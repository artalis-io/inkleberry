#!/usr/bin/env python3
"""Trusted, explicit release gate on an operator-selected idle GPU and local model."""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from model_release import FILES, digest, compare, records

ROOT = Path(__file__).resolve().parents[1]


def publish(report, manifest):
    """Readers see a complete receipt; a new attempt first invalidates old success."""
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=report.parent,
                                         prefix='.release-evidence-', suffix='.tmp',
                                         delete=False) as stream:
            name = stream.name
            json.dump(manifest, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, report)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def query(args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, timeout=30).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--baseline-engine', type=Path, required=True)
    p.add_argument('--baseline-facade', type=Path, required=True)
    p.add_argument('--source-revision', default=os.environ.get('IFN_RELEASE_SOURCE_REVISION'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--skip-package', action='store_true',
                   help='Use only when extracted-package qualification was separately executed')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report = a.out / 'release-evidence.json'
    manifest = {'schema_version': 1, 'run_id': uuid.uuid4().hex,
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'passed': False, 'checks': {}}
    # Before baseline parsing, model IO, Git/toolchain queries or any build/GPU work.
    publish(report, manifest)
    env = dict(os.environ, INKLEBERRY_TEST_GPU_UUID=a.gpu,
               INKLEBERRY_TEST_TIMEOUT_SECONDS='3600')

    def run(args, output=None):
        if output:
            with output.open('w') as f:
                subprocess.run(args, cwd=ROOT, env=env, stdout=f,
                               stderr=subprocess.STDOUT, check=True)
        else:
            subprocess.run(args, cwd=ROOT, env=env, check=True)

    try:
        if not a.gpu.startswith('GPU-'):
            raise SystemExit('Explicit GPU UUID required')
        for path in [a.baseline_engine, a.baseline_facade]:
            if not path.is_file():
                raise SystemExit('Missing baseline receipt: ' + str(path))
            records(path)
        expected = json.loads((ROOT / 'test/fixtures/inkling-golden-v1.json').read_text())
        actual = {name: digest(a.model / name) for name in FILES}
        if actual != expected['files']:
            raise SystemExit('Model/tokenizer/MTP identity mismatch; no inference started')
        revision = a.source_revision or query(['git', 'rev-parse', 'HEAD'])
        if not revision or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
            raise SystemExit('Supply exact --source-revision for an extracted source tree')
        status = query(['git', 'status', '--porcelain'])
        manifest.update(source_revision=revision, dirty=None if status is None else bool(status),
                        version=(ROOT / 'VERSION').read_text().strip(), model_files=actual,
                        gpu=query(['nvidia-smi', '-i', a.gpu,
                                   '--query-gpu=uuid,name,driver_version,compute_cap,power.limit',
                                   '--format=csv,noheader']), platform=platform.platform(),
                        compiler=query(['cc', '--version']),
                        cuda=query(['/usr/local/cuda/bin/nvcc', '--version']))
        manifest['checks']['preflight'] = True
        run(['make', '-j8', 'check', 'install-check', 'native-namespace-check',
             '.build/native-benchmark', '.build/facade-benchmark', '.build/verify'])
        manifest['build_flags'] = (ROOT / '.build/config').read_text()
        manifest['binaries'] = {str(f): digest(ROOT / f) for f in [
            '.build/native-benchmark', '.build/facade-benchmark', 'libinkleberry_backend.so.1']}
        guard = [sys.executable, str(ROOT / 'tools/run_cuda_smoke.py'), '/usr/local/cuda/lib64']
        identity_log = a.out / 'identity.log'
        run(guard + [str(ROOT / '.build/verify'), str(a.model)] +
            [actual[n] for n in FILES], identity_log)
        if expected['artifact_fingerprint'] not in identity_log.read_text():
            raise RuntimeError('C/Python artifact fingerprints differ')
        manifest['checks']['artifact_verification'] = True
        for kind, binary, baseline in [
                ('engine', 'native-benchmark', a.baseline_engine),
                ('facade', 'facade-benchmark', a.baseline_facade)]:
            output = a.out / (kind + '.jsonl')
            run(guard + [str(ROOT / '.build' / binary), str(a.model)], output)
            run([sys.executable, 'tools/model_release.py', 'verify', str(output),
                 '--golden', 'test/fixtures/inkling-golden-v1.json'])
            comparison = compare(baseline, output)
            (a.out / (kind + '-comparison.json')).write_text(json.dumps(comparison, indent=2) + '\n')
            if not comparison['performance_passed']:
                raise RuntimeError(kind + ' performance gate failed; investigate before release')
            manifest['checks'][kind + '_quality_performance'] = True
        if not a.skip_package:
            run(['make', 'native-release-check', 'MODEL_DIR=' + str(a.model)], a.out / 'packages.log')
            manifest['checks']['extracted_packages'] = True
        manifest['passed'] = True
    except BaseException as error:
        manifest['failure'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        manifest['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        publish(report, manifest)
    print(str(report))


if __name__ == '__main__':
    main()
