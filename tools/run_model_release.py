#!/usr/bin/env python3
"""Trusted, explicit release gate on an operator-selected idle GPU and local model."""
import argparse
import json
import os
import platform
import re
import shlex
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
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
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
        revision = query(['git', 'rev-parse', 'HEAD'])
        if not revision or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
            raise SystemExit('Release provenance requires a Git checkout; extracted packages have separate qualification')
        if a.source_revision and a.source_revision.lower() != revision.lower():
            raise SystemExit('Claimed source revision differs from checked-out HEAD')
        status = query(['git', 'status', '--porcelain', '--untracked-files=all'])
        if status is None or status:
            raise SystemExit('Release provenance requires a clean source checkout')
        top = query(['git', 'rev-parse', '--show-toplevel'])
        if not top or Path(top).resolve() != ROOT.resolve():
            raise SystemExit('Release checkout root mismatch')
        info_text = query(['make', '--no-print-directory', '-s', 'release-build-info'])
        if info_text is None:
            raise SystemExit('Cannot resolve selected Make build contract')
        info = dict(line.removeprefix('IFN_RELEASE_').split('=', 1)
                    for line in info_text.splitlines() if line.startswith('IFN_RELEASE_'))
        build = Path(info['BUILD_DIR'])
        suffix = info['EXE']
        names = [str(build / (n + suffix)) for n in ('native-benchmark', 'facade-benchmark', 'verify')]
        cc = shlex.split(info['CC'])
        nvcc = shlex.split(info['NVCC'])
        manifest.update(source_revision=revision, dirty=False,
                        version=(ROOT / 'VERSION').read_text().strip(), model_files=actual,
                        source_files={name: digest(ROOT/name) for name in
                                      json.loads((ROOT/'config/package.json').read_text())['source_files']},
                        gpu=query(['nvidia-smi', '-i', a.gpu,
                                   '--query-gpu=uuid,name,driver_version,compute_cap,power.limit',
                                   '--format=csv,noheader']), platform=platform.platform(),
                        compiler=query(cc + (['/?'] if Path(cc[0]).name.lower() in ('cl','cl.exe') else ['--version'])),
                        cuda=query(nvcc + ['--version']), build_contract=info)
        if not manifest['compiler'] or not manifest['cuda'] or not manifest['gpu']:
            raise SystemExit('Required compiler/CUDA/device provenance unavailable')
        manifest['checks']['preflight'] = True
        run(['make', '-j8', 'check', 'install-check', 'native-namespace-check', *names])
        manifest['build_flags'] = (ROOT / build / 'config').read_text()
        manifest['binaries'] = {f: digest(ROOT / f) for f in names + [info['PROVIDER']]}
        guard = [sys.executable, str(ROOT / 'tools/run_cuda_smoke.py'), info['CUDA_RUNTIME_PATH']]
        identity_log = a.out / 'identity.log'
        run(guard + [str(ROOT / names[2]), str(a.model)] +
            [actual[n] for n in FILES], identity_log)
        if expected['artifact_fingerprint'] not in identity_log.read_text():
            raise RuntimeError('C/Python artifact fingerprints differ')
        manifest['checks']['artifact_verification'] = True
        for kind, binary, baseline in [
                ('engine', 'native-benchmark', a.baseline_engine),
                ('facade', 'facade-benchmark', a.baseline_facade)]:
            output = a.out / (kind + '.jsonl')
            run(guard + [str(ROOT / build / (binary + suffix)), str(a.model)], output)
            run([sys.executable, 'tools/model_release.py', 'verify', str(output),
                 '--golden', 'test/fixtures/inkling-golden-v1.json'])
            records(output, require_identity=True)
            comparison = compare(baseline, output)
            (a.out / (kind + '-comparison.json')).write_text(json.dumps(comparison, indent=2) + '\n')
            if not comparison['performance_passed']:
                raise RuntimeError(kind + ' performance gate failed; investigate before release')
            manifest['checks'][kind + '_quality_performance'] = True
        if not a.skip_package:
            run(['make', 'native-release-check', 'MODEL_DIR=' + str(a.model)], a.out / 'packages.log')
            manifest['checks']['extracted_packages'] = True
        if query(['git', 'rev-parse', 'HEAD']) != revision or query(['git', 'status', '--porcelain', '--untracked-files=all']):
            raise RuntimeError('Source changed during qualification')
        if any(digest(ROOT/name)!=sha for name,sha in manifest['source_files'].items()):
            raise RuntimeError('Source contents changed during qualification')
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
