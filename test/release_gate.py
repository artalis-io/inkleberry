#!/usr/bin/env python3
"""Failed release attempts invalidate previous success before any build/GPU work."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ['tokenizer.json', 'model-00001-of-00001.safetensors', 'mtp.safetensors']
with tempfile.TemporaryDirectory(prefix='ifn-release-failure-') as tmp:
    base = Path(tmp)
    # Isolate the runner and a tiny synthetic release manifest. Never a real model.
    tree = base / 'source'
    (tree / 'tools').mkdir(parents=True)
    (tree / 'test/fixtures').mkdir(parents=True)
    for name in ['run_model_release.py', 'model_release.py']:
        shutil.copy2(ROOT / 'tools' / name, tree / 'tools' / name)
    (tree / 'VERSION').write_text('0.2.0-dev\n')
    model = base / 'model'
    model.mkdir()
    for name in FILES:
        (model / name).write_bytes(b'synthetic public test bytes')
    manifest = {'files': {n: hashlib.sha256((model/n).read_bytes()).hexdigest() for n in FILES}}
    fixture = tree / 'test/fixtures/inkling-golden-v1.json'
    fixture.write_text(json.dumps(manifest))
    baseline = base / 'baseline.jsonl'
    rows = [{'case': c, 'profile': p, 'warmup': False, 'run': run,
             'generated': 20, 'tokens': [17]*20, 'prompt_tokens': 32,
             'first_token_ns': 100, 'elapsed_ns': 1000}
            for c in range(7) for p in ['ar', 'fixed', 'adaptive', 'sampled'] for run in range(5)]
    baseline.write_text('\n'.join(json.dumps(r) for r in rows))
    out = base / 'evidence'
    out.mkdir()
    report = out / 'release-evidence.json'
    old = {'passed': True, 'run_id': 'previous-success', 'checks': {'old-check': True}}
    args = [sys.executable, str(tree / 'tools/run_model_release.py'),
            '--model', str(model), '--gpu', 'GPU-TEST-NOT-ACCESSED',
            '--baseline-engine', str(baseline), '--baseline-facade', str(baseline),
            '--source-revision', 'a'*40, '--out', str(out)]

    def rejected(extra, text, env=None):
        report.write_text(json.dumps(old))
        result = subprocess.run(args + extra, text=True, capture_output=True, env=env)
        assert result.returncode != 0, result.stdout
        current = json.loads(report.read_text())
        assert current['passed'] is False and current['run_id'] != old['run_id'], current
        assert 'old-check' not in current['checks'] and 'finished_utc' in current, current
        assert text in current['failure'], current
        assert not list(out.glob('.release-evidence-*.tmp'))
        return current

    rejected(['--gpu', 'bad'], 'Explicit GPU UUID')
    rejected(['--baseline-engine', str(base/'missing')], 'Missing baseline')
    bad = base / 'invalid.jsonl'
    bad.write_text('not a benchmark')
    rejected(['--baseline-engine', str(bad)], 'Expected all 28')
    rejected(['--model', str(base/'missing-model')], 'FileNotFoundError')
    (model / FILES[0]).write_bytes(b'changed input')
    rejected([], 'identity mismatch')
    (model / FILES[0]).write_bytes(b'synthetic public test bytes')
    rejected(['--source-revision', 'not-a-revision'], 'requires a Git checkout')

    # Exercise real receipt publication with mocked read-only provenance and a
    # failed child build. No GPU, toolchain or Git repository is required.
    import importlib.util
    from unittest.mock import patch
    sys.path.insert(0, str(tree/'tools'))
    spec = importlib.util.spec_from_file_location('ifn_release_runner', tree/'tools/run_model_release.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    (tree/'config').mkdir()
    (tree/'config/package.json').write_text(json.dumps({'source_files':['VERSION']}))
    revision='a'*40
    contract='\n'.join('IFN_RELEASE_'+k+'='+v for k,v in {
        'BUILD_DIR':'.build', 'EXE':'', 'PROVIDER':'backend.so', 'CC':'cc',
        'NVCC':'nvcc', 'CUDA_RUNTIME_PATH':'/not-used'}.items())
    def queries(command):
        if command[:3]==['git','rev-parse','HEAD']:return revision
        if command[:3]==['git','rev-parse','--show-toplevel']:return str(tree)
        if command[:2]==['git','status']:return ''
        if command[0]=='make':return contract
        return 'synthetic tool identity'
    def invoke(query_fn, error):
        report.write_text(json.dumps(old))
        with patch.object(sys,'argv',args[1:]), patch.object(runner,'query',side_effect=query_fn), patch.object(runner.subprocess,'run',side_effect=subprocess.CalledProcessError(23,['make'])):
            try:runner.main();raise AssertionError('failed release accepted')
            except (SystemExit,subprocess.CalledProcessError):pass
        current=json.loads(report.read_text())
        assert not current['passed'] and error in current['failure'],current
        return current
    current=invoke(queries,'CalledProcessError')
    assert current['checks']=={'preflight':True},current
    def wrong_revision(command):
        return 'b'*40 if command[:3]==['git','rev-parse','HEAD'] else queries(command)
    invoke(wrong_revision,'differs from checked-out HEAD')
    def dirty(command):
        return ' M VERSION' if command[:2]==['git','status'] else queries(command)
    invoke(dirty,'clean source checkout')
print('GPU/baseline/model/source preflight and child build failures invalidate stale success atomically')
