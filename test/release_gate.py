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
    rejected(['--source-revision', 'not-a-revision'], 'Supply exact')

    # A real child build failure, with synthetic preflight assets, cannot retain success.
    # Stub make only; do not run real build commands or access a GPU.
    commands = base / 'commands'
    commands.mkdir()
    make = commands / 'make'
    make.write_text('#!/bin/sh\nexit 23\n')
    make.chmod(0o755)
    env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ['PATH'])
    current = rejected([], 'CalledProcessError', env)
    assert current['checks'] == {'preflight': True}, current
print('GPU/baseline/model/revision preflight and child build failures replace stale success atomically')
