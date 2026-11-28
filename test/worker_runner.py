#!/usr/bin/env python3
"""Supervisor/reporting failure paths without GPUs, credentials or model downloads."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('worker_runner', ROOT / 'tools/run_worker_lifetime.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
PROGRAM = sys.argv.pop(1) if len(sys.argv) > 1 else '.build/worker-lifetime-mock'
HISTORY = subprocess.check_output([PROGRAM, 'mock-model', '42', '12'], text=True)


class Runner(unittest.TestCase):
    def invoke(self, failed=False, incomplete=False, telemetry=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            model.mkdir()
            for name in runner.FILES:
                (model / name).write_bytes(b'fake model for CPU test only')
            binary = root / 'client'
            binary.write_bytes(b'fake executable, never executed')
            provider = root / 'provider.so.1'
            provider.write_bytes(b'fake provider, never executed')
            output = root / 'report'
            argv = ['runner', '--executable', str(binary), '--provider', str(provider),
                    '--model', str(model), '--gpu', 'GPU-test', '--library-path', '/cuda/lib64',
                    '--source-revision', 'a' * 40, '--output', str(output)]

            def run(command, **kwargs):
                self.assertEqual(kwargs['env']['INKLEBERRY_TEST_GPU_UUID'], 'GPU-test')
                self.assertEqual(kwargs['env']['INKLEBERRY_TEST_GUARD_STDERR'], '1')
                kwargs['stdout'].write(HISTORY.rsplit('\n', 2)[0] + '\n' if incomplete else HISTORY)
                return subprocess.CompletedProcess(command, 1 if failed else 0)

            def query(*args, **kwargs):
                if telemetry:
                    raise subprocess.TimeoutExpired('nvidia-smi', 15)
                return 'GPU-test, mock device, driver, 100, 0, 10'

            with contextlib.redirect_stdout(io.StringIO()), patch.object(sys, 'platform', 'linux'), patch.object(sys, 'argv', argv), patch.object(runner, 'verify', return_value={'checked': True}), \
                 patch.object(runner, 'run', side_effect=run), \
                 patch.object(runner.subprocess, 'check_output', side_effect=query):
                if failed or incomplete or telemetry:
                    with self.assertRaises(SystemExit):
                        runner.main()
                else:
                    runner.main()
            return json.loads((output / 'summary.json').read_text())

    def test_success(self):
        self.assertTrue(self.invoke()['passed'])

    def test_worker_failure(self):
        self.assertFalse(self.invoke(failed=True)['passed'])

    def test_incomplete_history(self):
        self.assertFalse(self.invoke(incomplete=True)['passed'])

    def test_telemetry_failure(self):
        self.assertFalse(self.invoke(telemetry=True)['passed'])


if __name__ == '__main__':
    unittest.main()
