#!/usr/bin/env python3
"""Exercise the actual client with deliberately broken public-ABI test doubles."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('worker_history', ROOT / 'tools/check_worker_lifetime.py')
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)
PROGRAM = sys.argv.pop(1) if len(sys.argv) > 1 else 'build/worker-lifetime-mock'


class WorkerLifetime(unittest.TestCase):
    def run_client(self, fault=''):
        result = subprocess.run([PROGRAM, 'mock-model', '42', '12'], text=True,
                                capture_output=True, timeout=15,
                                env=dict(os.environ, INKLEBERRY_WORKER_MOCK_FAULT=fault))
        return result, [json.loads(line) for line in result.stdout.splitlines()]

    def test_composed_history(self):
        result, records = self.run_client()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = history.validate(records)
        self.assertTrue(summary['all_modes_exercised'])
        self.assertEqual(summary['requests'], 60)

    def test_replay(self):
        _, a = self.run_client()
        _, b = self.run_client()
        self.assertEqual([v for v in a if v['kind'] == 'begin'],
                         [v for v in b if v['kind'] == 'begin'])

    def test_broken_provider_rejected(self):
        for fault in ('cross-scope', 'corruption', 'reject', 'ignore-cancel'):
            with self.subTest(fault=fault):
                result, records = self.run_client(fault)
                self.assertNotEqual(result.returncode, 0)
                with self.assertRaises(ValueError):
                    history.validate(records)

    def test_corrupt_or_incomplete_history(self):
        _, records = self.run_client()
        mutations = [records[:-1], records[:5] + records[7:]]
        for field, value in [('output', [999]), ('reuse', 128), ('status', 8)]:
            changed = copy.deepcopy(records)
            changed[4][field] = value  # First miss / ordinary request
            mutations.append(changed)
        for changed in mutations:
            with self.assertRaises(ValueError):
                history.validate(changed)


if __name__ == '__main__':
    unittest.main()
