#!/usr/bin/env python3
"""Idle guard regressions, including GB10 unavailable VRAM reporting."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cuda_guard', ROOT / 'tools/run_cuda_smoke.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
UUID = 'GPU-test'

class GuardTests(unittest.TestCase):
    def invoke(self, memory='26', name='NVIDIA RTX PRO 6000', utilization='0',
               processes='', identity=UUID, succeeds=True, uuid=UUID):
        def query(command, **kwargs):
            if '--query-gpu=uuid,name,memory.used' in command:
                return f'{identity}, {name}, {memory}'
            if '--query-gpu=utilization.gpu' in command:
                return utilization
            if '--query-compute-apps=gpu_uuid,pid' in command:
                return processes
            raise AssertionError(command)
        with patch.dict(guard.os.environ, {'INKLEBERRY_TEST_GPU_UUID': uuid}, clear=True), \
             patch.object(guard.sys, 'argv', ['guard', '/cuda/lib64', '/benchmark']), \
             patch.object(guard.subprocess, 'check_output', side_effect=query), \
             patch.object(guard.subprocess, 'run') as run, \
             contextlib.redirect_stdout(io.StringIO()):
            if succeeds:
                guard.main()
                run.assert_called_once()
                self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], UUID)
            else:
                with self.assertRaises(SystemExit):
                    guard.main()
                run.assert_not_called()

    def test_idle_dedicated_gpu(self): self.invoke()
    def test_busy_vram(self): self.invoke(memory='257', succeeds=False)
    def test_invalid_counter(self): self.invoke(memory='garbage', succeeds=False)
    def test_negative_counter(self): self.invoke(memory='-1', succeeds=False)
    def test_wrong_identity(self): self.invoke(identity='GPU-other', succeeds=False)
    def test_explicit_uuid_required(self): self.invoke(uuid='', succeeds=False)
    def test_unknown_unavailable_vram(self): self.invoke(memory='[N/A]', succeeds=False)
    def test_idle_gb10(self): self.invoke(memory='[N/A]', name='NVIDIA GB10')
    def test_idle_gb10_plain_na(self): self.invoke(memory='N/A', name='NVIDIA GB10')
    def test_active_gb10(self): self.invoke(memory='[N/A]', name='NVIDIA GB10', utilization='1', succeeds=False)
    def test_unknown_gb10_activity(self): self.invoke(memory='[N/A]', name='NVIDIA GB10', utilization='[N/A]', succeeds=False)
    def test_allocated_idle_gb10(self): self.invoke(memory='[N/A]', name='NVIDIA GB10', processes=UUID+', 123', succeeds=False)
    def test_allocated_dedicated_gpu(self): self.invoke(processes=UUID+', 123', succeeds=False)
    def test_other_device_process(self): self.invoke(processes='GPU-other, 123')

if __name__ == '__main__': unittest.main()
