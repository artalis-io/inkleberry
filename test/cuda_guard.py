#!/usr/bin/env python3
"""Idle guard regressions, including GB10 unavailable VRAM reporting."""
import contextlib
import importlib.util
import io
import json
import tempfile
from types import SimpleNamespace
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

    def test_timeout_requires_explicit_long_run(self):
        for opt_in, timeout, accepted in [('', '28800', False), ('1', '28800', True),
                                         ('1', '86401', False), ('', '0', False)]:
            with self.subTest(opt_in=opt_in, timeout=timeout):
                def query(command, **kwargs):
                    if '--query-gpu=uuid,name,memory.used' in command:
                        return f'{UUID}, NVIDIA RTX PRO 6000, 2'
                    return ''
                with patch.dict(guard.os.environ, dict(INKLEBERRY_TEST_GPU_UUID=UUID,
                        INKLEBERRY_TEST_LONG_RUN=opt_in, INKLEBERRY_TEST_TIMEOUT_SECONDS=timeout), clear=True), \
                     patch.object(guard.sys, 'argv', ['guard', '/cuda/lib64', '/benchmark']), \
                     patch.object(guard.subprocess, 'check_output', side_effect=query), \
                     patch.object(guard.subprocess, 'run') as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    if accepted:
                        guard.main()
                        self.assertEqual(run.call_args.kwargs['timeout'], int(timeout))
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

class WindowsReservationTests(unittest.TestCase):
    def check(self, change=None, processes=None, starts=None, memory=1536, succeeds=True, native=True):
        value = {'format_version': 1, 'uuid': UUID, 'issued_at': 1000, 'expires_at': 2000,
                 'max_vram_mib': 1792, 'processes': {'123': {'name': 'desktop.exe', 'created': 'start-A'}}}
        if change:
            value.update(change)
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'reservation.json'
            file.write_text(json.dumps(value))
            with patch.object(guard, 'os', SimpleNamespace(name='nt' if native else 'posix')), \
                 patch.object(guard, 'windows_process_starts', return_value=starts or {'123': 'start-A'}):
                args = (file, UUID, memory, processes if processes is not None else UUID+', 123, desktop.exe')
                if succeeds:
                    self.assertEqual(guard.windows_reservation(*args, now=1500), value)
                else:
                    with self.assertRaises(SystemExit): guard.windows_reservation(*args, now=1500)

    def test_explicit_desktop_snapshot(self): self.check()
    def test_non_windows(self): self.check(native=False, succeeds=False)
    def test_expired(self): self.check({'expires_at': 1500}, succeeds=False)
    def test_future(self): self.check({'issued_at': 1501}, succeeds=False)
    def test_overlong(self): self.check({'expires_at': 10000}, succeeds=False)
    def test_wrong_gpu(self): self.check({'uuid': 'GPU-other'}, succeeds=False)
    def test_new_process(self): self.check(processes=UUID+', 124, model.exe', succeeds=False)
    def test_renamed_process(self): self.check(processes=UUID+', 123, model.exe', succeeds=False)
    def test_pid_recycled(self): self.check(starts={'123': 'start-B'}, succeeds=False)
    def test_unknown_pid(self): self.check(processes=UUID+', N/A, desktop.exe', succeeds=False)
    def test_empty_list(self): self.check(processes='', succeeds=False)
    def test_vram_exceeded(self): self.check(memory=1793, succeeds=False)
    def test_vram_negative(self): self.check(memory=-1, succeeds=False)
    def test_large_limit(self): self.check({'max_vram_mib': 4097}, succeeds=False)
    def test_bool_limit(self): self.check({'max_vram_mib': True}, succeeds=False)
    def test_infinite_expiry(self): self.check({'expires_at': float('inf')}, succeeds=False)

class WindowsGuardWorkflowTests(unittest.TestCase):
    def workflow(self, activity='0', second_process=None, succeeds=True):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'reservation.json'
            file.write_text(json.dumps({'format_version': 1, 'uuid': UUID, 'issued_at': 1000,
                'expires_at': 2000, 'max_vram_mib': 1792,
                'processes': {'123': {'name': 'desktop.exe', 'created': 'start-A'}}}))
            reads = 0
            def query(command, **kwargs):
                nonlocal reads
                if '--query-gpu=uuid,name,memory.used' in command: return UUID+', RTX 3090 Ti, 1536'
                if '--query-gpu=memory.used' in command: return '1540'
                if '--query-gpu=utilization.gpu' in command: return activity
                if '--query-compute-apps=gpu_uuid,pid,process_name' in command:
                    reads += 1
                    return (second_process if reads > 1 and second_process is not None else UUID+', 123, desktop.exe')
                if '--query-compute-apps=gpu_uuid,pid' in command: return UUID+', 123'
                raise AssertionError(command)
            env = {'INKLEBERRY_TEST_GPU_UUID': UUID, 'INKLEBERRY_TEST_WINDOWS_RESERVATION': str(file), 'PATH': 'old-path'}
            with patch.object(guard, 'os', SimpleNamespace(name='nt', environ=env, pathsep=';')), \
                 patch.object(guard.sys, 'argv', ['guard', 'cuda-bin', 'test.exe']), \
                 patch.object(guard.time, 'time', return_value=1500), \
                 patch.object(guard.time, 'sleep'), \
                 patch.object(guard, 'windows_process_starts', return_value={'123': 'start-A'}), \
                 patch.object(guard.subprocess, 'check_output', side_effect=query), \
                 patch.object(guard.subprocess, 'run') as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                if succeeds:
                    guard.main()
                    run.assert_called_once()
                    self.assertEqual(run.call_args.kwargs['env']['PATH'], 'cuda-bin;old-path')
                else:
                    with self.assertRaises(SystemExit): guard.main()
                    run.assert_not_called()
    def test_desktop_launch(self): self.workflow()
    def test_busy_desktop(self): self.workflow(activity='1', succeeds=False)
    def test_unknown_activity(self): self.workflow(activity='N/A', succeeds=False)
    def test_process_appears_during_sampling(self):
        self.workflow(second_process=UUID+', 124, model.exe', succeeds=False)

if __name__ == '__main__': unittest.main()
