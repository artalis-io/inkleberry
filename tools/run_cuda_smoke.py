#!/usr/bin/env python3
"""Explicit single-device, idle-GPU guard. Does not change any GPU settings."""
import os
import subprocess
import sys

def main():
    uuid = os.environ.get('INKLEBERRY_TEST_GPU_UUID', '')
    if not uuid.startswith('GPU-'):
        raise SystemExit('Set INKLEBERRY_TEST_GPU_UUID to the explicitly approved idle test GPU UUID')
    query = subprocess.check_output([
        'nvidia-smi', '-i', uuid, '--query-gpu=uuid,name,memory.used',
        '--format=csv,noheader,nounits'], text=True).strip()
    fields = [x.strip() for x in query.split(',')]
    if len(fields) != 3 or fields[0] != uuid:
        raise SystemExit('GPU identity preflight failed')
    if fields[2] in ('[N/A]', 'N/A') and fields[1] == 'NVIDIA GB10':
        # GB10 shares system memory and exposes no per-device VRAM counter.
        # Require no activity as well as the compute-process check below;
        # never interpret unavailable memory as zero on other devices.
        utilization = subprocess.check_output([
            'nvidia-smi', '-i', uuid, '--query-gpu=utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        if utilization != '0':
            raise SystemExit('GB10 GPU activity preflight failed')
    else:
        try:
            memory = int(fields[2])
        except ValueError:
            raise SystemExit('GPU VRAM measurement unavailable; refusing to run')
        if memory < 0 or memory > 256:
            raise SystemExit('GPU idle VRAM preflight failed')
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    if any(line.split(',')[0].strip() == uuid for line in processes.splitlines()):
        raise SystemExit('GPU has a compute workload; refusing to run')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, LD_LIBRARY_PATH=sys.argv[1])
    print('Idle GPU preflight:', query, flush=True)
    # A race with external schedulers remains; use a reserved GPU for release tests.
    timeout = int(os.environ.get('INKLEBERRY_TEST_TIMEOUT_SECONDS', '180'))
    if timeout < 1 or timeout > 3600:
        raise SystemExit('INKLEBERRY_TEST_TIMEOUT_SECONDS must be 1..3600')
    subprocess.run(sys.argv[2:], env=env, check=True, timeout=timeout)

if __name__ == '__main__':
    main()
