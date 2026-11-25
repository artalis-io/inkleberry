#!/usr/bin/env python3
"""Explicit single-device, idle-GPU guard. Does not change any GPU settings."""
import os
import subprocess
import sys

def main():
    uuid = os.environ.get('SQ_TEST_GPU_UUID', '')
    if not uuid.startswith('GPU-'):
        raise SystemExit('Set SQ_TEST_GPU_UUID to the explicitly approved idle test GPU UUID')
    query = subprocess.check_output([
        'nvidia-smi', '-i', uuid, '--query-gpu=uuid,name,memory.used',
        '--format=csv,noheader,nounits'], text=True).strip()
    fields = [x.strip() for x in query.split(',')]
    if len(fields) != 3 or fields[0] != uuid or int(fields[2]) > 256:
        raise SystemExit('GPU identity or idle VRAM preflight failed')
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    if any(line.split(',')[0].strip() == uuid for line in processes.splitlines()):
        raise SystemExit('GPU has a compute workload; refusing to run')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, LD_LIBRARY_PATH=sys.argv[1])
    print('Idle GPU preflight:', query, flush=True)
    # A race with external schedulers remains; use a reserved GPU for release tests.
    timeout = int(os.environ.get('SQ_TEST_TIMEOUT_SECONDS', '180'))
    if timeout < 1 or timeout > 3600:
        raise SystemExit('SQ_TEST_TIMEOUT_SECONDS must be 1..3600')
    subprocess.run(sys.argv[2:], env=env, check=True, timeout=timeout)

if __name__ == '__main__':
    main()
