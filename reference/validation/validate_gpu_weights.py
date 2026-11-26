#!/usr/bin/env python3
"""Explicit GPU qualification; external weights, no downloads or GPU settings changes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--cuda-lib', required=True)
    parser.add_argument('--oracle', default='.build/gpu-weight-oracle')
    args = parser.parse_args()
    uuid = os.environ.get('INKLEBERRY_TEST_GPU_UUID', '')
    if not uuid.startswith('GPU-'):
        parser.error('INKLEBERRY_TEST_GPU_UUID is required')
    results = []
    cases = [(mode, mtp, 0) for mode in range(3) for mtp in range(2)]
    cases += [(1, 1, 1), (2, 1, 1)]
    for mode, mtp, force in cases:
        env = dict(os.environ, SQUID_DRAFT_VOCAB='full', SQUID_MTP_BF16=str(force))
        command = [sys.executable, 'tools/run_cuda_smoke.py', args.cuda_lib,
                   args.oracle, args.model, str(mode), str(mtp)]
        peak = 0
        start = time.monotonic()
        with tempfile.TemporaryFile(mode='w+') as log:
            with subprocess.Popen(command, env=env, stdout=log, stderr=log) as child:
                while child.poll() is None:
                    value = subprocess.check_output([
                        'nvidia-smi', '-i', uuid, '--query-gpu=memory.used',
                        '--format=csv,noheader,nounits'], text=True, timeout=10)
                    peak = max(peak, int(value.strip()))
                    time.sleep(0.25)
                log.seek(0)
                output = log.read()
                print(output, file=sys.stderr, end='')
                if child.returncode:
                    raise SystemExit(child.returncode)
        rows = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
        if len(rows) != 1 or rows[0].get('oracle_parity') is not True:
            raise SystemExit('Missing successful parity record')
        rows[0].update(mtp_bf16=bool(force), elapsed_seconds=time.monotonic() - start,
                       sampled_combined_oracle_and_c_vram_mib=peak)
        results.append(rows[0])
    print(json.dumps(dict(gpu_uuid=uuid, model_directory=Path(args.model).name,
                         sampling_interval_seconds=0.25, cases=results), indent=2))


if __name__ == '__main__':
    main()
