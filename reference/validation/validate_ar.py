#!/usr/bin/env python3
"""Reserved-GPU, actual C/Zig AR parity. Never downloads or changes GPU settings."""
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--cuda-lib', required=True)
    parser.add_argument('--long', action='store_true')
    args = parser.parse_args()
    cases = [(mode, kv, graph, 33) for mode in range(3)
             for kv in range(3) for graph in range(2)]
    cases += [(2, 1, 1, n) for n in (1, 7, 8, 9, 31, 32, 2047, 2048, 2049)]
    if args.long:
        cases = [(2, 1, 1, n) for n in (8192, 131072)]
    records = []
    for mode, kv, graph, count in cases:
        print(f'Testing mode={mode} kv={kv} graphs={graph} prompt={count}',
              file=sys.stderr, flush=True)
        command = [sys.executable, 'tools/run_cuda_smoke.py', args.cuda_lib,
                   '.build/ar-oracle', args.model, str(mode), str(kv), str(graph),
                   str(count), 'What is 2 + 2? Answer briefly.']
        env = dict(os.environ)
        if args.long:
            env['INKLEBERRY_TEST_TIMEOUT_SECONDS'] = '3600'
        start = time.monotonic()
        run = subprocess.run(command, env=env, text=True, capture_output=True)
        print(run.stdout, file=sys.stderr, end='')
        print(run.stderr, file=sys.stderr, end='')
        if run.returncode:
            raise SystemExit(run.returncode)
        rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
        if len(rows) != 1 or rows[0].get('c_ar_parity') is not True:
            raise SystemExit('Missing complete successful parity evidence')
        rows[0].update(mode=mode, kv=kv, graphs=bool(graph),
                       wall_seconds=time.monotonic() - start)
        records.append(rows[0])
    print(json.dumps(dict(repetitions=1, timing='qualification, not benchmark',
                         contexts='real repeated tokenized text, not synthetic KV',
                         cases=records), indent=2))


if __name__ == '__main__':
    main()
