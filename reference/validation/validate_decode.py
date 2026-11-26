#!/usr/bin/env python3
"""Reserved-device C/Zig sampling and fixed-width DSpark parity, no downloads."""
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
    cases = [(mode, kv, graph, 33, width, 64) for mode in range(3)
             for kv in range(3) for graph in range(2) for width in (0, 1, 3, 7)]
    cases += [(2, 1, graph, 33, width, 64) for graph in range(2)
              for width in (2, 4, 5, 6)]
    cases += [(2, 1, 1, n, 7, 64) for n in (1, 7, 8, 9, 31, 32, 2047, 2048, 2049)]
    cases += [(2, 1, graph, 33, width, 32) for graph in range(2) for width in (1, 3, 7)]
    if args.long:
        cases = [(2, 1, 1, n, 7, 64) for n in (8192, 131072)]
    results = []
    for mode, kv, graph, n, width, extra in cases:
        print(f'Testing mode={mode} kv={kv} graphs={graph} tokens={n} drafts={width} room={extra}',
              file=sys.stderr, flush=True)
        env = dict(os.environ)
        if args.long:
            env['INKLEBERRY_TEST_TIMEOUT_SECONDS'] = '3600'
        command = [sys.executable, 'tools/run_cuda_smoke.py', args.cuda_lib,
                   '.build/decode-oracle', args.model, str(mode), str(kv), str(graph),
                   str(n), str(width), str(extra)]
        start = time.monotonic()
        run = subprocess.run(command, env=env, capture_output=True, text=True)
        print(run.stdout, file=sys.stderr, end='')
        print(run.stderr, file=sys.stderr, end='')
        if run.returncode:
            raise SystemExit(run.returncode)
        records = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
        if len(records) != 1 or not records[0].get('decode_parity'):
            raise SystemExit('Missing full successful parity record')
        records[0].update(mode=mode, kv=kv, graphs=bool(graph), prompt_tokens=n,
                          draft_tokens=width, context_room=extra,
                          wall_seconds=time.monotonic() - start)
        results.append(records[0])
    print(json.dumps(dict(repetitions=1, timing='qualification, not throughput benchmark',
                         cases=results), indent=2))


if __name__ == '__main__':
    main()
