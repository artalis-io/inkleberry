#!/usr/bin/env python3
"""Reserved-device C/Zig prefix checkpoint and shielding parity, no downloads."""
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--long', action='store_true')
    group.add_argument('--edges', action='store_true')
    args = parser.parse_args()
    cases = [(mode, kv, graph, 8192, mtp) for mode in range(3)
             for kv in range(3) for graph in range(2) for mtp in range(2)]
    if args.long:
        cases = [(2, kv, 1, 131072, 1) for kv in (1, 2)]
    if args.edges:
        cases = [(mode, kv, graph, 8192, 1) for mode in range(3)
                 for kv in range(3) for graph in range(2)]
    results = []
    for mode, kv, graph, n, mtp in cases:
        print(f'Testing mode={mode} kv={kv} graphs={graph} capacity={n} mtp={mtp}',
              file=sys.stderr, flush=True)
        env = dict(os.environ)
        if args.long:
            env['INKLEBERRY_TEST_TIMEOUT_SECONDS'] = '3600'
        command = [sys.executable, 'tools/run_cuda_smoke.py', args.cuda_lib,
                   '.build/prefix-oracle', args.model, str(mode), str(kv),
                   str(graph), str(n), str(mtp)]
        if args.edges:
            command.append('edges')
        start = time.monotonic()
        run = subprocess.run(command, env=env, capture_output=True, text=True)
        print(run.stdout, file=sys.stderr, end='')
        print(run.stderr, file=sys.stderr, end='')
        if run.returncode:
            raise SystemExit(run.returncode)
        records = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
        if len(records) != 1 or not records[0].get('prefix_parity'):
            raise SystemExit('Missing full successful parity record')
        if ((not args.edges and not records[0].get('live_hits')) or
                not records[0].get('shields') or not records[0].get('restores')):
            raise SystemExit('Missing executed live/checkpoint/shield coverage')
        if args.long and records[0].get('maximum_reused', 0) <= 65536:
            raise SystemExit('Long run did not demonstrate target reuse above 65536 tokens')
        records[0].update(mode=mode, kv=kv, graphs=bool(graph), context_capacity=n+64,
                          mtp=bool(mtp), edges=args.edges, wall_seconds=time.monotonic()-start)
        results.append(records[0])
    print(json.dumps(dict(repetitions=1, timing='qualification, not throughput benchmark',
                         cases=results), indent=2))


if __name__ == '__main__':
    main()
