#!/usr/bin/env python3
"""Reserved-device C/Zig adaptive DSpark and prompt lookup parity, no downloads."""
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
    group.add_argument('--copy-only', action='store_true')
    args = parser.parse_args()
    cases = [(mode, kv, 33, lookup, 64, False) for mode in range(3)
             for kv in range(3) for lookup in range(2)]
    cases += [(2, 1, n, 1, 64, False) for n in (1, 7, 8, 9, 31, 32, 2047, 2048, 2049)]
    cases += [(2, 1, 33, lookup, 32, False) for lookup in range(2)]
    copies = [(mode, kv, 1024, 1, 64, True) for mode in range(3) for kv in range(3)]
    cases += copies
    if args.copy_only:
        cases = copies
    if args.long:
        cases = [(2, 1, n, 1, 64, False) for n in (8192, 131072)]
    results = []
    for mode, kv, n, lookup, extra, copy in cases:
        print(f'Testing mode={mode} kv={kv} tokens={n} lookup={lookup} room={extra} copy={copy}',
              file=sys.stderr, flush=True)
        env = dict(os.environ)
        if args.long:
            env['SQ_TEST_TIMEOUT_SECONDS'] = '3600'
        command = [sys.executable, 'tools/run_cuda_smoke.py', args.cuda_lib,
                   '.build/adaptive-oracle', args.model, str(mode), str(kv),
                   str(n), str(lookup), str(extra)]
        if copy:
            command.append('copy')
        start = time.monotonic()
        run = subprocess.run(command, env=env, capture_output=True, text=True)
        print(run.stdout, file=sys.stderr, end='')
        print(run.stderr, file=sys.stderr, end='')
        if run.returncode:
            raise SystemExit(run.returncode)
        records = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
        if len(records) != 1 or not records[0].get('adaptive_parity'):
            raise SystemExit('Missing full successful parity record')
        records[0].update(mode=mode, kv=kv, graphs=True, prompt_tokens=records[0].get("prompt_tokens", n),
                          requested_prompt_capacity=n,
                          lookup=bool(lookup), copy_fixture=copy, context_room=extra,
                          wall_seconds=time.monotonic() - start)
        results.append(records[0])
    print(json.dumps(dict(repetitions=1, timing='qualification, not throughput benchmark',
                         cases=results), indent=2))


if __name__ == '__main__':
    main()
