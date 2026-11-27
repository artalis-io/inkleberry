#!/usr/bin/env python3
"""Bounded Linux public-ABI qualification; owns only its child process group."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from check_worker_lifetime import validate
from model_release import FILES, digest
from process_tree import run
from qualification_source import verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--provider', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--library-path', required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cycles', type=int, default=12)
    parser.add_argument('--seconds', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('This qualification client currently targets Linux')
    if not 0 < args.seed <= 0xffffffff or not 0 < args.cycles <= 10000000 or not 0 <= args.seconds <= 80000:
        parser.error('Invalid seed, cycles or duration')
    source = verify(ROOT, args.source_revision, args.source_manifest)
    args.executable = args.executable.resolve(strict=True)
    args.provider = args.provider.resolve(strict=True)
    args.model = args.model.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(source=source, executable_sha256=digest(args.executable),
                    provider_sha256=digest(args.provider),
                    model_sha256={name: digest(args.model / name) for name in FILES},
                    seed=args.seed, cycles=args.cycles, seconds=args.seconds, gpu=args.gpu)
    # Hashes identify supplied files, not a claim of a signed source-to-binary build.
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stop = threading.Event()
    ready = threading.Event()
    monitor_errors = []

    def monitor():
        with (args.output / 'device.jsonl').open('x') as log:
            while not stop.is_set():
                try:
                    sample = subprocess.check_output([
                        'nvidia-smi', '-i', args.gpu,
                        '--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu,power.draw',
                        '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
                    log.write(json.dumps(dict(time_ns=time.time_ns(), sample=sample)) + '\n')
                    log.flush()
                    ready.set()
                except (OSError, subprocess.SubprocessError) as error:
                    monitor_errors.append(str(error))
                    ready.set()
                    return
                stop.wait(5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    timeout = max(3600, args.seconds + 1200)
    env = dict(os.environ, INKLEBERRY_TEST_GPU_UUID=args.gpu,
               INKLEBERRY_TEST_TIMEOUT_SECONDS=str(timeout), INKLEBERRY_TEST_LONG_RUN='1')
    command = [sys.executable, str(ROOT / 'tools/run_cuda_smoke.py'), str(args.provider.parent) + os.pathsep + args.library_path,
               str(args.executable), str(args.model), str(args.seed), str(args.cycles)]
    if args.seconds:
        command.append(str(args.seconds))
    (args.output / 'command.json').write_text(json.dumps(command) + '\n')
    failure = None
    try:
        if not ready.wait(20) or monitor_errors:
            raise ValueError('Initial device telemetry unavailable')
        # Guard messages are separated from client JSONL by the guard option below.
        env['INKLEBERRY_TEST_GUARD_STDERR'] = '1'
        with (args.output / 'history.jsonl').open('x') as stdout, (args.output / 'stderr.log').open('x') as stderr:
            result = run(command, stdout=stdout, stderr=stderr, env=env, timeout=timeout + 30)
        if result.returncode:
            raise ValueError(f'Worker exited {result.returncode}')
        with (args.output / 'history.jsonl').open() as history:
            summary = validate([json.loads(line) for line in history])
        if not summary['all_modes_exercised']:
            raise ValueError('Seed/duration did not exercise all cancellation modes')
        stop.set()
        monitor_thread.join(timeout=20)
        if monitor_errors:
            raise ValueError('Device telemetry failed: ' + monitor_errors[0])
        if verify(ROOT, args.source_revision, args.source_manifest) != source:
            raise ValueError('Source changed during campaign')
        if (digest(args.executable) != manifest['executable_sha256'] or
                digest(args.provider) != manifest['provider_sha256']):
            raise ValueError('Client changed during campaign')
    except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        failure = str(error)
        summary = dict(passed=False, failure=failure, seed=args.seed,
                       replay='Retain manifest, command, stderr and complete ordered history')
    finally:
        stop.set()
        monitor_thread.join(timeout=20)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
