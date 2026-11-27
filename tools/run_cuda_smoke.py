#!/usr/bin/env python3
"""Explicit single-device, idle-GPU guard. Does not change any GPU settings."""
import os
import base64
import json
from pathlib import Path
import time
import subprocess
import sys

def windows_process_starts(pids):
    """Creation identities prevent a recycled desktop PID from inheriting approval."""
    if not pids or any(type(pid) is not int or pid <= 0 for pid in pids):
        raise SystemExit('Invalid Windows process identity')
    ids = ','.join(str(pid) for pid in sorted(pids))
    script = ("$ErrorActionPreference='Stop'; $ids=@(" + ids + "); "
              "@(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in $ids } | "
              "ForEach-Object { @{pid=[int]$_.ProcessId; "
              "created=$_.CreationDate.ToUniversalTime().ToString('o')} }) | ConvertTo-Json -Compress")
    encoded = base64.b64encode(script.encode('utf-16le')).decode('ascii')
    values = json.loads(subprocess.check_output([
        'powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive',
        '-EncodedCommand', encoded], text=True, timeout=30))
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, list):
        raise SystemExit('Windows process identities unavailable')
    result = {str(v['pid']): v['created'] for v in values}
    if set(result) != {str(pid) for pid in pids} or any(not isinstance(v, str) or not v for v in result.values()):
        raise SystemExit('Windows process identity incomplete')
    return result


def windows_reservation(path, uuid, memory, processes, now=None):
    if os.name != 'nt':
        raise SystemExit('Desktop reservation is only valid on native Windows')
    now = time.time() if now is None else now
    try:
        value = json.loads(Path(path).read_text())
        issued, expires, limit = (value[k] for k in ('issued_at', 'expires_at', 'max_vram_mib'))
        allowed = value['processes']
        if (type(value['format_version']) is not int or value['format_version'] != 1 or value['uuid'] != uuid or
                any(type(v) not in (int, float) for v in (issued, expires)) or
                not (0 <= issued <= now < expires <= issued + 7200) or
                type(limit) is not int or not (256 < limit <= 4096) or
                not isinstance(allowed, dict) or not allowed or memory < 0 or memory > limit):
            raise ValueError('Invalid/expired desktop reservation or VRAM exceeded')
        for pid, item in allowed.items():
            if (not isinstance(pid, str) or not pid.isdigit() or int(pid) <= 0 or
                    not isinstance(item, dict) or set(item) != {'name', 'created'} or
                    any(not isinstance(v, str) or not v for v in item.values())):
                raise ValueError('Malformed approved process identity')
        observed = {}
        for line in processes.splitlines():
            row = [field.strip() for field in line.split(',', 2)]
            if row[0] != uuid:
                continue
            if len(row) != 3 or not row[1].isdigit() or not row[2]:
                raise ValueError('Unobservable GPU process')
            observed[row[1]] = row[2]
        if not observed or not set(observed) <= set(allowed):
            raise ValueError('New or unobservable GPU process')
        starts = windows_process_starts([int(pid) for pid in observed])
        for pid, name in observed.items():
            if allowed[pid] != {'name': name, 'created': starts[pid]}:
                raise ValueError('GPU process identity changed')
        return value
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise SystemExit('Windows reservation rejected: ' + str(error)) from error


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
    reservation = os.environ.get('INKLEBERRY_TEST_WINDOWS_RESERVATION', '')
    if reservation:
        processes = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
            '--format=csv,noheader,nounits'], text=True, timeout=30)
        try:
            memory = int(fields[2])
        except ValueError:
            raise SystemExit('Desktop VRAM measurement unavailable')
        windows_reservation(reservation, uuid, memory, processes)
        for _ in range(2):
            utilization = subprocess.check_output([
                'nvidia-smi', '-i', uuid, '--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits'], text=True, timeout=30).strip()
            if utilization != '0':
                raise SystemExit('Windows desktop GPU is active; refusing to run')
            time.sleep(1)
        # Re-read after sampling so a process appearing during the idle window
        # cannot inherit the earlier successful process/VRAM check.
        current = subprocess.check_output([
            'nvidia-smi', '-i', uuid, '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True, timeout=30).strip()
        processes = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
            '--format=csv,noheader,nounits'], text=True, timeout=30)
        try:
            current = int(current)
        except ValueError:
            raise SystemExit('Desktop VRAM measurement unavailable')
        windows_reservation(reservation, uuid, current, processes)
        print('Explicit Windows desktop reservation verified', flush=True)
    elif fields[2] in ('[N/A]', 'N/A') and fields[1] == 'NVIDIA GB10':
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
    if not reservation and any(line.split(',')[0].strip() == uuid for line in processes.splitlines()):
        raise SystemExit('GPU has a compute workload; refusing to run')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, LD_LIBRARY_PATH=sys.argv[1])
    if os.name == 'nt':
        env['PATH'] = sys.argv[1] + os.pathsep + env.get('PATH', '')
    print('Idle GPU preflight:', query, flush=True)
    # A race with external schedulers remains; use a reserved GPU for release tests.
    timeout = int(os.environ.get('INKLEBERRY_TEST_TIMEOUT_SECONDS', '180'))
    if timeout < 1 or timeout > 3600:
        raise SystemExit('INKLEBERRY_TEST_TIMEOUT_SECONDS must be 1..3600')
    subprocess.run(sys.argv[2:], env=env, check=True, timeout=timeout)

if __name__ == '__main__':
    main()
