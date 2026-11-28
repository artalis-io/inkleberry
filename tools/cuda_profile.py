#!/usr/bin/env python3
"""Deterministic build identity; compilation never establishes qualification."""
import argparse
import json
from pathlib import Path

def profile(arch, experimental=False):
    if arch not in ('120a', '121a', '86') or not isinstance(experimental, bool):
        raise ValueError('Unsupported CUDA build profile')
    if arch == '86' and not experimental:
        raise ValueError('Ampere requires explicit experimental opt-in')
    return {'format_version': 1, 'cuda_arch': arch, 'target_sm': int(arch.removesuffix('a')),
            'experimental': arch == '86', 'launch_policy': 'ordered' if arch == '86' else 'pdl',
            'qualification': 'not-established-by-build'}

def validate(value):
    if (not isinstance(value, dict) or type(value.get('format_version')) is not int or
            type(value.get('target_sm')) is not int or
            value != profile(value.get('cuda_arch'), value.get('experimental'))):
        raise ValueError('Invalid CUDA build identity')
    return value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', required=True)
    parser.add_argument('--experimental', action='store_true')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    data = (json.dumps(profile(args.arch, args.experimental), sort_keys=True, indent=2) + '\n').encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out.exists() or args.out.read_bytes() != data:
        temporary = args.out.with_suffix('.tmp')
        temporary.write_bytes(data)
        temporary.replace(args.out)
