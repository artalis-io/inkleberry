#!/usr/bin/env python3
"""Build the production C targets in an isolated tree with no reference files."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default='cc')
    parser.add_argument('--cxx', default='c++')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='sq-reference-free-') as tmp:
        root = Path(tmp)
        for name in ['Makefile', 'LICENSE', 'UPSTREAM.json', 'sq.pc.in', 'squid-infer.pc.in']:
            shutil.copy2(ROOT / name, root / name)
        for name in ['include', 'src', 'cuda', 'cli', 'test', 'tools', 'abi']:
            shutil.copytree(ROOT / name, root / name, ignore=shutil.ignore_patterns('__pycache__', '*.zig'))
        assert not (root / 'reference').exists() and not list(root.rglob('*.zig'))
        verifier = [sys.executable, str(root / 'tools/verify_upstream.py')]
        subprocess.run(verifier + ['--production-only'], check=True, capture_output=True)
        # Production-only means Zig-independent, not weaker CUDA verification.
        kernel = root / 'cuda/config.hpp'
        original = kernel.read_bytes()
        kernel.write_bytes(original + b'\n// intentional isolated provenance failure\n')
        rejected = subprocess.run(verifier + ['--production-only'], capture_output=True, text=True)
        if rejected.returncode == 0 or 'cuda/config.hpp' not in rejected.stderr:
            raise SystemExit('Production verification failed to reject altered CUDA input')
        kernel.write_bytes(original)
        rejected = subprocess.run(verifier, capture_output=True, text=True)
        if rejected.returncode == 0 or '.zig' not in rejected.stderr:
            raise SystemExit('Full reference verification did not require the pinned Zig files')
        subprocess.run(['make', '-j2', 'CC=' + args.cc, 'CXX=' + args.cxx,
                        'check', 'install-check'], cwd=root, check=True)
    print('Production C checks/install checks passed without any Zig/reference files')


if __name__ == '__main__':
    main()
