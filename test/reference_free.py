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
    with tempfile.TemporaryDirectory(prefix='ifn-reference-free-') as tmp:
        root = Path(tmp)
        for name in ['VERSION', 'Makefile', 'LICENSE', 'UPSTREAM.json', 'inkleberry.pc.in', 'inkleberry-infer.pc.in']:
            shutil.copy2(ROOT / name, root / name)
        for name in ['include', 'src', 'cuda', 'cli', 'test', 'tools', 'abi', 'config', 'dist', 'examples', 'fuzz']:
            shutil.copytree(ROOT / name, root / name, ignore=shutil.ignore_patterns('__pycache__', '*.zig'))
        assert not (root / 'reference').exists() and not list(root.rglob('*.zig'))
        verifier = [sys.executable, str(root / 'tools/verify_upstream.py')]
        subprocess.run(verifier, check=True, capture_output=True)
        # Removing retired sources must not weaken CUDA provenance verification.
        kernel = root / 'cuda/config.hpp'
        original = kernel.read_bytes()
        kernel.write_bytes(original + b'\n// intentional isolated provenance failure\n')
        rejected = subprocess.run(verifier, capture_output=True, text=True)
        if rejected.returncode == 0 or 'cuda/config.hpp' not in rejected.stderr:
            raise SystemExit('Production verification failed to reject altered CUDA input')
        kernel.write_bytes(original)
        subprocess.run(['make', '-j2', 'CC=' + args.cc, 'CXX=' + args.cxx,
                        'check', 'install-check'], cwd=root, check=True)
    print('Production C checks/install checks passed without any Zig/reference files')


if __name__ == '__main__':
    main()
