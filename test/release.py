#!/usr/bin/env python3
"""Exercise extracted allowlisted artifacts, not the developer checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import release


def extract(archive, destination):
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        roots = {m.name.split('/')[0] for m in members}
        assert len(roots) == 1
        for member in members:
            release.safe(member.name)
            assert member.isfile() or member.issym()
            if member.issym():
                basename = Path(member.name).name
                allowed = basename in ('libinkleberry_backend.so', 'libinkleberry.so', 'libinkleberry.so.1', 'libinkleberry.dylib', 'libinkleberry.1.dylib')
                assert allowed and '/lib/' in member.name
                assert '/' not in member.linkname and member.linkname not in ('.', '..')
                assert (member.linkname == 'libinkleberry_backend.so.1' if basename == 'libinkleberry_backend.so' else member.linkname.startswith(('libinkleberry.so.', 'libinkleberry.')))
        tar.extractall(destination)
    root = destination / roots.pop()
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert not (root / 'reference').exists() and not list(root.rglob('*.zig'))
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default='cc')
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='infer-release-') as tmp:
        folder = Path(tmp)
        source = release.package('source', folder / 'archives')
        first = source.read_bytes()
        assert release.package('source', folder / 'archives').read_bytes() == first
        root = extract(source, folder / 'source')
        subprocess.run(['make', '-j2', 'CC=' + args.cc, 'CXX=' + args.cxx,
                        'check', 'install-check'], cwd=root, check=True)
        # A two-file distribution builds with the C compiler alone. It contains
        # no generator, Python, source tree or test dependencies.
        root = extract(release.package('embedding', folder / 'archives'), folder / 'embedding')
        assert not list(root.rglob('*.py')) and not (root / 'src').exists()
        cc, cxx = shlex.split(args.cc), shlex.split(args.cxx)
        subprocess.run(cc + ['-std=c11', '-O2', '-Idist', '-c', 'dist/ifn.c', '-o', 'host.o'], cwd=root, check=True)
        subprocess.run(cc + ['-std=c11', '-Idist', 'examples/inspect.c', 'host.o', '-lm', '-o', 'inspect'], cwd=root, check=True)
        subprocess.run(cxx + ['-std=c++11', '-Idist', '-x', 'c++', '-c', 'examples/inspect.c', '-o', 'consumer.o'], cwd=root, check=True)
        subprocess.run(cxx + ['consumer.o', 'host.o', '-lm', '-o', 'inspect-cpp'], cwd=root, check=True)
        fixture = folder / 'input.safetensors'
        header = b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
        fixture.write_bytes(struct.pack('<Q', len(header)) + header + b'*')
        for exe in ['inspect', 'inspect-cpp']:
            result = subprocess.run([str(root / exe), str(fixture)], capture_output=True, text=True, check=True)
            assert '1 tensors, 1 payload bytes' in result.stdout
        root = extract(release.package('runtime', folder / 'archives', gpu=args.gpu), folder / 'runtime')
        assert not list(root.rglob('*.py')) and not list(root.rglob('*.c'))
        subprocess.run([str(root / 'bin/inkleberry'), 'inspect', str(fixture)], check=True, capture_output=True)
        policy = json.loads((root / 'DEPENDENCIES.json').read_text())
        assert not policy['python_required_at_runtime'] and not policy['zig_required_at_runtime']
        if args.gpu:
            # Copying/linking is checked here. Actual GPU execution is an explicit
            # idle-device test, never an implicit side effect of packaging.
            assert (root / 'lib/libinkleberry_backend.so').resolve().is_file()
            assert (root / 'bin/inkleberry-generate').is_file()
    print('Deterministic source, C/C++ embedding and runtime archives verified from extracted contents')


if __name__ == '__main__':
    main()
