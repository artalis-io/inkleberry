#!/usr/bin/env python3
"""Create deterministic, allowlisted source/embedding/runtime archives.
No reference code, models, credentials or SDK/driver binaries are bundled.
SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def safe(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.as_posix() != name:
        raise ValueError('Unsafe package path')
    return name


def read_file(path, base=ROOT):
    # Reject symlinked parent directories too, while allowing an operator's
    # selected root itself to live under a platform alias such as macOS /tmp.
    path, base = path.absolute(), base.absolute()
    try:
        relative = path.relative_to(base)
    except ValueError:
        raise ValueError('Package input escapes its selected root') from None
    cursor = base
    for part in relative.parts:
        cursor = cursor / part
        if part == '..' or cursor.is_symlink():
            raise ValueError('Symlinked package inputs are forbidden')
    if path.is_symlink() or not path.is_file():
        raise ValueError('Package input must be a regular non-symlink file: ' + str(path))
    return path.read_bytes()


def package(kind, output, version='0.1.0-dev', gpu=False, build_root=ROOT):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', version):
        raise ValueError('Invalid package version')
    if gpu and (kind != 'runtime' or platform.system() != 'Linux'):
        raise ValueError('GPU runtime packaging currently requires Linux')
    policy = json.loads((ROOT / 'config/package.json').read_text())
    entries = {}
    modes = {}
    links = {}

    def add(name, path, executable=False, base=ROOT):
        safe(name)
        if name in entries or name in links:
            raise ValueError('Duplicate package entry')
        entries[name] = read_file(path, base)
        modes[name] = 0o755 if executable else 0o644

    for name in ['LICENSE', 'NOTICE', 'THIRD_PARTY.md']:
        add(name, ROOT / name)
    if kind == 'source':
        for name in policy['source_files']:
            safe(name)
            if name.startswith('reference/') or Path(name).suffix in ['.zig', '.safetensors', '.gguf', '.bin']:
                raise ValueError('Reference/private data forbidden in source package')
            if name not in entries:
                add(name, ROOT / name, name.endswith('.sh'))
    elif kind == 'embedding':
        for name in ['dist/sq.h', 'dist/sq.c', 'examples/inspect.c', 'examples/generate.c']:
            add(name, ROOT / name)
        entries['README.md'] = read_file(ROOT / 'docs/embedding.md')
        modes['README.md'] = 0o644
    elif kind == 'runtime':
        add('bin/squid-c', build_root / 'squid-c', True, build_root)
        shared = 'libsq.dylib' if platform.system() == 'Darwin' else 'libsq.so'
        add('lib/' + shared, build_root / shared, base=build_root)
        for name in policy['public_headers']:
            add('include/' + Path(name).name, ROOT / name)
        if gpu:
            add('bin/squid-c-generate', build_root / 'squid-c-generate', True, build_root)
            add('lib/libsq_backend.so.1', build_root / 'libsq_backend.so.1', base=build_root)
            add('lib/libsq-cuda.a', build_root / 'libsq-cuda.a', base=build_root)
            links['lib/libsq_backend.so'] = 'libsq_backend.so.1'
        dependencies = {
            'system': ['C runtime', 'libm'],
            'gpu': ['compatible NVIDIA driver (libcuda.so.1)', 'CUDA runtime (libcudart)',
                    'cuBLAS and its dependencies', 'C++ runtime'] if gpu else [],
            'bundled_sdk_driver_libraries': False,
            'python_required_at_runtime': False, 'zig_required_at_runtime': False,
            'model_files_bundled': False,
        }
        entries['DEPENDENCIES.json'] = (json.dumps(dependencies, indent=2) + '\n').encode()
        entries['README.md'] = read_file(ROOT / 'docs/runtime-package.md')
        modes.update({'README.md': 0o644, 'DEPENDENCIES.json': 0o644})
    else:
        raise ValueError('Unknown package kind')
    metadata = {
        'format_version': 1, 'kind': kind, 'version': version,
        'platform': platform.system().lower() + '-' + platform.machine() if kind == 'runtime' else None,
        'gpu': gpu,
        'files': {n: hashlib.sha256(data).hexdigest() for n, data in sorted(entries.items())},
        'symlinks': links,
    }
    entries['MANIFEST.json'] = (json.dumps(metadata, sort_keys=True, indent=2) + '\n').encode()
    modes['MANIFEST.json'] = 0o644
    suffix = ('-' + metadata['platform'] + ('-cuda' if gpu else '')) if kind == 'runtime' else ''
    prefix = 'native-inference-' + kind + '-' + version + suffix
    output.mkdir(parents=True, exist_ok=True)
    path = output / (prefix + '.tar.gz')
    temporary = path.with_suffix(path.suffix + '.tmp')
    epoch = int(os.environ.get('SOURCE_DATE_EPOCH', '0'))
    if not 0 <= epoch <= 2**31-1:
        raise ValueError('SOURCE_DATE_EPOCH must be in 0..2147483647')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as archive:
                for name in sorted(set(entries) | set(links)):
                    info = tarfile.TarInfo(prefix + '/' + safe(name))
                    info.mtime = epoch
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    if name in links:
                        info.type = tarfile.SYMTYPE
                        info.linkname = links[name]
                        info.mode = 0o777
                        archive.addfile(info)
                    else:
                        info.size = len(entries[name])
                        info.mode = modes[name]
                        archive.addfile(info, io.BytesIO(entries[name]))
    temporary.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['source', 'embedding', 'runtime'])
    parser.add_argument('--out', type=Path, default=ROOT / '.build/releases')
    parser.add_argument('--version', default='0.1.0-dev')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--build-root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        path = package(args.kind, args.out, args.version, args.gpu, args.build_root)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + '\n')
    print(path)


if __name__ == '__main__':
    main()
