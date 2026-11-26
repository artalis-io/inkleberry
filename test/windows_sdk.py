#!/usr/bin/env python3
"""Native SDK archives: deterministic packaging, extracted consumers and PE imports."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import release

if os.name != 'nt':
    raise SystemExit('This gate requires native Windows tools')
root = release.ROOT
build = root / os.environ['BUILD_DIR']
cc = os.environ['CC']
cxx = os.environ['CXX']
cflags = ['/nologo', '/utf-8', '/TC', '/std:c11', '/MD', '/W4', '/WX']
if Path(cc).name in ('cl', 'cl.exe'):
    cflags.append('/experimental:c11atomics')
with tempfile.TemporaryDirectory(prefix='ifn-windows-sdk-') as temporary:
    work = Path(temporary)
    for kind in ['source', 'embedding', 'runtime']:
        archive = release.package(kind, work, build_root=build)
        first = hashlib.sha256(archive.read_bytes()).digest()
        assert hashlib.sha256(release.package(kind, work, build_root=build).read_bytes()).digest() == first
        destination = work / kind
        destination.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(destination, filter='data')
        extracted, = destination.iterdir()
        manifest = json.loads((extracted / 'MANIFEST.json').read_text())
        for name, checksum in manifest['files'].items():
            assert hashlib.sha256((extracted / name).read_bytes()).hexdigest() == checksum, name
        if kind == 'source':
            subprocess.run(['make', '-j4', 'CC=' + cc, 'CXX=' + cxx, 'PYTHON="' + sys.executable.replace(chr(92), '/') + '"',
                            'all', 'platform-check', 'abi-contract-check'], cwd=extracted, check=True)
        elif kind == 'embedding':
            subprocess.run([cc, *cflags, '-Idist', '-c', 'dist/ifn.c', '/Fohost.obj'], cwd=extracted, check=True)
            subprocess.run([cc, *cflags, '-Idist', 'examples/inspect.c', '/Feconsumer.exe', '/link', 'host.obj'], cwd=extracted, check=True)
            fixture = extracted / 'fixture.safetensors'
            header = b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
            fixture.write_bytes(len(header).to_bytes(8, 'little') + header + b'x')
            subprocess.run([str(extracted / 'consumer.exe'), str(fixture)], cwd=extracted, check=True)
            subprocess.run([cxx, '/nologo', '/TP', '/std:c++14', '/MD', '/W4', '/WX', '-Idist',
                            'examples/inspect.c', '/Feconsumer-cpp.exe', '/link', 'host.obj'], cwd=extracted, check=True)
            subprocess.run([str(extracted / 'consumer-cpp.exe'), str(fixture)], cwd=extracted, check=True)
        else:
            source = root / 'test/installed_consumer.c'
            subprocess.run([cc, *cflags, '-Iinclude', str(source), '/Febin/static.exe', '/link', 'lib/inkleberry.lib'], cwd=extracted, check=True)
            subprocess.run([str(extracted / 'bin/static.exe')], cwd=extracted, check=True)
            subprocess.run([cxx, '/nologo', '/TP', '/std:c++14', '/MD', '/W4', '/WX', '-Iinclude', str(source),
                            '/Febin/shared.exe', '/link', 'lib/inkleberry-1.lib'], cwd=extracted, check=True)
            subprocess.run([str(extracted / 'bin/shared.exe')], cwd=extracted, check=True)
            for name in ['inkleberry-1.dll', 'static.exe', 'shared.exe']:
                imports = subprocess.check_output(['llvm-readobj', '--coff-imports', str(extracted / 'bin' / name)], text=True).lower()
                assert not any(x in imports for x in ['msys-', 'cygwin', 'winpthread', 'libgcc']), name
print('Windows deterministic source/embedding/runtime archives and extracted C/C++ consumers passed')
