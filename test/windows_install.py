#!/usr/bin/env python3
"""Native custom-prefix install and non-destructive uninstall contract."""
import os
from pathlib import Path
import subprocess
import tempfile
root = Path(__file__).resolve().parents[1]
assert os.name == 'nt'
with tempfile.TemporaryDirectory(prefix='ifn-install-') as temporary:
    stage = Path(temporary)
    args = ['make', 'CC=' + os.environ['CC'], 'CXX=' + os.environ['CXX'],
            'DESTDIR=' + stage.as_posix(), 'PREFIX=/opt/ifn', 'LIBDIR=/opt/ifn/lib64',
            'INCLUDEDIR=/opt/ifn/headers', 'BINDIR=/opt/ifn/tools', 'PKGCONFIGDIR=/opt/ifn/metadata']
    subprocess.run(args + ['install'], cwd=root, check=True)
    home = stage / 'opt/ifn'
    for name in ['headers/ifn.h', 'headers/ifn_infer.h', 'lib64/inkleberry.lib',
                 'lib64/inkleberry-infer.lib', 'lib64/inkleberry-1.lib',
                 'tools/inkleberry.exe', 'tools/inkleberry-1.dll']:
        assert (home / name).is_file(), name
    unrelated = home / 'lib64/customer-owned.txt'
    unrelated.write_text('preserve')
    other = home / 'lib64/inkleberry_backend-2.dll'
    other.write_bytes(b'other ABI')
    provider = home / 'lib64/inkleberry_backend-1.dll'
    provider.write_bytes(b'owned placeholder')
    subprocess.run(args + ['uninstall-infer-backend'], cwd=root, check=True)
    assert not provider.exists() and (home / 'tools/inkleberry.exe').is_file()
    subprocess.run(args + ['uninstall'], cwd=root, check=True)
    assert unrelated.read_text() == 'preserve' and other.read_bytes() == b'other ABI'
    assert sorted(p for p in home.rglob('*') if p.is_file()) == sorted([unrelated, other])
print('Native custom paths and non-destructive backend/full uninstall passed')
