#!/usr/bin/env python3
import json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='ifn-install-') as tmp:
    stage=Path(tmp);args=['make','DESTDIR='+tmp,'PREFIX=/opt/ifn','LIBDIR=/opt/ifn/lib64','INCLUDEDIR=/opt/ifn/headers','BINDIR=/opt/ifn/tools','PKGCONFIGDIR=/opt/ifn/metadata']
    subprocess.run(args+['install'],cwd=ROOT,check=True)
    env=dict(os.environ,PKG_CONFIG_PATH=str(stage/'opt/ifn/metadata'),PKG_CONFIG_SYSROOT_DIR=tmp)
    version=subprocess.check_output(['pkg-config','--modversion','inkleberry'],env=env,text=True).strip()
    assert version==(ROOT/'VERSION').read_text().strip()
    flags=subprocess.check_output(['pkg-config','--cflags','--libs','--static','inkleberry'],env=env,text=True)
    import shlex
    output=stage/'consumer'
    subprocess.run(shlex.split(os.environ.get('CC','cc'))+['-std=c11',str(ROOT/'test/installed_consumer.c')]+shlex.split(flags)+['-Wl,-rpath,'+str(stage/'opt/ifn/lib64'),'-o',str(output)],check=True)
    subprocess.run([str(output)],check=True)
    unrelated=stage/'opt/ifn/lib64/customer-owned.txt';unrelated.write_text('preserve')
    # Backend-only removal must preserve the installed host SDK and other files.
    lib=stage/'opt/ifn/lib64'; metadata=stage/'opt/ifn/metadata'
    (lib/'libinkleberry_backend.so.1').write_bytes(b'owned backend placeholder')
    (lib/'libinkleberry_backend.so').symlink_to('libinkleberry_backend.so.1')
    (metadata/'inkleberry-infer.pc').write_text('owned backend metadata')
    other=lib/'libinkleberry_backend.so.2';other.write_bytes(b'other ABI')
    subprocess.run(args+['uninstall-infer-backend'],cwd=ROOT,check=True)
    assert not (lib/'libinkleberry_backend.so.1').exists()
    assert not (lib/'libinkleberry_backend.so').is_symlink()
    assert not (metadata/'inkleberry-infer.pc').exists()
    assert (lib/'libinkleberry.a').is_file() and (stage/'opt/ifn/tools/inkleberry').is_file()
    assert (stage/'opt/ifn/headers/ifn.h').is_file() and (metadata/'inkleberry.pc').is_file()
    assert other.read_bytes()==b'other ABI'
    subprocess.run(args+['uninstall'],cwd=ROOT,check=True)
    assert unrelated.read_text()=='preserve'
    assert sorted(p for p in (stage/'opt/ifn').rglob('*') if p.is_file())==sorted([unrelated,other])
print('Custom install paths, pkg-config consumer, version, backend-only and full non-destructive uninstall passed')
