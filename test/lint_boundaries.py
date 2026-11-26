#!/usr/bin/env python3
import subprocess,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);(p/'src').mkdir();(p/'cuda').mkdir()
    (p/'src/ok.c').write_text('int ifn_ok(void) { return 0; }')
    for suffix in ['.cu','.cuh','.hpp','.cpp']:
        f=p/('cuda/hidden'+suffix)
        for bad in ['void ifn_bad(){std::exit(1);}', '#define IFN_BAD() abort()', 'void ifn_bad(){malloc(4);}']:
            f.write_text(bad)
            result=subprocess.run([sys.executable,str(root/'tools/lint.py'),'--root',str(p)],capture_output=True)
            assert result.returncode, (suffix,bad)
        f.write_text('/* std::exit(1) is forbidden */ int ifn_safe(){return 0;}')
        subprocess.run([sys.executable,str(root/'tools/lint.py'),'--root',str(p)],check=True,capture_output=True)
        f.unlink()
print('Included CUDA/C++ fatal and allocation paths are rejected; comments are ignored')
