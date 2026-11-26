#!/usr/bin/env python3
"""Explicit reserved-GPU qualification of extracted source/runtime/embedding SDKs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import release
from release import package
# The helper is named release.py in test/ as well; load its archive checker
# explicitly to avoid shadowing the production packager module.
import importlib.util
spec=importlib.util.spec_from_file_location('release_tests',ROOT/'test/release.py')
checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--cuda-root',type=Path,default=Path('/usr/local/cuda'))
    p.add_argument('--cc',default='cc');p.add_argument('--cxx',default='c++')
    a=p.parse_args()
    if not os.environ.get('INKLEBERRY_TEST_GPU_UUID','').startswith('GPU-') or not a.model.is_dir():
        p.error('An approved INKLEBERRY_TEST_GPU_UUID and existing external model directory are required')
    with tempfile.TemporaryDirectory(prefix='infer-native-release-') as tmp:
        folder=Path(tmp)
        source=checks.extract(package('source',folder/'archives'),folder/'source')
        # Exercise the documented operator target from clean extracted sources.
        # Do not pre-build infer or other archives: that hid a missing prerequisite.
        build=['make','-j8','CC='+a.cc,'CXX='+a.cxx,'CUDA_ROOT='+str(a.cuda_root)]
        subprocess.run(build+['runtime-gpu-package'],cwd=source,check=True)
        archives=list((source/'.build/releases').glob('inkleberry-runtime-*-cuda.tar.gz'))
        if len(archives)!=1:raise SystemExit('Expected exactly one native runtime archive')
        runtime=checks.extract(archives[0],folder/'runtime')
        subprocess.run(build+['native-namespace-check','.build/infer-native-shared'],cwd=source,check=True)
        subprocess.run([sys.executable,str(source/'tools/check_hardening.py'),
                        str(runtime/'bin/inkleberry'),str(runtime/'bin/inkleberry-generate'),
                        str(runtime/'lib/libinkleberry.so')],check=True)
        embed=checks.extract(package('embedding',folder/'archives'),folder/'embedding')
        env=dict(os.environ,INKLEBERRY_TEST_TIMEOUT_SECONDS='600')
        libraries=str(runtime/'lib')+':'+str(a.cuda_root/'lib64')
        guard=[sys.executable,str(ROOT/'tools/run_cuda_smoke.py'),libraries]
        subprocess.run(guard+[str(source/'.build/infer-native-shared'),str(a.model)],env=env,check=True)
        # Consume the installed static SDK through pkg-config, with the shared
        # host library present as in a real SDK. Explicit archives must prevent
        # the linker choosing a hidden-symbol shared host implementation.
        pkg_env=dict(os.environ,PKG_CONFIG_PATH=str(runtime/'lib/pkgconfig'))
        flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','--static','inkleberry-static-infer'],env=pkg_env,text=True))
        cc,cxx=shlex.split(a.cc),shlex.split(a.cxx)
        subprocess.run(cc+['-std=c11','-I'+str(runtime/'include'),'-c',str(source/'test/infer_native.c'),'-o',str(folder/'static-consumer.o')],check=True)
        subprocess.run(cxx+[str(folder/'static-consumer.o')]+flags+['-o',str(folder/'static-consumer')],check=True)
        subprocess.run(guard+[str(folder/'static-consumer'),str(a.model)],env=env,check=True)
        # Adapt only the public include spelling to the combined header, then
        # exercise the same native acceptance through a standalone C host.
        text=(source/'test/infer_native.c').read_text().replace('#include "ifn_infer.h"','#include "ifn.h"').replace('#include "ifn_infer_inkling.h"','')
        (embed/'consumer.c').write_text(text)
        cc,cxx=shlex.split(a.cc),shlex.split(a.cxx)
        subprocess.run(cc+['-std=c11','-O2','-DIFN_ENABLE_CUDA','-Idist','-c','dist/ifn.c','-o','host.o'],cwd=embed,check=True)
        subprocess.run(cc+['-std=c11','-O2','-Idist','-c','consumer.c','-o','consumer.o'],cwd=embed,check=True)
        subprocess.run(cxx+['consumer.o','host.o',str(runtime/'lib/libinkleberry-cuda.a'),'-L'+str(a.cuda_root/'lib64'),
                           '-L'+str(a.cuda_root/'lib64/stubs'),'-lcudart','-lcublas','-lcuda','-lstdc++','-lm','-lpthread','-o','consumer'],cwd=embed,check=True)
        subprocess.run(guard+[str(embed/'consumer'),str(a.model)],env=env,check=True)
        command=guard+[str(runtime/'bin/inkleberry-generate'),str(a.model),'What is 2 + 2? Answer briefly.','32','nvfp4','fp8','graph']
        result=subprocess.run(command,env=env,capture_output=True,text=True,check=True)
        if '4' not in result.stdout:raise SystemExit('Packaged generation did not produce the arithmetic answer')
        print(result.stdout)
    print(json.dumps({'extracted_source_native':True,'extracted_amalgamation_native':True,
                      'packaged_cli':True,'static_pkg_config_native':True,'reference_files_required':False}))


if __name__=='__main__':main()
