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
    if not os.environ.get('SQ_TEST_GPU_UUID','').startswith('GPU-') or not a.model.is_dir():
        p.error('An approved SQ_TEST_GPU_UUID and existing external model directory are required')
    with tempfile.TemporaryDirectory(prefix='infer-native-release-') as tmp:
        folder=Path(tmp)
        source=checks.extract(package('source',folder/'archives'),folder/'source')
        subprocess.run(['make','-j8','CC='+a.cc,'CXX='+a.cxx,'CUDA_ROOT='+str(a.cuda_root),
                        'generated-check','all','squid-c-generate','libsq-cuda.a','.build/infer-native-shared'],cwd=source,check=True)
        runtime=checks.extract(package('runtime',folder/'archives',gpu=True,build_root=source),folder/'runtime')
        embed=checks.extract(package('embedding',folder/'archives'),folder/'embedding')
        env=dict(os.environ,SQ_TEST_TIMEOUT_SECONDS='600')
        libraries=str(runtime/'lib')+':'+str(a.cuda_root/'lib64')
        guard=[sys.executable,str(ROOT/'tools/run_cuda_smoke.py'),libraries]
        subprocess.run(guard+[str(source/'.build/infer-native-shared'),str(a.model)],env=env,check=True)
        # Adapt only the public include spelling to the combined header, then
        # exercise the same native acceptance through a standalone C host.
        text=(source/'test/infer_native.c').read_text().replace('#include "sq_infer.h"','#include "sq.h"').replace('#include "sq_infer_squid.h"','')
        (embed/'consumer.c').write_text(text)
        cc,cxx=shlex.split(a.cc),shlex.split(a.cxx)
        subprocess.run(cc+['-std=c11','-O2','-DSQ_ENABLE_CUDA','-Idist','-c','dist/sq.c','-o','host.o'],cwd=embed,check=True)
        subprocess.run(cc+['-std=c11','-O2','-Idist','-c','consumer.c','-o','consumer.o'],cwd=embed,check=True)
        subprocess.run(cxx+['consumer.o','host.o',str(runtime/'lib/libsq-cuda.a'),'-L'+str(a.cuda_root/'lib64'),
                           '-L'+str(a.cuda_root/'lib64/stubs'),'-lcudart','-lcublas','-lcuda','-lstdc++','-lm','-lpthread','-o','consumer'],cwd=embed,check=True)
        subprocess.run(guard+[str(embed/'consumer'),str(a.model)],env=env,check=True)
        command=guard+[str(runtime/'bin/squid-c-generate'),str(a.model),'What is 2 + 2? Answer briefly.','32','nvfp4','fp8','graph']
        result=subprocess.run(command,env=env,capture_output=True,text=True,check=True)
        if '4' not in result.stdout:raise SystemExit('Packaged generation did not produce the arithmetic answer')
        print(result.stdout)
    print(json.dumps({'extracted_source_native':True,'extracted_amalgamation_native':True,
                      'packaged_cli':True,'reference_files_required':False}))


if __name__=='__main__':main()
