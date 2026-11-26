#!/usr/bin/env python3
"""Toolchain-only negative checks against the actual CUDA C++ bridge.
Requires a passing cuda-abi-check; does not execute GPU kernels.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', required=True)
    parser.add_argument('--cxx', required=True)
    parser.add_argument('--nvcc', required=True)
    parser.add_argument('--cuda-root', required=True)
    parser.add_argument('--cuda-flags', required=True)
    parser.add_argument('--cuda-objects', required=True)
    args = parser.parse_args()
    original = (ROOT / 'src/cuda_abi.h').read_text()
    with tempfile.TemporaryDirectory(prefix='abi-negative-', dir=ROOT / '.build') as tmp:
        folder = Path(tmp)
        header = folder / 'cuda_abi.h'
        # The same compiler/input flags must accept the original header first.
        command = [args.nvcc] + shlex.split(args.cuda_flags) + ['-I' + str(folder), '-Isrc',
                  '-c', 'test/cuda_abi_probe.cu', '-o', str(folder / 'cpp.o')]
        header.write_text(original)
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        # A signature disagreement with the real bridge must fail compilation.
        old = 'int32_t ifn_cuda_malloc('
        assert original.count(old) == 1
        header.write_text(original.replace(old, 'float ifn_cuda_malloc(', 1))
        failed = subprocess.run(command[:-1] + [str(folder / 'invalid-cpp.o')],
                                cwd=ROOT, capture_output=True, text=True)
        if failed.returncode == 0 or 'ifn_cuda_malloc' not in failed.stderr:
            raise SystemExit('Compiler did not reject the intentional CUDA signature mismatch')
        # Change only C record layout. C++ expected offsets/sizes still come
        # from the actual bridge included by the separately compiled probe.
        old = 'uint32_t pad;'
        assert original.count(old) == 1
        header.write_text(original.replace(old, 'uint64_t pad;', 1))
        c_object = folder / 'c.o'
        subprocess.run(shlex.split(args.cc) + ['-std=c11', '-I' + str(folder), '-c',
                       'test/cuda_abi_c.c', '-o', str(c_object)], cwd=ROOT, check=True)
        executable = folder / 'layout-probe'
        cuda_lib = str(Path(args.cuda_root) / 'lib64')
        subprocess.run(shlex.split(args.cxx) + [str(folder / 'cpp.o'), str(c_object)] +
                       shlex.split(args.cuda_objects) + ['-L' + cuda_lib,
                       '-L' + str(Path(cuda_lib) / 'stubs'), '-lcudart', '-lcublas',
                       '-lcuda', '-lstdc++', '-lm', '-lpthread', '-o', str(executable)],
                       cwd=ROOT, check=True)
        env = dict(os.environ, LD_LIBRARY_PATH=cuda_lib)
        failed = subprocess.run([str(executable)], env=env, capture_output=True, text=True)
        if failed.returncode != 1 or 'ifn_cuda_c_sample_config.' not in failed.stderr:
            raise SystemExit('Independent C/C++ probe did not reject the intentional layout mismatch')
    print('Actual CUDA C++ bridge rejected signature and record-layout mismatches')


if __name__ == '__main__':
    main()
