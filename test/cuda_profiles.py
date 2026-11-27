#!/usr/bin/env python3
"""CPU-only profile/compiler rejection and build isolation contracts; never runs CUDA."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--cxx', required=True)
p.add_argument('--flags', required=True)
a = p.parse_args()
base = shlex.split(a.cxx) + shlex.split(a.flags)
base += ['-I' + str(ROOT), '/Zs' if os.name == 'nt' else '-fsyntax-only']
cases = [({}, True), ({'IFN_CUDA_TARGET_SM': '121'}, True),
         ({'IFN_CUDA_TARGET_SM': '86', 'IFN_CUDA_ENABLE_EXPERIMENTAL_AMPERE': '1'}, True),
         ({'IFN_CUDA_TARGET_SM': '86'}, False),
         ({'IFN_CUDA_TARGET_SM': '86', 'IFN_CUDA_ENABLE_EXPERIMENTAL_AMPERE': '0'}, False),
         ({'IFN_CUDA_ENABLE_EXPERIMENTAL_AMPERE': '2'}, False)]
cases += [({'IFN_CUDA_TARGET_SM': str(sm)}, False) for sm in [80, 89, 90, 122]]
cases += [({'IFN_CUDA_TARGET_SM': '86', 'IFN_CUDA_ENABLE_EXPERIMENTAL_AMPERE': '1', '__CUDA_ARCH__': '1200'}, False),
          ({'IFN_CUDA_TARGET_SM': '120', '__CUDA_ARCH__': '860'}, False),
          ({'IFN_CUDA_TARGET_SM': '121', '__CUDA_ARCH__': '1210'}, True),
          ({'IFN_CUDA_TARGET_SM': '120', '__CUDA_ARCH__': '1200'}, True)]
with tempfile.TemporaryDirectory(prefix='ifn-target-') as tmp:
    source = Path(tmp) / 'probe.cpp'
    source.write_text('#include "src/cuda/target_profile.h"\n')
    for definitions, expected in cases:
        result = subprocess.run(base + ['-D' + k + '=' + v for k, v in definitions.items()] + [str(source)], capture_output=True, text=True)
        assert (result.returncode == 0) == expected, (definitions, result.stdout, result.stderr)
        if not expected:
            assert any(x in result.stdout + result.stderr for x in ['experimental', 'unsupported CUDA', 'disagrees', 'must be 0 or 1']), result.stderr

# Exercise the C/CUDA agreement contract, including an intentionally drifted C shape.
with tempfile.TemporaryDirectory(prefix='ifn-geometry-') as tmp:
    source = Path(tmp) / 'geometry.cpp'
    for drift in [False, True]:
        source.write_text('#include "config.hpp"\n#include "' + (ROOT/'src/inkling_geometry.h').as_posix() + '"\n' +
                          ('#undef IFN_INKLING_HIDDEN\n#define IFN_INKLING_HIDDEN 2305\n' if drift else '') +
                          '#include "' + (ROOT/'src/cuda/geometry_contract.hpp').as_posix() + '"\n')
        result = subprocess.run([x for x in base if x != '-I' + str(ROOT)] + ['-I' + str(ROOT/'test/cuda-stubs'),
                                        '-I' + str(ROOT/'cuda'), str(source)], capture_output=True, text=True)
        assert (result.returncode != 0) == drift, (result.stdout, result.stderr)
        if drift:
            assert 'Inkling hidden mismatch' in result.stdout + result.stderr

def dry(arguments, success=True):
    result = subprocess.run(['make', '--no-print-directory', '-n', *arguments], cwd=ROOT, capture_output=True, text=True)
    assert (result.returncode == 0) == success, (arguments, result.stdout, result.stderr)
    return result.stdout

for arguments in [['CUDA_ARCH=86'], ['CUDA_ARCH=80'], ['CUDA_ARCH='], ['CUDA_ARCH=120a 121a'],
                  ['EXPERIMENTAL_AMPERE=2'], ['EXPERIMENTAL_AMPERE=']]:
    dry([*arguments, 'gpu-build'], False)
for arch, opt in [('120a', False), ('121a', False), ('86', True)]:
    args = ['CUDA_ARCH=' + arch] + (['EXPERIMENTAL_AMPERE=1'] if opt else [])
    output = dry([*args, 'gpu-build'])
    assert '-DIFN_CUDA_TARGET_SM=' + arch.removesuffix('a') in output
    assert 'arch=compute_' + arch + ',code=sm_' + arch in output
    assert 'run_cuda_smoke.py' not in output and 'native_benchmark' not in output
    if opt:
        if os.name != 'nt':
            assert '.build/sm86-experimental/libinkleberry-gpu.a' in output
            assert '.build/sm86-experimental/inkleberry-generate' in output
            cleanup = dry([*args, 'clean'])
            commands = [shlex.split(line) for line in cleanup.splitlines() if line.startswith('rm -rf ')]
            assert len(commands) == 1, cleanup
            for name in commands[0][2:]:
                assert name.startswith('.build/sm86-experimental'), name
        package = dry([*args, 'runtime-gpu-package'])
        assert '--cuda-profile' in package and '--build-root' in package
print(f'{len(cases)} compiler profiles and build/clean/package isolation contracts passed (no CUDA execution)')
