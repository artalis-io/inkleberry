#!/usr/bin/env python3
"""Restricted translation of the pinned Zig ABI; fail closed on unknown types.
Generated files are checked in. No kernel or upstream bridge is modified.
SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
"""
import argparse
import difflib
from pathlib import Path
import re
from upstream import source_path

ROOT = Path(__file__).resolve().parents[1]
SCALARS = {'u8': 'uint8_t', 'u32': 'uint32_t', 'u64': 'uint64_t', 'i32': 'int32_t',
           'f32': 'float', 'usize': 'size_t', 'anyopaque': 'void', 'void': 'void'}
source = source_path(ROOT, 'src/cuda.zig').read_text()
records = []
for name, body in re.findall(r'pub const (\w+) = extern struct \{(.*?)^\};', source, re.M | re.S):
    fields = []
    for line in body.splitlines():
        line = line.split('//', 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r'(\w+):\s*(.*?)(?:\s*=.*)?,', line)
        if not m:
            raise ValueError('Unsupported ABI field: ' + line)
        fields.append((m[1], m[2].strip()))
    records.append((name, fields))
assert len(records) == 7
NAMES = {name for name, _ in records}

def ctype(value):
    if value.startswith('?'):
        assert value[1:].startswith(('*', '[*]')), 'Optional non-pointer ABI type'
        return ctype(value[1:])
    prefix = '[*]' if value.startswith('[*]') else '*' if value.startswith('*') else ''
    if prefix:
        rest = value[len(prefix):]
        const = rest.startswith('const ')
        base = ctype(rest[6:] if const else rest)
        if const:
            base = base + ' const' if base.endswith('*') else 'const ' + base
        return base + ' *'
    if value in SCALARS:
        return SCALARS[value]
    if value in NAMES:
        return value
    raise ValueError('Unknown ABI type: ' + value)

def declaration(name, value):
    if value.startswith('[constants.mtp_layer_count]'):
        return ctype(value[len('[constants.mtp_layer_count]'):]) + ' ' + name + '[3]'
    return ctype(value) + ' ' + name

banner = '/* Generated from pinned src/cuda.zig by tools/generate_cuda_abi.py; do not edit.\n * SPDX-License-Identifier: Apache-2.0\n * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis */\n'
header = [banner, '#ifndef SQ_CUDA_ABI_H\n#define SQ_CUDA_ABI_H\n#include <stddef.h>\n#include <stdint.h>\n', '#ifndef SQ_CUDA_ABI_NO_TYPES\n']
for name, fields in records:
    header += [f'typedef struct {name} {{\n', *['    ' + declaration(n, t) + ';\n' for n, t in fields], f'}} {name};\n']
header += ['#endif\n#ifdef __cplusplus\nextern "C" {\n#endif\n']
functions = re.findall(r'pub extern "c" fn (\w+)\((.*?)\)\s*(\w+);', source, re.S)
assert len(functions) == 76
for name, args, result in functions:
    params = []
    for arg in args.split(','):
        arg = arg.strip()
        if arg:
            n, t = arg.split(':', 1)
            params.append(declaration(n.strip(), t.strip()))
    header.append(ctype(result) + ' ' + name + '(' + (', '.join(params) if params else 'void') + ');\n')
header += ['#ifdef __cplusplus\n}\n#endif\n#endif\n']
# Two translation units compare the actual C and C++ representations. The
# C++ TU includes the original bridge (no copied substitute layout) and then
# redeclares every function using the generated header: prototype conflicts fail.
items = []
for name, fields in records:
    items += [(name + '.size', f'sizeof({name})'), (name + '.align', f'ALIGN({name})')]
    items += [(name + '.' + field, f'offsetof({name}, {field})') for field, _ in fields]
c = [banner, '#include "cuda_abi.h"\n', 'const size_t *sq_test_c_layout(size_t *count) {\n    static const size_t values[] = {\n']
c += ['        ' + expr.replace('ALIGN(', '_Alignof(') + ',\n' for _, expr in items]
c += ['    };\n    *count = sizeof values / sizeof values[0];\n    return values;\n}\n']
cpp = [banner, '#include <cstddef>\n#include <cstdio>\n#include "../cuda/cuda_abi.cu"\n', 'using CSampleConfig = squid::SampleConfig;\nusing CLoopCDConfig = squid::LoopCDConfig;\n', '#define SQ_CUDA_ABI_NO_TYPES\n#include "cuda_abi.h"\n', 'extern "C" const size_t *sq_test_c_layout(size_t *count);\nint main() {\n    const size_t expected[] = {\n']
cpp += ['        ' + expr.replace('ALIGN(', 'alignof(') + ',\n' for _, expr in items]
cpp += ['    };\n    const char *names[] = {\n', *['        "' + name + '",\n' for name, _ in items], '    };\n    size_t count = 0;\n    const size_t *actual = sq_test_c_layout(&count);\n    if (count != sizeof expected / sizeof expected[0]) return 1;\n    for (size_t i = 0; i < count; ++i) {\n        if (actual[i] != expected[i]) {\n            std::fprintf(stderr, "%s: C=%zu C++=%zu\\n", names[i], actual[i], expected[i]);\n            return 1;\n        }\n    }\n    std::printf("CUDA ABI: 76 prototypes, 7 records, %zu layout checks passed\\n", count);\n    return 0;\n}\n']
parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
outputs = {'src/cuda_abi.h': ''.join(header), 'test/cuda_abi_c.c': ''.join(c), 'test/cuda_abi_probe.cu': ''.join(cpp)}
for name, content in outputs.items():
    path = ROOT / name
    if args.check:
        current = path.read_text()
        if current != content:
            print(''.join(difflib.unified_diff(current.splitlines(True), content.splitlines(True), fromfile=name, tofile='generated')))
            raise SystemExit('Generated CUDA ABI is stale')
    else:
        path.write_text(content)
print('Verified' if args.check else 'Generated', len(functions), 'CUDA prototypes and', len(items), 'layout checks')
