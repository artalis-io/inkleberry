#!/usr/bin/env python3
"""Generate the native C/CUDA contract from a language-neutral schema.
No Zig source/compiler, upstream manifest or third-party Python dependency needed.
SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
"""
import argparse
import difflib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCALARS = {'uint8': 'uint8_t', 'uint32': 'uint32_t', 'uint64': 'uint64_t',
           'int32': 'int32_t', 'float32': 'float', 'usize': 'size_t', 'void': 'void'}
KEYWORDS = set(('auto bool break case char class const constexpr continue default delete '
                'do double else enum explicit extern false float for friend goto if inline '
                'int long namespace new nullptr operator private protected public register '
                'restrict return short signed sizeof static struct switch template this '
                'thread_local throw true try typedef typename union unsigned using virtual '
                'alignas alignof and and_eq asm bitand bitor catch char8_t char16_t '
                'char32_t compl concept consteval constinit const_cast co_await co_return '
                'co_yield decltype dynamic_cast export mutable noexcept not not_eq or '
                'or_eq reinterpret_cast requires static_assert static_cast typeid wchar_t xor xor_eq '
                'void volatile while _Alignas _Alignof _Atomic _Bool _Complex _Generic '
                '_Imaginary _Noreturn _Static_assert _Thread_local').split())


def fail(message):
    raise ValueError(message)


def object_keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        fail('Missing or unknown schema keys')


def identifier(value):
    if not isinstance(value, str) or len(value) > 128 or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value) or value in KEYWORDS:
        fail('Invalid C/C++ identifier')
    return value


def ctype(value, names, depth=0):
    if not isinstance(value, str) or len(value) > 256 or depth > 8:
        fail('Invalid or excessively nested type')
    if value.startswith('pointer<') and value.endswith('>'):
        inner = value[8:-1]
        const = inner.startswith('const,')
        base = ctype(inner[6:] if const else inner, names, depth + 1)
        if const:
            base = base + ' const' if base.endswith('*') else 'const ' + base
        return base + ' *'
    if value in SCALARS:
        return SCALARS[value]
    if value in names:
        return value
    fail('Unknown ABI type: ' + value)


def members(values, names, arrays=False):
    if not isinstance(values, list) or len(values) > 256:
        fail('Invalid or excessive fields/parameters')
    seen = set()
    for field in values:
        object_keys(field, ('name', 'type'), ('count',) if arrays else ())
        name = identifier(field['name'])
        if name in seen:
            fail('Duplicate field/parameter: ' + name)
        seen.add(name)
        ctype(field['type'], names)
        if field['type'] == 'void':
            fail('Void must be behind a pointer')
        if 'count' in field and (type(field['count']) is not int or not 1 <= field['count'] <= 65536):
            fail('Array count must be an integer in 1..65536')


def validate(schema):
    object_keys(schema, ('schema_version', 'provenance', 'records', 'functions'))
    if type(schema['schema_version']) is not int or schema['schema_version'] != 1:
        fail('Unsupported ABI schema version')
    object_keys(schema['provenance'], ('repository', 'revision', 'bootstrap_commit', 'bootstrap'))
    if any(not isinstance(v, str) or not v or len(v) > 1024 for v in schema['provenance'].values()):
        fail('Invalid provenance metadata')
    if not re.fullmatch(r'[0-9a-f]{40}', schema['provenance']['revision']):
        fail('Invalid upstream revision')
    if not isinstance(schema['records'], list) or not 1 <= len(schema['records']) <= 64:
        fail('Expected 1..64 records')
    names = set()
    cpp_names = set()
    for record in schema['records']:
        object_keys(record, ('name', 'fields'), ('cpp_type',))
        name = identifier(record['name'])
        if name in names or name in SCALARS or name in SCALARS.values():
            fail('Duplicate or reserved record name: ' + name)
        if not isinstance(record['fields'], list) or not record['fields']:
            fail('Empty records are not portable C')
        # By-value members must be declared before use; forward/cyclic types are unsupported.
        members(record['fields'], names, arrays=True)
        cpp_name = record.get('cpp_type', name)
        if not isinstance(cpp_name, str) or any(not part for part in cpp_name.split('::')):
            fail('Invalid CUDA C++ type name')
        for part in cpp_name.split('::'):
            identifier(part)
        if cpp_name in cpp_names:
            fail('Duplicate CUDA C++ type')
        cpp_names.add(cpp_name)
        names.add(name)
    if not isinstance(schema['functions'], list) or not 1 <= len(schema['functions']) <= 512:
        fail('Expected 1..512 functions')
    seen = set(names) | set(SCALARS.values())
    for function in schema['functions']:
        object_keys(function, ('name', 'returns', 'parameters'))
        name = identifier(function['name'])
        if name in seen:
            fail('Duplicate symbol: ' + name)
        seen.add(name)
        ctype(function['returns'], names)
        members(function['parameters'], names)
    return schema


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def load_schema(path):
    with path.open('rb') as source:
        data = source.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        fail('ABI schema exceeds 1 MiB')
    return validate(json.loads(data, object_pairs_hook=unique_object,
                               parse_constant=lambda value: fail('Invalid JSON constant: ' + value)))


def generate(schema):
    validate(schema)
    NAMES = {r['name'] for r in schema['records']}
    def declaration(field):
        result = ctype(field['type'], NAMES) + ' ' + field['name']
        if 'count' in field:
            result += '[' + str(field['count']) + ']'
        return result
    banner = '/* Generated from abi/cuda.json by tools/generate_cuda_abi.py; do not edit.\n * SPDX-License-Identifier: Apache-2.0\n * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis */\n'
    header = [banner, '#ifndef SQ_CUDA_ABI_H\n#define SQ_CUDA_ABI_H\n#include <stddef.h>\n#include <stdint.h>\n', '#ifndef SQ_CUDA_ABI_NO_TYPES\n']
    for record in schema['records']:
        name, fields = record['name'], record['fields']
        header += [f'typedef struct {name} {{\n', *['    ' + declaration(f) + ';\n' for f in fields], f'}} {name};\n']
    header += ['#endif\n#ifdef __cplusplus\nextern "C" {\n#endif\n']
    functions = schema['functions']
    for function in functions:
        params = [declaration(p) for p in function['parameters']]
        header.append(ctype(function['returns'], NAMES) + ' ' + function['name'] + '(' + (', '.join(params) if params else 'void') + ');\n')
    header += ['#ifdef __cplusplus\n}\n#endif\n#endif\n']
    # Two translation units compare the actual C and C++ representations. The
    # C++ TU includes the original bridge (no copied substitute layout) and then
    # redeclares every function using the generated header: prototype conflicts fail.
    items = []
    for record in schema['records']:
        name, fields = record['name'], record['fields']
        items += [(name + '.size', f'sizeof({name})'), (name + '.align', f'ALIGN({name})')]
        items += [(name + '.' + field, f'offsetof({name}, {field})') for field in [f['name'] for f in fields]]
    c = [banner, '#include "cuda_abi.h"\n', 'const size_t *sq_test_c_layout(size_t *count) {\n    static const size_t values[] = {\n']
    c += ['        ' + expr.replace('ALIGN(', '_Alignof(') + ',\n' for _, expr in items]
    c += ['    };\n    *count = sizeof values / sizeof values[0];\n    return values;\n}\n']
    cpp = [banner, '#include <cstddef>\n#include <cstdio>\n#include "../cuda/cuda_abi.cu"\n', ''.join('using ' + r['name'] + ' = ' + r['cpp_type'] + ';\n' for r in schema['records'] if 'cpp_type' in r), '#define SQ_CUDA_ABI_NO_TYPES\n#include "cuda_abi.h"\n', 'extern "C" const size_t *sq_test_c_layout(size_t *count);\nint main() {\n    const size_t expected[] = {\n']
    cpp += ['        ' + expr.replace('ALIGN(', 'alignof(') + ',\n' for _, expr in items]
    cpp += ['    };\n    const char *names[] = {\n', *['        "' + name + '",\n' for name, _ in items], '    };\n    size_t count = 0;\n    const size_t *actual = sq_test_c_layout(&count);\n    if (count != sizeof expected / sizeof expected[0]) return 1;\n    for (size_t i = 0; i < count; ++i) {\n        if (actual[i] != expected[i]) {\n            std::fprintf(stderr, "%s: C=%zu C++=%zu\\n", names[i], actual[i], expected[i]);\n            return 1;\n        }\n    }\n    std::printf("CUDA ABI: ' + str(len(functions)) + ' prototypes, ' + str(len(schema['records'])) + ' records, %zu layout checks passed\\n", count);\n    return 0;\n}\n']
    return {'src/cuda_abi.h': ''.join(header), 'test/cuda_abi_c.c': ''.join(c), 'test/cuda_abi_probe.cu': ''.join(cpp)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schema', type=Path, default=ROOT / 'abi/cuda.json')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    try:
        schema = load_schema(args.schema)
        outputs = generate(schema)
    except (ValueError, OSError, RecursionError) as error:
        parser.exit(1, 'Invalid CUDA ABI schema: ' + str(error) + '\n')
    for name, content in outputs.items():
        path = args.output_root / name
        if args.check:
            try:
                current = path.read_text()
            except OSError as error:
                parser.exit(1, 'Cannot check generated CUDA ABI: ' + str(error) + '\n')
            if current != content:
                print(''.join(difflib.unified_diff(current.splitlines(True), content.splitlines(True), fromfile=name, tofile='generated')))
                parser.exit(1, 'Generated CUDA ABI is stale\n')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    layouts = sum(2 + len(r['fields']) for r in schema['records'])
    print('Verified' if args.check else 'Generated', len(schema['functions']),
          'CUDA prototypes and', layouts, 'layout checks')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
