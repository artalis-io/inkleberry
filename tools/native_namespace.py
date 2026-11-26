#!/usr/bin/env python3
"""Reversible identifiers and reviewed host adaptations; device math remains pinned.
SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
"""
import json
import re
from pathlib import Path
from amalgamate import TOKEN

ROOT = Path(__file__).resolve().parents[1]
NATIVE_NOTICE = '// Inkleberry Finn: namespace and checked host error adaptation by Artalis; see config/native-namespace.json and config/native-host-adaptations.json.\n'


def names(root=ROOT):
    mapping = json.loads((root / 'config/native-namespace.json').read_text())
    if (not isinstance(mapping, dict) or not mapping or
            any(not isinstance(old, str) or not isinstance(new, str) or
                not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', old) or old.startswith('__') or
                not re.fullmatch(r'(?:ifn_|IFN_|INKLEBERRY_)[A-Za-z_0-9]+', new)
                for old, new in mapping.items()) or
            len(set(mapping.values())) != len(mapping)):
        raise ValueError('Native namespace mapping is not reversible')
    return mapping


def translate(text, mapping):
    output = []
    previous = ''
    for token in TOKEN.findall(text):
        # Preserve literals, comments, member names and arithmetic byte for byte.
        # The one renamed environment flag is an explicit string-literal change.
        if token.startswith('// namespace '):
            changed = re.sub(r'\b[A-Za-z_]\w*\b', lambda m:mapping.get(m[0],m[0]), token)
        elif token.startswith('"') and token[1:-1] in ('SQUID_MTP_BF16', 'INKLEBERRY_MTP_BF16') and token[1:-1] in mapping:
            changed = '"' + mapping[token[1:-1]] + '"'
        else:
            changed = mapping.get(token, token) if previous not in ('.', '->') else token
        output.append(changed)
        if not token.isspace() and not token.startswith(('/*', '//')):
            previous = token
    return ''.join(output)


def platform_patches(root=ROOT):
    value = json.loads((root / 'config/native-platform-adaptations.json').read_text())
    allowed = {'host-launch-policy': 'cuda/kernels_common.cuh',
               'host-resource-policy': 'cuda/kernels_prefill.cu'}
    if (not isinstance(value, dict) or set(value) != {'format_version', 'patches'} or
            value['format_version'] != 1 or not isinstance(value['patches'], list)):
        raise ValueError('Invalid platform adaptation manifest')
    seen = set()
    for patch in value['patches']:
        if (not isinstance(patch, dict) or set(patch) != {'path', 'kind', 'before', 'after'} or
                patch.get('kind') not in allowed or patch['path'] != allowed[patch['kind']] or
                patch['kind'] in seen or
                any(not isinstance(patch[k], str) or not patch[k] for k in ('before', 'after')) or
                any(t in patch['before'] or t in patch['after']
                    for t in ('__global__', '__device__', '__launch_bounds__'))):
            raise ValueError('Invalid host-only platform adaptation')
        if patch['kind'] == 'host-launch-policy':
            if (not patch['before'].startswith('template <class... KArgs, class... Args>\ninline void ifn_cuda_launch_pdl(') or
                    patch['after'] != '// Host target policy: config/native-platform-adaptations.json.\n#include "../src/cuda/launch_policy.hpp"\n'):
                raise ValueError('Platform launch hook escaped its reviewed host scope')
        else:
            original = ('    static const cudaError_t attr = cudaFuncSetAttribute(\n'
                        '        ifn_cuda_p_wgemm_kernel<FMT, MODE, K, TM>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);\n'
                        '    (void)attr;')
            expected = ('#if IFN_CUDA_USE_PDL\n' + original + '\n#endif\n'
                        '    // Ampere opt-in is checked per context during ifn_blas_create, before capture.')
            if patch['before'] != original or patch['after'] != expected:
                raise ValueError('Platform resource hook escaped its reviewed host scope')
        seen.add(patch['kind'])
    if seen != set(allowed):
        raise ValueError('Incomplete platform adaptation manifest')
    return value['patches']


def original_bytes(text, root=ROOT, path=None):
    if not text.startswith(NATIVE_NOTICE):
        raise ValueError('Missing native modification notice')
    text = text[len(NATIVE_NOTICE):]
    for patch in reversed(platform_patches(root)):
        if path is not None and path != patch['path']:
            continue
        count = text.count(patch['after'])
        if count > 1 or (path == patch['path'] and count != 1):
            raise ValueError('Missing or ambiguous platform adaptation')
        if count:
            text = text.replace(patch['after'], patch['before'], 1)
    patches = json.loads((root / 'config/native-host-adaptations.json').read_text())
    # Patches are ordered edits; reverse in the opposite order. Exact strings
    # preserve all unreviewed kernel bytes and arithmetic under original hashes.
    for patch in reversed(patches):
        if patch['after'] in text:
            if text.count(patch['after']) != 1:
                raise ValueError('Ambiguous host adaptation')
            text = text.replace(patch['after'], patch['before'], 1)
    mapping = names(root)
    return translate(text, {v:k for k,v in mapping.items()}).encode()


def kernel_names(text):
    """Read CUDA kernel names, balancing launch-bound attribute expressions."""
    tokens = [t for t in TOKEN.findall(text) if not t.isspace() and
              not t.startswith(('/*', '//'))]
    result = []
    for i, token in enumerate(tokens):
        if token != '__global__' or tokens[i + 1:i + 2] != ['void']:
            continue
        at = i + 2
        if tokens[at:at + 2] == ['__launch_bounds__', '(']:
            at += 2
            depth = 1
            while at < len(tokens) and depth:
                depth += (tokens[at] == '(') - (tokens[at] == ')')
                at += 1
            if depth:
                raise ValueError('Unbalanced CUDA launch-bound attribute')
        if (at + 1 >= len(tokens) or not re.fullmatch(r'[A-Za-z_]\w*', tokens[at]) or
                tokens[at + 1] != '(' or tokens[at].startswith('__')):
            raise ValueError('Unrecognized CUDA kernel declaration')
        result.append(tokens[at])
    return result
