#!/usr/bin/env python3
"""Reversible identifier-only transformation for the retained native implementation.
SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
"""
import json
import re
from pathlib import Path
from amalgamate import TOKEN

ROOT = Path(__file__).resolve().parents[1]
NATIVE_NOTICE = '// Inkleberry Finn: identifier-only namespace adaptation by Artalis; see config/native-namespace.json.\n'


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


def original_bytes(text, root=ROOT):
    if not text.startswith(NATIVE_NOTICE):
        raise ValueError('Missing native modification notice')
    mapping = names(root)
    return translate(text[len(NATIVE_NOTICE):], {v:k for k,v in mapping.items()}).encode()


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
