#!/usr/bin/env python3
"""Resolve original upstream names without rewriting their provenance or bytes.
SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
"""
import json
from pathlib import Path


def source_path(root: Path, name: str) -> Path:
    manifest = json.loads((root / 'UPSTREAM.json').read_text())
    if name not in manifest['files']:
        raise ValueError('Not a pinned upstream file: ' + name)
    locations = manifest.get('relocated_files', {})
    if set(locations) - set(manifest['files']):
        raise ValueError('Relocation without pinned provenance')
    target = Path(locations.get(name, name))
    if target.is_absolute() or '..' in target.parts:
        raise ValueError('Upstream relocation must stay inside the checkout')
    return root / target
