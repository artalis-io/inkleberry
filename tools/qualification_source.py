#!/usr/bin/env python3
"""Source identity for clean checkouts or explicitly attested source archives."""
import json
from pathlib import Path
import re
import subprocess
from model_release import digest


def files(root):
    names=json.loads((root/'config/package.json').read_text())['source_files']
    if not names or len(set(names))!=len(names):raise ValueError('Invalid source inventory')
    for name in names:
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or p.as_posix()!=name:
            raise ValueError('Unsafe source inventory path')
        if not (root/p).is_file() or (root/p).is_symlink():raise ValueError('Missing/nonregular source input: '+name)
    return {name:digest(root/name) for name in names}


def git(root,*args):
    p=subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True,timeout=30)
    if p.returncode:raise ValueError('Source checkout unavailable or Git command failed')
    return p.stdout.strip()


def verify(root,revision,archive=None):
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Exact source revision required')
    inventory=files(root)
    if archive:
        record=json.loads(archive.read_text())
        if (record.get('format_version')!=1 or record.get('source_revision')!=revision or
                record.get('source_files')!=inventory):
            raise ValueError('Source archive manifest mismatch')
        return {'mode':'archive-manifest','source_revision':revision,'source_files':inventory,
                'manifest_sha256':digest(archive),
                'association':'operator-attested archive; file hashes verified, not a signed build attestation'}
    # Relative Git prefix avoids MSYS/native-Windows spelling differences.
    if git(root,'rev-parse','--is-inside-work-tree')!='true' or git(root,'rev-parse','--show-prefix'):
        raise ValueError('Source checkout root mismatch')
    if git(root,'rev-parse','HEAD')!=revision:raise ValueError('Claimed source revision differs from HEAD')
    if git(root,'status','--porcelain','--untracked-files=all'):
        raise ValueError('Source verification requires a clean checkout')
    return {'mode':'clean-checkout','source_revision':revision,'source_files':inventory,
            'association':'checkout verified; executable separately identified, not a signed build attestation'}


def capture(root,out):
    revision=git(root,'rev-parse','HEAD')
    record=verify(root,revision)
    with out.open('x') as f:
        json.dump({'format_version':1,'source_revision':revision,'source_files':record['source_files']},f,indent=2)
        f.write('\n')
