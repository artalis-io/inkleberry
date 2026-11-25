#!/usr/bin/env python3
"""Reject unsupported configurations/malformed tokenizers without leaks or crashes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from tokenizer_fixtures import fixture

cases = []
def change(name, mutate):
    f = fixture()
    mutate(f)
    cases.append((name, json.dumps(f,ensure_ascii=False).encode()))
change('normalizer', lambda f:f.update(normalizer={'type':'NFC'}))
change('regex', lambda f:f['pre_tokenizer']['pretokenizers'][0]['pattern'].update(Regex='.*'))
change('dropout', lambda f:f['model'].update(dropout=0.1))
change('unknown-model-field', lambda f:f['model'].update(unknown=False))
change('missing-byte', lambda f:f['model']['vocab'].pop('Ā'))
change('duplicate-id', lambda f:f['model']['vocab'].update(ab=97))
change('negative-id', lambda f:f['model']['vocab'].update(ab=-1))
change('large-id', lambda f:f['model']['vocab'].update(ab=204800))
change('fraction-id', lambda f:f['model']['vocab'].update(ab=256.5))
change('invalid-alphabet', lambda f:f['model']['vocab'].update({'🚚':260}))
change('empty-token', lambda f:f['model']['vocab'].update({'':260}))
change('long-token', lambda f:f['model']['vocab'].update({'x'*4096:260}))
change('missing-merge-parent', lambda f:f['model']['merges'].append(['a','missing']))
change('long-merge-array', lambda f:f['model']['merges'].append(['a','b','c']))
change('empty-merge', lambda f:f['model']['merges'].append(['','a']))
change('invalid-merge-string', lambda f:f['model']['merges'].append('abc'))
change('bad-merge-type', lambda f:f['model']['merges'].append(5))
change('empty-vocab', lambda f:f['model'].update(vocab={}))
change('added-lstrip', lambda f:f['added_tokens'][0].update(lstrip=True))
change('added-normalized', lambda f:f['added_tokens'][0].update(normalized=True))
change('added-non-special', lambda f:f['added_tokens'][0].update(special=False))
change('added-prefix', lambda f:f['added_tokens'][0].update(content='stop'))
change('added-overlap-id', lambda f:f['added_tokens'][0].update(id=0))
change('added-duplicate', lambda f:f['added_tokens'].append({**f['added_tokens'][0], 'id':301}))
change('added-missing-content', lambda f:f['added_tokens'][0].pop('content'))
change('unknown-root', lambda f:f.update(unknown=1))
def deep_normalizer(f):
    value=None
    for _ in range(40):value=[value]
    f['normalizer']=value
change('deep-root',deep_normalizer)
raw=json.dumps(fixture(),ensure_ascii=False).encode()
cases += [('truncated', raw[:-1]), ('trailing',raw+b'null'),
          ('duplicate-root',raw.replace(b'"version": "1.0"',b'"version": "1.0", "version":"1.0"',1)),
          ('bad-utf8',raw.replace(b'"version"',b'"\xc0\x80"',1)),
          ('bad-escape',raw.replace(b'"version"',b'"\\q"',1)),
          ('trailing-comma',raw[:-1]+b',}'),
          ('duplicate-vocab-name',raw.replace(b'"aaaa": 259',b'"aaaa": 259, "a":260',1)),
          ('escaped-duplicate-name',raw.replace(b'"aaaa": 259',b'"aaaa": 259, "\\u0061":260',1))]
if len(sys.argv)>1 and sys.argv[1]=='--corpus':
    folder=Path(sys.argv[2]);folder.mkdir(parents=True,exist_ok=True)
    (folder/'valid').write_bytes(raw)
    for name,data in cases:(folder/name).write_bytes(data)
else:
    binary=sys.argv[1]
    with tempfile.TemporaryDirectory() as folder:
        for name,data in cases:
            p=Path(folder)/'bad.json';p.write_bytes(data)
            r=subprocess.run([binary,'--invalid',str(p)],text=True,capture_output=True,timeout=15)
            if r.returncode or 'Sanitizer' in r.stderr or 'runtime error:' in r.stderr:
                raise SystemExit(name+': '+r.stderr)
        if len(sys.argv)>2:
            p.write_text(json.dumps(fixture(),ensure_ascii=False))
            actual=json.loads(subprocess.check_output([sys.argv[2],'tokenize',str(p),'abc aaaa<|eos|>'],text=True))
            if actual != [257,32,259,300]:raise SystemExit('CLI tokenize mismatch')
            if json.loads(subprocess.check_output([sys.argv[2],'tokenize',str(p),''],text=True)) != []:raise SystemExit('CLI empty tokenization mismatch')
    print(len(cases),'tokenizer rejection cases passed with allocator cleanup checked')
