#!/usr/bin/env python3
"""External real tokenizer parity; no GPU or model payloads required."""
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

CASES = [
    '', 'Hello, world!', 'Árvíztűrő tükörfúrógép. Gyermelyi — 24 t.',
    'こんにちは世界。 مرحبا بالعالم', '😀👩‍💻 café e\u0301\n\t  spaces',
    'def add(a, b):\n    return a + b\n', '12345678901234567890',
    "I'm don't WE'RE you'll I'd they've", ' \t \r\n\n \tend  ',
    '<|message_user|><|content_text|>hello<|end_message|>',
    'a' * 70000, ('Chapter One: Hello, world!\n' * 45000),
    ('á🚚 ' * 180000), '\x00A\x00B',
]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('library', type=Path)
    p.add_argument('model', type=Path)
    p.add_argument('--zig-oracle', type=Path)
    p.add_argument('--reference', action='store_true')
    args = p.parse_args()
    lib = c.CDLL(str(args.library.resolve()))
    lib.ifn_tokenizer_open.argtypes = [c.c_char_p, c.c_void_p, c.POINTER(c.c_void_p)]
    lib.ifn_tokenizer_open.restype = c.c_uint32
    lib.ifn_tokenizer_close.argtypes = [c.c_void_p]
    lib.ifn_tokenizer_scratch_create.argtypes = [c.c_void_p, c.POINTER(c.c_void_p)]
    lib.ifn_tokenizer_scratch_create.restype = c.c_uint32
    lib.ifn_tokenizer_scratch_destroy.argtypes = [c.c_void_p]
    lib.ifn_tokenizer_encode.argtypes = [c.c_void_p, c.c_char_p, c.c_size_t, c.c_uint32, c.c_void_p, c.POINTER(c.c_int32), c.c_size_t, c.POINTER(c.c_size_t)]
    lib.ifn_tokenizer_encode.restype = c.c_uint32
    lib.ifn_tokenizer_decode.argtypes = [c.c_void_p, c.c_int32, c.c_uint32, c.POINTER(c.c_void_p), c.POINTER(c.c_size_t)]
    lib.ifn_tokenizer_decode.restype = c.c_uint32
    lib.ifn_render_user_prompt.argtypes = [c.c_char_p, c.c_size_t, c.c_uint32, c.c_void_p, c.c_size_t, c.POINTER(c.c_size_t)]
    lib.ifn_render_user_prompt.restype = c.c_uint32
    lib.ifn_last_error.restype = c.c_char_p
    t, scratch = c.c_void_p(), c.c_void_p()
    if lib.ifn_tokenizer_open(bytes(args.model/'tokenizer.json'), None, c.byref(t)):
        raise SystemExit(lib.ifn_last_error().decode())
    if lib.ifn_tokenizer_scratch_create(None, c.byref(scratch)):
        lib.ifn_tokenizer_close(t)
        raise SystemExit(lib.ifn_last_error().decode())
    reference = None
    if args.reference:
        from tokenizers import Tokenizer
        reference = Tokenizer.from_file(str(args.model/'tokenizer.json'))
    metadata=json.loads((args.model/'tokenizer.json').read_text())
    direct=list(range(33,127))+list(range(161,173))+list(range(174,256))
    inverse={chr(b):b for b in direct}
    for i,b in enumerate(b for b in range(256) if b not in direct):
        inverse[chr(256+i)]=b
    vocab_checked=0
    try:
        entries=[(token,bytes(inverse[ch] for ch in text),False)
                 for text,token in metadata['model']['vocab'].items()]
        entries += [(entry['id'],entry['content'].encode(),True) for entry in metadata['added_tokens']]
        for token,expected,special in entries:
            data,n=c.c_void_p(),c.c_size_t()
            if lib.ifn_tokenizer_decode(t,token,0,c.byref(data),c.byref(n)) or c.string_at(data,n.value)!=expected:
                raise SystemExit('Vocabulary byte mapping mismatch at ID '+str(token))
            if special and (lib.ifn_tokenizer_decode(t,token,1,c.byref(data),c.byref(n)) or n.value):
                raise SystemExit('Special-token skip mismatch')
            vocab_checked+=1
    except BaseException:
        lib.ifn_tokenizer_scratch_destroy(scratch)
        lib.ifn_tokenizer_close(t)
        raise
    rows = []
    template_zig = []
    try:
        with tempfile.TemporaryDirectory(prefix='ifn-tokenizer-') as folder:
            for i, text in enumerate(CASES):
                raw = text.encode()
                # Every output token consumes at least one input byte.
                ids = (c.c_int32 * max(len(raw), 1))()
                needed = c.c_size_t()
                if lib.ifn_tokenizer_encode(t, raw, len(raw), 1, scratch, ids, len(ids), c.byref(needed)):
                    raise SystemExit(lib.ifn_last_error().decode())
                actual = list(ids[:needed.value])
                restored = bytearray()
                for token in actual:
                    data, n = c.c_void_p(), c.c_size_t()
                    if lib.ifn_tokenizer_decode(t, token, 0, c.byref(data), c.byref(n)):
                        raise SystemExit('decode failed')
                    restored += c.string_at(data, n.value)
                row = {'case': i, 'bytes': len(raw), 'tokens': len(actual),
                       'roundtrip': restored == raw,
                       'ids_sha256': hashlib.sha256(b''.join(int(x).to_bytes(4,'little',signed=True) for x in actual)).hexdigest()}
                if args.zig_oracle:
                    path = Path(folder)/'input.txt'
                    path.write_bytes(raw)
                    r = subprocess.run([str(args.zig_oracle), 'tokenize', str(args.model), '--prompt-file', str(path), '--raw'], env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True, timeout=180)
                    match = re.search(r'^IDS:\s*([^\n]*)', r.stdout, re.M)
                    expected = [int(x) for x in match[1].split(',') if x.strip()] if match else None
                    row['zig'] = r.returncode == 0 and actual == expected
                if reference:
                    row['provider'] = actual == reference.encode(text, add_special_tokens=False).ids
                rows.append(row)
            for effort, level in enumerate(['0','0.1','0.2','0.7','0.9','0.99']):
                text = 'Árvíztűrő 🚚'.encode()
                expected = ('<|message_system|><|content_text|>Thinking effort level: '+level+'<|end_message|><|message_user|><|content_text|>').encode()+text+b'<|end_message|>'
                needed = c.c_size_t()
                out = c.create_string_buffer(len(expected))
                if lib.ifn_render_user_prompt(text,len(text),effort,out,len(out),c.byref(needed)) or out.raw != expected:
                    raise SystemExit('template mismatch')
                if args.zig_oracle:
                    ids=(c.c_int32*len(expected))()
                    n=c.c_size_t()
                    if lib.ifn_tokenizer_encode(t,out.raw,len(out.raw),1,scratch,ids,len(ids),c.byref(n)):
                        raise SystemExit('template encoding failed')
                    path=Path(folder)/'template.txt';path.write_bytes(text)
                    label=['none','minimal','low','medium','high','max'][effort]
                    r=subprocess.run([str(args.zig_oracle),'tokenize',str(args.model),'--prompt-file',str(path),'--effort',label],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},text=True,capture_output=True,timeout=30)
                    match=re.search(r'^IDS:\s*([^\n]*)',r.stdout,re.M)
                    wanted=[int(x) for x in match[1].split(',') if x.strip()] if match else None
                    template_zig.append(r.returncode==0 and list(ids[:n.value])==wanted)

    finally:
        lib.ifn_tokenizer_scratch_destroy(scratch)
        lib.ifn_tokenizer_close(t)
    print(json.dumps({'vocabulary_entries_matched': vocab_checked, 'tokenizer_cases': rows, 'template_efforts_matched': 6, 'template_zig': template_zig}, indent=2))
    if not all(template_zig):raise SystemExit('Zig template token mismatch')
    if not all(r['roundtrip'] and r.get('zig', True) and r.get('provider', True) for r in rows):
        raise SystemExit('Token parity mismatch')

if __name__ == '__main__':
    main()
