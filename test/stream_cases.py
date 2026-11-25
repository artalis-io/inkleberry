#!/usr/bin/env python3
"""Deterministic Unicode replacement/leftmost-stop oracle across partitions."""
import random,subprocess,sys
r=random.Random(20261125)
cases=[(b'abc',['b','abc']),(b'abX',['b','abc']),(b'ab',['b','abc']),('a🙂xZ'.encode(),['🙂x','x']),('á🚚 café'.encode(),['🚚']),(b'\xed\xa0\x80\xf4\x90\x80\x80\xe2\x82',[])]
cases += [(b'a'*64,['a'*64]),(b'a'*63+b'x',['a'*64]),(('a'*61+'🙂').encode(),['a'*61+'€']),(('🙂'*16).encode(),['🙂'*16])]
for i in range(80):
    raw=bytes(r.randrange(256) for _ in range(r.randrange(1,150)))
    if i%2:raw=b'abc'+raw+b'STOP'
    cases.append((raw,['b','abc','STOP'] if i%3 else []))
for data,stops in cases:
    normalized=data.decode('utf8','replace').encode()
    offsets=[normalized.find(s.encode()) for s in stops if s.encode() in normalized]
    end=min(offsets) if offsets else len(normalized);expected=normalized[:end]
    for chunk in sorted({1,2,3,7,max(len(data),1)}):
        p=subprocess.run([sys.argv[1],data.hex(),str(chunk),*[s.encode().hex() for s in stops]],capture_output=True,text=True,check=True)
        lines=p.stdout.splitlines();actual=bytes.fromhex(lines[0]);reason,replacements=map(int,lines[1].split())
        assert actual==expected,(data,stops,chunk,actual,expected)
        assert reason==(1 if offsets else 3),(data,reason,offsets)
        if not stops:assert replacements==data.decode('utf8','replace').count('\ufffd')
print(f'{len(cases)} Unicode/stop fixtures matched reference across chunk partitions')
