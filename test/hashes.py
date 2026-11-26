#!/usr/bin/env python3
import hashlib,subprocess,sys,random
r=random.Random(1234)
for n in [0,1,3,55,56,63,64,65,127,128,8191,8192,8193,1000000]:
    data=bytes(r.randrange(256) for _ in range(n))
    result=subprocess.check_output([sys.argv[1]],input=data).decode().strip()
    assert result==hashlib.sha256(data).hexdigest(),n
print('SHA-256 streaming matches hashlib across padding/chunk boundaries and 1 MiB input')
