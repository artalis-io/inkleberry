#!/usr/bin/env python3
"""Mismatch and missing-artifact failures must happen before any native upload."""
import hashlib,subprocess,sys,tempfile
from pathlib import Path
fixture=Path(sys.argv[2]).read_bytes()
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp)/'tokenizer.json').write_bytes(fixture)
    subprocess.run([sys.argv[1],'--reject-manifest',tmp,hashlib.sha256(fixture).hexdigest()],check=True)
print('Exact tokenizer checksum enforced before native upload; OS errors retain errno')
