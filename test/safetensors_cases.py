#!/usr/bin/env python3
"""Non-confidential fixtures; independent JSON/format edge cases via the real CLI."""
import argparse,json,os,struct,subprocess,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('binary',nargs='?',default='./inkleberry');p.add_argument('--corpus');a=p.parse_args()
def tensor(dtype='U8',shape=None,offsets=None):
 return {'dtype':dtype,'shape':[1] if shape is None else shape,'data_offsets':[0,1] if offsets is None else offsets}
def encode(header,payload=b'\x11'):
 if not isinstance(header,bytes):header=json.dumps(header,ensure_ascii=False,separators=(',',':')).encode()
 return struct.pack('<Q',len(header))+header+payload
cases=[]
def case(name,header,ok=False,payload=b'\x11'):
 cases.append((name,encode(header,payload),ok))
case('valid',{'a':tensor()},True)
case('scalar',{'a':tensor(shape=[])},True)
case('zero',{'a':tensor(shape=[0],offsets=[0,0])},True,b'')
case('empty',{},True,b'')
case('metadata',{'__metadata__':{'format':'pt'},'a':tensor()},True)
case('unicode',{'árvíztűrő🚚':tensor()},True)
case('surrogate-pair',b'{"\\ud83d\\ude9a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}',True)
case('out-of-order',{'b':tensor(offsets=[1,2]),'a':tensor()},True,b'\x11\x12')
case('all-dtypes',{'u':tensor(),'b':tensor('BF16',[1],[1,3]),'f':tensor('F32',[1],[3,7]),'q':tensor('F8_E4M3',[1],[7,8])},True,b'\0'*8)
case('padding',json.dumps({'a':tensor()}).encode()+b' '*13,True)
for name,value in [
 ('duplicate-tensor',b'{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[0],"data_offsets":[1,1]}}'),
 ('escaped-duplicate',b'{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"\\u0061":{"dtype":"U8","shape":[0],"data_offsets":[1,1]}}'),
 ('duplicate-field',b'{"a":{"dtype":"U8","dtype":"U8","shape":[1],"data_offsets":[0,1]}}'),
 ('nul',b'{"\\u0000":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'),
 ('unpaired-high',b'{"\\ud800":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'),
 ('unpaired-low',b'{"\\udc00":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'),
 ('invalid-utf8',b'{"\xc0\x80":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'),
 ('leading-zero',b'{"a":{"dtype":"U8","shape":[01],"data_offsets":[0,1]}}'),
 ('fraction',b'{"a":{"dtype":"U8","shape":[1.0],"data_offsets":[0,1]}}'),
 ('trailing-root',b'{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},}'),
 ('trailing-shape',b'{"a":{"dtype":"U8","shape":[1,],"data_offsets":[0,1]}}'),
 ('trailing-data',json.dumps({'a':tensor()}).encode()+b'null'),
 ('wrong-root',b'[]'),('unterminated',b'{"a":'),
]:case(name,value)
for name,t in [('dtype',tensor('F64')),('missing',{'shape':[1],'data_offsets':[0,1]}),
 ('rank',tensor(shape=[1]*9)),('size',tensor('BF16')),('negative',tensor(shape=[-1])),
 ('uint-overflow',tensor(shape=[2**64])),('shape-overflow',tensor(shape=[2**63,2])),
 ('reverse',tensor(offsets=[1,0])),('past-end',tensor(offsets=[1,2])),
 ('unknown-field',{**tensor(),'extra':False})]:case(name,{'a':t})
case('overlap',{'a':tensor(),'b':tensor()})
case('gap',{'a':tensor(offsets=[1,2])},False,b'\0\0')
case('leftover',{'a':tensor()},False,b'\0\0')
case('invalid-metadata',{'__metadata__':{'format':1},'a':tensor()})
case('too-many',{str(i):tensor(shape=[0],offsets=[0,0]) for i in range(2049)},False,b'')
case('max-tensors',{str(i):tensor(shape=[0],offsets=[0,0]) for i in range(2048)},True,b'')
case('name-limit',{'x'*255:tensor()},True)
case('name-overflow',{'x'*256:tensor()})
cases += [('empty-bytes',b'',False),('truncated-prefix',b'\0'*7,False),
 ('huge-header',struct.pack('<Q',2**64-1)+b'{}',False),
 ('header-limit',struct.pack('<Q',16*1024*1024+1)+b'{}',False)]
if a.corpus:
 root=Path(a.corpus);root.mkdir(parents=True,exist_ok=True)
 for name,data,_ in cases:(root/name).write_bytes(data)
 print('Wrote',len(cases),'synthetic corpus seeds')
else:
 with tempfile.TemporaryDirectory() as folder:
  path=Path(folder)/'fixture.safetensors'
  for name,data,ok in cases:
   path.write_bytes(data)
   r=subprocess.run([a.binary,'inspect',str(path)],capture_output=True,text=True,timeout=15)
   if r.returncode!=(0 if ok else 1) or 'Sanitizer' in r.stderr or 'runtime error:' in r.stderr:raise SystemExit(f'{name}: expected success={ok}, rc={r.returncode}: {r.stderr}')
   if ok:assert json.loads(r.stdout)['format']=='safetensors'
  for name in ('fifo', 'directory', 'missing'):
   special=Path(folder)/name
   if name=='fifo':os.mkfifo(special)
   if name=='directory':special.mkdir()
   r=subprocess.run([a.binary,'inspect',str(special)],capture_output=True,text=True,timeout=5)
   if r.returncode!=1 or 'Sanitizer' in r.stderr or 'runtime error:' in r.stderr:raise SystemExit(name+': expected normal failure, got '+str(r.returncode)+r.stderr)
 print(len(cases),'CLI format cases and 3 filesystem rejection cases passed')
