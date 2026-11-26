#!/usr/bin/env python3
"""Oracle-only visibility adapter; never modifies hash-pinned originals."""
from pathlib import Path
import shutil
import sys
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tools'))
from upstream import source_path
out=root/'.build/chat-oracle-src';out.mkdir(parents=True,exist_ok=True)
for source in source_path(root, 'src/server.zig').parent.glob('*.zig'):shutil.copy2(source,out/source.name)
p=out/'server.zig';s=p.read_text()
for before,after in [
 ('const tiger = @import','pub const tiger = @import'),
 ('const tokenizer_mod = @import','pub const tokenizer_mod = @import'),
 ('const ServerBuffers = struct','pub const ServerBuffers = struct'),
 ('    fn init() ServerBuffers','    pub fn init() ServerBuffers'),
 ('    fn deinit(self: *ServerBuffers)','    pub fn deinit(self: *ServerBuffers)'),
 ('fn encode_chat_template_zero_alloc(','pub fn encode_chat_template_zero_alloc(')]:
    assert s.count(before)==1,before
    s=s.replace(before,after)
p.write_text(s)
