#!/usr/bin/env python3
"""Synthetic byte-complete tokenizer fixture; no provider data is embedded."""
import json
from pathlib import Path
import sys

def fixture():
    direct = list(range(33,127))+list(range(161,173))+list(range(174,256))
    cp = {b:b for b in direct}
    for i,b in enumerate(b for b in range(256) if b not in cp):
        cp[b] = 256+i
    vocab = {chr(cp[b]):b for b in range(256)}
    vocab.update({'ab':256, 'abc':257, 'aa':258, 'aaaa':259})
    pattern = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    return {'version':'1.0','truncation':None,'padding':None,
        'added_tokens':[{'id':300,'content':'<|eos|>','single_word':False,'lstrip':False,'rstrip':False,'normalized':False,'special':True}],
        'normalizer':None,'pre_tokenizer':{'type':'Sequence','pretokenizers':[
            {'type':'Split','pattern':{'Regex':pattern},'behavior':'Isolated','invert':False},
            {'type':'ByteLevel','add_prefix_space':False,'trim_offsets':True,'use_regex':False}]},
        'post_processor':{'type':'ByteLevel','add_prefix_space':True,'trim_offsets':False,'use_regex':True},
        'decoder':{'type':'ByteLevel','add_prefix_space':True,'trim_offsets':True,'use_regex':True},
        'model':{'type':'BPE','dropout':None,'unk_token':None,'continuing_subword_prefix':None,
            'end_of_word_suffix':None,'fuse_unk':False,'byte_fallback':False,'ignore_merges':True,
            'vocab':vocab,'merges':[['a','b'],['ab','c'],['a','a'],['aa','aa']]}}

if __name__ == '__main__':
    p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(fixture(),ensure_ascii=False))
