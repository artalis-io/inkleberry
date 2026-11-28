#!/usr/bin/env python3
"""Optional pinned Transformers reference; original-checkpoint BF16, not IFN NVFP4 equivalence."""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def contexts(captures, differences):
    result=[]
    for d in differences:
        if d['profile']!='ar':continue
        path=captures/f"case-{d['case']}-profile-0.prompt.i32"
        data=path.read_bytes()
        if len(data)!=d['prompt_tokens']*4:raise ValueError('Prompt length mismatch')
        ids=array.array('i');ids.frombytes(data)
        if sys.byteorder!='little':ids.byteswap()
        if len(ids)!=d['prompt_tokens'] or any(not 0<=x<200058 for x in ids):
            raise ValueError('Invalid prompt IDs')
        prefix=d['shared_generated_prefix']
        if (len(prefix)!=d['first_output_index_zero_based'] or
                any(type(x) is not int or not 0<=x<200058 for x in prefix)):
            raise ValueError('Invalid shared prefix')
        result.append(dict(case=d['case'],prompt=list(ids),prefix=prefix,
                           prompt_sha256=hashlib.sha256(data).hexdigest()))
    if sorted(x['case'] for x in result)!=[4,5,6]:raise ValueError('Exactly three AR boundaries required')
    return result


def prompt_chunks(tokens):
    at=0
    while at<len(tokens):
        remain=len(tokens)-at
        count=min(2048,(remain-1)&~7) if remain>32 else min(8,remain)
        yield tokens[at:at+count]
        at+=count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--captures',type=Path,required=True)
    p.add_argument('--divergences',type=Path,required=True)
    p.add_argument('--reference-source',type=Path,required=True)
    p.add_argument('--reference-revision',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    source=a.reference_source.resolve()
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!=a.reference_revision or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise ValueError('Reference source must be clean and pinned')
    if os.environ.get('HF_HUB_OFFLINE')!='1':raise ValueError('Offline local-only execution required')
    inputs=contexts(a.captures,json.loads(a.divergences.read_text())['differences'])
    a.output.mkdir(exist_ok=False,parents=True)
    import torch
    import transformers
    from transformers import Inkling2LoopConfig, Inkling2LoopForConditionalGeneration
    if not Path(transformers.__file__).resolve().is_relative_to(source):
        raise ValueError('Imported Transformers does not match audited source')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    manifest=dict(reference_revision=revision,transformers=transformers.__version__,torch=torch.__version__,
                  source_files={str(f.relative_to(source)):digest(f) for f in [source/'src/transformers/models/inkling2_loop/modeling_inkling2_loop.py',source/'src/transformers/core_model_loading.py',source/'src/transformers/conversion_mapping.py']},
                  collector_sha256=digest(Path(__file__)),
                  representation='Original checkpoint dequantized to BF16 by official conversion; BF16 cache. '
                                 'NOT quantization-matched to Inkleberry default NVFP4/FP8-KV.',
                  model_sha256={n:digest(a.model/n) for n in ['tokenizer.json','model-00001-of-00001.safetensors','mtp.safetensors']},
                  contexts=inputs)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    config=Inkling2LoopConfig.from_pretrained(a.model,local_files_only=True)
    # Official conversion_mapping handles source expert dequantization. Do not
    # activate an unrelated ModelOpt quantizer/requantizer from envelope metadata.
    removed=getattr(config,'quantization_config',None)
    if removed is not None:delattr(config,'quantization_config')
    manifest['ignored_quantizer_envelope']=removed
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Loading independent BF16 checkpoint reference',flush=True)
    model,loading=Inkling2LoopForConditionalGeneration.from_pretrained(
        a.model,config=config,dtype=torch.bfloat16,
        local_files_only=True,trust_remote_code=False,use_kernels=False,
        attn_implementation='eager',experts_implementation='eager',output_loading_info=True)
    (a.output/'loading.json').write_text(json.dumps(loading,default=str,indent=2)+'\n')
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Reference checkpoint did not load completely; inspect loading.json')
    model.to('cuda:0')
    model.eval()
    summary=[]
    with torch.inference_mode():
        for item in inputs:
            previous=None
            for run in range(2):
                cache=None;output=None;start=time.monotonic()
                for chunk in prompt_chunks(item['prompt']):
                    ids=torch.tensor([chunk],dtype=torch.long,device='cuda:0')
                    output=model(input_ids=ids,cache=cache,use_cache=True,logits_to_keep=1)
                    cache=output.cache
                for token in item['prefix']:
                    output=model(input_ids=torch.tensor([[token]],device='cuda:0'),cache=cache,
                                 use_cache=True,logits_to_keep=1)
                    cache=output.cache
                logits=output.logits[0,-1].float().cpu().contiguous()
                if logits.numel()!=200058 or not torch.isfinite(logits).all():
                    raise ValueError('Invalid reference logits')
                payload=logits.numpy().tobytes()
                if sys.byteorder!='little':raise ValueError('Little-endian collector required')
                if previous is not None and previous!=payload:
                    raise ValueError('Reference repetitions differ')
                previous=payload
                with (a.output/f"case-{item['case']}-run-{run}.logits.f32").open('xb') as f:f.write(payload)
                row=dict(case=item['case'],run=run,seconds=time.monotonic()-start,
                         top1=int(logits.argmax()),sha256=hashlib.sha256(payload).hexdigest(),
                         max_allocated_bytes=torch.cuda.max_memory_allocated())
                summary.append(row);print(json.dumps(row),flush=True)
                del cache,output
                torch.cuda.empty_cache()
    (a.output/'summary.json').write_text(json.dumps(dict(passed=True,runs=summary),indent=2)+'\n')


if __name__=='__main__':main()
