#!/usr/bin/env python3
"""Compare exact-prefix boundary captures; no cross-device quality threshold is inferred."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import sys

from first_divergence import load as load_benchmark

PROFILES = ('ar', 'fixed', 'adaptive', 'sampled')


def floats(path, count):
    data = path.read_bytes()
    if len(data) != count * 4:
        raise ValueError('Truncated or wrong-sized float payload: ' + str(path))
    values = array.array('f')
    values.frombytes(data)
    if sys.byteorder != 'little':
        values.byteswap()
    if any(not math.isfinite(x) for x in values):
        raise ValueError('Non-finite capture: ' + str(path))
    return values


def read_capture(root, baseline, differences):
    rows = [json.loads(line) for line in (root / 'capture.jsonl').read_text().splitlines()
            if not line.startswith('Idle GPU preflight:')]
    index = {}
    for row in rows:
        if row['kind'] not in ('capture', 'result'):
            raise ValueError('Unknown capture record')
        key = (row['case'], row['profile'], row['kind'])
        if key in index:
            raise ValueError('Duplicate capture record')
        index[key] = row
    expected = {(d['case'], PROFILES.index(d['profile']), kind)
                for d in differences for kind in ('capture', 'result')}
    if set(index) != expected:
        raise ValueError('Missing/extra boundary captures')
    result = {}
    historical = load_benchmark(baseline)
    for d in differences:
        case, profile = d['case'], PROFILES.index(d['profile'])
        c, r = index[case, profile, 'capture'], index[case, profile, 'result']
        if (c['output_index'] != d['first_output_index_zero_based'] or
                type(c['width']) is not int or not 1 <= c['width'] <= 8 or
                type(c['row']) is not int or not 0 <= c['row'] < c['width'] or
                r['capture_preserved_output'] is not True or
                r['prompt_tokens'] != d['prompt_tokens'] or
                r['tokens'] != historical[case, d['profile']]['tokens'] or
                r['tokens'][:c['output_index']] != d['shared_generated_prefix'] or
                r['tokens'][c['output_index']] != c['selected_token']):
            raise ValueError('Capture did not reproduce historical path/prefix')
        stem = root / f'case-{case}-profile-{profile}'
        logits = floats(Path(str(stem) + '.logits.f32'), 200058)
        floats(Path(str(stem) + '.norm.f32'), c['width'] * 2304)
        prompt = Path(str(stem) + '.prompt.i32').read_bytes()
        if len(prompt) != d['prompt_tokens'] * 4:
            raise ValueError('Invalid prompt token payload')
        if d['profile'] != 'sampled' and logits[c['selected_token']] != max(logits):
            raise ValueError('Greedy captured head does not agree with actual selected token')
        result[case, d['profile']] = dict(metadata=c, logits=logits,
                                         prompt_sha256=hashlib.sha256(prompt).hexdigest(),
                                         logits_sha256=hashlib.sha256(Path(str(stem) + '.logits.f32').read_bytes()).hexdigest())
    return result


def probabilities(logits, temperature):
    maximum = max(logits)
    unnormalized = [math.exp((v - maximum) / temperature) for v in logits]
    total = math.fsum(unnormalized)
    return [v / total for v in unnormalized]


def vector_metrics(a, b):
    delta = [abs(x-y) for x, y in zip(a, b)]
    p, q = probabilities(a, 1), probabilities(b, 1)
    return dict(max_abs_logit=max(delta), rms_logit=math.sqrt(math.fsum(v*v for v in delta)/len(delta)),
                softmax_temperature=1, total_variation=math.fsum(abs(x-y) for x, y in zip(p, q))/2,
                kl_left_right=math.fsum(x*math.log(x/y) for x, y in zip(p, q) if x and y),
                kl_right_left=math.fsum(y*math.log(y/x) for x, y in zip(p, q) if x and y))


def describe(capture, candidates):
    v = capture['logits']
    top = sorted(range(len(v)), key=lambda i: (-v[i], i))[:10]
    probability = probabilities(v, 1)
    return dict(capture=capture['metadata'], prompt_sha256=capture['prompt_sha256'],
                logits_sha256=capture['logits_sha256'],
                top10=[dict(token=i, logit=v[i], probability=probability[i]) for i in top],
                top1_top2_margin=v[top[0]]-v[top[1]],
                original_candidates=[dict(token=i, logit=v[i], probability=probability[i],
                                          rank=1+sum(x>v[i] for x in v)) for i in candidates])


def compare(left, right, left_benchmark, right_benchmark, report):
    differences = report['differences']
    a = read_capture(left, left_benchmark, differences)
    b = read_capture(right, right_benchmark, differences)
    result = []
    for d in differences:
        key = d['case'], d['profile']
        x, y = a[key], b[key]
        if x['prompt_sha256'] != y['prompt_sha256']:
            raise ValueError('Different prompt token IDs')
        candidates = {d['left_token'], d['right_token']}
        if any(t is None for t in candidates):
            raise ValueError('Termination-only difference needs a separate boundary probe')
        candidates = sorted(candidates)
        result.append(dict(case=key[0], profile=key[1], output_index=d['first_output_index_zero_based'],
                           shared_generated_prefix=d['shared_generated_prefix'],
                           left=describe(x,candidates), right=describe(y,candidates),
                           metrics=vector_metrics(x['logits'],y['logits'])))
    return dict(format_version=1, passed_capture_validation=True, comparisons=result,
                interpretation='Identical-prefix, original-path final-hidden-state LM-head replay. '
                               'No independent full-model reference or quality equivalence established. '
                               'Temperature-1 softmax metrics are diagnostic; sampled production also uses temperature/top-p.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('left',type=Path);p.add_argument('right',type=Path)
    p.add_argument('--left-benchmark',type=Path,required=True)
    p.add_argument('--right-benchmark',type=Path,required=True)
    p.add_argument('--divergences',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=compare(a.left,a.right,a.left_benchmark,a.right_benchmark,json.loads(a.divergences.read_text()))
    with a.out.open('x') as f: json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__': main()
