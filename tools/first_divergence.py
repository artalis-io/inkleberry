#!/usr/bin/env python3
"""Locate repeatable first token differences; does not infer logit or quality agreement."""
import argparse
import hashlib
import json
from pathlib import Path

PROFILES = ('ar', 'fixed', 'adaptive', 'sampled')


def load(path):
    groups = {}
    for line in path.read_text().splitlines():
        if line.startswith('Idle GPU preflight:'):
            continue
        row = json.loads(line)
        case, profile, run = row['case'], row['profile'], row['run']
        tokens, count = row['tokens'], row['prompt_tokens']
        if (type(case) is not int or case not in range(7) or profile not in PROFILES or
                type(run) is not int or run not in range(6) or
                type(row['warmup']) is not bool or row['warmup'] != (run == 0) or
                type(count) is not int or not 0 < count <= 32768 or
                not isinstance(tokens, list) or not tokens or len(tokens) > 96 or
                any(type(t) is not int or not 0 <= t < 200058 for t in tokens) or
                type(row['generated']) is not int or row['generated'] != len(tokens)):
            raise ValueError('Invalid benchmark record')
        group = groups.setdefault((case, profile), {})
        if run in group:
            raise ValueError('Duplicate benchmark repetition')
        group[run] = row
    expected = {(case, profile) for case in range(7) for profile in PROFILES}
    if set(groups) != expected:
        raise ValueError('Incomplete benchmark case/profile matrix')
    result = {}
    for key, runs in groups.items():
        if set(runs) != set(range(6)):
            raise ValueError('Missing warm-up or measured repetition')
        first = runs[1]
        if any(r['tokens'] != first['tokens'] or r['prompt_tokens'] != first['prompt_tokens']
               for r in runs.values()):
            raise ValueError('Nondeterministic repetitions or changed prompt length')
        result[key] = first
    return result


def compare(left, right):
    a, b = load(left), load(right)
    differences = []
    for case, profile in sorted(a):
        x, y = a[case, profile], b[case, profile]
        if x['prompt_tokens'] != y['prompt_tokens']:
            raise ValueError('Prompt token counts differ; identical inputs are not established')
        if x['tokens'] == y['tokens']:
            continue
        at = next((i for i, (v, w) in enumerate(zip(x['tokens'], y['tokens'])) if v != w),
                  min(len(x['tokens']), len(y['tokens'])))
        differences.append(dict(case=case, profile=profile, first_output_index_zero_based=at,
                                prompt_tokens=x['prompt_tokens'],
                                shared_generated_prefix=x['tokens'][:at],
                                left_token=x['tokens'][at] if at < len(x['tokens']) else None,
                                right_token=y['tokens'][at] if at < len(y['tokens']) else None,
                                left_output_tokens=len(x['tokens']), right_output_tokens=len(y['tokens']),
                                interpretation='sampled outcome' if profile == 'sampled' else
                                'greedy/speculative outcome; requires execution-path-matched logits'))
    return dict(format_version=1, groups=28, matching_groups=28-len(differences),
                source_artifacts={label:dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                  for label, path in [('left', left), ('right', right)]},
                boundary='Historical token comparison only; prompt counts do not prove identical prompt bytes. '
                         'No logits, near-tie classification or reference quality claims.',
                differences=differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    with args.out.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')


if __name__ == '__main__':
    main()
