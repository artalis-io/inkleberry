#!/usr/bin/env python3
"""Independently validate complete persistent-worker JSONL; summarize comparable probes."""
import json
import statistics
import sys
from collections import defaultdict


def schedule(seed, cycles):
    def random():
        nonlocal seed
        seed ^= (seed << 13) & 0xffffffff
        seed ^= seed >> 17
        seed ^= (seed << 5) & 0xffffffff
        return seed
    for cycle in range(cycles):
        suffix, mode = random() & 1, random() % 3
        for step in range(5):
            yield cycle, step, mode, suffix ^ (step == 1)


def validate(records):
    if not records or records[0].get('kind') != 'start' or records[-1].get('kind') != 'end':
        raise ValueError('Missing start/end: interrupted or incomplete campaign')
    start, end = records[0], records[-1]
    cycles = end['completed_cycles']
    if end['passed'] is not True or not 0 < cycles <= start['cycles']:
        raise ValueError('Failed or empty campaign')
    baselines = records[1:3]
    if [b.get('suffix') for b in baselines] != [0, 1] or any(
            b.get('kind') != 'baseline' or not b.get('output') for b in baselines):
        raise ValueError('Missing clean-session baselines')
    history = records[3:-1]
    if len(history) != cycles * 10:
        raise ValueError('Missing or extra ordered requests')
    groups = defaultdict(list)
    modes = set()
    for index, (cycle, step, mode, suffix) in enumerate(schedule(start['seed'], cycles)):
        begin, result = history[2 * index:2 * index + 2]
        if begin != dict(kind='begin', cycle=cycle, step=step, mode=mode, suffix=suffix):
            raise ValueError('Seed/history mismatch')
        if (result.get('kind'), result.get('cycle'), result.get('step'), result.get('mode')) != (
                'result', cycle, step, mode):
            raise ValueError('Unpaired result')
        if result.get('valid') is not True or result.get('disposition') != 'next-request':
            raise ValueError('Invalid result/disposition')
        if result['tokens'] != len(result['output']):
            raise ValueError('Token accounting mismatch')
        if step == 2:
            if (result['tokens'] > 0) != (mode == 0):
                raise ValueError('Cancellation occurred in the wrong output phase')
            modes.add(mode)
            if result['status'] != 11:  # IFN_CANCELLED, frozen public status
                raise ValueError('Injected cancellation did not cancel')
        else:
            if result['status'] != 0 or result['output'] != baselines[suffix]['output']:
                raise ValueError('Ordinary request error or token corruption')
            if (result['reuse'] > 0) != (step in (1, 4)):
                raise ValueError('Forbidden cache reuse or missing expected hit')
            if result['tokens'] != len(result['output']):
                raise ValueError('Token accounting mismatch')
            # Step 4 is the repeated warm hit probe; compare same suffix/scope only.
            if step == 4 and cycle >= 2:
                groups[f'scope-{cycle % 2}/suffix-{suffix}'].append(result)
    summaries = {}
    for key, values in groups.items():
        metrics = {}
        for field in ('rss_bytes', 'ttft_ns', 'run_ns', 'decode_ns'):
            samples = [v[field] for v in values]
            metrics[field] = dict(min=min(samples), median=statistics.median(samples), max=max(samples),
                                  stdev=statistics.pstdev(samples))
        rates = [(v['tokens'] - 1) * 1e9 / v['decode_ns'] for v in values
                 if v['tokens'] > 1 and v['decode_ns'] > 0]
        metrics['decode_tokens_per_second'] = (dict(min=min(rates), median=statistics.median(rates),
                                                   max=max(rates), stdev=statistics.pstdev(rates))
                                               if rates else None)
        summaries[key] = dict(samples=len(values), metrics=metrics)
    return dict(passed=True, seed=start['seed'], cycles=cycles, requests=cycles * 5,
                cancellation_modes=sorted(modes), all_modes_exercised=modes == {0, 1, 2},
                comparable_warm_probes=summaries)


if __name__ == '__main__':
    try:
        with open(sys.argv[1], encoding='utf-8') as stream:
            result = validate([json.loads(line) for line in stream])
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        raise SystemExit('Worker qualification failed: ' + str(error)) from error
