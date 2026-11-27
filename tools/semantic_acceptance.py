#!/usr/bin/env python3
"""Small task-level GPU acceptance suite; no token goldens or kernel changes."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from model_release import FILES, digest

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('ar', 'fixed', 'adaptive', 'sampled')
CASES = {
    'arithmetic': 'Compute (37 * 19) - 248. Return only the integer answer.',
    'negative': 'Compute -17 * 6 + 49. Return only the integer answer.',
    'json': 'Extract name, quantity and urgent from: customer Acme, quantity 17, urgent no. '
            'Return only one JSON object with exactly those keys; urgent must be a boolean.',
    'code': 'Write C11 function int max_int_array(const int *a, size_t n). For n >= 1 return '
            'the largest element. Do not modify a. Include any needed standard headers. '
            'Return only the complete function and headers; no main, no explanation.',
    'copy': 'Copy the following identifier exactly, preserving case and punctuation. '
            'Return only the identifier: aB7-041_xZ',
    'retrieval': ('Record cedar has value CEDAR-0000.\n' * 180 +
                  'Record NEEDLE has value COBALT-7319.\n' +
                  'Record maple has value MAPLE-9999.\n' * 180 +
                  'What is the value of record NEEDLE? Return only its exact value.'),
    'prose': 'Write one complete paragraph of 40 to 70 words describing a quiet autumn morning. '
             'Use ordinary English sentences. No title, bullets or explanation.'
}


def final_answer(raw, meta):
    if (meta.get('semantic_version') != 1 or meta.get('finish_reason') != 1 or
            type(meta.get('generated_tokens')) is not int or
            not 0 < meta['generated_tokens'] <= 2048 or meta.get('max_output_tokens') != 2048):
        raise ValueError('Incomplete generation, non-EOS termination or invalid receipt')
    marker = '<|content_text|>'
    if raw.count(marker) != 1:
        raise ValueError('Expected one explicit final-answer channel')
    prefix, tail = raw.split(marker)
    if '<|content_thinking|>' not in prefix and prefix != '<|message_model|>':
        raise ValueError('Unexpected response framing')
    answer, sep, suffix = tail.partition('<|end_message|>')
    if not sep or suffix.strip() != '<|content_model_end_sampling|>' or '<|' in answer:
        raise ValueError('Incomplete or unexpected final-answer framing')
    answer = answer.strip()
    if not answer or '\ufffd' in answer or any(ord(c) < 32 and c not in '\n\r\t' for c in answer):
        raise ValueError('Empty, malformed UTF-8 or control characters in answer')
    # Detect loops in the entire response, including the reasoning channel.
    words = re.findall(r'\w+', raw.lower())
    grams = collections.Counter(tuple(words[i:i+8]) for i in range(max(0, len(words)-7)))
    if grams and max(grams.values()) >= 6:
        raise ValueError('Repeated eight-word sequence (possible runaway generation)')
    return answer


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def check_answer(case, text):
    expected = {'arithmetic': '455', 'negative': '-53', 'copy': 'aB7-041_xZ',
                'retrieval': 'COBALT-7319'}
    if case in expected:
        if text != expected[case]:
            raise ValueError('Incorrect exact answer')
    elif case == 'json':
        value = json.loads(text, object_pairs_hook=unique_object)
        if (not isinstance(value, dict) or set(value) != {'name', 'quantity', 'urgent'} or
                value['name'] != 'Acme' or type(value['quantity']) is not int or
                value['quantity'] != 17 or value['urgent'] is not False):
            raise ValueError('Incorrect JSON schema/types/values')
    elif case == 'prose':
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        if (not 40 <= len(words) <= 70 or '\n' in text or text[-1] not in '.!?' or
                len(set(w.lower() for w in words)) / len(words) < .45):
            raise ValueError('Paragraph length, structure or lexical-diversity failure')
    elif case == 'code':
        match = re.fullmatch(r'```(?:c|C)?\s*\n(.*?)\n```', text, re.S)
        return match[1] if match else text
    else:
        raise ValueError('Unknown semantic case')
    return None


def compile_and_test(code, image):
    """Generated C executes only in a restricted, offline disposable container."""
    harness = r'''
#include <stddef.h>
#include <limits.h>
''' + code + r'''
int main(void) {
    const int a[] = {-9, -3, -7};
    const int b[] = {INT_MIN};
    const int c[] = {2, INT_MAX, 2, -1};
    const int d[] = {5, 5, 5};
    const int e[] = {2, -9, 11};
    if (max_int_array(a, 3) != -3 || max_int_array(b, 1) != INT_MIN ||
        max_int_array(c, 4) != INT_MAX || max_int_array(d, 3) != 5 ||
        max_int_array(e, 3) != 11 ||
        a[0] != -9 || a[1] != -3 || a[2] != -7) return 1;
    return 0;
}
'''
    with tempfile.TemporaryDirectory(prefix='ifn-semantic-code-') as tmp:
        Path(tmp).chmod(0o755)
        Path(tmp, 'program.c').write_text(harness)
        command = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--read-only',
                   '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=64',
                   '--memory=256m', '--cpus=1', '--user=65534:65534',
                   '--tmpfs=/tmp:rw,exec,nosuid,size=64m,mode=1777',
                   '--mount', f'type=bind,src={tmp},dst=/input,readonly', image,
                   'sh', '-c', 'cc -std=c11 -O2 -Wall -Wextra -Werror /input/program.c '
                   '-o /tmp/check && /tmp/check']
        name = 'ifn-semantic-' + os.urandom(8).hex()
        command[3:3] = ['--name', name]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise ValueError('Generated C compile/test failed: ' + result.stderr[-2000:])
        finally:
            # Killing the CLI on timeout alone would leave the container running.
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=15)


def run(a):
    if not re.fullmatch('[0-9a-f]{40}', a.source_revision):
        raise ValueError('Exact source revision required')
    if len(set(a.profiles)) != len(a.profiles):
        raise ValueError('Duplicate execution profile')
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = {'suite': 'semantic-v1', 'source_revision': a.source_revision,
                'binary_sha256': digest(a.binary), 'model': {n: digest(a.model/n) for n in FILES},
                'profiles': a.profiles, 'started_at': time.time(), 'complete': False, 'runs': []}
    report = a.out/'run.json'
    report.write_text(json.dumps(manifest, indent=2))
    env = dict(os.environ, INKLEBERRY_TEST_GPU_UUID=a.gpu, INKLEBERRY_TEST_TIMEOUT_SECONDS='600')
    for profile in a.profiles:
        for case, prompt in CASES.items():
            name = case + '-' + profile
            prompt_file = a.out/(name+'.prompt.txt');prompt_file.write_text(prompt, encoding='utf-8')
            command = [sys.executable, str(ROOT/'tools/run_cuda_smoke.py'), a.cuda_runtime,
                       str(a.binary.resolve()), str(a.model.resolve()), str(prompt_file.resolve()),
                       str(PROFILES.index(profile))]
            with (a.out/(name+'.raw.txt')).open('wb') as out, (a.out/(name+'.stderr.txt')).open('wb') as err:
                result = subprocess.run(command, stdout=out, stderr=err, env=env, timeout=650)
            manifest['runs'].append({'case': case, 'profile': profile, 'returncode': result.returncode,
                                      'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
            report.write_text(json.dumps(manifest, indent=2))
            if result.returncode:
                raise ValueError('GPU run failed; see ' + name + '.stderr.txt')
            print('Completed', name, flush=True)
    manifest['complete'] = True
    report.write_text(json.dumps(manifest, indent=2))


def assess(a):
    manifest = json.loads((a.input/'run.json').read_text())
    if manifest.get('suite') != 'semantic-v1' or manifest.get('complete') is not True:
        raise ValueError('Incomplete or unknown semantic campaign')
    profiles = manifest['profiles']
    if not profiles or len(set(profiles)) != len(profiles) or not set(profiles) <= set(PROFILES):
        raise ValueError('Invalid profile coverage')
    expected = {(c, p) for c in CASES for p in profiles}
    rows = manifest['runs']
    if len(rows) != len(expected) or {(v['case'],v['profile']) for v in rows} != expected:
        raise ValueError('Missing or duplicated case/profile')
    image = subprocess.check_output(['docker','image','inspect',a.code_image,'--format={{.Id}}'], text=True, timeout=30).strip()
    report = {'suite': 'semantic-v1', 'passed': False, 'code_image_id': image,
              'validator_sha256': digest(Path(__file__)),
              'manual_review_required': ['prose'],
              'coherence_claim': 'Structural/topic smoke checks only; prose needs human review', 'cases': []}
    for row in rows:
        case, profile = row['case'], row['profile'];name = case+'-'+profile
        receipt = {'case': case, 'profile': profile, 'passed': False}
        try:
            if row['returncode'] != 0 or row['prompt_sha256'] != hashlib.sha256(CASES[case].encode()).hexdigest():
                raise ValueError('Run failed or input identity mismatch')
            raw = (a.input/(name+'.raw.txt')).read_text(encoding='utf-8')
            # The idle-GPU wrapper emits two informational lines before the child.
            begin = raw.find('<|message_model|>')
            if begin < 0:raise ValueError('Missing model response')
            metadata = [json.loads(s) for s in (a.input/(name+'.stderr.txt')).read_text().splitlines() if s.startswith('{')]
            if len(metadata) != 1 or metadata[0].get('profile') != PROFILES.index(profile):
                raise ValueError('Missing/ambiguous generation metadata')
            answer = final_answer(raw[begin:], metadata[0]);receipt['answer'] = answer
            code = check_answer(case, answer)
            if code is not None:compile_and_test(code, a.code_image)
            receipt['passed'] = True
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            receipt['error'] = str(error)
        report['cases'].append(receipt)
    report['passed'] = all(r['passed'] for r in report['cases'])
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': report['passed'], 'cases': len(rows),
                      'failures': [r for r in report['cases'] if not r['passed']]}, indent=2))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__);sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run');p.add_argument('--binary',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--cuda-runtime',required=True);p.add_argument('--source-revision',required=True)
    p.add_argument('--profiles',nargs='+',choices=PROFILES,default=['ar','adaptive']);p.add_argument('--out',type=Path,required=True)
    p = sub.add_parser('assess');p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--code-image',default='gcc:14.2.0')
    a = parser.parse_args()
    try:return run(a) if a.command == 'run' else assess(a)
    except (ValueError,OSError,subprocess.SubprocessError) as error:parser.exit(1,str(error)+'\n')

if __name__ == '__main__':sys.exit(main())
