#!/usr/bin/env python3
"""Acceptance-validator failure paths run in CPU CI without a model or Docker."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import semantic_acceptance as s

META={'semantic_version':1,'finish_reason':1,'generated_tokens':150,'max_output_tokens':2048}
def framed(answer):return '<|message_model|><|content_thinking|>Check the task.<|end_message|><|message_model|><|content_text|>'+answer+'<|end_message|><|content_model_end_sampling|>'

class Acceptance(unittest.TestCase):
    def test_exact_arithmetic_and_copy(self):
        for case,text in [('arithmetic','455'),('negative','-53'),('copy','aB7-041_xZ'),('retrieval','COBALT-7319')]:
            s.check_answer(case,s.final_answer(framed(text),META))
            with self.assertRaises(ValueError):s.check_answer(case,text+'x')
    def test_json_values_types_and_duplicates(self):
        s.check_answer('json','{"name":"Acme","quantity":17,"urgent":false}')
        for text in ['{"name":"Acme","quantity":"17","urgent":false}',
                     '{"name":"Acme","quantity":17,"urgent":0}',
                     '{"name":"Acme","quantity":17,"quantity":17,"urgent":false}',
                     '{"name":"Acme","quantity":17,"urgent":false,"extra":1}',
                     '```json\n{"name":"Acme","quantity":17,"urgent":false}\n```']:
            with self.assertRaises(ValueError):s.check_answer('json',text)
    def test_truncated_cancelled_or_missing_final(self):
        for finish in [0,2,3]:
            with self.assertRaises(ValueError):s.final_answer(framed('455'),dict(META,finish_reason=finish))
        for raw in [framed('455')[:-10],'<|content_thinking|>455',framed(''),framed('455')+'junk',framed('455')+framed('455')]:
            with self.assertRaises(ValueError):s.final_answer(raw,META)
    def test_reasoning_loop_and_bad_text(self):
        raw=framed('455').replace('Check the task.','one two three four five six seven eight '*8)
        with self.assertRaises(ValueError):s.final_answer(raw,META)
        with self.assertRaises(ValueError):s.final_answer(framed('\ufffd'),META)
    def test_prose_structure_without_literal_topic_requirement(self):
        text=('On a quiet autumn morning, pale sunlight settles across the garden while leaves drift onto the damp path. '
              'A small bird calls from the hedge, and a breeze carries the scent of earth toward the open window. '
              'The street remains still as the day begins.')
        s.check_answer('prose',text)
        s.check_answer('prose',text.replace('autumn','fall'))
        for bad in ['Too short.',text+'\nAnother paragraph.',text[:-1]]:
            with self.assertRaises(ValueError):s.check_answer('prose',bad)
    def test_code_fence_and_container_failure_cleanup(self):
        self.assertEqual(s.check_answer('code','```c\nint f(void){return 0;}\n```'),'int f(void){return 0;}')
        import subprocess
        with patch.object(s.subprocess,'run',side_effect=[subprocess.CompletedProcess([],1,'','syntax error'),subprocess.CompletedProcess([],0)]) as run:
            with self.assertRaises(ValueError):s.compile_and_test('broken','gcc:14.2.0')
            args=run.call_args_list[0].args[0]
            for flag in ['--network=none','--read-only','--cap-drop=ALL','--user=65534:65534','--pull=never']:
                self.assertIn(flag,args)
            self.assertEqual(run.call_args_list[1].args[0][:3],['docker','rm','-f'])
    def test_timeout_removes_container(self):
        import subprocess
        with patch.object(s.subprocess,'run',side_effect=[subprocess.TimeoutExpired('docker',30),subprocess.CompletedProcess([],0)]) as run:
            with self.assertRaises(subprocess.TimeoutExpired):s.compile_and_test('x','gcc:14.2.0')
            self.assertEqual(run.call_count,2)
    def test_complete_campaign_cannot_hide_wrong_answer(self):
        import argparse
        answers={'arithmetic':'999','negative':'-53','copy':'aB7-041_xZ','retrieval':'COBALT-7319',
                 'json':'{"name":"Acme","quantity":17,"urgent":false}',
                 'code':'int max_int_array(const int*a,size_t n){return a[n-1];}',
                 'prose':('On a quiet autumn morning, pale sunlight settles across the garden while leaves drift onto the damp path. '
                          'A small bird calls from the hedge, and a breeze carries the scent of earth toward the open window. '
                          'The street remains still as the day begins.')}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for case,prompt in s.CASES.items():
                rows.append({'case':case,'profile':'ar','returncode':0,'prompt_sha256':s.hashlib.sha256(prompt.encode()).hexdigest()})
                (root/(case+'-ar.raw.txt')).write_text('Idle GPU preflight: fixture\n'+framed(answers[case]))
                (root/(case+'-ar.stderr.txt')).write_text(json.dumps(dict(META,profile=0)))
            (root/'run.json').write_text(json.dumps({'suite':'semantic-v1','complete':True,'profiles':['ar'],'runs':rows}))
            args=argparse.Namespace(input=root,out=root/'report.json',code_image='gcc:14.2.0')
            with patch.object(s.subprocess,'check_output',return_value='image-id'),patch.object(s,'compile_and_test'),patch('builtins.print'):
                self.assertEqual(s.assess(args),1)
            report=json.loads(args.out.read_text());self.assertFalse(report['passed'])
            self.assertFalse(report['cases'][0]['passed']);self.assertEqual(len(report['cases']),7)

    def test_incomplete_or_duplicate_campaign_rejected(self):
        import argparse
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);receipt=root/'run.json'
            args=argparse.Namespace(input=root,out=root/'report.json',code_image='gcc:14.2.0')
            receipt.write_text(json.dumps({'suite':'semantic-v1','complete':False}))
            with self.assertRaises(ValueError):s.assess(args)
            rows=[{'case':c,'profile':'ar'} for c in s.CASES];rows[-1]=rows[0]
            receipt.write_text(json.dumps({'suite':'semantic-v1','complete':True,'profiles':['ar'],'runs':rows}))
            with self.assertRaises(ValueError):s.assess(args)

if __name__=='__main__':unittest.main()
