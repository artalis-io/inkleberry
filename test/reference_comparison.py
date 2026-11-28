#!/usr/bin/env python3
"""Reference-comparison evidence rejection; synthetic logits, no inference claims."""
import array
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from collect_reference_logits import contexts
from compare_reference_logits import compare


class Comparison(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.ref=self.root/'ref';self.ctrl=self.root/'control';self.base=self.root/'base'
        self.ref.mkdir();self.ctrl.mkdir();self.base.mkdir()
        self.report={'differences':[]};self.rows=[];self.controls=[]
        model={'weights':'0'*64}
        for name in ('rtx','gb10'):
            (self.base/name).mkdir()
            (self.base/name/'metadata.json').write_text(json.dumps(dict(model_sha256=model)))
        for c in (4,5,6):
            self.report['differences'].append(dict(case=c,profile='ar',prompt_tokens=3,
                                                  first_output_index_zero_based=2,shared_generated_prefix=[13,14]))
            (self.base/'rtx'/f'case-{c}-profile-0.prompt.i32').write_bytes(struct.pack('<iii',10,11,12))
            values=array.array('f',[-10])*200058;values[3]=5
            payload=values.tobytes();sha=hashlib.sha256(payload).hexdigest()
            for run in (0,1):
                (self.ref/f'case-{c}-run-{run}.logits.f32').write_bytes(payload)
                self.rows.append(dict(case=c,run=run,sha256=sha,top1=3))
                self.controls.append(dict(case=c,run=run,top1=3,prompt_tokens=3,total_tokens=5,
                                          input_sha256=hashlib.sha256(struct.pack('<iiiii',10,11,12,13,14)).hexdigest()))
            (self.ctrl/f'case-{c}.logits.f32').write_bytes(payload)
            for name in ('rtx','gb10'):
                (self.base/name/f'case-{c}-profile-0.logits.f32').write_bytes(payload)
        self.manifest=dict(contexts=contexts(self.base/'rtx',self.report['differences']),
                           model_sha256=model,reference_revision='0'*40)
        (self.ref/'manifest.json').write_text(json.dumps(self.manifest))
        self.save()

    def save(self):
        (self.ref/'summary.json').write_text(json.dumps(dict(passed=True,runs=self.rows)))
        (self.ctrl/'control.jsonl').write_text('\n'.join(json.dumps(x) for x in self.controls)+'\n')

    def compare(self):return compare(self.ref,self.ctrl,self.base,self.report)

    def test_valid_identical_vectors(self):
        result=self.compare();self.assertTrue(result['comparison_validation_passed'])
        self.assertEqual(len(result['cases']),3)
        self.assertTrue(all(x['metrics']['total_variation']==0 for c in result['cases'] for x in c['configurations'].values()))

    def test_incomplete_duplicate_repetition(self):
        original=copy.deepcopy(self.rows)
        for rows in (original[:-1],original+[original[0]]):
            self.rows=rows;self.save()
            with self.assertRaises(ValueError):self.compare()

    def test_changed_model_and_context(self):
        p=self.ref/'manifest.json'
        for field,value in [('model_sha256',{}),('contexts',[])]:
            m=copy.deepcopy(self.manifest);m[field]=value;p.write_text(json.dumps(m))
            with self.assertRaises(ValueError):self.compare()

    def test_changed_control_input(self):
        self.controls[0]['input_sha256']='0'*64;self.save()
        with self.assertRaises(ValueError):self.compare()

    def test_changed_payload_even_with_updated_digest(self):
        p=self.ref/'case-4-run-1.logits.f32';v=array.array('f');v.frombytes(p.read_bytes());v[10]=-9
        p.write_bytes(v.tobytes());self.rows[1]['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();self.save()
        with self.assertRaisesRegex(ValueError,'Nonrepeatable'):self.compare()


if __name__=='__main__':unittest.main()
