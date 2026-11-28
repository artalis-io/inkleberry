#!/usr/bin/env python3
"""Reject incomplete, changed-prefix, nonfinite and mis-mapped logit evidence."""
import array
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compare_boundary_logits import compare, floats, vector_metrics


class Boundary(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.report={'differences':[dict(case=4,profile='ar',first_output_index_zero_based=1,
                                        prompt_tokens=27,shared_generated_prefix=[11],
                                        left_token=12,right_token=14)]}
        for name,chosen in [('left',12),('right',14)]:
            root=self.root/name;root.mkdir()
            rows=[dict(case=c,profile=p,run=r,warmup=r==0,prompt_tokens=27,
                       generated=3,tokens=[11,chosen,13] if c==4 and p=='ar' else [11,12,13])
                  for c in range(7) for p in ('ar','fixed','adaptive','sampled') for r in range(6)]
            (self.root/(name+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            self.write_rows(name,[dict(kind='capture',case=4,profile=0,output_index=1,width=1,
                                       row=0,position=28,selected_token=chosen),
                                  dict(kind='result',case=4,profile=0,prompt_tokens=27,
                                       capture_preserved_output=True,tokens=[11,chosen,13])])
            v=array.array('f',[-10])*200058;v[chosen]=5
            (root/'case-4-profile-0.logits.f32').write_bytes(v.tobytes())
            (root/'case-4-profile-0.norm.f32').write_bytes(bytes(2304*4))
            (root/'case-4-profile-0.prompt.i32').write_bytes(array.array('i',[1]*27).tobytes())

    def write_rows(self,name,rows):
        (self.root/name/'capture.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')

    def compare(self):
        return compare(self.root/'left',self.root/'right',self.root/'left.jsonl',
                       self.root/'right.jsonl',self.report)

    def test_known_difference_and_probability_shift(self):
        result=self.compare();self.assertTrue(result['passed_capture_validation'])
        self.assertGreater(result['comparisons'][0]['metrics']['total_variation'],0)
        self.assertEqual(vector_metrics([1,2],[1,2])['total_variation'],0)
        self.assertLess(vector_metrics([1,2],[101,102])['total_variation'],1e-14)
        self.assertAlmostEqual(vector_metrics([0,1],[1,0])['total_variation'],
                               (math.exp(1)-1)/(math.exp(1)+1))

    def test_missing_duplicate_bad_row_and_changed_output(self):
        path=self.root/'right'/'capture.jsonl'
        original=[json.loads(x) for x in path.read_text().splitlines()]
        variants=[original[:-1],original+[original[0]]]
        for key,value in [('row',1),('width',0),('selected_token',12),('output_index',2)]:
            rows=json.loads(json.dumps(original));rows[0][key]=value;variants.append(rows)
        rows=json.loads(json.dumps(original));rows[1]['capture_preserved_output']=False;variants.append(rows)
        for rows in variants:
            self.write_rows('right',rows)
            with self.assertRaises(ValueError):self.compare()

    def test_changed_prompt_ids_rejected(self):
        (self.root/'right'/'case-4-profile-0.prompt.i32').write_bytes(array.array('i',[2]*27).tobytes())
        with self.assertRaises(ValueError):self.compare()

    def test_truncated_nonfinite_and_wrong_greedy_winner(self):
        p=self.root/'right'/'case-4-profile-0.logits.f32'
        for values in (array.array('f',[0]),array.array('f',[float('nan')])*200058,
                       array.array('f',[0])*200058):
            if len(values)==200058 and not math.isnan(values[0]):values[0]=100
            p.write_bytes(values.tobytes())
            with self.assertRaises(ValueError):self.compare()
        p.write_bytes(array.array('f',[float('inf')]).tobytes())
        with self.assertRaises(ValueError):floats(p,1)


if __name__=='__main__':unittest.main()
