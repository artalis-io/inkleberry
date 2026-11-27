#!/usr/bin/env python3
"""Independent failure-boundary tests for the offline diagnostic comparator."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('compare_numerics',ROOT/'tools/compare_numerics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class NumericComparison(unittest.TestCase):
    def test_first_divergence_and_decision(self):
        a={(0,0,'embedding'):struct.pack('<2f',1,2),(0,0,'qrgs'):struct.pack('<2f',3,4),
           (0,48,'logits'):struct.pack('<2f',5,6)}
        b=dict(a);b[0,0,'qrgs']=struct.pack('<2f',3,4.5);b[0,48,'logits']=struct.pack('<2f',7,6)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'spans.jsonl').write_text(json.dumps({'label':'test','sha256':'a'*64})+'\n')
            with patch.object(m,'load',side_effect=[a,b]):report=m.compare(root,root)
            self.assertEqual(report['first_difference']['stage'],'qrgs')
            self.assertEqual(report['first_difference']['max_abs'],.5)
            self.assertFalse(report['differences'][-1]['top1_equal'])
            self.assertFalse(report['numerical_acceptance_established'])
    def test_nonfinite_even_when_equal(self):
        a={(0,0,'embedding'):struct.pack('<f',float('nan'))}
        with patch.object(m,'load',side_effect=[a,a]):
            with self.assertRaisesRegex(ValueError,'Non-finite'):m.compare(Path('.'),Path('.'))
    def test_truncation_and_missing_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row={'batch':1,'position':0,'visit':0,'stage':'embedding','width':2304}
            (root/'trace.jsonl').write_text(json.dumps(row)+'\n')
            tensor=root/'b1-p0-v0-embedding.bin';tensor.write_bytes(b'')
            with self.assertRaisesRegex(ValueError,'Incomplete tensor'):m.load(root,1)
            tensor.write_bytes(bytes(2304*4))
            with self.assertRaisesRegex(ValueError,'coverage'):m.load(root,1)
            (root/'trace.jsonl').write_text((json.dumps(row)+'\n')*2)
            with self.assertRaisesRegex(ValueError,'Duplicate'):m.load(root,1)
    def test_changed_input_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            left,right=Path(tmp)/'left',Path(tmp)/'right';left.mkdir();right.mkdir()
            (left/'run.json').write_text(json.dumps({'tokens':[1]}))
            (right/'run.json').write_text(json.dumps({'tokens':[2]}))
            with self.assertRaisesRegex(ValueError,'identity mismatch'):m.compare(left,right)

    def test_unknown_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'trace.jsonl').write_text(json.dumps({'batch':1,'position':0,'visit':0,'stage':'../unsafe','width':1}))
            with self.assertRaisesRegex(ValueError,'Unknown stage'):m.load(root,1)

if __name__=='__main__':unittest.main()
