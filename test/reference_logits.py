#!/usr/bin/env python3
"""Reference input integrity and native-prefill chunk boundaries; no Torch required."""
import array
import copy
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from collect_reference_logits import contexts, prompt_chunks


class Reference(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.differences=[]
        for case in (4,5,6):
            data=array.array('i',[10,11,12])
            if sys.byteorder!='little':data.byteswap()
            (self.root/f'case-{case}-profile-0.prompt.i32').write_bytes(data.tobytes())
            self.differences.append(dict(case=case,profile='ar',prompt_tokens=3,
                                        first_output_index_zero_based=2,shared_generated_prefix=[13,14]))

    def test_exact_ids_no_chat_template(self):
        result=contexts(self.root,self.differences)
        self.assertEqual([x['case'] for x in result],[4,5,6])
        for x in result:
            self.assertEqual(x['prompt'],[10,11,12]);self.assertEqual(x['prefix'],[13,14])

    def test_truncated_prompt_and_invalid_ids(self):
        p=self.root/'case-4-profile-0.prompt.i32'
        for data in (b'\0',array.array('i',[-1,11,12]).tobytes()):
            p.write_bytes(data)
            with self.assertRaises(ValueError):contexts(self.root,self.differences)

    def test_missing_case_changed_or_invalid_prefix(self):
        for rows in (self.differences[:-1],self.differences+[self.differences[0]]):
            with self.assertRaises(ValueError):contexts(self.root,rows)
        for prefix in ([13],[13,True],[13,200058]):
            rows=copy.deepcopy(self.differences);rows[0]['shared_generated_prefix']=prefix
            with self.assertRaises(ValueError):contexts(self.root,rows)

    def test_original_prefill_shapes(self):
        self.assertEqual([len(x) for x in prompt_chunks(list(range(47)))],[40,7])
        self.assertEqual([len(x) for x in prompt_chunks(list(range(36)))],[32,4])
        self.assertEqual([len(x) for x in prompt_chunks(list(range(8844)))],
                         [2048,2048,2048,2048,648,4])
        for n in (1,8,27,32,33,36,47,64,2048,2049,8844):
            chunks=list(prompt_chunks(list(range(n))))
            self.assertEqual([t for c in chunks for t in c],list(range(n)))
            self.assertTrue(all(0<len(c)<=2048 for c in chunks))


if __name__=='__main__':unittest.main()
