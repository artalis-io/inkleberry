#!/usr/bin/env python3
"""Guard translation-unit isolation without modifying algorithm expressions."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import amalgamate

class NamespaceTests(unittest.TestCase):
    def test_functions_and_borrowed_names(self):
        text='''static int copy(int x) { return x; }
int use(void) { int (*cb)(int) = copy; return cb(2) + copy(3); }
int get(struct foreign *p) { return p->copy + p->result; }
const char *literal = "copy(result)"; /* copy must remain in comments */
'''
        value=amalgamate.namespace(text,'unit')
        self.assertIn('static int sq_am_unit_copy(',value)
        self.assertIn('= sq_am_unit_copy;',value)
        self.assertIn('sq_am_unit_copy(3)',value)
        self.assertIn('p->copy + p->result',value)
        self.assertIn('"copy(result)"',value)
        self.assertIn('/* copy must remain in comments */',value)
    def test_numeric_and_member_tokens(self):
        text='''enum { E=1, H=2 };
static int state = 0;
float get(struct foreign x) { return 1E-4f + 0xEF + E + x.E + H + state; }
'''
        value=amalgamate.namespace(text,'unit')
        self.assertIn('1E-4f + 0xEF',value)
        self.assertIn('x.E',value)
        self.assertIn('sq_am_unit_E + x.E + sq_am_unit_H + sq_am_unit_state',value)
    def test_outputs_are_hermetic(self):
        outputs=amalgamate.generated()
        for name,text in outputs.items():
            self.assertNotIn('#include "internal.h"',text)
            self.assertNotIn('#include "sq_chat.h"',text)
            self.assertNotIn(str(ROOT),text)
        self.assertIn('#include "sq.h"',outputs['dist/sq.c'])
        self.assertIn('#ifdef SQ_ENABLE_CUDA',outputs['dist/sq.c'])

if __name__=='__main__':unittest.main()
