#!/usr/bin/env python3
"""Mutation checks for the adapted lexical audit, without CUDA or model assets."""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import check_audit_boundaries as a

class Boundaries(unittest.TestCase):
    def fixture(self, root):
        (root/'src').mkdir();(root/'include').mkdir()
        (root/'include/ifn.h').write_text('#include <stdint.h>\n')
        text='\n'.join('int '+name+'(void){return 0;}' for name in sorted(a.CONTROL_ROOTS))
        (root/'src/control.c').write_text(text)
    def test_clean_and_transitive_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            self.assertEqual(a.audit(root)[0],[])
            p=root/'src/control.c'
            p.write_text(p.read_text().replace('ifn_session_cancel(void){return 0;}','ifn_session_cancel(void){return ifn_a();}')+
                         '\nint ifn_a(void){return ifn_b();}\nint ifn_b(void){ifn_a();return malloc(4)!=0;}')
            self.assertTrue(any('ifn_session_cancel -> ifn_a -> ifn_b -> malloc' in s for s in a.audit(root)[0]))
    def test_indirect_allocator_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);p=root/'src/control.c'
            p.write_text(p.read_text().replace('ifn_session_cancel(void){return 0;}','ifn_session_cancel(void){return ctx->allocate(3)!=0;}'))
            self.assertTrue(any('allocate' in s for s in a.audit(root)[0]))
    def test_private_cuda_bridge_is_not_a_control_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);p=root/'src/control.c'
            for name in ['ifn_cuda_malloc','ifn_cuda_event_sync','ifn_launch_custom','cublasCreate']:
                base='\n'.join('int '+n+'(void){return 0;}' for n in sorted(a.CONTROL_ROOTS))
                p.write_text(base.replace('ifn_session_cancel(void){return 0;}','ifn_session_cancel(void){return '+name+'();}'))
                self.assertTrue(any(name in s for s in a.audit(root)[0]))
    def test_strings_comments_and_braces(self):
        s='/* malloc(4); { */ int ifn_a(void){ const char*s="\\\" } abort(0)"; // free(x)\n return 0;}'
        text=a.strip(s)
        self.assertEqual(len(text),len(s));self.assertEqual(text.count('\n'),s.count('\n'))
        self.assertEqual(set(a.functions(text)),{'ifn_a'})
        self.assertNotIn('abort',a.calls(text))
    def test_missing_roots_or_empty_tree_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):a.audit(root)
            self.fixture(root);(root/'src/control.c').write_text('int ifn_other(void){return 0;}')
            self.assertTrue(any('Missing audited control root' in s for s in a.audit(root)[0]))
    def test_platform_calls_must_stay_in_platform_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            (root/'src/wrong.c').write_text('int ifn_x(void){return open("x",0);}')
            self.assertTrue(any('OS primitive' in s for s in a.audit(root)[0]))
            (root/'src/wrong.c').rename(root/'src/host_posix.c')
            self.assertEqual(a.audit(root)[0],[])
    def test_public_sdk_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);p=root/'include/ifn.h'
            for sdk in ['windows.h','cuda_runtime.h','pthread.h','sys/mman.h']:
                p.write_text('#include <'+sdk+'>\n')
                self.assertTrue(any('SDK leaked' in s for s in a.audit(root)[0]))
            p.write_text('/* #include <cuda_runtime.h> */\n#include <stddef.h>\n')
            self.assertEqual(a.audit(root)[0],[])
    def test_platform_alternatives_are_both_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            for name,body in [('host_posix.c','return 0;'),('host_win.c','return malloc(4)!=0;')]:
                (root/'src'/name).write_text('int ifn_helper(void){'+body+'}')
            p=root/'src/control.c';p.write_text(p.read_text().replace('ifn_session_cancel(void){return 0;}','ifn_session_cancel(void){return ifn_helper();}'))
            self.assertTrue(any('ifn_helper -> malloc' in s for s in a.audit(root)[0]))
    def test_unbalanced_input_rejected(self):
        with self.assertRaises(ValueError):a.functions('int ifn_bad(void){')
        with self.assertRaises(ValueError):a.strip('/* unclosed')

if __name__=='__main__':unittest.main()
