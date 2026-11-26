#!/usr/bin/env python3
"""Regression checks for the reversible native namespace and unchanged literals."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from native_namespace import names, original_bytes, translate, NATIVE_NOTICE, kernel_names


class NativeNamespaceTests(unittest.TestCase):
    def test_decorated_kernel_identifiers(self):
        text = '__global__ void __launch_bounds__(128, 2)\nmissed_kernel(int x);\n__global__ void ifn_ok(int x);'
        self.assertEqual(kernel_names(text), ['missed_kernel', 'ifn_ok'])
        self.assertEqual(kernel_names('__global__ void __launch_bounds__((32 * 4), (2))\nother_kernel(int x);'), ['other_kernel'])
        with self.assertRaises(ValueError):
            kernel_names('__global__ void __launch_bounds__((128) broken')

    def test_mapping(self):
        mapping = names()
        self.assertEqual(len(mapping), len(set(mapping.values())))
        for old, new in mapping.items():
            self.assertTrue(old.isidentifier(), old)
            self.assertTrue(new.startswith(('ifn_', 'IFN_', 'INKLEBERRY_')), new)

    def test_invalid_mappings_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config').mkdir()
            path = root / 'config/native-namespace.json'
            for mapping in [{'2304': 'ifn_value'}, {'old': 'new'},
                            {'a': 'ifn_same', 'b': 'ifn_same'},
                            {'__launch_bounds__': 'ifn_cuda_bounds'}]:
                path.write_text(json.dumps(mapping))
                with self.assertRaises(ValueError):
                    names(root)

    def test_literals_members_and_arithmetic(self):
        mapping = {'squid_launch': 'ifn_launch', 'BF16': 'IFN_CUDA_BF16',
                   'SQUID_MTP_BF16': 'INKLEBERRY_MTP_BF16'}
        text = 'squid_launch(x.BF16, x->BF16, BF16, 0x1.fp-2); "BF16"; "mma.sync BF16"; // BF16\n"SQUID_MTP_BF16";'
        expected = 'ifn_launch(x.BF16, x->BF16, IFN_CUDA_BF16, 0x1.fp-2); "BF16"; "mma.sync BF16"; // BF16\n"INKLEBERRY_MTP_BF16";'
        self.assertEqual(translate(text, mapping), expected)
        self.assertEqual(translate(expected, {v:k for k,v in mapping.items()}), text)

    def test_original_hashes(self):
        pins = json.loads((ROOT / 'UPSTREAM.json').read_text())
        count = 0
        for name, digest in pins['files'].items():
            if name.startswith('cuda/'):
                text = (ROOT / name).read_text()
                self.assertTrue(text.startswith(NATIVE_NOTICE), name)
                self.assertEqual(hashlib.sha256(original_bytes(text)).hexdigest(), digest, name)
                count += 1
        self.assertEqual(count, 9)

    def test_modified_math_is_rejected(self):
        text = (ROOT / 'cuda/config.hpp').read_text()
        self.assertEqual(text.count('IFN_CUDA_HIDDEN = 2304'), 1)
        before = hashlib.sha256(original_bytes(text)).hexdigest()
        after = hashlib.sha256(original_bytes(text.replace('IFN_CUDA_HIDDEN = 2304',
                                                          'IFN_CUDA_HIDDEN = 2305'))).hexdigest()
        self.assertNotEqual(before, after)
        with self.assertRaises(ValueError):
            original_bytes(text.removeprefix(NATIVE_NOTICE))


if __name__ == '__main__':
    unittest.main()
