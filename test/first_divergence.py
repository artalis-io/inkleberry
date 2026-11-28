#!/usr/bin/env python3
"""Adversarial cases for historical first-divergence extraction."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from first_divergence import compare


class Divergence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.left = Path(self.temp.name) / 'left'
        self.right = Path(self.temp.name) / 'right'
        self.rows = [dict(case=c, profile=p, run=r, warmup=r == 0, prompt_tokens=27,
                          generated=3, tokens=[11, 12, 13])
                     for c in range(7) for p in ('ar', 'fixed', 'adaptive', 'sampled') for r in range(6)]
        self.write(self.left, self.rows)

    def write(self, path, rows):
        path.write_text('\n'.join(json.dumps(x) for x in rows) + '\n')

    def test_agreement(self):
        self.write(self.right, self.rows)
        self.assertEqual(compare(self.left, self.right)['matching_groups'], 28)

    def test_shared_prefix_excludes_first_divergent_token(self):
        other = copy.deepcopy(self.rows)
        for r in other:
            if r['case'] == 4 and r['profile'] == 'ar':
                r['tokens'][1] = 99
        self.write(self.right, other)
        d = compare(self.left, self.right)['differences'][0]
        self.assertEqual((d['first_output_index_zero_based'], d['shared_generated_prefix'],
                          d['left_token'], d['right_token']), (1, [11], 12, 99))

    def test_early_termination_is_not_index_error(self):
        other = copy.deepcopy(self.rows)
        for r in other:
            if r['case'] == 4 and r['profile'] == 'sampled':
                r['tokens'] = [11, 12]
                r['generated'] = 2
        self.write(self.right, other)
        d = compare(self.left, self.right)['differences'][0]
        self.assertIsNone(d['right_token'])
        self.assertEqual(d['interpretation'], 'sampled outcome')

    def test_reject_incomplete_duplicate_changed_or_invalid(self):
        variants = [self.rows[:-1], self.rows + [self.rows[-1]]]
        for key, value in [('tokens', [10, 12, 13]), ('prompt_tokens', 28),
                           ('tokens', [True, 12, 13]), ('generated', 2), ('warmup', True)]:
            other = copy.deepcopy(self.rows)
            other[-1][key] = value
            variants.append(other)
        for other in variants:
            with self.subTest(other=other[-1]):
                self.write(self.right, other)
                with self.assertRaises(ValueError):
                    compare(self.left, self.right)


if __name__ == '__main__':
    unittest.main()
