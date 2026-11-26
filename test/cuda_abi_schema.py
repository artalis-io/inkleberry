#!/usr/bin/env python3
"""Host-only contract/failure tests; no Zig or CUDA toolkit required."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import generate_cuda_abi as gen


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = gen.load_schema(ROOT / 'abi/cuda.json')

    def test_initial_contract(self):
        outputs = gen.generate(self.schema)
        baseline = json.loads((ROOT / 'test/fixtures/cuda_abi_v1.json').read_text())
        self.assertEqual(self.schema['provenance']['revision'], baseline['upstream_revision'])
        self.assertEqual(len(self.schema['records']), 7)
        self.assertEqual(len(self.schema['functions']), 76)
        self.assertEqual(sum(2 + len(r['fields']) for r in self.schema['records']), 72)
        for path, digest in baseline['payload_sha256'].items():
            payload = outputs[path].split('*/\n', 1)[1].encode()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest, path)

    def test_nested_const_and_arrays(self):
        names = {r['name'] for r in self.schema['records']}
        self.assertEqual(gen.ctype('pointer<pointer<void>>', names), 'void * *')
        self.assertEqual(gen.ctype('pointer<const,pointer<const,float32>>', names), 'const float * const *')
        self.assertEqual(gen.ctype('pointer<const,pointer<float32>>', names), 'float * const *')
        self.assertIn('CDsparkLayerWeights layers[3];', gen.generate(self.schema)['src/cuda_abi.h'])

    def test_rejected_schema_mutations(self):
        def replace(path, value):
            def change(schema):
                node = schema
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
            return change
        mutations = [
            replace(['schema_version'], 2), replace(['schema_version'], True),
            replace(['records'], []), replace(['functions'], []),
            replace(['records', 0, 'fields'], []),
            replace(['records', 0, 'name'], 'int'),
            replace(['records', 0, 'name'], 'bad;system()'),
            replace(['records', 0, 'cpp_type'], 'squid::X;\n#error injected'),
            replace(['records', 0, 'fields', 0, 'type'], 'double'),
            replace(['records', 0, 'fields', 0, 'type'], 'void'),
            replace(['records', 0, 'fields', 0, 'type'], 'CDsparkWeights'),
            replace(['records', 0, 'fields', 0, 'type'], 'pointer<CSampleConfig>'),
            replace(['records', 0, 'fields', 0, 'type'], 'pointer<const>'),
            replace(['records', 0, 'fields', 0, 'type'], 'pointer<float32>>'),
            replace(['records', 0, 'fields', 0, 'type'], 'pointer<' * 10 + 'float32' + '>' * 10),
            replace(['records', 0, 'fields', 0, 'count'], 0),
            replace(['records', 0, 'fields', 0, 'count'], -1),
            replace(['records', 0, 'fields', 0, 'count'], True),
            replace(['records', 0, 'fields', 0, 'count'], 1.5),
            replace(['records', 0, 'fields', 0, 'count'], 65537),
            replace(['functions', 0, 'returns'], 'unknown'),
            replace(['functions', 0, 'parameters', 0, 'type'], 'void'),
            replace(['functions', 0, 'parameters', 0, 'count'], 3),
            replace(['provenance', 'revision'], 'not-a-revision'),
            replace(['provenance', 'extra'], 'unexpected'),
            replace(['records', 0, 'extra'], 'unexpected'),
            replace(['extra'], 'unexpected'),
            lambda s: s['records'].append(copy.deepcopy(s['records'][0])),
            lambda s: s['records'][0]['fields'].append(copy.deepcopy(s['records'][0]['fields'][0])),
            lambda s: s['functions'].append(copy.deepcopy(s['functions'][0])),
            lambda s: s['functions'][0]['parameters'].append(copy.deepcopy(s['functions'][0]['parameters'][0])),
            replace(['records', 1, 'cpp_type'], 'squid::SampleConfig'),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                schema = copy.deepcopy(self.schema)
                mutate(schema)
                with self.assertRaises(ValueError):
                    gen.generate(schema)

    def test_bad_json(self):
        with tempfile.TemporaryDirectory(prefix='sq-abi-json-') as tmp:
            path = Path(tmp) / 'schema.json'
            # A duplicate in an otherwise valid schema must fail too; rejecting
            # a schema that is already incomplete would not prove this guard.
            duplicate = json.dumps(self.schema).replace('"schema_version": 1',
                        '"schema_version": 1, "schema_version": 1', 1)
            for content in [duplicate, '{"schema_version":1,"schema_version":2}', '{"x":NaN}', '{', '\ud800', ' ' * (1024 * 1024 + 1)]:
                with self.subTest(content=content[:40]):
                    path.write_bytes(content.encode(errors='surrogatepass'))
                    with self.assertRaises((ValueError, UnicodeError)):
                        gen.load_schema(path)

    def test_no_reference_or_manifest_dependency_and_fail_closed(self):
        # A separate tree has only the generator and schema: no Zig, manifest,
        # upstream helper, kernels or copy of the current generated artifacts.
        with tempfile.TemporaryDirectory(prefix='sq-abi-independent-') as tmp:
            root = Path(tmp)
            (root / 'tools').mkdir()
            (root / 'abi').mkdir()
            shutil.copy2(ROOT / 'tools/generate_cuda_abi.py', root / 'tools')
            shutil.copy2(ROOT / 'abi/cuda.json', root / 'abi')
            command = [sys.executable, str(root / 'tools/generate_cuda_abi.py')]
            subprocess.run(command, check=True, capture_output=True)
            subprocess.run(command + ['--check'], check=True, capture_output=True)
            for name, value in gen.generate(self.schema).items():
                self.assertEqual((root / name).read_text(), value)
            # Stale output is detected, never silently regenerated by --check.
            header = root / 'src/cuda_abi.h'
            header.write_text('stale\n')
            self.assertNotEqual(subprocess.run(command + ['--check'], capture_output=True).returncode, 0)
            self.assertEqual(header.read_text(), 'stale\n')
            # Invalid schemas must not modify any preexisting output.
            before = {name: (root / name).read_bytes() for name in gen.generate(self.schema)}
            (root / 'abi/cuda.json').write_text('{"schema_version":99}')
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(before, {name: (root / name).read_bytes() for name in before})


if __name__ == '__main__':
    unittest.main()
