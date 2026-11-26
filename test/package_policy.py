#!/usr/bin/env python3
"""Package roots must not follow external/private symlinks."""
from pathlib import Path
import sys
import json
import tarfile
from unittest.mock import patch
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import release
from cuda_profile import profile, validate

class PackagePolicyTests(unittest.TestCase):
    def test_safe_names(self):
        for name in ['', '../secret', '/secret', './secret', 'x/../secret', 'x//secret']:
            with self.subTest(name=name),self.assertRaises(ValueError):release.safe(name)
        self.assertEqual(release.safe('src/module.c'),'src/module.c')
    def test_symlinks_and_escape(self):
        with tempfile.TemporaryDirectory(prefix='package-policy-') as tmp:
            base=Path(tmp); inside=base/'inside'; outside=base/'outside'
            inside.mkdir();outside.mkdir();(outside/'private').write_bytes(b'not a release input')
            (inside/'file').symlink_to(outside/'private')
            (inside/'dir').symlink_to(outside,target_is_directory=True)
            for path in [inside/'file',inside/'dir/private',outside/'private']:
                with self.subTest(path=path),self.assertRaises(ValueError):release.read_file(path,inside)
            (inside/'public').write_bytes(b'public')
            self.assertEqual(release.read_file(inside/'public',inside),b'public')
    def test_gpu_profiles_are_explicit_and_archives_do_not_collide(self):
        version = (ROOT / 'VERSION').read_text().strip()
        with tempfile.TemporaryDirectory(prefix='ifn-profile-package-') as tmp:
            base = Path(tmp); build = base / 'build'; build.mkdir()
            for name in ['inkleberry', 'libinkleberry.so.' + version, 'libinkleberry.a',
                         'inkleberry-generate', 'libinkleberry_backend.so.1',
                         'libinkleberry-cuda.a', 'libinkleberry-gpu.a', 'libinkleberry-infer.a']:
                (build / name).write_bytes(b'public synthetic packaging fixture; never executed')
            with patch.object(release.platform, 'system', return_value='Linux'):
                with self.assertRaises(ValueError):
                    release.package('runtime', base / 'archives', gpu=True, build_root=build)
                paths = []
                for arch in ['120a', '121a', '86']:
                    identity = profile(arch, arch == '86')
                    (build / 'cuda-profile.json').write_text(json.dumps(identity))
                    path = release.package('runtime', base / 'archives', gpu=True, build_root=build)
                    paths.append(path)
                    with tarfile.open(path) as archive:
                        manifest = json.load(archive.extractfile(next(m for m in archive if m.name.endswith('/MANIFEST.json'))))
                        self.assertEqual(manifest['cuda_profile'], identity)
                        self.assertIn('CUDA-PROFILE.json', manifest['files'])
                    self.assertIn('-sm' + arch, path.name)
                self.assertEqual(len(set(paths)), 3)
                self.assertTrue(all(path.is_file() for path in paths))
                self.assertIn('experimental', paths[-1].name)
                # Identity inputs may not escape the selected artifact root.
                outside = base / 'outside.json'; outside.write_text(json.dumps(profile('120a')))
                with self.assertRaises(ValueError):
                    release.package('runtime', base / 'archives', gpu=True, build_root=build, cuda_profile=outside)

    def test_profile_schema_fails_closed(self):
        for arch in ['80', '89', '90', '', None]:
            with self.subTest(arch=arch), self.assertRaises(ValueError): profile(arch)
        with self.assertRaises(ValueError): profile('86')
        for value in [{}, {'cuda_arch': '86', 'experimental': 1},
                      dict(profile('86', True), launch_policy='pdl'),
                      dict(profile('120a'), qualification='passed'),
                      dict(profile('120a'), format_version=True)]:
            with self.subTest(value=value), self.assertRaises(ValueError): validate(value)

    def test_bad_version_before_writing(self):
        with tempfile.TemporaryDirectory(prefix='package-version-') as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):release.package('source',root,version='../private')
            self.assertEqual(list(root.iterdir()),[])

if __name__=='__main__':unittest.main()
