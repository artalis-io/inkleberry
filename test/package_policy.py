#!/usr/bin/env python3
"""Package roots must not follow external/private symlinks."""
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import release

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
    def test_bad_version_before_writing(self):
        with tempfile.TemporaryDirectory(prefix='package-version-') as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):release.package('source',root,version='../private')
            self.assertEqual(list(root.iterdir()),[])

if __name__=='__main__':unittest.main()
