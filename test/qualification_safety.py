#!/usr/bin/env python3
"""Real process-tree and source-provenance regressions; no GPU/model required."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import process_tree as tree
import qualification_source as source


def alive(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes as w
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];api.OpenProcess.restype=w.HANDLE
        api.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD];api.WaitForSingleObject.restype=w.DWORD
        api.CloseHandle.argtypes=[w.HANDLE];api.CloseHandle.restype=w.BOOL
        handle=api.OpenProcess(0x100000,False,pid)
        if not handle:return False
        try:return api.WaitForSingleObject(handle,0)==258
        finally:api.CloseHandle(handle)
    p=subprocess.run(['ps','-p',str(pid),'-o','stat='],text=True,capture_output=True)
    return p.returncode==0 and p.stdout.strip() and not p.stdout.strip().startswith('Z')


class Safety(unittest.TestCase):
    def worker(self,root,finish=False):
        path=root/'worker.py';pidfile=root/'pids.json'
        path.write_text('import subprocess,sys,os,json,time\nfrom pathlib import Path\n'
                        'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])\n'
                        'Path(sys.argv[1]).write_text(json.dumps([os.getpid(),p.pid]))\n'+
                        ('' if finish else 'time.sleep(60)\n'))
        return [sys.executable,str(path),str(pidfile)],pidfile
    def assert_gone(self,pidfile):
        self.assertTrue(pidfile.exists(),'Fixture never launched')
        pids=json.loads(pidfile.read_text());deadline=time.monotonic()+10
        while any(alive(pid) for pid in pids) and time.monotonic()<deadline:time.sleep(.05)
        self.assertFalse(any(alive(pid) for pid in pids),'Owned descendant survived cleanup')
    def test_timeout_kills_descendants_and_preserves_unrelated_process(self):
        sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cmd,pids=self.worker(Path(tmp))
                with self.assertRaises(subprocess.TimeoutExpired):tree.run(cmd,timeout=3)
                self.assert_gone(pids);self.assertIsNone(sentinel.poll())
        finally:sentinel.terminate();sentinel.wait(timeout=15)
    def test_success_also_cleans_leftover_descendants(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd,pids=self.worker(Path(tmp),finish=True)
            self.assertEqual(tree.run(cmd,timeout=10).returncode,0);self.assert_gone(pids)
    def test_parent_termination_cleans_descendants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cmd,pids=self.worker(root)
            supervisor=root/'supervisor.py'
            supervisor.write_text('import sys\nsys.path.insert(0,'+repr(str(Path(tree.__file__).parent))+')\n'
                                  'from process_tree import run\nrun('+repr(cmd)+',timeout=60)\n')
            p=subprocess.Popen([sys.executable,str(supervisor)],stderr=subprocess.DEVNULL)
            try:
                deadline=time.monotonic()+10
                while not pids.exists() and time.monotonic()<deadline:time.sleep(.05)
                self.assertTrue(pids.exists());p.terminate();p.wait(timeout=15);self.assert_gone(pids)
            finally:
                if p.poll() is None:p.kill();p.wait(timeout=15)
    @unittest.skipUnless(os.name=='nt','Windows Job Object admission')
    def test_failed_job_assignment_never_starts_command(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'must-not-exist'
            cmd=[sys.executable,'-c','from pathlib import Path;Path('+repr(str(marker))+').write_text("bad")']
            with patch.object(tree.WindowsJob,'assign',side_effect=OSError('fixture denied')):
                with self.assertRaises(OSError):tree.run(cmd,timeout=5)
            self.assertFalse(marker.exists())
    def fixture_repo(self,root):
        (root/'config').mkdir()
        (root/'VERSION').write_text('fixture\n')
        (root/'config/package.json').write_text(json.dumps({'source_files':['VERSION','config/package.json']}))
        subprocess.run(['git','init','-q',str(root)],check=True)
        subprocess.run(['git','-C',str(root),'config','core.autocrlf','false'],check=True)
        subprocess.run(['git','-C',str(root),'add','.'],check=True)
        subprocess.run(['git','-C',str(root),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-qm','fixture'],check=True)
        return source.git(root,'rev-parse','HEAD')
    def test_verified_clean_checkout_and_mismatched_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);revision=self.fixture_repo(root)
            self.assertEqual(source.verify(root,revision)['mode'],'clean-checkout')
            with self.assertRaisesRegex(ValueError,'differs'):source.verify(root,'0'*40)
            (root/'VERSION').write_text('changed\n')
            with self.assertRaisesRegex(ValueError,'clean checkout'):source.verify(root,revision)
    def test_archive_manifest_and_modified_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);root=parent/'repo';root.mkdir();revision=self.fixture_repo(root)
            manifest=parent/'source.json';source.capture(root,manifest)
            verified=source.verify(root,revision,manifest)
            self.assertEqual(verified['mode'],'archive-manifest');self.assertIn('operator-attested',verified['association'])
            (root/'VERSION').write_text('changed\n')
            with self.assertRaisesRegex(ValueError,'manifest mismatch'):source.verify(root,revision,manifest)
            with self.assertRaises(ValueError):source.verify(root,'0'*40,manifest)
    def test_extracted_archive_without_git(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);root=parent/'repo';root.mkdir();revision=self.fixture_repo(root)
            manifest=parent/'source.json';source.capture(root,manifest)
            extracted=parent/'archive';shutil.copytree(root,extracted,ignore=shutil.ignore_patterns('.git'))
            self.assertEqual(source.verify(extracted,revision,manifest)['mode'],'archive-manifest')
            with self.assertRaises(ValueError):source.verify(extracted,revision)
    def test_collector_detects_source_change_before_completion(self):
        from unittest.mock import patch
        import semantic_acceptance as s
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);root=parent/'repo';root.mkdir();revision=self.fixture_repo(root)
            binary=parent/'binary';binary.write_bytes(b'fixture');model=parent/'model';model.mkdir()
            for name in s.FILES:(model/name).write_bytes(b'fixture')
            args=argparse.Namespace(gpu='GPU-fixture',source_revision=revision,profiles=['ar'],source_manifest=None,out=parent/'out',binary=binary,model=model,cuda_runtime='/not-used')
            def changed(*a,**kw):
                (root/'VERSION').write_text('changed\n');return subprocess.CompletedProcess([],0)
            with patch.object(s,'ROOT',root),patch.object(s.platform,'platform',return_value='CPU fixture'),patch.object(s,'run_owned',side_effect=changed),patch('builtins.print'):
                with self.assertRaises(ValueError):s.run(args)
            self.assertFalse(json.loads((args.out/'run.json').read_text())['complete'])
    def test_archive_explicit_manifest_required_and_missing_files_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config').mkdir();(root/'config/package.json').write_text(json.dumps({'source_files':['missing']}))
            with self.assertRaises(ValueError):source.verify(root,'a'*40)
    def test_collector_fails_before_gpu_or_model_on_source_mismatch(self):
        from unittest.mock import patch
        import semantic_acceptance as s
        a=argparse.Namespace(gpu='GPU-fixture',source_revision='0'*40,profiles=['ar'],source_manifest=None)
        with patch.object(s,'verify_source',side_effect=ValueError('mismatch')),patch.object(s,'run_owned') as run,patch.object(s,'digest') as digest:
            with self.assertRaises(ValueError):s.run(a)
            run.assert_not_called();digest.assert_not_called()

if __name__=='__main__':unittest.main()
