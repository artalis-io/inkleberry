#!/usr/bin/env python3
"""Own a qualification subprocess group/job; never target unrelated processes."""
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


class WindowsJob:
    def __init__(self):
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [('process_time',ctypes.c_int64),('job_time',ctypes.c_int64),
                        ('flags',w.DWORD),('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),
                        ('active',w.DWORD),('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(n,ctypes.c_uint64) for n in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class Limits(ctypes.Structure):
            _fields_ = [('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),
                        ('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        for name,args,result in [
            ('CreateJobObjectW',[ctypes.c_void_p,w.LPCWSTR],w.HANDLE),
            ('SetInformationJobObject',[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD],w.BOOL),
            ('AssignProcessToJobObject',[w.HANDLE,w.HANDLE],w.BOOL),
            ('CloseHandle',[w.HANDLE],w.BOOL)]:
            fn=getattr(self.api,name);fn.argtypes=args;fn.restype=result
        self.handle=self.api.CreateJobObjectW(None,None)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
        limits=Limits();limits.basic.flags=0x2000 # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            error=ctypes.WinError(ctypes.get_last_error());self.close();raise error

    def assign(self,process):
        if not self.api.AssignProcessToJobObject(self.handle,int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle);self.handle=None


def run(command, *, stdout=None, stderr=None, env=None, timeout=650):
    """Child cannot spawn the real command before containment is established."""
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('Qualification process supervision requires the main thread')
    handlers={}
    def interrupted(signum,frame):raise KeyboardInterrupt('Qualification interrupted by signal '+str(signum))
    job=None
    process=None
    try:
        for name in ('SIGTERM','SIGINT','SIGHUP','SIGBREAK'):
            sig=getattr(signal,name,None)
            if sig is not None:handlers[sig]=signal.signal(sig,interrupted)
        job=WindowsJob() if os.name=='nt' else None
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--child',*command],
                                 stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,env=env,
                                 start_new_session=os.name!='nt')
        if job:job.assign(process)
        process.stdin.write(b'1');process.stdin.close()
        result=process.wait(timeout=timeout)
        return subprocess.CompletedProcess(command,result)
    finally:
        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        try:
            if process:
                if job:
                    job.close() # includes descendants after wrapper exit
                else:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                if process.poll() is None:
                    process.kill();process.wait(timeout=15)
                if process.stdin and not process.stdin.closed:process.stdin.close()
        finally:
            if job:job.close()
            for sig,handler in handlers.items():signal.signal(sig,handler)



if __name__=='__main__':
    if len(sys.argv)<3 or sys.argv[1]!='--child':raise SystemExit(2)
    if sys.stdin.buffer.read(1)!=b'1':raise SystemExit(2)
    sys.stdin.close()
    raise SystemExit(subprocess.run(sys.argv[2:],stdin=subprocess.DEVNULL).returncode)
