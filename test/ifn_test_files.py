"""Native sparse test fixtures; never materialize model-sized zero payloads."""
import os
if os.name == 'nt':
    import ctypes as c
    from ctypes import wintypes as w
    import msvcrt
    kernel = c.WinDLL('kernel32', use_last_error=True)
    kernel.DeviceIoControl.argtypes = [w.HANDLE, w.DWORD, c.c_void_p, w.DWORD,
                                      c.c_void_p, w.DWORD, c.POINTER(w.DWORD), c.c_void_p]
    kernel.DeviceIoControl.restype = w.BOOL
    kernel.SetFilePointerEx.argtypes = [w.HANDLE, c.c_longlong, c.c_void_p, w.DWORD]
    kernel.SetFilePointerEx.restype = w.BOOL
    kernel.SetEndOfFile.argtypes = [w.HANDLE]
    kernel.SetEndOfFile.restype = w.BOOL
    kernel.GetCompressedFileSizeW.argtypes = [w.LPCWSTR, c.POINTER(w.DWORD)]
    kernel.GetCompressedFileSizeW.restype = w.DWORD

def sparse_resize(file, size):
    file.flush()
    if os.name != 'nt':
        file.truncate(size)
        return
    handle = msvcrt.get_osfhandle(file.fileno())
    returned = w.DWORD()
    if not kernel.DeviceIoControl(handle, 0x900C4, None, 0, None, 0, c.byref(returned), None):
        raise c.WinError(c.get_last_error())
    if not kernel.SetFilePointerEx(handle, size, None, 0) or not kernel.SetEndOfFile(handle):
        raise c.WinError(c.get_last_error())
    file.seek(0)

def allocated_bytes(path):
    if os.name != 'nt':
        return path.stat().st_blocks * 512
    high = w.DWORD()
    c.set_last_error(0)
    low = kernel.GetCompressedFileSizeW(str(path), c.byref(high))
    if low == 0xFFFFFFFF and c.get_last_error():
        raise c.WinError(c.get_last_error())
    return (high.value << 32) | low

def fixture_alias(source, destination):
    if os.name == 'nt':
        # NTFS hard links need no symlink privilege and do not copy huge holes.
        os.link(source, destination)
    else:
        destination.symlink_to(source.name)
