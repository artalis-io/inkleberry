#!/usr/bin/env python3
"""Native PE boundaries: DEP/ASLR, private symbols and runtime dependencies."""
import os
import re
import subprocess
import sys
from pathlib import Path
assert os.name == 'nt'
for filename in sys.argv[1:]:
    text = subprocess.check_output(['llvm-readobj', '--file-headers', '--coff-imports',
                                    '--coff-exports', filename], text=True)
    assert 'IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE' in text, filename
    assert 'IMAGE_DLL_CHARACTERISTICS_NX_COMPAT' in text, filename
    assert not any(x in text.lower() for x in ['msys-', 'cygwin', 'winpthread', 'libgcc']), filename
    exports = re.findall(r'Export \{.*?\n  Name: ([^\n]+)', text, re.S)
    if Path(filename).suffix == '.dll':
        assert exports and all(x.startswith('ifn_') for x in exports), exports
        if 'mock' in filename or 'backend' in filename:
            assert exports == ['ifn_infer_get_api'], exports
        else:
            assert 'ifn_status_string' in exports
            assert not set(exports) & {'ifn_file_map_open', 'ifn_host_now_ns', 'ifn_win32_error'}
print('Native PE ASLR/DEP, exports and absence of POSIX runtime dependencies passed')
