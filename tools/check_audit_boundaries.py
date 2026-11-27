#!/usr/bin/env python3
"""Jev Bush-style source boundary checks adapted to IFN's status/lease contract.

Traversal/lexical approach adapted from artalis-io/jev-bush tools/
check_error_boundaries.py and check_parallel_regions.py at 85fd51f
(BSD-0-Clause, Copyright 2026 Artalis). This is a lexical regression guard,
not a C/C++ parser or proof about indirect calls and preprocessor expansion.
"""
import argparse
from collections import deque
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONTROL_ROOTS = {
    'ifn_version', 'ifn_status_string', 'ifn_last_error', 'ifn_native_diagnostic_get',
    'ifn_size_add', 'ifn_size_mul', 'ifn_size_align', 'ifn_session_cancel',
    'ifn_infer_request_cancel', 'ifn_infer_request_progress',
}
# Indirect allocator/backend members are deliberately terminal names too.
BLOCKING = {
    'malloc','calloc','realloc','free','allocate','release','sync','open','fopen',
    'read','write','fread','fwrite','fclose','fprintf','printf','puts','fputs',
    'sleep','usleep','nanosleep','pthread_mutex_lock','mtx_lock',
    'ifn_file_map_open','ifn_file_map_close','cudaMalloc','cudaFree',
    'cudaDeviceSynchronize','cudaStreamSynchronize','ifn_cuda_stream_sync',
    'ifn_cuda_alloc','ifn_cuda_free','ifn_cuda_enter','ifn_cuda_leave',
    'abort','exit','_Exit','quick_exit','longjmp','setjmp','system','popen',
}
OS_CALLS = {'open','close','fstat','mmap','munmap','clock_gettime','CreateFileW',
            'CreateFileMappingW','MapViewOfFile','UnmapViewOfFile','CloseHandle',
            'QueryPerformanceCounter','QueryPerformanceFrequency'}


def strip(source):
    """Blank comments/literals without moving source offsets or line numbers."""
    out = list(source);i = 0
    while i < len(source):
        if source.startswith('/*', i):
            end = source.find('*/', i+2)
            if end < 0:raise ValueError('Unterminated block comment')
            end += 2
        elif source.startswith('//', i):
            end = source.find('\n', i)
            if end < 0:end = len(source)
        elif source[i] in "\"'":
            quote = source[i];end = i+1
            while end < len(source) and source[end] != quote:
                end += 2 if source[end] == '\\' else 1
            if end >= len(source):raise ValueError('Unterminated literal')
            end += 1
        else:
            i += 1;continue
        for k in range(i, end):
            if out[k] != '\n':out[k] = ' '
        i = end
    return ''.join(out)


def match(text, start):
    depth = 0
    for i in range(start,len(text)):
        if text[i] == '{':depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:return i+1
    raise ValueError('Unbalanced function body')


def functions(text):
    result = {}
    pattern = r'^[ \t]*[A-Za-z_][\w \t\*]*?\b(ifn_\w+)\s*\([^;{}]*\)\s*\{'
    for m in re.finditer(pattern,text,re.M):
        start = m.end()-1
        result[m[1]] = text[start:match(text,start)]
    return result


def calls(text):
    return set(re.findall(r'\b([A-Za-z_]\w*)\s*\(',text))


def path_to(graph, root, targets):
    pending = deque([(root,[root])]);seen = set()
    while pending:
        name,path = pending.popleft()
        if name in seen:continue
        seen.add(name)
        if name in targets:return path
        pending.extend((c,path+[c]) for c in sorted(graph.get(name,())))
    return None


def audit(root):
    graph = {};findings = [];files = sorted((root/'src').glob('*.c'))
    if not files:raise ValueError('No canonical C source files')
    for file in files:
        text = strip(file.read_text(encoding='utf-8'))
        for name,body in functions(text).items():
            # POSIX/Windows definitions are mutually exclusive; check both branches.
            graph.setdefault(name,set()).update(calls(body))
        if file.name not in ('host_posix.c','host_win.c'):
            for m in re.finditer(r'\b('+'|'.join(sorted(OS_CALLS))+r')\s*\(',text):
                findings.append(f'{file.relative_to(root)}:{text.count(chr(10),0,m.start())+1}: OS primitive outside platform layer: {m[1]}')
    hardware_calls = {c for callees in graph.values() for c in callees
                      if c.startswith(('ifn_cuda_', 'ifn_launch_', 'cuda', 'cublas'))}
    for name in sorted(CONTROL_ROOTS):
        if name not in graph:
            findings.append('Missing audited control root: '+name);continue
        path = path_to(graph,name,BLOCKING | hardware_calls)
        if path:findings.append('Control path may allocate, block or terminate: '+' -> '.join(path))
    headers = sorted((root/'include').glob('*.h'))
    if not headers:findings.append('Missing public headers')
    for file in headers:
        # Includes need their literal text; match directives only, not comments.
        text = re.sub(r'/\*.*?\*/|//[^\n]*','',file.read_text(encoding='utf-8'),flags=re.S)
        for m in re.finditer(r'^\s*#\s*include\s*[<"](windows\.h|cuda[^>"\n]*|pthread\.h|dlfcn\.h|sys/[^>"\n]*)[>"]',text,re.M):
            findings.append(f'{file.relative_to(root)}: platform SDK leaked into public API: {m[1]}')
    return findings,len(files),len(graph)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    try:findings,files,functions_count=audit(args.root)
    except (ValueError,OSError) as error:parser.exit(1,str(error)+'\n')
    if findings:parser.exit(1,'\n'.join(findings)+'\n')
    print(f'{files} C modules, {functions_count} lexical functions and {len(CONTROL_ROOTS)} control roots checked; platform boundaries intact')

if __name__=='__main__':main()
