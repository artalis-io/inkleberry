/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_TEST_PLATFORM_H
#define IFN_TEST_PLATFORM_H
#include <stddef.h>
#include <string.h>
/* Test infrastructure only: the library does not own worker threads. */
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#include <io.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    HANDLE handle;
    void *(*run)(void *);
    void *argument, *result;
} ifn_test_thread_record;

typedef ifn_test_thread_record *ifn_test_thread;

static unsigned __stdcall ifn_test_thread_entry(void *p) {
    ifn_test_thread thread = p;
    thread->result = thread->run(thread->argument);
    return 0;
}

static inline int ifn_test_thread_create(ifn_test_thread *thread, const void *attributes,
                                         void *(*run)(void *), void *argument) {
    (void)attributes;
    *thread = malloc(sizeof **thread);
    if (!*thread)
        return 1;
    (*thread)->run = run;
    (*thread)->argument = argument;
    (*thread)->result = NULL;
    (*thread)->handle = (HANDLE)_beginthreadex(NULL, 0, ifn_test_thread_entry, *thread, 0, NULL);
    if ((*thread)->handle)
        return 0;
    free(*thread);
    *thread = NULL;
    return 1;
}

static inline int ifn_test_thread_join(ifn_test_thread thread, void **result) {
    /* The caller's record remains alive until this wait completes. */
    if (WaitForSingleObject(thread->handle, INFINITE) != WAIT_OBJECT_0)
        return 1;
    if (!CloseHandle(thread->handle))
        return 1;
    if (result)
        *result = thread->result;
    free(thread);
    return 0;
}

static inline int ifn_test_yield(void) {
    (void)SwitchToThread();
    return 0;
}

static inline void *ifn_test_reserve(size_t bytes) {
    return VirtualAlloc(NULL, bytes, MEM_RESERVE, PAGE_NOACCESS);
}

static inline int ifn_test_unreserve(void *data, size_t bytes) {
    (void)bytes;
    return VirtualFree(data, 0, MEM_RELEASE) ? 0 : 1;
}
#else
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>
#ifdef _DEFAULT_SOURCE
#include <sys/mman.h>
#endif
typedef pthread_t ifn_test_thread;

static inline int ifn_test_thread_create(ifn_test_thread *t, const void *attributes,
                                         void *(*run)(void *), void *argument) {
    (void)attributes;
    return pthread_create(t, NULL, run, argument);
}

static inline int ifn_test_thread_join(ifn_test_thread t, void **result) {
    return pthread_join(t, result);
}

static inline int ifn_test_yield(void) {
    return sched_yield();
}
#ifdef _DEFAULT_SOURCE
static inline void *ifn_test_reserve(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

static inline int ifn_test_unreserve(void *data, size_t bytes) {
    return munmap(data, bytes);
}
#endif
#endif
#endif
