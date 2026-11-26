/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "inkling.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define IFN_INKLING_LOCAL_CHECK(x)                                                                 \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "inkling:%d: %s (%s)\n", __LINE__, #x, ifn_last_error());              \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

typedef struct {
    size_t main, mtp;
    bool dump;
    bool first;
} ifn_inkling_local_state;

static ifn_status ifn_inkling_local_check_spec(void *context, const ifn_inkling_spec *s) {
    ifn_inkling_local_state *st = context;
    if (s->source)
        st->mtp++;
    else
        st->main++;
    ifn_tensor_info t = {0};
    memcpy(t.name, s->name, strlen(s->name) + 1);
    t.dtype = s->dtype;
    t.rank = s->rank;
    for (uint32_t i = 0; i < s->rank; i++)
        t.shape[i] = s->shape[i];
    if (ifn_inkling_check_descriptor(&t, s) != IFN_OK)
        return IFN_FORMAT;
    t.rank++;
    if (ifn_inkling_check_descriptor(&t, s) != IFN_FORMAT)
        return IFN_INVALID_ARGUMENT;
    t.rank--;
    t.dtype = (s->dtype + 1) % 4;
    if (ifn_inkling_check_descriptor(&t, s) != IFN_FORMAT)
        return IFN_INVALID_ARGUMENT;
    t.dtype = s->dtype;
    t.shape[0]++;
    if (ifn_inkling_check_descriptor(&t, s) != IFN_FORMAT)
        return IFN_INVALID_ARGUMENT;
    if (st->dump) {
        printf("%s{\"source\":%u,\"name\":\"%s\",\"dtype\":%u,\"shape\":[", st->first ? "" : ",",
               s->source, s->name, s->dtype);
        for (uint32_t i = 0; i < s->rank; i++)
            printf("%s%llu", i ? "," : "", (unsigned long long)s->shape[i]);
        fputs("]}", stdout);
        st->first = false;
    }
    return IFN_OK;
}

typedef struct {
    size_t calls, fail, live;
} ifn_inkling_local_faults;

static void *ifn_inkling_local_alloc(void *context, size_t bytes) {
    ifn_inkling_local_faults *f = context;
    if (f->calls++ == f->fail)
        return NULL;
    void *p = malloc(bytes);
    if (p)
        f->live++;
    return p;
}

static void ifn_inkling_local_release(void *context, void *p) {
    ifn_inkling_local_faults *f = context;
    if (p)
        f->live--;
    free(p);
}

int main(int argc, char **argv) {
    ifn_inkling_local_state st = {0, 0, argc == 2 && !strcmp(argv[1], "--schema"), true};
    if (st.dump)
        fputs("[", stdout);
    IFN_INKLING_LOCAL_CHECK(ifn_inkling_visit_specs(1, ifn_inkling_local_check_spec, &st) ==
                                IFN_OK &&
                            st.main == 822 && st.mtp == 54);
    if (st.dump) {
        puts("]");
        return 0;
    }
    IFN_INKLING_LOCAL_CHECK(ifn_inkling_visit_specs(2, ifn_inkling_local_check_spec, &st) ==
                            IFN_INVALID_ARGUMENT);
    if (argc == 1) {
        puts("876 Inkling roles: dtype/rank/shape rejection passed");
        return 0;
    }
    bool invalid = argc == 4 && !strcmp(argv[1], "--invalid");
    IFN_INKLING_LOCAL_CHECK(invalid || argc == 3);
    const char *main_path = argv[invalid ? 2 : 1], *mtp = argv[invalid ? 3 : 2];
    for (size_t fail = invalid ? 5 : 0; fail <= 5; fail++) {
        ifn_inkling_local_faults f = {0, fail, 0};
        ifn_options o = IFN_OPTIONS_INIT;
        o.allocate = ifn_inkling_local_alloc;
        o.release = ifn_inkling_local_release;
        o.context = &f;
        ifn_inkling_checkpoint *c = (ifn_inkling_checkpoint *)(uintptr_t)1;
        ifn_status status = ifn_inkling_checkpoint_open(main_path, mtp, &o, &c);
        if (invalid) {
            IFN_INKLING_LOCAL_CHECK(status != IFN_OK && !c && !f.live);
            break;
        }
        IFN_INKLING_LOCAL_CHECK(status == (fail < 5 ? IFN_OUT_OF_MEMORY : IFN_OK));
        IFN_INKLING_LOCAL_CHECK(status == IFN_OK ? c != NULL : c == NULL);
        if (c) {
            IFN_INKLING_LOCAL_CHECK(ifn_inkling_checkpoint_count(c) == 876);
            for (size_t i = 0; i < 876; i++) {
                uint32_t source;
                const ifn_tensor_info *info = NULL;
                const void *data = NULL;
                size_t bytes = 0;
                IFN_INKLING_LOCAL_CHECK(ifn_inkling_checkpoint_at(c, i, &source, &info) == IFN_OK &&
                                        source == (i < 822 ? 0u : 1u));
                IFN_INKLING_LOCAL_CHECK(ifn_inkling_checkpoint_data(c, i, &data, &bytes) ==
                                            IFN_OK &&
                                        data && bytes == info->bytes);
            }
            uint32_t source = 99;
            const ifn_tensor_info *info = (const ifn_tensor_info *)(uintptr_t)1;
            IFN_INKLING_LOCAL_CHECK(ifn_inkling_checkpoint_at(c, 876, &source, &info) ==
                                        IFN_NOT_FOUND &&
                                    !source && !info);
        }
        ifn_inkling_checkpoint_close(c);
        IFN_INKLING_LOCAL_CHECK(!f.live);
    }
    puts(invalid ? "invalid checkpoint rejected and released"
                 : "bound data/metadata and all checkpoint allocation failure sites passed");
    return 0;
}
