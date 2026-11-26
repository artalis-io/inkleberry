/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "host_clock.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc == 3) {
        uint64_t ns;
        if (ifn_clock_from_ticks(strtoull(argv[1], NULL, 10), strtoull(argv[2], NULL, 10), &ns))
            printf("%" PRIu64 "\n", ns);
        else
            puts("overflow");
        return 0;
    }
    uint64_t first = ifn_host_now_ns();
    if (!first || ifn_clock_from_ticks(1, 0, &first) || first || ifn_clock_from_ticks(1, 1, NULL))
        return 1;
    first = ifn_host_now_ns();
    for (int i = 0; i < 10000; i++) {
        uint64_t next = ifn_host_now_ns();
        if (next < first)
            return 1;
        first = next;
    }
    puts("monotonic host clock passed");
    return 0;
}
