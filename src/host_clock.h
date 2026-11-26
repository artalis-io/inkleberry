/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_HOST_CLOCK_H
#define IFN_HOST_CLOCK_H
#include <stdint.h>
#include <stdbool.h>
/* Monotonic nanoseconds; zero on failure. Never substitutes wall time. */
uint64_t ifn_host_now_ns(void);
/* Exact, checked QPC conversion, independent of OS calls for boundary tests. */
bool ifn_clock_from_ticks(uint64_t, uint64_t, uint64_t *);
#endif
