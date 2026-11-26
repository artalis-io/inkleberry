/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "host_clock.h"

bool ifn_clock_from_ticks(uint64_t ticks, uint64_t frequency, uint64_t *out) {
    if (!out)
        return false;
    *out = 0;
    if (!frequency || ticks / frequency > UINT64_MAX / UINT64_C(1000000000))
        return false;
    uint64_t whole = (ticks / frequency) * UINT64_C(1000000000);
    uint64_t remainder = ticks % frequency, fraction = 0;
    if (remainder <= UINT64_MAX / UINT64_C(1000000000)) {
        fraction = remainder * UINT64_C(1000000000) / frequency;
    } else {
        /* floor(remainder * 1e9 / frequency), without a 128-bit extension.
         * At every bit, fraction/remainder hold the quotient/remainder of the
         * partial product. Subtraction tests avoid overflowing a doubled sum. */
        uint64_t rest = 0;
        for (int bit = 29; bit >= 0; bit--) {
            fraction *= 2;
            if (rest >= frequency - rest) {
                rest -= frequency - rest;
                fraction++;
            } else {
                rest += rest;
            }
            if ((UINT64_C(1000000000) >> bit) & 1) {
                if (rest >= frequency - remainder) {
                    rest -= frequency - remainder;
                    fraction++;
                } else {
                    rest += remainder;
                }
            }
        }
    }
    if (fraction > UINT64_MAX - whole)
        return false;
    *out = whole + fraction;
    return true;
}
