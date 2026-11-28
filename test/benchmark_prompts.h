/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_BENCHMARK_PROMPTS_H
#define IFN_BENCHMARK_PROMPTS_H
#include <stdio.h>
#include <string.h>
static const char *const ifn_benchmark_prompts[] = {
    "What is 9 + 8? Answer with the number only.",
    "Explain briefly why a delivery truck must obey a gross vehicle weight limit.",
    "Write one short paragraph describing a quiet autumn morning.",
    "Write a C function that computes the maximum of an integer array. Return only code.",
    "Fix this Python function to avoid a mutable default argument: def add(x, items=[]): "
    "items.append(x); return items. Return code.",
    "Extract JSON with name and quantity: The customer is Acme and the quantity is 17. Return "
    "only JSON.",
    "What is the access code for record NEEDLE? Copy just the code."};

static inline int ifn_benchmark_prompt(unsigned which, char *text, size_t capacity) {
    if (which >= 7 || capacity < 100000)
        return 1;
    text[0] = 0;
    if (which == 6) {
        const char *pad = "Record cedar contains ordinary inventory and no access code. ";
        for (unsigned i = 0; i < 400; i++)
            strcat(text, pad);
        strcat(text, "Record NEEDLE has access code COBALT-7319.\n");
        for (unsigned i = 0; i < 400; i++)
            strcat(text, pad);
    }
    strcat(text, ifn_benchmark_prompts[which]);
    return 0;
}
#endif
