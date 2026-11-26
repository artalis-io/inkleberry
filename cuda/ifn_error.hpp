/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#pragma once
namespace ifn_cuda {
/* Host launch failures are worker-thread-local. The existing checked launch
 * boundary consumes the first error; successful launches add no synchronization. */
inline thread_local int ifn_cuda_pending_error = 0;
inline void ifn_cuda_record_error(int code) {
    if (!ifn_cuda_pending_error) ifn_cuda_pending_error = code;
}
inline int ifn_cuda_take_error() {
    int code = ifn_cuda_pending_error;
    ifn_cuda_pending_error = 0;
    return code;
}
}
