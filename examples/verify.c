/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "ifn.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fputs("usage: verify MODEL_DIR TOKENIZER_SHA256 WEIGHTS_SHA256 [MTP_SHA256]\n", stderr);
        return 2;
    }
    const void *raw = NULL;
    if (ifn_infer_get_api(1, 1, &raw))
        return 1;
    const ifn_infer_api_v1 *api = raw;
    ifn_infer_model_config_v1 config;
    IFN_INFER_INIT(config);
    ifn_infer_source_v1 source;
    IFN_INFER_INIT(source);
    ifn_infer_artifact_manifest_v1 manifest;
    IFN_INFER_INIT(manifest);
    ifn_infer_extension_v1 ext;
    IFN_INFER_INIT(ext);
    ifn_infer_model_info_v1 info;
    IFN_INFER_INIT(info);
    ifn_infer_error_v1 error;
    IFN_INFER_INIT(error);
    ifn_infer_model *model = NULL;
    if (api->defaults(IFN_DEFAULT_MODEL, &config, &error))
        return 1;
    if (argc == 5)
        config.weight_profile = (ifn_infer_bytes){"nvfp4-mtp", 9};
    config.required_features |= IFN_INFER_CAP_ARTIFACT_VERIFY;
    manifest.tokenizer_sha256 = (ifn_infer_bytes){argv[2], strlen(argv[2])};
    manifest.weights_sha256 = (ifn_infer_bytes){argv[3], strlen(argv[3])};
    if (argc == 5)
        manifest.mtp_sha256 = (ifn_infer_bytes){argv[4], strlen(argv[4])};
    ext.id = IFN_INFER_EXT_ARTIFACT_MANIFEST;
    ext.version = 1;
    ext.payload = &manifest;
    ext.payload_bytes = sizeof manifest;
    config.extensions = (ifn_infer_array){&ext, 1, sizeof ext, 0};
    source.directory = (ifn_infer_bytes){argv[1], strlen(argv[1])};
    source.format_id = (ifn_infer_bytes){"inkling-safetensors", sizeof("inkling-safetensors") - 1};
    ifn_status rc = api->model_create(&config, &model, &error);
    if (!rc)
        rc = api->model_load(model, &source, &error);
    if (!rc)
        rc = api->model_info(model, &info, &error);
    if (!rc && (!info.fingerprint_verified || info.artifact_fingerprint.bytes != 64))
        rc = IFN_FORMAT;
    if (rc)
        fprintf(stderr, "verify: %s: %.*s\n", ifn_status_string(rc), (int)error.message_bytes,
                error.message);
    else
        printf("%.*s\n", (int)info.artifact_fingerprint.bytes,
               (const char *)info.artifact_fingerprint.data);
    if (api->model_close(&model, &error))
        rc = IFN_GPU_ERROR;
    return rc ? 1 : 0;
}
