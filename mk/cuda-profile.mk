# One target image per artifact. Experimental Ampere is never selected implicitly.
EXPERIMENTAL_AMPERE ?= 0
ifneq ($(words $(CUDA_ARCH)),1)
$(error CUDA_ARCH must select exactly one target)
endif
ifneq ($(words $(EXPERIMENTAL_AMPERE)),1)
$(error EXPERIMENTAL_AMPERE must be 0 or 1)
endif
ifneq ($(filter $(CUDA_ARCH),120a 121a 86),$(CUDA_ARCH))
$(error CUDA_ARCH must be 120a, 121a, or experimental 86)
endif
ifneq ($(filter $(EXPERIMENTAL_AMPERE),0 1),$(EXPERIMENTAL_AMPERE))
$(error EXPERIMENTAL_AMPERE must be 0 or 1)
endif
ifeq ($(CUDA_ARCH),86)
ifneq ($(EXPERIMENTAL_AMPERE),1)
$(error CUDA_ARCH=86 requires EXPERIMENTAL_AMPERE=1; runtime qualification is pending)
endif
endif
CUDA_TARGET_FLAGS = -DIFN_CUDA_TARGET_SM=$(patsubst %a,%,$(CUDA_ARCH)) -DIFN_CUDA_ENABLE_EXPERIMENTAL_AMPERE=$(EXPERIMENTAL_AMPERE)
IFN_PACKAGE_BUILD_ROOT = .
ifneq ($(IFN_NATIVE_MSVC),)
IFN_PACKAGE_BUILD_ROOT = $(BUILD_DIR)
else ifeq ($(CUDA_ARCH),86)
# Preserve caller-selected base directories and avoid appending on recursive make.
ifneq ($(notdir $(BUILD_DIR)),sm86-experimental)
IFN_CUDA_BASE_BUILD_DIR := $(BUILD_DIR)
override BUILD_DIR := $(IFN_CUDA_BASE_BUILD_DIR)/sm86-experimental
endif
HOST_STATIC = $(BUILD_DIR)/libinkleberry.a
INFER_STATIC = $(BUILD_DIR)/libinkleberry-infer.a
GPU_STATIC = $(BUILD_DIR)/libinkleberry-gpu.a
CUDA_STATIC = $(BUILD_DIR)/libinkleberry-cuda.a
CLI = $(BUILD_DIR)/inkleberry
GENERATE = $(BUILD_DIR)/inkleberry-generate
WEIGHTS_CLI = $(BUILD_DIR)/inkleberry-weights
PROVIDER = $(BUILD_DIR)/libinkleberry_backend.so.1
IFN_CUDA_SHARED := $(notdir $(SHARED))
IFN_CUDA_SHARED_REAL := $(notdir $(SHARED_REAL))
IFN_CUDA_SHARED_ABI := $(notdir $(SHARED_ABI))
SHARED = $(BUILD_DIR)/$(IFN_CUDA_SHARED)
SHARED_REAL = $(BUILD_DIR)/$(IFN_CUDA_SHARED_REAL)
SHARED_ABI = $(BUILD_DIR)/$(IFN_CUDA_SHARED_ABI)
ifeq ($(shell uname -s),Darwin)
SHARED_FLAGS = -dynamiclib -Wl,-install_name,@rpath/$(notdir $(SHARED_ABI))
else
SHARED_FLAGS = -shared -Wl,-soname,$(notdir $(SHARED_ABI))
endif
RPATH_ROOT = -Wl,-rpath,'$(abspath $(BUILD_DIR))'
IFN_PACKAGE_BUILD_ROOT = $(BUILD_DIR)
endif
export CUDA_ARCH EXPERIMENTAL_AMPERE
