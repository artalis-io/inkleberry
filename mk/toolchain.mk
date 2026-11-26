# Compiler intent only; OS services remain in the private host adapters.
# Never assign LIB, INCLUDE, LINK or CL: these are MSVC environment variables.
IFN_CC_NAME := $(notdir $(CC))
IFN_NATIVE_MSVC := $(filter cl cl.exe clang-cl clang-cl.exe,$(IFN_CC_NAME))
IFN_WINDOWS := $(if $(filter Windows_NT windows,$(OS)),1,0)
PYTHON ?= python3
BUILD_DIR ?= .build
HOST_STATIC = libinkleberry.a
INFER_STATIC = libinkleberry-infer.a
GPU_STATIC = libinkleberry-gpu.a
CUDA_STATIC = libinkleberry-cuda.a
CLI = inkleberry
GENERATE = inkleberry-generate
WEIGHTS_CLI = inkleberry-weights
PROVIDER = libinkleberry_backend.so.1
MOCK_PROVIDER = $(BUILD_DIR)/infer-mock.so
EXE :=
HOST_IMPORT :=
INSTALLED_DLLS :=
BACKEND_IMPORT :=
C_STD := -std=c11
CXX_WARNFLAGS := -std=c++11 -Wall -Wextra -Werror
CXX20_WARNFLAGS := -O2 -std=c++20 -Wall -Wextra -Werror
DEPFLAGS := -MMD -MP
PIC_FLAGS := -fPIC
HIDDEN_FLAGS := -fvisibility=hidden
THREAD_LIBS := -pthread
CXX_LANGUAGE := -x c++
SYNTAX_ONLY := -fsyntax-only
RPATH_ROOT = -Wl,-rpath,'$(CURDIR)'
RPATH_STAGE = -Wl,-rpath,'$(CURDIR)/$(BUILD_DIR)/stage/usr/local/lib'
INSTALLED_SHARED_LINK = -L$(BUILD_DIR)/stage/usr/local/lib -linkleberry
INSTALLED_SHARED_PROGRAM = $(BUILD_DIR)/installed-shared$(EXE)
PROVIDER_FLAGS = -shared -Wl,-soname,libinkleberry_backend.so.1 -Wl,--version-script=src/infer.exports -Wl,-z,defs
BACKEND_DEFS := -DIFN_SHARED
CUDA_PIC_FLAGS = -Xcompiler=$(PIC_FLAGS),$(HIDDEN_FLAGS)
CLI_HOST = $(PIC)
PROVIDER_HOST = $(PIC)
ifn_archive = $(AR) rcs $(1) $(2)
# Common calls keep one dependency/source graph. Windows native drivers require
# link inputs after /link; pure object links use link.exe/lld-link directly.
ifn_compile = $(1) $(2) -c $(3) -o $(4)
ifn_program = $(1) $(2) $(3) $(4) -o $(5)
ifn_shared = $(call ifn_program,$(1),$(2),$(3),$(4),$(5))
ifneq ($(IFN_NATIVE_MSVC),)
ifeq ($(IFN_WINDOWS),0)
$(error Native MSVC/clang-cl needs OS=windows or a native Windows shell)
endif
IFN_TOOLCHAIN := $(if $(filter clang-cl clang-cl.exe,$(IFN_CC_NAME)),clang-cl,msvc)
ifeq ($(origin CXX),default)
CXX := $(CC)
endif
BUILD_DIR := .build/windows-$(IFN_TOOLCHAIN)-x64-release-cuda$(CUDA_ARCH)
HOST_STATIC = $(BUILD_DIR)/inkleberry.lib
INFER_STATIC = $(BUILD_DIR)/inkleberry-infer.lib
GPU_STATIC = $(BUILD_DIR)/inkleberry-gpu.lib
CUDA_STATIC = $(BUILD_DIR)/inkleberry-cuda.lib
CLI = $(BUILD_DIR)/inkleberry.exe
GENERATE = $(BUILD_DIR)/inkleberry-generate.exe
WEIGHTS_CLI = $(BUILD_DIR)/inkleberry-weights.exe
PROVIDER = $(BUILD_DIR)/inkleberry_backend-1.dll
MOCK_PROVIDER = $(BUILD_DIR)/infer-mock.dll
EXE := .exe
C_STD := /nologo /utf-8 /TC /std:c11
ifeq ($(IFN_TOOLCHAIN),msvc)
C_STD += /experimental:c11atomics
endif
export PYTHONUTF8 := 1
CFLAGS := /O2 /MD /GS
WARNFLAGS := $(C_STD) /W4 /WX
CPPFLAGS := -Iinclude -Isrc -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_WARNINGS
CXX_WARNFLAGS := /nologo /utf-8 /std:c++14 /W4 /WX /EHsc /MD
CXX20_WARNFLAGS := /nologo /utf-8 /std:c++20 /W4 /WX /EHsc /MD /O2
DEPFLAGS :=
# Conservative header prerequisites replace GCC .d files for native MSVC.
IFN_HEADER_DEPS := $(wildcard include/*.h src/*.h test/*.h src/cuda/*.h)
PIC_FLAGS :=
HIDDEN_FLAGS :=
THREAD_LIBS :=
CXX_LANGUAGE := /TP
SYNTAX_ONLY := /Zs
LDLIBS :=
LDFLAGS := /DYNAMICBASE /NXCOMPAT /WX
PROGRAM_COMPILE :=
PROGRAM_HARDEN :=
RPATH_ROOT :=
RPATH_STAGE :=
ifn_archive = lib.exe /nologo /OUT:$(1) $(2)
IFN_LINKER := $(if $(filter clang-cl,$(IFN_TOOLCHAIN)),lld-link.exe,"$(subst \,/,$(VCToolsInstallDir))bin/Hostx64/x64/link.exe")
HOST_IMPORT = $(BUILD_DIR)/inkleberry-1.lib
INSTALLED_DLLS := inkleberry-1.dll
BACKEND_IMPORT := inkleberry_backend-1.lib
SHARED = $(BUILD_DIR)/inkleberry-1.dll
SHARED_REAL = $(SHARED)
SHARED_ABI = $(SHARED)
SHARED_FLAGS := /DLL /IMPLIB:$(BUILD_DIR)/inkleberry-1.lib
PLAIN_SHARED_FLAGS := /DLL
MOCK_EXPORT_FLAGS := /DEF:src/infer.def
INSTALLED_SHARED_LINK = $(BUILD_DIR)/stage/usr/local/lib/inkleberry-1.lib
INSTALLED_SHARED_PROGRAM = $(BUILD_DIR)/stage/usr/local/bin/installed-shared.exe
PROVIDER_FLAGS = /DLL /DEF:src/infer.def /IMPLIB:$(BUILD_DIR)/inkleberry_backend-1.lib
BACKEND_DEFS :=
CLI_HOST = $(OBJ)
PROVIDER_HOST = $(OBJ)
CUDA_ROOT ?= $(subst \,/,$(CUDA_PATH))
NVCC = "$(CUDA_ROOT)/bin/nvcc.exe"
# CUDA 13.3 cuda.lib contains UUID helper objects requesting LIBCMT. All
# owned C/CUDA objects use /MD; resolve those helpers against the same CRT.
CUDA_LIBS = /LIBPATH:"$(CUDA_ROOT)/lib/x64" cudart.lib cublas.lib cuda.lib /NODEFAULTLIB:LIBCMT
CUDAFLAGS += -Xcompiler=/MD
CUDA_PIC_FLAGS :=
ifn_compile = $(1) $(2) -c $(3) /Fo$(4)
# Give combined compile/link commands their own intermediate directory, so
# parallel consumers of the same C source never race on a default .obj/PDB.
ifn_program = $(if $(filter %.c %.cpp,$(3)),mkdir -p $(5).objects && $(1) $(2) /Fo$(5).objects/ /Fe$(5) $(filter %.c %.cpp,$(3)) /link $(filter-out %.c %.cpp,$(3)) $(4),$(IFN_LINKER) /NOLOGO /OUT:$(5) $(3) $(4))
ifn_shared = $(IFN_LINKER) /NOLOGO /OUT:$(5) $(3) $(4)
endif
