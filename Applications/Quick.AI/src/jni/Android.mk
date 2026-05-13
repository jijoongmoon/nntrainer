LOCAL_PATH := $(call my-dir)

# ── Path configuration ──────────────────────────────────────────────────
ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../nntrainer
endif

CAUSALLM_ROOT := $(NNTRAINER_ROOT)/Applications/CausalLM

ML_API_COMMON_INCLUDES := $(NNTRAINER_ROOT)/ml_api_common/include
NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/builddir/android_build_result/include/nntrainer

# ── Prebuilt nntrainer libraries ─────────────────────────────────────────
include $(CLEAR_VARS)
LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/android_build_result/lib/$(TARGET_ARCH_ABI)/libnntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/android_build_result/lib/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

# ── Tokenizer library ───────────────────────────────────────────────────
include $(CLEAR_VARS)
LOCAL_MODULE := tokenizers_c
LOCAL_SRC_FILES := $(CAUSALLM_ROOT)/lib/libtokenizers_android_c.a
include $(PREBUILT_STATIC_LIBRARY)

# ── Common flags ─────────────────────────────────────────────────────────
COMMON_CFLAGS := -std=c++17 -Ofast -mcpu=cortex-a53 \
    -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 \
    -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 \
    -mtune=cortex-a76 -O3 -ffast-math \
    -pthread -fexceptions -fopenmp -static-openmp

COMMON_LDFLAGS := -fexceptions -fopenmp -static-openmp \
    -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 \
    -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 \
    -mtune=cortex-a76 -O3 -ffast-math

# ── Common include paths (auto-discover model subdirectories) ────────────
CAUSALLM_INCLUDES := $(NNTRAINER_INCLUDES) \
    $(CAUSALLM_ROOT) \
    $(CAUSALLM_ROOT)/layers \
    $(CAUSALLM_ROOT)/models \
    $(sort $(dir $(wildcard $(CAUSALLM_ROOT)/models/*/)))

# ── CausalLM source files (auto-discover via wildcard) ───────────────────
# Uses Make-native $(wildcard) instead of $(shell find) for correct
# path resolution in ndk-build. Automatically picks up new .cpp files
# when upstream adds them — no manual maintenance needed.
CAUSALLM_ALL_SRC := \
    $(wildcard $(CAUSALLM_ROOT)/*.cpp) \
    $(wildcard $(CAUSALLM_ROOT)/layers/*.cpp) \
    $(wildcard $(CAUSALLM_ROOT)/models/*.cpp) \
    $(wildcard $(CAUSALLM_ROOT)/models/*/*.cpp)
# Exclude main.cpp — we add it explicitly in the executable module
# Exclude quantize.cpp — built as separate executable
CAUSALLM_ALL_SRC := $(filter-out %/main.cpp %/quantize.cpp,$(CAUSALLM_ALL_SRC))

# ══════════════════════════════════════════════════════════════════════════
# Module: libcausallm.so  (CausalLM shared library for API use)
#
# All upstream CausalLM sources compiled into a shared library.
# Used by api to link against CausalLM model classes.
# ══════════════════════════════════════════════════════════════════════════
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += $(COMMON_CFLAGS)
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_LDFLAGS += $(COMMON_LDFLAGS)
LOCAL_MODULE := causallm

LOCAL_SRC_FILES := $(CAUSALLM_ALL_SRC)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c
LOCAL_LDLIBS := -llog -landroid

LOCAL_C_INCLUDES += $(CAUSALLM_INCLUDES)

include $(BUILD_SHARED_LIBRARY)

# ══════════════════════════════════════════════════════════════════════════
# Module 1: libquick_dot_ai.so  (self-registering plugin for LD_PRELOAD)
#
# Contains custom model definitions only.
# Auto-registers with Factory via __attribute__((constructor)).
# Undefined base class symbols are resolved at load time when injected
# into an executable that already contains them.
# ══════════════════════════════════════════════════════════════════════════
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += $(COMMON_CFLAGS)
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_LDFLAGS += $(COMMON_LDFLAGS)
LOCAL_ALLOW_UNDEFINED_SYMBOLS := true
LOCAL_MODULE := quick_dot_ai

LOCAL_SRC_FILES := \
	../models/qnn/android_memory_allocator.cpp \
	../models/qnn/quick_dot_ai_qnn.cpp \
	../models/qnn/graph_parser.cpp \
	../models/qnn/generate_qnn_utils.cpp \


LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_LDLIBS := -llog -landroid

LOCAL_C_INCLUDES := $(CAUSALLM_INCLUDES) \
    $(LOCAL_PATH)/../models/qnn \

include $(BUILD_SHARED_LIBRARY)

# ══════════════════════════════════════════════════════════════════════════
# Module 2: libquick_dot_ai_static.a  (for linking into executable)
#
# Same sources as the shared library, but statically linked via
# --whole-archive to ensure __attribute__((constructor)) is retained.
# ══════════════════════════════════════════════════════════════════════════
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += $(COMMON_CFLAGS)
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_MODULE := quick_dot_ai_static

LOCAL_SRC_FILES := \
	../models/qnn/android_memory_allocator.cpp \
	../models/qnn/quick_dot_ai_qnn.cpp \
	../models/qnn/graph_parser.cpp \
	../models/qnn/generate_qnn_utils.cpp \
    

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_LDLIBS := -llog -landroid

LOCAL_C_INCLUDES := $(CAUSALLM_INCLUDES) \
    $(LOCAL_PATH)/../models/qnn \

include $(BUILD_STATIC_LIBRARY)

# ══════════════════════════════════════════════════════════════════════════
# Module 3: quick_dot_ai_exe (standalone executable)
#
# Same structure as the original Applications/CausalLM/jni/Android.mk:
# all CausalLM sources compiled directly into the executable.
# Custom models are added via --whole-archive so the
# __attribute__((constructor)) registration always runs.
# ══════════════════════════════════════════════════════════════════════════
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += $(COMMON_CFLAGS)
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_LDFLAGS += $(COMMON_LDFLAGS)
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := quick_dot_ai_exe
LOCAL_LDLIBS := -llog -landroid -fopenmp -static-openmp

# main.cpp + all CausalLM sources (mirrors original Android.mk)
LOCAL_SRC_FILES := \
    $(CAUSALLM_ROOT)/main.cpp \
    $(CAUSALLM_ALL_SRC)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c
LOCAL_WHOLE_STATIC_LIBRARIES := quick_dot_ai_static


LOCAL_C_INCLUDES := $(CAUSALLM_INCLUDES) \
    $(LOCAL_PATH)/../models/qnn \

include $(BUILD_EXECUTABLE)

# ══════════════════════════════════════════════════════════════════════════
# Module 4: quantize_exe (standalone quantize tool)
#
# Separate executable for model quantization.
# ══════════════════════════════════════════════════════════════════════════
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += $(COMMON_CFLAGS)
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_LDFLAGS += $(COMMON_LDFLAGS)
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := quantize_exe
LOCAL_LDLIBS := -llog -landroid -fopenmp -static-openmp

# quantize.cpp + all CausalLM sources (for quantize tool)
LOCAL_SRC_FILES := \
    $(CAUSALLM_ROOT)/quantize.cpp \
    $(CAUSALLM_ALL_SRC)

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(CAUSALLM_INCLUDES)

include $(BUILD_EXECUTABLE)
