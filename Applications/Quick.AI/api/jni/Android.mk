LOCAL_PATH := $(call my-dir)

# ── Path configuration ──────────────────────────────────────────────────
ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../nntrainer
endif

CAUSALLM_ROOT := $(NNTRAINER_ROOT)/Applications/CausalLM
QUICK_DOT_AI_ROOT := $(LOCAL_PATH)/../../src

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

# ── Prebuilt causallm library (from src Android build) ──────────
include $(CLEAR_VARS)
LOCAL_MODULE := causallm
LOCAL_SRC_FILES := $(QUICK_DOT_AI_ROOT)/jni/libs/$(TARGET_ARCH_ABI)/libcausallm.so
include $(PREBUILT_SHARED_LIBRARY)

# ── Prebuilt quick_dot_ai library (from src Android build) ──────
include $(CLEAR_VARS)
LOCAL_MODULE := quick_dot_ai
LOCAL_SRC_FILES := $(QUICK_DOT_AI_ROOT)/jni/libs/$(TARGET_ARCH_ABI)/libquick_dot_ai.so
include $(PREBUILT_SHARED_LIBRARY)

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

# ── CausalLM includes ───────────────────────────────────────────────────
CAUSALLM_INCLUDES := $(NNTRAINER_INCLUDES) \
    $(CAUSALLM_ROOT) \
    $(CAUSALLM_ROOT)/layers \
    $(CAUSALLM_ROOT)/models \
    $(sort $(dir $(wildcard $(CAUSALLM_ROOT)/models/*/)))

# ══════════════════════════════════════════════════════════════════════════
# Module: libquick_dot_ai_api.so  (C API library)
# ══════════════════════════════════════════════════════════════════════════
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += $(COMMON_CFLAGS)
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_LDFLAGS += $(COMMON_LDFLAGS)
LOCAL_MODULE := quick_dot_ai_api

LOCAL_SRC_FILES := \
    ../quick_dot_ai_api.cpp \
    ../model_config.cpp \
../streamer.cpp \
../callback_streamer.cpp

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer causallm quick_dot_ai
LOCAL_LDLIBS := -llog -landroid -ldl

LOCAL_C_INCLUDES += $(CAUSALLM_INCLUDES) \
    $(LOCAL_PATH)/.. \
    $(QUICK_DOT_AI_ROOT)/models/qnn

include $(BUILD_SHARED_LIBRARY)
