LOCAL_PATH := $(call my-dir)

# ── Path configuration ──────────────────────────────────────────────────
ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../nntrainer
endif

API_ROOT := $(LOCAL_PATH)/../../api
INCLUDE_ROOT := $(LOCAL_PATH)/../include

# ── Prebuilt nntrainer libraries (needed at runtime) ─────────────────────
include $(CLEAR_VARS)
LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/android_build_result/lib/$(TARGET_ARCH_ABI)/libnntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/android_build_result/lib/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

# ── Prebuilt causallm library (needed at runtime) ────────────────────────
include $(CLEAR_VARS)
LOCAL_MODULE := causallm
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../src/jni/libs/$(TARGET_ARCH_ABI)/libcausallm.so
include $(PREBUILT_SHARED_LIBRARY)

# ── Prebuilt API library ────────────────────────────────────────────────
include $(CLEAR_VARS)
LOCAL_MODULE := quick_dot_ai_api
LOCAL_SRC_FILES := $(API_ROOT)/jni/libs/$(TARGET_ARCH_ABI)/libquick_dot_ai_api.so
include $(PREBUILT_SHARED_LIBRARY)

# ══════════════════════════════════════════════════════════════════════════
# Module: quick_dot_ai_test  (test executable)
#
# Only includes quick_dot_ai_api.h, links only libquick_dot_ai_api.so.
# Runtime dependencies (nntrainer, causallm) are resolved transitively.
# ══════════════════════════════════════════════════════════════════════════
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_MODULE := quick_dot_ai_test
LOCAL_LDLIBS := -llog -landroid

LOCAL_SRC_FILES := ../test_api.cpp

# API headers (copied to include/ by build script)
LOCAL_C_INCLUDES += $(INCLUDE_ROOT)

# Link against the API library; runtime deps are pulled transitively
LOCAL_SHARED_LIBRARIES := quick_dot_ai_api nntrainer ccapi-nntrainer causallm

include $(BUILD_EXECUTABLE)
