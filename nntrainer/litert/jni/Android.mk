LOCAL_PATH := $(call my-dir)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../..
endif

ifndef LITERT_LM_ROOT
$(error LITERT_LM_ROOT is not defined! Set to LiteRT-LM source root.)
endif

# Abseil headers are from Bazel's external: bazel-LiteRT-LM/external/com_google_absl

# Bazel 생성 파일 경로 (proto .pb.h 등)
ifndef LITERT_LM_BAZEL_BIN
LITERT_LM_BAZEL_BIN := $(LITERT_LM_ROOT)/bazel-bin
endif

ML_API_COMMON_INCLUDES := $(NNTRAINER_ROOT)/ml_api_common/include
NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
	$(NNTRAINER_ROOT)/nntrainer/dataset \
	$(NNTRAINER_ROOT)/nntrainer/models \
	$(NNTRAINER_ROOT)/nntrainer/layers \
	$(NNTRAINER_ROOT)/nntrainer/compiler \
	$(NNTRAINER_ROOT)/nntrainer/graph \
	$(NNTRAINER_ROOT)/nntrainer/optimizers \
	$(NNTRAINER_ROOT)/nntrainer/tensor \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/fallback \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm \
	$(NNTRAINER_ROOT)/nntrainer/utils \
	$(NNTRAINER_ROOT)/api \
	$(NNTRAINER_ROOT)/api/ccapi/include \
	$(ML_API_COMMON_INCLUDES)

####################################################################
# Prebuilt: nntrainer
####################################################################
include $(CLEAR_VARS)
LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/android_build_result/lib/$(TARGET_ARCH_ABI)/libnntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

####################################################################
# Prebuilt: ccapi-nntrainer
####################################################################
include $(CLEAR_VARS)
LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/android_build_result/lib/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

####################################################################
# Prebuilt: liblitert_lm_all.so (Bazel cc_binary linkshared)
# 모든 deps가 정적 링크된 fat shared library
####################################################################
include $(CLEAR_VARS)
LOCAL_MODULE := litert_lm_all

ifndef LITERT_LM_LIB_PATH
LITERT_LM_LIB_PATH := $(LITERT_LM_BAZEL_BIN)/runtime/engine
endif

LOCAL_SRC_FILES := $(LITERT_LM_LIB_PATH)/liblitert_lm_all.so
include $(PREBUILT_SHARED_LIBRARY)

####################################################################
# Prebuilt static: Abseil (모든 .a를 합친 fat archive)
# 생성: find bazel-bin/external/com_google_absl -name "*.a" -exec ar x {} \;
#       ar rcs libabsl_all.a *.o
####################################################################
include $(CLEAR_VARS)
LOCAL_MODULE := absl_all
LOCAL_SRC_FILES := $(LITERT_LM_LIB_PATH)/libabsl_all.a
include $(PREBUILT_STATIC_LIBRARY)

####################################################################
# liblitert_context.so - LiteRT-LM context plugin for nntrainer
####################################################################
include $(CLEAR_VARS)

LOCAL_MODULE        := litert_context
LOCAL_MODULE_TAGS   := optional
LOCAL_ARM_NEON      := true
LOCAL_ARM_MODE      := arm

LOCAL_SRC_FILES     := \
	$(NNTRAINER_ROOT)/nntrainer/litert/litert_context.cpp \
	$(NNTRAINER_ROOT)/nntrainer/litert/litert_graph.cpp

LOCAL_C_INCLUDES    := \
	$(NNTRAINER_INCLUDES) \
	$(NNTRAINER_ROOT)/nntrainer/litert \
	$(LITERT_LM_ROOT) \
	$(LITERT_LM_BAZEL_BIN) \
	$(LITERT_LM_ROOT)/../LiteRT \
	$(LITERT_LM_ROOT)/LiteRT \
	$(LITERT_LM_ROOT)/bazel-LiteRT-LM/external/com_google_absl \
	$(LITERT_LM_ROOT)/bazel-LiteRT-LM/external/com_google_protobuf/src

LOCAL_CFLAGS        += -pthread -fexceptions -Wno-deprecated-declarations
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions -DPLUGGABLE -DENABLE_LITERT_LM
LOCAL_LDLIBS        := -llog -landroid
LOCAL_LDFLAGS       += "-Wl,-z,max-page-size=16384" "-Wl,--allow-multiple-definition"

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer litert_lm_all
LOCAL_STATIC_LIBRARIES := absl_all

include $(BUILD_SHARED_LIBRARY)
