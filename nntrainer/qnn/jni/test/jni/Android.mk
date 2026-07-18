# Device-side (arm64) build for unittest_qnn_graph.
# Links the prebuilt libnntrainer.so + libccapi-nntrainer.so (model API). The
# QNN backend (libqnn_context.so) is dlopen'd at runtime by engine=npu, so it is
# NOT linked here — only deployed alongside on the device.
#
# Build:
#   ANDROID_NDK=<ndk> $ANDROID_NDK/ndk-build \
#     NDK_PROJECT_PATH=nntrainer/qnn/jni/test/jni \
#     APP_BUILD_SCRIPT=nntrainer/qnn/jni/test/jni/Android.mk \
#     NDK_APPLICATION_MK=nntrainer/qnn/jni/test/jni/Application.mk \
#     NDK_LIBS_OUT=nntrainer/qnn/jni/test/jni/libs \
#     NDK_OUT=nntrainer/qnn/jni/test/jni/obj
LOCAL_PATH := $(call my-dir)
ROOT := /home/nntrainer/nntrainer

include $(CLEAR_VARS)
LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(ROOT)/builddir/libs/arm64-v8a/libnntrainer.so
LOCAL_EXPORT_C_INCLUDES := $(ROOT)/nntrainer $(ROOT)/api
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(ROOT)/builddir/libs/arm64-v8a/libccapi-nntrainer.so
LOCAL_EXPORT_C_INCLUDES := $(ROOT)/api/ccapi/include $(ROOT)/api
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := unittest_qnn_graph
LOCAL_SRC_FILES := ../unittest_qnn_graph.cpp
LOCAL_C_INCLUDES := $(ROOT)/api/ccapi/include $(ROOT)/api $(ROOT)/nntrainer
LOCAL_CPPFLAGS := -std=c++17 -frtti -fexceptions
LOCAL_SHARED_LIBRARIES := ccapi-nntrainer nntrainer
include $(BUILD_EXECUTABLE)
