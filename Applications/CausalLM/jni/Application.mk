APP_ABI := arm64-v8a
APP_PLATFORM := android-29
APP_STL := c++_shared
APP_CPPFLAGS += -fexceptions -frtti -std=c++17
APP_CFLAGS += -Wno-nan-infinity-disabled -Wno-comment -Wno-deprecated-literal-operator
APP_CPPFLAGS += -Wno-nan-infinity-disabled -Wno-comment -Wno-deprecated-literal-operator
APP_LDFLAGS += -fopenmp
NDK_TOOLCHAIN_VERSION := clang
