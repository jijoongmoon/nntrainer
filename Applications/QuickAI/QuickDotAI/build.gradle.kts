// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
//
// QuickDotAI — reusable AAR that bundles the QuickDotAI interface and
// both concrete implementations (LiteRTLm + NativeQuickDotAI) plus the
// JNI shim (libquickai_jni.so) and the CausalLM prebuilt shared
// libraries. Third-party apps can depend on this AAR to run on-device
// LLMs without linking QuickAIService or any of LauncherApp's REST
// plumbing.

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.serialization)
}

// Copies the flat prebuilt .so files from QuickDotAI/prebuilt_libs/ into an
// ABI-nested directory (build/generated/jniLibs/arm64-v8a/) so that Android
// Gradle's standard jniLibs machinery can bundle them into the AAR.
val prebuiltNativeLibsDir =
    layout.buildDirectory.dir("generated/jniLibs/arm64-v8a")

val copyPrebuiltNativeLibs = tasks.register<Copy>("copyPrebuiltNativeLibs") {
    from(project.file("prebuilt_libs"))
    include("*.so")
    into(prebuiltNativeLibsDir)
}

android {
    namespace = "com.example.quickdotai"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        minSdk = 33

        ndk {
            // Only arm64-v8a is supported by the prebuilt libcausallm_api.so.
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
            }
        }

        consumerProguardFiles("consumer-rules.pro")
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    sourceSets {
        getByName("main") {
            // Pick up the generated/jniLibs/<abi>/*.so tree produced by
            // copyPrebuiltNativeLibs above, alongside any hand-placed files
            // in src/main/jniLibs/.
            jniLibs.srcDirs(
                "src/main/jniLibs",
                file("${buildDir}/generated/jniLibs")
            )
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "consumer-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
}

// The merge*JniLibFolders task reads android.sourceSets.main.jniLibs and
// stages the native libraries for packaging into the AAR, so make it
// depend on the copy task. ExternalNativeBuild also benefits because the
// CMake link step reads libcausallm_api.so directly from prebuilt_libs.
tasks.matching {
    it.name.startsWith("merge") && it.name.endsWith("JniLibFolders")
}.configureEach {
    dependsOn(copyPrebuiltNativeLibs)
}
tasks.matching { it.name.startsWith("externalNativeBuild") }.configureEach {
    dependsOn(copyPrebuiltNativeLibs)
}

dependencies {
    // kotlinx.serialization is exposed as an `api` dependency because the
    // public types (ModelId, BackendType, LoadModelRequest, …) carry
    // @Serializable annotations so consumers that want to JSON-ify them
    // can do so without pulling the runtime in themselves.
    api(libs.kotlinx.serialization.json)

    // LiteRT-LM is the engine used by LiteRTLm.kt for Gemma-family models.
    // Exposed as `api` so consumers don't have to redeclare it.
    api("com.google.ai.edge.litertlm:litertlm-android:latest.release")
}
