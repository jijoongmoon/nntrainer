plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.serialization)
}

// Copies the flat prebuilt .so files from Applications/QuickAI/prebuilt_libs/
// into an ABI-nested directory (build/generated/jniLibs/arm64-v8a/) so that
// Android Gradle's standard jniLibs machinery can package them into the APK.
// The source files are checked into git via Applications/QuickAI/prebuilt_libs/
// so that consumers do not have to rebuild libcausallm_api themselves; see
// Architecture.md §7 and Applications/CausalLM/build_api_lib.sh.
val prebuiltNativeLibsDir =
    layout.buildDirectory.dir("generated/jniLibs/arm64-v8a")

val copyPrebuiltNativeLibs = tasks.register<Copy>("copyPrebuiltNativeLibs") {
    from(rootProject.file("prebuilt_libs"))
    include("*.so")
    into(prebuiltNativeLibsDir)
}

android {
    namespace = "com.example.QuickAI"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.example.QuickAI"
        minSdk = 33
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            // Only arm64-v8a is supported by the prebuilt libcausallm_api.so
            // (see Applications/CausalLM/build_api_lib.sh). Add other ABIs
            // here when matching .so files become available.
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
            }
        }
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
                layout.buildDirectory.dir("generated/jniLibs")
            )
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

// The merge*JniLibFolders task is what reads android.sourceSets.main.jniLibs
// and stages the native libraries for packaging, so make it depend on the
// copy task. ExternalNativeBuild also benefits because the CMake link step
// reads libcausallm_api.so from the same prebuilt_libs folder directly.
tasks.matching {
    it.name.startsWith("merge") && it.name.endsWith("JniLibFolders")
}.configureEach {
    dependsOn(copyPrebuiltNativeLibs)
}
tasks.matching { it.name.startsWith("externalNativeBuild") }.configureEach {
    dependsOn(copyPrebuiltNativeLibs)
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.activity)
    implementation(libs.material)

    // REST server embedded inside QuickAIService.
    implementation(libs.nanohttpd)

    // JSON (wire format for the REST API).
    implementation(libs.kotlinx.serialization.json)

    // Coroutines for future async work (foreground service helpers).
    implementation(libs.kotlinx.coroutines.android)

    // LiteRT-LM for Gemma-family models (Architecture.md §4).
    // See how-to-use-litert-lm-guide.md at the repo root for the Kotlin API.
    implementation(libs.litertlm.android)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
