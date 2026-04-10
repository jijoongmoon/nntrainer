plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "com.example.QuickAI"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.QuickAI"
        minSdk = 33
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            // The QuickDotAI AAR dependency bundles arm64-v8a prebuilt
            // .so files; restrict the host APK to the matching ABI so we
            // don't generate empty ABI slices for armv7 / x86_64.
            abiFilters += listOf("arm64-v8a")
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
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // The QuickDotAI interface + both implementations (LiteRTLm and
    // NativeQuickDotAI) + libquickai_jni.so + the CausalLM prebuilt
    // shared libraries all come from this module. LauncherApp contains
    // only service / REST plumbing on top.
    implementation(project(":QuickDotAI"))

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

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
