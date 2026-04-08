plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.quickai.service"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.quickai.service"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
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
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.code.gson:gson:2.11.0")
    // NanoHTTPD for REST API server
    implementation("org.nanohttpd:nanohttpd:2.3.1")
    // OkHttp for model downloads
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    // WorkManager for background downloads
    implementation("androidx.work:work-runtime-ktx:2.10.0")
}
