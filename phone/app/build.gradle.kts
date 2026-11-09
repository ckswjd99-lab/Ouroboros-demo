plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.ouroboros.demo"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.ouroboros.demo"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += listOf("arm64-v8a") }
    }
    buildTypes { release { isMinifyEnabled = false } }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    packaging { jniLibs { useLegacyPackaging = true } }
}

val cameraxVersion = "1.3.4"

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    // On-device mode: the server's own ViTDet-B, exported to ONNX (tools/export_vitdet_onnx.py).
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.1")
    // Mobile-side mirror of the server's registration + dirtiness map (same 5.0.0 as the server's cv2).
    implementation("org.opencv:opencv:5.0.0")
}
