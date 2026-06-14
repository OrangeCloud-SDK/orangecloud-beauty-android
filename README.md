# OrangeCloud Beauty SDK (Android)

实时美颜 SDK（Android / OpenGL ES）：磨皮、美白、美型、美妆、LUT 滤镜、哈哈镜、AR 贴纸。

> 本仓库以**二进制**形式分发（`.aar`），不含核心算法源码。
> 运行需有效 License（`BeautyAppId + SecretKey + AuthToken`）。

## 获取 AAR

从 [Releases](https://github.com/OrangeCloud-SDK/orangecloud-beauty-android/releases) 下载 `orangecloud-beauty-android-release.aar`（见 v1.0.1）。

## 集成

把 aar 放入 `app/libs/`，在 `app/build.gradle`：

```groovy
android {
    defaultConfig { minSdk 24 }   // 引擎要求 minSdk 24
}
dependencies {
    implementation files('libs/orangecloud-beauty-android-release.aar')
    // AAR 不携带传递依赖，需自行声明引擎运行时依赖：
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
}
```

权限：`<uses-permission android:name="android.permission.CAMERA" />`

## 使用

```kotlin
val pipeline = BeautyPipeline(context)
pipeline.initialize(sharedEglContext)          // 在 GL 线程
val outTexId = pipeline.processFrame(texId, beautyParams)
```

完整 API 见 `BeautyPipeline`。Flutter 集成见 `orangecloud-beauty-flutter`（已内置本 aar）。

## 人脸模型

美型 / 美妆 / 贴纸需要 `face_landmark_98.tflite`（二进制，单独交付，放入 assets）。缺模型时自动降级：磨皮 / 美白 / LUT / 哈哈镜仍正常。
