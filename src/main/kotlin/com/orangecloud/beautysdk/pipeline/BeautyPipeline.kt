package com.orangecloud.beautysdk.pipeline

import android.content.Context
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES30
import android.util.Log
import com.orangecloud.beautysdk.detector.FaceDetector
import com.orangecloud.beautysdk.filter.BeautyFilter
import com.orangecloud.beautysdk.filter.LutFilter
import com.orangecloud.beautysdk.filter.DistortionFilter
import com.orangecloud.beautysdk.filter.AdvancedBeautyFilter
import com.orangecloud.beautysdk.filter.MakeupFilter
import com.orangecloud.beautysdk.deformer.FaceDeformer
import com.orangecloud.beautysdk.perf.PerfMonitor
import com.orangecloud.beautysdk.models.BeautyParams
import com.orangecloud.beautysdk.models.FaceLandmarks
import com.orangecloud.beautysdk.sticker.StickerEngine

/**
 * OpenGL ES 渲染管线管理器
 *
 * 管理 EGL context、OpenGL ES 环境生命周期，串联完整美颜处理管线：
 * FaceDetector → BeautyFilter → FaceDeformer → StickerEngine
 *
 * 功能：
 * - 初始化 EGL context 和 OpenGL ES 3.0 环境
 * - processFrame 串联完整渲染管线，端到端延迟 ≤33ms
 * - 后台暂停：释放非必要 GPU 资源（≤500ms）
 * - 前台恢复：重新创建 GPU 资源（≤300ms）
 * - dispose() 释放所有 GPU 资源
 *
 * 需求: 5.7, 8.1, 8.6, 8.7
 */
class BeautyPipeline(private val context: Context) {

    companion object {
        private const val TAG = "BeautyPipeline"

        /** EGL 配置属性 */
        private val EGL_CONFIG_ATTRIBUTES = intArrayOf(
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT or 0x0040, // EGL_OPENGL_ES3_BIT
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_NONE
        )

        /** EGL Context 属性（OpenGL ES 3.0） */
        private val EGL_CONTEXT_ATTRIBUTES = intArrayOf(
            EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
            EGL14.EGL_NONE
        )

        /** PBuffer Surface 属性 */
        private val PBUFFER_ATTRIBUTES = intArrayOf(
            EGL14.EGL_WIDTH, 1,
            EGL14.EGL_HEIGHT, 1,
            EGL14.EGL_NONE
        )
    }

    // ==================== EGL 状态 ====================

    /** EGL Display */
    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY

    /** EGL Context */
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT

    /** EGL Surface（PBuffer，用于离屏渲染） */
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE

    /** EGL Config */
    private var eglConfig: EGLConfig? = null

    // ==================== 子模块 ====================

    /** 人脸检测器 */
    private var faceDetector: FaceDetector? = null

    /** 美颜滤镜 */
    private var beautyFilter: BeautyFilter? = null

    /** 高级美颜滤镜 */
    private var advancedBeautyFilter: AdvancedBeautyFilter? = null

    /** 美妆滤镜 */
    private var makeupFilter: MakeupFilter? = null

    /** 性能监控器（对外暴露供 Plugin 层订阅） */
    val perfMonitor: PerfMonitor = PerfMonitor()

    /** LUT 颜色分级滤镜 */
    private var lutFilter: LutFilter? = null

    /** 哈哈镜滤镜 */
    private var distortionFilter: DistortionFilter? = null

    /** 人脸变形器 */
    private var faceDeformer: FaceDeformer? = null

    /** 贴纸引擎 */
    private var stickerEngine: StickerEngine? = null

    // ==================== 管线状态 ====================

    /** 管线是否已初始化 */
    @Volatile
    private var isInitialized: Boolean = false

    /** 管线是否已释放 */
    @Volatile
    private var isDisposed: Boolean = false

    /** 管线是否已暂停（后台状态） */
    @Volatile
    private var isPaused: Boolean = false

    /** 共享 EGL Context（用于与外部 GL 线程共享纹理） */
    private var sharedEglContext: EGLContext = EGL14.EGL_NO_CONTEXT

    // ==================== 公开方法 ====================

    /**
     * 初始化 EGL + OpenGL ES 环境及子模块
     *
     * @param sharedContext 可选的共享 EGL Context（用于与外部 GL 线程共享纹理）
     * @return 是否初始化成功
     */
    fun initialize(sharedContext: EGLContext = EGL14.EGL_NO_CONTEXT): Boolean {
        if (isInitialized) return true
        if (isDisposed) {
            Log.e(TAG, "Cannot initialize: pipeline is disposed")
            return false
        }

        sharedEglContext = sharedContext

        return try {
            // 1. 初始化 EGL 环境
            if (!initializeEGL()) {
                Log.e(TAG, "EGL initialization failed")
                return false
            }

            // 2. 使当前线程绑定 EGL Context
            if (!makeCurrent()) {
                Log.e(TAG, "Failed to make EGL context current")
                releaseEGL()
                return false
            }

            // 3. 初始化子模块
            initializeSubmodules()

            isInitialized = true
            Log.i(TAG, "BeautyPipeline initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "BeautyPipeline initialization failed: ${e.message}", e)
            releaseEGL()
            false
        }
    }

    /**
     * 处理单帧：FaceDetector → BeautyFilter → FaceDeformer → StickerEngine
     *
     * @param textureId 输入纹理 ID（OpenGL ES 纹理）
     * @param params 美颜参数
     * @return 处理后的输出纹理 ID，失败或已暂停时返回输入纹理 ID
     */
    fun processFrame(textureId: Int, params: BeautyParams): Int {
        if (isDisposed) {
            Log.e(TAG, "Cannot process frame: pipeline is disposed")
            return textureId
        }

        if (isPaused) {
            Log.d(TAG, "Pipeline is paused, skipping frame")
            return textureId
        }

        if (!isInitialized) {
            Log.w(TAG, "Pipeline not initialized, returning input texture")
            return textureId
        }

        // 确保 EGL Context 绑定到当前线程
        if (!makeCurrent()) {
            Log.e(TAG, "Failed to make EGL context current for frame processing")
            return textureId
        }

        var currentTexture = textureId

        perfMonitor.beginFrame()

        val landmarks = cachedLandmarks

        // Step 1: FaceDetector（Android 采用 Camera 侧先检测的缓存结果，此 stage 仅用于计时占位）
        perfMonitor.beginStage("faceDetector")
        perfMonitor.endStage("faceDetector")

        // Step 2: BeautyFilter → 磨皮 + 美白
        perfMonitor.beginStage("beautyFilter")
        val filter = beautyFilter
        if (filter != null && perfMonitor.isStageEnabled("beautyFilter")) {
            if (landmarks.isNotEmpty()) {
                currentTexture = filter.apply(
                    currentTexture,
                    landmarks.first(),
                    params.smoothingIntensity,
                    params.whiteningIntensity
                )
            } else if (params.smoothingIntensity > 0f || params.whiteningIntensity > 0f) {
                currentTexture = filter.apply(
                    currentTexture, null,
                    params.smoothingIntensity, params.whiteningIntensity
                )
            }
        }
        perfMonitor.endStage("beautyFilter")

        // Step 3: FaceDeformer → 全量美型
        perfMonitor.beginStage("faceDeformer")
        val deformer = faceDeformer
        if (deformer != null && perfMonitor.isStageEnabled("faceDeformer") && landmarks.isNotEmpty()) {
            currentTexture = deformer.deform(currentTexture, landmarks, params)
        }
        perfMonitor.endStage("faceDeformer")

        // Step 3.1: 高级美颜
        perfMonitor.beginStage("advancedBeauty")
        val adv = advancedBeautyFilter
        if (adv != null && adv.isActive && perfMonitor.isStageEnabled("advancedBeauty") && landmarks.isNotEmpty()) {
            currentTexture = adv.apply(currentTexture, landmarks, inputImageWidth, inputImageHeight)
        }
        perfMonitor.endStage("advancedBeauty")

        // Step 3.5: LUT
        perfMonitor.beginStage("lutFilter")
        val lut = lutFilter
        if (lut != null && lut.isActive && perfMonitor.isStageEnabled("lutFilter")) {
            currentTexture = lut.apply(currentTexture)
        }
        perfMonitor.endStage("lutFilter")

        // Step 3.6: 哈哈镜
        perfMonitor.beginStage("distortionFilter")
        val dist = distortionFilter
        if (dist != null && dist.isActive && perfMonitor.isStageEnabled("distortionFilter")) {
            currentTexture = dist.apply(currentTexture)
        }
        perfMonitor.endStage("distortionFilter")

        // Step 3.7: 美妆
        perfMonitor.beginStage("makeupFilter")
        val mk = makeupFilter
        if (mk != null && mk.isActive && perfMonitor.isStageEnabled("makeupFilter") && landmarks.isNotEmpty()) {
            currentTexture = mk.apply(currentTexture, landmarks, inputImageWidth, inputImageHeight)
        }
        perfMonitor.endStage("makeupFilter")

        // Step 4: StickerEngine → AR 贴纸
        perfMonitor.beginStage("stickerEngine")
        val sticker = stickerEngine
        if (sticker != null && perfMonitor.isStageEnabled("stickerEngine") && landmarks.isNotEmpty()) {
            currentTexture = sticker.render(currentTexture, landmarks)
        }
        perfMonitor.endStage("stickerEngine")

        perfMonitor.endFrame()

        return currentTexture
    }

    /**
     * 更新人脸检测结果（由外部 Camera 回调线程调用）
     * 用于解耦人脸检测与渲染管线的线程模型
     *
     * @param landmarks 最新的人脸关键点检测结果
     */
    @Volatile
    private var cachedLandmarks: List<FaceLandmarks> = emptyList()

    /** 输入图像尺寸（用于 landmark 归一化；由 camera 回调写入） */
    @Volatile
    private var inputImageWidth: Int = 1280
    @Volatile
    private var inputImageHeight: Int = 720

    fun updateFaceDetectionResult(landmarks: List<FaceLandmarks>) {
        cachedLandmarks = landmarks
    }

    /** 更新输入图像尺寸（Camera 端在分辨率变化时调用） */
    fun updateInputImageSize(width: Int, height: Int) {
        if (width > 0) inputImageWidth = width
        if (height > 0) inputImageHeight = height
    }

    /**
     * 获取人脸检测器（供外部 Camera 回调使用）
     */
    fun getFaceDetector(): FaceDetector? = faceDetector

    /**
     * 获取贴纸引擎（供外部加载/移除贴纸）
     */
    fun getStickerEngine(): StickerEngine? = stickerEngine

    /** 获取 LUT 滤镜（用于加载/清除 LUT） */
    fun getLutFilter(): LutFilter? = lutFilter

    /** 获取哈哈镜滤镜 */
    fun getDistortionFilter(): DistortionFilter? = distortionFilter

    /** 获取高级美颜滤镜 */
    fun getAdvancedBeautyFilter(): AdvancedBeautyFilter? = advancedBeautyFilter

    /** 获取美妆滤镜 */
    fun getMakeupFilter(): MakeupFilter? = makeupFilter

    /**
     * 暂停 GPU 操作（后台状态）
     * 释放非必要 GPU 资源，目标 ≤500ms 完成
     */
    fun pause() {
        if (isDisposed || isPaused) return

        val startTime = System.nanoTime()
        isPaused = true

        if (makeCurrent()) {
            // 释放 BeautyFilter 的 FBO 资源
            beautyFilter?.release()

            // 释放 LUT FBO 资源（保留 LUT 纹理）
            lutFilter?.release()

            // 释放哈哈镜资源
            distortionFilter?.release()

            // 释放高级美颜资源
            advancedBeautyFilter?.release()

            // 释放美妆资源
            makeupFilter?.release()

            // 释放 FaceDeformer 的 FBO 和平滑状态
            faceDeformer?.release()

            // 清除缓存的检测结果
            cachedLandmarks = emptyList()

            // 刷新 GL 命令
            GLES30.glFlush()
        }

        val elapsed = (System.nanoTime() - startTime) / 1_000_000.0
        Log.i(TAG, "BeautyPipeline paused in ${String.format("%.1f", elapsed)}ms")
    }

    /**
     * 恢复 GPU 操作（前台状态）
     * 重新创建 GPU 资源，目标 ≤300ms 完成
     */
    fun resume() {
        if (isDisposed || !isPaused) return

        val startTime = System.nanoTime()

        if (makeCurrent()) {
            // 重新初始化 BeautyFilter
            beautyFilter = BeautyFilter().also { it.initialize() }

            // 重新初始化 LUT（用户需要重新 load 才会生效）
            lutFilter = LutFilter().also { it.initialize() }

            // 重新初始化哈哈镜（用户需重新 set 才会生效）
            distortionFilter = DistortionFilter().also { it.initialize() }

            // 重新初始化高级美颜
            advancedBeautyFilter = AdvancedBeautyFilter().also { it.initialize() }

            // 重新初始化美妆
            makeupFilter = MakeupFilter().also { it.initialize() }

            // 重新初始化 FaceDeformer
            faceDeformer = FaceDeformer().also { it.initialize() }
        }

        isPaused = false

        val elapsed = (System.nanoTime() - startTime) / 1_000_000.0
        Log.i(TAG, "BeautyPipeline resumed in ${String.format("%.1f", elapsed)}ms")
    }

    /**
     * 释放所有 GPU 资源
     */
    fun dispose() {
        if (isDisposed) return
        isDisposed = true

        Log.i(TAG, "Disposing BeautyPipeline...")

        if (makeCurrent()) {
            // 释放子模块
            faceDetector?.release()
            faceDetector = null

            beautyFilter?.release()
            beautyFilter = null

            advancedBeautyFilter?.release()
            advancedBeautyFilter = null

            makeupFilter?.release()
            makeupFilter = null

            lutFilter?.release()
            lutFilter = null

            distortionFilter?.release()
            distortionFilter = null

            faceDeformer?.release()
            faceDeformer = null

            stickerEngine?.dispose()
            stickerEngine = null
        }

        // 释放 EGL 资源
        releaseEGL()

        cachedLandmarks = emptyList()
        isInitialized = false

        Log.i(TAG, "BeautyPipeline disposed")
    }

    // ==================== EGL 管理 ====================

    /**
     * 初始化 EGL 环境
     * @return 是否成功
     */
    private fun initializeEGL(): Boolean {
        // 获取 EGL Display
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            Log.e(TAG, "eglGetDisplay failed: ${EGL14.eglGetError()}")
            return false
        }

        // 初始化 EGL
        val version = IntArray(2)
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            Log.e(TAG, "eglInitialize failed: ${EGL14.eglGetError()}")
            eglDisplay = EGL14.EGL_NO_DISPLAY
            return false
        }
        Log.d(TAG, "EGL initialized: version ${version[0]}.${version[1]}")

        // 选择 EGL Config
        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        if (!EGL14.eglChooseConfig(
                eglDisplay, EGL_CONFIG_ATTRIBUTES, 0,
                configs, 0, 1, numConfigs, 0
            ) || numConfigs[0] == 0
        ) {
            Log.e(TAG, "eglChooseConfig failed: ${EGL14.eglGetError()}")
            return false
        }
        eglConfig = configs[0]

        // 创建 EGL Context（可选共享 Context）
        eglContext = EGL14.eglCreateContext(
            eglDisplay,
            eglConfig,
            sharedEglContext,
            EGL_CONTEXT_ATTRIBUTES, 0
        )
        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            Log.e(TAG, "eglCreateContext failed: ${EGL14.eglGetError()}")
            return false
        }

        // 创建 PBuffer Surface（离屏渲染用）
        eglSurface = EGL14.eglCreatePbufferSurface(
            eglDisplay,
            eglConfig,
            PBUFFER_ATTRIBUTES, 0
        )
        if (eglSurface == EGL14.EGL_NO_SURFACE) {
            Log.e(TAG, "eglCreatePbufferSurface failed: ${EGL14.eglGetError()}")
            return false
        }

        return true
    }

    /**
     * 使当前线程绑定 EGL Context
     * @return 是否成功
     */
    private fun makeCurrent(): Boolean {
        if (eglDisplay == EGL14.EGL_NO_DISPLAY || eglContext == EGL14.EGL_NO_CONTEXT) {
            return false
        }

        // 检查当前线程是否已绑定此 Context
        if (EGL14.eglGetCurrentContext() == eglContext) {
            return true
        }

        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            Log.e(TAG, "eglMakeCurrent failed: ${EGL14.eglGetError()}")
            return false
        }
        return true
    }

    /**
     * 释放 EGL 资源
     */
    private fun releaseEGL() {
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            // 解绑当前 Context
            EGL14.eglMakeCurrent(
                eglDisplay,
                EGL14.EGL_NO_SURFACE,
                EGL14.EGL_NO_SURFACE,
                EGL14.EGL_NO_CONTEXT
            )

            // 销毁 Surface
            if (eglSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, eglSurface)
                eglSurface = EGL14.EGL_NO_SURFACE
            }

            // 销毁 Context
            if (eglContext != EGL14.EGL_NO_CONTEXT) {
                EGL14.eglDestroyContext(eglDisplay, eglContext)
                eglContext = EGL14.EGL_NO_CONTEXT
            }

            // 终止 Display
            EGL14.eglTerminate(eglDisplay)
            eglDisplay = EGL14.EGL_NO_DISPLAY
        }

        eglConfig = null
        Log.d(TAG, "EGL resources released")
    }

    // ==================== 子模块初始化 ====================

    /**
     * 初始化子模块（FaceDetector、BeautyFilter、FaceDeformer、StickerEngine）
     */
    private fun initializeSubmodules() {
        // 初始化 FaceDetector
        try {
            faceDetector = FaceDetector(context).also { detector ->
                if (!detector.loadModel()) {
                    Log.w(TAG, "FaceDetector model load failed, detector will be unavailable")
                    faceDetector = null
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "FaceDetector initialization failed: ${e.message}")
            faceDetector = null
        }

        // 初始化 BeautyFilter
        try {
            beautyFilter = BeautyFilter().also { filter ->
                if (!filter.initialize()) {
                    Log.w(TAG, "BeautyFilter initialization failed")
                    beautyFilter = null
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "BeautyFilter initialization failed: ${e.message}")
            beautyFilter = null
        }

        // 初始化 LutFilter（可选，若初始化失败只是不能用 LUT）
        try {
            lutFilter = LutFilter().also { filter ->
                if (!filter.initialize()) {
                    Log.w(TAG, "LutFilter initialization failed")
                    lutFilter = null
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "LutFilter initialization failed: ${e.message}")
            lutFilter = null
        }

        // 初始化 DistortionFilter
        try {
            distortionFilter = DistortionFilter().also { filter ->
                if (!filter.initialize()) {
                    Log.w(TAG, "DistortionFilter initialization failed")
                    distortionFilter = null
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "DistortionFilter initialization failed: ${e.message}")
            distortionFilter = null
        }

        // 初始化 AdvancedBeautyFilter
        try {
            advancedBeautyFilter = AdvancedBeautyFilter().also { filter ->
                if (!filter.initialize()) {
                    Log.w(TAG, "AdvancedBeautyFilter initialization failed")
                    advancedBeautyFilter = null
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "AdvancedBeautyFilter initialization failed: ${e.message}")
            advancedBeautyFilter = null
        }

        // 初始化 MakeupFilter
        try {
            makeupFilter = MakeupFilter().also { filter ->
                if (!filter.initialize()) {
                    Log.w(TAG, "MakeupFilter initialization failed")
                    makeupFilter = null
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "MakeupFilter initialization failed: ${e.message}")
            makeupFilter = null
        }

        // 初始化 FaceDeformer
        try {
            faceDeformer = FaceDeformer().also { deformer ->
                if (!deformer.initialize()) {
                    Log.w(TAG, "FaceDeformer initialization failed")
                    faceDeformer = null
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "FaceDeformer initialization failed: ${e.message}")
            faceDeformer = null
        }

        // 初始化 StickerEngine
        try {
            stickerEngine = StickerEngine(context).also { sticker ->
                if (!sticker.initialize()) {
                    Log.w(TAG, "StickerEngine initialization failed")
                    stickerEngine = null
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "StickerEngine initialization failed: ${e.message}")
            stickerEngine = null
        }
    }
}
