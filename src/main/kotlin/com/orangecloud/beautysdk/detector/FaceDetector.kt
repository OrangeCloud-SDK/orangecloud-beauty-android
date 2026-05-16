package com.orangecloud.beautysdk.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import com.orangecloud.beautysdk.models.FaceLandmarks
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * 人脸关键点检测器 - 基于 TensorFlow Lite + MobileNet v2 架构
 *
 * 功能：
 * - 加载 98 点关键点模型（TFLite 格式，≤5MB）
 * - GPU Delegate 加速推理，不可用时回退 CPU
 * - 单帧推理 ≤15ms
 * - 支持最多 5 张脸同时检测
 * - 无人脸时 ≤10ms 返回空列表
 * - 支持部分遮挡（≤30%）的降级检测
 * - 低功耗模式下降低推理频率，维持 ≥15fps
 *
 * 需求: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
 */
class FaceDetector(private val context: Context) {

    companion object {
        private const val TAG = "FaceDetector"

        /** 模型输入尺寸 */
        private const val INPUT_SIZE = 192

        /** 每张脸的关键点数量 */
        private const val LANDMARK_COUNT = 98

        /** 最大检测人脸数 */
        private const val MAX_FACES = 5

        /** 最小置信度阈值 */
        private const val CONFIDENCE_THRESHOLD = 0.5f

        /** 遮挡降级检测的最低置信度 */
        private const val OCCLUSION_CONFIDENCE_THRESHOLD = 0.3f
    }

    /** TFLite 解释器 */
    private var interpreter: Interpreter? = null

    /** GPU Delegate（可能为 null，表示回退到 CPU） */
    private var gpuDelegate: GpuDelegate? = null

    /** 是否使用 GPU 加速 */
    private var isGpuEnabled: Boolean = false

    /** 模型是否已加载 */
    private var isModelLoaded: Boolean = false

    /** 低功耗模式标志 */
    @Volatile
    private var lowPowerMode: Boolean = false

    /** 帧计数器（用于低功耗模式跳帧） */
    private var frameCounter: Long = 0

    /** 低功耗模式下的跳帧间隔（每 N 帧推理一次，维持 ≥15fps） */
    private var skipFrameInterval: Int = 2

    /** 上一次检测结果缓存（低功耗模式跳帧时复用） */
    private var lastDetectionResult: List<FaceLandmarks> = emptyList()

    /** 输入缓冲区 */
    private var inputBuffer: ByteBuffer? = null

    /** Android 内置人脸检测器（用于人脸区域定位） */
    private val androidFaceDetector: android.media.FaceDetector by lazy {
        android.media.FaceDetector(INPUT_SIZE, INPUT_SIZE, MAX_FACES)
    }

    /**
     * 加载 TFLite 关键点模型
     *
     * @param modelPath assets 目录下的模型文件路径
     * @return 是否加载成功
     */
    fun loadModel(modelPath: String = "face_landmark_98.tflite"): Boolean {
        return try {
            val modelBuffer = loadModelFile(modelPath)
            val options = Interpreter.Options()

            // 尝试启用 GPU Delegate 加速
            isGpuEnabled = try {
                val delegateOptions = GpuDelegate.Options().apply {
                    setPrecisionLossAllowed(true) // 允许精度损失以提升速度
                }
                gpuDelegate = GpuDelegate(delegateOptions)
                options.addDelegate(gpuDelegate)
                Log.i(TAG, "GPU Delegate 已启用")
                true
            } catch (e: Exception) {
                Log.w(TAG, "GPU Delegate 不可用，回退到 CPU: ${e.message}")
                gpuDelegate = null
                // CPU 模式下使用多线程
                options.setNumThreads(4)
                false
            }

            interpreter = Interpreter(modelBuffer, options)

            // 预分配输入缓冲区
            inputBuffer = ByteBuffer.allocateDirect(
                INPUT_SIZE * INPUT_SIZE * 3 * 4 // float32, RGB
            ).apply {
                order(ByteOrder.nativeOrder())
            }

            isModelLoaded = true
            Log.i(TAG, "模型加载成功: $modelPath, GPU加速: $isGpuEnabled")
            true
        } catch (e: Exception) {
            Log.e(TAG, "模型加载失败: ${e.message}", e)
            isModelLoaded = false
            false
        }
    }

    /**
     * 检测人脸关键点
     *
     * @param bitmap 输入图像（Camera Frame）
     * @return 检测到的人脸关键点列表（最多 5 张脸，每张 98 个点）
     */
    fun detect(bitmap: Bitmap): List<FaceLandmarks> {
        if (!isModelLoaded || interpreter == null) {
            Log.w(TAG, "模型未加载，无法执行检测")
            return emptyList()
        }

        // 低功耗模式：跳帧处理，维持 ≥15fps
        if (lowPowerMode) {
            frameCounter++
            if (frameCounter % skipFrameInterval != 0L) {
                return lastDetectionResult
            }
        }

        // 1. 预处理：检测人脸区域（使用 Android FaceDetector 快速定位）
        val faceRegions = detectFaceRegions(bitmap)

        // 无人脸快速返回（≤10ms）
        if (faceRegions.isEmpty()) {
            lastDetectionResult = emptyList()
            return emptyList()
        }

        // 2. 对每个人脸区域执行关键点推理（最多 MAX_FACES 张脸）
        val results = mutableListOf<FaceLandmarks>()
        val facesToProcess = faceRegions.take(MAX_FACES)

        for ((index, region) in facesToProcess.withIndex()) {
            val landmarks = detectLandmarksForFace(bitmap, region, index)
            if (landmarks != null) {
                results.add(landmarks)
            }
        }

        lastDetectionResult = results
        return results
    }

    /**
     * 设置低功耗模式
     * 低功耗模式下降低推理频率，通过跳帧维持 ≥15fps 检测率
     *
     * @param enabled 是否启用低功耗模式
     */
    fun setLowPowerMode(enabled: Boolean) {
        lowPowerMode = enabled
        if (enabled) {
            // 假设正常 30fps 输入，跳帧间隔 2 → 实际推理 15fps
            skipFrameInterval = 2
            Log.i(TAG, "低功耗模式已启用，推理频率降至 ~15fps")
        } else {
            frameCounter = 0
            Log.i(TAG, "低功耗模式已关闭，恢复全帧推理")
        }
    }

    /**
     * 释放资源
     */
    fun release() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
        inputBuffer = null
        isModelLoaded = false
        Log.i(TAG, "FaceDetector 资源已释放")
    }

    // ==================== 私有方法 ====================

    /**
     * 检测人脸区域（快速定位，用于裁剪 ROI）
     * 使用 Android 内置 FaceDetector 进行快速人脸区域检测
     */
    private fun detectFaceRegions(bitmap: Bitmap): List<RectF> {
        // 缩放到检测尺寸
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        // Android FaceDetector 需要 RGB_565 格式
        val rgb565Bitmap = scaledBitmap.copy(Bitmap.Config.RGB_565, false)

        val faces = arrayOfNulls<android.media.FaceDetector.Face>(MAX_FACES)
        val faceCount = androidFaceDetector.findFaces(rgb565Bitmap, faces)

        if (faceCount == 0) {
            if (scaledBitmap != bitmap) scaledBitmap.recycle()
            rgb565Bitmap.recycle()
            return emptyList()
        }

        val scaleX = bitmap.width.toFloat() / INPUT_SIZE
        val scaleY = bitmap.height.toFloat() / INPUT_SIZE

        val regions = mutableListOf<RectF>()
        for (i in 0 until faceCount) {
            val face = faces[i] ?: continue
            val midPoint = android.graphics.PointF()
            face.getMidPoint(midPoint)
            val eyeDistance = face.eyesDistance()

            // 根据双眼距离估算人脸边界框
            val faceWidth = eyeDistance * 2.5f
            val faceHeight = eyeDistance * 3.0f
            val left = (midPoint.x - faceWidth / 2) * scaleX
            val top = (midPoint.y - faceHeight / 2.5f) * scaleY
            val right = (midPoint.x + faceWidth / 2) * scaleX
            val bottom = (midPoint.y + faceHeight / 1.8f) * scaleY

            regions.add(RectF(
                left.coerceIn(0f, bitmap.width.toFloat()),
                top.coerceIn(0f, bitmap.height.toFloat()),
                right.coerceIn(0f, bitmap.width.toFloat()),
                bottom.coerceIn(0f, bitmap.height.toFloat())
            ))
        }

        if (scaledBitmap != bitmap) scaledBitmap.recycle()
        rgb565Bitmap.recycle()

        return regions
    }

    /**
     * 对单张人脸区域执行 98 点关键点推理
     *
     * @param bitmap 原始图像
     * @param faceRegion 人脸区域边界框
     * @param faceId 人脸 ID
     * @return 关键点结果，置信度过低时返回 null
     */
    private fun detectLandmarksForFace(
        bitmap: Bitmap,
        faceRegion: RectF,
        faceId: Int
    ): FaceLandmarks? {
        val buffer = inputBuffer ?: return null
        buffer.rewind()

        // 裁剪人脸区域并缩放到模型输入尺寸
        val cropLeft = faceRegion.left.toInt().coerceIn(0, bitmap.width - 1)
        val cropTop = faceRegion.top.toInt().coerceIn(0, bitmap.height - 1)
        val cropWidth = (faceRegion.width()).toInt().coerceIn(1, bitmap.width - cropLeft)
        val cropHeight = (faceRegion.height()).toInt().coerceIn(1, bitmap.height - cropTop)

        val croppedBitmap = Bitmap.createBitmap(bitmap, cropLeft, cropTop, cropWidth, cropHeight)
        val resizedBitmap = Bitmap.createScaledBitmap(croppedBitmap, INPUT_SIZE, INPUT_SIZE, true)

        // 归一化像素值到 [0, 1] 并填充输入缓冲区
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }

        // 执行推理：输出为 [1, 98*2+1] → 98 个 (x,y) 坐标 + 1 个置信度
        val outputSize = LANDMARK_COUNT * 2 + 1
        val outputBuffer = Array(1) { FloatArray(outputSize) }

        try {
            interpreter?.run(buffer, outputBuffer)
        } catch (e: Exception) {
            Log.e(TAG, "推理执行失败: ${e.message}", e)
            croppedBitmap.recycle()
            if (resizedBitmap != croppedBitmap) resizedBitmap.recycle()
            return null
        }

        val output = outputBuffer[0]
        val confidence = output[LANDMARK_COUNT * 2] // 最后一个值为置信度

        // 支持部分遮挡（≤30%）的降级检测：使用更低的置信度阈值
        if (confidence < OCCLUSION_CONFIDENCE_THRESHOLD) {
            croppedBitmap.recycle()
            if (resizedBitmap != croppedBitmap) resizedBitmap.recycle()
            return null
        }

        // 将归一化坐标转换回原图坐标
        val points = mutableListOf<PointF>()
        for (i in 0 until LANDMARK_COUNT) {
            val normalizedX = output[i * 2]
            val normalizedY = output[i * 2 + 1]

            // 从模型输出空间 [0,1] 映射回原图坐标
            val x = faceRegion.left + normalizedX * faceRegion.width()
            val y = faceRegion.top + normalizedY * faceRegion.height()
            points.add(PointF(x, y))
        }

        croppedBitmap.recycle()
        if (resizedBitmap != croppedBitmap) resizedBitmap.recycle()

        return FaceLandmarks(
            faceId = faceId,
            points = points,
            boundingBox = faceRegion,
            confidence = confidence
        )
    }

    /**
     * 从 assets 加载 TFLite 模型文件
     */
    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}
