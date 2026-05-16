package com.orangecloud.beautysdk.deformer

import android.opengl.GLES30
import android.util.Log
import com.orangecloud.beautysdk.models.FaceLandmarks
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * 人脸变形器 - MLS 移动最小二乘法 + OpenGL ES 3.0 Mesh 渲染
 *
 * 功能：
 * - MLS（Moving Least Squares）仿射变形算法：基于控制点对计算变形场
 * - 瘦脸：下颌线 Landmark 向脸部中心收缩
 * - 大眼：眼部 Landmark 从眼部中心向外扩张
 * - OpenGL Mesh 渲染：40×40 均匀三角形网格 + 顶点/片段着色器
 * - EMA 时间平滑（alpha=0.6）防止帧间抖动
 * - 多脸独立变形
 * - 零强度恒等性：slimFace=0.0 且 enlargeEye=0.0 时直接返回输入纹理 ID
 * - 使用 FBO 进行离屏渲染
 *
 * 需求: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8
 */
class FaceDeformer {

    companion object {
        private const val TAG = "FaceDeformer"

        /** Mesh 网格分辨率（每行/列的顶点数） */
        private const val MESH_GRID_SIZE = 40

        /** EMA 平滑因子（0.0~1.0，越大越跟随当前帧） */
        private const val EMA_SMOOTHING_FACTOR = 0.6f

        /** 瘦脸最大位移比例（相对于人脸宽度） */
        private const val MAX_SLIM_DISPLACEMENT = 0.06f

        /** 大眼最大缩放比例 */
        private const val MAX_ENLARGE_SCALE = 0.3f

        /** MLS 权重衰减指数 */
        private const val MLS_ALPHA = 1.5f

        /** 下颌线 Landmark 索引（98 点模型中的下颌线区域） */
        private val JAWLINE_INDICES = intArrayOf(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        )

        /** 左眼 Landmark 索引 */
        private val LEFT_EYE_INDICES = intArrayOf(60, 61, 62, 63, 64, 65, 66, 67)

        /** 右眼 Landmark 索引 */
        private val RIGHT_EYE_INDICES = intArrayOf(68, 69, 70, 71, 72, 73, 74, 75)

        // ==================== 顶点着色器 ====================
        private const val VERTEX_SHADER = """
            #version 300 es
            layout(location = 0) in vec2 aPosition;
            layout(location = 1) in vec2 aTexCoord;
            out vec2 vTexCoord;
            void main() {
                // 将归一化坐标 [0,1] 映射到裁剪空间 [-1,1]，Y 轴翻转
                gl_Position = vec4(aPosition.x * 2.0 - 1.0, 1.0 - aPosition.y * 2.0, 0.0, 1.0);
                vTexCoord = aTexCoord;
            }
        """

        // ==================== 片段着色器 ====================
        private const val FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            in vec2 vTexCoord;
            out vec4 fragColor;
            uniform sampler2D uTexture;
            void main() {
                fragColor = texture(uTexture, vTexCoord);
            }
        """
    }

    // ==================== 状态字段 ====================

    /** 是否已初始化 */
    private var isInitialized = false

    /** Shader 程序 ID */
    private var program = 0

    /** FBO ID */
    private var fboId = 0

    /** FBO 附加的输出纹理 */
    private var fboTextureId = 0

    /** 渲染目标宽度 */
    private var textureWidth = 0

    /** 渲染目标高度 */
    private var textureHeight = 0

    /** 基础 Mesh 顶点（未变形状态，position + texCoord 交错） */
    private val baseMeshVertices: FloatArray

    /** 基础 Mesh 三角形索引 */
    private val baseMeshIndices: ShortArray

    /** 每张脸的 EMA 平滑状态（faceId → 上一帧平滑后的 Landmark） */
    private val smoothedLandmarks = mutableMapOf<Int, FloatArray>()

    init {
        // 预计算基础 Mesh 网格
        val (vertices, indices) = generateBaseMesh(MESH_GRID_SIZE)
        baseMeshVertices = vertices
        baseMeshIndices = indices
    }

    // ==================== 公开方法 ====================

    /**
     * 初始化 FaceDeformer
     * 编译 Shader 程序，创建 FBO
     *
     * @return 是否初始化成功
     */
    fun initialize(): Boolean {
        if (isInitialized) return true

        return try {
            program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
            if (program == 0) {
                Log.e(TAG, "Mesh Shader 程序创建失败")
                return false
            }
            isInitialized = true
            Log.i(TAG, "FaceDeformer 初始化成功")
            true
        } catch (e: Exception) {
            Log.e(TAG, "FaceDeformer 初始化异常: ${e.message}")
            false
        }
    }


    /**
     * 对单帧执行美型变形（全量参数版）
     *
     * @param inputTexture 输入纹理 ID
     * @param landmarks 人脸关键点列表（支持多脸）
     * @param params 美颜参数（包含所有美型强度）
     * @return 输出纹理 ID（零强度时返回输入纹理 ID）
     */
    fun deform(
        inputTexture: Int,
        landmarks: List<FaceLandmarks>,
        params: com.orangecloud.beautysdk.models.BeautyParams
    ): Int {
        // 零强度恒等性
        if (!params.hasAnyDeformation) {
            return inputTexture
        }

        // 无人脸时直接返回
        if (landmarks.isEmpty()) {
            return inputTexture
        }

        if (!isInitialized) {
            Log.w(TAG, "FaceDeformer 未初始化，返回原始纹理")
            return inputTexture
        }

        // 确保 FBO 尺寸匹配
        ensureFboSize(inputTexture)

        // 绑定 FBO 进行离屏渲染
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
        GLES30.glViewport(0, 0, textureWidth, textureHeight)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        GLES30.glUseProgram(program)

        // 绑定输入纹理
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTexture)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uTexture"), 0)

        // 启用混合（支持多脸叠加渲染）
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)

        // 为每张脸独立计算变形 Mesh 并渲染
        for (face in landmarks) {
            val smoothedPoints = applyTemporalSmoothing(face.faceId, face.points)
            val deformedVertices = computeMLSDeformationFull(
                smoothedPoints,
                face.boundingBox.left, face.boundingBox.top,
                face.boundingBox.width(), face.boundingBox.height(),
                params
            )
            renderMesh(deformedVertices, baseMeshIndices)
        }

        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)

        return fboTextureId
    }

    /**
     * 兼容旧签名（仅瘦脸 + 大眼）
     */
    fun deform(
        inputTexture: Int,
        landmarks: List<FaceLandmarks>,
        slimFace: Float,
        enlargeEye: Float
    ): Int = deform(
        inputTexture,
        landmarks,
        com.orangecloud.beautysdk.models.BeautyParams(
            slimFaceIntensity = slimFace,
            enlargeEyeIntensity = enlargeEye
        )
    )

    /**
     * 设置渲染目标尺寸
     */
    fun setSize(width: Int, height: Int) {
        if (width != textureWidth || height != textureHeight) {
            textureWidth = width
            textureHeight = height
            recreateFbo()
        }
    }

    /**
     * 释放所有 GPU 资源并清除平滑状态
     */
    fun release() {
        if (program != 0) {
            GLES30.glDeleteProgram(program)
            program = 0
        }
        deleteFbo()
        smoothedLandmarks.clear()
        isInitialized = false
        Log.i(TAG, "FaceDeformer 资源已释放")
    }

    /**
     * 重置时间平滑状态（场景切换时调用）
     */
    fun resetSmoothing() {
        smoothedLandmarks.clear()
    }

    // ==================== MLS 变形算法 ====================

    // ==================== 扩展版 MLS 变形（全量 BeautyParams） ====================

    /**
     * 扩展版：接受全量 BeautyParams（含所有美型）
     */
    private fun computeMLSDeformationFull(
        landmarks: FloatArray,
        faceLeft: Float,
        faceTop: Float,
        faceWidth: Float,
        faceHeight: Float,
        params: com.orangecloud.beautysdk.models.BeautyParams
    ): FloatArray {
        val controlPoints = computeControlPointsFull(landmarks, faceWidth, faceHeight, params)

        if (controlPoints.isEmpty()) {
            return baseMeshVertices.copyOf()
        }

        val deformed = baseMeshVertices.copyOf()
        val vertexCount = MESH_GRID_SIZE * MESH_GRID_SIZE
        for (i in 0 until vertexCount) {
            val baseIdx = i * 4
            val px = deformed[baseIdx]
            val py = deformed[baseIdx + 1]
            val (dx, dy) = mlsAffineDeformation(px, py, controlPoints)
            deformed[baseIdx] = dx
            deformed[baseIdx + 1] = dy
        }
        return deformed
    }

    /**
     * 扩展版控制点计算，覆盖全部 11 项美型（4 基础 + 9 扩展）
     */
    private fun computeControlPointsFull(
        landmarks: FloatArray,
        faceWidth: Float,
        faceHeight: Float,
        params: com.orangecloud.beautysdk.models.BeautyParams
    ): FloatArray {
        val numLandmarks = landmarks.size / 2
        if (numLandmarks < 98) return FloatArray(0)

        val points = mutableListOf<Float>()
        val faceCenterX = landmarks[54 * 2]
        val faceCenterY = landmarks[54 * 2 + 1]

        fun addPoint(srcX: Float, srcY: Float, dstX: Float, dstY: Float) {
            points.add(srcX); points.add(srcY); points.add(dstX); points.add(dstY)
        }

        fun towardCenter(idx: Int, disp: Float) {
            if (idx >= numLandmarks) return
            val sx = landmarks[idx * 2]; val sy = landmarks[idx * 2 + 1]
            val dx = faceCenterX - sx; val dy = faceCenterY - sy
            val len = sqrt(dx * dx + dy * dy)
            if (len < 1e-6f) return
            addPoint(sx, sy, sx + dx / len * disp, sy + dy / len * disp)
        }

        fun horizontalTowardCenter(idx: Int, disp: Float) {
            if (idx >= numLandmarks) return
            val sx = landmarks[idx * 2]; val sy = landmarks[idx * 2 + 1]
            val dx = faceCenterX - sx
            if (kotlin.math.abs(dx) < 1e-6f) return
            val sign = if (dx > 0) 1f else -1f
            addPoint(sx, sy, sx + sign * disp, sy)
        }

        if (params.slimFaceIntensity > 0f) {
            val disp = MAX_SLIM_DISPLACEMENT * params.slimFaceIntensity.coerceIn(0f, 1f) * faceWidth
            for (i in 2 until JAWLINE_INDICES.size step 2) {
                towardCenter(JAWLINE_INDICES[i], disp)
            }
        }

        if (params.enlargeEyeIntensity > 0f) {
            val scale = 1.0f + MAX_ENLARGE_SCALE * params.enlargeEyeIntensity.coerceIn(0f, 1f)
            addEnlargeEyeControlPoints(landmarks, LEFT_EYE_INDICES, scale, numLandmarks, points)
            addEnlargeEyeControlPoints(landmarks, RIGHT_EYE_INDICES, scale, numLandmarks, points)
        }

        if (params.slimChinIntensity > 0f) {
            val disp = 0.05f * params.slimChinIntensity.coerceIn(0f, 1f) * faceHeight
            for (idx in intArrayOf(14, 15, 16, 17, 18)) towardCenter(idx, disp)
        }

        if (params.slimNoseIntensity > 0f && numLandmarks > 58) {
            val disp = 0.03f * params.slimNoseIntensity.coerceIn(0f, 1f) * faceWidth
            val bridgeX = landmarks[51 * 2]
            for (idx in intArrayOf(55, 56, 57, 58, 59)) {
                if (idx >= numLandmarks) continue
                val sx = landmarks[idx * 2]; val sy = landmarks[idx * 2 + 1]
                val dx = bridgeX - sx
                if (kotlin.math.abs(dx) < 1e-6f) continue
                val sign = if (dx > 0) 1f else -1f
                addPoint(sx, sy, sx + sign * disp, sy)
            }
        }

        if (params.mouthShapeIntensity > 0f && numLandmarks > 87) {
            val disp = 0.02f * params.mouthShapeIntensity.coerceIn(0f, 1f) * faceWidth
            val mcX = (landmarks[76 * 2] + landmarks[82 * 2]) / 2f
            val mcY = (landmarks[76 * 2 + 1] + landmarks[82 * 2 + 1]) / 2f
            for (idx in intArrayOf(76, 77, 81, 82, 83, 87)) {
                if (idx >= numLandmarks) continue
                val sx = landmarks[idx * 2]; val sy = landmarks[idx * 2 + 1]
                val dx = mcX - sx; val dy = mcY - sy
                val len = sqrt(dx * dx + dy * dy)
                if (len < 1e-6f) continue
                addPoint(sx, sy, sx + dx / len * disp, sy + dy / len * disp)
            }
        }

        if (params.foreheadIntensity > 0f) {
            val disp = 0.02f * params.foreheadIntensity.coerceIn(0f, 1f) * faceHeight
            for (idx in 33..42) {
                if (idx >= numLandmarks) continue
                val sx = landmarks[idx * 2]; val sy = landmarks[idx * 2 + 1]
                addPoint(sx, sy, sx, sy - disp)
            }
        }

        if (params.hairlineIntensity > 0f) {
            val disp = 0.04f * params.hairlineIntensity.coerceIn(0f, 1f) * faceHeight
            for (idx in intArrayOf(33, 38, 42)) {
                if (idx >= numLandmarks) continue
                val sx = landmarks[idx * 2]
                val sy = landmarks[idx * 2 + 1] - 0.3f * faceHeight
                addPoint(sx, sy, sx, sy - disp)
            }
        }

        if (params.slimCheekboneIntensity > 0f) {
            val disp = 0.04f * params.slimCheekboneIntensity.coerceIn(0f, 1f) * faceWidth
            for (idx in intArrayOf(2, 3, 4, 5, 6, 26, 27, 28, 29, 30)) {
                horizontalTowardCenter(idx, disp)
            }
        }

        if (params.eyebrowShapeIntensity > 0f && numLandmarks > 46) {
            val disp = 0.02f * params.eyebrowShapeIntensity.coerceIn(0f, 1f) * faceWidth
            if (37 < numLandmarks) {
                val sx = landmarks[37 * 2]; val sy = landmarks[37 * 2 + 1]
                addPoint(sx, sy, sx - disp, sy)
            }
            if (46 < numLandmarks) {
                val sx = landmarks[46 * 2]; val sy = landmarks[46 * 2 + 1]
                addPoint(sx, sy, sx + disp, sy)
            }
        }

        if (params.vShapeIntensity > 0f) {
            val disp = 0.05f * params.vShapeIntensity.coerceIn(0f, 1f) * faceWidth
            for (idx in 10..22) towardCenter(idx, disp)
        }

        if (params.jawboneIntensity > 0f) {
            val disp = 0.03f * params.jawboneIntensity.coerceIn(0f, 1f) * faceWidth
            for (idx in intArrayOf(3, 4, 5, 27, 28, 29)) horizontalTowardCenter(idx, disp)
        }

        return points.toFloatArray()
    }

    // ==================== 原 MLS 变形（保留兼容） ====================

    /**
     * 计算 MLS 变形后的 Mesh 顶点
     *
     * @param landmarks 平滑后的 Landmark 点（归一化坐标，x/y 交错存储）
     * @param faceLeft 人脸边界框左边
     * @param faceTop 人脸边界框上边
     * @param faceWidth 人脸边界框宽度
     * @param faceHeight 人脸边界框高度
     * @param slimFace 瘦脸强度
     * @param enlargeEye 大眼强度
     * @return 变形后的顶点数据（position + texCoord 交错）
     */
    private fun computeMLSDeformation(
        landmarks: FloatArray,
        faceLeft: Float,
        faceTop: Float,
        faceWidth: Float,
        faceHeight: Float,
        slimFace: Float,
        enlargeEye: Float
    ): FloatArray {
        // 计算控制点对（源点 → 目标点）
        val controlPoints = computeControlPoints(
            landmarks, faceWidth, slimFace, enlargeEye
        )

        // 如果没有有效控制点，返回原始 Mesh
        if (controlPoints.isEmpty()) {
            return baseMeshVertices.copyOf()
        }

        // 对每个 Mesh 顶点应用 MLS 仿射变形
        val deformed = baseMeshVertices.copyOf()
        val vertexCount = MESH_GRID_SIZE * MESH_GRID_SIZE
        for (i in 0 until vertexCount) {
            val baseIdx = i * 4  // 每个顶点 4 个 float: posX, posY, texU, texV
            val px = deformed[baseIdx]
            val py = deformed[baseIdx + 1]

            val (dx, dy) = mlsAffineDeformation(px, py, controlPoints)
            deformed[baseIdx] = dx
            deformed[baseIdx + 1] = dy
            // texCoord 保持不变
        }

        return deformed
    }


    /**
     * MLS 仿射变形（Affine Deformation）
     * 基于加权最小二乘法计算点的变形位置
     *
     * @param px 待变形点的 X 坐标
     * @param py 待变形点的 Y 坐标
     * @param controlPoints 控制点对列表 [srcX, srcY, dstX, dstY, ...]
     * @return 变形后的坐标 (x, y)
     */
    private fun mlsAffineDeformation(
        px: Float,
        py: Float,
        controlPoints: FloatArray
    ): Pair<Float, Float> {
        val numPoints = controlPoints.size / 4

        // 计算每个控制点的权重 w_i = 1 / |p_i - v|^(2*alpha)
        val weights = FloatArray(numPoints)
        var totalWeight = 0.0f

        for (i in 0 until numPoints) {
            val srcX = controlPoints[i * 4]
            val srcY = controlPoints[i * 4 + 1]
            val diffX = px - srcX
            val diffY = py - srcY
            val distSq = diffX * diffX + diffY * diffY

            // 点与控制点重合时直接返回目标点
            if (distSq < 1e-10f) {
                return Pair(controlPoints[i * 4 + 2], controlPoints[i * 4 + 3])
            }

            val w = 1.0f / distSq.toDouble().pow(MLS_ALPHA.toDouble()).toFloat()
            weights[i] = w
            totalWeight += w
        }

        // 计算加权质心
        var pStarX = 0.0f
        var pStarY = 0.0f
        var qStarX = 0.0f
        var qStarY = 0.0f

        for (i in 0 until numPoints) {
            val nw = weights[i] / totalWeight
            pStarX += nw * controlPoints[i * 4]
            pStarY += nw * controlPoints[i * 4 + 1]
            qStarX += nw * controlPoints[i * 4 + 2]
            qStarY += nw * controlPoints[i * 4 + 3]
        }

        // 构建仿射变换矩阵
        // sumPPt = sum(w_i * pHat_i^T * pHat_i)
        // sumPQt = sum(w_i * pHat_i^T * qHat_i)
        var pp00 = 0.0f; var pp01 = 0.0f
        var pp10 = 0.0f; var pp11 = 0.0f
        var pq00 = 0.0f; var pq01 = 0.0f
        var pq10 = 0.0f; var pq11 = 0.0f

        for (i in 0 until numPoints) {
            val w = weights[i]
            val phX = controlPoints[i * 4] - pStarX
            val phY = controlPoints[i * 4 + 1] - pStarY
            val qhX = controlPoints[i * 4 + 2] - qStarX
            val qhY = controlPoints[i * 4 + 3] - qStarY

            pp00 += w * phX * phX
            pp01 += w * phX * phY
            pp10 += w * phY * phX
            pp11 += w * phY * phY

            pq00 += w * phX * qhX
            pq01 += w * phX * qhY
            pq10 += w * phY * qhX
            pq11 += w * phY * qhY
        }

        // 求逆矩阵 det = pp00*pp11 - pp01*pp10
        val det = pp00 * pp11 - pp01 * pp10
        if (abs(det) < 1e-10f) {
            // 矩阵奇异，使用简单加权平均
            var resultX = 0.0f
            var resultY = 0.0f
            for (i in 0 until numPoints) {
                val nw = weights[i] / totalWeight
                resultX += nw * (controlPoints[i * 4 + 2] - controlPoints[i * 4])
                resultY += nw * (controlPoints[i * 4 + 3] - controlPoints[i * 4 + 1])
            }
            return Pair(px + resultX, py + resultY)
        }

        val invDet = 1.0f / det
        // inv(sumPPt)
        val inv00 = pp11 * invDet
        val inv01 = -pp01 * invDet
        val inv10 = -pp10 * invDet
        val inv11 = pp00 * invDet

        // M = inv(sumPPt) * sumPQt
        val m00 = inv00 * pq00 + inv01 * pq10
        val m01 = inv00 * pq01 + inv01 * pq11
        val m10 = inv10 * pq00 + inv11 * pq10
        val m11 = inv10 * pq01 + inv11 * pq11

        // 变形结果：q* + (v - p*) * M
        val vpX = px - pStarX
        val vpY = py - pStarY
        val transformedX = vpX * m00 + vpY * m10
        val transformedY = vpX * m01 + vpY * m11

        return Pair(qStarX + transformedX, qStarY + transformedY)
    }

    // ==================== 控制点计算 ====================

    /**
     * 计算瘦脸/大眼的控制点对（源点 → 目标点）
     *
     * @param landmarks 归一化的 Landmark 坐标（x/y 交错存储）
     * @param faceWidth 人脸宽度
     * @param slimFace 瘦脸强度
     * @param enlargeEye 大眼强度
     * @return 控制点对数组 [srcX, srcY, dstX, dstY, ...]
     */
    private fun computeControlPoints(
        landmarks: FloatArray,
        faceWidth: Float,
        slimFace: Float,
        enlargeEye: Float
    ): FloatArray {
        val numLandmarks = landmarks.size / 2
        if (numLandmarks < 98) return FloatArray(0)

        val points = mutableListOf<Float>()

        // 瘦脸控制点：下颌线向内收缩
        if (slimFace > 0.0f) {
            // 使用鼻尖（索引 54）作为脸部中心参考
            val faceCenterX = landmarks[54 * 2]
            val faceCenterY = landmarks[54 * 2 + 1]
            val displacement = MAX_SLIM_DISPLACEMENT * slimFace * faceWidth

            // 间隔采样下颌线关键点以减少计算量
            for (i in 2 until JAWLINE_INDICES.size step 2) {
                val landmarkIdx = JAWLINE_INDICES[i]
                if (landmarkIdx >= numLandmarks) continue

                val srcX = landmarks[landmarkIdx * 2]
                val srcY = landmarks[landmarkIdx * 2 + 1]

                // 计算向脸部中心的方向
                val dirX = faceCenterX - srcX
                val dirY = faceCenterY - srcY
                val dirLen = sqrt(dirX * dirX + dirY * dirY)
                if (dirLen < 1e-6f) continue

                val normDirX = dirX / dirLen
                val normDirY = dirY / dirLen
                val dstX = srcX + normDirX * displacement
                val dstY = srcY + normDirY * displacement

                points.add(srcX); points.add(srcY)
                points.add(dstX); points.add(dstY)
            }
        }

        // 大眼控制点：眼部 Landmark 向外扩张
        if (enlargeEye > 0.0f) {
            val scale = 1.0f + MAX_ENLARGE_SCALE * enlargeEye

            // 左眼
            addEnlargeEyeControlPoints(landmarks, LEFT_EYE_INDICES, scale, numLandmarks, points)

            // 右眼
            addEnlargeEyeControlPoints(landmarks, RIGHT_EYE_INDICES, scale, numLandmarks, points)
        }

        return points.toFloatArray()
    }

    /**
     * 添加大眼变形的控制点
     */
    private fun addEnlargeEyeControlPoints(
        landmarks: FloatArray,
        eyeIndices: IntArray,
        scale: Float,
        numLandmarks: Int,
        points: MutableList<Float>
    ) {
        // 计算眼部中心
        var centerX = 0.0f
        var centerY = 0.0f
        var count = 0
        for (idx in eyeIndices) {
            if (idx >= numLandmarks) continue
            centerX += landmarks[idx * 2]
            centerY += landmarks[idx * 2 + 1]
            count++
        }
        if (count == 0) return
        centerX /= count
        centerY /= count

        // 从眼部中心向外扩张
        for (idx in eyeIndices) {
            if (idx >= numLandmarks) continue
            val srcX = landmarks[idx * 2]
            val srcY = landmarks[idx * 2 + 1]
            val dirX = srcX - centerX
            val dirY = srcY - centerY
            val dstX = centerX + dirX * scale
            val dstY = centerY + dirY * scale

            points.add(srcX); points.add(srcY)
            points.add(dstX); points.add(dstY)
        }
    }


    // ==================== 时间平滑（EMA） ====================

    /**
     * 应用指数移动平均（EMA）时间平滑
     *
     * @param faceId 人脸 ID
     * @param landmarkPoints 当前帧的 Landmark 点列表
     * @return 平滑后的 Landmark 坐标（x/y 交错存储的 FloatArray）
     */
    private fun applyTemporalSmoothing(
        faceId: Int,
        landmarkPoints: List<android.graphics.PointF>
    ): FloatArray {
        val currentPoints = FloatArray(landmarkPoints.size * 2)
        for (i in landmarkPoints.indices) {
            currentPoints[i * 2] = landmarkPoints[i].x
            currentPoints[i * 2 + 1] = landmarkPoints[i].y
        }

        val previousPoints = smoothedLandmarks[faceId]
        if (previousPoints == null || previousPoints.size != currentPoints.size) {
            // 首帧或点数变化，直接使用当前帧
            smoothedLandmarks[faceId] = currentPoints.copyOf()
            return currentPoints
        }

        // EMA 平滑：smoothed = alpha * current + (1 - alpha) * previous
        val smoothed = FloatArray(currentPoints.size)
        for (i in currentPoints.indices) {
            smoothed[i] = EMA_SMOOTHING_FACTOR * currentPoints[i] +
                (1.0f - EMA_SMOOTHING_FACTOR) * previousPoints[i]
        }

        // 更新平滑状态
        smoothedLandmarks[faceId] = smoothed
        return smoothed
    }

    // ==================== OpenGL Mesh 渲染 ====================

    /**
     * 渲染变形后的 Mesh
     *
     * @param vertices 变形后的顶点数据（position + texCoord 交错）
     * @param indices 三角形索引
     */
    private fun renderMesh(vertices: FloatArray, indices: ShortArray) {
        // 创建顶点缓冲区
        val vertexBuffer = ByteBuffer.allocateDirect(vertices.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(vertices)
                position(0)
            }

        // 创建索引缓冲区
        val indexBuffer = ByteBuffer.allocateDirect(indices.size * 2)
            .order(ByteOrder.nativeOrder())
            .asShortBuffer()
            .apply {
                put(indices)
                position(0)
            }

        // 设置顶点属性：position (location=0)
        vertexBuffer.position(0)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(
            0, 2, GLES30.GL_FLOAT, false,
            4 * 4,  // stride = 4 floats (posX, posY, texU, texV)
            vertexBuffer
        )

        // 设置顶点属性：texCoord (location=1)
        vertexBuffer.position(2)  // 偏移 2 个 float 到 texCoord
        GLES30.glEnableVertexAttribArray(1)
        GLES30.glVertexAttribPointer(
            1, 2, GLES30.GL_FLOAT, false,
            4 * 4,  // stride = 4 floats
            vertexBuffer
        )

        // 绘制三角形 Mesh
        GLES30.glDrawElements(
            GLES30.GL_TRIANGLES,
            indices.size,
            GLES30.GL_UNSIGNED_SHORT,
            indexBuffer
        )

        GLES30.glDisableVertexAttribArray(0)
        GLES30.glDisableVertexAttribArray(1)
    }

    // ==================== Mesh 生成 ====================

    /**
     * 生成基础 Mesh 网格（均匀分布的三角形网格）
     * 每个顶点包含 4 个 float: posX, posY, texU, texV
     *
     * @param gridSize 网格每行/列的顶点数
     * @return (顶点数组, 索引数组)
     */
    private fun generateBaseMesh(gridSize: Int): Pair<FloatArray, ShortArray> {
        val vertexCount = gridSize * gridSize
        val vertices = FloatArray(vertexCount * 4)  // 4 floats per vertex

        // 生成均匀分布的顶点
        for (row in 0 until gridSize) {
            for (col in 0 until gridSize) {
                val idx = (row * gridSize + col) * 4
                val x = col.toFloat() / (gridSize - 1).toFloat()
                val y = row.toFloat() / (gridSize - 1).toFloat()
                vertices[idx] = x       // posX
                vertices[idx + 1] = y   // posY
                vertices[idx + 2] = x   // texU
                vertices[idx + 3] = y   // texV
            }
        }

        // 生成三角形索引（每个网格单元分为两个三角形）
        val cellCount = (gridSize - 1) * (gridSize - 1)
        val indices = ShortArray(cellCount * 6)
        var indexPos = 0

        for (row in 0 until gridSize - 1) {
            for (col in 0 until gridSize - 1) {
                val topLeft = (row * gridSize + col).toShort()
                val topRight = (topLeft + 1).toShort()
                val bottomLeft = ((row + 1) * gridSize + col).toShort()
                val bottomRight = (bottomLeft + 1).toShort()

                // 三角形 1：左上 → 右上 → 左下
                indices[indexPos++] = topLeft
                indices[indexPos++] = topRight
                indices[indexPos++] = bottomLeft

                // 三角形 2：右上 → 右下 → 左下
                indices[indexPos++] = topRight
                indices[indexPos++] = bottomRight
                indices[indexPos++] = bottomLeft
            }
        }

        return Pair(vertices, indices)
    }

    // ==================== FBO 管理 ====================

    /**
     * 确保 FBO 尺寸与输入纹理匹配
     */
    private fun ensureFboSize(inputTexture: Int) {
        if (textureWidth == 0 || textureHeight == 0) {
            val width = IntArray(1)
            val height = IntArray(1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTexture)
            GLES30.glGetTexLevelParameteriv(
                GLES30.GL_TEXTURE_2D, 0, GLES30.GL_TEXTURE_WIDTH, width, 0
            )
            GLES30.glGetTexLevelParameteriv(
                GLES30.GL_TEXTURE_2D, 0, GLES30.GL_TEXTURE_HEIGHT, height, 0
            )

            if (width[0] > 0 && height[0] > 0) {
                setSize(width[0], height[0])
            } else {
                // 默认 720p
                setSize(1280, 720)
            }
        }
    }

    /**
     * 创建/重建 FBO
     */
    private fun recreateFbo() {
        deleteFbo()

        if (textureWidth <= 0 || textureHeight <= 0) return

        val fboArray = IntArray(1)
        val texArray = IntArray(1)

        GLES30.glGenFramebuffers(1, fboArray, 0)
        GLES30.glGenTextures(1, texArray, 0)

        fboId = fboArray[0]
        fboTextureId = texArray[0]

        // 配置输出纹理
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fboTextureId)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA,
            textureWidth, textureHeight, 0,
            GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
        )
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        // 绑定纹理到 FBO
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D, fboTextureId, 0
        )

        val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
        if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "FBO 创建失败，状态: $status")
        }

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
    }

    /**
     * 删除 FBO 资源
     */
    private fun deleteFbo() {
        if (fboId != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(fboId), 0)
            fboId = 0
        }
        if (fboTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(fboTextureId), 0)
            fboTextureId = 0
        }
    }

    // ==================== Shader 工具方法 ====================

    /**
     * 编译并链接 Shader 程序
     */
    private fun createProgram(vertexSource: String, fragmentSource: String): Int {
        val vertexShader = compileShader(GLES30.GL_VERTEX_SHADER, vertexSource)
        if (vertexShader == 0) return 0

        val fragmentShader = compileShader(GLES30.GL_FRAGMENT_SHADER, fragmentSource)
        if (fragmentShader == 0) {
            GLES30.glDeleteShader(vertexShader)
            return 0
        }

        val program = GLES30.glCreateProgram()
        if (program == 0) {
            Log.e(TAG, "创建 Shader 程序失败")
            return 0
        }

        GLES30.glAttachShader(program, vertexShader)
        GLES30.glAttachShader(program, fragmentShader)
        GLES30.glLinkProgram(program)

        val linkStatus = IntArray(1)
        GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, linkStatus, 0)
        if (linkStatus[0] == 0) {
            val log = GLES30.glGetProgramInfoLog(program)
            Log.e(TAG, "Shader 程序链接失败: $log")
            GLES30.glDeleteProgram(program)
            GLES30.glDeleteShader(vertexShader)
            GLES30.glDeleteShader(fragmentShader)
            return 0
        }

        GLES30.glDeleteShader(vertexShader)
        GLES30.glDeleteShader(fragmentShader)

        return program
    }

    /**
     * 编译单个 Shader
     */
    private fun compileShader(type: Int, source: String): Int {
        val shader = GLES30.glCreateShader(type)
        if (shader == 0) {
            Log.e(TAG, "创建 Shader 失败，类型: $type")
            return 0
        }

        GLES30.glShaderSource(shader, source)
        GLES30.glCompileShader(shader)

        val compileStatus = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, compileStatus, 0)
        if (compileStatus[0] == 0) {
            val log = GLES30.glGetShaderInfoLog(shader)
            val typeName = if (type == GLES30.GL_VERTEX_SHADER) "Vertex" else "Fragment"
            Log.e(TAG, "$typeName Shader 编译失败: $log")
            GLES30.glDeleteShader(shader)
            return 0
        }

        return shader
    }
}
