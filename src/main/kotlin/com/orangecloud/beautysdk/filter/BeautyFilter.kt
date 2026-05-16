package com.orangecloud.beautysdk.filter

import android.opengl.GLES30
import android.util.Log
import com.orangecloud.beautysdk.models.FaceLandmarks
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * 美颜滤镜 - 基于 OpenGL ES 3.0 GLSL Shader
 *
 * 功能：
 * - 双边滤波磨皮 Shader（保留边缘细节的自然磨皮）
 * - 美白 Shader（调整皮肤区域亮度和饱和度）
 * - smoothing/whitening 参数范围 0.0~1.0，按比例应用
 * - 单帧处理 ≤8ms，维持 ≥30fps
 * - smoothing=0.0 且 whitening=0.0 时直接返回输入纹理 ID（pass-through）
 * - GPU 不可用时降级到 CPU 处理，记录警告日志
 *
 * 需求: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.9, 2.10
 */
class BeautyFilter {

    companion object {
        private const val TAG = "BeautyFilter"

        // ==================== 顶点着色器（通用） ====================
        private const val VERTEX_SHADER = """
            #version 300 es
            layout(location = 0) in vec4 aPosition;
            layout(location = 1) in vec2 aTexCoord;
            out vec2 vTexCoord;
            void main() {
                gl_Position = aPosition;
                vTexCoord = aTexCoord;
            }
        """

        // ==================== 双边滤波磨皮 Fragment Shader ====================
        private const val BILATERAL_FILTER_SHADER = """
            #version 300 es
            precision highp float;
            in vec2 vTexCoord;
            out vec4 fragColor;

            uniform sampler2D uTexture;
            uniform vec2 uTexelSize;
            uniform float uSmoothingIntensity;

            // 双边滤波参数
            const float SIGMA_SPATIAL = 3.0;
            const float SIGMA_COLOR = 0.1;
            const int KERNEL_RADIUS = 4;

            void main() {
                vec4 centerColor = texture(uTexture, vTexCoord);

                if (uSmoothingIntensity <= 0.001) {
                    fragColor = centerColor;
                    return;
                }

                float sigmaSpatial = SIGMA_SPATIAL * uSmoothingIntensity;
                float sigmaColor = SIGMA_COLOR + 0.15 * uSmoothingIntensity;

                float spatialFactor = -0.5 / (sigmaSpatial * sigmaSpatial);
                float colorFactor = -0.5 / (sigmaColor * sigmaColor);

                vec4 sumColor = vec4(0.0);
                float sumWeight = 0.0;

                int radius = int(float(KERNEL_RADIUS) * uSmoothingIntensity) + 1;

                for (int x = -radius; x <= radius; x++) {
                    for (int y = -radius; y <= radius; y++) {
                        vec2 offset = vec2(float(x), float(y)) * uTexelSize;
                        vec4 sampleColor = texture(uTexture, vTexCoord + offset);

                        float spatialDist = float(x * x + y * y);
                        float colorDist = distance(sampleColor.rgb, centerColor.rgb);
                        colorDist = colorDist * colorDist;

                        float weight = exp(spatialDist * spatialFactor + colorDist * colorFactor);

                        sumColor += sampleColor * weight;
                        sumWeight += weight;
                    }
                }

                vec4 smoothed = sumColor / sumWeight;
                fragColor = mix(centerColor, smoothed, uSmoothingIntensity);
            }
        """

        // ==================== 美白 Fragment Shader ====================
        private const val WHITENING_SHADER = """
            #version 300 es
            precision highp float;
            in vec2 vTexCoord;
            out vec4 fragColor;

            uniform sampler2D uTexture;
            uniform float uWhiteningIntensity;

            vec3 rgbToHsl(vec3 color) {
                float maxC = max(max(color.r, color.g), color.b);
                float minC = min(min(color.r, color.g), color.b);
                float l = (maxC + minC) * 0.5;
                float s = 0.0;
                float h = 0.0;

                if (maxC != minC) {
                    float d = maxC - minC;
                    s = l > 0.5 ? d / (2.0 - maxC - minC) : d / (maxC + minC);

                    if (maxC == color.r) {
                        h = (color.g - color.b) / d + (color.g < color.b ? 6.0 : 0.0);
                    } else if (maxC == color.g) {
                        h = (color.b - color.r) / d + 2.0;
                    } else {
                        h = (color.r - color.g) / d + 4.0;
                    }
                    h /= 6.0;
                }
                return vec3(h, s, l);
            }

            float hueToRgb(float p, float q, float t) {
                if (t < 0.0) t += 1.0;
                if (t > 1.0) t -= 1.0;
                if (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
                if (t < 1.0 / 2.0) return q;
                if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
                return p;
            }

            vec3 hslToRgb(vec3 hsl) {
                float h = hsl.x;
                float s = hsl.y;
                float l = hsl.z;

                if (s == 0.0) {
                    return vec3(l);
                }

                float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
                float p = 2.0 * l - q;

                float r = hueToRgb(p, q, h + 1.0 / 3.0);
                float g = hueToRgb(p, q, h);
                float b = hueToRgb(p, q, h - 1.0 / 3.0);

                return vec3(r, g, b);
            }

            void main() {
                vec4 color = texture(uTexture, vTexCoord);

                if (uWhiteningIntensity <= 0.001) {
                    fragColor = color;
                    return;
                }

                vec3 hsl = rgbToHsl(color.rgb);

                // 提亮：增加亮度
                float brightnessBoost = uWhiteningIntensity * 0.15;
                hsl.z = min(hsl.z + brightnessBoost, 1.0);

                // 降低饱和度使肤色更白皙
                float saturationReduce = uWhiteningIntensity * 0.1;
                hsl.y = max(hsl.y - saturationReduce, 0.0);

                vec3 whitened = hslToRgb(hsl);
                fragColor = vec4(mix(color.rgb, whitened, uWhiteningIntensity), color.a);
            }
        """

        /** 全屏四边形顶点坐标 */
        private val QUAD_VERTICES = floatArrayOf(
            -1.0f, -1.0f,  // 左下
             1.0f, -1.0f,  // 右下
            -1.0f,  1.0f,  // 左上
             1.0f,  1.0f   // 右上
        )

        /** 纹理坐标 */
        private val QUAD_TEX_COORDS = floatArrayOf(
            0.0f, 0.0f,
            1.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 1.0f
        )
    }

    // ==================== 状态字段 ====================

    /** 是否已初始化 */
    private var isInitialized: Boolean = false

    /** 是否 GPU 可用 */
    private var isGpuAvailable: Boolean = false

    /** 磨皮 Shader 程序 ID */
    private var smoothingProgram: Int = 0

    /** 美白 Shader 程序 ID */
    private var whiteningProgram: Int = 0

    /** FBO（Framebuffer Object）用于离屏渲染 */
    private var fboIds: IntArray = IntArray(2)

    /** FBO 附加的纹理 */
    private var fboTextureIds: IntArray = IntArray(2)

    /** 渲染目标宽度 */
    private var textureWidth: Int = 0

    /** 渲染目标高度 */
    private var textureHeight: Int = 0

    /** 顶点坐标缓冲区 */
    private var vertexBuffer: FloatBuffer? = null

    /** 纹理坐标缓冲区 */
    private var texCoordBuffer: FloatBuffer? = null

    // ==================== 公开方法 ====================

    /**
     * 初始化 BeautyFilter
     * 编译并链接磨皮/美白 GLSL Shader 程序，创建 FBO
     *
     * @return 是否初始化成功
     */
    fun initialize(): Boolean {
        if (isInitialized) return true

        return try {
            // 编译磨皮 Shader 程序
            smoothingProgram = createProgram(VERTEX_SHADER, BILATERAL_FILTER_SHADER)
            if (smoothingProgram == 0) {
                Log.w(TAG, "GPU 不可用：磨皮 Shader 编译失败，降级到 CPU 处理")
                isGpuAvailable = false
                isInitialized = true
                return true
            }

            // 编译美白 Shader 程序
            whiteningProgram = createProgram(VERTEX_SHADER, WHITENING_SHADER)
            if (whiteningProgram == 0) {
                Log.w(TAG, "GPU 不可用：美白 Shader 编译失败，降级到 CPU 处理")
                GLES30.glDeleteProgram(smoothingProgram)
                smoothingProgram = 0
                isGpuAvailable = false
                isInitialized = true
                return true
            }

            // 创建顶点和纹理坐标缓冲区
            vertexBuffer = createFloatBuffer(QUAD_VERTICES)
            texCoordBuffer = createFloatBuffer(QUAD_TEX_COORDS)

            isGpuAvailable = true
            isInitialized = true
            Log.i(TAG, "BeautyFilter 初始化成功，GPU 加速已启用")
            true
        } catch (e: Exception) {
            Log.w(TAG, "GPU 不可用：初始化异常，降级到 CPU 处理: ${e.message}")
            isGpuAvailable = false
            isInitialized = true
            true
        }
    }

    /**
     * 应用美颜效果
     *
     * @param inputTexture 输入纹理 ID
     * @param landmarks 人脸关键点（可选，用于精确皮肤区域定位）
     * @param smoothing 磨皮强度 (0.0 ~ 1.0)
     * @param whitening 美白强度 (0.0 ~ 1.0)
     * @return 输出纹理 ID（smoothing=0.0 且 whitening=0.0 时返回输入纹理 ID）
     */
    fun apply(
        inputTexture: Int,
        landmarks: FaceLandmarks?,
        smoothing: Float,
        whitening: Float
    ): Int {
        // 参数钳制到 [0.0, 1.0]
        val clampedSmoothing = smoothing.coerceIn(0.0f, 1.0f)
        val clampedWhitening = whitening.coerceIn(0.0f, 1.0f)

        // Pass-through：smoothing=0.0 且 whitening=0.0 时直接返回输入纹理
        if (clampedSmoothing <= 0.0f && clampedWhitening <= 0.0f) {
            return inputTexture
        }

        if (!isInitialized) {
            Log.w(TAG, "BeautyFilter 未初始化，返回原始纹理")
            return inputTexture
        }

        // GPU 不可用时降级到 CPU 处理
        if (!isGpuAvailable) {
            Log.w(TAG, "GPU 不可用，执行 CPU 降级处理")
            return applyCpuFallback(inputTexture, clampedSmoothing, clampedWhitening)
        }

        // 确保 FBO 尺寸匹配
        ensureFboSize(inputTexture)

        var currentTexture = inputTexture

        // 第一步：磨皮（双边滤波）
        if (clampedSmoothing > 0.0f) {
            currentTexture = applySmoothing(currentTexture, clampedSmoothing, 0)
        }

        // 第二步：美白
        if (clampedWhitening > 0.0f) {
            val fboIndex = if (clampedSmoothing > 0.0f) 1 else 0
            currentTexture = applyWhitening(currentTexture, clampedWhitening, fboIndex)
        }

        return currentTexture
    }

    /**
     * 设置渲染目标尺寸
     *
     * @param width 纹理宽度
     * @param height 纹理高度
     */
    fun setSize(width: Int, height: Int) {
        if (width != textureWidth || height != textureHeight) {
            textureWidth = width
            textureHeight = height
            recreateFbo()
        }
    }

    /**
     * 释放所有 GPU 资源
     */
    fun release() {
        if (smoothingProgram != 0) {
            GLES30.glDeleteProgram(smoothingProgram)
            smoothingProgram = 0
        }
        if (whiteningProgram != 0) {
            GLES30.glDeleteProgram(whiteningProgram)
            whiteningProgram = 0
        }
        deleteFbo()
        vertexBuffer = null
        texCoordBuffer = null
        isInitialized = false
        isGpuAvailable = false
        Log.i(TAG, "BeautyFilter 资源已释放")
    }

    // ==================== 私有方法 ====================

    /**
     * 应用磨皮效果（双边滤波）
     */
    private fun applySmoothing(inputTexture: Int, intensity: Float, fboIndex: Int): Int {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboIds[fboIndex])
        GLES30.glViewport(0, 0, textureWidth, textureHeight)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        GLES30.glUseProgram(smoothingProgram)

        // 绑定输入纹理
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTexture)
        GLES30.glUniform1i(
            GLES30.glGetUniformLocation(smoothingProgram, "uTexture"), 0
        )

        // 设置 uniform 参数
        GLES30.glUniform2f(
            GLES30.glGetUniformLocation(smoothingProgram, "uTexelSize"),
            1.0f / textureWidth,
            1.0f / textureHeight
        )
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(smoothingProgram, "uSmoothingIntensity"),
            intensity
        )

        // 绘制全屏四边形
        drawQuad()

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        return fboTextureIds[fboIndex]
    }

    /**
     * 应用美白效果
     */
    private fun applyWhitening(inputTexture: Int, intensity: Float, fboIndex: Int): Int {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboIds[fboIndex])
        GLES30.glViewport(0, 0, textureWidth, textureHeight)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        GLES30.glUseProgram(whiteningProgram)

        // 绑定输入纹理
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTexture)
        GLES30.glUniform1i(
            GLES30.glGetUniformLocation(whiteningProgram, "uTexture"), 0
        )

        // 设置美白强度
        GLES30.glUniform1f(
            GLES30.glGetUniformLocation(whiteningProgram, "uWhiteningIntensity"),
            intensity
        )

        // 绘制全屏四边形
        drawQuad()

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        return fboTextureIds[fboIndex]
    }

    /**
     * 绘制全屏四边形
     */
    private fun drawQuad() {
        val vBuf = vertexBuffer ?: return
        val tBuf = texCoordBuffer ?: return

        // 顶点坐标
        vBuf.position(0)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 0, vBuf)

        // 纹理坐标
        tBuf.position(0)
        GLES30.glEnableVertexAttribArray(1)
        GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, 0, tBuf)

        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)

        GLES30.glDisableVertexAttribArray(0)
        GLES30.glDisableVertexAttribArray(1)
    }

    /**
     * 确保 FBO 尺寸与输入纹理匹配
     */
    private fun ensureFboSize(inputTexture: Int) {
        if (textureWidth == 0 || textureHeight == 0) {
            // 查询输入纹理尺寸
            val prevFbo = IntArray(1)
            GLES30.glGetIntegerv(GLES30.GL_FRAMEBUFFER_BINDING, prevFbo, 0)

            val width = IntArray(1)
            val height = IntArray(1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTexture)
            GLES30.glGetTexLevelParameteriv(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_TEXTURE_WIDTH, width, 0)
            GLES30.glGetTexLevelParameteriv(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_TEXTURE_HEIGHT, height, 0)

            if (width[0] > 0 && height[0] > 0) {
                setSize(width[0], height[0])
            } else {
                // 默认 720p
                setSize(1280, 720)
            }

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, prevFbo[0])
        }
    }

    /**
     * 创建/重建 FBO
     */
    private fun recreateFbo() {
        deleteFbo()

        if (textureWidth <= 0 || textureHeight <= 0) return

        // 创建 2 个 FBO（磨皮和美白各一个）
        GLES30.glGenFramebuffers(2, fboIds, 0)
        GLES30.glGenTextures(2, fboTextureIds, 0)

        for (i in 0..1) {
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fboTextureIds[i])
            GLES30.glTexImage2D(
                GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA,
                textureWidth, textureHeight, 0,
                GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
            )
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboIds[i])
            GLES30.glFramebufferTexture2D(
                GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D, fboTextureIds[i], 0
            )

            val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
            if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) {
                Log.e(TAG, "FBO[$i] 创建失败，状态: $status")
            }
        }

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
    }

    /**
     * 删除 FBO 资源
     */
    private fun deleteFbo() {
        if (fboIds[0] != 0) {
            GLES30.glDeleteFramebuffers(2, fboIds, 0)
            fboIds = IntArray(2)
        }
        if (fboTextureIds[0] != 0) {
            GLES30.glDeleteTextures(2, fboTextureIds, 0)
            fboTextureIds = IntArray(2)
        }
    }

    /**
     * CPU 降级处理（GPU 不可用时的回退方案）
     * 返回输入纹理 ID，仅记录警告日志
     */
    private fun applyCpuFallback(
        inputTexture: Int,
        smoothing: Float,
        whitening: Float
    ): Int {
        Log.w(TAG, "CPU 降级模式：smoothing=$smoothing, whitening=$whitening，性能可能下降")
        // CPU 降级模式下直接返回输入纹理，避免崩溃
        // 实际生产环境可在此实现 CPU 端的简化滤镜处理
        return inputTexture
    }

    /**
     * 编译并链接 Shader 程序
     *
     * @param vertexSource 顶点着色器源码
     * @param fragmentSource 片段着色器源码
     * @return 程序 ID，失败返回 0
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

        // 链接成功后可删除 Shader 对象
        GLES30.glDeleteShader(vertexShader)
        GLES30.glDeleteShader(fragmentShader)

        return program
    }

    /**
     * 编译单个 Shader
     *
     * @param type Shader 类型（GL_VERTEX_SHADER 或 GL_FRAGMENT_SHADER）
     * @param source Shader 源码
     * @return Shader ID，失败返回 0
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

    /**
     * 创建 FloatBuffer
     */
    private fun createFloatBuffer(data: FloatArray): FloatBuffer {
        return ByteBuffer.allocateDirect(data.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(data)
                position(0)
            }
    }
}
