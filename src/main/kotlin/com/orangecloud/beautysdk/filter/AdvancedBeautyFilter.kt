package com.orangecloud.beautysdk.filter

import android.opengl.GLES30
import android.util.Log
import com.orangecloud.beautysdk.models.FaceLandmarks
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * 高级美颜：亮眼 / 白牙 / 祛黑眼圈 / 祛法令纹 / 祛皱纹
 *
 * 基于 landmarks 定位局部 mask，用 fragment shader 做区域增强。
 */
class AdvancedBeautyFilter {

    data class Params(
        var brightEye: Float = 0f,
        var whiteTeeth: Float = 0f,
        var removeDarkCircles: Float = 0f,
        var removeNasolabial: Float = 0f,
        var removeWrinkle: Float = 0f,
    ) {
        val hasAny: Boolean get() =
            brightEye > 0 || whiteTeeth > 0 || removeDarkCircles > 0 ||
            removeNasolabial > 0 || removeWrinkle > 0
    }

    companion object {
        private const val TAG = "AdvancedBeautyFilter"
        private const val MAX_FACES = 5
        private const val REGIONS_PER_FACE = 10

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

        /**
         * GLSL uniform 数组布局：
         *  uRegions[50]: 50 个 vec2，展平为 vec2 数组
         *  uRadii: 5 个 float
         * Android OpenGL ES 3.0 的 uniform vec2 数组按 16B 对齐（std140），
         * 为避免对齐问题，我们把它拆分为 uRegions（uniform vec2[50]）——
         * GLSL 默认 vec2 数组按自然对齐（8B），但 WebGL/GLES 可能按 16B。
         * 最安全的方法是在 CPU 侧按 16B stride 打包，shader 里用 vec4 数组读。
         */
        private const val FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            in vec2 vTexCoord;
            out vec4 fragColor;

            uniform sampler2D uTexture;
            uniform vec2 uTextureSize;
            uniform float uBrightEye;
            uniform float uWhiteTeeth;
            uniform float uRemoveDarkCircles;
            uniform float uRemoveNasolabial;
            uniform float uRemoveWrinkle;
            uniform int uFaceCount;
            uniform float uFaceRadius[5];
            // 50 个 vec2 区域点，GLES3 下 vec2 uniform 数组按 vec4 对齐，所以每个 vec2 占 16B
            // 在 CPU 侧填 vec4(x, y, 0, 0)
            uniform vec4 uRegions[50];

            float falloff(float d, float r) {
                float t = d / max(r, 0.001);
                return exp(-t * t * 2.5);
            }

            vec3 rgb2hsl(vec3 c) {
                float mx = max(max(c.r, c.g), c.b);
                float mn = min(min(c.r, c.g), c.b);
                float l = (mx + mn) * 0.5;
                float h = 0.0, s = 0.0;
                if (mx != mn) {
                    float d = mx - mn;
                    s = l > 0.5 ? d / (2.0 - mx - mn) : d / (mx + mn);
                    if (mx == c.r)      h = (c.g - c.b) / d + (c.g < c.b ? 6.0 : 0.0);
                    else if (mx == c.g) h = (c.b - c.r) / d + 2.0;
                    else                h = (c.r - c.g) / d + 4.0;
                    h /= 6.0;
                }
                return vec3(h, s, l);
            }
            float hue2rgb(float p, float q, float t) {
                if (t < 0.0) t += 1.0;
                if (t > 1.0) t -= 1.0;
                if (t < 1.0/6.0) return p + (q - p) * 6.0 * t;
                if (t < 1.0/2.0) return q;
                if (t < 2.0/3.0) return p + (q - p) * (2.0/3.0 - t) * 6.0;
                return p;
            }
            vec3 hsl2rgb(vec3 hsl) {
                float h = hsl.x, s = hsl.y, l = hsl.z;
                if (s < 0.0001) return vec3(l);
                float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
                float p = 2.0 * l - q;
                return vec3(hue2rgb(p, q, h + 1.0/3.0), hue2rgb(p, q, h), hue2rgb(p, q, h - 1.0/3.0));
            }

            bool isTeethPixel(vec3 rgb, vec3 hsl) {
                return hsl.z > 0.4 && hsl.y < 0.3 && rgb.r > rgb.b - 0.1;
            }
            bool isEyeballPixel(vec3 hsl) {
                return hsl.y < 0.5;
            }

            float getRadius(int i) {
                if (i == 0) return uFaceRadius[0];
                if (i == 1) return uFaceRadius[1];
                if (i == 2) return uFaceRadius[2];
                if (i == 3) return uFaceRadius[3];
                return uFaceRadius[4];
            }

            vec2 getRegion(int idx) { return uRegions[idx].xy; }

            void main() {
                vec2 uv = vTexCoord;
                vec4 src = texture(uTexture, uv);
                vec3 rgb = src.rgb;
                vec3 hsl = rgb2hsl(rgb);

                float eyeMask = 0.0, teethMask = 0.0, darkCircleMask = 0.0;
                float nasoMask = 0.0, wrinkleMask = 0.0;

                for (int i = 0; i < uFaceCount && i < 5; i++) {
                    float r = getRadius(i);
                    vec2 leftEye        = getRegion(i * 10 + 0);
                    vec2 rightEye       = getRegion(i * 10 + 1);
                    vec2 mouth          = getRegion(i * 10 + 2);
                    vec2 leftUnder      = getRegion(i * 10 + 3);
                    vec2 rightUnder     = getRegion(i * 10 + 4);
                    vec2 leftNostril    = getRegion(i * 10 + 5);
                    vec2 rightNostril   = getRegion(i * 10 + 6);
                    vec2 leftMouthCorn  = getRegion(i * 10 + 7);
                    vec2 rightMouthCorn = getRegion(i * 10 + 8);
                    vec2 faceCenter     = getRegion(i * 10 + 9);

                    float eyeR   = r * 0.06;
                    float mouthR = r * 0.08;
                    float cheekR = r * 0.05;
                    float faceR  = r * 0.5;

                    eyeMask = max(eyeMask, max(falloff(length(uv - leftEye), eyeR), falloff(length(uv - rightEye), eyeR)));
                    teethMask = max(teethMask, falloff(length(uv - mouth), mouthR));
                    darkCircleMask = max(darkCircleMask, max(falloff(length(uv - leftUnder), eyeR), falloff(length(uv - rightUnder), eyeR)));

                    vec2 aL = leftNostril, bL = leftMouthCorn;
                    vec2 dirL = bL - aL;
                    float tL = clamp(dot(uv - aL, dirL) / max(dot(dirL, dirL), 1e-6), 0.0, 1.0);
                    float nasoL = falloff(length(uv - (aL + dirL * tL)), cheekR);

                    vec2 aR = rightNostril, bR = rightMouthCorn;
                    vec2 dirR = bR - aR;
                    float tR = clamp(dot(uv - aR, dirR) / max(dot(dirR, dirR), 1e-6), 0.0, 1.0);
                    float nasoR = falloff(length(uv - (aR + dirR * tR)), cheekR);

                    nasoMask = max(nasoMask, max(nasoL, nasoR));
                    wrinkleMask = max(wrinkleMask, falloff(length(uv - faceCenter), faceR));
                }

                if (uBrightEye > 0.0 && eyeMask > 0.0 && isEyeballPixel(hsl)) {
                    float k = uBrightEye * eyeMask * 0.5;
                    hsl.z = min(hsl.z + k * 0.25, 1.0);
                }
                if (uWhiteTeeth > 0.0 && teethMask > 0.0 && isTeethPixel(rgb, hsl)) {
                    float k = uWhiteTeeth * teethMask;
                    hsl.y = max(hsl.y - k * 0.4, 0.0);
                    hsl.z = min(hsl.z + k * 0.15, 1.0);
                }
                if (uRemoveDarkCircles > 0.0 && darkCircleMask > 0.0) {
                    float k = uRemoveDarkCircles * darkCircleMask;
                    float darkness = 1.0 - hsl.z;
                    if (darkness > 0.3) {
                        hsl.z = min(hsl.z + k * 0.25 * darkness, 1.0);
                        hsl.y = max(hsl.y - k * 0.2, 0.0);
                    }
                }

                float smoothWeight = 0.0;
                if (uRemoveNasolabial > 0.0 && nasoMask > 0.0)
                    smoothWeight = max(smoothWeight, uRemoveNasolabial * nasoMask);
                if (uRemoveWrinkle > 0.0 && wrinkleMask > 0.0)
                    smoothWeight = max(smoothWeight, uRemoveWrinkle * wrinkleMask * 0.6);

                vec3 finalRgb = hsl2rgb(hsl);
                if (smoothWeight > 0.0) {
                    vec2 px = 1.0 / uTextureSize;
                    vec3 acc = vec3(0.0);
                    int count = 0;
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            acc += texture(uTexture, uv + vec2(float(dx), float(dy)) * px).rgb;
                            count++;
                        }
                    }
                    vec3 smoothed = acc / float(count);
                    finalRgb = mix(finalRgb, smoothed, clamp(smoothWeight, 0.0, 0.8));
                }

                fragColor = vec4(finalRgb, src.a);
            }
        """

        private val QUAD_VERTICES = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
        private val QUAD_TEX_COORDS = floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f)
    }

    private var program = 0
    private var fbo = 0
    private var fboTexture = 0
    private var textureWidth = 0
    private var textureHeight = 0
    private var params = Params()
    private var vertexBuffer: FloatBuffer? = null
    private var texCoordBuffer: FloatBuffer? = null

    // 缓存 uniform 数组（50 个 vec4 = 200 float）
    private val regionArray = FloatArray(200)
    private val radiusArray = FloatArray(5)

    fun initialize(): Boolean {
        return try {
            program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
            if (program == 0) return false
            vertexBuffer = buf(QUAD_VERTICES)
            texCoordBuffer = buf(QUAD_TEX_COORDS)
            true
        } catch (e: Exception) {
            Log.e(TAG, "initialize failed: ${e.message}")
            false
        }
    }

    fun setParams(p: Params) { this.params = p }
    fun clear() { this.params = Params() }
    val isActive: Boolean get() = params.hasAny

    fun apply(inputTexture: Int, landmarks: List<FaceLandmarks>, imageWidth: Int, imageHeight: Int): Int {
        if (!isActive || landmarks.isEmpty()) return inputTexture
        ensureFbo(inputTexture)

        // 填充 uniform 数据
        java.util.Arrays.fill(regionArray, 0f)
        java.util.Arrays.fill(radiusArray, 0f)
        val faceCount = landmarks.size.coerceAtMost(MAX_FACES)
        val imgW = imageWidth.toFloat().coerceAtLeast(1f)
        val imgH = imageHeight.toFloat().coerceAtLeast(1f)

        fun nx(i: Int, p: List<android.graphics.PointF>): Float =
            if (i < p.size) p[i].x / imgW else 0f
        fun ny(i: Int, p: List<android.graphics.PointF>): Float =
            if (i < p.size) p[i].y / imgH else 0f

        fun setRegion(faceIdx: Int, regionIdx: Int, x: Float, y: Float) {
            val base = (faceIdx * 10 + regionIdx) * 4
            regionArray[base + 0] = x
            regionArray[base + 1] = y
        }

        for (i in 0 until faceCount) {
            val p = landmarks[i].points
            setRegion(i, 0, (nx(60, p) + nx(64, p)) / 2f, (ny(60, p) + ny(64, p)) / 2f)
            setRegion(i, 1, (nx(68, p) + nx(72, p)) / 2f, (ny(68, p) + ny(72, p)) / 2f)
            setRegion(i, 2, (nx(85, p) + nx(86, p)) / 2f, (ny(85, p) + ny(86, p)) / 2f)
            setRegion(i, 3, (nx(64, p) + nx(67, p)) / 2f, (ny(64, p) + ny(67, p)) / 2f)
            setRegion(i, 4, (nx(72, p) + nx(75, p)) / 2f, (ny(72, p) + ny(75, p)) / 2f)
            setRegion(i, 5, nx(55, p), ny(55, p))
            setRegion(i, 6, nx(59, p), ny(59, p))
            setRegion(i, 7, nx(76, p), ny(76, p))
            setRegion(i, 8, nx(82, p), ny(82, p))
            setRegion(i, 9, nx(54, p), ny(54, p))
            radiusArray[i] = landmarks[i].boundingBox.width() / imgW
        }

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
        GLES30.glViewport(0, 0, textureWidth, textureHeight)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glUseProgram(program)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTexture)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uTexture"), 0)
        GLES30.glUniform2f(GLES30.glGetUniformLocation(program, "uTextureSize"),
            textureWidth.toFloat(), textureHeight.toFloat())
        GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "uBrightEye"), params.brightEye)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "uWhiteTeeth"), params.whiteTeeth)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "uRemoveDarkCircles"), params.removeDarkCircles)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "uRemoveNasolabial"), params.removeNasolabial)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "uRemoveWrinkle"), params.removeWrinkle)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uFaceCount"), faceCount)
        GLES30.glUniform1fv(GLES30.glGetUniformLocation(program, "uFaceRadius"), 5, radiusArray, 0)
        GLES30.glUniform4fv(GLES30.glGetUniformLocation(program, "uRegions"), 50, regionArray, 0)

        drawQuad()

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        return fboTexture
    }

    fun release() {
        if (program != 0) { GLES30.glDeleteProgram(program); program = 0 }
        if (fbo != 0) { GLES30.glDeleteFramebuffers(1, intArrayOf(fbo), 0); fbo = 0 }
        if (fboTexture != 0) { GLES30.glDeleteTextures(1, intArrayOf(fboTexture), 0); fboTexture = 0 }
        clear()
    }

    private fun ensureFbo(inputTexture: Int) {
        if (textureWidth == 0 || textureHeight == 0) {
            val w = IntArray(1); val h = IntArray(1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTexture)
            GLES30.glGetTexLevelParameteriv(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_TEXTURE_WIDTH, w, 0)
            GLES30.glGetTexLevelParameteriv(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_TEXTURE_HEIGHT, h, 0)
            textureWidth = if (w[0] > 0) w[0] else 1280
            textureHeight = if (h[0] > 0) h[0] else 720
        }
        if (fbo == 0) {
            val fboIds = IntArray(1); val texIds = IntArray(1)
            GLES30.glGenFramebuffers(1, fboIds, 0)
            GLES30.glGenTextures(1, texIds, 0)
            fbo = fboIds[0]; fboTexture = texIds[0]
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fboTexture)
            GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA, textureWidth, textureHeight, 0,
                GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
            GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D, fboTexture, 0)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        }
    }

    private fun drawQuad() {
        val vBuf = vertexBuffer ?: return
        val tBuf = texCoordBuffer ?: return
        vBuf.position(0); GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 0, vBuf)
        tBuf.position(0); GLES30.glEnableVertexAttribArray(1)
        GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, 0, tBuf)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        GLES30.glDisableVertexAttribArray(0)
        GLES30.glDisableVertexAttribArray(1)
    }

    private fun createProgram(vSrc: String, fSrc: String): Int {
        val v = compile(GLES30.GL_VERTEX_SHADER, vSrc); if (v == 0) return 0
        val f = compile(GLES30.GL_FRAGMENT_SHADER, fSrc); if (f == 0) { GLES30.glDeleteShader(v); return 0 }
        val p = GLES30.glCreateProgram()
        GLES30.glAttachShader(p, v); GLES30.glAttachShader(p, f); GLES30.glLinkProgram(p)
        val st = IntArray(1); GLES30.glGetProgramiv(p, GLES30.GL_LINK_STATUS, st, 0)
        GLES30.glDeleteShader(v); GLES30.glDeleteShader(f)
        if (st[0] == 0) {
            Log.e(TAG, "link failed: ${GLES30.glGetProgramInfoLog(p)}"); GLES30.glDeleteProgram(p); return 0
        }
        return p
    }

    private fun compile(type: Int, src: String): Int {
        val s = GLES30.glCreateShader(type)
        GLES30.glShaderSource(s, src); GLES30.glCompileShader(s)
        val st = IntArray(1); GLES30.glGetShaderiv(s, GLES30.GL_COMPILE_STATUS, st, 0)
        if (st[0] == 0) { Log.e(TAG, "compile failed: ${GLES30.glGetShaderInfoLog(s)}"); GLES30.glDeleteShader(s); return 0 }
        return s
    }

    private fun buf(data: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder())
            .asFloatBuffer().apply { put(data); position(0) }
}
