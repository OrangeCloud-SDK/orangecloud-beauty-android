package com.orangecloud.beautysdk.filter

import android.opengl.GLES30
import android.util.Log
import com.orangecloud.beautysdk.models.FaceLandmarks
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * 美妆：口红 / 腮红 / 眉毛 / 眼影 / 眼线 / 睫毛 / 美瞳
 *
 * 基于 98 点 landmarks 定位区域 mask，与指定 color 混合。
 * 每张脸 42 个实用点（压到 64 个），最多 5 张脸 = 320 个 vec2 = 160 个 vec4。
 */
class MakeupFilter {

    data class Params(
        var lipstickIntensity: Float = 0f,  var lipstickColor: IntArray = intArrayOf(204, 51, 68),
        var blushIntensity: Float = 0f,     var blushColor: IntArray = intArrayOf(255, 136, 153),
        var eyebrowIntensity: Float = 0f,   var eyebrowColor: IntArray = intArrayOf(58, 40, 32),
        var eyeshadowIntensity: Float = 0f, var eyeshadowColor: IntArray = intArrayOf(136, 85, 170),
        var eyelinerIntensity: Float = 0f,  var eyelinerColor: IntArray = intArrayOf(16, 16, 16),
        var eyelashIntensity: Float = 0f,   var eyelashColor: IntArray = intArrayOf(16, 16, 16),
        var pupilIntensity: Float = 0f,     var pupilColor: IntArray = intArrayOf(107, 142, 90),
    ) {
        val hasAny: Boolean get() =
            lipstickIntensity > 0 || blushIntensity > 0 ||
            eyebrowIntensity > 0 || eyeshadowIntensity > 0 ||
            eyelinerIntensity > 0 || eyelashIntensity > 0 || pupilIntensity > 0
    }

    companion object {
        private const val TAG = "MakeupFilter"
        private const val MAX_FACES = 5
        private const val POINTS_PER_FACE = 64
        private const val TOTAL_POINTS = MAX_FACES * POINTS_PER_FACE

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

        // 每 2 个 vec2 打包为 1 个 vec4：uv0 = vec4(p0.xy, p1.xy)
        // 160 个 vec4 = 320 个 vec2
        private const val FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            in vec2 vTexCoord;
            out vec4 fragColor;

            uniform sampler2D uTexture;

            uniform float uLipstickI;   uniform vec3 uLipstickC;
            uniform float uBlushI;      uniform vec3 uBlushC;
            uniform float uEyebrowI;    uniform vec3 uEyebrowC;
            uniform float uEyeshadowI;  uniform vec3 uEyeshadowC;
            uniform float uEyelinerI;   uniform vec3 uEyelinerC;
            uniform float uEyelashI;    uniform vec3 uEyelashC;
            uniform float uPupilI;      uniform vec3 uPupilC;

            uniform int uFaceCount;
            uniform float uFaceRadius[5];
            uniform vec4 uPtsPacked[160]; // 160 vec4 = 320 vec2

            vec2 getPt(int i) {
                int pk = i / 2;
                if (mod(float(i), 2.0) < 0.5) return uPtsPacked[pk].xy;
                return uPtsPacked[pk].zw;
            }

            float getRadius(int i) {
                if (i == 0) return uFaceRadius[0];
                if (i == 1) return uFaceRadius[1];
                if (i == 2) return uFaceRadius[2];
                if (i == 3) return uFaceRadius[3];
                return uFaceRadius[4];
            }

            vec3 softBlend(vec3 base, vec3 overlay, float a) {
                return mix(base, overlay, clamp(a, 0.0, 1.0));
            }

            float circleFalloff(vec2 p, vec2 c, float r) {
                float d = length(p - c);
                float t = d / max(r, 0.001);
                return exp(-t * t * 2.0);
            }

            float segmentFalloff(vec2 p, vec2 a, vec2 b, float r) {
                vec2 dir = b - a;
                float t = clamp(dot(p - a, dir) / max(dot(dir, dir), 1e-6), 0.0, 1.0);
                return circleFalloff(p, a + dir * t, r);
            }

            bool pointInLipPoly(vec2 p, int faceIdx) {
                int base = faceIdx * 64;
                bool inside = false;
                int prev = 11;
                for (int k = 0; k < 12; k++) {
                    vec2 a = getPt(base + k);
                    vec2 b = getPt(base + prev);
                    if (((a.y > p.y) != (b.y > p.y)) &&
                        (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y + 1e-8) + a.x)) {
                        inside = !inside;
                    }
                    prev = k;
                }
                return inside;
            }

            void main() {
                vec2 uv = vTexCoord;
                vec4 src = texture(uTexture, uv);
                vec3 rgb = src.rgb;

                for (int i = 0; i < uFaceCount && i < 5; i++) {
                    int base = i * 64;
                    float r = getRadius(i);

                    if (uLipstickI > 0.0 && pointInLipPoly(uv, i)) {
                        rgb = softBlend(rgb, uLipstickC, uLipstickI * 0.55);
                    }

                    if (uBlushI > 0.0) {
                        float k = max(circleFalloff(uv, getPt(base + 40), r * 0.06),
                                      circleFalloff(uv, getPt(base + 41), r * 0.06));
                        if (k > 0.0) rgb = softBlend(rgb, uBlushC, uBlushI * k * 0.5);
                    }

                    if (uEyebrowI > 0.0) {
                        float bestK = 0.0;
                        for (int j = 0; j < 4; j++) {
                            bestK = max(bestK, segmentFalloff(uv, getPt(base + 12 + j), getPt(base + 12 + j + 1), r * 0.008));
                            bestK = max(bestK, segmentFalloff(uv, getPt(base + 17 + j), getPt(base + 17 + j + 1), r * 0.008));
                        }
                        if (bestK > 0.0) rgb = softBlend(rgb, uEyebrowC, uEyebrowI * bestK * 0.7);
                    }

                    if (uEyeshadowI > 0.0) {
                        vec2 leftCenter  = (getPt(base + 22) + getPt(base + 26)) * 0.5;
                        vec2 rightCenter = (getPt(base + 30) + getPt(base + 34)) * 0.5;
                        vec2 leftShadow  = leftCenter  + vec2(0.0, -r * 0.03);
                        vec2 rightShadow = rightCenter + vec2(0.0, -r * 0.03);
                        float k = max(circleFalloff(uv, leftShadow, r * 0.035),
                                      circleFalloff(uv, rightShadow, r * 0.035));
                        if (k > 0.0 && uv.y < min(leftCenter.y, rightCenter.y)) {
                            rgb = softBlend(rgb, uEyeshadowC, uEyeshadowI * k * 0.5);
                        }
                    }

                    if (uEyelinerI > 0.0) {
                        float bestK = 0.0;
                        for (int j = 0; j < 7; j++) {
                            bestK = max(bestK, segmentFalloff(uv, getPt(base + 22 + j), getPt(base + 22 + j + 1), r * 0.003));
                            bestK = max(bestK, segmentFalloff(uv, getPt(base + 30 + j), getPt(base + 30 + j + 1), r * 0.003));
                        }
                        bestK = max(bestK, segmentFalloff(uv, getPt(base + 29), getPt(base + 22), r * 0.003));
                        bestK = max(bestK, segmentFalloff(uv, getPt(base + 37), getPt(base + 30), r * 0.003));
                        if (bestK > 0.0) rgb = softBlend(rgb, uEyelinerC, uEyelinerI * bestK);
                    }

                    if (uEyelashI > 0.0) {
                        float bestK = 0.0;
                        for (int j = 0; j < 2; j++) {
                            vec2 a  = getPt(base + 22 + j)     + vec2(0.0, -r * 0.005);
                            vec2 b  = getPt(base + 22 + j + 1) + vec2(0.0, -r * 0.005);
                            bestK = max(bestK, segmentFalloff(uv, a, b, r * 0.004));
                            vec2 a2 = getPt(base + 30 + j)     + vec2(0.0, -r * 0.005);
                            vec2 b2 = getPt(base + 30 + j + 1) + vec2(0.0, -r * 0.005);
                            bestK = max(bestK, segmentFalloff(uv, a2, b2, r * 0.004));
                        }
                        if (bestK > 0.0) rgb = softBlend(rgb, uEyelashC, uEyelashI * bestK * 0.8);
                    }

                    if (uPupilI > 0.0) {
                        float pR = r * 0.012;
                        float k = max(circleFalloff(uv, getPt(base + 38), pR),
                                      circleFalloff(uv, getPt(base + 39), pR));
                        if (k > 0.0) rgb = softBlend(rgb, uPupilC, uPupilI * k * 0.7);
                    }
                }

                fragColor = vec4(rgb, src.a);
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

    private val ptsPackedArray = FloatArray(160 * 4) // 160 vec4 → 640 float
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

        // 填充 points
        java.util.Arrays.fill(ptsPackedArray, 0f)
        java.util.Arrays.fill(radiusArray, 0f)
        val faceCount = landmarks.size.coerceAtMost(MAX_FACES)
        val imgW = imageWidth.toFloat().coerceAtLeast(1f)
        val imgH = imageHeight.toFloat().coerceAtLeast(1f)

        fun setPt(globalIdx: Int, x: Float, y: Float) {
            val pk = globalIdx / 2
            val slot = globalIdx % 2
            val base = pk * 4 + slot * 2
            ptsPackedArray[base + 0] = x
            ptsPackedArray[base + 1] = y
        }

        fun nx(i: Int, p: List<android.graphics.PointF>): Float = if (i < p.size) p[i].x / imgW else 0f
        fun ny(i: Int, p: List<android.graphics.PointF>): Float = if (i < p.size) p[i].y / imgH else 0f

        for (f in 0 until faceCount) {
            val p = landmarks[f].points
            val base = f * POINTS_PER_FACE
            // 外唇 12 点：76..87
            for (k in 0 until 12) setPt(base + k, nx(76 + k, p), ny(76 + k, p))
            // 左眉 5 点 33..37
            for (k in 0 until 5) setPt(base + 12 + k, nx(33 + k, p), ny(33 + k, p))
            // 右眉 5 点 42..46
            for (k in 0 until 5) setPt(base + 17 + k, nx(42 + k, p), ny(42 + k, p))
            // 左眼 8 点 60..67
            for (k in 0 until 8) setPt(base + 22 + k, nx(60 + k, p), ny(60 + k, p))
            // 右眼 8 点 68..75
            for (k in 0 until 8) setPt(base + 30 + k, nx(68 + k, p), ny(68 + k, p))
            // 左右瞳孔 = 眼部 8 点中心
            var lx = 0f; var ly = 0f; var rx = 0f; var ry = 0f
            for (k in 0 until 8) {
                val li = base + 22 + k
                val ri = base + 30 + k
                val pkL = li / 2; val slotL = li % 2
                val pkR = ri / 2; val slotR = ri % 2
                lx += ptsPackedArray[pkL * 4 + slotL * 2 + 0]
                ly += ptsPackedArray[pkL * 4 + slotL * 2 + 1]
                rx += ptsPackedArray[pkR * 4 + slotR * 2 + 0]
                ry += ptsPackedArray[pkR * 4 + slotR * 2 + 1]
            }
            setPt(base + 38, lx / 8f, ly / 8f)
            setPt(base + 39, rx / 8f, ry / 8f)
            // 左右脸颊中心：瞳孔下方
            val faceHNorm = landmarks[f].boundingBox.height() / imgH
            setPt(base + 40, lx / 8f, ly / 8f + faceHNorm * 0.18f)
            setPt(base + 41, rx / 8f, ry / 8f + faceHNorm * 0.18f)

            radiusArray[f] = landmarks[f].boundingBox.width() / imgW
        }

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
        GLES30.glViewport(0, 0, textureWidth, textureHeight)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glUseProgram(program)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTexture)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uTexture"), 0)

        setU1f("uLipstickI", params.lipstickIntensity);  setU3f("uLipstickC", params.lipstickColor)
        setU1f("uBlushI", params.blushIntensity);         setU3f("uBlushC", params.blushColor)
        setU1f("uEyebrowI", params.eyebrowIntensity);     setU3f("uEyebrowC", params.eyebrowColor)
        setU1f("uEyeshadowI", params.eyeshadowIntensity); setU3f("uEyeshadowC", params.eyeshadowColor)
        setU1f("uEyelinerI", params.eyelinerIntensity);   setU3f("uEyelinerC", params.eyelinerColor)
        setU1f("uEyelashI", params.eyelashIntensity);     setU3f("uEyelashC", params.eyelashColor)
        setU1f("uPupilI", params.pupilIntensity);         setU3f("uPupilC", params.pupilColor)

        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uFaceCount"), faceCount)
        GLES30.glUniform1fv(GLES30.glGetUniformLocation(program, "uFaceRadius"), 5, radiusArray, 0)
        GLES30.glUniform4fv(GLES30.glGetUniformLocation(program, "uPtsPacked"), 160, ptsPackedArray, 0)

        drawQuad()

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        return fboTexture
    }

    private fun setU1f(name: String, v: Float) {
        GLES30.glUniform1f(GLES30.glGetUniformLocation(program, name), v)
    }

    private fun setU3f(name: String, rgb: IntArray) {
        val r = (rgb.getOrElse(0) { 0 }.coerceIn(0, 255)) / 255f
        val g = (rgb.getOrElse(1) { 0 }.coerceIn(0, 255)) / 255f
        val b = (rgb.getOrElse(2) { 0 }.coerceIn(0, 255)) / 255f
        GLES30.glUniform3f(GLES30.glGetUniformLocation(program, name), r, g, b)
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
