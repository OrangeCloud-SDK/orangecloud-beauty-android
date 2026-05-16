package com.orangecloud.beautysdk.filter

import android.opengl.GLES30
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * 哈哈镜扭曲滤镜 - OpenGL ES 3.0 实现
 * 支持 10 种扭曲类型（与 iOS DistortionType 对齐）。
 */
class DistortionFilter {

    enum class Type(val code: Int) {
        NONE(0),
        SPHERE_BULGE(1),
        SPHERE_SQUEEZE(2),
        HORIZONTAL_STRETCH(3),
        VERTICAL_STRETCH(4),
        WAVE_HORIZONTAL(5),
        WAVE_VERTICAL(6),
        SWIRL(7),
        FISHEYE(8),
        CHROMATIC(9),
        PIXELATE(10);

        companion object {
            fun fromName(name: String): Type = when (name) {
                "none" -> NONE
                "sphereBulge" -> SPHERE_BULGE
                "sphereSqueeze" -> SPHERE_SQUEEZE
                "horizontalStretch" -> HORIZONTAL_STRETCH
                "verticalStretch" -> VERTICAL_STRETCH
                "waveHorizontal" -> WAVE_HORIZONTAL
                "waveVertical" -> WAVE_VERTICAL
                "swirl" -> SWIRL
                "fisheye" -> FISHEYE
                "chromatic" -> CHROMATIC
                "pixelate" -> PIXELATE
                else -> NONE
            }
        }
    }

    companion object {
        private const val TAG = "DistortionFilter"

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

        private const val FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;
            in vec2 vTexCoord;
            out vec4 fragColor;
            uniform sampler2D uTexture;
            uniform int uType;
            uniform float uIntensity;
            uniform int uFrame;

            vec2 sphereBulge(vec2 uv, float k) {
                vec2 c = vec2(0.5);
                vec2 d = uv - c;
                float r = length(d);
                if (r == 0.0) return uv;
                float f = 1.0 - k * (1.0 - r);
                return c + d * f;
            }
            vec2 sphereSqueeze(vec2 uv, float k) {
                vec2 c = vec2(0.5);
                vec2 d = uv - c;
                float r = length(d);
                return c + d * (1.0 + k * (1.0 - r));
            }
            vec2 horizontalStretch(vec2 uv, float k) {
                vec2 c = vec2(0.5);
                uv.x = c.x + (uv.x - c.x) / (1.0 + k * 0.6);
                return uv;
            }
            vec2 verticalStretch(vec2 uv, float k) {
                vec2 c = vec2(0.5);
                uv.y = c.y + (uv.y - c.y) / (1.0 + k * 0.6);
                return uv;
            }
            vec2 waveHorizontal(vec2 uv, float k, int frame) {
                float t = float(frame) * 0.05;
                uv.x += sin(uv.y * 20.0 + t) * 0.03 * k;
                return uv;
            }
            vec2 waveVertical(vec2 uv, float k, int frame) {
                float t = float(frame) * 0.05;
                uv.y += sin(uv.x * 20.0 + t) * 0.03 * k;
                return uv;
            }
            vec2 swirl(vec2 uv, float k) {
                vec2 c = vec2(0.5);
                vec2 d = uv - c;
                float r = length(d);
                float theta = atan(d.y, d.x);
                theta += (1.0 - r) * 3.14159 * k;
                return c + vec2(cos(theta), sin(theta)) * r;
            }
            vec2 fisheye(vec2 uv, float k) {
                vec2 c = vec2(0.5);
                vec2 d = uv - c;
                float r2 = dot(d, d);
                return c + d * (1.0 + k * r2);
            }

            void main() {
                vec2 uv = vTexCoord;
                float k = clamp(uIntensity, 0.0, 1.0);
                vec2 srcUV = uv;
                if (uType == 1) srcUV = sphereBulge(uv, k);
                else if (uType == 2) srcUV = sphereSqueeze(uv, k);
                else if (uType == 3) srcUV = horizontalStretch(uv, k);
                else if (uType == 4) srcUV = verticalStretch(uv, k);
                else if (uType == 5) srcUV = waveHorizontal(uv, k, uFrame);
                else if (uType == 6) srcUV = waveVertical(uv, k, uFrame);
                else if (uType == 7) srcUV = swirl(uv, k);
                else if (uType == 8) srcUV = fisheye(uv, k);
                else if (uType == 10) {
                    float blockSize = max(1.0, k * 40.0);
                    vec2 size = vec2(textureSize(uTexture, 0));
                    vec2 px = uv * size;
                    px = floor(px / blockSize) * blockSize + blockSize * 0.5;
                    srcUV = px / size;
                }

                vec4 color;
                if (uType == 9) {
                    float offset = k * 0.01;
                    float r = texture(uTexture, vec2(uv.x + offset, uv.y)).r;
                    float g = texture(uTexture, uv).g;
                    float b = texture(uTexture, vec2(uv.x - offset, uv.y)).b;
                    color = vec4(r, g, b, 1.0);
                } else {
                    color = texture(uTexture, srcUV);
                }
                fragColor = color;
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
    private var type: Type = Type.NONE
    private var intensity: Float = 0f
    private var frameCounter: Int = 0

    private var vertexBuffer: FloatBuffer? = null
    private var texCoordBuffer: FloatBuffer? = null

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

    fun set(type: Type, intensity: Float) {
        this.type = type
        this.intensity = intensity.coerceIn(0f, 1f)
    }

    fun clear() {
        type = Type.NONE
        intensity = 0f
    }

    val isActive: Boolean get() = type != Type.NONE && intensity > 0f

    fun apply(inputTexture: Int): Int {
        if (!isActive) return inputTexture
        ensureFbo(inputTexture)

        frameCounter++

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
        GLES30.glViewport(0, 0, textureWidth, textureHeight)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        GLES30.glUseProgram(program)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTexture)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uTexture"), 0)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uType"), type.code)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(program, "uIntensity"), intensity)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uFrame"), frameCounter)

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
        vBuf.position(0)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 0, vBuf)
        tBuf.position(0)
        GLES30.glEnableVertexAttribArray(1)
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
