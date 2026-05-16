package com.orangecloud.beautysdk.perf

import android.util.Log

/**
 * 管线性能监控 + 自动降级。
 * 每秒聚合一次数据，通过 [onStatsUpdated] 回调上报。
 */
class PerfMonitor {

    enum class DegradationLevel {
        NONE, LIGHT, MEDIUM, HEAVY;

        val key: String get() = name.lowercase()

        companion object {
            fun fromKey(k: String): DegradationLevel = when (k) {
                "light" -> LIGHT
                "medium" -> MEDIUM
                "heavy" -> HEAVY
                else -> NONE
            }
        }
    }

    data class Sample(
        var fps: Double = 0.0,
        var avgFrameTimeMs: Double = 0.0,
        var p95FrameTimeMs: Double = 0.0,
        var droppedFrames: Int = 0,
        var stageAvgMs: Map<String, Double> = emptyMap(),
        var degradation: DegradationLevel = DegradationLevel.NONE,
    )

    companion object { private const val TAG = "PerfMonitor" }

    @Volatile var targetFps: Int = 30
    @Volatile var autoDegradation: Boolean = true
        @JvmName("setAutoDegradationValue") set

    var degradation: DegradationLevel = DegradationLevel.NONE
        private set

    /** 回调：每秒一次 */
    @Volatile var onStatsUpdated: ((Sample) -> Unit)? = null

    private var forcedLevel: DegradationLevel? = null
    private var frameStart: Long = 0
    private val stageStart = HashMap<String, Long>()
    private val frameTimes = ArrayList<Double>()
    private val stageSamples = HashMap<String, ArrayList<Double>>()
    private var droppedInWindow: Int = 0
    private var windowStart: Long = System.nanoTime()
    private var consecutiveBadSeconds: Int = 0

    fun isStageEnabled(stage: String): Boolean = when (degradation) {
        DegradationLevel.NONE -> true
        DegradationLevel.LIGHT ->
            stage != "distortionFilter" && stage != "advancedBeauty"
        DegradationLevel.MEDIUM ->
            stage != "distortionFilter" && stage != "advancedBeauty" &&
            stage != "lutFilter" && stage != "makeupFilter"
        DegradationLevel.HEAVY ->
            stage == "faceDetector" || stage == "beautyFilter"
    }

    fun start() { reset() }
    fun stop()  { reset() }

    fun setForcedLevel(level: DegradationLevel?) {
        forcedLevel = level
        degradation = level ?: DegradationLevel.NONE
    }

    fun beginFrame() {
        frameStart = System.nanoTime()
    }

    fun endFrame() {
        val now = System.nanoTime()
        val elapsedMs = (now - frameStart) / 1_000_000.0
        frameTimes.add(elapsedMs)
        val frameBudgetMs = 1000.0 / maxOf(targetFps, 1)
        if (elapsedMs > frameBudgetMs * 1.2) droppedInWindow++

        if ((now - windowStart) >= 1_000_000_000L) {
            val sample = buildSample(now)
            if (forcedLevel == null && autoDegradation) {
                updateAutoDegradation(sample)
            }
            onStatsUpdated?.invoke(sample)
            resetWindow(now)
        }
    }

    fun beginStage(name: String) {
        stageStart[name] = System.nanoTime()
    }

    fun endStage(name: String) {
        val s = stageStart[name] ?: return
        val elapsedMs = (System.nanoTime() - s) / 1_000_000.0
        stageSamples.getOrPut(name) { ArrayList() }.add(elapsedMs)
    }

    private fun buildSample(now: Long): Sample {
        val sample = Sample()
        val count = frameTimes.size
        if (count > 0) {
            val total = frameTimes.sum()
            sample.avgFrameTimeMs = total / count
            val sorted = frameTimes.sorted()
            val idx = (count * 0.95).toInt().coerceAtMost(count - 1)
            sample.p95FrameTimeMs = sorted[idx]
            val elapsedSec = (now - windowStart) / 1_000_000_000.0
            sample.fps = if (elapsedSec > 0) count / elapsedSec else 0.0
        }
        sample.droppedFrames = droppedInWindow
        val stageAvg = HashMap<String, Double>()
        for ((name, values) in stageSamples) {
            if (values.isEmpty()) continue
            stageAvg[name] = values.sum() / values.size
        }
        sample.stageAvgMs = stageAvg
        sample.degradation = degradation
        return sample
    }

    private fun updateAutoDegradation(sample: Sample) {
        val targetMs = 1000.0 / maxOf(targetFps, 1)
        val bad = sample.p95FrameTimeMs > targetMs * 1.1 ||
                  sample.fps < targetFps * 0.85
        val good = sample.p95FrameTimeMs < targetMs * 0.8 &&
                   sample.fps > targetFps * 0.95
        when {
            bad -> {
                if (consecutiveBadSeconds < 0) consecutiveBadSeconds = 0
                consecutiveBadSeconds++
                if (consecutiveBadSeconds >= 3) {
                    escalate()
                    consecutiveBadSeconds = 0
                }
            }
            good -> {
                if (consecutiveBadSeconds > 0) consecutiveBadSeconds = 0
                consecutiveBadSeconds--
                if (consecutiveBadSeconds <= -5) {
                    deescalate()
                    consecutiveBadSeconds = 0
                }
            }
            else -> { /* 中间区不动 */ }
        }
    }

    private fun escalate() {
        degradation = when (degradation) {
            DegradationLevel.NONE -> DegradationLevel.LIGHT
            DegradationLevel.LIGHT -> DegradationLevel.MEDIUM
            DegradationLevel.MEDIUM -> DegradationLevel.HEAVY
            DegradationLevel.HEAVY -> degradation
        }
        Log.i(TAG, "Auto degradation escalated → $degradation")
    }

    private fun deescalate() {
        degradation = when (degradation) {
            DegradationLevel.HEAVY -> DegradationLevel.MEDIUM
            DegradationLevel.MEDIUM -> DegradationLevel.LIGHT
            DegradationLevel.LIGHT -> DegradationLevel.NONE
            DegradationLevel.NONE -> degradation
        }
        Log.i(TAG, "Auto degradation de-escalated → $degradation")
    }

    private fun resetWindow(now: Long) {
        frameTimes.clear()
        stageSamples.clear()
        droppedInWindow = 0
        windowStart = now
    }

    private fun reset() {
        frameTimes.clear()
        stageSamples.clear()
        droppedInWindow = 0
        consecutiveBadSeconds = 0
        windowStart = System.nanoTime()
    }
}
