package com.orangecloud.beautysdk.models

import android.graphics.PointF
import android.graphics.RectF

/**
 * 人脸关键点检测结果
 * @param faceId 人脸唯一标识
 * @param points 98 个关键点坐标
 * @param boundingBox 人脸边界框
 * @param confidence 检测置信度 (0.0 ~ 1.0)
 */
data class FaceLandmarks(
    val faceId: Int,
    val points: List<PointF>,
    val boundingBox: RectF,
    val confidence: Float
)

/**
 * 美颜参数
 * @param smoothingIntensity 磨皮强度 (0.0 ~ 1.0)
 * @param whiteningIntensity 美白强度 (0.0 ~ 1.0)
 * @param slimFaceIntensity 瘦脸强度 (0.0 ~ 1.0)
 * @param enlargeEyeIntensity 大眼强度 (0.0 ~ 1.0)
 * 以下为扩展美型参数（对标腾讯美颜 SDK）
 */
data class BeautyParams(
    val smoothingIntensity: Float = 0f,
    val whiteningIntensity: Float = 0f,
    val slimFaceIntensity: Float = 0f,
    val enlargeEyeIntensity: Float = 0f,
    val slimChinIntensity: Float = 0f,
    val slimNoseIntensity: Float = 0f,
    val mouthShapeIntensity: Float = 0f,
    val foreheadIntensity: Float = 0f,
    val hairlineIntensity: Float = 0f,
    val slimCheekboneIntensity: Float = 0f,
    val eyebrowShapeIntensity: Float = 0f,
    val vShapeIntensity: Float = 0f,
    val jawboneIntensity: Float = 0f,
) {
    /** 是否存在任何美型强度（> 0） */
    val hasAnyDeformation: Boolean
        get() = slimFaceIntensity > 0f || enlargeEyeIntensity > 0f ||
                slimChinIntensity > 0f || slimNoseIntensity > 0f ||
                mouthShapeIntensity > 0f || foreheadIntensity > 0f ||
                hairlineIntensity > 0f || slimCheekboneIntensity > 0f ||
                eyebrowShapeIntensity > 0f || vShapeIntensity > 0f ||
                jawboneIntensity > 0f
}
