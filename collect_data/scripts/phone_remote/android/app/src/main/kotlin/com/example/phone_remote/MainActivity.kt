package com.example.phone_remote

import android.os.Bundle
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

/**
 * MainActivity - Flutter入口Activity
 *
 * 注册ARCore位姿获取的MethodChannel，
 * 供Flutter端通过原生层获取ARCore相机6DoF位姿数据。
 */
class MainActivity : FlutterActivity() {
    // MethodChannel名称（与Flutter端arcore_service.dart中的定义一致）
    private val ARCORE_POSE_CHANNEL = "com.example.phone_remote/arcore_pose"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // 注册ARCore位姿MethodChannel
        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            ARCORE_POSE_CHANNEL
        ).setMethodCallHandler { call, result ->
            when (call.method) {
                "getCameraPose" -> {
                    // TODO: 集成ARCore Session后，从Frame中提取Camera Pose
                    // 当前返回默认值（零位姿），待ARCore集成完成后替换为真实数据
                    val poseMap = mapOf(
                        "tx" to 0.0,
                        "ty" to 0.0,
                        "tz" to 0.0,
                        "qw" to 1.0,
                        "qx" to 0.0,
                        "qy" to 0.0,
                        "qz" to 0.0
                    )
                    result.success(poseMap)
                }
                else -> result.notImplemented()
            }
        }
    }
}
