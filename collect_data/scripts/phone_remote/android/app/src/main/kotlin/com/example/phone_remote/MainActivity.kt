package com.example.phone_remote

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.opengl.EGL14
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.util.Log
import android.view.Surface
import android.view.View
import android.widget.FrameLayout
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.platform.PlatformView
import io.flutter.plugin.platform.PlatformViewFactory
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicReference
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class MainActivity : FlutterActivity() {
    companion object {
        private const val TAG = "MainActivity"
        private const val ARCORE_POSE_CHANNEL = "com.example.phone_remote/arcore_pose"
        private const val ARCORE_VIEW_TYPE = "arcore-view"
        private const val CAMERA_PERMISSION_REQUEST_CODE = 1001
    }

    private var arSession: Session? = null
    private var installRequested = false
    private var arView: ARCorePlatformView? = null
    private var cameraTextureId = -1
    private var needSessionResume = false
    private var backgroundTrackingThread: Thread? = null
    private var backgroundTrackingRunning = false
    private val sessionLock = Object()

    private var hasGLSurfaceView = false

    private val latestCameraPose = AtomicReference<Map<String, Double>>(mapOf(
        "tx" to 0.0, "ty" to 0.0, "tz" to 0.0,
        "qw" to 1.0, "qx" to 0.0, "qy" to 0.0, "qz" to 0.0,
        "tracking" to 0.0
    ))

    // ARCore初始化状态
    enum class ARState { NOT_INITIALIZED, INSTALLING, INITIALIZED, FAILED }
    private var arState = ARState.NOT_INITIALIZED

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // 注册ARCore位姿MethodChannel
        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            ARCORE_POSE_CHANNEL
        ).setMethodCallHandler { call, result ->
            when (call.method) {
                "getCameraPose" -> {
                    result.success(latestCameraPose.get())
                }
                "isARSupported" -> {
                    val availability = ArCoreApk.getInstance().checkAvailability(this@MainActivity)
                    val supported = availability.isSupported || availability.isUnknown
                    result.success(supported)
                }
                "requestInstall" -> {
                    // 标记需要安装，下次onResume时处理
                    needSessionResume = true
                    result.success(true)
                }
                else -> result.notImplemented()
            }
        }

        // 注册ARCore PlatformView
        flutterEngine.platformViewsController.registry.registerViewFactory(
            ARCORE_VIEW_TYPE,
            ARCoreViewFactory(flutterEngine.dartExecutor.binaryMessenger)
        )
    }

    override fun onResume() {
        super.onResume()

        // 检查相机权限
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST_CODE)
            return
        }

        // 尝试创建或恢复ARCore Session
        ensureARCoreSession()
    }

    override fun onPause() {
        super.onPause()
        stopBackgroundTracking()
        hasGLSurfaceView = false
        try {
            arSession?.pause()
        } catch (_: Exception) {}
    }

    override fun onDestroy() {
        super.onDestroy()
        stopBackgroundTracking()
        hasGLSurfaceView = false
        try {
            arSession?.pause()
            arSession?.close()
        } catch (_: Exception) {}
        arSession = null
        arState = ARState.NOT_INITIALIZED
    }

    @Deprecated("Deprecated in Java")
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                ensureARCoreSession()
            } else {
                arState = ARState.FAILED
                Log.e(TAG, "相机权限被拒绝")
            }
        }
    }

    /**
     * 确保ARCore Session已创建并恢复
     * 必须在onResume中调用
     *
     * 策略：先直接尝试创建Session（ARCore已安装的情况），
     * 失败后再尝试requestInstall（需要安装ARCore的情况）
     */
    private fun ensureARCoreSession() {
        // 如果已有Session，直接resume
        if (arSession != null) {
            try {
                arSession!!.resume()
                arState = ARState.INITIALIZED
                return
            } catch (e: CameraNotAvailableException) {
                Log.e(TAG, "相机不可用: ${e.message}")
                arState = ARState.FAILED
                return
            }
        }

        // 第一步：直接尝试创建Session（适用于ARCore已安装的设备，包括中国区手机）
        try {
            arSession = Session(this)
            configureAndResumeSession()
            Log.i(TAG, "ARCore Session直接创建成功")
            return
        } catch (e: UnavailableDeviceNotCompatibleException) {
            Log.w(TAG, "设备不兼容ARCore: ${e.message}")
            arState = ARState.FAILED
            return
        } catch (e: UnavailableApkTooOldException) {
            Log.w(TAG, "ARCore APK版本过旧，尝试更新: ${e.message}")
            // 继续尝试requestInstall
        } catch (e: UnavailableSdkTooOldException) {
            Log.e(TAG, "ARCore SDK版本过旧: ${e.message}")
            arState = ARState.FAILED
            return
        } catch (e: Exception) {
            Log.w(TAG, "直接创建Session失败: ${e.message}，尝试requestInstall")
            // 继续尝试requestInstall
        }

        // 第二步：尝试通过requestInstall安装/更新ARCore
        try {
            val installStatus = ArCoreApk.getInstance().requestInstall(this, !installRequested)
            installRequested = true
            if (installStatus == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                arState = ARState.INSTALLING
                Log.i(TAG, "ARCore服务正在安装...")
                return
            }

            // 安装完成，再次尝试创建Session
            arSession = Session(this)
            configureAndResumeSession()
            Log.i(TAG, "ARCore Session安装后创建成功")

        } catch (e: UnavailableUserDeclinedInstallationException) {
            Log.e(TAG, "用户拒绝安装ARCore: ${e.message}")
            arState = ARState.FAILED
        } catch (e: UnavailableDeviceNotCompatibleException) {
            Log.e(TAG, "设备不兼容ARCore: ${e.message}")
            arState = ARState.FAILED
        } catch (e: UnavailableApkTooOldException) {
            Log.e(TAG, "ARCore APK版本过旧: ${e.message}")
            arState = ARState.FAILED
        } catch (e: UnavailableSdkTooOldException) {
            Log.e(TAG, "ARCore SDK版本过旧: ${e.message}")
            arState = ARState.FAILED
        } catch (e: CameraNotAvailableException) {
            Log.e(TAG, "相机不可用: ${e.message}")
            arState = ARState.FAILED
        } catch (e: Exception) {
            Log.e(TAG, "ARCore初始化最终失败: ${e.message}")
            arState = ARState.FAILED
        }
    }

    private fun configureAndResumeSession() {
        val session = arSession ?: throw RuntimeException("Session is null")

        val config = Config(session)
        config.planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
        config.lightEstimationMode = Config.LightEstimationMode.ENVIRONMENTAL_HDR
        config.depthMode = if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
            Config.DepthMode.AUTOMATIC
        } else {
            Config.DepthMode.DISABLED
        }
        session.configure(config)

        session.resume()
        arState = ARState.INITIALIZED

        if (!hasGLSurfaceView) {
            startBackgroundEglTracking()
        }

        Log.i(TAG, "ARCore Session已配置并启动, hasGLSurfaceView=$hasGLSurfaceView")
    }

    /**
     * 使用离屏 EGL PbufferSurface 启动后台追踪线程
     * 这样即使没有可见的 GLSurfaceView，也能持续调用 session.update() 获取位姿
     */
    private fun startBackgroundEglTracking() {
        stopBackgroundTracking()
        backgroundTrackingRunning = true
        backgroundTrackingThread = Thread({
            Log.i(TAG, "后台EGL追踪线程启动")

            var eglDisplay: EGLDisplay? = null
            var eglContext: EGLContext? = null
            var eglSurface: EGLSurface? = null
            var textureId = -1

            try {
                // 1. 初始化 EGL
                eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
                if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
                    throw RuntimeException("无法获取EGL Display")
                }
                val version = IntArray(2)
                if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
                    throw RuntimeException("EGL初始化失败")
                }

                // 2. 选择 EGL Config
                val configAttribs = intArrayOf(
                    EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                    EGL14.EGL_RED_SIZE, 8,
                    EGL14.EGL_GREEN_SIZE, 8,
                    EGL14.EGL_BLUE_SIZE, 8,
                    EGL14.EGL_ALPHA_SIZE, 8,
                    EGL14.EGL_DEPTH_SIZE, 16,
                    EGL14.EGL_NONE
                )
                val configs = arrayOfNulls<android.opengl.EGLConfig>(1)
                    val numConfigs = IntArray(1)
                if (!EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0)) {
                    throw RuntimeException("EGL选择Config失败")
                }
                val eglConfig = configs[0]!!

                // 3. 创建 EGL Context (OpenGL ES 2.0)
                val contextAttribs = intArrayOf(
                    EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
                    EGL14.EGL_NONE
                )
                eglContext = EGL14.eglCreateContext(eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttribs, 0)
                if (eglContext == EGL14.EGL_NO_CONTEXT) {
                    throw RuntimeException("EGL创建Context失败")
                }

                // 4. 创建离屏 PbufferSurface (1x1)
                val surfaceAttribs = intArrayOf(
                    EGL14.EGL_WIDTH, 1,
                    EGL14.EGL_HEIGHT, 1,
                    EGL14.EGL_NONE
                )
                eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, eglConfig, surfaceAttribs, 0)
                if (eglSurface == EGL14.EGL_NO_SURFACE) {
                    throw RuntimeException("EGL创建PbufferSurface失败")
                }

                // 5. 绑定 Context 和 Surface
                if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
                    throw RuntimeException("EGL MakeCurrent失败")
                }

                // 6. 创建相机纹理
                val textures = IntArray(1)
                GLES20.glGenTextures(1, textures, 0)
                textureId = textures[0]
                GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
                GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
                GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
                GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
                GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

                synchronized(sessionLock) {
                    cameraTextureId = textureId
                    arSession?.setCameraTextureName(textureId)
                    Log.i(TAG, "后台EGL: 已设置相机纹理ID=$textureId")
                }

                // 7. 主循环：调用 session.update()
                while (backgroundTrackingRunning && !hasGLSurfaceView) {
                    val session = arSession
                    if (session != null && arState == ARState.INITIALIZED) {
                        try {
                            synchronized(sessionLock) {
                                val frame = session.update()
                                val camera = frame.camera
                                val pose = camera.pose
                                val isTracking = camera.trackingState == TrackingState.TRACKING

                                latestCameraPose.set(mapOf(
                                    "tx" to pose.tx().toDouble(),
                                    "ty" to pose.ty().toDouble(),
                                    "tz" to pose.tz().toDouble(),
                                    "qw" to pose.qw().toDouble(),
                                    "qx" to pose.qx().toDouble(),
                                    "qy" to pose.qy().toDouble(),
                                    "qz" to pose.qz().toDouble(),
                                    "tracking" to if (isTracking) 1.0 else 0.0
                                ))
                            }
                        } catch (e: Exception) {
                            Log.w(TAG, "后台EGL: frame更新失败: ${e.message}")
                        }
                    }
                    Thread.sleep(33)
                }

            } catch (e: Exception) {
                Log.e(TAG, "后台EGL追踪线程失败: ${e.message}", e)
            } finally {
                // 清理 EGL 资源
                try {
                    if (eglDisplay != null && eglDisplay != EGL14.EGL_NO_DISPLAY) {
                        EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
                        if (eglSurface != null && eglSurface != EGL14.EGL_NO_SURFACE) {
                            EGL14.eglDestroySurface(eglDisplay, eglSurface)
                        }
                        if (eglContext != null && eglContext != EGL14.EGL_NO_CONTEXT) {
                            EGL14.eglDestroyContext(eglDisplay, eglContext)
                        }
                        EGL14.eglTerminate(eglDisplay)
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "清理EGL资源失败: ${e.message}")
                }
                Log.i(TAG, "后台EGL追踪线程退出")
            }
        }, "ARBackgroundEglTracking")
        backgroundTrackingThread?.isDaemon = true
        backgroundTrackingThread?.start()
    }

    private fun stopBackgroundTracking() {
        backgroundTrackingRunning = false
        try {
            backgroundTrackingThread?.join(1500)
        } catch (_: InterruptedException) {}
        backgroundTrackingThread = null
    }

    /**
     * 通知Activity有可见的GLSurfaceView了，后台EGL追踪可以停止
     */
    fun onGLSurfaceViewReady() {
        hasGLSurfaceView = true
        if (backgroundTrackingRunning) {
            Log.i(TAG, "GLSurfaceView已就绪，停止后台EGL追踪")
            stopBackgroundTracking()
        }
    }

    /**
     * 通知Activity可见的GLSurfaceView销毁了，需要重启后台EGL追踪
     */
    fun onGLSurfaceViewDestroyed() {
        hasGLSurfaceView = false
        if (arSession != null && arState == ARState.INITIALIZED && !backgroundTrackingRunning) {
            Log.i(TAG, "GLSurfaceView已销毁，启动后台EGL追踪")
            startBackgroundEglTracking()
        }
    }

    /**
     * 更新相机位姿缓存
     */
    fun updateCameraPose(frame: Frame) {
        val camera = frame.camera
        val pose = camera.pose
        val isTracking = camera.trackingState == TrackingState.TRACKING

        latestCameraPose.set(mapOf(
            "tx" to pose.tx().toDouble(),
            "ty" to pose.ty().toDouble(),
            "tz" to pose.tz().toDouble(),
            "qw" to pose.qw().toDouble(),
            "qx" to pose.qx().toDouble(),
            "qy" to pose.qy().toDouble(),
            "qz" to pose.qz().toDouble(),
            "tracking" to if (isTracking) 1.0 else 0.0
        ))
    }

    /**
     * 设置相机纹理ID（由Renderer在GL线程创建后调用）
     */
    fun setCameraTextureId(textureId: Int) {
        cameraTextureId = textureId
        arSession?.setCameraTextureName(textureId)
    }

    /**
     * ARCore PlatformView工厂
     */
    inner class ARCoreViewFactory(private val messenger: io.flutter.plugin.common.BinaryMessenger) :
        PlatformViewFactory(io.flutter.plugin.common.StandardMessageCodec.INSTANCE) {
        override fun create(context: Context, viewId: Int, args: Any?): PlatformView {
            arView = ARCorePlatformView(context, this@MainActivity, messenger, viewId)
            return arView!!
        }
    }

    /**
     * ARCore PlatformView - 渲染AR相机画面
     */
    class ARCorePlatformView(
        context: Context,
        private val activity: MainActivity,
        messenger: io.flutter.plugin.common.BinaryMessenger,
        viewId: Int
    ) : PlatformView {

        private val container: FrameLayout = FrameLayout(context)
        private val glSurfaceView: GLSurfaceView = GLSurfaceView(context)
        private val renderer: ARCoreRenderer

        private val methodChannel = MethodChannel(messenger, "arcore_view_$viewId")

        private var lastPlaneNotifyTime = 0L
        private val planeNotifyIntervalMs = 500L

        init {
            glSurfaceView.setEGLContextClientVersion(2)
            glSurfaceView.preserveEGLContextOnPause = true
            // 设置透明背景，确保混合合成模式下正确渲染
            glSurfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            glSurfaceView.holder.setFormat(android.graphics.PixelFormat.TRANSLUCENT)
            glSurfaceView.setZOrderOnTop(true)

            renderer = ARCoreRenderer(activity, this)
            glSurfaceView.setRenderer(renderer)
            glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY

            container.addView(glSurfaceView, FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            ))

            // MethodChannel处理
            methodChannel.setMethodCallHandler { call, result ->
                when (call.method) {
                    "initialize" -> {
                        // 检查Session状态
                        when (activity.arState) {
                            ARState.INITIALIZED -> {
                                result.success(true)
                            }
                            ARState.INSTALLING -> {
                                result.error("INSTALLING", "ARCore正在安装，请重启应用", null)
                            }
                            ARState.FAILED -> {
                                result.error("FAILED", "ARCore初始化失败，设备可能不支持", null)
                            }
                            ARState.NOT_INITIALIZED -> {
                                // 还没onResume过，标记需要初始化
                                result.error("NOT_READY", "ARCore尚未初始化，请稍后重试", null)
                            }
                        }
                    }
                    else -> result.notImplemented()
                }
            }
        }

        fun onFrameUpdated(frame: Frame) {
            activity.updateCameraPose(frame)

            // 平面检测通知（节流）
            val now = System.currentTimeMillis()
            if (now - lastPlaneNotifyTime >= planeNotifyIntervalMs) {
                val session = activity.arSession ?: return
                val planes = session.getAllTrackables(Plane::class.java)
                val activePlanes = planes.filter { it.trackingState == TrackingState.TRACKING }
                if (activePlanes.isNotEmpty()) {
                    lastPlaneNotifyTime = now
                    val planeData = activePlanes.map { plane ->
                        val pose = plane.centerPose
                        mapOf(
                            "id" to plane.hashCode().toString(),
                            "type" to when (plane.type) {
                                Plane.Type.HORIZONTAL_UPWARD_FACING -> "horizontal"
                                Plane.Type.HORIZONTAL_DOWNWARD_FACING -> "horizontal"
                                Plane.Type.VERTICAL -> "vertical"
                                else -> "unknown"
                            },
                            "center" to mapOf(
                                "x" to pose.tx().toDouble(),
                                "y" to pose.ty().toDouble(),
                                "z" to pose.tz().toDouble()
                            ),
                            "extent" to mapOf(
                                "x" to plane.extentX.toDouble(),
                                "y" to 0.0,
                                "z" to plane.extentZ.toDouble()
                            )
                        )
                    }
                    container.post {
                        methodChannel.invokeMethod("onPlanesUpdated", planeData)
                    }
                }
            }
        }

        override fun getView(): View = container

        override fun dispose() {
            glSurfaceView.onPause()
            activity.onGLSurfaceViewDestroyed()
        }
    }

    /**
     * ARCore GL渲染器
     */
    class ARCoreRenderer(
        private val activity: MainActivity,
        private val platformView: ARCorePlatformView
    ) : GLSurfaceView.Renderer {

        private var cameraTextureId = -1
        private var backgroundRenderer: CameraBackgroundRenderer? = null
        private var viewportWidth = 0
        private var viewportHeight = 0

        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            val textures = IntArray(1)
            GLES20.glGenTextures(1, textures, 0)
            cameraTextureId = textures[0]
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

            backgroundRenderer = CameraBackgroundRenderer()
            backgroundRenderer?.cameraTextureId = cameraTextureId

            activity.setCameraTextureId(cameraTextureId)
            activity.onGLSurfaceViewReady()
        }

        override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
            viewportWidth = width
            viewportHeight = height
            GLES20.glViewport(0, 0, width, height)
            activity.arSession?.setDisplayGeometry(0, width, height)
        }

        override fun onDrawFrame(gl: GL10?) {
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

            val session = activity.arSession
            if (session == null || activity.arState != ARState.INITIALIZED) return

            try {
                // session.update() 必须在 GL 线程执行（onDrawFrame 自身就在 GL 线程）
                // sessionLock 保留以兼容潜在的多线程访问场景
                synchronized(activity.sessionLock) {
                    val frame = session.update()
                    backgroundRenderer?.draw(frame)
                    platformView.onFrameUpdated(frame)
                }
            } catch (e: Exception) {
                Log.e(TAG, "渲染帧失败: ${e.message}")
            }
        }
    }

    /**
     * 相机背景渲染器
     */
    class CameraBackgroundRenderer {
        var cameraTextureId: Int = -1

        private val vertexShaderCode = """
            attribute vec4 aPosition;
            attribute vec2 aTexCoord;
            varying vec2 vTexCoord;
            void main() {
                gl_Position = aPosition;
                vTexCoord = aTexCoord;
            }
        """.trimIndent()

        private val fragmentShaderCode = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            varying vec2 vTexCoord;
            uniform samplerExternalOES sTexture;
            void main() {
                gl_FragColor = texture2D(sTexture, vTexCoord);
            }
        """.trimIndent()

        private val program: Int
        private val positionHandle: Int
        private val texCoordHandle: Int
        private val textureUniformHandle: Int
        private val quadVertices: FloatBuffer
        private var quadTexCoords: FloatBuffer

        init {
            val vertices = floatArrayOf(
                -1f, -1f,
                 1f, -1f,
                -1f,  1f,
                 1f,  1f
            )
            quadVertices = ByteBuffer.allocateDirect(vertices.size * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(vertices)
            quadVertices.position(0)

            val texCoords = floatArrayOf(
                0f, 1f,
                1f, 1f,
                0f, 0f,
                1f, 0f
            )
            quadTexCoords = ByteBuffer.allocateDirect(texCoords.size * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(texCoords)
            quadTexCoords.position(0)

            val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderCode)
            val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderCode)

            program = GLES20.glCreateProgram()
            GLES20.glAttachShader(program, vertexShader)
            GLES20.glAttachShader(program, fragmentShader)
            GLES20.glLinkProgram(program)

            positionHandle = GLES20.glGetAttribLocation(program, "aPosition")
            texCoordHandle = GLES20.glGetAttribLocation(program, "aTexCoord")
            textureUniformHandle = GLES20.glGetUniformLocation(program, "sTexture")
        }

        fun draw(frame: Frame) {
            GLES20.glUseProgram(program)

            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId)
            GLES20.glUniform1i(textureUniformHandle, 0)

            quadVertices.position(0)
            GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, quadVertices)
            GLES20.glEnableVertexAttribArray(positionHandle)

            // 根据相机旋转调整纹理坐标（输入输出buffer大小必须一致：8个float）
            val uvCoords = ByteBuffer.allocateDirect(8 * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f))
            uvCoords.position(0)
            val transform = ByteBuffer.allocateDirect(8 * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
            frame.transformDisplayUvCoords(uvCoords, transform)
            transform.position(0)
            quadTexCoords = transform

            quadTexCoords.position(0)
            GLES20.glVertexAttribPointer(texCoordHandle, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)
            GLES20.glEnableVertexAttribArray(texCoordHandle)

            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

            GLES20.glDisableVertexAttribArray(positionHandle)
            GLES20.glDisableVertexAttribArray(texCoordHandle)
        }

        private fun loadShader(type: Int, shaderCode: String): Int {
            val shader = GLES20.glCreateShader(type)
            GLES20.glShaderSource(shader, shaderCode)
            GLES20.glCompileShader(shader)
            return shader
        }
    }
}
