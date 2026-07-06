import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_svg/flutter_svg.dart';
import '../services/arcore_service.dart';

/// AR可视化视图组件
///
/// 使用自定义PlatformView（arcore-view）渲染ARCore相机画面，
/// 并通过ARCoreService获取6DoF位姿数据。
///
/// 使用混合合成（Hybrid Composition）模式，确保GLSurfaceView正确渲染。
/// 虚拟显示模式会导致ARCore的GL_TEXTURE_EXTERNAL_OES纹理黑屏。
class ARVisualizationView extends StatefulWidget {
  final ARCoreService? arService;
  final Function()? onControllerReady;
  final Function(Map<String, dynamic> plane)? onPlaneDetected;
  final bool enableTracking;
  final ARVisualizationConfig config;
  final double height;

  const ARVisualizationView({
    super.key,
    this.arService,
    this.onControllerReady,
    this.onPlaneDetected,
    this.enableTracking = true,
    this.config = const ARVisualizationConfig(),
    this.height = 240,
  });

  @override
  State<ARVisualizationView> createState() => _ARVisualizationViewState();
}

class _ARVisualizationViewState extends State<ARVisualizationView> {
  bool _isInitialized = false;
  bool _arSupported = true;
  String _statusMessage = '初始化中...';
  MethodChannel? _viewChannel;

  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: widget.height,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Stack(
          children: [
            // ARCore相机画面 - 使用混合合成模式
            if (_arSupported)
              Positioned.fill(
                child: PlatformViewLink(
                  viewType: 'arcore-view',
                  surfaceFactory: (context, controller) {
                    return AndroidViewSurface(
                      controller: controller as AndroidViewController,
                      gestureRecognizers: const <Factory<
                          OneSequenceGestureRecognizer>>{},
                      hitTestBehavior: PlatformViewHitTestBehavior.opaque,
                    );
                  },
                  onCreatePlatformView: (params) {
                    final controller =
                        PlatformViewsService.initExpensiveAndroidView(
                      id: params.id,
                      viewType: 'arcore-view',
                      layoutDirection: TextDirection.ltr,
                      creationParams: null,
                      creationParamsCodec: const StandardMessageCodec(),
                      onFocus: () {
                        params.onFocusChanged(true);
                      },
                    );
                    controller.addOnPlatformViewCreatedListener((id) {
                      _onPlatformViewCreated(id);
                      params.onPlatformViewCreated(id);
                    });
                    controller.create();
                    return controller;
                  },
                ),
              )
            else
              Positioned.fill(
                child: Container(
                  color: Colors.black87,
                  child: Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        SvgPicture.asset(
                          'assets/icons/alert_triangle.svg',
                          width: 32,
                          height: 32,
                          colorFilter: const ColorFilter.mode(
                            Colors.orangeAccent,
                            BlendMode.srcIn,
                          ),
                        ),
                        const SizedBox(height: 8),
                        const Text(
                          '此设备不支持ARCore',
                          style: TextStyle(color: Colors.white70, fontSize: 13),
                        ),
                      ],
                    ),
                  ),
                ),
              ),

            // 状态覆盖层
            Positioned(
              top: 8,
              left: 8,
              right: 8,
              child: _buildStatusOverlay(),
            ),

            // 追踪状态指示器
            Positioned(
              bottom: 12,
              left: 0,
              right: 0,
              child: Center(child: _buildTrackingIndicator()),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusOverlay() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 6,
            height: 6,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: _isInitialized
                  ? (widget.arService?.isTracking ?? false)
                      ? Colors.greenAccent
                      : Colors.orangeAccent
                  : Colors.grey,
            ),
          ),
          const SizedBox(width: 6),
          Flexible(
            child: Text(
              _getStatusText(),
              style: const TextStyle(
                color: Colors.white,
                fontSize: 11,
                fontWeight: FontWeight.w500,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }

  String _getStatusText() {
    if (!_arSupported) return 'ARCore不可用';
    if (!_isInitialized) return _statusMessage;
    if (!(widget.arService?.isTracking ?? false)) return 'AR已就绪 - 等待追踪';
    return '正在追踪';
  }

  Widget _buildTrackingIndicator() {
    final isTracking = widget.arService?.isTracking ?? false;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      decoration: BoxDecoration(
        color:
            isTracking ? Colors.green.withValues(alpha: 0.9) : Colors.black54,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isTracking ? Colors.greenAccent : Colors.grey.shade600,
          width: 1.5,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SvgPicture.asset(
            isTracking
                ? 'assets/icons/radio_checked.svg'
                : 'assets/icons/radio_unchecked.svg',
            width: 14,
            height: 14,
            colorFilter: ColorFilter.mode(
              isTracking ? Colors.white : Colors.grey.shade400,
              BlendMode.srcIn,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            isTracking ? '追踪中' : '未追踪',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: isTracking ? Colors.white : Colors.grey.shade300,
            ),
          ),
        ],
      ),
    );
  }

  void _onPlatformViewCreated(int viewId) {
    _viewChannel = MethodChannel('arcore_view_$viewId');

    // 监听平面检测事件
    _viewChannel!.setMethodCallHandler((call) async {
      if (call.method == 'onPlanesUpdated') {
        final planes = call.arguments as List;
        if (planes.isNotEmpty) {
          widget.onPlaneDetected?.call(Map<String, dynamic>.from(planes.first));
          if (mounted) {
            setState(() {
              _statusMessage = '检测到平面';
            });
          }
        }
      }
    });

    // 初始化AR Session
    _initializeAR();
  }

  Future<void> _initializeAR() async {
    try {
      final result = await _viewChannel?.invokeMethod<bool>('initialize');
      if (result == true) {
        if (mounted) {
          setState(() {
            _isInitialized = true;
            _arSupported = true;
            _statusMessage = 'AR系统就绪';
          });
        }

        widget.onControllerReady?.call();

        // 自动开始追踪
        if (widget.enableTracking && widget.arService != null) {
          Future.delayed(const Duration(milliseconds: 500), () {
            if (mounted) {
              widget.arService!.startTracking();
            }
          });
        }
      } else {
        if (mounted) {
          setState(() {
            _arSupported = false;
            _statusMessage = 'AR初始化失败';
          });
        }
      }
    } on PlatformException catch (e) {
      if (mounted) {
        setState(() {
          _arSupported = false;
          _statusMessage = 'AR不可用: ${e.message}';
        });
      }
    }
  }

  @override
  void dispose() {
    _viewChannel?.setMethodCallHandler(null);
    super.dispose();
  }
}
