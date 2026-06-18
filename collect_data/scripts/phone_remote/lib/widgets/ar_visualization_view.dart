import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:augen/augen.dart';
import '../services/arcore_service.dart';

/// AR可视化视图组件
///
/// 功能：
/// - 显示ARCore相机画面
/// - 可视化检测到的平面（辅助定位）
/// - 显示追踪状态指示
/// - 集成ARCoreService进行6DoF追踪
class ARVisualizationView extends StatefulWidget {
  final ARCoreService? arService;
  final Function(AugenController controller)? onControllerReady;
  final Function(ARPlane plane)? onPlaneDetected;
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
  String _statusMessage = '初始化中...';

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: widget.height,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Stack(
          children: [
            // ARCore相机视图（使用AugenView）
            Positioned.fill(
              child: AugenView(
                onViewCreated: _onAugenViewCreated,
                config: ARSessionConfig(
                  planeDetection: widget.config.showPlanes,
                  lightEstimation: widget.config.enableLightEstimation,
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

  void _onAugenViewCreated(AugenController controller) async {
    _isInitialized = true;

    // 初始化AR服务
    try {
      await controller.initialize(ARSessionConfig(
        planeDetection: widget.config.showPlanes,
        lightEstimation: widget.config.enableLightEstimation,
      ));
    } catch (e) {
      if (mounted) {
        setState(() {
          _statusMessage = 'AR初始化失败: $e';
        });
      }
      return;
    }

    // 初始化ARCore服务
    if (widget.arService != null) {
      widget.arService!.initController(controller);

      // 设置平面检测回调（单个ARPlane）
      widget.arService!.onPlaneDetected = (ARPlane plane) {
        widget.onPlaneDetected?.call(plane);
        if (mounted) {
          setState(() {
            _statusMessage = '检测到平面';
          });
        }
      };
    }

    // 通知外部控制器已就绪
    widget.onControllerReady?.call(controller);

    if (mounted) {
      setState(() {
        _statusMessage = 'AR系统就绪';
      });
    }

    // 如果启用自动追踪
    if (widget.enableTracking && widget.arService != null) {
      Future.delayed(const Duration(milliseconds: 500), () {
        if (mounted) {
          widget.arService!.startTracking();
        }
      });
    }
  }

  @override
  void dispose() {
    // 不在这里dispose controller，它由AugenView管理
    super.dispose();
  }
}
