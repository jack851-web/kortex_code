import 'dart:async';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:vector_math/vector_math_64.dart' as vm;
import 'package:augen/augen.dart' hide Vector3, Quaternion;

/// ARCore追踪服务 - 获取6DoF位姿增量
///
/// 实现策略：
/// 1. 使用 augen 插件进行视觉AR（平面检测、相机画面）
/// 2. 通过原生 MethodChannel 获取ARCore相机位姿（6DoF）
/// 3. 增量计算：以每次resetReference()为基准点，计算相对位移
class ARCoreService {
  // ========== 状态变量 ==========
  bool _isTracking = false;
  AugenController? _augenController;

  // 位姿数据（当前帧）
  vm.Vector3 _currentPosition = vm.Vector3.zero();
  vm.Quaternion _currentOrientation = vm.Quaternion.identity();

  // 基准位姿（用于计算增量）
  vm.Vector3 _referencePosition = vm.Vector3.zero();
  vm.Quaternion _referenceOrientation = vm.Quaternion.identity();

  // 上一帧位姿（用于差分）
  vm.Vector3 _lastPosition = vm.Vector3.zero();
  vm.Quaternion _lastOrientation = vm.Quaternion.identity();

  // 原生MethodChannel（获取ARCore相机位姿）
  static const MethodChannel _poseChannel =
      MethodChannel('com.example.phone_remote/arcore_pose');

  // 位姿更新定时器
  Timer? _poseUpdateTimer;

  // ========== 回调函数 ==========

  /// 位姿增量回调（高频 ~30Hz）
  Function(double dx, double dy, double dz, double droll, double dpitch,
      double dyaw)? onPoseDelta;

  /// 绝对位姿回调（低频 ~5Hz，用于显示）
  Function(double x, double y, double z, double roll, double pitch, double yaw)?
      onAbsolutePose;

  /// AR状态变化回调
  Function(bool isTracking)? onTrackingStateChanged;

  /// 平面检测回调（单个平面，匹配augen API）
  Function(ARPlane plane)? onPlaneDetected;

  /// 错误回调
  Function(String error)? onError;

  // ========== 公共属性 ==========

  bool get isTracking => _isTracking;
  vm.Vector3 get currentPosition => _currentPosition;
  vm.Quaternion get currentOrientation => _currentOrientation;

  bool get hasReference =>
      _referencePosition.length > 0.001 ||
      (_referenceOrientation.x.abs() > 0.001 ||
          _referenceOrientation.y.abs() > 0.001 ||
          _referenceOrientation.z.abs() > 0.001);

  // ========== 核心方法 ==========

  /// 初始化AR控制器
  ///
  /// [controller] 由AugenView widget创建的控制器实例
  void initController(AugenController controller) {
    _augenController = controller;

    // 设置平面检测回调（augen发送ARPlane列表）
    controller.planesStream.listen((planes) {
      if (planes.isNotEmpty) {
        onPlaneDetected?.call(planes.first);
      }
    });

    // 设置错误回调
    controller.errorStream.listen((error) {
      onError?.call('ARCore错误: $error');
    });
  }

  /// 开始ARCore追踪
  void startTracking() {
    if (_isTracking) return;

    if (_augenController == null) {
      onError?.call('ARCore控制器未初始化');
      return;
    }

    _isTracking = true;
    onTrackingStateChanged?.call(true);

    // 启动位姿轮询（通过原生MethodChannel获取相机位姿）
    _startPosePolling();
  }

  /// 停止追踪
  void stopTracking() {
    if (!_isTracking) return;

    _isTracking = false;
    _poseUpdateTimer?.cancel();
    _poseUpdateTimer = null;

    onTrackingStateChanged?.call(false);
  }

  /// 重置基准点（每次开始新Episode时调用）
  void resetReference() {
    _referencePosition = vm.Vector3.copy(_currentPosition);
    _referenceOrientation = vm.Quaternion.copy(_currentOrientation);
    _lastPosition = vm.Vector3.copy(_currentPosition);
    _lastOrientation = vm.Quaternion.copy(_currentOrientation);
  }

  /// 获取相对于基准点的累积位移
  vm.Vector3 getCumulativeDelta() {
    return _currentPosition - _referencePosition;
  }

  /// 获取相对于基准点的累积旋转（欧拉角，度）
  List<double> getCumulativeRotationDelta() {
    final deltaQuat = _referenceOrientation * _currentOrientation.conjugated();
    return _quaternionToEuler(deltaQuat);
  }

  // ========== 内部方法 ==========

  /// 启动位姿轮询
  ///
  /// 通过原生MethodChannel定期获取ARCore相机位姿。
  /// 频率约30Hz（平衡精度和性能）。
  void _startPosePolling() {
    const updateIntervalMs = 33; // ~30Hz

    _poseUpdateTimer = Timer.periodic(
      const Duration(milliseconds: updateIntervalMs),
      (_) => _pollCameraPose(),
    );
  }

  /// 从原生层获取相机位姿
  Future<void> _pollCameraPose() async {
    if (!_isTracking || onPoseDelta == null) return;

    try {
      final result = await _poseChannel.invokeMethod<Map>('getCameraPose');

      if (result != null) {
        // 解析原生层返回的位姿数据
        _currentPosition = vm.Vector3(
          (result['tx'] as num?)?.toDouble() ?? 0.0,
          (result['ty'] as num?)?.toDouble() ?? 0.0,
          (result['tz'] as num?)?.toDouble() ?? 0.0,
        );
        _currentOrientation = vm.Quaternion(
          (result['qw'] as num?)?.toDouble() ?? 1.0,
          (result['qx'] as num?)?.toDouble() ?? 0.0,
          (result['qy'] as num?)?.toDouble() ?? 0.0,
          (result['qz'] as num?)?.toDouble() ?? 0.0,
        );

        _calculateAndEmitDelta();
      }
    } on PlatformException {
      // MethodChannel未实现时静默处理（可能在模拟器或未配置原生代码的设备上）
      // 不输出日志避免刷屏
    } on MissingPluginException {
      // 原生插件未注册，静默处理
    }
  }

  /// 计算并发出位姿增量
  void _calculateAndEmitDelta() {
    // 位置增量（相对于上一帧）
    final positionDelta = _currentPosition - _lastPosition;

    // 姿态增量（四元数差分）
    final orientationDelta =
        _currentOrientation * _lastOrientation.conjugated();
    final eulerDelta = _quaternionToEuler(orientationDelta);

    // 发出增量回调
    onPoseDelta!.call(
      positionDelta.x,
      positionDelta.y,
      positionDelta.z,
      eulerDelta[0],
      eulerDelta[1],
      eulerDelta[2],
    );

    // 更新上一帧数据
    _lastPosition = vm.Vector3.copy(_currentPosition);
    _lastOrientation = vm.Quaternion.copy(_currentOrientation);

    // 低频发送绝对位姿
    onAbsolutePose?.call(
      _currentPosition.x,
      _currentPosition.y,
      _currentPosition.z,
      eulerDelta[0],
      eulerDelta[1],
      eulerDelta[2],
    );
  }

  /// 四元数转欧拉角（度）
  List<double> _quaternionToEuler(vm.Quaternion q) {
    q.normalize();

    final sinrCosp = 2 * (q.w * q.x + q.y * q.z);
    final cosrCosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    final roll = atan2(sinrCosp, cosrCosp);

    final sinp = 2 * (q.w * q.y - q.z * q.x);
    double pitch;
    if (sinp.abs() >= 1) {
      pitch = _copySign(pi / 2, sinp);
    } else {
      pitch = asin(sinp);
    }

    final sinyCosp = 2 * (q.w * q.z + q.x * q.y);
    final cosyCosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    final yaw = atan2(sinyCosp, cosyCosp);

    return [_radToDeg(roll), _radToDeg(pitch), _radToDeg(yaw)];
  }

  double _radToDeg(double rad) => rad * 180 / pi;

  /// 复制符号（兼容性替代copysign）
  static double _copySign(double magnitude, double sign) {
    return sign >= 0 ? magnitude.abs() : -magnitude.abs();
  }

  // ========== 生命周期管理 ==========

  void dispose() {
    stopTracking();
    // 不在这里dispose AugenController，它由AugenView管理
    _augenController = null;
  }
}

/// AR可视化配置
class ARVisualizationConfig {
  final bool showPlanes;
  final Color planeColor;
  final bool showAxes;
  final double axesLength;
  final bool enableLightEstimation;

  const ARVisualizationConfig({
    this.showPlanes = true,
    this.planeColor = const Color(0xFFFFFFFF),
    this.showAxes = false,
    this.axesLength = 0.2,
    this.enableLightEstimation = true,
  });
}
