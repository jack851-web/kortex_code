import 'dart:async';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:flutter/material.dart';
import 'package:vector_math/vector_math_64.dart' as vm;

/// ARCore追踪服务 - 获取6DoF位姿增量
///
/// 通过原生MethodChannel获取ARCore相机位姿数据。
/// Android端由MainActivity.kt中的ARCore Session提供。
/// 使用单例模式，确保全局只有一个实例，避免重复轮询。
class ARCoreService {
  // ========== 单例实现 ==========
  static final ARCoreService _instance = ARCoreService._internal();
  factory ARCoreService() => _instance;
  ARCoreService._internal();

  // ========== 状态变量 ==========
  bool _isTracking = false;
  bool _isARAvailable = false;

  // 位姿数据（当前帧）
  vm.Vector3 _currentPosition = vm.Vector3.zero();
  vm.Quaternion _currentOrientation = vm.Quaternion.identity();

  // 基准位姿（用于计算增量）
  vm.Vector3 _referencePosition = vm.Vector3.zero();
  vm.Quaternion _referenceOrientation = vm.Quaternion.identity();

  // 上一帧位姿（用于差分）
  vm.Vector3 _lastPosition = vm.Vector3.zero();
  vm.Quaternion _lastOrientation = vm.Quaternion.identity();
  bool _hasReceivedFirstPose = false;

  // 原生MethodChannel（获取ARCore相机位姿）
  static const MethodChannel _poseChannel =
      MethodChannel('com.example.phone_remote/arcore_pose');

  // 位姿更新定时器
  Timer? _poseUpdateTimer;

  // ========== 回调函数（支持多监听器）==========
  final List<
      Function(double dx, double dy, double dz, double droll, double dpitch,
          double dyaw)> _poseDeltaListeners = [];
  final List<
      Function(double x, double y, double z, double roll, double pitch,
          double yaw)> _absolutePoseListeners = [];
  final List<Function(bool isTracking)> _trackingListeners = [];
  final List<Function(Map<String, dynamic> plane)> _planeListeners = [];
  final List<Function(String error)> _errorListeners = [];

  /// 添加位姿增量监听器
  void addPoseDeltaListener(
      Function(double dx, double dy, double dz, double droll, double dpitch,
              double dyaw)
          listener) {
    if (!_poseDeltaListeners.contains(listener)) {
      _poseDeltaListeners.add(listener);
    }
  }

  /// 移除位姿增量监听器
  void removePoseDeltaListener(
      Function(double dx, double dy, double dz, double droll, double dpitch,
              double dyaw)
          listener) {
    _poseDeltaListeners.remove(listener);
  }

  /// 添加绝对位姿监听器
  void addAbsolutePoseListener(
      Function(double x, double y, double z, double roll, double pitch,
              double yaw)
          listener) {
    if (!_absolutePoseListeners.contains(listener)) {
      _absolutePoseListeners.add(listener);
    }
  }

  void removeAbsolutePoseListener(
      Function(double x, double y, double z, double roll, double pitch,
              double yaw)
          listener) {
    _absolutePoseListeners.remove(listener);
  }

  /// 添加追踪状态监听器
  void addTrackingListener(Function(bool isTracking) listener) {
    if (!_trackingListeners.contains(listener)) {
      _trackingListeners.add(listener);
    }
  }

  void removeTrackingListener(Function(bool isTracking) listener) {
    _trackingListeners.remove(listener);
  }

  /// 添加错误监听器
  void addErrorListener(Function(String error) listener) {
    if (!_errorListeners.contains(listener)) {
      _errorListeners.add(listener);
    }
  }

  void removeErrorListener(Function(String error) listener) {
    _errorListeners.remove(listener);
  }

  // ========== 兼容旧API的setter（设置单个回调，会清空之前的同类型监听器）==========
  set onPoseDelta(
      Function(double dx, double dy, double dz, double droll, double dpitch,
              double dyaw)?
          callback) {
    _poseDeltaListeners.clear();
    if (callback != null) {
      _poseDeltaListeners.add(callback);
    }
  }

  set onAbsolutePose(
      Function(double x, double y, double z, double roll, double pitch,
              double yaw)?
          callback) {
    _absolutePoseListeners.clear();
    if (callback != null) {
      _absolutePoseListeners.add(callback);
    }
  }

  set onTrackingStateChanged(Function(bool isTracking)? callback) {
    _trackingListeners.clear();
    if (callback != null) {
      _trackingListeners.add(callback);
    }
  }

  set onPlaneDetected(Function(Map<String, dynamic> plane)? callback) {
    _planeListeners.clear();
    if (callback != null) {
      _planeListeners.add(callback);
    }
  }

  set onError(Function(String error)? callback) {
    _errorListeners.clear();
    if (callback != null) {
      _errorListeners.add(callback);
    }
  }

  // ========== 公共属性 ==========

  bool get isTracking => _isTracking;
  bool get isARAvailable => _isARAvailable;
  vm.Vector3 get currentPosition => _currentPosition;
  vm.Quaternion get currentOrientation => _currentOrientation;

  bool get hasReference =>
      _referencePosition.length > 0.001 ||
      (_referenceOrientation.x.abs() > 0.001 ||
          _referenceOrientation.y.abs() > 0.001 ||
          _referenceOrientation.z.abs() > 0.001);

  // ========== 核心方法 ==========

  /// 检查AR是否可用
  Future<bool> checkARSupport() async {
    try {
      final supported = await _poseChannel.invokeMethod<bool>('isARSupported');
      _isARAvailable = supported ?? false;
      return _isARAvailable;
    } on PlatformException {
      _isARAvailable = false;
      return false;
    }
  }

  /// 请求安装ARCore服务
  Future<void> requestInstall() async {
    try {
      await _poseChannel.invokeMethod<bool>('requestInstall');
    } on PlatformException {
      // 忽略
    }
  }

  /// 开始ARCore追踪
  void startTracking() {
    if (_isTracking) return;

    _isTracking = true;
    _hasReceivedFirstPose = false;
    // 通知所有追踪状态监听器
    for (final listener in List.of(_trackingListeners)) {
      listener(true);
    }

    // 启动位姿轮询
    _startPosePolling();
  }

  /// 停止追踪
  void stopTracking() {
    if (!_isTracking) return;

    _isTracking = false;
    _poseUpdateTimer?.cancel();
    _poseUpdateTimer = null;

    // 通知所有追踪状态监听器
    for (final listener in List.of(_trackingListeners)) {
      listener(false);
    }
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
  void _startPosePolling() {
    const updateIntervalMs = 33; // ~30Hz

    _poseUpdateTimer = Timer.periodic(
      const Duration(milliseconds: updateIntervalMs),
      (_) => _pollCameraPose(),
    );
  }

  /// 从原生层获取相机位姿
  Future<void> _pollCameraPose() async {
    if (!_isTracking || _poseDeltaListeners.isEmpty) return;

    try {
      final result = await _poseChannel.invokeMethod<Map>('getCameraPose');

      if (result != null) {
        final isTracking = (result['tracking'] as num?)?.toDouble() == 1.0;

        if (!isTracking) {
          return;
        }

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
    } on MissingPluginException {}
  }

  /// 计算并发出位姿增量
  void _calculateAndEmitDelta() {
    // 第一帧：只保存位姿，不发出delta（避免初始跳变）
    if (!_hasReceivedFirstPose) {
      _lastPosition = vm.Vector3.copy(_currentPosition);
      _lastOrientation = vm.Quaternion.copy(_currentOrientation);
      _hasReceivedFirstPose = true;
      return;
    }

    final positionDelta = _currentPosition - _lastPosition;

    final orientationDelta =
        _currentOrientation * _lastOrientation.conjugated();
    final eulerDelta = _quaternionToEuler(orientationDelta);

    // 通知所有位姿增量监听器
    for (final listener in List.of(_poseDeltaListeners)) {
      listener(
        positionDelta.x,
        positionDelta.y,
        positionDelta.z,
        eulerDelta[0],
        eulerDelta[1],
        eulerDelta[2],
      );
    }

    _lastPosition = vm.Vector3.copy(_currentPosition);
    _lastOrientation = vm.Quaternion.copy(_currentOrientation);

    // 通知所有绝对位姿监听器
    if (_absolutePoseListeners.isNotEmpty) {
      final absEuler = _quaternionToEuler(_currentOrientation);
      for (final listener in List.of(_absolutePoseListeners)) {
        listener(
          _currentPosition.x,
          _currentPosition.y,
          _currentPosition.z,
          absEuler[0],
          absEuler[1],
          absEuler[2],
        );
      }
    }
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

  /// 复制符号
  static double _copySign(double magnitude, double sign) {
    return sign >= 0 ? magnitude.abs() : -magnitude.abs();
  }

  // ========== 生命周期管理 ==========

  void dispose() {
    stopTracking();
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
