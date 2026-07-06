import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import '../core/theme.dart';
import '../core/constants.dart';
import '../providers/connection_provider.dart' as conn;
import '../providers/pose_provider.dart';
import '../providers/gripper_provider.dart';
import '../providers/task_provider.dart';
import '../providers/ar_provider.dart';
import '../services/arcore_service.dart';
import '../widgets/ar_visualization_view.dart';

/// 主控页面
class MainControlScreen extends ConsumerStatefulWidget {
  const MainControlScreen({super.key});

  @override
  ConsumerState<MainControlScreen> createState() => _MainControlScreenState();
}

class _MainControlScreenState extends ConsumerState<MainControlScreen> {
  bool _arVisualizationEnabled = true;
  // 调试：统计产生的位姿增量数量，用于诊断"手机移动但机械臂不动"
  int _poseDeltaCount = 0;
  DateTime _lastDeltaDebug = DateTime.now();

  late final void Function(double, double, double, double, double, double)
      _poseDeltaDebugListener;
  late final void Function(bool) _trackingDebugListener;
  late final void Function(String) _errorDebugListener;

  @override
  void initState() {
    super.initState();
    // 页面加载后主动请求一次状态同步，确保与 PC 端状态一致
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final wsService = ref.read(conn.webSocketServiceProvider);
      if (ref.read(conn.connectionProvider).isConnected) {
        wsService.requestStateSync();
      }
      // 添加调试监听器（AR服务已在app.dart中全局启动）
      _setupDebugListeners();
    });
  }

  void _setupDebugListeners() {
    final arService = ref.read(arCoreServiceProvider);

    _poseDeltaDebugListener = (dx, dy, dz, droll, dpitch, dyaw) {
      _poseDeltaCount++;
      final now = DateTime.now();
      if (now.difference(_lastDeltaDebug).inSeconds >= 1) {
        debugPrint('[AR] pose_delta count=$_poseDeltaCount, '
            'dx=${dx.toStringAsFixed(4)} dy=${dy.toStringAsFixed(4)} dz=${dz.toStringAsFixed(4)}');
        _poseDeltaCount = 0;
        _lastDeltaDebug = now;
      }
    };
    arService.addPoseDeltaListener(_poseDeltaDebugListener);

    _trackingDebugListener = (isTracking) {
      if (mounted) setState(() {});
    };
    arService.addTrackingListener(_trackingDebugListener);

    _errorDebugListener = (error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('AR错误: $error'),
            backgroundColor: Colors.red,
          ),
        );
      }
    };
    arService.addErrorListener(_errorDebugListener);
  }

  @override
  void dispose() {
    final arService = ref.read(arCoreServiceProvider);
    arService.removePoseDeltaListener(_poseDeltaDebugListener);
    arService.removeTrackingListener(_trackingDebugListener);
    arService.removeErrorListener(_errorDebugListener);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final connectionState = ref.watch(conn.connectionProvider);
    final poseState = ref.watch(poseProvider);
    final arState = ref.watch(arServiceProvider);
    final arService = ref.watch(arCoreServiceProvider);

    return Scaffold(
      backgroundColor: AppTheme.background,
      body: SafeArea(
        child: Column(
          children: [
            _buildAppBar(),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(
                    horizontal: AppTheme.pagePaddingH),
                child: Column(
                  children: [
                    const SizedBox(height: 16),
                    _buildConnectionAndARSection(
                        connectionState, arState, arService),
                    const SizedBox(height: 20),
                    _buildTcpPoseSection(poseState),
                    const SizedBox(height: 24),
                    _buildGripperSection(),
                    const SizedBox(height: 8),
                    const Text(
                      '${AppConstants.arcoreUpdateFrequency} Hz',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: AppTheme.fontSizeCaption - 1,
                        color: AppTheme.textHint,
                      ),
                    ),
                    const SizedBox(height: 20),
                    _buildStatusHint(),
                    const SizedBox(height: 12),
                    _buildCalibrationButton(),
                    const SizedBox(height: 12),
                    _buildTaskControlButtons(),
                    const SizedBox(height: 20),
                    _buildEmergencyStopButton(),
                    const SizedBox(height: 40),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAppBar() {
    final connectionState = ref.watch(conn.connectionProvider);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          GestureDetector(
            onTap: () =>
                Navigator.of(context).pushReplacementNamed('/settings'),
            child: Row(
              children: [
                SvgPicture.asset(
                  'assets/icons/arrow_left.svg',
                  width: 24,
                  height: 24,
                  colorFilter: const ColorFilter.mode(
                    AppTheme.textPrimary,
                    BlendMode.srcIn,
                  ),
                ),
                const SizedBox(width: 12),
                const Text(
                  '设置',
                  style: TextStyle(
                    fontSize: AppTheme.fontSizeTitle,
                    fontWeight: FontWeight.w600,
                    color: AppTheme.textPrimary,
                  ),
                ),
              ],
            ),
          ),
          Row(
            children: [
              Container(
                width: 12,
                height: 12,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: connectionState.isConnected
                      ? AppTheme.dotGreen
                      : AppTheme.dotRed,
                ),
              ),
              const SizedBox(width: 8),
              Text(
                connectionState.isConnected ? '已连接' : '未连接',
                style: TextStyle(
                  fontSize: AppTheme.fontSizeBody,
                  color: connectionState.isConnected
                      ? AppTheme.primaryGreen
                      : AppTheme.primaryRed,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(width: 12),
              SizedBox(
                height: 32,
                child: OutlinedButton(
                  onPressed: () {
                    if (connectionState.isConnected) {
                      ref.read(conn.connectionProvider.notifier).disconnect();
                    }
                    Navigator.of(context).pushReplacementNamed('/settings');
                  },
                  style: OutlinedButton.styleFrom(
                    foregroundColor: AppTheme.textPrimary,
                    side: const BorderSide(color: Color(0xFFBDBDBD), width: 1),
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    minimumSize: Size.zero,
                    tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  ),
                  child: Text(
                    connectionState.isConnected ? '断开' : '连接',
                    style: const TextStyle(fontSize: 13),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildConnectionAndARSection(conn.ConnectionState connectionState,
      ARState arState, ARCoreService arService) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: AppTheme.cardBackground,
            borderRadius: BorderRadius.circular(AppTheme.radiusCard),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'AR 追踪',
                      style: TextStyle(
                        fontSize: AppTheme.fontSizeBody,
                        fontWeight: FontWeight.w500,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      arState.isTracking
                          ? '追踪中 - 移动手机控制机械臂'
                          : (arState.isAvailable
                              ? 'AR已就绪，请移动手机扫描环境'
                              : '正在初始化AR...'),
                      style: TextStyle(
                        fontSize: AppTheme.fontSizeCaption - 1,
                        color: arState.isTracking
                            ? AppTheme.primaryGreen
                            : (arState.isAvailable
                                ? AppTheme.textSecondary
                                : AppTheme.textHint),
                      ),
                    ),
                  ],
                ),
              ),
              Switch(
                value: _arVisualizationEnabled,
                onChanged: (value) {
                  setState(() {
                    _arVisualizationEnabled = value;
                  });
                },
                activeTrackColor: AppTheme.primaryGreen,
              ),
            ],
          ),
        ),
        Visibility(
          visible: _arVisualizationEnabled,
          maintainState: true,
          maintainAnimation: true,
          maintainSize: false,
          maintainInteractivity: false,
          child: Opacity(
            opacity: _arVisualizationEnabled ? 1.0 : 0.0,
            child: IgnorePointer(
              ignoring: !_arVisualizationEnabled,
              child: Padding(
                padding: const EdgeInsets.only(top: 12),
                child: ARVisualizationView(
                  arService: arService,
                  enableTracking: true,
                  config: const ARVisualizationConfig(
                    showPlanes: true,
                    planeColor: Color(0xFFFFFFFF),
                    enableLightEstimation: true,
                  ),
                  onControllerReady: () {},
                  onPlaneDetected: (plane) {},
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildCalibrationButton() {
    final taskState = ref.watch(taskProvider);
    final canCalibrate = taskState.status == TaskStatus.collectionStarted;

    return SizedBox(
      width: double.infinity,
      height: AppTheme.buttonHeight,
      child: OutlinedButton(
        onPressed: canCalibrate
            ? () {
                Navigator.of(context).pushNamed('/calibration');
              }
            : null,
        style: OutlinedButton.styleFrom(
          foregroundColor: AppTheme.primaryBlue,
          side: BorderSide(
              color: canCalibrate ? AppTheme.primaryBlue : AppTheme.textHint,
              width: 1.5),
        ),
        child: const Text('标定坐标'),
      ),
    );
  }

  Widget _buildTcpPoseSection(PoseState poseState) {
    final pose = poseState.pose;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.cardBackground,
        borderRadius: BorderRadius.circular(AppTheme.radiusCard),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Text(
                'TCP 位姿',
                style: TextStyle(
                  fontSize: AppTheme.fontSizeBody,
                  fontWeight: FontWeight.w500,
                  color: AppTheme.textPrimary,
                ),
              ),
              Spacer(),
              Text(
                '${AppConstants.arcoreUpdateFrequency} Hz',
                style: TextStyle(
                  fontSize: AppTheme.fontSizeCaption - 1,
                  color: AppTheme.primaryGreen,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          GridView.count(
            crossAxisCount: 2,
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            mainAxisSpacing: 10,
            crossAxisSpacing: 20,
            childAspectRatio: 3,
            children: [
              _buildPoseValueItem('X', pose.x.toStringAsFixed(3)),
              _buildPoseValueItem('Y', pose.y.toStringAsFixed(3)),
              _buildPoseValueItem('Z', pose.z.toStringAsFixed(3)),
              _buildPoseValueItem('RX', '${pose.rx.toStringAsFixed(2)} deg'),
              _buildPoseValueItem('RY', '${pose.ry.toStringAsFixed(2)} deg'),
              _buildPoseValueItem('RZ', '${pose.rz.toStringAsFixed(2)} deg'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildPoseValueItem(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: AppTheme.fontSizeBody,
            color: AppTheme.textSecondary,
          ),
        ),
        Text(
          value,
          style: const TextStyle(
            fontSize: AppTheme.fontSizeValue,
            color: AppTheme.textPrimary,
            fontFamily: 'monospace',
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }

  Widget _buildGripperSection() {
    return Row(
      children: [
        Expanded(
          child: _buildGripperButton(
            label: '张开',
            subLabel: '按住松开物体',
            iconPath: 'assets/icons/plus_circle.svg',
            backgroundColor: AppTheme.primaryGreen,
            textColor: Colors.white,
            onStart: () => ref.read(gripperProvider.notifier).startOpen(),
            onStop: () => ref.read(gripperProvider.notifier).stop(),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _buildGripperButton(
            label: '闭合',
            subLabel: '按住抓住物体',
            iconPath: 'assets/icons/minus_circle.svg',
            backgroundColor: AppTheme.primaryBlack,
            textColor: Colors.white,
            onStart: () => ref.read(gripperProvider.notifier).startClose(),
            onStop: () => ref.read(gripperProvider.notifier).stop(),
          ),
        ),
      ],
    );
  }

  Widget _buildGripperButton({
    required String label,
    required String subLabel,
    required String iconPath,
    required Color backgroundColor,
    required Color textColor,
    required VoidCallback onStart,
    required VoidCallback onStop,
  }) {
    return GestureDetector(
      onTapDown: (_) => onStart(),
      onTapUp: (_) => onStop(),
      onTapCancel: () => onStop(),
      child: Container(
        height: 120,
        decoration: BoxDecoration(
          color: backgroundColor,
          borderRadius: BorderRadius.circular(AppTheme.radiusButton),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SvgPicture.asset(
              iconPath,
              width: 48,
              height: 48,
              colorFilter: ColorFilter.mode(textColor, BlendMode.srcIn),
            ),
            const SizedBox(height: 8),
            Text(
              label,
              style: TextStyle(
                fontSize: AppTheme.fontSizeButton,
                fontWeight: FontWeight.w600,
                color: textColor,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              subLabel,
              style: TextStyle(
                fontSize: AppTheme.fontSizeCaption - 1,
                color: textColor.withValues(alpha: 0.8),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusHint() {
    final taskState = ref.watch(taskProvider);
    Color textColor;
    switch (taskState.status) {
      case TaskStatus.idle:
        textColor = AppTheme.textSecondary;
        break;
      case TaskStatus.preparing:
        textColor = AppTheme.accentYellowOrange;
        break;
      case TaskStatus.calibrating:
      case TaskStatus.episodeRunning:
        textColor = AppTheme.primaryGreen;
        break;
      default:
        textColor = AppTheme.textSecondary;
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: AppTheme.cardBackground,
        borderRadius: BorderRadius.circular(AppTheme.radiusCard),
      ),
      child: Text(
        taskState.statusText,
        textAlign: TextAlign.center,
        style: TextStyle(
          fontSize: AppTheme.fontSizeBody,
          color: textColor,
          fontWeight: FontWeight.w500,
        ),
      ),
    );
  }

  Widget _buildTaskControlButtons() {
    final taskState = ref.watch(taskProvider);

    return Column(
      children: [
        Row(
          children: [
            Expanded(
              child: SizedBox(
                height: AppTheme.buttonHeight,
                child: ElevatedButton(
                  onPressed: taskState.canStartCollection
                      ? () => ref.read(taskProvider.notifier).startCollection()
                      : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primaryGreen,
                    foregroundColor: Colors.white,
                    disabledBackgroundColor:
                        AppTheme.textHint.withValues(alpha: 0.3),
                  ),
                  child: const Text('开启收集'),
                ),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: SizedBox(
                height: AppTheme.buttonHeight,
                child: ElevatedButton(
                  onPressed: taskState.canEndCollection
                      ? () => ref.read(taskProvider.notifier).endCollection()
                      : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primaryRed,
                    foregroundColor: Colors.white,
                    disabledBackgroundColor:
                        AppTheme.textHint.withValues(alpha: 0.3),
                  ),
                  child: const Text('结束收集'),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
        Row(
          children: [
            Expanded(
              child: SizedBox(
                height: AppTheme.buttonHeight,
                child: OutlinedButton(
                  onPressed: taskState.canStartTask
                      ? () => ref.read(taskProvider.notifier).startTask()
                      : null,
                  style: OutlinedButton.styleFrom(
                    foregroundColor: AppTheme.accentYellowOrange,
                    side: BorderSide(
                        color: taskState.canStartTask
                            ? AppTheme.accentYellowOrange
                            : AppTheme.textHint,
                        width: 1.5),
                  ),
                  child: const Text('开启任务'),
                ),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: SizedBox(
                height: AppTheme.buttonHeight,
                child: OutlinedButton(
                  onPressed: taskState.canEndTask
                      ? () => ref.read(taskProvider.notifier).endTask()
                      : null,
                  style: OutlinedButton.styleFrom(
                    foregroundColor: AppTheme.primaryBlue,
                    side: BorderSide(
                        color: taskState.canEndTask
                            ? AppTheme.primaryBlue
                            : AppTheme.textHint,
                        width: 1.5),
                  ),
                  child: const Text('结束任务'),
                ),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: SizedBox(
                height: AppTheme.buttonHeight,
                child: OutlinedButton(
                  onPressed: taskState.canRetryTask
                      ? () => ref.read(taskProvider.notifier).retryTask()
                      : null,
                  style: OutlinedButton.styleFrom(
                    foregroundColor: AppTheme.accentPurple,
                    side: BorderSide(
                        color: taskState.canRetryTask
                            ? AppTheme.accentPurple
                            : AppTheme.textHint,
                        width: 1.5),
                  ),
                  child: const Text('重做任务'),
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildEmergencyStopButton() {
    return SizedBox(
      width: double.infinity,
      height: AppTheme.buttonHeight,
      child: ElevatedButton(
        onPressed: () {
          showDialog(
            context: context,
            builder: (ctx) => AlertDialog(
              title: const Text('确认急停？'),
              content: const Text('急停将立即停止所有运动并可能丢弃当前数据。'),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(ctx).pop(),
                  child: const Text('取消'),
                ),
                TextButton(
                  onPressed: () {
                    Navigator.of(ctx).pop();
                    ref.read(taskProvider.notifier).emergencyStop();
                  },
                  style: TextButton.styleFrom(
                    foregroundColor: AppTheme.primaryRed,
                  ),
                  child: const Text('确认急停'),
                ),
              ],
            ),
          );
        },
        style: ElevatedButton.styleFrom(
          backgroundColor: AppTheme.primaryRed,
          foregroundColor: Colors.white,
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SvgPicture.asset(
              'assets/icons/alert_triangle.svg',
              width: 20,
              height: 20,
              colorFilter: const ColorFilter.mode(
                Colors.white,
                BlendMode.srcIn,
              ),
            ),
            const SizedBox(width: 8),
            const Text(
              '急停',
              style: TextStyle(
                fontSize: AppTheme.fontSizeButton,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
