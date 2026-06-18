import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:augen/augen.dart';
import '../core/theme.dart';
import '../core/constants.dart';
import '../providers/connection_provider.dart' as conn;
import '../providers/mode_provider.dart';
import '../providers/pose_provider.dart';
import '../providers/gripper_provider.dart';
import '../providers/task_provider.dart';
import '../services/arcore_service.dart';
import '../services/sound_service.dart';
import '../widgets/ar_visualization_view.dart';

/// 主控页面
class MainControlScreen extends ConsumerStatefulWidget {
  const MainControlScreen({super.key});

  @override
  ConsumerState<MainControlScreen> createState() => _MainControlScreenState();
}

class _MainControlScreenState extends ConsumerState<MainControlScreen> {
  bool _arVisualizationEnabled = false;
  bool _showARView = false;
  ARCoreService? _arService;

  // WebSocket回调订阅（防止累积）
  StreamSubscription? _wsSubscription;

  @override
  void initState() {
    super.initState();
    _arService = ARCoreService();
    _setupARCallbacks();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _setupWebSocketCallbacks();
    });
  }

  void _setupARCallbacks() {
    if (_arService == null) return;

    _arService!.onPoseDelta = (dx, dy, dz, droll, dpitch, dyaw) {
      final wsService = ref.read(conn.webSocketServiceProvider);
      if (ref.read(conn.connectionProvider).isConnected) {
        wsService.sendPoseDelta(dx, dy, dz, droll, dpitch, dyaw);
      }
    };

    _arService!.onTrackingStateChanged = (isTracking) {
      if (mounted) setState(() {});
    };

    _arService!.onPlaneDetected = (ARPlane plane) {
      // 平面检测回调（视觉辅助）
    };

    _arService!.onError = (error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('AR错误: $error'),
            backgroundColor: Colors.red,
          ),
        );
      }
    };
  }

  void _setupWebSocketCallbacks() {
    final wsService = ref.read(conn.webSocketServiceProvider);

    // 清理旧的监听器
    _wsSubscription?.cancel();

    // 设置WebSocket回调
    wsService.onTcpPoseReceived = (pose) {
      ref.read(poseProvider.notifier).updatePose(pose);
    };

    wsService.onModeReceived = (mode) {
      ref.read(modeProvider.notifier).updateMode(mode);
    };

    wsService.onEpisodeSaved = () {
      ref.read(taskProvider.notifier).episodeSaved();
    };

    wsService.onTaskComplete = () {
      ref.read(taskProvider.notifier).endTask();
    };

    wsService.onError = (error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(error),
            backgroundColor: AppTheme.primaryRed,
          ),
        );
      }
    };

    wsService.onConnectionChanged = (connected) {
      if (connected) {
        SoundService().playConnected();
      }
    };
  }

  @override
  void dispose() {
    _wsSubscription?.cancel();
    _arService?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final connectionState = ref.watch(conn.connectionProvider);
    final poseState = ref.watch(poseProvider);

    return Scaffold(
      backgroundColor: AppTheme.background,
      body: SafeArea(
        child: Column(
          children: [
            _buildAppBar(),
            Expanded(
              child: SingleChildScrollView(
                padding:
                    const EdgeInsets.symmetric(horizontal: AppTheme.pagePaddingH),
                child: Column(
                  children: [
                    const SizedBox(height: 16),
                    _buildConnectionAndARSection(connectionState),
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

  Widget _buildConnectionAndARSection(conn.ConnectionState connectionState) {
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
                      'AR 可视化',
                      style: TextStyle(
                        fontSize: AppTheme.fontSizeBody,
                        fontWeight: FontWeight.w500,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _arVisualizationEnabled
                          ? 'AR追踪已启动 - 移动手机控制机械臂'
                          : '点击开启或关闭AR显示',
                      style: TextStyle(
                        fontSize: AppTheme.fontSizeCaption - 1,
                        color: _arVisualizationEnabled
                            ? AppTheme.primaryGreen
                            : AppTheme.textSecondary,
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
                    _showARView = value;
                    if (!value && _arService != null) {
                      _arService!.stopTracking();
                    }
                  });
                },
                activeTrackColor: AppTheme.primaryGreen,
              ),
            ],
          ),
        ),
        if (_showARView && _arService != null)
          Padding(
            padding: const EdgeInsets.only(top: 12),
            child: ARVisualizationView(
              arService: _arService,
              enableTracking: _arVisualizationEnabled,
              config: const ARVisualizationConfig(
                showPlanes: true,
                planeColor: Color(0xFFFFFFFF),
                enableLightEstimation: true,
              ),
              onControllerReady: (controller) {
                if (_arVisualizationEnabled) {
                  Future.delayed(const Duration(milliseconds: 300), () {
                    _arService?.startTracking();
                  });
                }
              },
              onPlaneDetected: (ARPlane plane) {
                // 平面检测视觉辅助
              },
            ),
          ),
      ],
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
            subLabel: '松开物体',
            iconPath: 'assets/icons/plus_circle.svg',
            backgroundColor: AppTheme.primaryGreen,
            textColor: Colors.white,
            onTap: () => ref.read(gripperProvider.notifier).open(),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _buildGripperButton(
            label: '闭合',
            subLabel: '抓住物体',
            iconPath: 'assets/icons/minus_circle.svg',
            backgroundColor: AppTheme.primaryBlack,
            textColor: Colors.white,
            onTap: () => ref.read(gripperProvider.notifier).close(),
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
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
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

  Widget _buildTaskControlButtons() {
    final taskState = ref.watch(taskProvider);

    return Row(
      children: [
        Expanded(
          child: SizedBox(
            height: AppTheme.buttonHeight,
            child: OutlinedButton(
              onPressed: taskState.status == TaskStatus.idle ||
                      taskState.status == TaskStatus.taskEnded
                  ? () => ref.read(taskProvider.notifier).startTask()
                  : null,
              style: OutlinedButton.styleFrom(
                foregroundColor: AppTheme.primaryGreen,
                side:
                    const BorderSide(color: AppTheme.primaryGreen, width: 1.5),
              ),
              child: const Text('开始任务'),
            ),
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          flex: 2,
          child: SizedBox(
            height: AppTheme.buttonHeight,
            child: OutlinedButton(
              onPressed: taskState.status == TaskStatus.taskStarted ||
                      taskState.status == TaskStatus.episodeRunning
                  ? () => ref.read(taskProvider.notifier).toggleEpisode()
                  : null,
              style: OutlinedButton.styleFrom(
                foregroundColor: AppTheme.accentYellowOrange,
                side:
                    const BorderSide(color: AppTheme.accentYellowOrange, width: 1.5),
              ),
              child: Text(taskState.isEpisodeRunning ? '结束Ep' : '开始Ep'),
            ),
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: SizedBox(
            height: AppTheme.buttonHeight,
            child: OutlinedButton(
              onPressed: taskState.status != TaskStatus.idle
                  ? () => ref.read(taskProvider.notifier).endTask()
                  : null,
              style: OutlinedButton.styleFrom(
                foregroundColor: AppTheme.accentPurple,
                side: const BorderSide(color: AppTheme.accentPurple, width: 1.5),
              ),
              child: const Text('结束任务'),
            ),
          ),
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
