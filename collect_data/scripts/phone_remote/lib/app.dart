import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'core/theme.dart';
import 'providers/connection_provider.dart' as conn;
import 'providers/mode_provider.dart';
import 'providers/pose_provider.dart';
import 'providers/task_provider.dart';
import 'providers/ar_provider.dart';
import 'screens/settings_screen.dart';
import 'screens/main_control_screen.dart';
import 'screens/calibration_screen.dart';
import 'screens/smart_calibration_screen.dart';

class PhoneRemoteApp extends ConsumerWidget {
  const PhoneRemoteApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // 在首次 build 时设置 WebSocket 回调桥接
    // 这确保回调在任何页面导航之前就已注册，不会丢失 PC 端发送的初始状态
    _setupCallbackBridge(ref);

    return MaterialApp(
      title: 'Kortex Remote',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      initialRoute: '/settings',
      routes: {
        '/settings': (context) => const SettingsScreen(),
        '/main': (context) => const MainControlScreen(),
        '/calibration': (context) => const CalibrationScreen(),
        '/smart_calibration': (context) => const SmartCalibrationScreen(),
      },
    );
  }

  static bool _callbacksSetup = false;

  void _setupCallbackBridge(WidgetRef ref) {
    if (_callbacksSetup) return;
    _callbacksSetup = true;

    // 立即初始化AR服务，确保AR追踪在App启动时就自动开始
    // 不依赖任何特定页面，pose_delta会自动发送到WebSocket
    ref.read(arServiceProvider);

    final wsService = ref.read(conn.webSocketServiceProvider);
    final soundService = ref.read(conn.soundServiceProvider);

    // 连接状态回调 - 同时更新 ConnectionNotifier 和播放音效
    wsService.onConnectionChanged = (connected) {
      ref.read(conn.connectionProvider.notifier).onConnectionChanged(connected);
      if (connected) {
        soundService.playConnected();
      }
    };

    // 模式回调
    wsService.onModeReceived = (mode) {
      ref.read(modeProvider.notifier).updateMode(mode);
    };

    // TCP 位姿回调
    wsService.onTcpPoseReceived = (pose) {
      ref.read(poseProvider.notifier).updatePose(pose);
    };

    // Episode 保存完成回调
    wsService.onEpisodeSaved = () {
      ref.read(taskProvider.notifier).episodeSaved();
    };

    // 任务完成回调
    wsService.onTaskComplete = () {
      soundService.playTaskComplete();
    };

    // 状态同步回调（核心 - PC 端状态的真实来源）
    wsService.onStateSync = (stateData) {
      ref.read(taskProvider.notifier).syncFromPC(stateData);
    };

    // 错误回调
    wsService.onError = (error) {
      // 错误由各页面自行监听处理
    };
  }
}
