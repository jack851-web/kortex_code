import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'connection_provider.dart';
import '../services/websocket_service.dart';
import '../services/sound_service.dart';

/// 任务状态（与 PC 端严格对齐）
enum TaskStatus {
  idle, // 空闲 - 未开启收集
  collectionStarted, // 已开启收集，等待开启任务/标定
  preparing, // 准备中 - PC端正在创建仿真场景
  calibrating, // 标定模式中（仿真已启动，可移动手机调试方向）
  episodeRunning, // 任务执行中（仿真已启动、已对齐、控制循环运行，可遥操作，数据录制中）
}

/// 统一任务状态管理
/// PC 端是状态的唯一真实来源，手机端通过 state_sync 消息同步状态
class TaskState {
  final TaskStatus status;
  final int currentEpisode;
  final int taskIndex;
  final bool taskInProgress;
  final bool collecting;
  final String mode; // simu / real
  final bool simuReady; // 仿真是否已启动就绪
  final bool aligned; // 是否已完成对齐
  final bool controlRunning; // 控制循环是否正在运行
  final bool canControl; // 是否可以控制机械臂（所有条件满足）

  const TaskState({
    this.status = TaskStatus.idle,
    this.currentEpisode = 0,
    this.taskIndex = 0,
    this.taskInProgress = false,
    this.collecting = false,
    this.mode = 'simu',
    this.simuReady = false,
    this.aligned = false,
    this.controlRunning = false,
    this.canControl = false,
  });

  TaskState copyWith({
    TaskStatus? status,
    int? currentEpisode,
    int? taskIndex,
    bool? taskInProgress,
    bool? collecting,
    String? mode,
    bool? simuReady,
    bool? aligned,
    bool? controlRunning,
    bool? canControl,
  }) {
    return TaskState(
      status: status ?? this.status,
      currentEpisode: currentEpisode ?? this.currentEpisode,
      taskIndex: taskIndex ?? this.taskIndex,
      taskInProgress: taskInProgress ?? this.taskInProgress,
      collecting: collecting ?? this.collecting,
      mode: mode ?? this.mode,
      simuReady: simuReady ?? this.simuReady,
      aligned: aligned ?? this.aligned,
      controlRunning: controlRunning ?? this.controlRunning,
      canControl: canControl ?? this.canControl,
    );
  }

  // ========== 按钮启用条件（严格按照流程） ==========

  /// 是否可以开启收集（空闲状态下）
  bool get canStartCollection => status == TaskStatus.idle;

  /// 是否可以开启任务（收集已开启，等待执行任务）
  bool get canStartTask => status == TaskStatus.collectionStarted;

  /// 是否可以结束任务（任务执行中）
  bool get canEndTask => status == TaskStatus.episodeRunning;

  /// 是否可以重做任务（任务执行中）
  bool get canRetryTask => status == TaskStatus.episodeRunning;

  /// 是否可以结束收集（收集已开启、准备中、任务执行中都可以结束）
  bool get canEndCollection =>
      status == TaskStatus.collectionStarted ||
      status == TaskStatus.preparing ||
      status == TaskStatus.episodeRunning;

  /// 是否任务执行中
  bool get isTaskRunning => status == TaskStatus.episodeRunning;

  /// 是否在标定模式
  bool get isCalibrating => status == TaskStatus.calibrating;

  /// 是否在准备中
  bool get isPreparing => status == TaskStatus.preparing;

  /// 是否应该发送pose_delta给PC（仅标定中或任务执行中发送）
  bool get shouldSendPoseDelta =>
      status == TaskStatus.calibrating || status == TaskStatus.episodeRunning;

  /// 状态提示文本
  String get statusText {
    switch (status) {
      case TaskStatus.idle:
        return '未开启收集，请先点击「开启收集」';
      case TaskStatus.collectionStarted:
        return '收集已开启，可以开始标定坐标或开启任务';
      case TaskStatus.preparing:
        return '正在准备场景，请稍候...';
      case TaskStatus.calibrating:
        return simuReady
            ? (aligned ? '标定中：移动手机调整机械臂方向' : '标定中：正在对齐...')
            : '标定中：正在启动仿真...';
      case TaskStatus.episodeRunning:
        return canControl ? '任务执行中：移动手机控制机械臂，数据录制中' : '任务执行中：正在初始化控制...';
    }
  }
}

class TaskNotifier extends StateNotifier<TaskState> {
  final WebSocketService _wsService;
  final SoundService _soundService;

  TaskNotifier(this._wsService, this._soundService) : super(const TaskState());

  /// 开启收集（发送start_task命令给PC，开启整个收集流程）
  void startCollection() {
    _wsService.sendStartTask();
  }

  /// 开启任务（发送start_episode命令给PC，开启单条任务）
  void startTask() {
    _wsService.sendStartEpisode();
  }

  /// 结束任务（发送end_episode命令给PC，结束当前任务保存数据）
  void endTask() {
    _wsService.sendEndEpisode();
  }

  /// 重做任务（发送retry_episode命令给PC，丢弃当前数据重新执行）
  void retryTask() {
    _wsService.sendRetryEpisode();
  }

  /// 结束收集（发送end_task命令给PC，结束整个收集流程）
  void endCollection() {
    _wsService.sendEndTask();
  }

  /// 急停
  void emergencyStop() {
    _wsService.sendEmergencyStop();
    _soundService.playEmergency();
    state = state.copyWith(status: TaskStatus.idle);
  }

  /// Episode保存完成（由WebSocket回调触发）
  void episodeSaved() {
    _soundService.playEpisodeSaved();
  }

  /// PC端状态同步（由WebSocket state_sync消息触发）
  /// 这是手机端状态更新的唯一来源
  void syncFromPC(Map<String, dynamic> pcStateData) {
    final pcState = pcStateData['state'] as String? ?? 'idle';
    final pcEpisode = pcStateData['episode'] as int? ?? 0;
    final pcTaskIndex = pcStateData['task_index'] as int? ?? 0;
    final pcTaskInProgress = pcStateData['task_in_progress'] as bool? ?? false;
    final pcCollecting = pcStateData['collecting'] as bool? ?? false;
    final pcSimuReady = pcStateData['simu_ready'] as bool? ?? false;
    final pcAligned = pcStateData['aligned'] as bool? ?? false;
    final pcControlRunning = pcStateData['control_running'] as bool? ?? false;
    final pcCanControl = pcStateData['can_control'] as bool? ?? false;

    // 调试日志：状态变化时打印
    debugPrint(
        '[TaskProvider] 收到state_sync: state=$pcState, aligned=$pcAligned, '
        'simuReady=$pcSimuReady, controlRunning=$pcControlRunning, canControl=$pcCanControl');

    TaskStatus newStatus;
    switch (pcState) {
      case 'idle':
        newStatus = TaskStatus.idle;
        break;
      case 'collection_started':
        newStatus = TaskStatus.collectionStarted;
        break;
      case 'preparing':
        newStatus = TaskStatus.preparing;
        break;
      case 'calibrating':
        newStatus = TaskStatus.calibrating;
        break;
      case 'episode_running':
        newStatus = TaskStatus.episodeRunning;
        break;
      default:
        newStatus = TaskStatus.idle;
    }

    state = state.copyWith(
      status: newStatus,
      currentEpisode: pcEpisode,
      taskIndex: pcTaskIndex,
      taskInProgress: pcTaskInProgress,
      collecting: pcCollecting,
      simuReady: pcSimuReady,
      aligned: pcAligned,
      controlRunning: pcControlRunning,
      canControl: pcCanControl,
    );
  }

  /// 进入标定模式（发送命令给PC，等PC同步状态回来）
  void enterCalibration() {
    _wsService.send({'type': 'calibration_start'});
  }

  /// 退出标定模式（发送命令给PC，等PC同步状态回来）
  void exitCalibration() {
    _wsService.send({'type': 'calibration_end'});
  }

  /// 更新模式
  void updateMode(String mode) {
    state = state.copyWith(mode: mode);
  }
}

/// TaskProvider
final taskProvider = StateNotifierProvider<TaskNotifier, TaskState>((ref) {
  final wsService = ref.watch(webSocketServiceProvider);
  final soundService = ref.watch(soundServiceProvider);
  return TaskNotifier(wsService, soundService);
});
