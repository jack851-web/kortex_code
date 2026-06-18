import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'connection_provider.dart';
import '../services/websocket_service.dart';
import '../services/sound_service.dart';

/// 任务状态管理
enum TaskStatus {
  idle, // 空闲
  taskStarted, // 任务已开始
  episodeRunning, // Episode采集中
  taskEnded, // 任务已结束
}

class TaskState {
  final TaskStatus status;
  final String currentEpisodeId;
  final int totalEpisodes;

  const TaskState({
    this.status = TaskStatus.idle,
    this.currentEpisodeId = '',
    this.totalEpisodes = 0,
  });

  TaskState copyWith({
    TaskStatus? status,
    String? currentEpisodeId,
    int? totalEpisodes,
  }) {
    return TaskState(
      status: status ?? this.status,
      currentEpisodeId: currentEpisodeId ?? this.currentEpisodeId,
      totalEpisodes: totalEpisodes ?? this.totalEpisodes,
    );
  }

  /// 是否正在采集Episode
  bool get isEpisodeRunning => status == TaskStatus.episodeRunning;

  /// 是否可以开始Episode
  bool get canStartEpisode =>
      status == TaskStatus.taskStarted || status == TaskStatus.idle;
}

class TaskNotifier extends StateNotifier<TaskState> {
  final WebSocketService _wsService;
  final SoundService _soundService;

  TaskNotifier(this._wsService, this._soundService) : super(const TaskState());

  /// 开始任务
  void startTask() {
    _wsService.sendStartTask();
    state = state.copyWith(status: TaskStatus.taskStarted);
  }

  /// 开始/结束 Episode（切换）
  void toggleEpisode() {
    if (state.status == TaskStatus.episodeRunning) {
      // 结束当前Episode
      _wsService.sendEndEpisode();
      state = state.copyWith(status: TaskStatus.taskStarted);
    } else {
      // 开始新Episode
      final episodeId = 'ep_${state.totalEpisodes + 1}';
      _wsService.sendStartEpisode();
      state = state.copyWith(
        status: TaskStatus.episodeRunning,
        currentEpisodeId: episodeId,
        totalEpisodes: state.totalEpisodes + 1,
      );
    }
  }

  /// 结束任务
  void endTask() {
    _wsService.sendEndTask();
    state = state.copyWith(status: TaskStatus.taskEnded);
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
}

/// TaskProvider - 不使用.family，直接从connectionProvider获取WS服务
final taskProvider = StateNotifierProvider<TaskNotifier, TaskState>((ref) {
  final wsService = ref.watch(webSocketServiceProvider);
  final soundService = SoundService();
  return TaskNotifier(wsService, soundService);
});
