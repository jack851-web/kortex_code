import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/tcp_pose.dart';

/// TCP位姿状态管理
class PoseState {
  final TcpPose pose;
  final DateTime lastUpdate;

  PoseState({
    TcpPose? pose,
    required this.lastUpdate,
  }) : pose = pose ?? TcpPose();

  PoseState copyWith({
    TcpPose? pose,
    DateTime? lastUpdate,
  }) {
    return PoseState(
      pose: pose ?? this.pose,
      lastUpdate: lastUpdate ?? this.lastUpdate,
    );
  }
}

class PoseNotifier extends StateNotifier<PoseState> {
  PoseNotifier() : super(PoseState(lastUpdate: DateTime.now()));

  /// 更新位姿
  void updatePose(TcpPose newPose) {
    state = state.copyWith(pose: newPose, lastUpdate: DateTime.now());
  }

  /// 清空位姿
  void clear() {
    state = PoseState(lastUpdate: DateTime.now());
  }
}

final poseProvider = StateNotifierProvider<PoseNotifier, PoseState>((ref) {
  return PoseNotifier();
});
