import 'package:flutter_riverpod/flutter_riverpod.dart';

/// 模式状态管理（real/simu）
class ModeState {
  final String mode;  // 'real' 或 'simu'
  final bool isReceived;

  const ModeState({
    this.mode = '',
    this.isReceived = false,
  });

  ModeState copyWith({
    String? mode,
    bool? isReceived,
  }) {
    return ModeState(
      mode: mode ?? this.mode,
      isReceived: isReceived ?? this.isReceived,
    );
  }
}

class ModeNotifier extends StateNotifier<ModeState> {
  ModeNotifier() : super(const ModeState());

  /// 更新模式
  void updateMode(String newMode) {
    state = state.copyWith(mode: newMode, isReceived: true);
  }

  /// 重置模式
  void reset() {
    state = const ModeState();
  }
}

final modeProvider =
    StateNotifierProvider<ModeNotifier, ModeState>((ref) {
  return ModeNotifier();
});
