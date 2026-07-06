import 'dart:async';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../core/constants.dart';
import 'connection_provider.dart';
import '../services/websocket_service.dart';

/// 夹爪状态管理
class GripperState {
  final double value; // 0.0(完全打开) ~ 1.0(完全闭合)
  final bool isHoldingOpen; // 是否正在按住"打开"
  final bool isHoldingClose; // 是否正在按住"闭合"

  const GripperState({
    this.value = AppConstants.gripperDefault,
    this.isHoldingOpen = false,
    this.isHoldingClose = false,
  });

  GripperState copyWith({
    double? value,
    bool? isHoldingOpen,
    bool? isHoldingClose,
  }) {
    return GripperState(
      value: value ?? this.value,
      isHoldingOpen: isHoldingOpen ?? this.isHoldingOpen,
      isHoldingClose: isHoldingClose ?? this.isHoldingClose,
    );
  }
}

class GripperNotifier extends StateNotifier<GripperState> {
  final WebSocketService _wsService;
  Timer? _repeatTimer;

  GripperNotifier(this._wsService) : super(const GripperState());

  /// 开始持续张开（按住时调用）
  void startOpen() {
    state = state.copyWith(isHoldingOpen: true, isHoldingClose: false);
    _startRepeatTimer(decrement: true);
  }

  /// 开始持续闭合（按住时调用）
  void startClose() {
    state = state.copyWith(isHoldingOpen: false, isHoldingClose: true);
    _startRepeatTimer(decrement: false);
  }

  /// 停止持续操作（松开时调用）
  void stop() {
    state = state.copyWith(isHoldingOpen: false, isHoldingClose: false);
    _repeatTimer?.cancel();
    _repeatTimer = null;
  }

  /// 启动持续增减定时器
  void _startRepeatTimer({required bool decrement}) {
    _repeatTimer?.cancel();
    // 立即执行一次
    _applyStep(decrement);
    // 每50ms执行一次（约20Hz）
    _repeatTimer = Timer.periodic(const Duration(milliseconds: 50), (_) {
      _applyStep(decrement);
    });
  }

  /// 应用一步增量
  void _applyStep(bool decrement) {
    double newValue;
    if (decrement) {
      // 张开：value 减小（0=完全打开）
      newValue = (state.value - AppConstants.gripperStep)
          .clamp(AppConstants.gripperMin, AppConstants.gripperMax);
    } else {
      // 闭合：value 增大（1=完全闭合）
      newValue = (state.value + AppConstants.gripperStep)
          .clamp(AppConstants.gripperMin, AppConstants.gripperMax);
    }

    if (newValue != state.value) {
      state = state.copyWith(value: newValue);
      _wsService.sendGripper(newValue);
    }
  }

  @override
  void dispose() {
    _repeatTimer?.cancel();
    super.dispose();
  }
}

/// GripperProvider
final gripperProvider =
    StateNotifierProvider<GripperNotifier, GripperState>((ref) {
  final wsService = ref.watch(webSocketServiceProvider);
  return GripperNotifier(wsService);
});
