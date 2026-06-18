import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../core/constants.dart';
import 'connection_provider.dart';
import '../services/websocket_service.dart';

/// 夹爪状态管理
class GripperState {
  final double value; // 0.0(闭合) ~ 1.0(张开)
  final bool isOpening; // 当前是否为张开状态

  const GripperState({
    this.value = AppConstants.gripperDefault,
    this.isOpening = true,
  });

  GripperState copyWith({
    double? value,
    bool? isOpening,
  }) {
    return GripperState(
      value: value ?? this.value,
      isOpening: isOpening ?? this.isOpening,
    );
  }
}

class GripperNotifier extends StateNotifier<GripperState> {
  final WebSocketService _wsService;

  GripperNotifier(this._wsService) : super(const GripperState());

  /// 张开夹爪
  void open() {
    state = state.copyWith(value: 1.0, isOpening: true);
    _wsService.sendGripper(1.0);
  }

  /// 闭合夹爪
  void close() {
    state = state.copyWith(value: 0.0, isOpening: false);
    _wsService.sendGripper(0.0);
  }

  /// 切换状态
  void toggle() {
    if (state.isOpening) {
      close();
    } else {
      open();
    }
  }
}

/// GripperProvider - 不使用.family，直接从connectionProvider获取WS服务
final gripperProvider =
    StateNotifierProvider<GripperNotifier, GripperState>((ref) {
  final wsService = ref.watch(webSocketServiceProvider);
  return GripperNotifier(wsService);
});
