import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../core/constants.dart';
import '../core/storage_service.dart';
import '../services/websocket_service.dart';

/// 连接状态管理
class ConnectionState {
  final String ip;
  final int port;
  final bool isConnected;
  final bool isConnecting;
  final String? error;

  const ConnectionState({
    this.ip = AppConstants.defaultIp,
    this.port = AppConstants.defaultPort,
    this.isConnected = false,
    this.isConnecting = false,
    this.error,
  });

  ConnectionState copyWith({
    String? ip,
    int? port,
    bool? isConnected,
    bool? isConnecting,
    String? error,
  }) {
    return ConnectionState(
      ip: ip ?? this.ip,
      port: port ?? this.port,
      isConnected: isConnected ?? this.isConnected,
      isConnecting: isConnecting ?? this.isConnecting,
      error: error,
    );
  }
}

class ConnectionNotifier extends StateNotifier<ConnectionState> {
  final WebSocketService _wsService;

  ConnectionNotifier(this._wsService) : super(const ConnectionState()) {
    _loadSavedConfig();
    _wsService.onConnectionChanged = _onConnectionChanged;
  }

  /// 暴露WebSocketService实例供外部使用
  WebSocketService get wsService => _wsService;

  void _loadSavedConfig() {
    final savedIp = StorageService.getString(
      AppConstants.keyIp,
      defaultValue: AppConstants.defaultIp,
    );
    final savedPort = StorageService.getInt(
      AppConstants.keyPort,
      defaultValue: AppConstants.defaultPort,
    );
    state = state.copyWith(ip: savedIp, port: savedPort);
  }

  void updateIp(String ip) {
    state = state.copyWith(ip: ip);
  }

  void updatePort(int port) {
    state = state.copyWith(port: port);
  }

  Future<bool> saveAndConnect() async {
    state = state.copyWith(isConnecting: true, error: null);

    await StorageService.setString(AppConstants.keyIp, state.ip);
    await StorageService.setInt(AppConstants.keyPort, state.port);

    final success = await _wsService.connect(state.ip, state.port);

    if (success) {
      state = state.copyWith(isConnecting: false, isConnected: true);
    } else {
      state = state.copyWith(
        isConnecting: false,
        isConnected: false,
        error: '连接失败，请检查IP和端口',
      );
    }

    return success;
  }

  void disconnect() {
    _wsService.disconnect();
    state = state.copyWith(isConnected: false);
  }

  void _onConnectionChanged(bool connected) {
    state = state.copyWith(isConnected: connected);
  }

  @override
  void dispose() {
    _wsService.dispose();
    super.dispose();
  }
}

final connectionProvider =
    StateNotifierProvider<ConnectionNotifier, ConnectionState>((ref) {
  final wsService = WebSocketService();
  return ConnectionNotifier(wsService);
});

/// 暴露WebSocketService实例的provider
final webSocketServiceProvider = Provider<WebSocketService>((ref) {
  return ref.watch(connectionProvider.notifier).wsService;
});
