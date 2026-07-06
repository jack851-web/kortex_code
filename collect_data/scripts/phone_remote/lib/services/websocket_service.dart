import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../core/constants.dart';
import '../models/tcp_pose.dart';

/// WebSocket通信服务
class WebSocketService {
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  Timer? _heartbeatTimer;
  Timer? _heartbeatTimeoutTimer;
  bool _isConnected = false;

  // 回调函数
  Function(TcpPose)? onTcpPoseReceived;
  Function(String mode)? onModeReceived;
  Function()? onEpisodeSaved;
  Function()? onTaskComplete;
  Function(String error)? onError;
  Function(bool connected)? onConnectionChanged;
  Function(Map<String, dynamic> state)? onStateSync;
  Function(Map<String, dynamic> calibResult)? onCalibObsResult;
  Function(Map<String, dynamic> mappingUpdated)? onCalibMappingUpdated;

  bool get isConnected => _isConnected;

  /// 连接到服务器
  Future<bool> connect(String ip, int port) async {
    // 先清理旧连接资源，避免残留的定时器和订阅干扰新连接
    _cleanupOldConnection();

    try {
      final uri = Uri.parse('ws://$ip:$port');
      _channel = WebSocketChannel.connect(uri);

      // 给 TCP 握手加超时，避免网络不可达时长时间挂起
      await _channel!.ready.timeout(
        const Duration(seconds: 5),
        onTimeout: () => throw TimeoutException('连接超时（5秒），请检查 IP 和端口'),
      );

      _isConnected = true;
      onConnectionChanged?.call(true);

      _startListening();
      _startHeartbeat();

      return true;
    } catch (e) {
      _isConnected = false;
      // 清理半连接资源
      try {
        _channel?.sink.close();
      } catch (_) {}
      _channel = null;
      onConnectionChanged?.call(false);
      onError?.call('连接失败: $e');
      return false;
    }
  }

  /// 清理旧连接资源（定时器、订阅、channel），不触发 onConnectionChanged 回调
  void _cleanupOldConnection() {
    _heartbeatTimer?.cancel();
    _heartbeatTimer = null;
    _heartbeatTimeoutTimer?.cancel();
    _heartbeatTimeoutTimer = null;
    _subscription?.cancel();
    _subscription = null;
    if (_channel != null) {
      try {
        _channel?.sink.close();
      } catch (_) {}
      _channel = null;
    }
    _isConnected = false;
  }

  /// 断开连接
  void disconnect() {
    _heartbeatTimer?.cancel();
    _heartbeatTimer = null;
    _heartbeatTimeoutTimer?.cancel();
    _heartbeatTimeoutTimer = null;
    _subscription?.cancel();
    _subscription = null;
    _channel?.sink.close();
    _channel = null;
    _isConnected = false;
    onConnectionChanged?.call(false);
  }

  void _startListening() {
    _subscription = _channel!.stream.listen(
      (message) {
        _handleMessage(message.toString());
      },
      onError: (error) {
        onError?.call('通信错误: $error');
        disconnect();
      },
      onDone: () {
        disconnect();
      },
    );
  }

  void _handleMessage(String rawMessage) {
    try {
      final json = jsonDecode(rawMessage);
      final type = json['type'] as String? ?? '';

      switch (type) {
        case 'mode':
          final mode = json['value'] as String? ?? 'unknown';
          onModeReceived?.call(mode);
          break;

        case 'tcp_pose':
          final pose = TcpPose.fromJson(json);
          onTcpPoseReceived?.call(pose);
          break;

        case 'episode_saved':
          onEpisodeSaved?.call();
          break;

        case 'task_complete':
          onTaskComplete?.call();
          break;

        case 'state_sync':
          onStateSync?.call(json as Map<String, dynamic>);
          break;

        case 'calib_obs_result':
          debugPrint('[WS] 收到标定观察结果: $json');
          onCalibObsResult?.call(json as Map<String, dynamic>);
          break;

        case 'calib_mapping_updated':
          debugPrint('[WS] 收到映射更新: $json');
          onCalibMappingUpdated?.call(json as Map<String, dynamic>);
          break;

        case 'error':
          final errorMsg = json['message'] as String? ?? '未知错误';
          onError?.call(errorMsg);
          break;

        case 'pong':
          // 收到pong，取消超时计时
          _heartbeatTimeoutTimer?.cancel();
          _heartbeatTimeoutTimer = null;
          break;
      }
    } catch (e) {
      onError?.call('消息解析失败: $e');
    }
  }

  void send(Map<String, dynamic> message) {
    if (_isConnected && _channel != null) {
      try {
        _channel!.sink.add(jsonEncode(message));
        // 对关键命令打印发送日志，便于调试
        final type = message['type'] as String? ?? '';
        if (type == 'calibration_start' ||
            type == 'calibration_end' ||
            type == 'start_task' ||
            type == 'end_task' ||
            type == 'start_episode' ||
            type == 'end_episode') {
          debugPrint('[WS] 已发送命令: $type');
        }
      } catch (e) {
        debugPrint('[WS] 发送失败: $e');
        // 发送失败说明连接已断开
        disconnect();
      }
    } else {
      debugPrint('[WS] 未连接，无法发送: ${message['type']}');
    }
  }

  void sendPoseDelta(double dx, double dy, double dz, double droll,
      double dpitch, double dyaw) {
    send({
      'type': 'pose_delta',
      'dx': dx,
      'dy': dy,
      'dz': dz,
      'droll': droll,
      'dpitch': dpitch,
      'dyaw': dyaw,
    });
  }

  void sendGripper(double value) {
    final v = value.clamp(0.0, 1.0);
    debugPrint('[WS] 发送夹爪控制: value=$v');
    send({
      'type': 'gripper',
      'value': v,
    });
  }

  void sendAlignDirection(String axis, String phoneAxis) {
    send({
      'type': 'align_direction',
      'axis': axis,
      'phone_axis': phoneAxis,
    });
  }

  void sendStartTask() {
    send({'type': 'start_task'});
  }

  void sendStartEpisode() {
    send({'type': 'start_episode'});
  }

  void sendEndEpisode() {
    send({'type': 'end_episode'});
  }

  void sendRetryEpisode() {
    send({'type': 'retry_episode'});
  }

  void sendEndTask() {
    send({'type': 'end_task'});
  }

  void sendEmergencyStop() {
    send({'type': 'emergency_stop'});
  }

  /// 请求PC端发送当前状态同步
  void requestStateSync() {
    send({'type': 'request_state_sync'});
  }

  /// 启动心跳（含超时检测）
  void _startHeartbeat() {
    _heartbeatTimer = Timer.periodic(
      const Duration(seconds: AppConstants.heartbeatInterval),
      (_) {
        send({'type': 'ping'});

        // 启动超时计时器
        _heartbeatTimeoutTimer?.cancel();
        _heartbeatTimeoutTimer = Timer(
          const Duration(seconds: AppConstants.heartbeatTimeout),
          () {
            // 心跳超时，断开连接
            onError?.call('心跳超时，连接已断开');
            disconnect();
          },
        );
      },
    );
  }

  void dispose() {
    disconnect();
  }
}
