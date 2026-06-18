/// TCP位姿数据模型
class TcpPose {
  final double x;
  final double y;
  final double z;
  final double rx;
  final double ry;
  final double rz;
  final DateTime timestamp;

  TcpPose({
    this.x = 0.0,
    this.y = 0.0,
    this.z = 0.0,
    this.rx = 0.0,
    this.ry = 0.0,
    this.rz = 0.0,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  // 从服务器消息创建
  factory TcpPose.fromJson(Map<String, dynamic> json) {
    return TcpPose(
      x: (json['x'] ?? 0.0).toDouble(),
      y: (json['y'] ?? 0.0).toDouble(),
      z: (json['z'] ?? 0.0).toDouble(),
      rx: (json['rx'] ?? 0.0).toDouble(),
      ry: (json['ry'] ?? 0.0).toDouble(),
      rz: (json['rz'] ?? 0.0).toDouble(),
    );
  }

  // 复制并修改
  TcpPose copyWith({
    double? x,
    double? y,
    double? z,
    double? rx,
    double? ry,
    double? rz,
  }) {
    return TcpPose(
      x: x ?? this.x,
      y: y ?? this.y,
      z: z ?? this.z,
      rx: rx ?? this.rx,
      ry: ry ?? this.ry,
      rz: rz ?? this.rz,
    );
  }

  @override
  String toString() {
    return 'TcpPose(x: ${x.toStringAsFixed(3)}, y: ${y.toStringAsFixed(3)}, '
        'z: ${z.toStringAsFixed(3)}, rx: ${rx.toStringAsFixed(2)}, '
        'ry: ${ry.toStringAsFixed(2)}, rz: ${rz.toStringAsFixed(2)})';
  }
}
