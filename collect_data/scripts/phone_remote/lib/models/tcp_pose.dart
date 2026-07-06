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
    // 注意：JSON 中整数（如 0）会被 Dart 解析为 int，直接调用 .toDouble() 会抛异常，
    // 必须先转 num 再 toDouble()
    double parseNum(dynamic v) => (v as num? ?? 0.0).toDouble();
    return TcpPose(
      x: parseNum(json['x']),
      y: parseNum(json['y']),
      z: parseNum(json['z']),
      rx: parseNum(json['rx']),
      ry: parseNum(json['ry']),
      rz: parseNum(json['rz']),
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
