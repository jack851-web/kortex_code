/// 校准数据模型
class CalibrationData {
  final Map<String, String> axisMapping;  // 机械臂方向 -> 手机方向
  final DateTime calibratedAt;

  CalibrationData({
    required this.axisMapping,
    required this.calibratedAt,
  });

  // 从JSON列表解析
  factory CalibrationData.fromList(List<String> data) {
    final map = <String, String>{};
    for (final item in data) {
      final parts = item.split(':');
      if (parts.length == 2) {
        map[parts[0]] = parts[1];
      }
    }
    return CalibrationData(
      axisMapping: map,
      calibratedAt: DateTime.now(),
    );
  }

  // 转换为JSON列表存储
  List<String> toList() {
    return axisMapping.entries.map((e) => '${e.key}:${e.value}').toList();
  }

  // 是否已完成所有6个方向的标定
  bool get isComplete => axisMapping.length == 6;

  // 创建空校准数据
  factory CalibrationData.empty() {
    return CalibrationData(
      axisMapping: {},
      calibratedAt: DateTime.now(),
    );
  }
}
