class AppConstants {
  AppConstants._();

  // 默认连接配置
  // defaultIp 留空，强制用户填写 PC 局域网 IP（避免硬编码错误 IP 误导）
  static const String defaultIp = '';
  static const int defaultPort = 8765;

  // WebSocket配置
  static const int heartbeatInterval = 3; // 心跳间隔（秒）
  static const int heartbeatTimeout = 15; // 心跳超时（秒）

  // ARCore配置
  static const int arcoreUpdateFrequency = 60; // ARCore更新频率(Hz)

  // 夹爪控制范围
  static const double gripperMin = 0.0; // 完全打开
  static const double gripperMax = 1.0; // 完全闭合
  static const double gripperDefault = 0.0; // 默认完全打开
  static const double gripperStep = 0.05; // 每次增量步长

  // 校准方向列表
  static const List<String> calibrationDirections = [
    'Y+', // 手机向上
    'Y-', // 手机向下
    'X-', // 手机向左
    'X+', // 手机向右
    'Z+', // 手机向前
    'Z-', // 手机向后
  ];

  // 校准方向显示名称
  static const Map<String, String> directionNames = {
    'Y+': '手机向上',
    'Y-': '手机向下',
    'X-': '手机向左',
    'X+': '手机向右',
    'Z+': '手机向前',
    'Z-': '手机向后',
  };

  // 校准方向轴说明
  static const Map<String, String> directionAxisLabels = {
    'Y+': 'Y+ 方向',
    'Y-': 'Y- 方向',
    'X-': 'X- 方向',
    'X+': 'X+ 方向',
    'Z+': 'Z+ 方向',
    'Z-': 'Z- 方向',
  };

  // 应用信息
  static const String appName = 'Kortex 遥操作控制';
  static const String appVersion = 'v0.1.0';
  static const String appDescription = '连接至手机端将可进行数据收集';

  // 存储键名
  static const String keyIp = 'saved_ip';
  static const String keyPort = 'saved_port';
  static const String keyCalibration = 'calibration_data';
  static const String keyCalibrationTime = 'calibration_time';
}
