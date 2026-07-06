import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/storage_service.dart';
import '../../core/constants.dart';

/// 标定状态
class CalibrationState {
  final int _permIndex; // 位置排列索引 (0~5)
  final List<bool> _flips; // 位置每轴是否反转 [X, Y, Z]
  final int _rotPermIndex; // 姿态排列索引 (0~5)
  final List<bool> _rotFlips; // 姿态每轴是否反转 [X, Y, Z]
  final Map<String, String> axisMapping; // 位置映射: 手机方向 -> 机械臂方向
  final Map<String, String> rotMapping; // 姿态映射: 手机旋转 -> 机械臂旋转
  final bool isCalibrated;
  final String? lastCalibrationTime;
  final bool isCalibrating;

  const CalibrationState({
    int permIndex = 0,
    List<bool>? flips,
    int rotPermIndex = 0,
    List<bool>? rotFlips,
    this.axisMapping = const {},
    this.rotMapping = const {},
    this.isCalibrated = false,
    this.lastCalibrationTime,
    this.isCalibrating = false,
  })  : _permIndex = permIndex,
        _flips = flips ?? const [false, false, false],
        _rotPermIndex = rotPermIndex,
        _rotFlips = rotFlips ?? const [false, false, false];

  CalibrationState copyWith({
    int? permIndex,
    List<bool>? flips,
    int? rotPermIndex,
    List<bool>? rotFlips,
    Map<String, String>? axisMapping,
    Map<String, String>? rotMapping,
    bool? isCalibrated,
    String? lastCalibrationTime,
    bool? isCalibrating,
  }) {
    return CalibrationState(
      permIndex: permIndex ?? _permIndex,
      flips: flips ?? _flips,
      rotPermIndex: rotPermIndex ?? _rotPermIndex,
      rotFlips: rotFlips ?? _rotFlips,
      axisMapping: axisMapping ?? this.axisMapping,
      rotMapping: rotMapping ?? this.rotMapping,
      isCalibrated: isCalibrated ?? this.isCalibrated,
      lastCalibrationTime: lastCalibrationTime ?? this.lastCalibrationTime,
      isCalibrating: isCalibrating ?? this.isCalibrating,
    );
  }

  int get permIndex => _permIndex;
  List<bool> get flips => _flips;
  int get rotPermIndex => _rotPermIndex;
  List<bool> get rotFlips => _rotFlips;
}

/// 轴名
const List<String> axisNames = ['X', 'Y', 'Z'];
const List<String> axisLower = ['x', 'y', 'z'];
const List<String> rotNames = ['Roll', 'Pitch', 'Yaw'];
const List<String> rotLower = ['rx', 'ry', 'rz'];

/// 6种排列：phone(x,y,z) 的每个元素映射到 robot 哪个轴 (0=X,1=Y,2=Z)
const List<List<int>> permutations = [
  [0, 1, 2], // x->X, y->Y, z->Z
  [0, 2, 1], // x->X, y->Z, z->Y
  [1, 0, 2], // x->Y, y->X, z->Z
  [1, 2, 0], // x->Y, y->Z, z->X
  [2, 0, 1], // x->Z, y->X, z->Y
  [2, 1, 0], // x->Z, y->Y, z->X
];

class CalibrationNotifier extends StateNotifier<CalibrationState> {
  CalibrationNotifier() : super(const CalibrationState()) {
    _loadCalibrationData();
  }

  void _loadCalibrationData() {
    final dataList = StorageService.getStringList(AppConstants.keyCalibration);
    final rotList =
        StorageService.getStringList('${AppConstants.keyCalibration}_rot');
    final time = StorageService.getString(AppConstants.keyCalibrationTime);

    if (dataList != null && dataList.isNotEmpty) {
      final map = <String, String>{};
      for (final item in dataList) {
        final parts = item.split(':');
        if (parts.length == 2) {
          map[parts[0]] = parts[1];
        }
      }

      final rotMap = <String, String>{};
      if (rotList != null) {
        for (final item in rotList) {
          final parts = item.split(':');
          if (parts.length == 2) {
            rotMap[parts[0]] = parts[1];
          }
        }
      }

      if (map.length >= 6) {
        state = state.copyWith(
            axisMapping: map,
            rotMapping: rotMap.isNotEmpty ? rotMap : const {},
            isCalibrated: true,
            lastCalibrationTime: time);
      }
    }
  }

  void startCalibration() {
    state = state.copyWith(
        permIndex: 0,
        flips: const [false, false, false],
        rotPermIndex: 0,
        rotFlips: const [false, false, false],
        axisMapping: {},
        rotMapping: {},
        isCalibrated: false,
        isCalibrating: true);
  }

  /// 获取手机 phoneIdx 轴 映射到的机械臂方向字符串 (如 "y+")
  String getMappedDirection(int phoneIdx) {
    final robotAxisIdx = permutations[state._permIndex][phoneIdx];
    final sign = state._flips[phoneIdx] ? '-' : '+';
    return '${axisLower[robotAxisIdx]}$sign';
  }

  /// 获取手机旋转 phoneIdx 轴 映射到的机械臂旋转方向字符串 (如 "ry+")
  String getMappedRotDirection(int phoneIdx) {
    final robotAxisIdx = permutations[state._rotPermIndex][phoneIdx];
    final sign = state._rotFlips[phoneIdx] ? '-' : '+';
    return '${rotLower[robotAxisIdx]}$sign';
  }

  /// 获取当前位置映射显示文本列表
  List<String> getCurrentMappingDisplay() {
    return [
      for (int i = 0; i < 3; i++)
        '${axisNames[i]} -> ${getMappedDirection(i).toUpperCase()}',
    ];
  }

  /// 获取当前姿态映射显示文本列表
  List<String> getCurrentRotMappingDisplay() {
    return [
      for (int i = 0; i < 3; i++)
        '${rotNames[i]} -> ${getMappedRotDirection(i).toUpperCase()}',
    ];
  }

  /// 切换位置排列（循环6种）
  void switchPermutation() {
    final next = (state._permIndex + 1) % permutations.length;
    state = state.copyWith(permIndex: next);
  }

  /// 翻转位置某轴的正负
  void flipAxis(int axisIdx) {
    final newFlips = List<bool>.from(state._flips);
    newFlips[axisIdx] = !newFlips[axisIdx];
    state = state.copyWith(flips: newFlips);
  }

  /// 切换姿态排列（循环6种）
  void switchRotPermutation() {
    final next = (state._rotPermIndex + 1) % permutations.length;
    state = state.copyWith(rotPermIndex: next);
  }

  /// 翻转姿态某轴的正负
  void flipRotAxis(int axisIdx) {
    final newFlips = List<bool>.from(state._rotFlips);
    newFlips[axisIdx] = !newFlips[axisIdx];
    state = state.copyWith(rotFlips: newFlips);
  }

  /// 确认标定，生成位置+姿态映射，保存并发送
  Map<String, dynamic>? confirmCalibration() {
    final mappings = buildCurrentMappings();
    if (mappings == null) return null;

    state = state.copyWith(
      axisMapping: mappings['pos'] as Map<String, String>,
      rotMapping: mappings['rot'] as Map<String, String>,
      isCalibrated: true,
      isCalibrating: false,
    );
    _saveCalibration();
    return mappings;
  }

  /// 根据当前排列和翻转生成映射（不保存，用于实时预览/发送）
  Map<String, dynamic> buildCurrentMappings() {
    // 位置映射
    final posMapping = <String, String>{};
    for (int i = 0; i < 3; i++) {
      final dirPlus = getMappedDirection(i); // 如 "y+"
      final dirMinus = _reverseDir(dirPlus); // 如 "y-"
      posMapping['${axisLower[i]}+'] = dirPlus;
      posMapping['${axisLower[i]}-'] = dirMinus;
    }

    // 姿态映射
    final rotMapping = <String, String>{};
    for (int i = 0; i < 3; i++) {
      final dirPlus = getMappedRotDirection(i); // 如 "ry+"
      final dirMinus = _reverseDir(dirPlus); // 如 "ry-"
      rotMapping['${axisLower[i]}+'] = dirPlus;
      rotMapping['${axisLower[i]}-'] = dirMinus;
    }

    return {'pos': posMapping, 'rot': rotMapping};
  }

  String _reverseDir(String dir) {
    // 处理 "ry+" -> "ry-", "y+" -> "y-" 等
    if (dir.endsWith('+')) {
      return '${dir.substring(0, dir.length - 1)}-';
    } else {
      return '${dir.substring(0, dir.length - 1)}+';
    }
  }

  Future<void> _saveCalibration() async {
    final now = DateTime.now();
    final formattedTime =
        '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}';
    final dataList =
        state.axisMapping.entries.map((e) => '${e.key}:${e.value}').toList();
    final rotList =
        state.rotMapping.entries.map((e) => '${e.key}:${e.value}').toList();
    await StorageService.setStringList(AppConstants.keyCalibration, dataList);
    await StorageService.setStringList(
        '${AppConstants.keyCalibration}_rot', rotList);
    await StorageService.setString(
        AppConstants.keyCalibrationTime, formattedTime);
    state = state.copyWith(lastCalibrationTime: formattedTime);
  }

  void cancelCalibration() {
    _loadCalibrationData();
    state = state.copyWith(isCalibrating: false);
  }

  void recalibrate() {
    startCalibration();
  }
}

final calibrationProvider =
    StateNotifierProvider<CalibrationNotifier, CalibrationState>(
        (ref) => CalibrationNotifier());
