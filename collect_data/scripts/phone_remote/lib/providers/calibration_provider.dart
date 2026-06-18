import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/storage_service.dart';
import '../../core/constants.dart';
import '../../models/calibration_data.dart';

/// 校准状态管理
class CalibrationState {
  final CalibrationData data;
  final String? lastCalibrationTime;
  final bool isInProgress;

  const CalibrationState({
    required this.data,
    this.lastCalibrationTime,
    this.isInProgress = false,
  });

  CalibrationState copyWith({
    CalibrationData? data,
    String? lastCalibrationTime,
    bool? isInProgress,
    bool clearData = false,
  }) {
    return CalibrationState(
      data: clearData ? CalibrationData.empty() : (data ?? this.data),
      lastCalibrationTime: lastCalibrationTime ?? this.lastCalibrationTime,
      isInProgress: isInProgress ?? this.isInProgress,
    );
  }

  /// 是否已完成标定
  bool get isCalibrated => data.isComplete && !isInProgress;
}

class CalibrationNotifier extends StateNotifier<CalibrationState> {
  CalibrationNotifier() : super(CalibrationState(data: CalibrationData.empty())) {
    _loadCalibrationData();
  }

  /// 加载本地存储的校准数据
  void _loadCalibrationData() {
    final dataList = StorageService.getStringList(AppConstants.keyCalibration);
    final time = StorageService.getString(AppConstants.keyCalibrationTime);

    if (dataList != null && dataList.isNotEmpty) {
      final calibrationData = CalibrationData.fromList(dataList);
      state = state.copyWith(
        data: calibrationData,
        lastCalibrationTime: time,
      );
    }
  }

  /// 开始新的校准流程
  void startCalibration() {
    state = state.copyWith(
      isInProgress: true,
      clearData: true,
    );
  }

  /// 设置某个方向的映射关系
  void setDirectionMapping(String robotAxis, String phoneAxis) {
    final newMapping = Map<String, String>.from(state.data.axisMapping);
    newMapping[robotAxis] = phoneAxis;

    state = state.copyWith(
      data: CalibrationData(
        axisMapping: newMapping,
        calibratedAt: DateTime.now(),
      ),
    );
  }

  /// 完成校准并保存
  Future<void> completeCalibration() async {
    final now = DateTime.now();
    final formattedTime = '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}';

    // 保存到本地存储
    await StorageService.setStringList(
      AppConstants.keyCalibration,
      state.data.toList(),
    );
    await StorageService.setString(
      AppConstants.keyCalibrationTime,
      formattedTime,
    );

    state = state.copyWith(
      isInProgress: false,
      lastCalibrationTime: formattedTime,
    );
  }

  /// 取消当前校准
  void cancelCalibration() {
    _loadCalibrationData();  // 恢复之前的数据
    state = state.copyWith(isInProgress: false);
  }

  /// 重新校准
  void recalibrate() {
    startCalibration();
  }
}

final calibrationProvider =
    StateNotifierProvider<CalibrationNotifier, CalibrationState>((ref) {
  return CalibrationNotifier();
});
