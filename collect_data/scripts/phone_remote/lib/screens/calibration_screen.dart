import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import '../core/theme.dart';
import '../providers/calibration_provider.dart';
import '../providers/connection_provider.dart';
import '../providers/ar_provider.dart';

/// 方向标定页面
/// 位置映射 + 姿态映射，各自独立排列/翻转
class CalibrationScreen extends ConsumerStatefulWidget {
  const CalibrationScreen({super.key});

  @override
  ConsumerState<CalibrationScreen> createState() => _CalibrationScreenState();
}

class _CalibrationScreenState extends ConsumerState<CalibrationScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final connState = ref.read(connectionProvider);
      if (!connState.isConnected) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('请先连接到PC端'),
            backgroundColor: Colors.red,
          ),
        );
        Navigator.of(context).pop();
        return;
      }
      ref.read(calibrationProvider.notifier).startCalibration();
      _sendCalibrationStart();
    });
  }

  void _sendCalibrationStart() {
    final wsService = ref.read(webSocketServiceProvider);
    final connState = ref.read(connectionProvider);
    if (connState.isConnected) {
      wsService.send({'type': 'calibration_start'});
      debugPrint('[Calibration] 发送calibration_start命令');
    } else {
      debugPrint('[Calibration] WebSocket未连接，无法发送calibration_start');
    }
  }

  void _sendCalibrationEnd() {
    final wsService = ref.read(webSocketServiceProvider);
    debugPrint('[Calibration] 发送calibration_end命令');
    print('[Calibration] 发送calibration_end命令', ); // 确保命令发送可观测
    wsService.send({'type': 'calibration_end'});
  }

  void _sendAllMappings(Map<String, dynamic> mappings) {
    final wsService = ref.read(webSocketServiceProvider);
    // 发送位置映射
    final posMapping = mappings['pos'] as Map<String, String>;
    for (final entry in posMapping.entries) {
      wsService.sendAlignDirection(entry.value, entry.key);
    }
    // 发送姿态映射：统一使用 sendAlignDirection
    // axis=机械臂旋转轴 (如 "rx+")，phone_axis=手机旋转轴 (如 "x+")
    final rotMapping = mappings['rot'] as Map<String, String>;
    for (final entry in rotMapping.entries) {
      wsService.sendAlignDirection(entry.value, entry.key);
    }
  }

  /// 实时发送当前映射到PC端（标定过程中每次调整后调用）
  void _sendCurrentMappings() {
    final notifier = ref.read(calibrationProvider.notifier);
    final mappings = notifier.buildCurrentMappings();
    if (mappings != null) {
      _sendAllMappings(mappings);
    }
  }

  @override
  void dispose() {
    final calState = ref.read(calibrationProvider);
    if (calState.isCalibrating) {
      _sendCalibrationEnd();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final calState = ref.watch(calibrationProvider);
    final arState = ref.watch(arServiceProvider);

    return Scaffold(
      backgroundColor: AppTheme.background,
      body: SafeArea(
        child: Column(
          children: [
            _buildAppBar(),
            _buildARStatusBar(arState),
            Expanded(
              child: calState.isCalibrated && !calState.isCalibrating
                  ? _buildCompletedView(calState)
                  : _buildCalibrationView(calState),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildARStatusBar(ARState arState) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: arState.isTracking
            ? const Color(0xFFE8F5E9)
            : const Color(0xFFFFF3E0),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: arState.isTracking
                  ? AppTheme.primaryGreen
                  : AppTheme.accentYellowOrange,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              arState.isTracking
                  ? 'AR追踪中 - 移动手机观察机械臂方向变化'
                  : '等待AR追踪...请移动手机扫描环境',
              style: TextStyle(
                fontSize: 13,
                color: arState.isTracking
                    ? AppTheme.primaryGreen
                    : AppTheme.textSecondary,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAppBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        children: [
          GestureDetector(
            onTap: () {
              ref.read(calibrationProvider.notifier).cancelCalibration();
              _sendCalibrationEnd();
              Navigator.of(context).pop();
            },
            child: SvgPicture.asset('assets/icons/arrow_left.svg',
                width: 24,
                height: 24,
                colorFilter: const ColorFilter.mode(
                    AppTheme.textPrimary, BlendMode.srcIn)),
          ),
          const SizedBox(width: 12),
          const Text('方向标定',
              style: TextStyle(
                fontSize: AppTheme.fontSizeSection,
                fontWeight: FontWeight.w600,
                color: AppTheme.textPrimary,
              )),
        ],
      ),
    );
  }

  // ========== 标定完成视图 ==========
  Widget _buildCompletedView(CalibrationState calState) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
                width: 72,
                height: 72,
                decoration: const BoxDecoration(
                    color: Color(0xFFE8F5E9), shape: BoxShape.circle),
                child: Center(
                    child: SvgPicture.asset('assets/icons/check.svg',
                        width: 36,
                        height: 36,
                        colorFilter: const ColorFilter.mode(
                            AppTheme.primaryGreen, BlendMode.srcIn)))),
            const SizedBox(height: 20),
            const Text('标定完成',
                style: TextStyle(
                    fontSize: AppTheme.fontSizeTitle,
                    fontWeight: FontWeight.bold,
                    color: AppTheme.textPrimary)),
            const SizedBox(height: 16),
            ..._buildMappingList(calState.axisMapping),
            const SizedBox(height: 32),
            SizedBox(
              width: double.infinity,
              height: AppTheme.buttonHeight,
              child: OutlinedButton(
                onPressed: () {
                  ref.read(calibrationProvider.notifier).recalibrate();
                  _sendCalibrationStart();
                },
                style: OutlinedButton.styleFrom(
                    foregroundColor: AppTheme.primaryBlack,
                    side: const BorderSide(color: AppTheme.primaryBlack)),
                child: const Text('重新标定'),
              ),
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              height: AppTheme.buttonHeight,
              child: ElevatedButton(
                onPressed: () => Navigator.of(context).pop(),
                style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primaryBlack,
                    foregroundColor: Colors.white),
                child: const Text('返回'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  List<Widget> _buildMappingList(Map<String, String> mapping) {
    final widgets = <Widget>[];
    for (final entry in mapping.entries) {
      if (!entry.key.endsWith('+')) continue;
      widgets.add(Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Text('${entry.key} -> ${entry.value}',
            style: const TextStyle(
                fontSize: AppTheme.fontSizeBody,
                color: AppTheme.textSecondary)),
      ));
    }
    return widgets;
  }

  // ========== 标定进行中视图 ==========
  Widget _buildCalibrationView(CalibrationState calState) {
    final notifier = ref.read(calibrationProvider.notifier);
    final posDisplayLines = notifier.getCurrentMappingDisplay();
    final rotDisplayLines = notifier.getCurrentRotMappingDisplay();

    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: AppTheme.pagePaddingH),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // ---- 位置映射区域 ----
          _buildSectionHeader('位置映射', calState.permIndex),
          const SizedBox(height: 8),
          _buildMappingCard(posDisplayLines),
          const SizedBox(height: 8),
          _buildFlipRow(notifier, false),
          const SizedBox(height: 6),
          Center(
            child: SizedBox(
              width: double.infinity,
              height: 36,
              child: OutlinedButton(
                onPressed: () {
                  setState(() => notifier.switchPermutation());
                  _sendCurrentMappings();
                },
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppTheme.accentYellowOrange,
                  side: const BorderSide(color: AppTheme.accentYellowOrange),
                ),
                child: const Text('切换位置排列', style: TextStyle(fontSize: 13)),
              ),
            ),
          ),

          const SizedBox(height: 20),

          // ---- 姿态映射区域 ----
          _buildSectionHeader('姿态映射', calState.rotPermIndex),
          const SizedBox(height: 8),
          _buildMappingCard(rotDisplayLines),
          const SizedBox(height: 8),
          _buildFlipRow(notifier, true),
          const SizedBox(height: 6),
          Center(
            child: SizedBox(
              width: double.infinity,
              height: 36,
              child: OutlinedButton(
                onPressed: () {
                  setState(() => notifier.switchRotPermutation());
                  _sendCurrentMappings();
                },
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppTheme.accentYellowOrange,
                  side: const BorderSide(color: AppTheme.accentYellowOrange),
                ),
                child: const Text('切换姿态排列', style: TextStyle(fontSize: 13)),
              ),
            ),
          ),

          const SizedBox(height: 24),

          // ---- 底部操作按钮 ----
          SizedBox(
            width: double.infinity,
            height: AppTheme.buttonHeight,
            child: ElevatedButton(
              onPressed: () {
                final mappings = notifier.confirmCalibration();
                if (mappings != null) {
                  _sendAllMappings(mappings);
                  _sendCalibrationEnd();
                  Navigator.of(context).pop();
                }
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: AppTheme.primaryGreen,
                foregroundColor: Colors.white,
              ),
              child: const Text('确认完成'),
            ),
          ),
          const SizedBox(height: 10),
          SizedBox(
            width: double.infinity,
            height: AppTheme.buttonHeight,
            child: OutlinedButton(
              onPressed: () {
                notifier.cancelCalibration();
                _sendCalibrationEnd();
                Navigator.of(context).pop();
              },
              style: OutlinedButton.styleFrom(
                foregroundColor: AppTheme.textSecondary,
                side: const BorderSide(color: AppTheme.textSecondary),
              ),
              child: const Text('退出标定'),
            ),
          ),
          const SizedBox(height: 32),
        ],
      ),
    );
  }

  /// 区域标题 + 方案序号
  Widget _buildSectionHeader(String title, int permIndex) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(title,
            style: const TextStyle(
                fontSize: AppTheme.fontSizeSection,
                fontWeight: FontWeight.w600,
                color: AppTheme.textPrimary)),
        Text('方案 ${permIndex + 1}/6',
            style: const TextStyle(
                fontSize: AppTheme.fontSizeCaption,
                color: AppTheme.textSecondary)),
      ],
    );
  }

  /// 映射显示卡片
  Widget _buildMappingCard(List<String> lines) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F5F5),
        borderRadius: BorderRadius.circular(AppTheme.radiusCard),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: lines
            .map((line) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 3),
                  child: Text(
                    line,
                    style: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                        color: AppTheme.textPrimary),
                  ),
                ))
            .toList(),
      ),
    );
  }

  /// 轴翻转按钮行
  Widget _buildFlipRow(CalibrationNotifier notifier, bool isRotation) {
    final calState = ref.read(calibrationProvider);
    final flips = isRotation ? calState.rotFlips : calState.flips;
    final names = isRotation ? rotNames : axisNames;
    return Row(
      children: List.generate(3, (i) {
        final isFlipped = flips[i];
        return Expanded(
          child: Padding(
            padding: EdgeInsets.only(left: i > 0 ? 8 : 0),
            child: SizedBox(
              height: 34,
              child: ElevatedButton(
                onPressed: () {
                  setState(() {
                    if (isRotation) {
                      notifier.flipRotAxis(i);
                    } else {
                      notifier.flipAxis(i);
                    }
                  });
                  _sendCurrentMappings();
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: isFlipped
                      ? AppTheme.accentYellowOrange
                      : AppTheme.cardBackground,
                  foregroundColor:
                      isFlipped ? Colors.white : AppTheme.textPrimary,
                  padding: const EdgeInsets.symmetric(horizontal: 4),
                  elevation: 0,
                ),
                child: FittedBox(
                  child: Text('${names[i]} ${isFlipped ? '-' : '+'}',
                      style: const TextStyle(fontSize: 13)),
                ),
              ),
            ),
          ),
        );
      }),
    );
  }
}
