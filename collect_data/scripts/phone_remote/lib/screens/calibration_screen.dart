import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import '../core/theme.dart';
import '../core/constants.dart';
import '../providers/calibration_provider.dart';

/// 方向校准页面 - 参考左侧UI
class CalibrationScreen extends ConsumerStatefulWidget {
  const CalibrationScreen({super.key});

  @override
  ConsumerState<CalibrationScreen> createState() => _CalibrationScreenState();
}

class _CalibrationScreenState extends ConsumerState<CalibrationScreen> {
  @override
  void initState() {
    super.initState();
    // 开始新的校准流程
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(calibrationProvider.notifier).startCalibration();
    });
  }

  @override
  Widget build(BuildContext context) {
    final calibrationState = ref.watch(calibrationProvider);

    return Scaffold(
      backgroundColor: AppTheme.background,
      body: SafeArea(
        child: Column(
          children: [
            // 顶部导航栏
            _buildAppBar(),

            // 可滚动内容区域
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(
                    horizontal: AppTheme.pagePaddingH),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 16),

                    // 标题
                    const Text(
                      '方向校准',
                      style: TextStyle(
                        fontSize: AppTheme.fontSizeTitle + 2,
                        fontWeight: FontWeight.bold,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      '${calibrationState.data.axisMapping.length}/6',
                      style: const TextStyle(
                        fontSize: AppTheme.fontSizeCaption,
                        color: AppTheme.textSecondary,
                      ),
                    ),

                    const SizedBox(height: AppTheme.sectionGap),

                    // 说明文字区域
                    _buildInstructions(),

                    const SizedBox(height: AppTheme.sectionGap),

                    // Section标题
                    const Text(
                      '手机移动 → 机械臂响应',
                      style: TextStyle(
                        fontSize: AppTheme.fontSizeBody,
                        fontWeight: FontWeight.w500,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 12),

                    // 6个方向列表
                    ...AppConstants.calibrationDirections.map((direction) =>
                        _buildDirectionItem(direction, calibrationState)),

                    const SizedBox(height: 12),

                    // 提示文字
                    const Center(
                      child: Text(
                        '完成后自动保存，下次连接直接使用',
                        style: TextStyle(
                          fontSize: AppTheme.fontSizeCaption - 1,
                          color: AppTheme.textHint,
                        ),
                      ),
                    ),

                    const SizedBox(height: 20),

                    // 完成校准按钮（黑色填充）
                    SizedBox(
                      width: double.infinity,
                      height: AppTheme.buttonHeight,
                      child: ElevatedButton(
                        onPressed: calibrationState.data.isComplete
                            ? () async {
                                // ignore: use_build_context_synchronously
                                if (!mounted) return;
                                await ref
                                    .read(calibrationProvider.notifier)
                                    .completeCalibration();
                                // ignore: use_build_context_synchronously
                                if (!mounted) return;
                                Navigator.of(context).pop();
                              }
                            : null,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: calibrationState.data.isComplete
                              ? AppTheme.primaryBlack
                              : Colors.grey.shade400,
                          foregroundColor: Colors.white,
                        ),
                        child: const Text('完成校准'),
                      ),
                    ),

                    const SizedBox(height: 40),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// 构建顶部导航栏
  Widget _buildAppBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        children: [
          // 返回箭头
          GestureDetector(
            onTap: () {
              // 取消校准并返回
              ref.read(calibrationProvider.notifier).cancelCalibration();
              Navigator.of(context).pop();
            },
            child: SvgPicture.asset(
              'assets/icons/arrow_left.svg',
              width: 24,
              height: 24,
              colorFilter: const ColorFilter.mode(
                AppTheme.textPrimary,
                BlendMode.srcIn,
              ),
            ),
          ),
        ],
      ),
    );
  }

  /// 构建说明文字区域
  Widget _buildInstructions() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF8E1), // 浅黄色背景
        borderRadius: BorderRadius.circular(AppTheme.radiusCard),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '依次向 6 个方向移动手机',
            style: TextStyle(
              fontSize: AppTheme.fontSizeBody,
              fontWeight: FontWeight.w500,
              color: AppTheme.textPrimary,
            ),
          ),
          SizedBox(height: 6),
          Text(
            '屏幕朝向：屏幕朝上',
            style: TextStyle(
              fontSize: AppTheme.fontSizeCaption,
              color: AppTheme.textSecondary,
            ),
          ),
          SizedBox(height: 2),
          Text(
            '移动后保持机械臂响应方向',
            style: TextStyle(
              fontSize: AppTheme.fontSizeCaption,
              color: AppTheme.textSecondary,
            ),
          ),
        ],
      ),
    );
  }

  /// 构建单个方向条目
  Widget _buildDirectionItem(String direction, dynamic calibrationState) {
    final directionName = AppConstants.directionNames[direction] ?? direction;
    final axisLabel = AppConstants.directionAxisLabels[direction] ?? '';
    final isMapped = calibrationState.data.axisMapping.containsKey(direction);

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: AppTheme.cardBackground,
        borderRadius: BorderRadius.circular(AppTheme.radiusCard),
        border: Border.all(
          color: isMapped ? AppTheme.primaryGreen : const Color(0xFFE0E0E0),
          width: isMapped ? 1.5 : 1.0,
        ),
      ),
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
        title: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            // 方向名称
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  directionName,
                  style: const TextStyle(
                    fontSize: AppTheme.fontSizeBody,
                    fontWeight: FontWeight.w500,
                    color: AppTheme.textPrimary,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  axisLabel,
                  style: const TextStyle(
                    fontSize: AppTheme.fontSizeCaption - 1,
                    color: AppTheme.textSecondary,
                  ),
                ),
              ],
            ),

            // 选择按钮或已完成标记
            if (isMapped)
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: const Color(0xFFE8F5E9),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    SvgPicture.asset(
                      'assets/icons/check.svg',
                      width: 14,
                      height: 14,
                      colorFilter: const ColorFilter.mode(
                        AppTheme.primaryGreen,
                        BlendMode.srcIn,
                      ),
                    ),
                    const SizedBox(width: 4),
                    const Text(
                      '已选择',
                      style: TextStyle(
                        fontSize: AppTheme.fontSizeCaption,
                        color: AppTheme.primaryGreen,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              )
            else
              OutlinedButton(
                onPressed: () => _showDirectionSelector(direction),
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppTheme.textSecondary,
                  side: const BorderSide(color: Color(0xFFBDBDBD)),
                  minimumSize: const Size(60, 32),
                  padding: const EdgeInsets.symmetric(horizontal: 12),
                  tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                ),
                child: const Text('选择', style: TextStyle(fontSize: 13)),
              ),
          ],
        ),
      ),
    );
  }

  /// 显示方向选择对话框
  void _showDirectionSelector(String robotAxis) {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (context) => DirectionSelectorSheet(
        robotAxis: robotAxis,
        onSelected: (phoneAxis) {
          // 设置映射关系
          ref
              .read(calibrationProvider.notifier)
              .setDirectionMapping(robotAxis, phoneAxis);
          Navigator.of(context).pop();
        },
      ),
    );
  }
}

/// 方向选择底部弹窗组件
class DirectionSelectorSheet extends StatelessWidget {
  final String robotAxis;
  final Function(String) onSelected;

  const DirectionSelectorSheet({
    super.key,
    required this.robotAxis,
    required this.onSelected,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '选择 ${AppConstants.directionNames[robotAxis]} 对应的手机方向',
            style: const TextStyle(
              fontSize: AppTheme.fontSizeSection,
              fontWeight: FontWeight.w600,
              color: AppTheme.textPrimary,
            ),
          ),
          const SizedBox(height: 20),

          // 手机方向选项列表
          ...[
            'phone_x+',
            'phone_x-',
            'phone_y+',
            'phone_y-',
            'phone_z+',
            'phone_z-'
          ].map((phoneAxis) {
            final displayNames = {
              'phone_x+': '手机 X轴正向',
              'phone_x-': '手机 X轴负向',
              'phone_y+': '手机 Y轴正向',
              'phone_y-': '手机 Y轴负向',
              'phone_z+': '手机 Z轴正向',
              'phone_z-': '手机 Z轴负向',
            };

            return ListTile(
              title: Text(displayNames[phoneAxis] ?? phoneAxis),
              trailing: SvgPicture.asset(
                'assets/icons/chevron_right.svg',
                width: 16,
                height: 16,
                colorFilter: const ColorFilter.mode(
                  AppTheme.textSecondary,
                  BlendMode.srcIn,
                ),
              ),
              onTap: () => onSelected(phoneAxis.replaceFirst('phone_', '')),
            );
          }),

          const SizedBox(height: 20),
        ],
      ),
    );
  }
}
