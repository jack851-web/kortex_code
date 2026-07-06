import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import '../core/theme.dart';
import '../core/constants.dart';
import '../core/storage_service.dart';
import '../providers/connection_provider.dart';
import '../providers/calibration_provider.dart';

/// 设置页面 - 参考右侧UI
class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  final TextEditingController _ipController = TextEditingController();
  final TextEditingController _portController = TextEditingController();
  bool _isConnecting = false;

  @override
  void initState() {
    super.initState();
    // 延迟初始化，确保provider已就绪
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final connectionState = ref.read(connectionProvider);
      _ipController.text = connectionState.ip;
      _portController.text = connectionState.port.toString();
    });
  }

  @override
  void dispose() {
    _ipController.dispose();
    _portController.dispose();
    super.dispose();
  }

  /// 保存并连接
  Future<void> _saveAndConnect() async {
    final ip = _ipController.text.trim();
    final port = int.tryParse(_portController.text) ?? AppConstants.defaultPort;

    // 校验 IP 不能为空
    if (ip.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text('请填写 PC 端 IP 地址（在 PC 端启动日志中查看"手机请连接到: x.x.x.x"）'),
          backgroundColor: AppTheme.accentYellowOrange,
          duration: const Duration(seconds: 3),
        ),
      );
      return;
    }

    setState(() => _isConnecting = true);

    // 更新状态
    ref.read(connectionProvider.notifier).updateIp(ip);
    ref.read(connectionProvider.notifier).updatePort(port);

    // 保存配置
    await StorageService.setString(AppConstants.keyIp, ip);
    await StorageService.setInt(AppConstants.keyPort, port);

    // 尝试连接
    final success =
        await ref.read(connectionProvider.notifier).saveAndConnect();

    if (mounted) {
      if (success) {
        // 连接成功，跳转到主控页
        Navigator.of(context).pushReplacementNamed('/main');
      } else {
        // 连接失败，停留在设置页让用户重试
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content:
                Text(ref.read(connectionProvider).error ?? '连接失败，请检查 IP 和端口'),
            backgroundColor: Colors.red,
            duration: const Duration(seconds: 3),
          ),
        );
      }
    }

    setState(() => _isConnecting = false);
  }

  /// 直接进入主页（不连接）
  void _goToMainPage() {
    // 保存当前配置
    final ip = _ipController.text.trim();
    final port = int.tryParse(_portController.text) ?? AppConstants.defaultPort;
    ref.read(connectionProvider.notifier).updateIp(ip);
    ref.read(connectionProvider.notifier).updatePort(port);
    StorageService.setString(AppConstants.keyIp, ip);
    StorageService.setInt(AppConstants.keyPort, port);

    Navigator.of(context).pushReplacementNamed('/main');
  }

  /// 跳转到校准页面
  void _goToCalibration() {
    Navigator.of(context).pushNamed('/calibration');
  }

  @override
  Widget build(BuildContext context) {
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
                    const SizedBox(height: AppTheme.sectionGap),

                    // PG 连接配置 Section
                    _buildConnectionSection(),

                    const SizedBox(height: AppTheme.sectionGap),

                    // 方向标定 Section
                    _buildCalibrationSection(),

                    const SizedBox(height: AppTheme.sectionGap),

                    // 关于 Section
                    _buildAboutSection(),

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
          // 返回箭头（如果需要）
          GestureDetector(
            onTap: () => Navigator.of(context).maybePop(),
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
          const SizedBox(width: 16),
          // 标题
          const Text(
            '设置',
            style: TextStyle(
              fontSize: AppTheme.fontSizeTitle,
              fontWeight: FontWeight.w600,
              color: AppTheme.textPrimary,
            ),
          ),
        ],
      ),
    );
  }

  /// 构建PG连接配置Section
  Widget _buildConnectionSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Section标题
        const Text(
          'PG 连接配置',
          style: TextStyle(
            fontSize: AppTheme.fontSizeCaption,
            color: AppTheme.textSecondary,
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 6),
        const Text(
          '保存后下次自动连接',
          style: TextStyle(
            fontSize: AppTheme.fontSizeCaption - 1,
            color: AppTheme.textHint,
          ),
        ),
        const SizedBox(height: 12),

        // PC IP 地址输入框
        TextField(
          controller: _ipController,
          keyboardType: const TextInputType.numberWithOptions(
              decimal: true, signed: false),
          decoration: const InputDecoration(
            labelText: 'PC IP 地址',
            labelStyle: TextStyle(color: AppTheme.textSecondary),
            hintText: '192.168.1.100',
            hintStyle: TextStyle(color: AppTheme.textHint),
          ),
          onChanged: (value) {
            ref.read(connectionProvider.notifier).updateIp(value);
          },
        ),
        const SizedBox(height: 12),

        // 端口输入框
        TextField(
          controller: _portController,
          keyboardType: TextInputType.number,
          decoration: const InputDecoration(
            labelText: '端口',
            labelStyle: TextStyle(color: AppTheme.textSecondary),
            hintText: '${AppConstants.defaultPort}',
            hintStyle: TextStyle(color: AppTheme.textHint),
          ),
          onChanged: (value) {
            final port = int.tryParse(value) ?? AppConstants.defaultPort;
            ref.read(connectionProvider.notifier).updatePort(port);
          },
        ),
        const SizedBox(height: 20),

        // 保存并连接按钮（黑色填充）
        SizedBox(
          width: double.infinity,
          height: AppTheme.buttonHeight,
          child: ElevatedButton(
            onPressed: _isConnecting ? null : _saveAndConnect,
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.primaryBlack,
              foregroundColor: Colors.white,
              disabledBackgroundColor: Colors.grey.shade400,
            ),
            child: _isConnecting
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                    ),
                  )
                : const Text('保存并连接'),
          ),
        ),
        const SizedBox(height: 10),
        // 进入主页按钮（白色边框）
        SizedBox(
          width: double.infinity,
          height: AppTheme.buttonHeight,
          child: OutlinedButton(
            onPressed: _goToMainPage,
            style: OutlinedButton.styleFrom(
              foregroundColor: AppTheme.textPrimary,
              side: const BorderSide(color: Color(0xFFBDBDBD), width: 1.5),
            ),
            child: const Text('进入主页'),
          ),
        ),
      ],
    );
  }

  /// 构建方向标定Section
  Widget _buildCalibrationSection() {
    final calibrationState = ref.watch(calibrationProvider);
    final isCalibrated = calibrationState.isCalibrated;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Section标题
        const Text(
          '方向标定',
          style: TextStyle(
            fontSize: AppTheme.fontSizeCaption,
            color: AppTheme.textSecondary,
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 12),

        // 校准状态显示或重新标定按钮
        if (isCalibrated)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            decoration: BoxDecoration(
              color: AppTheme.cardBackground,
              borderRadius: BorderRadius.circular(AppTheme.radiusCard),
              border: Border.all(color: const Color(0xFFE8F5E9)),
            ),
            child: Row(
              children: [
                // 绿色对勾图标
                SvgPicture.asset(
                  'assets/icons/check.svg',
                  width: 18,
                  height: 18,
                  colorFilter: const ColorFilter.mode(
                    AppTheme.primaryGreen,
                    BlendMode.srcIn,
                  ),
                ),
                const SizedBox(width: 10),
                // 状态文字
                Expanded(
                  child: Text(
                    '已标定 · 上次标定 ${calibrationState.lastCalibrationTime}',
                    style: const TextStyle(
                      fontSize: AppTheme.fontSizeBody,
                      color: AppTheme.textPrimary,
                    ),
                  ),
                ),
              ],
            ),
          )
        else
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            decoration: BoxDecoration(
              color: AppTheme.cardBackground,
              borderRadius: BorderRadius.circular(AppTheme.radiusCard),
              border: Border.all(color: const Color(0xFFE0E0E0)),
            ),
            child: Row(
              children: [
                SvgPicture.asset(
                  'assets/icons/alert.svg',
                  width: 18,
                  height: 18,
                  colorFilter: const ColorFilter.mode(
                    AppTheme.accentYellowOrange,
                    BlendMode.srcIn,
                  ),
                ),
                const SizedBox(width: 10),
                const Expanded(
                  child: Text(
                    '未标定',
                    style: TextStyle(
                      fontSize: AppTheme.fontSizeBody,
                      color: AppTheme.textSecondary,
                    ),
                  ),
                ),
              ],
            ),
          ),

        const SizedBox(height: 12),

        // 重新标定按钮（白色边框）
        SizedBox(
          width: double.infinity,
          height: AppTheme.buttonHeight,
          child: OutlinedButton(
            onPressed: _goToCalibration,
            style: OutlinedButton.styleFrom(
              foregroundColor: AppTheme.textPrimary,
              side: const BorderSide(color: Color(0xFFBDBDBD), width: 1.5),
            ),
            child: Text(isCalibrated ? '重新标定' : '开始标定'),
          ),
        ),

        const SizedBox(height: 8),

        // 智能标定按钮（推荐）
        SizedBox(
          width: double.infinity,
          height: AppTheme.buttonHeight,
          child: ElevatedButton(
            onPressed: () {
              Navigator.of(context).pushNamed('/smart_calibration');
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.primaryGreen,
              foregroundColor: Colors.white,
            ),
            child: const Text('智能标定（推荐）'),
          ),
        ),
      ],
    );
  }

  /// 构建关于Section
  Widget _buildAboutSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Section标题
        const Text(
          '关于',
          style: TextStyle(
            fontSize: AppTheme.fontSizeCaption,
            color: AppTheme.textSecondary,
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 12),

        // 应用信息卡片
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: AppTheme.cardBackground,
            borderRadius: BorderRadius.circular(AppTheme.radiusCard),
          ),
          child: const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '${AppConstants.appName} ${AppConstants.appVersion}',
                style: TextStyle(
                  fontSize: AppTheme.fontSizeBody,
                  fontWeight: FontWeight.w500,
                  color: AppTheme.textPrimary,
                ),
              ),
              SizedBox(height: 6),
              Text(
                AppConstants.appDescription,
                style: TextStyle(
                  fontSize: AppTheme.fontSizeCaption,
                  color: AppTheme.textSecondary,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
