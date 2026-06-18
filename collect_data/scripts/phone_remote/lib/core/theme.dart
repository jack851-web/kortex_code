import 'package:flutter/material.dart';

class AppTheme {
  // 私有构造函数，防止实例化
  AppTheme._();

  // 主色调（严格匹配参考截图）
  static const Color primaryGreen = Color(0xFF4CAF50);       // 张开按钮 / 开始任务边框
  static const Color primaryBlack = Color(0xFF212121);        // 闭合按钮 / 主要填充按钮
  static const Color primaryRed = Color(0xFFF44336);          // 急停按钮
  static const Color accentYellowOrange = Color(0xFFFFB300);  // 开始/结束 Ep 边框
  static const Color accentPurple = Color(0xFF9C27B0);        // 结束任务边框

  // 状态颜色
  static const Color dotGreen = Color(0xFF4CAF50);            // 已连接绿点
  static const Color dotRed = Color(0xFFE57373);              // 未连接红点

  // 背景色
  static const Color background = Color(0xFFFAFAFA);          // 页面背景（接近白色）
  static const Color cardBackground = Colors.white;            // 卡片/输入框背景

  // 文字颜色
  static const Color textPrimary = Color(0xFF212121);          // 主要文字
  static const Color textSecondary = Color(0xFF757575);        // 次要文字
  static const Color textHint = Color(0xFFBDBDBD);             // 提示文字

  // 字体大小
  static const double fontSizeTitle = 18.0;      // 页面标题
  static const double fontSizeSection = 16.0;    // Section 标题
  static const double fontSizeBody = 14.0;       // 正文
  static const double fontSizeCaption = 12.0;    // 副标题/说明
  static const double fontSizeValue = 15.0;      // TCP 数值
  static const double fontSizeButton = 16.0;     // 按钮文字

  // 圆角
  static const double radiusCard = 8.0;          // 卡片/输入框
  static const double radiusButton = 8.0;        // 按钮

  // 间距
  static const double pagePaddingH = 20.0;       // 页面水平内边距
  static const double sectionGap = 28.0;         // Section 之间间距
  static const double itemGapV = 8.0;            // 条目垂直间距
  static const double buttonHeight = 48.0;       // 标准按钮高度

  // 浅色主题
  static ThemeData get lightTheme => ThemeData(
    useMaterial3: true,
    brightness: Brightness.light,
    primaryColor: primaryGreen,
    scaffoldBackgroundColor: background,
    colorScheme: const ColorScheme.light(
      primary: primaryGreen,
      secondary: accentYellowOrange,
      error: primaryRed,
      surface: cardBackground,
      onPrimary: Colors.white,
      onSecondary: Colors.white,
      onError: Colors.white,
      onSurface: textPrimary,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: Colors.transparent,
      elevation: 0,
      foregroundColor: textPrimary,
      titleTextStyle: TextStyle(
        color: textPrimary,
        fontSize: fontSizeTitle,
        fontWeight: FontWeight.w600,
      ),
    ),
    cardTheme: CardThemeData(
      color: cardBackground,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(radiusCard),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: cardBackground,
      contentPadding: const EdgeInsets.symmetric(
        horizontal: 16,
        vertical: 14,
      ),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radiusCard),
        borderSide: BorderSide.none,
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radiusCard),
        borderSide: const BorderSide(color: Color(0xFFE0E0E0), width: 1),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radiusCard),
        borderSide: const BorderSide(color: primaryGreen, width: 1.5),
      ),
      hintStyle: const TextStyle(
        color: textHint,
        fontSize: fontSizeBody,
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        elevation: 0,
        minimumSize: const Size(double.infinity, buttonHeight),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusButton),
        ),
        textStyle: const TextStyle(
          fontSize: fontSizeButton,
          fontWeight: FontWeight.w500,
        ),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        minimumSize: const Size(double.infinity, buttonHeight),
        side: const BorderSide(width: 1.5),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusButton),
        ),
        textStyle: const TextStyle(
          fontSize: fontSizeButton,
          fontWeight: FontWeight.w500,
        ),
      ),
    ),
    textTheme: const TextTheme(
      titleLarge: TextStyle(
        fontSize: fontSizeTitle,
        fontWeight: FontWeight.w600,
        color: textPrimary,
      ),
      titleMedium: TextStyle(
        fontSize: fontSizeSection,
        fontWeight: FontWeight.w500,
        color: textPrimary,
      ),
      bodyMedium: TextStyle(
        fontSize: fontSizeBody,
        color: textSecondary,
      ),
      bodySmall: TextStyle(
        fontSize: fontSizeCaption,
        color: textSecondary,
      ),
    ),
  );
}
