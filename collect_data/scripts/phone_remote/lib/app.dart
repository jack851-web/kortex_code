import 'package:flutter/material.dart';
import 'core/theme.dart';
import 'screens/settings_screen.dart';
import 'screens/main_control_screen.dart';
import 'screens/calibration_screen.dart';

class PhoneRemoteApp extends StatelessWidget {
  const PhoneRemoteApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Kortex Remote',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      initialRoute: '/settings',
      routes: {
        '/settings': (context) => const SettingsScreen(),
        '/main': (context) => const MainControlScreen(),
        '/calibration': (context) => const CalibrationScreen(),
      },
    );
  }
}
