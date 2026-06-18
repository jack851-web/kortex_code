import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'app.dart';
import 'core/storage_service.dart';

void main() async {
  // 确保Flutter绑定初始化（StorageService需要）
  WidgetsFlutterBinding.ensureInitialized();

  // 初始化本地存储服务
  await StorageService.init();

  runApp(const ProviderScope(child: PhoneRemoteApp()));
}
