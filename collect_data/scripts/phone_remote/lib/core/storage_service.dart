import 'package:shared_preferences/shared_preferences.dart';

class StorageService {
  static SharedPreferences? _prefs;
  static bool _initialized = false;

  // 初始化
  static Future<void> init() async {
    if (!_initialized) {
      _prefs = await SharedPreferences.getInstance();
      _initialized = true;
    }
  }

  // 确保已初始化
  static void _ensureInitialized() {
    if (!_initialized || _prefs == null) {
      throw Exception('StorageService 未初始化，请先调用 init()');
    }
  }

  // 字符串操作
  static Future<bool> setString(String key, String value) async {
    _ensureInitialized();
    return await _prefs!.setString(key, value);
  }

  static String? getString(String key, {String? defaultValue}) {
    _ensureInitialized();
    return _prefs!.getString(key) ?? defaultValue;
  }

  // 整数操作
  static Future<bool> setInt(String key, int value) async {
    _ensureInitialized();
    return await _prefs!.setInt(key, value);
  }

  static int? getInt(String key, {int? defaultValue}) {
    _ensureInitialized();
    return _prefs!.getInt(key) ?? defaultValue;
  }

  // 布尔值操作
  static Future<bool> setBool(String key, bool value) async {
    _ensureInitialized();
    return await _prefs!.setBool(key, value);
  }

  static bool? getBool(String key, {bool? defaultValue}) {
    _ensureInitialized();
    return _prefs!.getBool(key) ?? defaultValue;
  }

  // 字符串列表操作（用于校准数据）
  static Future<bool> setStringList(String key, List<String> value) async {
    _ensureInitialized();
    return await _prefs!.setStringList(key, value);
  }

  static List<String>? getStringList(String key) {
    _ensureInitialized();
    return _prefs!.getStringList(key);
  }

  // 删除数据
  static Future<bool> remove(String key) async {
    _ensureInitialized();
    return await _prefs!.remove(key);
  }

  // 清空所有数据
  static Future<bool> clear() async {
    _ensureInitialized();
    return await _prefs!.clear();
  }
}
