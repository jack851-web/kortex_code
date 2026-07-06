import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:audioplayers/audioplayers.dart';
import 'package:path_provider/path_provider.dart';

/// 音效服务 - 程序化生成WAV音效并播放
class SoundService {
  static final SoundService _instance = SoundService._internal();
  factory SoundService() => _instance;
  SoundService._internal();

  final AudioPlayer _player = AudioPlayer();
  bool _initialized = false;

  /// 初始化音效系统
  Future<void> init() async {
    if (_initialized) return;
    _initialized = true;
  }

  /// 播放Episode保存完成提示音（清脆"叮咚"双音）
  Future<void> playEpisodeSaved() async {
    await _playProgrammaticSound(
      frequencies: [523.25, 659.25], // C5, E5
      durations: [200, 400],
      volumes: [0.8, 0.6],
      gapMs: 100,
    );
  }

  /// 播放连接成功音效（上升琶音）
  Future<void> playConnected() async {
    await _playProgrammaticSound(
      frequencies: [440.00, 554.37, 659.25, 880.00], // A4, C#5, E5, A5
      durations: [75, 75, 75, 75],
      volumes: [0.7, 0.8, 0.9, 1.0],
      gapMs: 0,
    );
  }

  /// 播放急停警告音效（急促脉冲）
  Future<void> playEmergency() async {
    await _playProgrammaticSound(
      frequencies: [880, 880, 880, 880, 880, 880],
      durations: [80, 80, 80, 80, 80, 80],
      volumes: [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
      gapMs: 40,
    );
  }

  /// 播放任务完成提示音（三连升音）
  Future<void> playTaskComplete() async {
    await _playProgrammaticSound(
      frequencies: [523.25, 659.25, 783.99], // C5, E5, G5
      durations: [150, 150, 300],
      volumes: [0.8, 0.8, 0.7],
      gapMs: 50,
    );
  }

  /// 程序化生成声音并播放
  Future<void> _playProgrammaticSound({
    required List<double> frequencies,
    required List<int> durations,
    required List<double> volumes,
    required int gapMs,
  }) async {
    const sampleRate = 44100;
    final pcmData = <int>[];

    for (var i = 0; i < frequencies.length; i++) {
      final freq = frequencies[i];
      final durationMs = durations[i];
      final volume = volumes[i];
      final samplesCount = (sampleRate * durationMs / 1000).round();

      for (var s = 0; s < samplesCount; s++) {
        final t = s / sampleRate;
        var sample = sin(2 * pi * freq * t);

        // ADSR包络
        const attackTime = 0.01;
        const decayTime = 0.05;
        const releaseTime = 0.03;
        final totalDuration = durationMs / 1000.0;

        double envelope = 1.0;
        if (t < attackTime) {
          envelope = t / attackTime;
        } else if (t < attackTime + decayTime) {
          envelope = 1.0 - (0.3 * (t - attackTime) / decayTime);
        } else if (t > totalDuration - releaseTime) {
          envelope = 0.7 * (totalDuration - t) / releaseTime;
        } else {
          envelope = 0.7;
        }

        // 添加泛音
        sample += 0.15 * sin(2 * pi * freq * 2 * t);
        sample += 0.08 * sin(2 * pi * freq * 3 * t);

        final value = (sample * envelope * volume * 32767).clamp(-32767, 32767);
        pcmData.add(value.toInt());
      }

      // 间隙静音
      if (gapMs > 0 && i < frequencies.length - 1) {
        final gapSamples = (sampleRate * gapMs / 1000).round();
        for (var g = 0; g < gapSamples; g++) {
          pcmData.add(0);
        }
      }
    }

    // 生成WAV并写入临时文件播放
    final wavBytes = _pcmToWav(pcmData);
    final tempDir = await getTemporaryDirectory();
    final tempFile = File(
        '${tempDir.path}/sound_${DateTime.now().millisecondsSinceEpoch}.wav');
    await tempFile.writeAsBytes(wavBytes);

    try {
      await _player.play(DeviceFileSource(tempFile.path));
      // 播放完成后清理临时文件
      _player.onPlayerComplete.first.then((_) {
        try {
          tempFile.deleteSync();
        } catch (_) {}
      }).timeout(const Duration(seconds: 5), onTimeout: () {
        try {
          tempFile.deleteSync();
        } catch (_) {}
      });
    } catch (e) {
      // 播放失败时清理临时文件
      try {
        tempFile.deleteSync();
      } catch (_) {}
    }
  }

  /// PCM数据转WAV格式（使用ByteData确保二进制正确）
  Uint8List _pcmToWav(List<int> pcmData) {
    final dataSize = pcmData.length * 2;
    final byteData = ByteData(44 + dataSize);

    // RIFF头
    _writeString(byteData, 0, 'RIFF');
    byteData.setUint32(4, 36 + dataSize, Endian.little);
    _writeString(byteData, 8, 'WAVE');

    // fmt子块
    _writeString(byteData, 12, 'fmt ');
    byteData.setUint32(16, 16, Endian.little); // Chunk size
    byteData.setUint16(20, 1, Endian.little); // PCM format
    byteData.setUint16(22, 1, Endian.little); // Mono
    byteData.setUint32(24, 44100, Endian.little); // Sample rate
    byteData.setUint32(28, 44100 * 2, Endian.little); // Byte rate
    byteData.setUint16(32, 2, Endian.little); // Block align
    byteData.setUint16(34, 16, Endian.little); // Bits per sample

    // data子块
    _writeString(byteData, 36, 'data');
    byteData.setUint32(40, dataSize, Endian.little);

    // PCM样本数据
    for (var i = 0; i < pcmData.length; i++) {
      byteData.setInt16(44 + i * 2, pcmData[i], Endian.little);
    }

    return byteData.buffer.asUint8List();
  }

  void _writeString(ByteData byteData, int offset, String s) {
    for (var i = 0; i < s.length; i++) {
      byteData.setUint8(offset + i, s.codeUnitAt(i));
    }
  }

  /// 停止播放
  void stop() {
    _player.stop();
  }

  /// 释放资源
  void dispose() {
    _player.dispose();
    _initialized = false;
  }
}
