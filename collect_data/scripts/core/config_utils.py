"""配置路径解析工具

提供 resolve_path 和 resolve_config_paths 两个公共函数，
供 main.py、main_qt.py、gui/object_profile_tuner_window.py 等模块统一使用。
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_path(path_str: str, base_dir: Path = None, allowed_roots: list = None) -> str:
    """解析路径，支持相对路径和绝对路径

    Args:
        path_str: 路径字符串
        base_dir: 基准目录，默认为调用方自行指定
        allowed_roots: 允许的路径根目录列表，用于安全校验。为 None 时不校验。

    Returns:
        解析后的绝对路径字符串
    """
    if not path_str:
        return path_str

    path = Path(path_str)

    # 已经是绝对路径
    if path.is_absolute():
        resolved = path
    else:
        # 相对路径：基于 base_dir 解析
        base = base_dir or Path.cwd()
        resolved = (base / path_str).resolve()

    # 路径安全校验：检测路径遍历攻击
    resolved_str = str(resolved)
    if '..' in path_str.split('/') or '..' in path_str.split('\\'):
        logger.warning(f"配置路径包含路径遍历字符: {path_str} -> {resolved_str}")

    # 如果指定了允许的根目录，校验路径是否在其下
    if allowed_roots:
        allowed_roots_resolved = [Path(r).resolve() for r in allowed_roots]
        if not any(resolved_str.startswith(str(r)) for r in allowed_roots_resolved):
            logger.warning(f"配置路径超出允许范围: {resolved_str} 不在 {allowed_roots} 下")

    return resolved_str


def resolve_config_paths(config: dict, base_dir: Path = None) -> dict:
    """递归解析配置中的路径字段

    自动识别以 _path, _root 结尾的字段以及 xml_path, model_xml_path 等常见路径字段。
    """
    path_keys = {
        'xml_path', 'model_xml_path', 'scene_base_xml_path',
        'real_data_root', 'simu_data_root', 'mock_simu_data_root',
        'data_root', 'output_path', 'log_path',
    }

    def _resolve_value(key: str, value):
        if isinstance(value, str):
            is_path = (
                key in path_keys
                or key.endswith('_path')
                or key.endswith('_root')
                or key.endswith('_xml')
                or key.endswith('_dir')
                or '.xml' in value.lower()
                or '.yaml' in value.lower()
                or '.json' in value.lower()
            )
            if is_path:
                return resolve_path(value, base_dir)
            return value
        elif isinstance(value, dict):
            return {k: _resolve_value(k, v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_resolve_value(key, item) for item in value]
        return value

    if config is None:
        return config

    return {k: _resolve_value(k, v) for k, v in config.items()}
