"""
数据后处理工具集

提供四个核心工具:
- lerobot_to_hdf5:   LeRobot v3.0 → HDF5  格式转换
- merge_datasets:    多个 LeRobot v3.0 数据集合并
- delete_episode:    单条/批量 episode 删除
- consolidate_videos: 多 chunk 视频/数据合并（已废弃，保留兼容）
"""
from pathlib import Path

__all__ = [
    "lerobot_to_hdf5",
    "merge_datasets",
    "delete_episode",
]
