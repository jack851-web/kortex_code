"""
仿真机器人接口
支持两种模式:
1. 实机同步模式: 接收真实机器人关节数据并同步到仿真
2. 纯仿真模式: 使用 IK 根据笛卡尔坐标控制仿真机械臂
"""
# 从原始位置导入，保持向后兼容
from scripts.simu_interface import SimuInterface, MockSimuInterface

__all__ = ['SimuInterface', 'MockSimuInterface']
