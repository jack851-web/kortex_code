"""
逆运动学 (Inverse Kinematics) 模块
用于纯仿真模式下的机械臂控制

主要功能:
1. 正运动学: 从关节角度计算笛卡尔坐标
2. 逆运动学: 从笛卡尔坐标计算关节角度
3. 仿真控制器: 纯仿真模式下的机械臂控制

使用示例:
    from kortex_simu.ik import SimulationController
    
    # 创建控制器
    controller = SimulationController('path/to/model.xml')
    
    # 笛卡尔坐标转关节角度
    joint_angles, success = controller.cartesian_to_joint([0.4, 0.0, 0.3])
    
    # 关节角度转笛卡尔坐标
    position = controller.joint_to_cartesian(joint_angles)
    
    # 移动到目标位置
    controller.move_to_position([0.4, 0.0, 0.3])
"""

from .ik_solver import IKSolver, KinovaGen3LiteIK
from .mujoco_ik import MuJoCoIK
from .simulation_controller import SimulationController

__all__ = [
    'IKSolver',
    'KinovaGen3LiteIK', 
    'MuJoCoIK',
    'SimulationController'
]
