"""
IK 模块使用示例
展示如何在纯仿真模式下使用 IK 控制机械臂
"""

import numpy as np
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ik import SimulationController, KinovaGen3LiteIK


def example_1_basic_fk_ik():
    """示例1: 基本的正逆运动学"""
    print("=" * 50)
    print("示例1: 基本正逆运动学")
    print("=" * 50)
    
    # 使用纯数学 IK 求解器
    ik = KinovaGen3LiteIK()
    
    # 测试正运动学
    joint_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    position = ik.forward_kinematics(joint_angles)
    print(f"关节角度: {joint_angles}")
    print(f"末端位置: {position}")
    
    # 测试逆运动学
    target_pos = np.array([0.4, 0.0, 0.3])
    print(f"\n目标位置: {target_pos}")
    
    ik_solution = ik.inverse_kinematics(target_pos, initial_guess=joint_angles)
    print(f"IK 解: {ik_solution}")
    
    # 验证
    actual_pos = ik.forward_kinematics(ik_solution)
    print(f"实际位置: {actual_pos}")
    print(f"误差: {np.linalg.norm(target_pos - actual_pos):.6f} m")
    

def example_2_simulation_controller():
    """示例2: 使用仿真控制器"""
    print("\n" + "=" * 50)
    print("示例2: 仿真控制器")
    print("=" * 50)
    
    # 模型路径
    xml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'simu', 'env', 'task_pick_place.xml'
    )
    
    if not os.path.exists(xml_path):
        print(f"模型文件不存在: {xml_path}")
        return
    
    # 创建控制器
    controller = SimulationController(xml_path)
    
    # 获取初始状态
    print("初始状态:")
    state = controller.get_state()
    print(f"  关节角度: {state['joint_angles']}")
    print(f"  末端位置: {state['end_effector_position']}")
    
    # 笛卡尔到关节转换
    target_pos = np.array([0.4, 0.0, 0.3])
    print(f"\n目标位置: {target_pos}")
    
    joint_angles, success = controller.cartesian_to_joint(target_pos)
    print(f"IK 解: {joint_angles}")
    print(f"求解成功: {success}")
    
    # 验证
    actual_pos = controller.joint_to_cartesian(joint_angles)
    print(f"实际位置: {actual_pos}")
    print(f"误差: {np.linalg.norm(target_pos - actual_pos):.6f} m")
    

def example_3_trajectory_planning():
    """示例3: 轨迹规划"""
    print("\n" + "=" * 50)
    print("示例3: 轨迹规划")
    print("=" * 50)
    
    xml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'simu', 'env', 'task_pick_place.xml'
    )
    
    if not os.path.exists(xml_path):
        print(f"模型文件不存在: {xml_path}")
        return
    
    controller = SimulationController(xml_path)
    
    # 定义路径点
    waypoints = [
        np.array([0.4, 0.0, 0.3]),
        np.array([0.4, 0.1, 0.3]),
        np.array([0.4, 0.1, 0.2]),
        np.array([0.3, 0.0, 0.2]),
    ]
    
    print(f"路径点数量: {len(waypoints)}")
    
    # 规划轨迹
    trajectory = controller.plan_trajectory(
        waypoints,
        duration_per_segment=1.0,
        steps_per_segment=20
    )
    
    print(f"轨迹点数量: {len(trajectory)}")
    
    # 显示轨迹的起始和结束
    print(f"起始关节角度: {trajectory[0]}")
    print(f"结束关节角度: {trajectory[-1]}")
    

def example_4_integration_with_simulation():
    """示例4: 与仿真环境集成"""
    print("\n" + "=" * 50)
    print("示例4: 与仿真环境集成")
    print("=" * 50)
    
    xml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'simu', 'env', 'task_pick_place.xml'
    )
    
    if not os.path.exists(xml_path):
        print(f"模型文件不存在: {xml_path}")
        return
    
    controller = SimulationController(xml_path)
    
    # 设置初始位置
    initial_joints = np.array([0.0, 0.5, -0.5, 0.0, 0.0, 0.0])
    controller.set_joint_positions_direct(initial_joints)
    
    print(f"初始关节角度: {initial_joints}")
    print(f"当前位置: {controller.get_current_position()}")
    
    # 定义抓取任务
    print("\n执行抓取任务:")
    
    # 1. 移动到物体上方
    pre_grasp_pos = np.array([0.3, -0.1, 0.25])
    print(f"1. 移动到预抓取位置: {pre_grasp_pos}")
    success = controller.move_to_position(pre_grasp_pos, duration=1.0, steps=50)
    print(f"   成功: {success}")
    
    # 2. 下降抓取
    grasp_pos = np.array([0.3, -0.1, 0.15])
    print(f"2. 下降抓取: {grasp_pos}")
    success = controller.move_to_position(grasp_pos, duration=0.5, steps=25)
    print(f"   成功: {success}")
    
    # 3. 抬起
    lift_pos = np.array([0.3, -0.1, 0.25])
    print(f"3. 抬起物体: {lift_pos}")
    success = controller.move_to_position(lift_pos, duration=0.5, steps=25)
    print(f"   成功: {success}")
    
    # 4. 移动到放置位置
    place_pos = np.array([-0.1, -0.4, 0.25])
    print(f"4. 移动到放置位置: {place_pos}")
    success = controller.move_to_position(place_pos, duration=1.0, steps=50)
    print(f"   成功: {success}")
    
    print(f"\n最终关节角度: {controller.get_current_joint_angles()}")


if __name__ == "__main__":
    # 运行示例
    example_1_basic_fk_ik()
    example_2_simulation_controller()
    example_3_trajectory_planning()
    example_4_integration_with_simulation()
    
    print("\n" + "=" * 50)
    print("所有示例完成!")
    print("=" * 50)
