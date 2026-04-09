"""
简单的 IK 测试脚本 - 使用 MuJoCo Viewer
测试任务：从 home 位置移动到 cup 抓取位置
"""

import sys
import time
import numpy as np

# 添加路径
sys.path.insert(0, 'd:/VLA/kortex_code/kortex_simu/ik')
sys.path.insert(0, 'd:/VLA/kortex_code/collect_data')

import mujoco
import mujoco.viewer

from mujoco_ik import MuJoCoIK


def test_ik():
    """测试 IK 功能"""
    
    # 任务配置 (来自 tasks_config.yaml)
    object_position = np.array([0.212, -0.4, 0.03])  # cup 位置
    pre_grasp_height = 0.203  # 预抓取高度 (0.03 + 0.173)
    pre_grasp_position = np.array([0.212, -0.4, pre_grasp_height])
    
    print("=" * 60)
    print("IK 测试 - 任务: grasp cup")
    print("=" * 60)
    print(f"物体位置: {object_position}")
    print(f"预抓取位置: {pre_grasp_position}")
    
    # 加载模型
    xml_path = 'd:/VLA/kortex_code/kortex_simu/simu/env/task_pick_place.xml'
    print(f"\n加载模型: {xml_path}")
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    print(f"模型加载成功!")
    print(f"  - 关节数: {model.njnt}")
    print(f"  - 执行器数: {model.nu}")
    print(f"  - 自由度: {model.nv}")
    
    # 创建 IK 求解器
    print("\n初始化 IK 求解器...")
    ik_solver = MuJoCoIK(model, data, 'tcp')
    print(f"IK 求解器初始化成功!")
    print(f"  - 关节: {ik_solver.joint_names}")
    print(f"  - 关节数: {ik_solver.n_joints}")
    
    # 获取初始状态
    initial_q = np.zeros(ik_solver.n_joints)
    for i, jnt_id in enumerate(ik_solver.joint_ids):
        jnt_qposadr = model.jnt_qposadr[jnt_id]
        initial_q[i] = data.qpos[jnt_qposadr]
    
    initial_pos = ik_solver.forward_kinematics(initial_q)
    print(f"\n初始状态:")
    print(f"  - 关节角度: {np.rad2deg(initial_q)}")
    print(f"  - 末端位置: {initial_pos}")
    
    # 测试 1: 正运动学
    print("\n" + "=" * 60)
    print("测试 1: 正运动学")
    print("=" * 60)
    test_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fk_pos = ik_solver.forward_kinematics(test_joints)
    print(f"输入关节: {np.rad2deg(test_joints)}")
    print(f"末端位置: {fk_pos}")
    
    # 测试 2: 逆运动学 - 到预抓取位置
    print("\n" + "=" * 60)
    print("测试 2: 逆运动学到预抓取位置")
    print("=" * 60)
    print(f"目标位置: {pre_grasp_position}")
    
    start_time = time.time()
    target_q = ik_solver.inverse_kinematics(
        pre_grasp_position,
        initial_guess=initial_q,
        max_iterations=100,
        tolerance=1e-4
    )
    ik_time = time.time() - start_time
    
    print(f"IK 求解完成! 耗时: {ik_time:.3f}s")
    print(f"目标关节: {np.rad2deg(target_q)}")
    
    # 验证
    final_pos = ik_solver.forward_kinematics(target_q)
    error = np.linalg.norm(pre_grasp_position - final_pos)
    print(f"实际位置: {final_pos}")
    print(f"误差: {error:.6f}m")
    
    # 检查关节限制
    is_valid, violations = ik_solver.check_joint_limits(target_q)
    if is_valid:
        print("✓ 关节限制检查通过")
    else:
        print(f"✗ 关节限制违反: {violations}")
    
    # 测试 3: 使用 MuJoCo Viewer 可视化
    print("\n" + "=" * 60)
    print("测试 3: MuJoCo Viewer 可视化")
    print("=" * 60)
    print("启动 Viewer，按空格键开始运动...")
    
    # 重置数据
    mujoco.mj_resetData(model, data)
    
    # 设置初始关节角度
    for i, jnt_id in enumerate(ik_solver.joint_ids):
        jnt_qposadr = model.jnt_qposadr[jnt_id]
        data.qpos[jnt_qposadr] = initial_q[i]
    
    # 获取执行器索引
    actuator_ids = []
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if act_name and 'J' in act_name and 'BOTTOM' not in act_name and 'TIP' not in act_name:
            actuator_ids.append(i)
    
    print(f"找到 {len(actuator_ids)} 个执行器")
    
    # 轨迹插值
    steps = 100
    trajectory = []
    for i in range(steps + 1):
        alpha = i / steps
        interp_q = initial_q + alpha * (target_q - initial_q)
        trajectory.append(interp_q)
    
    step_idx = 0
    running = True  # 自动开始运动
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer 已启动")
        print("  - 自动执行 IK 运动")
        print("  - 按 ESC 退出")
        
        while viewer.is_running() and step_idx < len(trajectory):
            if running:
                # 设置目标关节角度
                q_target = trajectory[step_idx]
                
                # 设置控制值
                for i, act_id in enumerate(actuator_ids[:6]):
                    if i < len(q_target):
                        ctrl_range = model.actuator_ctrlrange[act_id]
                        clipped_val = np.clip(q_target[i], ctrl_range[0], ctrl_range[1])
                        data.ctrl[act_id] = clipped_val
                
                # 步进仿真
                mujoco.mj_step(model, data)
                
                step_idx += 1
                
                # 每 10 步打印进度
                if step_idx % 10 == 0:
                    current_q = np.array([data.qpos[model.jnt_qposadr[jid]] 
                                         for jid in ik_solver.joint_ids])
                    current_pos = ik_solver.forward_kinematics(current_q)
                    print(f"Step {step_idx}/{steps}, pos: {current_pos}")
            
            viewer.sync()
            time.sleep(0.01)
        
        # 运动完成后保持显示
        print("\n运动完成! 保持显示，按 ESC 退出...")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_ik()
