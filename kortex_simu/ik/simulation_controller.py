"""
纯仿真模式控制器
使用 IK 控制仿真机械臂
"""

import numpy as np
import mujoco
from typing import Optional, Tuple, Dict, Any
from .mujoco_ik import MuJoCoIK


class SimulationController:
    """
    纯仿真模式控制器
    使用 IK 将笛卡尔坐标转换为关节角度
    """
    
    def __init__(self, model_path: str, end_effector_name: str = "tcp"):
        """
        初始化仿真控制器
        
        Args:
            model_path: MuJoCo XML 模型路径
            end_effector_name: 末端执行器 site 名称
        """
        # 加载 MuJoCo 模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 创建 IK 求解器
        self.ik_solver = MuJoCoIK(self.model, self.data, end_effector_name)
        
        # 获取关节信息
        self.joint_names = self.ik_solver.joint_names
        self.n_joints = self.ik_solver.n_joints
        
        # 获取执行器信息
        self.actuator_ids = []
        for i in range(self.model.nu):
            act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if act_name and 'J' in act_name:
                self.actuator_ids.append(i)
        
        # 初始化状态
        self.current_joint_angles = np.zeros(self.n_joints)
        self.target_joint_angles = np.zeros(self.n_joints)
        
        # 控制参数
        self.position_gain = 5000  # 位置控制增益
        self.velocity_gain = 100   # 速度控制增益
        
    def reset(self):
        """重置仿真状态"""
        mujoco.mj_resetData(self.model, self.data)
        self.current_joint_angles = np.zeros(self.n_joints)
        self.target_joint_angles = np.zeros(self.n_joints)
        
    def get_current_position(self) -> np.ndarray:
        """获取当前末端执行器位置"""
        return self.ik_solver.forward_kinematics(self.current_joint_angles)
    
    def get_current_joint_angles(self) -> np.ndarray:
        """获取当前关节角度"""
        # 从 MuJoCo 数据中获取
        q = np.zeros(self.n_joints)
        for i, jnt_id in enumerate(self.ik_solver.joint_ids):
            jnt_qposadr = self.model.jnt_qposadr[jnt_id]
            q[i] = self.data.qpos[jnt_qposadr]
        self.current_joint_angles = q
        return q
    
    def move_to_position(self, 
                        target_position: np.ndarray,
                        target_orientation: Optional[np.ndarray] = None,
                        duration: float = 2.0,
                        steps: int = 100) -> bool:
        """
        移动末端执行器到目标位置
        
        Args:
            target_position: 目标位置 [x, y, z]
            target_orientation: 目标姿态 (3x3旋转矩阵), 可选
            duration: 运动持续时间 (秒)
            steps: 插值步数
            
        Returns:
            是否成功
        """
        # 获取当前关节状态
        current_q = self.get_current_joint_angles()
        
        # 使用 IK 计算目标关节角度
        target_q = self.ik_solver.inverse_kinematics(
            target_position,
            target_orientation,
            initial_guess=current_q
        )
        
        # 检查解的有效性
        is_valid, violations = self.ik_solver.check_joint_limits(target_q)
        solve_info = self.ik_solver.get_last_solve_info() if hasattr(self.ik_solver, 'get_last_solve_info') else {}
        ik_success = solve_info.get('success', True)
        if not is_valid or not ik_success:
            if not is_valid:
                print(f"IK solution violates joint limits: {violations}")
            if not ik_success:
                print(f"IK failed to converge: {solve_info}")
            return False
        
        # 轨迹插值
        dt = duration / steps
        for i in range(steps + 1):
            alpha = i / steps
            # 关节空间插值
            interp_q = current_q + alpha * (target_q - current_q)
            
            # 设置目标位置
            self._set_joint_positions(interp_q)
            
            # 仿真步进
            mujoco.mj_step(self.model, self.data)
        
        self.target_joint_angles = target_q
        return True
    
    def _set_joint_positions(self, joint_angles: np.ndarray):
        """设置关节目标位置"""
        # 设置控制信号
        for i, act_id in enumerate(self.actuator_ids):
            if i < len(joint_angles):
                self.data.ctrl[act_id] = joint_angles[i]
    
    def set_joint_positions_direct(self, joint_angles: np.ndarray):
        """直接设置关节位置（用于初始化）"""
        for i, jnt_id in enumerate(self.ik_solver.joint_ids):
            if i < len(joint_angles):
                jnt_qposadr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[jnt_qposadr] = joint_angles[i]
        
        # 前向运动学更新
        mujoco.mj_forward(self.model, self.data)
        self.current_joint_angles = joint_angles.copy()
    
    def step(self, joint_commands: Optional[np.ndarray] = None):
        """
        执行一个仿真步
        
        Args:
            joint_commands: 关节控制命令 (可选)
        """
        if joint_commands is not None:
            self._set_joint_positions(joint_commands)
        
        mujoco.mj_step(self.model, self.data)
        self.get_current_joint_angles()  # 更新当前角度
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'joint_angles': self.get_current_joint_angles(),
            'end_effector_position': self.get_current_position(),
            'time': self.data.time
        }
    
    def cartesian_to_joint(self, 
                          position: np.ndarray,
                          orientation: Optional[np.ndarray] = None,
                          initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """
        笛卡尔坐标转关节角度
        
        Args:
            position: 目标位置 [x, y, z]
            orientation: 目标姿态 (可选)
            initial_guess: 初始猜测
            
        Returns:
            (joint_angles, success)
        """
        if initial_guess is None:
            initial_guess = self.get_current_joint_angles()
        
        joint_angles = self.ik_solver.inverse_kinematics(
            position,
            orientation,
            initial_guess=initial_guess
        )
        
        is_valid, _ = self.ik_solver.check_joint_limits(joint_angles)
        solve_info = self.ik_solver.get_last_solve_info() if hasattr(self.ik_solver, 'get_last_solve_info') else {}
        return joint_angles, is_valid and solve_info.get('success', True)
    
    def joint_to_cartesian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        关节角度转笛卡尔坐标
        
        Args:
            joint_angles: 关节角度
            
        Returns:
            末端执行器位置 [x, y, z]
        """
        return self.ik_solver.forward_kinematics(joint_angles)
    
    def plan_trajectory(self, 
                       waypoints: list,
                       duration_per_segment: float = 2.0,
                       steps_per_segment: int = 100) -> list:
        """
        规划轨迹
        
        Args:
            waypoints: 路径点列表，每个点是 [x, y, z] 或 (position, orientation)
            duration_per_segment: 每段持续时间
            steps_per_segment: 每段步数
            
        Returns:
            轨迹点列表 (关节角度)
        """
        trajectory = []
        current_q = self.get_current_joint_angles()
        
        for waypoint in waypoints:
            if isinstance(waypoint, tuple):
                target_pos, target_rot = waypoint
            else:
                target_pos = waypoint
                target_rot = None
            
            # IK 求解
            target_q = self.ik_solver.inverse_kinematics(
                target_pos,
                target_rot,
                initial_guess=current_q
            )
            
            # 插值
            for i in range(steps_per_segment + 1):
                alpha = i / steps_per_segment
                interp_q = current_q + alpha * (target_q - current_q)
                trajectory.append(interp_q.copy())
            
            current_q = target_q
        
        return trajectory
    
    def execute_trajectory(self, trajectory: list):
        """执行规划好的轨迹"""
        for joint_angles in trajectory:
            self._set_joint_positions(joint_angles)
            mujoco.mj_step(self.model, self.data)
        
        self.get_current_joint_angles()


def test_simulation_controller():
    """测试仿真控制器"""
    import os
    
    xml_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'simu', 'env', 'task_pick_place.xml'
    )
    
    if not os.path.exists(xml_path):
        print(f"Model not found: {xml_path}")
        return
    
    controller = SimulationController(xml_path)
    
    print("Initial state:")
    state = controller.get_state()
    print(f"Joint angles: {state['joint_angles']}")
    print(f"End effector: {state['end_effector_position']}")
    
    # 测试笛卡尔到关节转换
    target_pos = np.array([0.4, 0.0, 0.3])
    print(f"\nTarget position: {target_pos}")
    
    joint_angles, success = controller.cartesian_to_joint(target_pos)
    print(f"IK solution: {joint_angles}")
    print(f"Success: {success}")
    
    actual_pos = controller.joint_to_cartesian(joint_angles)
    print(f"Actual position: {actual_pos}")
    print(f"Error: {np.linalg.norm(target_pos - actual_pos)}")
    
    # 测试轨迹规划
    waypoints = [
        np.array([0.4, 0.0, 0.3]),
        np.array([0.4, 0.1, 0.3]),
        np.array([0.4, 0.1, 0.2]),
    ]
    
    print("\nPlanning trajectory...")
    trajectory = controller.plan_trajectory(waypoints, duration_per_segment=1.0, steps_per_segment=50)
    print(f"Trajectory length: {len(trajectory)}")


if __name__ == "__main__":
    test_simulation_controller()
