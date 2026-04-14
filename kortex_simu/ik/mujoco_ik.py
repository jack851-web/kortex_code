"""
MuJoCo 集成的 IK 求解器
利用 MuJoCo 的 mj_jac 和 mj_ik 功能
"""

import numpy as np
import mujoco
from typing import Optional, Tuple, Dict, Any


class MuJoCoIK:
    """
    基于 MuJoCo 的 IK 求解器
    使用 MuJoCo 的内置功能进行正逆运动学计算
    """
    
    def __init__(self, model, data, end_effector_name: str = "tcp"):
        """
        初始化 MuJoCo IK 求解器
        
        Args:
            model: MuJoCo 模型 (mujoco.MjModel)
            data: MuJoCo 数据 (mujoco.MjData) - 用于复制初始状态
            end_effector_name: 末端执行器 site 名称
        """
        self.model = model
        # 创建独立的 data 副本，避免与主仿真冲突
        self.data = mujoco.MjData(model)
        # 复制初始状态
        self.data.qpos[:] = data.qpos[:]
        self.end_effector_name = end_effector_name
        
        # 获取末端执行器 ID
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, end_effector_name)
        if self.ee_id < 0:
            raise ValueError(f"Site '{end_effector_name}' not found in model")
        
        # 获取关节信息
        self.joint_names = []
        self.joint_ids = []
        self.joint_ranges = []
        
        for i in range(model.njnt):
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name and jnt_name.startswith('J') and jnt_name[1:].isdigit():
                self.joint_names.append(jnt_name)
                self.joint_ids.append(i)
                jnt_range = model.jnt_range[i]
                self.joint_ranges.append((jnt_range[0], jnt_range[1]))
        
        self.n_joints = len(self.joint_ids)
        
        if self.n_joints == 0:
            raise ValueError("No valid joints found in model. Expected joints with names starting with 'J' followed by a digit.")

        self.joint_qposadrs = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.joint_dofadrs = [self.model.jnt_dofadr[jid] for jid in self.joint_ids]
        self.last_solve_info: Dict[str, Any] = {
            'success': False,
            'iterations': 0,
            'position_error': float('inf'),
            'orientation_error': float('inf'),
            'timed_out': False,
        }
        
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        正运动学：计算末端执行器位置
        
        Args:
            joint_angles: 关节角度 (弧度)
            
        Returns:
            末端执行器位置 [x, y, z]
        """
        # 设置关节角度
        for i, qpos_adr in enumerate(self.joint_qposadrs):
            if i < len(joint_angles):
                self.data.qpos[qpos_adr] = joint_angles[i]
        
        # 使用 mj_forward 确保完整的数据更新
        mujoco.mj_forward(self.model, self.data)
        
        # 获取末端执行器位置
        return self.data.site_xpos[self.ee_id].copy()
    
    def forward_kinematics_with_orientation(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        正运动学：计算位置和姿态
        
        Args:
            joint_angles: 关节角度 (弧度)
            
        Returns:
            (position, rotation_matrix) - 位置和旋转矩阵
        """
        # 设置关节角度
        for i, qpos_adr in enumerate(self.joint_qposadrs):
            if i < len(joint_angles):
                self.data.qpos[qpos_adr] = joint_angles[i]
        
        # 使用 mj_forward 确保完整的数据更新
        mujoco.mj_forward(self.model, self.data)
        
        # 获取位置和姿态
        position = self.data.site_xpos[self.ee_id].copy()
        rotation = self.data.site_xmat[self.ee_id].reshape(3, 3).copy()
        
        return position, rotation
    
    def jacobian(self, joint_angles: Optional[np.ndarray] = None, skip_forward: bool = False) -> np.ndarray:
        """
        计算雅可比矩阵
        
        Args:
            joint_angles: 关节角度 (可选，默认使用当前状态)
            skip_forward: 是否跳过 mj_forward（如果刚调用过 forward_kinematics 则为 True）
            
        Returns:
            雅可比矩阵 (6 x n_joints)
        """
        # 如果提供了关节角度，更新状态
        if joint_angles is not None:
            for i, qpos_adr in enumerate(self.joint_qposadrs):
                if i < len(joint_angles):
                    self.data.qpos[qpos_adr] = joint_angles[i]
        
        # 只有在需要时才调用 mj_forward
        if not skip_forward:
            mujoco.mj_forward(self.model, self.data)
        
        # 计算雅可比矩阵
        jacp = np.zeros((3, self.model.nv))  # 位置雅可比
        jacr = np.zeros((3, self.model.nv))  # 旋转雅可比
        
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_id)
        
        # 提取关节对应的列
        jacobian = np.zeros((6, self.n_joints))
        for i, dof_adr in enumerate(self.joint_dofadrs):
            jacobian[:3, i] = jacp[:, dof_adr]
            jacobian[3:, i] = jacr[:, dof_adr]
        
        return jacobian
    
    def inverse_kinematics(self,
                          target_position: np.ndarray,
                          target_orientation: Optional[np.ndarray] = None,
                          initial_guess: Optional[np.ndarray] = None,
                          max_iterations: int = 50,  # 增加默认迭代次数
                          tolerance: float = 5e-4,   # 收紧容差到 0.5mm
                          damping: float = 0.05) -> np.ndarray:
        """
        逆运动学：阻尼最小二乘法 (DLS)
        返回关节角；求解状态可通过 get_last_solve_info() 获取。
        """
        target_position = np.asarray(target_position, dtype=float).reshape(3)
        target_orientation = None if target_orientation is None else np.asarray(target_orientation, dtype=float).reshape(3, 3)

        if initial_guess is not None:
            q = np.asarray(initial_guess, dtype=float).copy()
        else:
            q = np.zeros(self.n_joints)

        if q.shape[0] != self.n_joints:
            raise ValueError(f"Expected initial_guess with shape ({self.n_joints},), got {q.shape}")

        step_scale = 0.6  # 增大步长缩放，加快收敛
        max_step_norm = 0.2  # 减小最大步长，提高精度
        success = False
        pos_err_norm = float('inf')
        rot_err_norm = float('inf')
        iteration_done = 0

        for iteration in range(max_iterations):
            iteration_done = iteration + 1
            current_pos, current_rot = self.forward_kinematics_with_orientation(q)
            pos_error = target_position - current_pos
            pos_err_norm = float(np.linalg.norm(pos_error))

            if target_orientation is not None:
                rot_error_mat = target_orientation @ current_rot.T
                rot_error = self._rotation_matrix_to_axis_angle(rot_error_mat)
                rot_err_norm = float(np.linalg.norm(rot_error))
                error = np.concatenate([pos_error, rot_error])
            else:
                rot_error = np.zeros(3)
                rot_err_norm = 0.0
                error = pos_error

            if pos_err_norm < tolerance and (target_orientation is None or rot_err_norm < tolerance):
                success = True
                break

            # 关键优化：forward_kinematics_with_orientation 已调用 mj_forward
            # 这里跳过重复调用
            J = self.jacobian(q, skip_forward=True)
            if target_orientation is None:
                J = J[:3, :]

            JTJ = J.T @ J
            A = JTJ + damping**2 * np.eye(JTJ.shape[0])
            b = J.T @ error

            try:
                delta_q = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                delta_q = np.linalg.pinv(J) @ error

            step_norm = np.linalg.norm(delta_q)
            if step_norm > max_step_norm:
                delta_q = delta_q / (step_norm + 1e-12) * max_step_norm

            q += step_scale * delta_q

            for i in range(self.n_joints):
                q[i] = np.clip(q[i], self.joint_ranges[i][0], self.joint_ranges[i][1])

        self.last_solve_info = {
            'success': success,
            'iterations': iteration_done,
            'position_error': pos_err_norm,
            'orientation_error': rot_err_norm,
            'timed_out': False,
        }
        return q
    
    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """将旋转矩阵转换为轴角表示"""
        # 使用 trace 方法计算轴角
        trace = np.trace(R)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if np.abs(angle) < 1e-6:
            return np.zeros(3)
        
        # 计算旋转轴
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(angle))
        
        return axis * angle
    
    def get_last_solve_info(self) -> Dict[str, Any]:
        """获取最近一次 IK 求解状态"""
        return self.last_solve_info.copy()

    def get_joint_limits(self) -> list:
        """获取关节限制"""
        return self.joint_ranges.copy()
    
    def check_joint_limits(self, joint_angles: np.ndarray) -> Tuple[bool, list]:
        """
        检查关节角度是否在限制范围内
        
        Returns:
            (is_valid, violations)
        """
        violations = []
        for i, angle in enumerate(joint_angles):
            if angle < self.joint_ranges[i][0]:
                violations.append(f'{self.joint_names[i]}: {angle:.3f} < {self.joint_ranges[i][0]:.3f}')
            elif angle > self.joint_ranges[i][1]:
                violations.append(f'{self.joint_names[i]}: {angle:.3f} > {self.joint_ranges[i][1]:.3f}')
        
        return len(violations) == 0, violations


def test_ik():
    """测试 IK 功能"""
    import os
    
    # 加载模型
    xml_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'simu', 'env', 'task_pick_place.xml'
    )
    
    if not os.path.exists(xml_path):
        print(f"Model not found: {xml_path}")
        return
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 创建 IK 求解器
    ik = MuJoCoIK(model, data)
    
    print("Testing Forward Kinematics:")
    q_test = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pos = ik.forward_kinematics(q_test)
    print(f"Joint angles: {q_test}")
    print(f"End effector position: {pos}")
    
    print("\nTesting Inverse Kinematics:")
    target_pos = pos + np.array([0.05, 0.05, 0.05])
    q_ik = ik.inverse_kinematics(target_pos, initial_guess=q_test)
    print(f"Target position: {target_pos}")
    print(f"IK solution: {q_ik}")
    
    pos_ik = ik.forward_kinematics(q_ik)
    print(f"Actual position: {pos_ik}")
    print(f"Error: {np.linalg.norm(target_pos - pos_ik)}")


if __name__ == "__main__":
    test_ik()
