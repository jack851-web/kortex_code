"""
Kinova Gen3 Lite 逆运动学求解器
基于DH参数的正逆运动学实现
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any


class IKSolver:
    """IK求解器基类"""
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """正运动学：从关节角度计算笛卡尔坐标"""
        raise NotImplementedError
    
    def inverse_kinematics(self, position: np.ndarray, orientation: Optional[np.ndarray] = None) -> np.ndarray:
        """逆运动学：从笛卡尔坐标计算关节角度"""
        raise NotImplementedError
    
    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """计算雅可比矩阵"""
        raise NotImplementedError


class KinovaGen3LiteIK(IKSolver):
    """
    Kinova Gen3 Lite 逆运动学求解器
    
    机械臂结构:
    - 6个关节 (J0-J5) + 夹爪
    - 采用标准的6-DOF机械臂逆运动学
    
    DH参数 (单位: 米, 弧度):
    基于 gen3_lite_gen3_lite_2f.xml 中的关节定义
    """
    
    def __init__(self):
        self.dh_params = {
            'd1': 0.12825,      # 基座高度
            'd2': 0.115,        # Shoulder到Elbow距离
            'd3': 0.28,         # Elbow到Wrist距离
            'd4': 0.14,         # Wrist offset
            'd5': 0.02,         # Wrist高度
            'd6': 0.105,        # End effector offset
            'a2': 0.05955,      # Shoulder offset
        }
        
        self.joint_limits = {
            'J0': (-2.76, 2.76),
            'J1': (-2.76, 2.76),
            'J2': (-2.76, 2.76),
            'J3': (-2.67, 2.67),
            'J4': (-2.67, 2.67),
            'J5': (-2.67, 2.67),
        }
        self.last_solve_info: Dict[str, Any] = {
            'success': False,
            'iterations': 0,
            'position_error': float('inf'),
            'orientation_error': float('inf'),
        }
    
    def _transform_matrix(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """
        计算DH变换矩阵
        
        Args:
            theta: 绕z轴旋转角度
            d: 沿z轴平移距离
            a: 沿x轴平移距离
            alpha: 绕x轴旋转角度
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        正运动学：从关节角度计算末端执行器位置
        
        Args:
            joint_angles: 6个关节角度 (弧度), shape=(6,)
            
        Returns:
            末端执行器位置 [x, y, z] (米)
        """
        if len(joint_angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")
        
        d1 = self.dh_params['d1']
        d2 = self.dh_params['d2']
        d3 = self.dh_params['d3']
        d4 = self.dh_params['d4']
        d5 = self.dh_params['d5']
        d6 = self.dh_params['d6']
        a2 = self.dh_params['a2']
        
        q = joint_angles
        
        T01 = self._transform_matrix(q[0], d1, 0, np.pi/2)
        T12 = self._transform_matrix(q[1], 0, -a2, 0)
        T23 = self._transform_matrix(q[2] - np.pi/2, 0, -d2, -np.pi/2)
        T34 = self._transform_matrix(q[3], d3, 0, np.pi/2)
        T45 = self._transform_matrix(q[4], d4, 0, -np.pi/2)
        T56 = self._transform_matrix(q[5], d5 + d6, 0, 0)
        
        T02 = T01 @ T12
        T03 = T02 @ T23
        T04 = T03 @ T34
        T05 = T04 @ T45
        T06 = T05 @ T56
        
        position = T06[:3, 3]
        return position
    
    def forward_kinematics_with_orientation(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        正运动学：计算位置和姿态
        
        Args:
            joint_angles: 6个关节角度 (弧度)
            
        Returns:
            (position, rotation_matrix) - 位置和旋转矩阵
        """
        if len(joint_angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")
        
        d1 = self.dh_params['d1']
        d2 = self.dh_params['d2']
        d3 = self.dh_params['d3']
        d4 = self.dh_params['d4']
        d5 = self.dh_params['d5']
        d6 = self.dh_params['d6']
        a2 = self.dh_params['a2']
        
        q = joint_angles
        
        T01 = self._transform_matrix(q[0], d1, 0, np.pi/2)
        T12 = self._transform_matrix(q[1], 0, -a2, 0)
        T23 = self._transform_matrix(q[2] - np.pi/2, 0, -d2, -np.pi/2)
        T34 = self._transform_matrix(q[3], d3, 0, np.pi/2)
        T45 = self._transform_matrix(q[4], d4, 0, -np.pi/2)
        T56 = self._transform_matrix(q[5], d5 + d6, 0, 0)
        
        T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56
        
        position = T06[:3, 3]
        rotation = T06[:3, :3]
        
        return position, rotation
    
    def inverse_kinematics(self, position: np.ndarray, 
                          orientation: Optional[np.ndarray] = None,
                          initial_guess: Optional[np.ndarray] = None,
                          max_iterations: int = 100,
                          tolerance: float = 1e-4) -> np.ndarray:
        """
        数值逆运动学：阻尼最小二乘法 (DLS)
        支持位置约束，若提供姿态则同时优化姿态。
        """
        target_position = np.asarray(position, dtype=float).reshape(3)
        target_orientation = None if orientation is None else np.asarray(orientation, dtype=float).reshape(3, 3)

        n_joints = 6
        q = np.zeros(n_joints) if initial_guess is None else np.asarray(initial_guess, dtype=float).copy()
        if q.shape[0] != n_joints:
            raise ValueError(f"Expected initial_guess with shape (6,), got {q.shape}")

        damping = 0.08
        step_scale = 0.6
        max_step_norm = 0.25
        eps = 1e-6

        success = False
        pos_err_norm = float('inf')
        rot_err_norm = float('inf')
        iteration_done = 0

        for iteration in range(max_iterations):
            iteration_done = iteration + 1
            current_pos, current_rot = self.forward_kinematics_with_orientation(q)
            pos_error = target_position - current_pos
            pos_err_norm = float(np.linalg.norm(pos_error))

            if target_orientation is None:
                error = pos_error
                rot_err_norm = 0.0
            else:
                rot_error = self._rotation_matrix_to_axis_angle(target_orientation @ current_rot.T)
                rot_err_norm = float(np.linalg.norm(rot_error))
                error = np.concatenate([pos_error, rot_error])

            if pos_err_norm < tolerance and (target_orientation is None or rot_err_norm < tolerance):
                success = True
                break

            J = np.zeros((error.shape[0], n_joints))
            for j in range(n_joints):
                q_plus = q.copy()
                q_plus[j] += eps
                p_plus, r_plus = self.forward_kinematics_with_orientation(q_plus)

                dp = (p_plus - current_pos) / eps
                if target_orientation is None:
                    J[:, j] = dp
                else:
                    dR = r_plus @ current_rot.T
                    domega = self._rotation_matrix_to_axis_angle(dR) / eps
                    J[:, j] = np.concatenate([dp, domega])

            JTJ = J.T @ J
            A = JTJ + (damping ** 2) * np.eye(n_joints)
            b = J.T @ error

            try:
                delta_q = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                delta_q = np.linalg.pinv(J) @ error

            delta_norm = np.linalg.norm(delta_q)
            if delta_norm > max_step_norm:
                delta_q = delta_q / (delta_norm + 1e-12) * max_step_norm

            q += step_scale * delta_q

            for i in range(n_joints):
                q[i] = np.clip(q[i], self.joint_limits[f'J{i}'][0], self.joint_limits[f'J{i}'][1])

        self.last_solve_info = {
            'success': success,
            'iterations': iteration_done,
            'position_error': pos_err_norm,
            'orientation_error': rot_err_norm,
        }
        return q
    
    def inverse_kinematics_analytical(self, position: np.ndarray, 
                                      preferred_angles: Optional[np.ndarray] = None) -> np.ndarray:
        """
        解析逆运动学（简化版本，不考虑姿态）
        
        Args:
            position: 目标位置 [x, y, z] (米)
            preferred_angles: 优先角度用于解决多解问题
            
        Returns:
            6个关节角度 (弧度), shape=(6,)
        """
        x, y, z = position
        d1 = self.dh_params['d1']
        d2 = self.dh_params['d2']
        d3 = self.dh_params['d3']
        d4 = self.dh_params['d4']
        d5 = self.dh_params['d5']
        d6 = self.dh_params['d6']
        a2 = self.dh_params['a2']
        
        d_base_to_shoulder = d1
        shoulder_to_wrist = d2 + d3 + d4 + d5 + d6
        
        r_xy = np.sqrt(x**2 + y**2)
        r_xz = z - d_base_to_shoulder
        
        r = np.sqrt(r_xy**2 + r_xz**2)
        
        cos_q1_1 = x / r_xy if r_xy > 1e-6 else 1
        sin_q1_1 = y / r_xy if r_xy > 1e-6 else 0
        q1 = np.arctan2(sin_q1_1, cos_q1_1)
        
        L1 = a2
        L2 = d2
        L_total = np.sqrt(r**2 + a2**2 - 2 * r * a2 * np.cos(np.arctan2(r_xz, r_xy)))
        
        cos_q2 = (L1**2 + L2**2 - L_total**2) / (2 * L1 * L2) if L_total < L1 + L2 else 1
        cos_q2 = np.clip(cos_q2, -1, 1)
        q2 = np.pi - np.arccos(cos_q2)
        
        phi1 = np.arctan2(r_xz, r_xy)
        phi2 = np.arctan2(L_total * np.sin(np.arccos(cos_q2)), L1 + L2 * cos_q2)
        q3 = phi1 - phi2
        
        if preferred_angles is not None:
            q = np.array([q1, q2, q3, preferred_angles[3], preferred_angles[4], preferred_angles[5]])
        else:
            shoulder_angle = np.arctan2(r_xz, r_xy)
            q = np.array([q1, q2, q3, 0.0, 0.0, 0.0])
        
        for i in range(6):
            q[i] = np.clip(q[i], self.joint_limits[f'J{i}'][0], self.joint_limits[f'J{i}'][1])
        
        return q
    
    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        数值雅可比矩阵 (6 x 6)
        前3行: 位置雅可比，后3行: 姿态(轴角微分)雅可比。
        """
        q = np.asarray(joint_angles, dtype=float)
        n = len(q)
        J = np.zeros((6, n))
        eps = 1e-6

        p0, r0 = self.forward_kinematics_with_orientation(q)
        for i in range(n):
            q_plus = q.copy()
            q_plus[i] += eps
            p1, r1 = self.forward_kinematics_with_orientation(q_plus)

            J[:3, i] = (p1 - p0) / eps
            dR = r1 @ r0.T
            J[3:, i] = self._rotation_matrix_to_axis_angle(dR) / eps

        return J

    @staticmethod
    def _rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
        """将旋转矩阵转换为轴角向量 (axis * angle)"""
        trace = np.trace(R)
        angle = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
        if np.abs(angle) < 1e-10:
            return np.zeros(3)

        denom = 2.0 * np.sin(angle)
        if np.abs(denom) < 1e-10:
            return np.zeros(3)

        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / denom
        return axis * angle

    def get_last_solve_info(self) -> Dict[str, Any]:
        """获取最近一次 IK 求解信息"""
        return self.last_solve_info.copy()
    
    def get_joint_limits(self) -> dict:
        """获取关节限制"""
        return self.joint_limits.copy()
    
    def is_within_limits(self, joint_angles: np.ndarray) -> Tuple[bool, List[str]]:
        """
        检查关节角度是否在限制范围内
        
        Args:
            joint_angles: 关节角度数组
            
        Returns:
            (is_valid, violations) - 是否有效及违反限制的关节名称列表
        """
        violations = []
        for i, angle in enumerate(joint_angles):
            jname = f'J{i}'
            if angle < self.joint_limits[jname][0]:
                violations.append(f'{jname} below minimum')
            elif angle > self.joint_limits[jname][1]:
                violations.append(f'{jname} above maximum')
        
        return len(violations) == 0, violations
