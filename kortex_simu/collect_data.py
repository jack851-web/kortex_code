"""
使用 MuJoCo 官方库的数据收集脚本
"""
import sys
import random
import numpy as np
import os
from pathlib import Path

# 添加 lerobot 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "lerobot" / "src"))

from PIL import Image
import mujoco
import mujoco.viewer
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 配置
SEED = 0
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
REPO_NAME = 'kortex_pick_place'
NUM_DEMO = 1
ROOT = "./demo_data"

# 关节名称
JOINT_NAMES = ['J0', 'J1', 'J2', 'J3', 'J4', 'J5']
GRIPPER_JOINTS = ['RIGHT_BOTTOM', 'RIGHT_TIP', 'LEFT_BOTTOM', 'LEFT_TIP']


class MujocoEnv:
    """使用 MuJoCo 官方库的环境类"""
    
    def __init__(self, xml_path, seed=None, show_viewer=True):
        """初始化环境
        
        Args:
            xml_path: XML 文件路径
            seed: 随机种子
            show_viewer: 是否显示 MuJoCo 查看器
        """
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 获取关节 ID 和 qpos 地址
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                         for name in JOINT_NAMES]
        self.joint_qpos_addrs = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        
        self.gripper_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                                  for name in GRIPPER_JOINTS]
        self.gripper_qpos_addrs = [self.model.jnt_qposadr[jid] for jid in self.gripper_joint_ids]
        
        # 获取执行器 ID
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator_{name}") 
                            for name in JOINT_NAMES]
        self.gripper_actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator_{name}") 
                                     for name in GRIPPER_JOINTS]
        
        # 获取物体和站点 ID
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'place_target')
        
        # 相机名称
        self.camera_names = ['agentview', 'top', 'side', 'robot0_eye_in_hand']
        
        # 渲染器
        self.renderer = mujoco.Renderer(self.model, height=256, width=256)
        
        # 状态
        self.seed = seed
        self.gripper_state = False
        
        # 查看器
        self.show_viewer = show_viewer
        self.viewer = None
        if show_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # 设置相机视角
            self.viewer.cam.azimuth = 45
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
        
        # 重置环境
        self.reset()
    
    def reset(self):
        """重置环境"""
        # 重置数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始关节位置 (使用正确的 qpos 地址)
        q_init = np.deg2rad([0, 0, 0, 0, 0, 0])
        for i, q in enumerate(q_init):
            self.data.qpos[self.joint_qpos_addrs[i]] = q
        
        # 设置夹爪初始位置
        for addr in self.gripper_qpos_addrs:
            self.data.qpos[addr] = 0.0
        
        # 前向运动学
        mujoco.mj_forward(self.model, self.data)
        
        # 稳定仿真
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        
        # 记录初始状态
        self.obj_init_pos = self.data.xpos[self.cube_body_id].copy()
        
        return self.get_state()
    
    def get_state(self):
        """获取当前状态"""
        # 获取关节角度 (使用正确的 qpos 地址)
        qpos = np.array([self.data.qpos[addr] for addr in self.joint_qpos_addrs])
        
        # 获取 TCP 位置和姿态
        tcp_pos = self.data.site_xpos[self.tcp_site_id].copy()
        tcp_rot = self.data.site_xmat[self.tcp_site_id].reshape(3, 3).copy()
        
        # 转换为欧拉角 (简化)
        euler = self._rot2euler(tcp_rot)
        
        return {
            'qpos': qpos,
            'tcp_pos': tcp_pos,
            'tcp_rot': tcp_rot,
            'tcp_euler': euler,
            'obj_pos': self.data.xpos[self.cube_body_id].copy(),
        }
    
    def _rot2euler(self, R):
        """旋转矩阵转欧拉角 (简化版)"""
        # 简化的欧拉角计算
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        if sy < 1e-6:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        else:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        return np.array([x, y, z])
    
    def step(self, action, n_steps=50):
        """
        执行动作
        action: [6个关节角度 + 1个夹爪开合]
        n_steps: 仿真步数（默认50步，约0.1秒）
        """
        # 直接设置关节位置 (类似 test_physics.py 的方式)
        for i, addr in enumerate(self.joint_qpos_addrs):
            self.data.qpos[addr] = action[i]
        
        # 设置夹爪位置
        gripper_cmd = action[6] if len(action) > 6 else 0.0
        for addr in self.gripper_qpos_addrs:
            self.data.qpos[addr] = gripper_cmd
        
        # 前向运动学更新位置
        mujoco.mj_forward(self.model, self.data)
        
        # 执行多步仿真，让物理特性生效
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
            # 同步更新查看器
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()
        
        return self.get_state()
    
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
    
    def get_camera_image(self, camera_name):
        """获取相机图像"""
        try:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            self.renderer.update_scene(self.data, camera=camera_name)
            img = self.renderer.render()
            return img
        except:
            return None
    
    def check_success(self):
        """检查是否成功放置"""
        obj_pos = self.data.xpos[self.cube_body_id]
        target_pos = self.data.site_xpos[self.target_site_id]
        dist = np.linalg.norm(obj_pos - target_pos)
        return dist < 0.05  # 5cm 阈值


def main():
    """主函数"""
    TASK_NAME = 'Pick cube and place on target'
    xml_path = os.path.join(os.path.dirname(__file__), 'simu/env/task_pick_place.xml')
    
    # 创建环境
    env = MujocoEnv(xml_path, seed=SEED)
    
    # 数据集设置
    create_new = True
    if os.path.exists(ROOT):
        print(f"Directory {ROOT} already exists.")
        ans = input("Do you want to delete it? (y/n) ")
        if ans == 'y':
            import shutil
            shutil.rmtree(ROOT)
        else:
            create_new = False
    
    if create_new:
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            root=ROOT,
            robot_type="kortex",
            fps=20,
            features={
                "observation.image": {
                    "dtype": "video",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action"],
                },
                "obj_init": {
                    "dtype": "float32",
                    "shape": (3,),
                    "names": ["obj_init"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )
    else:
        dataset = LeRobotDataset(REPO_NAME, root=ROOT)
    
    print("\n使用 MuJoCo 官方库")
    print("按 Ctrl+C 退出")
    
    episode_id = 0
    record_flag = False
    
    # 简单的随机动作示例
    try:
        while episode_id < NUM_DEMO:
            # 随机动作
            action = np.random.randn(7) * 0.1
            action[6] = np.random.rand()  # 夹爪
            
            # 执行动作
            state = env.step(action)
            
            # 获取图像
            img = env.get_camera_image('agentview')
            if img is None:
                img = np.zeros((256, 256, 3), dtype=np.uint8)
            
            # 调整图像大小
            img = np.ascontiguousarray(Image.fromarray(img).resize((256, 256)))
            
            # 检查成功
            if env.check_success():
                print(f"Episode {episode_id} completed!")
                dataset.save_episode()
                env.reset()
                episode_id += 1
            
            # 记录数据
            if record_flag:
                dataset.add_frame({
                    "observation.image": img,
                    "observation.state": state['tcp_euler'].astype(np.float32),
                    "action": action.astype(np.float32),
                    "obj_init": env.obj_init_pos.astype(np.float32),
                }, task=TASK_NAME)
            
            # 开始记录
            if not record_flag:
                record_flag = True
                print("Start recording!")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # 关闭环境
        env.close()
    
    print("Cleanup completed.")


if __name__ == "__main__":
    main()
