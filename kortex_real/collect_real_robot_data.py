#!/usr/bin/env python
"""
实机数据收集脚本
用于 Gen3 Lite 机械臂的数据采集

使用方式:
    python collect_real_robot_data.py
    
控制方式:
    - 键盘控制 (默认): 
        W/S: 关节1 前后
        A/D: 关节2 左右
        Q/E: 关节3 旋转
        R/F: 关节4 上下
        T/G: 关节5 前后
        Y/H: 关节6 旋转
        空格: 夹爪开/闭
        Z: 保存并重置
        X: 退出
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent / "lerobot" / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.cameras import CameraConfig

# 导入 Gen3 Lite
from kortex_real.gen3.gen3_lite import Gen3Lite
from kortex_real.gen3.config_gen3_lite import Gen3LiteConfig

# ============= 配置 =============
REPO_NAME = "kortex_real_data"
DATASET_ROOT = "./real_robot_data"
FPS = 20
NUM_EPISODES = 10

# 机械臂配置
ROBOT_IP = "192.168.1.10"

# 相机配置 (根据你的实际相机调整)
CAMERAS = {
    "top": {
        "type": "opencv",
        "index_or_path": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
    },
    "wrist": {
        "type": "opencv", 
        "index_or_path": 1,
        "width": 640,
        "height": 480,
        "fps": 30,
    }
}


def create_robot_config() -> Gen3LiteConfig:
    """创建机械臂配置"""
    cameras = {}
    for name, cfg in CAMERAS.items():
        cameras[name] = CameraConfig(
            type=cfg["type"],
            index_or_path=cfg["index_or_path"],
            width=cfg["width"],
            height=cfg["height"],
            fps=cfg["fps"],
        )
    
    return Gen3LiteConfig(
        ip_address=ROBOT_IP,
        username="admin",
        password="admin",
        control_mode="joint",
        gripper_enabled=True,
        cameras=cameras,
    )


def keyboard_control(action: np.ndarray, gripper_state: bool) -> tuple[np.ndarray, bool, bool, bool]:
    """
    键盘控制机械臂
    
    Returns:
        (action, reset, exit, gripper_state)
    """
    import msvcrt  # Windows 专用
    
    delta = np.zeros(6)
    reset = False
    exit_flag = False
    
    # 检测按键 (非阻塞)
    if msvcrt.kbhit():
        key = msvcrt.getch()
        key_char = key.decode('utf-8', errors='ignore').lower()
        
        step = 0.1  # 关节移动步长 (弧度)
        
        if key_char == 'w':
            delta[0] = -step
        elif key_char == 's':
            delta[0] = step
        elif key_char == 'a':
            delta[1] = -step
        elif key_char == 'd':
            delta[1] = step
        elif key_char == 'q':
            delta[2] = -step
        elif key_char == 'e':
            delta[2] = step
        elif key_char == 'r':
            delta[3] = -step
        elif key_char == 'f':
            delta[3] = step
        elif key_char == 't':
            delta[4] = -step
        elif key_char == 'g':
            delta[4] = step
        elif key_char == 'y':
            delta[5] = -step
        elif key_char == 'h':
            delta[5] = step
        elif key_char == ' ':
            gripper_state = not gripper_state
        elif key_char == 'z':
            reset = True
        elif key_char == 'x':
            exit_flag = True
    
    action = np.concatenate([delta, [float(gripper_state), float(gripper_state)]])  # 6 joints + 2 fingers
    return action, reset, exit_flag, gripper_state


def main():
    print("=" * 60)
    print("Gen3 Lite 实机数据收集")
    print("=" * 60)
    
    # 1. 创建或加载数据集
    print("\n[1/4] 初始化数据集...")
    if os.path.exists(DATASET_ROOT):
        print(f"  加载已有数据集: {DATASET_ROOT}")
        dataset = LeRobotDataset(REPO_NAME, root=DATASET_ROOT)
    else:
        print(f"  创建新数据集: {DATASET_ROOT}")
        FEATURES = {
            "observation.image": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.wrist_image": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (22,),  # 6 pos + 6 vel + 6 ee + 3 gripper (finger1, finger2, avg)
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),  # 6 joints + finger1 + finger2
            },
        }
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            root=DATASET_ROOT,
            fps=FPS,
            robot_type="kortex",
            features=FEATURES,
        )
    
    print(f"  视频键: {dataset.meta.video_keys}")
    
    # 2. 连接机械臂
    print("\n[2/4] 连接机械臂...")
    robot_config = create_robot_config()
    robot = Gen3Lite(robot_config)
    
    print(f"  机械臂 IP: {ROBOT_IP}")
    print(f"  控制模式: {robot_config.control_mode}")
    
    try:
        robot.connect()
        print("  ✓ 机械臂已连接")
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        return
    
    # 3. 数据收集循环
    print("\n[3/4] 开始数据收集...")
    print("-" * 40)
    print("控制说明:")
    print("  W/S: 关节1  |  R/F: 关节4")
    print("  A/D: 关节2  |  T/G: 关节5")
    print("  Q/E: 关节3  |  Y/H: 关节6")
    print("  空格: 夹爪  |  Z: 保存并重置 | X: 退出")
    print("-" * 40)
    
    episode_id = 0
    gripper_state = False
    action = np.zeros(8)  # 6 joints + 2 fingers
    recording = False
    frame_count = 0
    
    try:
        while episode_id < NUM_EPISODES:
            # 获取观测
            obs = robot.get_observation()
            
            # 获取图像
            images = {}
            for key in ["image", "wrist_image"]:
                full_key = f"observation.{key}"
                if full_key in obs and obs[full_key] is not None:
                    images[full_key] = obs[full_key]
            
            # 获取状态: 6 pos + 6 vel + 6 ee + 3 gripper (finger1, finger2, avg) = 21
            state = []
            for joint in robot.joint_names:
                if f"{joint}.pos" in obs:
                    state.append(obs[f"{joint}.pos"])
                if f"{joint}.vel" in obs:
                    state.append(obs[f"{joint}.vel"])
            # 末端位姿
            for key in ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz"]:
                state.append(obs.get(key, 0.0))
            # 双指夹爪
            state.append(obs.get("gripper.finger_1.pos", 0.0))
            state.append(obs.get("gripper.finger_2.pos", 0.0))
            state.append(obs.get("gripper.pos", 0.0))
            
            # 键盘控制
            action, reset, exit_flag, gripper_state = keyboard_control(action, gripper_state)
            
            if exit_flag:
                break
                
            if reset:
                if recording:
                    print(f"  保存 Episode {episode_id}, 帧数: {frame_count}")
                    dataset.save_episode()
                    episode_id += 1
                    recording = False
                    frame_count = 0
                robot.reset()
                continue
            
            # 执行动作 (支持双指夹爪)
            action_dict = {}
            for i, joint in enumerate(robot.joint_names):
                action_dict[f"{joint}.pos"] = action[i]
            # 双指夹爪控制
            action_dict["gripper.finger_1.pos"] = action[6]
            action_dict["gripper.finger_2.pos"] = action[6]
            
            robot.send_action(action_dict)
            
            # 记录数据
            if not recording and np.sum(np.abs(action[:6])) > 0.001:
                recording = True
                print(f"  ▶ 开始记录 Episode {episode_id}")
            
            if recording:
                frame = {
                    "observation.image": images.get("observation.image", np.zeros((480, 640, 3), dtype=np.uint8)),
                    "observation.wrist_image": images.get("observation.wrist_image", np.zeros((480, 640, 3), dtype=np.uint8)),
                    "observation.state": np.array(state[:19], dtype=np.float32),
                    "action": action.astype(np.float32),
                }
                dataset.add_frame(frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"    帧数: {frame_count}")
            
            time.sleep(1.0 / FPS)
            
    except KeyboardInterrupt:
        print("\n  用户中断")
    finally:
        if recording and frame_count > 0:
            print(f"  保存 Episode {episode_id}, 帧数: {frame_count}")
            dataset.save_episode()
        
        print("\n[4/4] 断开机械臂连接...")
        robot.disconnect()
        print("  ✓ 完成")
    
    print(f"\n数据集保存位置: {DATASET_ROOT}")
    print(f"总 Episode 数: {episode_id + 1}")


if __name__ == "__main__":
    main()