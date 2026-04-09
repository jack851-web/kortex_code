#!/usr/bin/env python
"""
测试 Gen3 Lite 机械臂连接
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "lerobot" / "src"))

from kortex_real.gen3.gen3_lite import Gen3Lite
from kortex_real.gen3.config_gen3_lite import Gen3LiteConfig
from lerobot.cameras import CameraConfig

# 配置
ROBOT_IP = "192.168.1.10"

# 创建配置
config = Gen3LiteConfig(
    ip_address=ROBOT_IP,
    control_mode="joint",
    gripper_enabled=True,
    cameras={
        "top": CameraConfig(type="opencv", index_or_path=0, width=640, height=480, fps=30),
    }
)

# 创建机械臂
robot = Gen3Lite(config)

# 连接
print("尝试连接机械臂...")
try:
    robot.connect()
    print("✓ 连接成功!")
    
    # 测试获取观测
    print("\n获取观测数据...")
    obs = robot.get_observation()
    print(f"  关节位置: {[f'{k}: {v:.3f}' for k, v in obs.items() if 'pos' in k and 'ee' not in k][:3]}")
    print(f"  末端位置: x={obs.get('ee.x', 0):.3f}, y={obs.get('ee.y', 0):.3f}, z={obs.get('ee.z', 0):.3f}")
    print(f"  夹爪位置: {obs.get('gripper.pos', 0):.3f}")
    print(f"  相机: {list(obs.keys())}")
    
    # 测试动作
    print("\n测试发送动作...")
    action = {f"joint_{i+1}.pos": 0.0 for i in range(6)}
    action["gripper.pos"] = 0.0
    robot.send_action(action)
    print("✓ 动作发送成功!")
    
    # 断开
    robot.disconnect()
    print("\n✓ 测试完成!")
    
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()