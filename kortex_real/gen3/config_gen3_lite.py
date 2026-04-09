#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'lerobot', 'src'))

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig


@dataclass
class Gen3LiteConfig(RobotConfig):
    """Gen3Lite机械臂配置类"""
    # 网络配置
    ip_address: str = "192.168.1.10"
    username: str = "admin"
    password: str = "admin"
    
    # 控制配置
    control_mode: str = "joint"  # "joint" 或 "cartesian"
    max_velocity: float = 0.5  # 最大关节速度 (rad/s)
    max_acceleration: float = 0.5  # 最大关节加速度 (rad/s²)
    
    # 末端执行器配置
    gripper_enabled: bool = True
    
    # 相机配置
    cameras: dict[str, CameraConfig] = field(default_factory=dict)