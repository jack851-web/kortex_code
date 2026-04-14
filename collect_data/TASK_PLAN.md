# Kortex 实机-仿真同步数据收集系统

## 项目概述

本项目是一个实机与仿真同步的数据收集系统，支持 **Mock 模式**（纯仿真）和 **Real 模式**（实机同步），用于抓取任务的数据采集。通过 MuJoCo 仿真环境与 Kinova Gen3 Lite 机械臂的协同控制，自动化收集 LeRobot v3.0 格式的数据集。

### 核心特性

- **双模式运行**: Mock 模式（纯仿真 IK 控制）和 Real 模式（实机同步 + 吸附机制）
- **MuJoCo 物理仿真**: 高精度物理引擎，支持刚体动力学和碰撞检测
- **逆运动学求解**: 基于 MuJoCo 的实时 IK 求解器
- **多相机支持**: 支持多视角相机渲染和数据采集
- **键盘遥操作**: 类似 lerobot 的键盘控制方式
- **随机位置生成**: 支持 `random` 关键字自动生成物体位置
- **LeRobot v3.0 格式**: 兼容 HuggingFace 数据集格式

---

## 功能模块

### 1. 实机数据采集模块 (`scripts/real/`)

**功能**：获取实机机械臂关节状态和相机图像数据
- 通过网络连接获取 Gen3 Lite 机械臂的实时关节角度
- 从多个相机（top、side、hand）获取 RGB 图像
- 支持实时数据流输出到仿真环境
- 发布数据到消息总线供其他模块订阅

### 2. 仿真环境模块 (`scripts/simu/`)

**功能**：MuJoCo 仿真环境管理
- 加载和管理 MuJoCo 模型
- GLFW 可视化窗口渲染
- IK 求解和运动控制
- 多相机渲染
- 物体动态加载和位置控制
- 流式视频编码

### 3. 同步控制模块 (`scripts/control/`)

**功能**：实机-仿真状态同步
- 实时关节状态同步
- 物体吸附机制（Real 模式）
- 状态一致性检查
- 异常处理与恢复

### 4. 抓取执行模块 (`scripts/control/`)

**功能**：执行预定义路径点的抓取任务
- 预抓取位姿计算
- 抓取轨迹规划
- 物体类型适配（cup、mug、plate 等）
- 抬升和放置动作
- 支持键盘微调（Mock 模式）

### 5. 数据收集模块 (`scripts/simu/data_collector.py`, `scripts/real/data_collector.py`)

**功能**：LeRobot v3.0 格式数据收集
- LeRobot API 直录
- 流式视频编码 (libsvtav1)
- Episode 管理
- 进度保存与恢复
- Chunk/File 自动管理

### 6. 消息总线模块 (`scripts/core/`)

**功能**：模块间通信
- 发布-订阅模式
- 实机数据发布
- 仿真数据发布
- 线程安全

---

## 技术架构

```
collect_data/
├── main_qt.py                   # 主程序入口（Qt GUI）
├── config/                      # 配置文件
│   ├── tasks_config.yaml        # 任务和物体配置（实机模式）
│   ├── real_config.yaml         # 实机模式配置
│   ├── simu_config.yaml         # 仿真模式配置
│   └── object/                  # 物体特定配置
│       └── cup.yaml
├── gui/                         # Qt GUI 界面
│   ├── main_window.py           # 主窗口
│   ├── mock_task_tuner_window.py # Mock模式调参窗口
│   └── camera_widget.py         # 相机显示组件
├── scripts/                     # 核心脚本
│   ├── control/                 # 控制模块
│   │   ├── grasp_executor.py    # 抓取执行器
│   │   └── sync_controller.py   # 虚实同步控制器
│   ├── core/                    # 核心模块
│   │   ├── message_bus.py       # 消息总线
│   │   └── topic_defs.py        # 主题定义
│   ├── real/                    # 实机模块
│   │   ├── interface.py         # 真实机器人接口
│   │   ├── data_collector.py    # 实机数据收集器
│   │   ├── publisher.py         # 实机数据发布者
│   │   └── camera.py            # 相机接口
│   ├── simu/                    # 仿真模块
│   │   ├── interface.py         # 仿真接口
│   │   ├── data_collector.py    # 仿真数据收集器
│   │   ├── manager.py           # 仿真管理器
│   │   ├── publisher.py         # 仿真数据发布者
│   │   └── render_process.py    # 渲染进程
│   └── _legacy/                 # 旧版代码（已弃用）
└── TASK_PLAN.md                 # 本文档
```

---

## 工作流程

### Mock 模式工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                     Mock 模式工作流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 启动程序                                                 │
│     python main_qt.py --mock                                │
│            │                                                │
│            ▼                                                │
│  2. 初始化仿真环境                                           │
│     - 加载 MuJoCo 模型                                       │
│     - 启动 GLFW 可视化窗口                                   │
│     - 初始化 IK 求解器                                       │
│            │                                                │
│            ▼                                                │
│  3. 加载任务配置                                             │
│     - 从 simu_config.yaml 读取任务列表                       │
│     - 支持 object_position: "random" 随机生成               │
│            │                                                │
│            ▼                                                │
│  4. 执行任务                                                 │
│     ┌─────────────────────────────────────────┐             │
│     │  a. IK 解算到预抓取位置                  │             │
│     │  b. 键盘微调位置                         │             │
│     │  c. 下降到抓取位置                       │             │
│     │  d. 闭合夹爪                            │             │
│     │  e. 抬升物体                            │             │
│     │  f. IK 解算到放置位置                   │             │
│     │  g. 下降放置                            │             │
│     │  h. 松开夹爪                            │             │
│     │  i. 返回初始位置                        │             │
│     └─────────────────────────────────────────┘             │
│            │                                                │
│            ▼                                                │
│  5. 保存数据集                                               │
│     - LeRobot v3.0 格式                                      │
│     - 视频流式编码 (libsvtav1)                               │
│            │                                                │
│            ▼                                                │
│  6. 下一个任务或结束                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Real 模式工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                     Real 模式工作流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 启动程序                                                 │
│     python main_qt.py --real                                │
│            │                                                │
│            ▼                                                │
│  2. 连接实机                                                 │
│     - 连接 Kinova 机械臂                                     │
│     - 连接实机相机                                           │
│     - 初始化仿真环境（同步用）                               │
│            │                                                │
│            ▼                                                │
│  3. 同步控制                                                 │
│     ┌─────────────────────────────────────────┐             │
│     │  实机关节状态 ──→ 仿真关节状态           │             │
│     │        │                                │             │
│     │        ▼                                │             │
│     │  吸附机制：夹爪接近物体时自动吸附        │             │
│     │        │                                │             │
│     │        ▼                                │             │
│     │  仿真物体跟随夹爪移动                    │             │
│     └─────────────────────────────────────────┘             │
│            │                                                │
│            ▼                                                │
│  4. 执行任务                                                 │
│     - 遥操作控制实机                                         │
│     - 仿真同步显示                                           │
│     - 自动吸附辅助抓取                                       │
│            │                                                │
│            ▼                                                │
│  5. 保存双数据流                                             │
│     ┌─────────────────────────────────────────┐             │
│     │  realdata/  - 实机相机数据              │             │
│     │  simudata/  - 同步仿真数据              │             │
│     └─────────────────────────────────────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 接口定义

### RealInterface (`scripts/real/interface.py`)

```python
class RealInterface:
    def connect(self, ip: str) -> bool
    def disconnect(self)
    def get_joint_state(self) -> np.ndarray      # 返回6个关节角度（度）
    def get_gripper_state(self) -> float         # 0.0-1.0
    def get_camera_images(self) -> dict          # {"top": image, ...}
    def get_full_state(self) -> Tuple[np.ndarray, np.ndarray, float]
    def is_connected(self) -> bool
```

### SimuInterface (`scripts/simu/interface.py`)

```python
class SimuInterface:
    def __init__(self, xml_path: str, use_ik: bool = False)
    
    # 关节控制
    def set_joint_state(self, joints: np.ndarray) -> bool
    def get_joint_state(self) -> np.ndarray
    
    # IK 控制
    def set_end_effector_pose(self, position: np.ndarray, orientation: np.ndarray) -> bool
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]
    
    # 物体控制
    def set_object_position(self, name: str, position: np.ndarray) -> bool
    def get_object_position(self, name: str) -> np.ndarray
    def attach_object(self, object_body: str) -> bool
    def detach_object(self) -> bool
    
    # 相机渲染
    def get_camera_images(self, camera_names: List[str]) -> dict
    def render(self) -> np.ndarray
    
    # 仿真步进
    def step(self, n_steps: int = 1)
    
    # GLFW 可视化
    def start_glfw_viewer(self)
    def close_glfw_viewer(self)
```

### GraspExecutor (`scripts/control/grasp_executor.py`)

```python
class GraspExecutor:
    def __init__(self, real_interface, simu_interface, home_position: List[float], ...)
    
    def set_task(self, object_position: np.ndarray, plate_position: np.ndarray, 
                 object_type: str = "cube")
    def set_object_type(self, object_type: str)
    
    def move_to_home(self) -> bool
    def move_to_pre_grasp(self) -> bool
    def move_to_grasp(self) -> bool
    def close_gripper(self) -> bool
    def lift_object(self) -> bool
    def move_to_place(self) -> bool
    def release_object(self) -> bool
    
    def execute_full_task(self, callback: Callable = None) -> bool
```

### SyncController (`scripts/control/sync_controller.py`)

```python
class SyncController:
    def __init__(self, real_interface, simu_interface)
    
    def start_sync(self)
    def stop_sync(self)
    def is_synced(self) -> bool
    
    # 吸附机制
    def enable_adhesion(self, object_body: str, threshold: float = 0.05)
    def disable_adhesion(self)
    def check_and_attach(self) -> bool
```

### DataCollector (`scripts/simu/data_collector.py`)

```python
class SimuDataCollector:
    def __init__(self, simu_interface, data_root: str, fps: int = 20, ...)
    
    def start_collection(self)
    def stop_collection(self)
    
    def start_episode(self, episode_id: int, camera_names: list, task_info: dict)
    def end_episode(self, episode_id: int, success: bool = True) -> bool
    
    def start_recording(self)
    def stop_recording(self)
    def discard_current_task()
    
    def collect_frame(self)
    def finalize(self)
```

---

## 数据格式

### LeRobot v3.0 数据集结构

```
data/
├── Real/
│   ├── realdata/                    # 实机数据
│   │   ├── meta/
│   │   │   ├── info.json            # 数据集元信息
│   │   │   ├── stats.json           # 统计信息
│   │   │   ├── tasks.parquet        # 任务描述
│   │   │   └── episodes/            # Episode 元数据
│   │   │       └── chunk-000/
│   │   │           └── file-000.parquet
│   │   ├── videos/                  # 视频数据
│   │   │   └── observation.images.top/
│   │   │       └── chunk-000/
│   │   │           └── file-000.mp4
│   │   └── data/                    # 状态/动作数据
│   │       └── chunk-000/
│   │           └── file-000.parquet
│   └── simudata/                    # 同步仿真数据（结构相同）
│
└── Simu/
    └── simu_data/                   # 纯仿真数据（结构相同）
```

### 数据特征

| 特征 | 形状 | 说明 |
|------|------|------|
| `observation.state` | (7,) | [6关节角(度), 1夹爪] |
| `action` | (7,) | [6关节增量(度), 1夹爪增量] |
| `observation.images.top` | (480, 640, 3) | 顶部相机图像 |
| `observation.images.side` | (480, 640, 3) | 侧面相机图像 |
| `observation.images.hand` | (480, 640, 3) | 手眼相机图像 |

### meta/info.json 格式

```json
{
  "codebase_version": "v3.0",
  "robot_type": "kortex",
  "total_episodes": 13,
  "total_frames": 19700,
  "total_videos": 39,
  "total_chunks": 1,
  "chunks_size": 1000,
  "data_files_size_in_mb": 100,
  "video_files_size_in_mb": 200,
  "fps": 30,
  "splits": {
    "train": "0:13"
  }
}
```

---

## 配置说明

### 任务配置 (`config/simu_config.yaml`)

```yaml
# 工作空间约束（用于 random 位置生成）
workspace:
  table_bounds: [0.25, 0.50, -0.25, 0.25, 0.44]
  safety_margin: 0.05
  min_object_plate_distance: 0.15

simulation:
  xml_path: "D:/VLA/kortex_code/kortex_simu/simu/env/task_pick_place.xml"
  initial_joints: [-0.594, 0.135, 0.81, -1.79, -1.33, 0.0]
  
  object_library:
    cup:
      model_xml_path: "path/to/cup/model.xml"
      body_name: "body_obj_cup"
    mug:
      model_xml_path: "path/to/mug/model.xml"
      body_name: "body_obj_mug_5"

grasp:
  lift_height: 0.1
  micro_lift_height: 0.02
  release_lift_height: 0.08
  
  object_profiles:
    default:
      orientation: [180, 0, 0]
      pre_grasp_offset: [0, 0, 0.05]
      grasp_offset: [0.0, 0.0, 0.0]
      gripper: {open: 0.0, close: 0.65}
    cup:
      orientation: [176.02, 5.81, -83.23]
      pre_grasp_offset: [0, 0, 0.01]
      grasp_offset: [0.0, 0.012, 0]
      gripper: {open: 0.80, close: 0.80}

tasks:
  task1:
    object_name: "mug"
    object_position: [0.35, 0.15, 0.44]  # 固定位置
    plate_position: [0.35, -0.15, 0.44]
    description: "grasp mug and place to right side"
  
  task2:
    object_name: "mug"
    object_position: "random"  # 随机位置
    plate_position: [0.35, -0.15, 0.44]
    description: "random grasp and place task"
```

---

## 使用流程

### 1. Mock 模式（纯仿真）

```bash
cd D:\VLA\kortex_code\collect_data
python main_qt.py --mock
```

**操作步骤**：
1. 系统自动初始化仿真环境
2. 加载任务配置
3. 点击"执行任务"开始
4. 使用键盘微调位置
5. 任务完成后自动保存数据

### 2. Real 模式（实机同步）

```bash
cd D:\VLA\kortex_code\collect_data
python main_qt.py --real
```

**操作步骤**：
1. 系统连接实机机械臂
2. 初始化仿真环境
3. 建立同步连接
4. 遥操作控制实机执行任务
5. 系统自动保存双数据流

---

## 键盘控制（Mock 模式）

| 按键 | 功能 |
|------|------|
| W/S | 前后移动 (Y轴) |
| A/D | 左右移动 (X轴) |
| R/F | 上下移动 (Z轴) |
| Q/E | 旋转末端执行器 |
| Space | 切换夹爪开合 |
| 1-6 | 预设关节位置 |
| Enter | 确认当前位置 |
| Esc | 取消当前操作 |

---

## 开发状态

### 已完成 ✅

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1.1 | 项目目录结构 | ✅ 完成 |
| 1.2 | 实机接口封装 | ✅ 完成 |
| 1.3 | 仿真接口封装 | ✅ 完成 |
| 2.1 | 同步控制器 | ✅ 完成 |
| 2.2 | 仿真跟随模式 | ✅ 完成 |
| 3.1 | 抓取路径点设计 | ✅ 完成 |
| 3.2 | 抓取执行器 | ✅ 完成 |
| 4.1 | 任务配置 | ✅ 完成 |
| 4.2 | 数据收集器 | ✅ 完成 |
| 5.1 | 主程序集成 | ✅ 完成 |
| 5.2 | Qt GUI 界面 | ✅ 完成 |
| 6.1 | LeRobot v3.0 格式支持 | ✅ 完成 |
| 6.2 | 随机位置生成 | ✅ 完成 |
| 6.3 | 键盘遥操作 | ✅ 完成 |
| 6.4 | GLFW 可视化窗口 | ✅ 完成 |

### 待优化 🔧

| 内容 | 说明 |
|------|------|
| 单元测试 | 添加完整的单元测试覆盖 |
| 性能优化 | 渲染线程优化 |
| 异常处理 | 完善异常恢复机制 |

---

## 常见问题

### 1. rerun 可视化报错 `missing field 'total_chunks'`

**解决方案**:
```bash
pip install --upgrade rerun==1.0.31 rerun-sdk==0.31.2
```

### 2. 第二次启动数据保存位置

LeRobot 会自动管理数据文件：
- 每个 episode 保存后检查文件大小
- 文件大小接近限制时自动创建新文件
- 文件数量达到 `chunks_size` 时自动创建新 chunk

### 3. GLFW 窗口黑屏

**原因**: OpenGL 上下文冲突或渲染线程问题

**解决方案**: 确保渲染在主线程执行，或使用独立的渲染进程。

### 4. 可视化数据集

```bash
# 使用 rerun 可视化
rerun data/Simu/simu_data

# 使用 LeRobot 可视化
python -m lerobot.common.datasets.visualize_dataset --repo-id local/simu_data --root data/Simu
```

---

## 许可证

MIT License

## 致谢

- [MuJoCo](https://mujoco.org/) - 物理仿真引擎
- [LeRobot](https://github.com/huggingface/lerobot) - 数据集格式参考
- [Kinova Robotics](https://www.kinovarobotics.com/) - Gen3 Lite 机械臂
