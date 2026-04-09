# Kortex 实机-仿真同步数据收集系统

## 项目概述

本项目旨在实现一个实机与仿真同步的数据收集系统，用于抓取任务的数据采集。通过实机机械臂与Mujoco仿真环境的同步控制，自动化收集LeRobot格式的数据集。

---

## 功能需求详细描述

### 1. 实机数据采集模块
**功能**：获取实机机械臂关节状态和相机图像数据
- 通过网络连接获取Gen3 Lite机械臂的实时关节角度
- 从多个相机（top、wrist等）获取RGB图像
- 支持实时数据流输出到仿真环境

### 2. 仿真跟随模块
**功能**：将实机关节状态映射到Mujoco仿真环境
- 接收实机关节角度
- 同步控制仿真机械臂跟随实机运动
- 保持实机与仿真的状态一致性

### 3. 抓取任务执行模块
**功能**：执行预定义路径点的抓取任务
- 读取任务配置（100个任务目标位置）
- 按固定路径点执行抓取：
  1. 移动到预抓取位置
  2. 下沉接近物体
  3. 闭合夹爪抓取
  4. 抬起物体
  5. 移动到放置位置
  6. 释放夹爪
- 记录每个动作的图像和状态数据

### 4. 自动化数据集构建模块
**功能**：自动化收集LeRobot格式数据集
- 任务列表定义（目标物体位置）
- 自动执行完一个任务后暂停
- 提示人工切换物品位置
- 继续执行下一个任务
- 支持断点续传

### 5. 双路数据同步采集
**功能**：同时收集实机和仿真数据
- 实机侧：关节角度 + 相机图像
- 仿真侧：关节角度 + 仿真图像（可选）
- 同一时间戳存储，便于后续对齐

### 6. 仿真物体位置控制
**功能**：每个任务执行后自动更新仿真物体位置
- 读取下一个任务的目标物体位置
- 在仿真环境中移动物体到指定位置
- 无需人工干预仿真侧

---

## 技术架构

```
collect_data/
├── config/
│   └── tasks_config.yaml      # 任务列表配置
├── scripts/
│   ├── __init__.py
│   ├── real_interface.py       # 实机接口封装
│   ├── simu_interface.py      # 仿真接口封装
│   ├── sync_controller.py     # 同步控制器
│   ├── grasp_executor.py       # 抓取任务执行器
│   └── data_collector.py      # 数据收集器
├── main.py                     # 主程序入口
└── README.md
```

---

## 实施计划

### 阶段1：环境搭建与接口封装（1-2天）

**任务1.1**：创建项目目录结构
```
collect_data/
├── config/
├── scripts/
├── data/          # 收集的数据存放
└── logs/          # 日志存放
```

**任务1.2**：实现实机接口（`real_interface.py`）
- 封装Gen3Lite类连接
- 实现关节状态读取
- 实现夹爪控制
- 实现相机图像读取

**任务1.3**：实现仿真接口（`simu_interface.py`）
- 封装SimpleEnv环境
- 实现关节状态设置与读取
- 实现物体位置控制
- 实现图像渲染

### 阶段2：同步控制器开发（1-2天）

**任务2.1**：实现同步控制（`sync_controller.py`）
- 创建实机-仿真状态同步线程
- 实现关节角度映射
- 添加状态一致性检查
- 处理异常情况

**任务2.2**：实现仿真跟随模式
- 实机运动时仿真实时跟随
- 支持平滑插值
- 添加延迟补偿

### 阶段3：抓取执行器开发（2-3天）

**任务3.1**：设计抓取路径点
```python
# 标准抓取路径
HOME_POSITION = [0°, 0°, 0°, 0°, 0°, 0°]
PRE_GRASP = [x, y, z_above, roll, pitch, yaw]
GRASP = [x, y, z_on_object, roll, pitch, yaw]
LIFT = [x, y, z_above, roll, pitch, yaw]
PLACE = [x_target, y_target, z_above, roll, pitch, yaw]
```

**任务3.2**：实现抓取流程（`grasp_executor.py`）
- 路径点插值平滑移动
- 夹爪开闭时序控制
- 抓取成功检测
- 异常处理与恢复

### 阶段4：数据收集器开发（2-3天）

**任务4.1**：配置任务列表（`config/tasks_config.yaml`）
```yaml
tasks:
  - task_id: 1
    object_position: [0.3, 0.0, 0.05]
    target_position: [0.4, 0.2, 0.05]
    description: "抓取红色方块放到蓝色目标"
  - task_id: 2
    object_position: [0.25, -0.1, 0.05]
    target_position: [0.35, 0.15, 0.05]
  # ... 共100个任务
```

**任务4.2**：实现数据收集（`data_collector.py`）
- LeRobot数据集格式封装
- 图像+状态同步存储
- 任务进度保存与加载
- 统计信息记录

### 阶段5：主程序集成（1-2天）

**任务5.1**：实现主程序（`main.py`）
- 启动实机连接
- 启动仿真环境
- 启动同步控制
- 加载任务配置
- 执行数据收集主循环

**任务5.2**：添加交互功能
- 任务完成后暂停提示
- 人工确认继续
- 进度显示
- 中断保存

### 阶段6：测试与优化（2-3天）

**任务6.1**：单元测试
- 实机接口测试
- 仿真接口测试
- 同步控制测试

**任务6.2**：集成测试
- 端到端流程测试
- 100个任务连续执行测试
- 数据质量检查

**任务6.3**：优化
- 同步延迟优化
- 数据存储效率优化
- 异常处理完善

---

## 接口定义

### RealInterface
```python
class RealInterface:
    def connect(self, ip: str) -> bool
    def get_joint_state(self) -> np.ndarray  # 返回6个关节角度
    def set_joint_target(self, joints: np.ndarray) -> bool
    def get_gripper_state(self) -> float  # 0.0-1.0
    def set_gripper(self, position: float) -> bool
    def get_camera_images(self) -> dict  # {"top": image, "wrist": image}
    def disconnect(self)
```

### SimuInterface
```python
class SimuInterface:
    def __init__(self, xml_path: str)
    def set_joint_state(self, joints: np.ndarray) -> bool
    def get_joint_state(self) -> np.ndarray
    def set_object_position(self, name: str, position: np.ndarray) -> bool
    def get_object_position(self, name: str) -> np.ndarray
    def render(self) -> np.ndarray
    def step(self)
    def close(self)
```

### SyncController
```python
class SyncController:
    def __init__(self, real: RealInterface, simu: SimuInterface)
    def start_sync(self)
    def stop_sync(self)
    def is_synced(self) -> bool
```

---

## 数据格式

### 收集的数据结构
```
data/
├── episode_001/
│   ├── observation/
│   │   ├── top_image/          # 顶部相机图像序列
│   │   ├── wrist_image/       # 手腕相机图像序列
│   │   └── state.json         # 关节状态序列
│   ├── action/
│   │   └── action.json        # 动作序列
│   └── metadata.json           # 任务元信息
├── episode_002/
│   └── ...
└── dataset_info.json           # 数据集总体信息
```

### metadata.json 格式
```json
{
    "task_id": 1,
    "object_position": [0.3, 0.0, 0.05],
    "target_position": [0.4, 0.2, 0.05],
    "start_time": "2024-01-01T10:00:00",
    "end_time": "2024-01-01T10:01:00",
    "success": true,
    "real_joints": [[...], [...], ...],
    "simu_joints": [[...], [...], ...]
}
```

---

## 任务配置示例

### tasks_config.yaml
```yaml
robot:
  ip: "192.168.1.10"
  control_mode: "joint"
  gripper_enabled: true

cameras:
  top:
    type: "opencv"
    index: 0
    width: 640
    height: 480
  wrist:
    type: "opencv"
    index: 1
    width: 640
    height: 480

simulation:
  xml_path: "./kortex_simu/simu/env/task_pick_place.xml"
  object_name: "ball"
  target_name: "place_target"

grasp:
  home_position: [0, 0, 0, 0, 0, 0]
  pre_grasp_offset: [0, 0, 0.1, 0, 0, 0]  # 相对于物体的偏移
  lift_height: 0.15
  approach_height: 0.05

dataset:
  repo_name: "kortex_grasp_data"
  root: "./collected_data"
  fps: 20

tasks:
  total: 100
  positions:
    - object: [0.3, 0.0, 0.05]
      target: [0.4, 0.2, 0.05]
    - object: [0.25, -0.1, 0.05]
      target: [0.35, 0.15, 0.05]
    # ... 更多任务
```

---

## 使用流程

1. **启动准备**
   ```bash
   cd D:\VLA\kortex_code\collect_data
   python main.py --config config/tasks_config.yaml
   ```

2. **系统初始化**
   - 连接实机机械臂
   - 启动仿真环境
   - 建立同步连接
   - 加载第一个任务

3. **数据收集循环**
   ```
   for task in tasks:
       1. 显示任务信息（物体位置、目标位置）
       2. 提示人工放置物体
       3. 等待确认开始
       4. 执行抓取任务
       5. 保存Episode数据
       6. 更新仿真物体位置
       7. 提示切换下一个物体
       8. 继续下一个任务
   ```

4. **完成**
   - 显示收集统计
   - 保存最终数据集
   - 断开连接

---

## 注意事项

1. **安全检查**
   - 实机运动前确认环境安全
   - 设置关节角度限制
   - 监控异常情况

2. **数据备份**
   - 定期保存进度
   - 支持中断恢复
   - 验证数据完整性

3. **性能优化**
   - 图像压缩存储
   - 多线程数据写入
   - 内存管理

---

## 预计开发时间

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| 1 | 环境搭建与接口封装 | 1-2天 |
| 2 | 同步控制器开发 | 1-2天 |
| 3 | 抓取执行器开发 | 2-3天 |
| 4 | 数据收集器开发 | 2-3天 |
| 5 | 主程序集成 | 1-2天 |
| 6 | 测试与优化 | 2-3天 |
| **总计** | | **9-15天** |
