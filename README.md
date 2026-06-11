# Kortex Code - 机器人数据收集与仿真系统

基于 MuJoCo 的 Kinova Gen3 Lite 机械臂开发框架，支持实机/仿真数据收集、LeRobot v3.0 格式直录、模型训练与评估。

## 版本信息

| 版本 | 日期 | 说明 |
|------|------|------|
| **v2.0** | 2025-06 | LeRobot v3.0 直录模式、MessageBroker 解耦架构、streaming video 编码、images/ 文件夹 bug 修复 |

### v2.0 核心变更

- **LeRobot v3.0 streaming_encoding 直录**：使用 `LeRobotDataset.create(streaming_encoding=True)` 直接写入 MP4 视频，不再生成中间 PNG 帧
- **MessageBroker 进程内消息总线**：解耦 Qt GUI / MuJoCo 仿真 / 真实机器人三层，发布-订阅模式替代直接调用
- **分段锁渲染策略**：SimuPublisher 逐相机短锁渲染，解决 GLFW viewer 与采集线程的 OpenGL 竞争
- **修复：加载已有数据集时 `images/` 文件夹误创建问题**（`_streaming_encoder` 在 `__init__` 中为 None 导致 add_frame 回退到 PNG 路径）
- **支持多物体动态场景加载**：通过 XML 模板替换实现运行时切换抓取物体（cube/cup/mug/bottle/bowl）
- **Mock 吸附机制**：仿真模式下模拟夹爪抓取-搬运-放置的物理效果

## 项目结构

```
kortex_code/
├── collect_data/           # 数据收集模块（核心）
│   ├── main_qt.py         # Qt GUI 主入口
│   ├── main.py            # CLI 主入口（无GUI）
│   ├── config/            # 配置文件 (real/simu/tasks)
│   ├── scripts/
│   │   ├── real/          # 实机接口 + 数据收集器
│   │   │   ├── interface.py      # RealInterface / MockRealInterface
│   │   │   ├── data_collector.py # RealDataCollector (LeRobot直录)
│   │   │   ├── camera.py        # CameraManager / SimpleCamera
│   │   │   └── publisher.py     # RealPublisher (broker发布者)
│   │   ├── simu/          # 仿真接口 + 数据收集器
│   │   │   ├── interface.py      # SimuInterface (MuJoCo+IK)
│   │   │   ├── data_collector.py # SimuDataCollector (LeRobot直录)
│   │   │   ├── manager.py        # SimuManager (生命周期管理)
│   │   │   ├── publisher.py     # SimuPublisher (broker发布者)
│   │   │   └── render_process.py # 进程渲染器
│   │   ├── core/          # 基础设施
│   │   │   ├── message_bus.py    # MessageBroker (进程内Pub/Sub)
│   │   │   └── topic_defs.py     # 话题名称定义
│   │   └── control/       # 执行器
│   │       ├── grasp_executor.py # GraspExecutor (抓取路径规划)
│   │       └── sync_controller.py # SyncController (实机→仿真同步)
│   ├── gui/               # Qt 界面
│   │   ├── main_window.py  # MainWindow (broker订阅显示)
│   │   ├── camera_widget.py # 相机画面组件
│   │   ├── control_panel.py # 控制面板
│   │   ├── status_panel.py # 状态面板
│   │   └── log_panel.py    # 日志面板
│   └── data_process/      # 数据后处理工具
├── kortex_simu/            # MuJoCo 仿真环境
│   ├── ik/                 # IK 求解 (MuJoCoIK, SimulationController)
│   └── simu/               # 场景XML、机器人模型、物体模型
├── kortex_real/            # Kinova API 封装
│   └── gen3/              # Gen3Lite 驱动
├── scripts/
│   ├── train/             # 训练脚本 (ACT++, OpenPI)
│   └── eval/              # 评估框架 (远程推理、安全监控)
├── data/                  # LeRobot 格式数据集输出目录
│   ├── Real/realdata/     # 实机数据集
│   └── Simu/simu_data/    # 仿真数据集
└── environment.yml         # conda 环境配置
```

## 环境配置

### 方法：使用 conda 环境文件

```bash

# 1. 创建 conda 环境
cd kortex_code
conda env create -f environment.yml
conda activate lerobot

# 2. 安装 lerobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e .
```



## 模块说明

| 模块 | 说明 | 关键类 |
|------|------|--------|
| **collect_data/scripts/real** | 实机数据采集 | `RealInterface`, `RealDataCollector`, `RealPublisher` |
| **collect_data/scripts/simu** | 仿真数据采集 (MuJoCo+IK) | `SimuInterface`, `SimuDataCollector`, `SimuPublisher` |
| **collect_data/scripts/core** | 进程内消息总线 | `MessageBroker`, `Topic` |
| **collect_data/scripts/control** | 抓取执行 & 同步 | `GraspExecutor`, `SyncController` |
| **collect_data/gui** | Qt5 GUI 界面 | `MainWindow`, `CameraPanel`, `ControlPanel` |
| **kortex_simu** | MuJoCo 仿真 + IK 求解 | `MuJoCoIK`, `SimulationController` |
| **scripts/train** | 训练框架 (ACT++, OpenPI) | 训练脚本、数据转换 |
| **scripts/eval** | 模型评估 | 远程推理、安全监控 |

## 数据集格式

输出为 **LeRobot v3.0 标准格式**，可直接用于训练：

```
data/
├── Real/realdata/          # 实机数据集
│   ├── data/chunk-000/     # Parquet 数据文件 (state/action)
│   ├── videos/observation.images.hand/  # MP4 视频文件
│   └── meta/               # info.json, tasks.parquet, stats.json
└── Simu/simu_data/         # 仿真数据集（结构同上）
```

### State / Action 维度

| 维度 | 含义 | 实机范围 | 仿真范围 |
|------|------|---------|---------|
| j1-j6 | 6关节角度 | 原始度数 (0~360) | 归一化 [-1,1] |
| gripper | 夹爪开度 | 0~1 | 0~1 |
| ee_x/y/z | 末端位姿 | 米 (原始值) | 归一化 [-1,1] |

## 快速开始

```bash
# 1. 创建环境
conda env create -f environment.yml && conda activate lerobot

# 2. 安装 LeRobot (v3.0+)
git clone https://github.com/huggingface/lerobot.git && cd lerobot && pip install -e .

# 3. 数据收集
cd collect_data

# Mock 模式（纯仿真，无需硬件）
python main_qt.py --mock

# Real 模式（连接真实机械臂）
python main_qt.py --real

# Simu 模式（仿真 + IK 控制）
python main_qt.py --simu --show-viewer
```

## 运行模式说明

| 参数 | 说明 | 需要硬件 |
|------|------|---------|
| `--mock` | 纯仿真模式（MockRealInterface） | 否 |
| `--real` | 实机模式 | Kinova Gen3 Lite + 相机 |
| `--simu` | 仿真模式 + IK 控制 | 否 |
| `--show-viewer` | 显示 MuJoCo GLFW 3D 视图 | 否 |

## 已知问题 & 待办

- [ ] 统一 Real/Simu state 归一化策略（当前实机用原始值，仿真用归一化）
- [ ] 抽取 BaseDataCollector 基类消除 ~400 行重复代码
- [ ] MessageBroker 单订阅者异常不应中断后续订阅者
- [ ] SyncController 同步线程需增加全局异常保护

## 依赖说明

- **MuJoCo 3.0+** - 物理仿真
- **PyQt5** - GUI 界面
- **LeRobot** - 数据格式、训练框架
- **PyTorch** - 模型推理
- **OpenCV** - 图像处理

## 许可证

MIT License
