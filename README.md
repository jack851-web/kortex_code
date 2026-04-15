# Kortex Code - 机器人数据收集与仿真系统

基于 MuJoCo 的 Kinova Gen3 Lite 机械臂开发框架，支持数据收集、仿真控制和模型评估。

## 项目结构

```
kortex_code/
├── collect_data/           # 数据收集模块 → [TASK_PLAN.md](collect_data/TASK_PLAN.md)
├── kortex_simu/            # MuJoCo 仿真环境
│   ├── ik/                 # 逆运动学求解
│   └── simu/               # 场景、机器人、物体模型
├── eval/                   # 模型评估模块 → [EVAL_DESIGN.md](eval/EVAL_DESIGN.md)
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

| 模块 | 说明 | 详细文档 |
|------|------|----------|
| **collect_data** | 实机/仿真数据收集，LeRobot v3.0 格式输出 | [TASK_PLAN.md](collect_data/TASK_PLAN.md) |
| **kortex_simu** | MuJoCo 仿真场景、IK 求解、物体模型 | - |
| **eval** | 训练模型评估（无头模式，支持远程 GPU） | [EVAL_DESIGN.md](eval/EVAL_DESIGN.md) |

## 快速开始

```bash
cd collect_data

# Mock 模式（纯仿真）
python main_qt.py --mock

# Real 模式（实机连接）
python main_qt.py --real
```

## 依赖说明

- **MuJoCo 3.0+** - 物理仿真
- **PyQt5** - GUI 界面
- **LeRobot** - 数据格式、训练框架
- **PyTorch** - 模型推理
- **OpenCV** - 图像处理

## 许可证

MIT License
