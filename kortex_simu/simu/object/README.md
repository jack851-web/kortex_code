# MuJoCo 可抓取物体说明

## 目录结构

```
object/
├── objects.xml          # 所有物体定义 (可通过 include 导入)
└── README.md            # 本说明文件

env/
├── task_pick_place.xml        # 原始场景 (单个球)
├── task_pick_place_simple.xml # 简化场景 (球 + 立方体)
└── task_pick_place_multi.xml  # 多物体场景 (所有物体)
```

## 物体类型

### 球体 (Sphere)
| 名称 | 尺寸 (半径) | 质量 | 颜色 | 抓取难度 |
|------|-------------|------|------|----------|
| `ball_green` | 33mm | 58g | 绿色 | 中等 |
| `ball_red` | 33mm | 58g | 红色 | 中等 |
| `ball_blue` | 33mm | 58g | 蓝色 | 中等 |
| `ball_small` | 25mm | 30g | 黄色 | 简单 |

### 立方体 (Box)
| 名称 | 尺寸 (半边长) | 质量 | 颜色 | 抓取难度 |
|------|---------------|------|------|----------|
| `cube_red` | 20mm | 50g | 红色 | 中等 |
| `cube_blue` | 20mm | 50g | 蓝色 | 中等 |
| `cube_green` | 20mm | 50g | 绿色 | 中等 |
| `cube_small` | 15mm | 20g | 黄色 | 简单 |

### 圆柱体 (Cylinder)
| 名称 | 半径 × 半高 | 质量 | 颜色 | 抓取难度 |
|------|-------------|------|------|----------|
| `cylinder_red` | 20mm × 30mm | 80g | 红色 | 中等 |
| `cylinder_blue` | 20mm × 30mm | 80g | 蓝色 | 中等 |
| `cylinder_small` | 15mm × 20mm | 30g | 青色 | 简单 |

### 特殊形状
| 名称 | 类型 | 尺寸 | 质量 | 颜色 |
|------|------|------|------|------|
| `box_long` | 长方体 | 40×15×15mm | 40g | 棕色 |
| `ellipsoid` | 椭球体 | 30×20×20mm | 50g | 紫色 |

## 使用方法

### 方法 1: 使用预设场景

```python
# 使用多物体场景
xml_path = "simu/env/task_pick_place_multi.xml"
env = SimpleEnv(xml_path)

# 使用简化场景
xml_path = "simu/env/task_pick_place_simple.xml"
env = SimpleEnv(xml_path)
```

### 方法 2: 导入物体定义

在自定义场景 XML 中添加：

```xml
<include file="../object/objects.xml"/>
```

### 方法 3: 单独添加物体

```xml
<!-- 添加一个球 -->
<body name="my_ball" pos="0.4 0.0 0.033">
  <freejoint/>
  <geom type="sphere" size="0.033" mass="0.058"
        rgba="0.2 0.8 0.2 1" friction="1 0.01 0.001"/>
</body>

<!-- 添加一个立方体 -->
<body name="my_cube" pos="0.45 0.15 0.02">
  <freejoint/>
  <geom type="box" size="0.02 0.02 0.02" mass="0.05"
        rgba="0.9 0.1 0.1 1" friction="1 0.01 0.001"/>
</body>
```

## 物理参数说明

- **friction**: `1 0.01 0.001` (滑动摩擦、扭转摩擦、滚动摩擦)
- **mass**: 根据物体体积和密度计算
- **freejoint**: 允许物体自由移动 (6个自由度)

## 抓取建议

1. **初学者**: 使用 `ball_small` 或 `cube_small` (尺寸小，易抓取)
2. **中等难度**: 使用标准球体或立方体
3. **高难度**: 使用椭球体或长方体 (形状不规则)

## 注意事项

- 物体位置应与机器人工作空间匹配
- 确保 `freejoint` 在 body 内部
- 调整 `pos` 参数改变物体初始位置