# 世界模型发展方向调研

本文独立于 LeWM 小数据验证 README，用来整理当前世界模型领域的发展方向、代表方法，以及不同方向之间的区别。

## 1. 先给结论

世界模型不是一个单一模型结构，而是一组围绕“预测世界如何变化”的方法族。

核心问题可以写成：

```text
当前状态 + 动作 / 条件
        ↓
未来状态 / 未来观测 / 未来 latent / 未来 3D 世界
```

不同方向的差别主要在于：

```text
预测什么：像素、latent、3D/4D、occupancy、reward、文本状态
服务什么：控制、规划、仿真、数据生成、表征学习、Agent 推理
是否闭环：只生成视频，还是能让 agent 在模型里行动
是否需要动作：无条件未来生成，还是 action-conditioned 预测
```

## 2. 主要方向总览

| 方向 | 核心问题 | 代表方法 | 和其他方向的区别 |
|---|---|---|---|
| Model-Based RL / Latent Control | 在 latent 中想象未来，用于强化学习和控制 | Dreamer、TD-MPC2、IRIS、STORM | 最偏控制，通常绑定 reward、value、policy |
| Action-Conditioned Video Prediction | 当前图像和动作预测未来图像 | Visual Foresight、GAIA-1、GameNGen | 最直观，可看到未来帧，但像素预测重且长时易漂移 |
| JEPA / Feature Predictive World Model | 不预测像素，预测未来表征 | V-JEPA、DINO-WM、LeWM | 更轻，更适合规划，但 latent 不如图像直观 |
| Driving World Model | 预测自动驾驶未来场景 | GAIA、DriveDreamer4D、HERMES、UniFuture | 强领域化，强调多视角、BEV、3D/4D、occupancy |
| 3D / 4D / Geometry World Model | 预测空间结构和时间演化 | GaussianWorld、UniScene、DriveDreamer4D | 比 2D 视频更强调几何一致性和物理空间 |
| World Foundation Model | 通用物理世界基础模型 | Sora、Cosmos、Genie | 野心最大，强调大规模视频/多模态/交互仿真 |
| Interactive Game / Neural Simulator | 神经网络生成可交互环境 | Genie、Genie 2、GameNGen | 像“神经游戏引擎”，动作输入后实时生成后续画面 |
| Robotics / Embodied World Model | 机器人感知、行动、接触和规划 | DayDreamer、RoboDreamer、DINO-WM、LeWM | 更关注真实身体、动作、安全和 sim-to-real |
| LLM / Agent World Model | 预测网页、代码、社会系统等状态变化 | WebDreamer、LLM causal WM、social simulacra | 不一定是物理世界，更多是符号/交互环境状态转移 |

## 3. Model-Based RL / Latent Control

代表：

- DreamerV1/V2/V3
- TD-MPC / TD-MPC2
- IRIS
- STORM
- TWM
- DayDreamer

核心形式：

```text
observation -> encoder -> latent state
latent state + action -> next latent state
latent rollout -> reward / value / policy
```

这一派的目标是控制：让 agent 在环境里获得更高 reward。

典型系统会学习：

```text
dynamics model
reward model
value model
policy
```

和 JEPA / LeWM 的区别：

- Model-Based RL 通常需要 reward 或任务信号。
- 它最终往往训练出一个 policy，可以直接输出动作。
- JEPA / LeWM 更像预测未来 feature，再用 CEM/MPC 等 planner 搜动作。

## 4. Action-Conditioned Video Prediction

代表：

- Visual Foresight / Visual MPC
- GAIA-1
- GameNGen
- 部分 driving video generation 方法

核心形式：

```text
过去几帧图像 + 动作序列
        ↓
未来几帧图像
```

如果摄像头固定在小车上，那么动作可以理解为移动摄像头：

```text
车载图像 + 小车线速度/角速度 -> 下一帧车载图像
```

但动作不一定是相机运动。对机器人手臂，动作可能是关节速度；对游戏，动作可能是键鼠输入；对自动驾驶，动作可能是 ego trajectory。

优点：

- 预测结果直观。
- 适合仿真和可视化。
- 对小车、游戏、驾驶都很自然。

缺点：

- 像素预测计算重。
- 未来帧容易模糊。
- 长 horizon 误差累积明显。
- 画面真实不等于可控制或物理正确。

## 5. JEPA / Feature Predictive World Model

代表：

- V-JEPA / V-JEPA 2
- DINO-WM
- LeWorldModel

核心形式：

```text
当前图像 -> encoder -> z_t
动作 -> action encoder
predictor(z_t, action) -> z_hat_t+1
未来图像 -> encoder -> z_t+1
loss = distance(z_hat_t+1, z_t+1)
```

这一派不追求生成清晰未来图像，而是预测未来图像的 latent / feature。

优点：

- 比像素预测轻。
- 更适合目标图像规划。
- 不必重建所有视觉细节。

缺点：

- latent 不直观。
- feature distance 不一定等价于任务成功。
- 仍然依赖训练分布。
- 控制时通常需要 CEM/MPC 等 planner。

LeWM 属于这个方向。它的推理形式是：

```text
当前图像 + 目标图像 + 候选动作序列 -> cost
CEM 搜索低 cost 动作
```

## 6. Driving World Model

代表：

- GAIA-1 / GAIA-2
- DriveDreamer / DriveDreamer4D
- HERMES / HERMES++
- UniFuture
- DrivingGPT
- GaussianWorld
- UniScene
- X-World

核心问题：

```text
自动驾驶车辆看到当前场景后，未来道路、车辆、行人和交通状态会如何演化？
```

常见输入：

```text
多摄像头图像
LiDAR / occupancy
BEV
地图
ego trajectory
文本或驾驶意图
```

常见输出：

```text
未来视频
未来 BEV
3D occupancy
轨迹
场景生成
规划辅助信号
```

和通用视频生成不同，driving world model 必须考虑：

- 多视角一致性。
- 几何一致性。
- ego motion。
- 交通参与者交互。
- 闭环仿真和安全评估。

## 7. 3D / 4D / Geometry World Model

代表：

- DriveDreamer4D
- GaussianWorld
- UniScene
- DIO
- UniOcc

它们不满足于预测 2D 图像，而是想预测：

```text
3D occupancy
4D dynamic scene
point cloud
Gaussian splatting
neural field
BEV flow
```

和视频预测的区别：

- 视频预测关心“看起来像不像”。
- 3D/4D world model 关心“空间结构和时间运动是否真实”。

这条线对自动驾驶和机器人尤其重要，因为控制系统最终要知道：

```text
障碍物在哪里？
物体会怎么动？
我能不能通过？
```

## 8. World Foundation Model

代表：

- Sora
- NVIDIA Cosmos
- Genie / Genie 2
- Runway General World Model 类方向

目标：

```text
用大规模视频、多模态数据和生成模型，构建可泛化的物理世界模拟器。
```

这条线的雄心最大，但风险也最大。

它需要回答：

- 是否真的有物理理解，还是只生成看起来合理的视频？
- 是否支持动作条件控制？
- 是否能闭环交互？
- 是否能用于真实机器人或自动驾驶训练？
- 是否有可靠评测，而不是只看视觉质量？

## 9. Robotics / Embodied World Model

这是和“具身智能”关系最密切的方向。

Embodied AI 可以翻译为“具身智能”。它指的是：

```text
有身体或具身载体的 agent，
能够感知环境、做出动作、影响世界，
并根据交互反馈完成任务。
```

这里的身体可以是真实机器人，也可以是仿真环境中的 agent。

例子：

```text
机器人手臂抓取物体
小车靠近目标
自动驾驶车辆行驶
Minecraft agent 采集方块
Blender 里的仿真机器人推动物体
```

和普通视觉模型的区别：

```text
普通视觉模型：看图 -> 识别
具身智能：看图 -> 决策 -> 行动 -> 世界变化 -> 再观察
```

因此 embodied AI 天然需要 world model，因为它必须预测：

```text
如果我执行这个动作，环境会如何变化？
```

## 10. 四篇综述的阅读笔记

### Understanding World or Predicting Future?

本地 PDF：

```text
resources/papers/understanding_world_or_predicting_future_2411.14499.pdf
```

在线 HTML：

```text
https://arxiv.org/html/2411.14499v4
```

这篇是综合入口。它把世界模型归纳成两个核心功能：

```text
理解当前世界：构建 internal representation
预测未来世界：simulate future states
```

它的价值在于概念地图，而不是某一个技术细节。适合用来回答：

```text
世界模型到底是在理解世界，还是只是在预测未来？
```

### Is Sora a World Simulator?

本地 PDF：

```text
resources/papers/is_sora_a_world_simulator_2405.03520.pdf
```

在线 HTML：

```text
https://arxiv.org/html/2405.03520
```

这篇从 Sora 出发，讨论视频生成模型是否能被看作 general world model。

它更关注：

- text-to-video 技术。
- diffusion / transformer 架构。
- 视频生成模型的物理一致性。
- 视频生成和 autonomous driving / embodied AI 的关系。

它适合看大模型趋势，但不应直接等同于可控机器人 world model。

### A Comprehensive Survey on World Models for Embodied AI

本地 PDF：

```text
resources/papers/comprehensive_survey_world_models_embodied_ai_2510.16732.pdf
```

在线页面：

```text
https://arxiv.org/abs/2510.16732
https://arxiv.org/html/2510.16732
```

这篇专门讨论 embodied AI 中的 world model。它强调 world model 是 embodied agent 的 internal simulator，可以支持 forward rollout 和 counterfactual rollout。

它提出三轴分类：

```text
1. Functionality:
   Decision-Coupled vs General-Purpose

2. Temporal Modeling:
   Sequential Simulation and Inference vs Global Difference Prediction

3. Spatial Representation:
   Global Latent Vector
   Token Feature Sequence
   Spatial Latent Grid
   Decomposed Rendering Representation
```

这篇特别适合理解 LeWM、DINO-WM、Dreamer、Sora、robotics world model 之间的关系。

它强调的挑战：

- 缺少统一数据集。
- 评估指标过度关注像素质量，缺少物理一致性评估。
- 实时控制需要计算效率。
- 长 horizon 预测会误差累积。
- 空间表示会影响 occlusion、object permanence 和 geometry-aware planning。

### A Survey: Learning Embodied Intelligence from Physical Simulators and World Models

本地 PDF：

```text
resources/papers/learning_embodied_intelligence_simulators_world_models_2507.00917.pdf
```

在线页面：

```text
https://arxiv.org/abs/2507.00917
```

这篇把 embodied intelligence 放在两个基础技术之间看：

```text
Physical Simulator: 外部仿真环境
World Model: agent 内部世界模型
```

它的核心区别很清楚：

```text
Simulator:
显式建模真实世界，提供可控、安全、高保真的训练和评估环境。

World Model:
学习 agent 内部对环境的表示，让机器人能预测、规划和适应。
```

这篇适合做机器人/小车/Blender 仿真时阅读，因为它直接讨论：

- locomotion
- manipulation
- human-robot interaction
- physical simulators
- realistic rendering
- autonomous driving
- articulated object manipulation
- sim-to-real

## 11. Physical Simulator 和 World Model 的区别

这是 embodied AI 里很关键的一组概念。

Physical simulator 是外部世界：

```text
MuJoCo
Isaac Sim
PyBullet
Habitat
CARLA
Blender
```

它通常基于显式物理、渲染和碰撞规则。

World model 是 agent 内部模型：

```text
我当前看到什么？
如果我这么做，世界会怎么变？
哪个动作更接近目标？
```

它通常通过数据学习。

两者的关系：

```text
simulator 提供可控环境和训练数据
world model 学习环境动态和预测能力
agent 用 world model 做规划和决策
最终再回到真实世界验证
```

对小车项目来说：

```text
Blender / Isaac / MuJoCo 可以是 simulator
LeWM / DINO-WM / Dreamer 可以是 world model
真实小车是最终部署环境
```

## 12. 对小车/机器人方向的启发

如果目标是“视觉小车接近目标物体”，最现实的研究路线不是直接用 Sora/Cosmos 这种大模型，而是：

```text
1. 固定摄像头和动作空间
2. 收集真实或仿真轨迹
3. 训练 action-conditioned video 或 latent world model
4. 用目标图像或 reward 定义任务
5. 用 MPC/CEM 或 policy 输出动作
6. 加入安全层和低速测试
```

更务实的第一版：

```text
目标检测/分割 + 视觉伺服 + 避障/急停
```

世界模型适合第二阶段：

```text
学习短 horizon 的视觉动力学
处理简单绕行、遮挡、目标变化
在仿真中做更多试错
```

## 13. 推荐阅读顺序

如果只读两篇：

```text
1. Understanding World or Predicting Future?
2. A Comprehensive Survey on World Models for Embodied AI
```

如果关注视频生成和大模型趋势：

```text
1. Is Sora a World Simulator?
2. Understanding World or Predicting Future?
```

如果关注小车、机器人、Blender：

```text
1. A Survey: Learning Embodied Intelligence from Physical Simulators and World Models
2. A Comprehensive Survey on World Models for Embodied AI
3. Understanding World or Predicting Future?
```

## 14. 参考资料

- Understanding World or Predicting Future?: https://arxiv.org/abs/2411.14499
- Understanding World or Predicting Future? HTML: https://arxiv.org/html/2411.14499v4
- Is Sora a World Simulator?: https://arxiv.org/abs/2405.03520
- Is Sora a World Simulator? HTML: https://arxiv.org/html/2405.03520
- A Comprehensive Survey on World Models for Embodied AI: https://arxiv.org/abs/2510.16732
- A Comprehensive Survey on World Models for Embodied AI HTML: https://arxiv.org/html/2510.16732
- Learning Embodied Intelligence from Physical Simulators and World Models: https://arxiv.org/abs/2507.00917
- AwesomeWorldModels: https://github.com/Li-Zn-H/AwesomeWorldModels
- Embodied-World-Models-Survey: https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey
