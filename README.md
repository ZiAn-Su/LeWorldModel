# LeWM 小数据验证与方法理解总结

本文是对 LeWM 论文和仓库的本地小样本验证总结。重点不是完整复现论文指标，而是回答几个核心问题：

- LeWM 到底在做什么？
- 数据集里的任务是什么？
- 它是怎么训练的？
- 推理时怎么给目标、怎么选动作？
- CEM 是模型的一部分吗？
- 它能不能迁移到小车、Blender、Minecraft 这类新场景？

结论先说：**TwoRoom 小样本评估链路已经在本机跑通，但这不等于论文完整复现，也不等于模型可以直接迁移到真实小车或其他环境。**

## 1. 本次实际验证结果

本次只验证了 TwoRoom，因为它是官方数据中较小的一份。没有下载和验证 PushT、Cube、Reacher 的完整数据。

本地环境：

```text
Python: 3.10.5
uv: 0.6.17
PyTorch: 2.7.0+cu128
CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 5090
Transformers: 4.57.6
```

运行结果：

```text
eval.num_eval=5:
success_rate = 100.0%
evaluation_time = 16.86s

eval.num_eval=10:
success_rate = 100.0%
evaluation_time = 25.89s
```

结果文件：

```text
F:\Research\LeWorldModel\.stable-wm\tworoom\tworoom_results.txt
```

生成视频：

```text
F:\Research\LeWorldModel\.stable-wm\tworoom\env_0.mp4 ... env_9.mp4
```

这个结果只能说明：

- 依赖可以导入。
- CUDA 可用。
- TwoRoom 数据可以读取。
- TwoRoom checkpoint 可以加载。
- `eval.py` 的小样本评估链路可以跑通。

不能说明：

- 论文完整指标已复现。
- 其他三个任务也能跑通。
- 模型可以迁移到真实小车、Blender 或 Minecraft。

## 2. 四个任务是什么

论文里涉及的任务可以理解为四类目标条件控制问题。

### TwoRoom

二维导航任务。智能体在两个房间结构中移动，需要到达目标位置。目标通常来自数据集未来某一帧。

本次只验证了这个任务。

### PushT

二维推物任务。智能体需要推动一个 T 形物体，使物体接近目标位置或目标姿态。这个任务考察接触、推动和物体姿态变化。

### Cube

机器人操作任务。机械臂需要操作一个 cube，使 cube 到达目标状态。图像来自仿真器渲染的 RGB 画面。

仓库配置中 Cube 默认使用单视角，默认相机是 `front_pixels`，图像大小是 224x224。

### Reacher

DMControl/MuJoCo 里的机械臂 reaching 或 qpos matching 任务。机械臂需要达到目标关节状态或目标末端状态。

Reacher 的图像来自 DMControl/MuJoCo 渲染，当前 wrapper 对 Reacher 使用默认 `camera_id=0`，图像大小是 224x224。

## 3. TwoRoom 数据里有什么

TwoRoom 数据文件：

```text
F:\Research\LeWorldModel\.stable-wm\datasets\tworoom.h5
```

主要统计：

```text
样本数: 920,809
episode 数: 10,000
pixels shape: (920809, 224, 224, 3)
action shape: (920809, 2)
proprio shape: (920809, 2)
episode length min/mean/max: 31 / 92.0809 / 101
```

关键字段：

```text
pixels              RGB 图像，模型主要视觉输入
action              二维连续动作
proprio             agent 自身状态，TwoRoom 中主要对应位置
pos_agent           agent 位置
pos_target          目标位置
distance_to_target  到目标的距离
terminated          是否正常结束
truncated           是否被截断
```

生成的视频有三个面板：

```text
agent    当前 policy 在环境中的 rollout
dataset  数据集里的参考状态
goal     目标状态图像
```

视频只是评估过程可视化，不是训练视频，也不是模型内部预测出来的视频。

## 4. LeWM 真正能做什么

LeWM 不是直接输出动作的 policy，也不是目标检测器或通用智能体。

它真正做的是：

```text
当前图像 + 目标图像 + 候选动作序列
        ↓
估计这串动作执行后是否更接近目标
        ↓
输出 cost
```

所以它更像：

```text
视觉 world model + latent dynamics + 目标代价评估器
```

它能帮助 planner 判断：

```text
这条动作序列会不会让未来状态更接近目标图像？
```

它不能单独完成：

- 语言理解。
- 任意物体识别。
- 长期任务规划。
- 安全避障。
- 跨场景泛化。
- 真实小车底层电机控制。

## 5. 它是怎么训练的

LeWM 的训练不是传统分类监督，也不是行为克隆。

它不是学习：

```text
当前图像 -> 正确动作
```

而是学习：

```text
当前/历史图像 latent + 动作
        ↓
未来图像 latent
```

训练数据来自离线轨迹：

```text
I0, A0, I1, A1, I2, A2, I3 ...
```

模型先用 encoder 把图像编码成 latent：

```text
I0 -> z0
I1 -> z1
I2 -> z2
I3 -> z3
```

然后 predictor 根据历史 latent 和动作预测未来 latent：

```text
z0, z1, z2 + A0, A1, A2 -> z1_hat, z2_hat, z3_hat
```

loss 是：

```text
MSE(predicted latent, future image latent) + SIGReg regularization
```

所以可以说它“用未来图像做监督”，但更精确地说：

> 未来图像不是以像素形式被预测，而是先经过 encoder 变成 latent embedding，再作为监督目标。

当前仓库配置：

```yaml
history_size: 3
num_preds: 1
frameskip: 5
```

也就是说，一个训练片段大致使用 4 个时间点，并在 latent space 中做短期预测。

## 6. 训练的是 predictor，还是 encoder 也训练

按本地仓库代码看，**不是只训练 predictor，而是 encoder、predictor、action encoder、projector、pred_proj 一起训练。**

训练配置里 encoder 是：

```yaml
encoder:
  _target_: stable_pretraining.backbone.utils.vit_hf
  size: tiny
  patch_size: 14
  image_size: ${img_size}
  pretrained: false
  use_mask_token: false
```

关键是：

```text
pretrained: false
```

也就是说这个配置不是加载一个现成冻结的 ViT，而是使用 ViT tiny 结构参与训练。

从代码看，也没有明显的 frozen target encoder 或 EMA target encoder。`tgt_emb` 没有 detach，所以 target 分支也会影响 encoder。

这也是为什么需要 SIGReg：如果 encoder 和 predictor 一起训练，模型可能把所有图像编码成相似 latent 来降低 MSE，SIGReg 用来抑制这种 embedding 坍塌。

## 7. 推理时是不是给一张目标图像

是的，典型推理任务是给：

```text
当前图像 current image
目标图像 goal image
```

然后让 planner 搜索动作。

在 TwoRoom 评估里，目标图像不是人工手动给的，而是从数据集未来状态中取出来：

```yaml
goal_offset_steps: 25
```

意思是：

```text
从当前轨迹往后取约 25 步的状态作为目标
让 agent 从当前状态出发，尝试接近这个目标状态
```

推理时 LeWM 做的是：

```text
当前图像 + 目标图像 + 候选动作序列 -> cost
```

cost 越低，表示模型认为这串动作更可能让未来状态接近目标图像。

## 8. CEM 是什么，属于模型吗

CEM 是 Cross-Entropy Method，是在线规划器，不是 LeWM 模型本体。

它不在模型权重里，也没有训练出来的参数。`weights.pt` 保存的是 LeWM/JEPA 相关神经网络参数，不包含 CEM。

CEM 的作用是从动作空间里搜索低 cost 的动作序列。

TwoRoom 配置：

```yaml
num_samples: 300
n_steps: 30
topk: 30
```

流程：

```text
1. 随机采样 300 条动作序列
2. LeWM 给每条动作序列打 cost
3. 保留 cost 最低的 30 条
4. 根据这 30 条更新采样分布
5. 重复 30 轮
6. 执行最优动作序列的前几步
7. 重新观察真实环境，再规划
```

所以 CEM 一开始不知道哪些动作好。它只是反复提出候选动作，真正判断好坏的是 LeWM 的 cost。

## 9. “有 A2/A3/A4 干扰，怎么预测 5 步后？”

这里的关键点是：**A2/A3/A4 不是干扰，而是输入条件。**

LeWM 不是只用：

```text
I1 + A1 -> 预测 5 步后
```

而是用完整候选动作序列：

```text
I1 + A1 + A2 + A3 + A4 + A5 -> predicted future latent
```

推理时未来真实动作当然未知，所以 CEM 会假设很多种可能的未来动作序列：

```text
seq1 = A1, A2, A3, A4, A5
seq2 = B1, B2, B3, B4, B5
seq3 = C1, C2, C3, C4, C5
```

LeWM 分别预测这些候选动作序列的后果，再选 cost 最低的。

限制也很明显：

- horizon 太长会误差累积。
- 场景超出训练分布会失效。
- cost 不准时 CEM 会被误导。
- 所以它通常使用短 horizon 和 receding horizon/MPC：执行几步就重新观察真实环境，再重新规划。

## 10. 它能用在小车、Blender、Minecraft 吗

方法上可以迁移，当前 TwoRoom 权重不能直接用。

### 视觉小车

可以作为研究方案，用于：

```text
当前摄像头图像 + 目标图像 -> 搜索动作，让小车接近目标状态
```

但需要重新采集小车数据并训练。真实系统还必须有安全层：

- 限速。
- 急停。
- 避障。
- 人工接管。
- 动作平滑。

如果只是“靠近看到的物体”，工程上更实用的第一版通常是：

```text
目标检测/分割 + 视觉伺服 + 安全控制
```

LeWM 更适合第二阶段做学习型短期规划。

### Blender

Blender 相对更可行，因为可以搭受控仿真环境：

```text
Blender scene
render() -> 224x224 RGB
step(action)
reset()
收集轨迹
训练 LeWM
CEM 推理
```

适合做小车移动、相机移动、机械臂推动等受控任务。

### Minecraft

Minecraft 难很多。它有复杂画面、长时任务、离散键鼠动作、物品栏、合成、开放世界目标。LeWM 可以研究小任务，例如固定地图导航或接近目标方块，但不能直接变成通用 Minecraft 智能体。

## 11. 最核心的限制

对话中反复出现的关键疑问是合理的：LeWM 不是通用智能，不会训练一次到处用。

它依赖：

- 预先采集的离线轨迹。
- 训练分布内的视觉外观。
- 训练分布内的动作效果。
- 较短的预测 horizon。
- 合理的目标图像。
- planner 能在动作空间中搜到可行序列。

因此它更适合作为论文研究方法：

```text
在有限任务分布中，用 latent world model + planner 做目标条件控制
```

而不是直接作为真实世界通用控制系统。

## 12. 本地关键文件

本地仓库代码：

```text
F:\Research\LeWorldModel\resources\le-wm
```

TwoRoom 数据：

```text
F:\Research\LeWorldModel\.stable-wm\datasets\tworoom.h5
```

TwoRoom checkpoint：

```text
F:\Research\LeWorldModel\.stable-wm\checkpoints\tworoom\lewm\weights.pt
F:\Research\LeWorldModel\.stable-wm\checkpoints\tworoom\lewm\config.json
F:\Research\LeWorldModel\.stable-wm\checkpoints\tworoom\lewm\config_hf_original.json
```

评估结果：

```text
F:\Research\LeWorldModel\.stable-wm\tworoom\tworoom_results.txt
```

主要代码：

```text
train.py   训练入口
jepa.py    LeWM/JEPA encode、predict、rollout、get_cost
module.py  predictor、action encoder、SIGReg 等模块
eval.py    小样本评估入口
```

## 13. 一句话总结

LeWM 学的是：

```text
动作如何改变视觉 latent 状态
```

推理时做的是：

```text
给当前图像和目标图像，让 CEM 搜索一串模型认为能接近目标的动作
```

它有研究价值，但不是通用自学习智能体，也不是可以直接部署到新场景的现成控制模型。

## 14. 参考资料

核心资料：

- LeWorldModel 论文 arXiv 页面：[LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312)
- LeWorldModel HTML 论文页面：[arXiv HTML v3](https://arxiv.org/html/2603.19312v3)
- 官方代码仓库：[lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)
- stable-worldmodel 平台仓库：[galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)

Hugging Face 数据和权重：

- Hugging Face paper 页面：[huggingface.co/papers/2603.19312](https://huggingface.co/papers/2603.19312)
- TwoRoom 数据/权重入口：[quentinll/lewm-tworooms](https://huggingface.co/datasets/quentinll/lewm-tworooms)
- PushT 数据/权重入口：[quentinll/lewm-pusht](https://huggingface.co/datasets/quentinll/lewm-pusht)
- Cube 数据/权重入口：[quentinll/lewm-cube](https://huggingface.co/datasets/quentinll/lewm-cube)
- Reacher 数据/权重入口：[quentinll/lewm-reacher](https://huggingface.co/datasets/quentinll/lewm-reacher)

相关背景：

- stable-worldmodel 论文：[stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation](https://arxiv.org/abs/2605.21800)
- DeepMind Control Suite：[dm_control](https://github.com/google-deepmind/dm_control)
- OGBench：[seohongpark/ogbench](https://github.com/seohongpark/ogbench)
- PushT 环境来源之一：[real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)

说明：以上资料用于理解 LeWM 方法、复现实验环境和定位数据来源。本仓库只记录本地 TwoRoom 小样本验证和方法理解，不包含官方完整实验复现。
