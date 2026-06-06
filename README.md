# LeWM 小数据验证总结报告

## 1. 验证目标与结论摘要

本次验证目标不是完整复现论文指标，而是进行“小样本实用验证”：确认 LeWM 仓库中的依赖、模型权重、数据读取、环境评估和规划链路可以在本机跑通。

结论：TwoRoom 小数据链路可运行。已完成 zero-data smoke test、TwoRoom 数据读取、checkpoint 加载到 GPU，以及 `eval.num_eval=5` 和 `eval.num_eval=10` 两组小样本评估。两次评估均成功生成结果文件和视频。

需要明确：这不是论文完整指标复现。本文只说明本地 TwoRoom 小样本链路可运行，不能代表 PushT、Cube、Reacher 的完整结果，也不能代表论文表格中的正式复现实验。

## 2. 本次验证范围

本次只验证官方数据中较小的 TwoRoom 任务。

执行内容包括：

- 创建 Python 3.10 环境。
- 安装 `stable-worldmodel[train,env]` 及训练/环境相关依赖。
- 配置 CUDA 版 PyTorch。
- 下载 TwoRoom 模型权重和 TwoRoom 数据集。
- 读取 TwoRoom HDF5 数据。
- 加载 LeWM checkpoint 到 GPU。
- 使用 CEM planner 运行小样本环境评估。

未执行内容包括：

- 未下载 PushT、Cube、Reacher 数据集。
- 未训练模型。
- 未复现论文全部表格指标。
- 未修改仓库源码。

## 3. 论文、仓库与本地资源说明

在线论文地址：

- https://arxiv.org/html/2603.19312v3

本地仓库代码位置：

- `F:\Research\LeWorldModel\resources\le-wm`

当前代码提交：

- `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`

本次实验数据、权重、结果统一放在：

- `F:\Research\LeWorldModel\.stable-wm`

## 4. 环境创建与依赖状态

环境位于：

- `F:\Research\LeWorldModel\resources\le-wm\.venv`

环境是用 `uv venv` 创建的。`.venv\pyvenv.cfg` 中记录：

```text
home = D:\Python3.10.5
implementation = CPython
uv = 0.6.17
version_info = 3.10.5
include-system-site-packages = false
```

关键版本：

```text
Python: 3.10.5
uv: 0.6.17
PyTorch: 2.7.0+cu128
CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 5090
Transformers: 4.57.6
stable_pretraining: 0.1.7
stable_worldmodel: unknown version metadata
```

CUDA 检查结果：

```text
torch.cuda.is_available() = True
```

说明：

- 环境确实由 uv 创建和安装依赖。
- 当前没有 `pyproject.toml`、`requirements.txt` 或 `uv.lock` 记录完整依赖锁定。
- 因此 `.venv` 本身可继续使用，但从零复刻环境时仍需要重新执行安装步骤；严格可复现性不如带 lock 文件的 uv 项目。

## 5. 下载内容与文件目录解释

`.stable-wm` 当前主要目录如下：

```text
F:\Research\LeWorldModel\.stable-wm
├── checkpoints
├── datasets
├── hf_tworooms_data
├── hf_tworooms_model
├── tworoom
└── tworoom.h5
```

### hf_tworooms_model

路径：

- `F:\Research\LeWorldModel\.stable-wm\hf_tworooms_model`

内容：

```text
config.json
weights.pt
```

这是从 Hugging Face 下载的 TwoRoom 模型原始暂存目录。它保留了原始模型配置和权重。

该目录中没有 `config_hf_original.json`，因为 `config_hf_original.json` 是后来在 checkpoint 运行目录中保存的备份副本，不是 Hugging Face 原始下载文件名。

### hf_tworooms_data

路径：

- `F:\Research\LeWorldModel\.stable-wm\hf_tworooms_data`

内容：

```text
tworoom.tar.zst
```

这是从 Hugging Face 下载的 TwoRoom 数据集压缩包，大小约 3.4 GB。

### datasets 与 tworoom.h5

解压后的 HDF5 数据位于：

- `F:\Research\LeWorldModel\.stable-wm\tworoom.h5`
- `F:\Research\LeWorldModel\.stable-wm\datasets\tworoom.h5`

`datasets\tworoom.h5` 是为了适配当前 `stable_worldmodel` 默认查找路径而放置的入口。当前检查结果表明它和根目录下的 `tworoom.h5` 指向同一份数据内容，不应理解为两份不同数据集。

### checkpoints

路径：

- `F:\Research\LeWorldModel\.stable-wm\checkpoints\tworoom\lewm`

内容：

```text
config.json
config_hf_original.json
weights.pt
```

其中：

- `weights.pt` 是模型权重。
- `config_hf_original.json` 是 Hugging Face 原始配置备份。
- `config.json` 是为本地仓库运行而调整后的配置，目标类改为本地 `jepa.JEPA`、`module.ARPredictor`、`module.Embedder`、`module.MLP` 等。

调整配置的原因是：HF 原始配置目标类指向 `stable_worldmodel.wm.lewm.LeWM` 及包内模块，但本地仓库代码和已安装包的结构、Transformers 权重命名存在不完全匹配。为了加载已下载权重并使用本地仓库评估脚本，需要在 checkpoint 目录中保留一份可运行配置。

## 6. TwoRoom 数据集内容解读

TwoRoom 是二维导航任务。智能体需要在两个房间结构中从当前位置移动到目标位置，通常需要穿过连接两个房间的通道。

数据文件：

- `F:\Research\LeWorldModel\.stable-wm\datasets\tworoom.h5`

HDF5 keys：

```text
action
distance_to_target
ep_idx
ep_len
ep_offset
id
observation
pixels
pos_agent
pos_target
proprio
render_time
reward
step_idx
terminated
truncated
```

主要统计：

```text
样本数: 920,809
episode 数: 10,000
pixels shape: (920809, 224, 224, 3), uint8
action shape: (920809, 2), float32
proprio shape: (920809, 2), float32
episode length min/mean/max: 31 / 92.0809 / 101
terminated count: 4035
truncated count: 6056
```

字段含义：

- `pixels`：224x224 RGB 图像，是模型主要视觉输入。
- `action`：二维连续动作。
- `proprio` / `pos_agent`：智能体自身位置或本体状态。
- `pos_target`：目标位置。
- `distance_to_target`：当前位置到目标的距离。
- `ep_idx`、`step_idx`、`ep_len`、`ep_offset`：轨迹编号和步数索引。
- `terminated`、`truncated`：episode 结束标记。

## 7. 模型权重与配置文件说明

当前评估使用的 checkpoint 目录：

- `F:\Research\LeWorldModel\.stable-wm\checkpoints\tworoom\lewm`

权重文件：

- `weights.pt`，约 72 MB。

兼容对象 checkpoint：

- `F:\Research\LeWorldModel\.stable-wm\tworoom\lewm_object.ckpt`

说明：

- 当前 `eval.py policy=tworoom/lewm` 实际使用的是 `checkpoints\tworoom\lewm\config.json` 和 `weights.pt`。
- `lewm_object.ckpt` 是额外保存的兼容 checkpoint，不是当前评估命令的主要入口。
- CEM planner 不在 `weights.pt` 中；权重只保存 LeWM/JEPA 相关神经网络参数。

## 8. Smoke Test 验证过程

Smoke test 覆盖以下内容：

- Python 环境可用。
- CUDA 可用。
- `stable_worldmodel` 可导入。
- `stable_pretraining` 可导入。
- `torch`、`hydra`、`lightning`、`torchvision`、`huggingface_hub` 等关键依赖可导入。
- TwoRoom HDF5 数据可读取。
- checkpoint 可加载并移动到 GPU。

通过标准均已满足：

```text
torch.cuda.is_available() = True
GPU = NVIDIA GeForce RTX 5090
TwoRoom rows = 920,809
checkpoint load = success
```

## 9. 小样本评估程序说明

评估入口：

- `F:\Research\LeWorldModel\resources\le-wm\eval.py`

使用配置：

- `F:\Research\LeWorldModel\resources\le-wm\config\eval\tworoom.yaml`
- `F:\Research\LeWorldModel\resources\le-wm\config\eval\solver\cem.yaml`

核心命令：

```powershell
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm eval.num_eval=5
```

可选扩大样本命令：

```powershell
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm eval.num_eval=10
```

评估配置重点：

```yaml
world:
  env_name: swm/TwoRoom-v1
  num_envs: ${eval.num_eval}
  max_episode_steps: 100

plan_config:
  horizon: 5
  receding_horizon: 5
  action_block: 5

eval:
  goal_offset_steps: 25
  eval_budget: 50
  img_size: 224
```

含义：

- 从数据集中取起点和目标。
- 设置 TwoRoom 环境状态和目标状态。
- LeWM 根据当前图像、目标图像和候选动作序列估计 cost。
- CEM 在候选动作序列中搜索较优动作。
- policy 执行动作并记录是否到达目标。

## 10. 评估结果与指标解读

结果文件：

- `F:\Research\LeWorldModel\.stable-wm\tworoom\tworoom_results.txt`

`eval.num_eval=5`：

```text
success_rate: 100.0
episode_successes: [True, True, True, True, True]
evaluation_time: 16.861632585525513 seconds
```

`eval.num_eval=10`：

```text
success_rate: 100.0
episode_successes: [True, True, True, True, True, True, True, True, True, True]
evaluation_time: 25.89249610900879 seconds
```

解读：

- 两次小样本评估均成功完成。
- 结果说明 TwoRoom 的模型加载、环境交互、目标设置、CEM 规划和结果写出链路可运行。
- 样本数很小，不能用来代表论文正式成功率。

## 11. 生成视频说明

视频输出目录：

- `F:\Research\LeWorldModel\.stable-wm\tworoom`

生成文件：

```text
env_0.mp4
env_1.mp4
env_2.mp4
env_3.mp4
env_4.mp4
env_5.mp4
env_6.mp4
env_7.mp4
env_8.mp4
env_9.mp4
```

还提取过一张首帧图：

- `F:\Research\LeWorldModel\.stable-wm\tworoom\env_0_frame0.png`

视频画面由三个面板组成：

- `agent`：当前 policy 在环境中的 rollout。
- `dataset`：数据集中的参考轨迹状态。
- `goal`：目标状态图像。

视频的意义是辅助检查评估过程是否合理，例如智能体是否朝目标移动、目标图像是否正确、环境渲染是否异常。它不是训练视频，也不是模型内部预测视频。

## 12. 四个任务说明：TwoRoom、PushT、Cube、Reacher

### TwoRoom

二维导航任务。智能体在两个房间结构中移动，需要到达指定目标位置。输入包含图像、动作和位置状态。当前验证只使用了这个任务。

### PushT

二维推物任务。智能体通常需要推动一个 T 形物体到目标位置或目标姿态。该任务验证模型是否能理解接触、推动和物体位姿变化。

### Cube

机器人操作任务。机械臂需要移动或操作一个 cube，使其达到目标状态。图像输入来自仿真器渲染的 RGB 画面。当前配置中 Cube 使用单视角，默认渲染相机为 `front_pixels`，图像大小为 224x224。

### Reacher

DMControl 中的机械臂 reaching / qpos matching 任务。机械臂需要达到目标关节状态或目标末端状态。图像输入来自 DMControl/MuJoCo 的默认渲染相机，当前 wrapper 对 Reacher 使用 `camera_id=0`，图像大小为 224x224。

## 13. LeWM 模型能力与 CEM 规划器关系

LeWM 不是一个直接输出动作的 policy。它更准确地说是 world model / latent dynamics model。

它能做的事情：

- 编码当前图像和目标图像。
- 根据候选动作序列预测 latent 变化。
- 对动作序列计算到目标的 cost。
- 为 planner 提供动作选择依据。

CEM 的角色：

- CEM 是 Cross-Entropy Method，是在线规划器。
- 它采样很多候选动作序列。
- 调用 LeWM 的 cost 估计。
- 保留较优候选并迭代更新采样分布。
- 最终输出当前要执行的动作。

CEM 不属于模型权重。它没有训练得到的参数，不保存在 `weights.pt` 中。若只做表征学习或预测分析，不一定需要 CEM；若要用 LeWM 控制环境，则需要 CEM、Adam solver 或其他 planner 来把 cost 转换成动作。

### 13.1 训练过程：用未来图像的 latent 作为监督

LeWM 的训练不是传统的“图像分类标签监督”，也不是直接学习：

```text
当前图像 -> 正确动作
```

它使用离线轨迹中的时间关系进行训练。每条轨迹包含连续图像和动作：

```text
I0, A0, I1, A1, I2, A2, I3 ...
```

模型先用 ViT encoder 把每一帧图像编码成 latent embedding：

```text
I0 -> z0
I1 -> z1
I2 -> z2
I3 -> z3
```

然后 predictor 根据历史 latent 和对应动作预测未来 latent：

```text
z0, z1, z2 + A0, A1, A2 -> z1_hat, z2_hat, z3_hat
```

训练目标是让预测 latent 接近真实未来图像编码出来的 latent：

```text
loss = MSE(predicted latent, future image latent) + sigreg regularization
```

对应代码逻辑在 `train.py` 中：

```python
emb = output["emb"]
act_emb = output["act_emb"]

ctx_emb = emb[:, :ctx_len]
ctx_act = act_emb[:, :ctx_len]

tgt_emb = emb[:, n_preds:]
pred_emb = self.model.predict(ctx_emb, ctx_act)

pred_loss = (pred_emb - tgt_emb).pow(2).mean()
```

当前训练配置中：

```yaml
history_size: 3
num_preds: 1
frameskip: 5
```

因此一个训练片段大致使用 4 个时间点。模型不是直接预测清晰未来图像像素，而是在 latent space 中预测未来状态。未来图像提供监督信号，但监督对象是未来图像的 embedding。

### 13.2 推理过程：给当前图像和目标图像，让 planner 搜动作

推理时，输入通常包括：

```text
当前图像 current image
目标图像 goal image
候选动作序列 action candidates
```

LeWM 本身不直接输出动作。它输出的是每条候选动作序列的 cost：

```text
current image + goal image + action sequence -> cost
```

cost 越低，表示模型认为这串动作执行后越接近目标图像对应的 latent 状态。

在 TwoRoom 评估中，目标图像不是人工手动指定的，而是从数据集未来状态中取出。配置中：

```yaml
goal_offset_steps: 25
```

含义是：从轨迹中取一个未来状态作为目标，让 agent 从当前状态出发，尝试接近这个未来目标状态。

### 13.3 CEM 动作序列如何产生

CEM 一开始并不知道哪些动作会接近目标。它从动作空间中随机采样很多候选动作序列。

以 TwoRoom 配置为例：

```yaml
num_samples: 300
n_steps: 30
topk: 30
```

含义是：

```text
每轮采样 300 条动作序列
用 LeWM 给每条动作序列打分
保留 cost 最低的 30 条
根据这 30 条更新采样分布
重复 30 轮
```

流程可以写成：

```text
1. CEM 随机采样动作序列
2. LeWM rollout 每条动作序列的未来 latent
3. LeWM 比较未来 latent 和 goal latent
4. CEM 保留低 cost 的动作序列
5. CEM 更新动作分布并再次采样
6. 最后执行最优序列的前几步
7. 重新观察真实环境，再重复规划
```

因此，A2、A3、A4 这类后续动作不是预测中的“干扰”，而是候选动作序列的一部分。LeWM 不是只用 `I1 + A1` 预测 5 步之后，而是用：

```text
I1 + A1 + A2 + A3 + A4 + A5 -> predicted future latent
```

真正的限制在于：如果预测 horizon 太长、环境超出训练分布、目标图像不合理或模型 cost 不准，CEM 会被错误 cost 引导，规划就会失败。

## 14. 复现结论、限制与风险

当前可以确认：

- 本机 CUDA 环境可用。
- TwoRoom 数据可以读取。
- TwoRoom LeWM checkpoint 可以加载。
- `eval.py` 可以完成小样本评估。
- 结果文件和视频可以生成。

当前不能确认：

- 不能确认论文完整表格指标可复现。
- 不能确认 PushT、Cube、Reacher 链路可运行。
- 不能确认训练脚本可从零复现模型权重。
- 不能确认不同随机种子、大规模评估下结果稳定。

主要风险：

- 当前没有 uv lock 文件，环境可复制性依赖手工记录。
- HF 原始配置与本地运行配置存在差异。
- 小样本 success rate 偏高不代表正式评估结果。
- 当前只覆盖 TwoRoom，一个任务不能外推到所有任务。

## 15. 后续可复现实验建议

建议下一步按优先级推进：

1. 固化环境：生成 `requirements.txt` 或 `uv.lock`，记录 CUDA wheel 来源。
2. 扩大 TwoRoom 评估：运行 `eval.num_eval=50` 或更接近官方设置的样本数。
3. 验证 PushT：下载对应较小权重和数据，重复 smoke test 与小样本 eval。
4. 验证 Cube / Reacher：重点检查 MuJoCo、DMControl、OGBench 环境渲染和相机输入。
5. 记录每次评估的运行时间、显存占用、结果文件和视频。

## Appendix A. 关键命令记录

创建环境：

```powershell
uv venv --python D:\Python3.10.5\python.exe .venv
```

运行 TwoRoom 小样本评估：

```powershell
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm eval.num_eval=5
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm eval.num_eval=10
```

检查 CUDA：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Appendix B. 关键文件清单

仓库代码：

```text
F:\Research\LeWorldModel\resources\le-wm
```

Python 环境：

```text
F:\Research\LeWorldModel\resources\le-wm\.venv
```

TwoRoom 数据：

```text
F:\Research\LeWorldModel\.stable-wm\datasets\tworoom.h5
F:\Research\LeWorldModel\.stable-wm\tworoom.h5
```

HF 原始模型下载：

```text
F:\Research\LeWorldModel\.stable-wm\hf_tworooms_model\config.json
F:\Research\LeWorldModel\.stable-wm\hf_tworooms_model\weights.pt
```

HF 原始数据下载：

```text
F:\Research\LeWorldModel\.stable-wm\hf_tworooms_data\tworoom.tar.zst
```

当前评估 checkpoint：

```text
F:\Research\LeWorldModel\.stable-wm\checkpoints\tworoom\lewm\config.json
F:\Research\LeWorldModel\.stable-wm\checkpoints\tworoom\lewm\config_hf_original.json
F:\Research\LeWorldModel\.stable-wm\checkpoints\tworoom\lewm\weights.pt
```

结果与视频：

```text
F:\Research\LeWorldModel\.stable-wm\tworoom\tworoom_results.txt
F:\Research\LeWorldModel\.stable-wm\tworoom\env_0.mp4
F:\Research\LeWorldModel\.stable-wm\tworoom\env_1.mp4
F:\Research\LeWorldModel\.stable-wm\tworoom\env_2.mp4
F:\Research\LeWorldModel\.stable-wm\tworoom\env_3.mp4
F:\Research\LeWorldModel\.stable-wm\tworoom\env_4.mp4
F:\Research\LeWorldModel\.stable-wm\tworoom\env_5.mp4
F:\Research\LeWorldModel\.stable-wm\tworoom\env_6.mp4
F:\Research\LeWorldModel\.stable-wm\tworoom\env_7.mp4
F:\Research\LeWorldModel\.stable-wm\tworoom\env_8.mp4
F:\Research\LeWorldModel\.stable-wm\tworoom\env_9.mp4
```

## Appendix C. 环境与版本信息

```text
Python: 3.10.5
uv: 0.6.17
PyTorch: 2.7.0+cu128
CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 5090
Transformers: 4.57.6
stable_pretraining: 0.1.7
stable_worldmodel: installed, version metadata unknown
```

当前报告生成时间：

```text
2026-06-06
```
