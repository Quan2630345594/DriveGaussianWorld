<div align="center">

# DriveGaussianWorld

### Geometry Grounded World Modeling for Autonomous Driving

[![Tests](https://github.com/Quan2630345594/DriveGaussianWorld/actions/workflows/tests.yml/badge.svg)](https://github.com/Quan2630345594/DriveGaussianWorld/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-orange)
![CUDA](https://img.shields.io/badge/CUDA-supported-76B900)

**DGGT 几何前端 · 动作条件潜空间动力学 · Gaussian 场景预测**

[技术概览](#技术概览) • [安装](#安装) • [数据准备](#数据准备) • [训练与评估](#训练与评估) • [技术文档](#技术文档) • [引用](#引用)

</div>

DriveGaussianWorld 将预训练 DGGT 的几何重建能力与动作条件时序建模连接起来，面向道路行车序列预测未来图像、局部 Gaussian 场景和自车运动。项目包含数据采集与审计、因果表征缓存、模型训练、基线与消融、可视化报告和推理 API。

## 项目特性

- **原生 DGGT 前端**：冻结上游模型，预测相机参数、深度、Gaussian 属性和动态概率，输出统一的几何状态。
- **因果观测缓存**：每个时刻只重建已观测历史，避免完整视频重建引入未来信息。
- **动作条件世界模型**：融合图像、Gaussian 集合和速度，以递归潜状态建模未来演化。
- **几何运动先验**：结合解析自车运动和学习残差，联合预测场景、观测与自车位移。
- **完整实验接口**：支持 CUDA、GPU 索引、FP16/BF16 训练，提供基线、消融、日志与检查点。

## 技术概览

```mermaid
flowchart LR
    A[历史行车图像] --> B[DGGT 几何前端]
    S[同步双目与标定] --> M[历史尺度锚定]
    B --> M
    M --> G[Gaussian 状态与有效采样 Mask]
    A --> E[图像编码器]
    G --> Q[Gaussian 集合编码器]
    E --> H[历史场景状态]
    Q --> H
    V[历史自车速度] --> H
    H --> D[GRU / RSSM / 条件 Diffusion]
    U[动作条件与时间间隔] --> D
    D --> P[解析运动先验与学习残差]
    P --> I[未来 RGB]
    P --> F[未来 Gaussian 场景]
    P --> T[未来自车运动]
```

### DGGT 几何前端

前端直接使用 [xiaomi-research/dggt](https://github.com/xiaomi-research/dggt) 的 `VGGT` 模型。图像经过 patch embedding 和帧内/跨帧交替注意力，多个预测头联合输出相机位姿编码、深度、Gaussian 外观与几何属性以及动态概率。本项目沿用上游 `use_depth=True` 推理路径，由预测深度和相机内外参反投影得到场景点，而不是直接使用 point head 作为最终几何。

对时刻 `t`，缓存只接收 `[t-C+1, …, t]` 的历史帧；每次在当前预测相机系中提取状态，借助当时的双目深度估计尺度。每帧在 **4×16 网格**上选取 **64 个 Gaussian 槽位**，每个槽位为 16 维向量：位置、RGB、不透明度、三轴尺度、四元数、动态概率与置信度。网格槽位是采样位置，不代表持续追踪的物体身份。

尺度锚定为中位数深度比例，并非完整的相机—车体配准；状态仍保留前端预测误差。相机坐标采用 **右、下、前**，位置与尺度分别按 40 m、5 m 归一化。

### Gaussian 世界模型

世界模型支持 `gru`、`rssm` 和 `diffusion` 三个后端，共用输入契约、运动先验和预测头。以下三部分描述原始 GRU 基线：

1. **观测编码**：CNN 将 `3×32×96` RGB 编码为 64 维向量；共享 MLP 与 masked mean 将 `64×16` Gaussian 集合编码为 48 维向量，再与速度拼接输入历史 GRU。
2. **动作条件递推**：动力学 GRU 每步接收 `[a/5, yaw_rate, dt, speed/20]`，递归更新潜状态。解析平面运动将上一时刻 Gaussian 变换到下一相机局部系，网络预测几何、属性、有效采样 mask 和自车运动残差。
3. **联合解码**：图像残差解码器预测未来 RGB；Gaussian 解码器预测局部场景；自车解码器预测 `[forward, left, delta_yaw]`。开环 rollout 不读取未来观测，也不使用 teacher forcing。

训练目标为：

```text
L = L_rgb + 2 L_geometry + 0.2 L_attributes + 0.05 L_mask + 0.1 L_ego
```

其中几何项使用双向最近邻集合距离，属性项包含固定采样槽属性误差与四元数符号不变项；检查点仅由验证集损失选择。默认历史 3 帧、预测 4 步，潜状态维度与预测长度可配置；当前图像解码分辨率和 Gaussian 槽位数固定。

RSSM 增加连续随机潜状态、先验/后验和 balanced KL；条件 Diffusion 使用潜空间加噪训练与 DDIM 采样。新增后端目前完成框架与小张量测试，尚未进行数据集性能或 GPU 资源比较。架构、损失和实验限制见 [三类世界模型说明](docs/WORLD_MODEL_BACKENDS.md)。

## 技术文档

| 模块 | 文档 | 内容 |
|---|---|---|
| DGGT 前端 | [DGGT 前端技术设计](docs/DGGT_Frontend.docx) | 原生网络结构、深度反投影、坐标转换、因果缓存、尺度锚定与状态契约 |
| 世界模型 | [Gaussian 世界模型技术设计](docs/World_Model.docx) | GRU、RSSM 与条件 Diffusion 结构、共享预测头、训练目标和比较协议 |

## 安装

### 1 创建环境

```bash
git clone --recurse-submodules https://github.com/Quan2630345594/DriveGaussianWorld.git DriveGaussianWorld
cd DriveGaussianWorld
conda create -n drivegaussianworld python=3.10 -y
conda activate drivegaussianworld

# PyTorch 2.4.1 CUDA 12.4 示例；请按驱动选择对应的官方 CUDA wheel。
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e '.[test,dggt]'
```

### 2 准备 DGGT 运行时

```bash
# 非 Git 源码包可使用该脚本获取固定版本的上游源码。
bash scripts/setup_dggt_source.sh
pip install -r third_party/dggt/requirements.txt
mkdir -p checkpoints
curl -L --fail \
  https://huggingface.co/xiaomi-research/dggt/resolve/main/model_latest_waymo.pt \
  -o checkpoints/model_latest_waymo.pth
python -m drive_gaussian_world doctor
```

前端缓存不调用扩散精修或跟踪模块；如单独运行上游完整训练/渲染，请另外遵循其安装说明。依赖冲突时，以 PyTorch、torchvision 和 CUDA 版本匹配为先，避免混装多个 torch 版本。

## 数据准备

当前提供 KITTI raw 适配：同步双目、OXTS、时间戳和标定进入主流程，少量 LiDAR 用于独立深度诊断。示例划分按整条日志隔离：`0005 / 0013 / 0009` 分别用于 train / val / test。

```bash
# 选择公开 ZIP 中所需成员，校验 CRC32 和 SHA256。
python -m drive_gaussian_world collect --root data/kitti --frames 128 --stride 4

# 构建同步序列、运动条件与双目参考几何。
python -m drive_gaussian_world prepare \
  --raw data/kitti/raw --output data/processed/kitti_stereo

# 在 GPU 上构建因果 DGGT 缓存；默认宽度 518，最多 4 帧历史。
python -m drive_gaussian_world dggt-cache \
  --dataset data/processed/kitti_stereo --raw data/kitti/raw \
  --output data/processed/kitti_dggt \
  --device cuda:0 --width 518 --context 4 --precision fp32

python -m drive_gaussian_world validate --dataset data/processed/kitti_dggt
python -m drive_gaussian_world audit \
  --dataset data/processed/kitti_dggt --output runs/data_audit.json
```

输出目录须为新目录，防止覆盖已有实验。初始采集配置是工程示例；扩大数据时需调整日志清单、时间区间及划分，并重新生成对应缓存。数据来源、许可和划分限制见 [数据卡](docs/DATA_CARD.md)。

### 扩展行车数据与代理

`download-drives` 按 `configs/data/kitti_driving.json` 下载更多完整行车日志，保留彩色双目、OXTS、时间戳、标定和 LiDAR。按日志划分 train/val/test，先抽查行驶速度；下载前检查源 ZIP 总量及保守磁盘峰值，均低于 50GB。已校验解压的临时 ZIP 会删除，原始传感器文件保留；支持断点续传和逐文件 CRC32/SHA256。

```bash
# 先查看实际文件大小与预算；只读取元数据和少量 OXTS。
python -m drive_gaussian_world download-drives --preflight --root data/kitti_driving
python -m drive_gaussian_world download-drives --root data/kitti_driving

# 后续准备全部日志的训练窗口；此步骤会计算双目参考几何。
python -m drive_gaussian_world prepare --raw data/kitti_driving/raw \
  --all-collected --stride 4 --output data/processed/kitti_driving_stereo
```

需要代理时，可在本机运行 `ssh -N -R 127.0.0.1:17892:127.0.0.1:7890 <server>`，在服务器显式设置 `DGW_HTTP_PROXY=http://127.0.0.1:17892`。转发只监听服务器回环地址；端口被占用时选择其他空闲端口。默认忽略其他 shell 代理变量，可直接访问官方 S3。`DGW_TRANSFER_WORKERS=16` 可调整每条日志的分块并发数，允许范围为 1 到 32；最多同时处理两条日志。代码和 CI 使用 CUDA 版 PyTorch；在没有 GPU 任务配额时仍可显式选择 `--device cpu` 做小型验证，无需安装 CPU 专用发行版。


## 训练与评估

### 训练世界模型

```bash
python -m drive_gaussian_world train \
  --dataset data/processed/kitti_dggt --output runs/world_model \
  --device cuda:0 --precision fp16 \
  --steps 20000 --batch-size 32 --history 3 --horizon 4 --hidden 96
```

默认设备为 `cuda`；可指定 `cuda:N`，或使用 `--device auto` 自动选择设备。没有可用 CUDA 时，默认路径明确报错；本地调试可显式传入 `--device cpu --precision fp32`。FP16 使用梯度缩放，BF16 需硬件支持；显存占用取决于 DGGT 分辨率、历史长度和训练批大小。

### 切换动力学后端

训练命令增加 `--backend gru|rssm|diffusion`，默认保持 `gru`，旧检查点兼容加载。

```bash
# RSSM：连续随机潜状态与先验/后验训练
python -m drive_gaussian_world train \
  --dataset data/processed/kitti_dggt --output runs/rssm \
  --backend rssm --model-options '{"stochastic":32,"kl_weight":0.1}' --device cuda:0

# Diffusion：潜空间噪声预测训练，20 步 DDIM 采样
python -m drive_gaussian_world train \
  --dataset data/processed/kitti_dggt --output runs/diffusion \
  --backend diffusion --model-options '{"diffusion_steps":100,"sampling_steps":20}' --device cuda:0

# 数据与算力就绪后，使用同一份缓存运行三个后端
python -m drive_gaussian_world experiments \
  --dataset data/processed/kitti_dggt --config configs/comparison.json \
  --output runs/model_comparison
```

单后端配置位于 `configs/world_models/`；评估、Demo 和 API 根据检查点自动选择后端。评估时添加 `--samples 5 --seed 42` 可输出平均样本 ADE、best-of-K ADE 和轨迹离散度。RSSM 当前只实现世界模型，不包含 Dreamer actor/critic；Diffusion 当前使用联合学习的紧凑场景编码器。相同隐藏维度与训练步数不代表相同计算预算，后续比较需补充多种子和显存/延迟测量。

### 评估与消融

```bash
python -m drive_gaussian_world evaluate \
  --dataset data/processed/kitti_dggt --checkpoint runs/world_model/best.pt \
  --device cuda:0 --split test --output runs/world_model/test.json

python -m drive_gaussian_world experiments \
  --dataset data/processed/kitti_dggt \
  --stereo-dataset data/processed/kitti_stereo \
  --config configs/default.json --output runs/ablations
```

实验包括完整模型、RGB-only、移除动作条件及 stereo 表征对照；基线包括末帧保持、解析刚体运动和匀速运动。输出 RGB L1/PSNR、自车 ADE/FDE、Gaussian 集合距离和有效采样 mask IoU。

**条件协议**：当前动作是由 OXTS 差分得到的 `[acceleration, yaw_rate]` 运动代理，不是实测油门/方向盘命令。`learned_logged_motion` 提供未来日志运动作为条件；`learned_current_motion` 不提供未来运动标签。Gaussian 指标主要针对前端伪标签，不能替代独立真实几何评估。已有小样本参考结果见 [实验记录](docs/RESULTS.md)，并非 GPU 大规模训练结果或基准性能承诺。

### 可视化与推理服务

```bash
python -m drive_gaussian_world demo \
  --dataset data/processed/kitti_dggt --checkpoint runs/world_model/best.pt \
  --device cuda:0 --output runs/demo

python -m drive_gaussian_world serve \
  --dataset data/processed/kitti_dggt --checkpoint runs/world_model/best.pt \
  --device cuda:0 --port 8765
```

报告包含历史观测、未来图像、Gaussian 表面、自车轨迹和基线对比。API 默认绑定 `127.0.0.1`：

```bash
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/api/rollout \
  -H 'Content-Type: application/json' \
  -d '{"window":0,"dt":0.4,"actions":[[0,0],[0,0],[0,0],[0,0]]}'
```

动作长度应与检查点的预测 horizon 一致。离线候选动作比较仅用于研究，当前没有真实车辆执行器或闭环碰撞验证。

## 项目结构

```text
DriveGaussianWorld/
├── configs/                 # 训练与回归配置
├── docs/                    # 两份核心技术 DOCX、数据卡与实验记录
├── drive_gaussian_world/
│   ├── data/                # 采集、同步、契约、窗口与来源审计
│   ├── geometry/            # DGGT 适配与双目尺度参考
│   ├── models/              # 多模态编码、递归动力学、解码与损失
│   ├── planning/            # 离线候选动作比较
│   ├── train.py             # 训练、混合精度与检查点
│   ├── evaluate.py          # 条件评估与解析基线
│   ├── demo.py              # 可视化报告
│   └── serve.py             # 推理 API
├── scripts/                 # 上游源码准备与完整流水线
├── tests/                   # 契约、因果性、设备路由与集成测试
└── third_party/dggt/        # 固定版本上游模型
```

## 开发与测试

```bash
make test
make pipeline DEVICE=cuda CONFIG=configs/default.json
```

测试使用小型 fixture 验证数据与数值契约，不启动完整训练。提交改动时请覆盖坐标、时间或设备相关的回归检查，记录数据与模型来源，并保持训练目标和推理输入之间的因果边界。

## 引用

本项目基于 [DGGT 官方实现](https://github.com/xiaomi-research/dggt) 和 [KITTI raw 数据](https://www.cvlibs.net/datasets/kitti/raw_data.php)。DGGT 及 VGGT 派生文件保留上游许可证与 NOTICE；KITTI 数据遵循 CC BY-NC-SA 3.0，使用和分享数据及派生产物时需遵守其条款。

```bibtex
@article{geiger2013vision,
  title={Vision meets Robotics: The KITTI Dataset},
  author={Geiger, Andreas and Lenz, Philip and Stiller, Christoph and Urtasun, Raquel},
  journal={The International Journal of Robotics Research},
  year={2013}
}
```

DGGT 论文与推荐引用格式请参阅上游仓库。
