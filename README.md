# GaussianParkWorld

项目转向世界模型训练，目前仅保留第一方高斯渲染接口及必要辅助代码。
停车建图、规划、控制、仿真、诊断、数据准备和预训练重建入口已移除。
现有数据、权重、历史实验输出和第三方源码保留，不属于当前第一方执行流程。
Python 包名为 `gaussian_park_world`，不再包含停车业务。

## 保留接口

```text
gaussian_park_world/
  common/experiments.py        渲染实验目录
  rendering/
    renderer.py               整图渲染、相机加载/插值、视频编码
    preprocessing.py          离线指标所需的参考图预处理
scripts/rendering/render.py    离线整图渲染入口
```

### 整图渲染

```python
import numpy as np
from PIL import Image
from gaussian_park_world.rendering.renderer import Renderer

renderer = Renderer("scene.npz", width=640, height=360)
pose = np.eye(4)  # Optical camera-to-world transform, in the scene's gauge.
K = np.array([[500., 0., 320.], [0., 500., 180.], [0., 0., 1.]])
rgb, alpha = renderer.render(pose, K)
Image.fromarray(rgb).save("render.png")
```

输出为完整尺寸的 RGB `uint8[H,W,3]` 和 alpha `[H,W]`，不裁剪图像、不提供点云替代渲染。
相机内参为像素单位，位姿与高斯必须使用一致的坐标系。
NPZ 包含浮点数组 `means[N,3]`、正尺度 `scales[N,3]`、非零 `quats[N,4]`（wxyz）、
`colors[N,3]`（RGB，范围 0–1）、`opacities[N]` 或 `[N,1]`（范围 0–1）。

时间透明度通过 `Renderer(..., frame_times=..., temporal_opacity=True)` 启用，
需要额外的 `source_frames[N]`、正值 `temporal_widths[N]`，调用时传入源帧坐标 `timestamp`。
归档 SH 场景 checkpoint 的加载和渲染接口也保留。

**当前 `Renderer.render()` 使用推理模式并返回 NumPy，不支持训练反向传播。**
尚未新增世界模型训练器或可微训练接口。

### 离线整图渲染

从项目根目录运行，对已有高斯和相机产物生成整图 PNG 及视频：

```bash
python scripts/rendering/render.py \
  --reconstruction-root /path/to/four_view_scenes \
  --experiment world_model_render_check \
  --temporal-opacity --input-metrics --seconds 3 --fps 30 --samples 8
```

输入根目录包含 `reconstruction_single_{front,back,left,right}/`，每路包含：

- `scene.npz`：高斯参数及可选时间字段。
- `predictions.npz`：`extrinsics`、`intrinsics`、`depth`、`input_paths`。
- `selected_indices.json`：共同的递增源帧索引。
- 默认输入指标还需要 `scene.json`（记录 `horizontal_fov_deg`）、`input_intrinsics.json` 和对应原始输入图像。

`--no-input-metrics` 可仅生成媒体；缺少时间字段时显式使用 `--no-temporal-opacity`。
四路独立模型坐标不直接融合。视频分辨率由预测深度尺寸确定，四路必须相同。
输出六段视频（四路、2x2 拼接、前视新视角）及整图 PNG，记录在 `output/<experiment>/`。
输入一致性指标包括 PSNR、SSIM 和 alpha 覆盖率，不等于留出测试结果。
已有非空渲染目录拒绝覆盖；不指定实验名称时自动创建时间戳目录并归档输入。
新实验不再创建停车目录；`reconstruction/` 仅用于兼容已有输入布局和存放渲染输入副本。

归档场景 checkpoint 模式保留，必须显式指定 checkpoint 和完整 SfM 目录：

```bash
python scripts/rendering/render.py \
  --checkpoint /path/to/scene_checkpoint.pt --sfm /path/to/sfm \
  --experiment archived_render_check --seconds 3 --samples 8
```

## 运行依赖

渲染需要 CUDA、匹配的 PyTorch 和 gsplat，以及 NumPy、SciPy、Pillow。
离线视频需要系统 `ffmpeg`（libx264）和 `ffprobe`。
带 FOV 调整的参考图指标需要 OpenCV；归档 SfM 模式额外需要 pycolmap。
渲染已有 NPZ 不需要 DGGT 网络权重、ROS 或停车模块。

可使用现有 `parkings` 环境，按机器环境配置 CUDA：

```bash
conda activate parkings
export PYTHONNOUSERSITE=1
python scripts/rendering/render.py --help
```

`data/`、`checkpoints/`、`output/` 和 `third_party/` 均未删除；其上游许可及来源记录保持原样。
