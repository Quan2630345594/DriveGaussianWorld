# Gaussian Park

车载停车场图像 -> DGGT 高斯重建 -> 风险栅格 -> Hybrid A* -> 跟踪仿真。
实验使用 ParkingE2E 真实车载泊车记录，并与记录的参考轨迹比较。
前端是原生 HTML/JS，不需要 Node.js。

## 环境

```bash
conda activate parkings
```

已配置 Python 3.10、PyTorch 2.4.1+cu124、NumPy 1.26.4、gsplat 1.5.3、
DGGT 依赖、pointops2、ROS bag 读取和视频工具。
该环境设置 `PYTHONNOUSERSITE=1`、`CUDA_HOME=/usr/local/cuda-12.5`，并清空失效的 `LD_PRELOAD`。
若此前已激活环境，先退出再重新激活。依赖入口是 `requirements-dggt.txt`。

需要重新安装时，先安装配套 CUDA PyTorch，再安装依赖和扩展：

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-dggt.txt
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 python -m pip install \
  --no-build-isolation --no-deps ./third_party/dggt/third_party/pointops2
```

## 数据与权重

- DGGT 主权重：`checkpoints/model_latest_waymo.pth`，官方 SHA-256 已核验。
- [ParkingE2E](https://github.com/qintonguav/ParkingE2E) Demo Bag：`data/raw/parkinge2e/demo_bag.bag`，约 9.02 GB。
- 标定和来源记录：`data/raw/parkinge2e/calibration/`。
- 选中候选 14：32.3 秒，包含停止、倒车转弯、停止，四相机各 222 帧。
- 参考轨迹来自融合定位，不是独立测量的高精度真值；原始数据许可需要遵循作者要求。
- 旧 Hilti 数据仅保留作历史材料，不用于这次泊车轨迹实验。

## 已完成实验

结果目录：`outputs/parkinge2e_20261010_01/`。
本地归档：`outputs/parkinge2e_20261010_01_results.zip`，不重复包含原始 bag 和模型权重。

| 材料 | 路径 |
|---|---|
| 结果报告 | `EXPERIMENT_REPORT.md` |
| 浏览入口 | `index.html` |
| 原始输入与参考轨迹 | `preparation/selected_candidate14/` |
| 四路真实 DGGT 推理 | `reconstruction_single_{front,back,left,right}/` |
| 对齐场景与严格风险地图 | `mapping/` |
| 规划、仿真和比较结果 | `evaluation.json` |
| 主流程图表、失败证据视频 | `analysis/` |
| 无障碍控制诊断视频 | `analysis_control/` |
| 实际高斯渲染视频 | `rendering/gaussian_orbit.mp4` |

**主流程未通过：** 真实重建完成，但严格地图阻塞起终点车身范围，不能生成可执行规划。
无障碍控制诊断得到成功轨迹，但不代表重建地图可用于安全泊车。
完整复现实验命令保存在结果目录的 `reproduce.sh`，输出到新的目录，不覆盖本次结果。

## 网页 Demo

```bash
python -m parking_gs.server --host 127.0.0.1 --port 8080
```

访问 `http://127.0.0.1:8080`。初始场景是合成 Demo，不是本次真实数据的成功结果。
网页上传不读取 `configs/site.json`；真实场地需明确尺度、Z 向上、起终点和已知区域。
车辆位姿使用后轴中心 `[x,y,yaw]`，距离单位米，角度单位弧度。

## 限制

- 已修复原始 NPZ 缺 metadata 的交接问题，以及场景旋转时四元数未同步的问题。
- 普通视频 CLI 仍只取首个窗口；实验脚本另外保存了覆盖完整片段的采样及多路结果。
- 本次为离线重建，使用相机高度和水平地面假设定尺度，定位原点与后轴关系未独立核验。
- 轨迹误差按弧长比较，不是时间 ADE；视频对比是进度归一化，不是真车闭环验证。
