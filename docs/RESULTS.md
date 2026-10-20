# 实测结果与验收

本文件根据当前服务器上的实际 JSON 产物生成。项目：`/home/tiger/Disk/DriveGaussianWorld`。实验路径：`runs/driving_cpu/`。

## 已执行

- 采集官方 KITTI 行车子集：train 0005、val 0013、test 0009，共 96 个采样时刻、192 张双目 RGB，另有 9 帧真实 LiDAR、OXTS、时间戳与标定。
- 318 个已选原始文件通过 CRC32/字节数审计，RGB/OXTS/LiDAR 等 active 数据约 185 MB；原始选择和哈希见 `data/kitti/collection.json`。
- 原生 DGGT 冻结预训练权重，96 个观测均实际执行 CPU 前向，逐时刻最多 2 个历史采样图像，宽度 140、高度 42。未调用 GPU 推理/训练；上游禁用 autocast 的弃用提示不是 CUDA kernel 执行。
- 四组新增世界模型各完成 200 步 CPU 训练，共 800 次更新，参数量 319,551。每组 train/val/test 各有 26 个重叠时间窗口，历史 3、预测 4。只按 val loss 选择 checkpoint。
- 22 项自动测试通过，覆盖数据契约、未来帧隔离、几何单位、RGB-only 几何旁路隔离、checkpoint/评估/报告与 CPU HTTP API。
- 实际训练模型的 API smoke 已通过：health、4 步输出、每帧 64 Gaussian，越界动作返回 HTTP 400；测试结束后服务已关闭。

DGGT 已记录前向耗时合计 226.72 秒，平均 2.36 秒/观测；四组轻量训练合计 26.87 秒。耗时不包含数据传输、权重哈希、所有导入或安装，不是跨机器性能基准。

## 留出 test 指标

以下学习模型行均为“给定未来日志运动条件”的开环预测；未来条件不等于部署时已知控制。每行是 26 个相关窗口的均值。

| 方法 | RGB L1 ↓ | PSNR dB ↑ | 自车 ADE m ↓ | 自车 FDE m ↓ | Gaussian 伪标签距离 m ↓ |
|---|---:|---:|---:|---:|---:|
| DGGT world | 0.2302 | 10.0486 | 0.2507 | 0.3870 | 3.4907 |
| RGB-only | 0.2278 | 10.0706 | 0.2690 | 0.4221 | 12.4786 |
| No actions | 0.2276 | 10.0935 | 0.2731 | 0.4661 | 3.3182 |
| Stereo world | 0.2284 | 10.0773 | 0.2179 | 0.3526 | 4.7381 |
| 末帧保持 + 日志运动自车先验 | 0.2267 | 10.0868 | 0.1819 | 0.2799 | 1.8789 |
| 解析 Gaussian 刚体运动 | 0.2267 | 10.0868 | 0.1819 | 0.2799 | 6.0799 |

不提供未来日志运动时，主模型采用匀速直行条件：RGB PSNR 10.0488 dB，自车 ADE/FDE 0.2983/0.5016 m。相同条件下的解析匀速基线 ADE/FDE 为 0.2460/0.4285 m，不能拿使用未来日志运动的基线冒充公平的无未来条件对比。

这些结果**没有证明学习模型全面优于简单基线**。主模型的图像保持很强，小样本残差模型未稳定改善；Gaussian 输入移除后几何误差明显恶化，但这个对照同时移除了几何运动先验，不能把全部差异归因于网络学到的几何推理。stereo 与 DGGT 行的 Gaussian 真值来源不同，几何距离不能直接作跨后端精度排名。

DGGT 有效采样 mask 在该设置基本恒为有效，所以对应 mask IoU=1 是退化指标，不能称为遮挡建模、语义占据或可通行区识别能力。

## 独立 LiDAR 诊断

- DGGT：test 前三个 LiDAR 时刻匹配 135 个稀疏图像 patch，depth MAE 4.900 m，AbsRel 0.404。
- Stereo reference：test 前三个 LiDAR 时刻匹配 99 个稀疏图像 patch，depth MAE 2.332 m，AbsRel 0.158。

这是粗稀疏 patch 诊断，不是标准深度 benchmark；不同后端有效采样支持不同，匹配数量不同，不应以这些均值直接排名。DGGT 的低分辨率、跨域、相机预测与中位数尺度锚定仍带来明显误差。当前缓存不能作真实规划碰撞地图。

## 产物位置

- `data/kitti/collection.json`：active 数据来源与哈希。
- `data/processed/kitti_stereo/validation.json`、`data/processed/kitti_dggt/validation.json`：数据契约与采样验证。
- `data/processed/kitti_dggt/*.dggt.json`：实际 checkpoint hash、上游 commit、历史帧与尺度残差。
- `runs/driving_cpu/{dggt_world,rgb_only,no_actions,stereo_world}/`：best.pt、配置、逐步日志、训练摘要与 test.json。
- `runs/driving_cpu/comparison.json`：完整方法/基线/消融结果。
- `runs/driving_cpu/demo/report.html`、`rollout.gif`：自包含报告与动态预览。
- `output/driving_tests_final.log`、`driving_api_smoke.json`、`driving_data_audit_*.json`：验收证据。

完整环境版本见 `environment.txt`。这些是 CPU 工程原型结果；长时高保真世界模型、大规模数据训练与真实闭环驾驶仍属于资源获批后的研究任务，不列为本次已完成能力。
