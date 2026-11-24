# Dreamer 风格 RSSM 世界模型

本工程使用连续对角高斯 RSSM，研究给定历史观测及未来自车运动条件的场景预测。它不包含完整 Dreamer 的 actor、critic、奖励或 continuation 预测。

## 观测与状态

图像 CNN 输出 64 维特征，Gaussian token MLP 和 masked mean 输出 48 维特征，拼接归一化速度为 113 维观测。状态包含确定性 h 和随机 z。默认训练 h=256，z=64；代码构造器保留 hidden=96 以支持已有 RSSM 检查点。

历史过滤时，循环单元读取前一 z 和已观测速度；历史控制和 dt 尚未纳入观测契约，因此使用零未知值。未来动力学读取前一 z、加速度、偏航率、dt 和速度。先验 p(z|h) 与后验 q(z|h,o) 均为 MLP，标准差为 softplus(raw)+0.1。训练采用重参数采样，推理仅从先验生成；sample=False 使用分布均值进行诊断。

## 输出与训练

图像头将状态映射为 32×4×12，再通过三层反卷积输出 3×32×96 的残差，每步幅度限为 0.15。Gaussian 头将状态、16 维槽位 embedding 与解析运动后的 16 维几何先验拼接，经过 96 隐藏维的 MLP 输出 16 个属性及 1 个 mask logit。自车头为 hidden→32→3 的 MLP，并叠加解析运动。RGB 与几何独立解码，未使用可微 Gaussian 渲染约束二者一致。

重建损失为 RGB L1 + 2×双向几何距离 + 0.2×属性误差 + 0.05×mask BCE + 0.1×自车 Huber。总损失加 0.1×balanced KL。KL 的动力学项固定后验，表征项固定先验，两者权重为 0.8 和 0.2；潜变量维度求和后应用 free nats=1。

验证使用开环先验预测及公共重建损失，固定随机种子，并仅按验证损失保存最佳模型。测试日志不参与梯度、模型选择或早停。

## 数据与评估边界

KITTI 20 条日志按 train/val/test=16/2/2 划分，共 7,915 帧。缓存只读当前及历史图像。尺度由同一时刻 stereo 锚定；失败的几何明确无效，不使用未来插值。motion-derived acceleration/yaw-rate 是未来自车运动条件，不是真实车辆控制。几何标签来自 DGGT 伪几何，跨日志留出测试不等于跨数据集泛化。

`CUDA_VISIBLE_DEVICES` 必须只包含一张卡，缓存支持逐帧恢复及协议校验。最终保留逐日志 NPZ、来源、几何元数据和完整性报告；临时分片在完成后删除。单卡缓存与训练应串行运行。

全量缓存已完成并校验，详见 [缓存清单](DGGT_CACHE_MANIFEST.json)。原始数据、生成缓存和模型权重均不进入 Git。

## 参考

[PlaNet 原论文](https://proceedings.mlr.press/v97/hafner19a.html)；[Dreamer 原论文](https://arxiv.org/abs/1912.01603)。本项目是面向行车场景预测的 RSSM 适配，未声称完整复现论文系统。
