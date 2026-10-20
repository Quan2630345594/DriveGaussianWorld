# 架构与数据契约

## 因果边界

样本 anchor 为 `t`，历史长度 3，预测长度 4。历史 RGB/Gaussian/speed 进入 `observe()`；未来 motion proxy 与 dt 进入 `rollout()`。训练目标只进入 loss。模型接口拒绝 observation 内出现额外字段。

DGGT 每个时刻单独从最多两个已观测图像重建，缓存 metadata 明确记录 `source_frame_ids` 和 `target_frame_id`。验证器要求全部 source ≤ target；采样窗口中的最早 Gaussian 可能包含更早一个历史帧，所以有效历史感受野最多为 4 个采样时刻。没有未来 RGB、目标位姿或全序列重建隐式进入初始潜状态。

原生适配器使用 `VGGT`、严格 checkpoint 加载与上游 `use_depth=True` 路径：depth + predicted extrinsics/intrinsics → 反投影 → activated Gaussian 属性。直接 point-head 输出不是该权重的正式重建路径。没有修改上游网络或注入 mock 模块。`accelerate.init_empty_weights()` 避免初始化重复权重，保留上游标量/缓冲区初始化所需 CPU 行为。

## Gaussian 表征

每帧在 4×16 图像网格采样 64 个候选 Gaussian。网格是采样槽，不是持久物体身份。invalid 槽有 boolean mask，不能用零填充假装有效几何。

| channels | 含义 | 网络内单位 |
|---|---|---|
| 0:3 | 光学相机局部 xyz，右/下/前 | xyz / 40 m |
| 3:6 | activated RGB | 0..1 |
| 6 | activated opacity | 0..1 |
| 7:10 | 正的三轴 scale | scale / 5 m |
| 10:14 | unit quaternion | wxyz |
| 14 | DGGT dynamic probability | sigmoid，stereo 参考缺此证据 |
| 15 | 几何 confidence | 0..1 |

Stereo 参考用 SGBM、反向视差一致性和局部中位数深度，得到米制可见表面。DGGT 的预测世界坐标通过预测 extrinsics 变为当前相机系，与**当时**的 stereo 深度作中位数尺度锚定；这不是完整 Sim(3) 或车体标定配准。尺度锚点数、残差逐帧保存，独立 LiDAR 仅用于事后诊断。

## 时钟、坐标和动作

KITTI 时间先在 int64 纳秒域减去 epoch，再转浮点秒；左右图像与 OXTS 相对源时钟偏差不得超过 50 ms。帧号必须完整匹配配置中的选择集合，避免未完成的文件传输被误当作有效短日志。

OXTS 经纬度以 KITTI Mercator 近似转局部米制 ENU。相邻位置差旋转到当前 heading 得到 `[forward_m,left_m,delta_yaw_rad]`。状态来源是 IMU/OXTS 参考点；简化 Gaussian 运动先验把车辆平面运动近似施加到相机，尚未补偿杆臂、俯仰与滚转。camera optical pose 不等于严格实车后轴 pose。

条件 `[a, yaw_rate]` 由速度/航向差分计算，所以包含相邻未来运动信息。训练时这是合法条件标签，但部署时不能自动得到：提供日志未来动作的测试必须称为条件预测。无未来动作测试固定为当前速度下的直行条件。

## 动力学与监督

CNN 将 96×32 图像编码为 64 维；共享 MLP + masked mean 将 Gaussian 集合编码为 48 维；速度除以 20 m/s 后拼入 history GRU。action GRU 每步接收 `[a/5,yaw_rate,dt,speed/20]`，产生未来潜状态。

解析平面运动先验从当前速度、加速度、yaw rate 和 dt 产生下一自车位移，并将上一步 Gaussian 变换到下一局部相机系。学习模块预测几何/属性残差、有效采样 mask、自车运动残差和 RGB 残差。未来步骤递归依赖自己的状态，**没有 teacher forcing**。

损失：RGB L1 + 双向最近邻 Gaussian 集合距离（归一化空间）+ 固定采样槽属性 L1/四元数 sign-invariant 项 + mask BCE + ego Huber。几何集合距离避免把移动的采样槽解释为同一物体。动态概率作为状态输入/递归属性保留，目前不对运动轨迹或 actor ID 作独立监督。

基线中 RGB 保持末帧；rigid action-warp 只做解析自车运动；没有偷偷使用未来目标图像。mask IoU 是网格可见表面支持集的比较，不能称为语义占据 IoU。LiDAR 稀疏 patch 深度误差与 Gaussian 伪标签距离分别报告。

## 工程证据

- 数据源 URL、CRC32、SHA256、所选成员及日志运动检查。
- 数据集相对路径、episode SHA256、schema version 与采样配置。
- DGGT 上游 commit、权重 hash、输入分辨率、逐帧历史索引与耗时。
- 训练配置、随机种子、版本、实际步数、梯度范数、val 选择点、checkpoint hash。
- test 日志与窗口数、条件语义、baseline 和 ablation 的数值输出。
- API 仅绑定 loopback，有请求长度/动作范围检查，不加载车辆控制接口。
