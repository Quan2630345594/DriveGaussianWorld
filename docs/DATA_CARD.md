# 数据卡

主数据来自 KITTI raw 官方公开 S3。来源主页：https://www.cvlibs.net/datasets/kitti/raw_data.php ，许可：https://www.cvlibs.net/datasets/kitti/ 。采集器只访问公开对象，不保存账户口令，不使用第三方绕过授权的镜像。

当前 active drive 为 train 0005、val 0013、test 0009，日期 2011_09_26。各选择 32 个采样时刻（原始索引 0..124，步长 4），采集左右 RGB、对应 OXTS、各传感器时间戳及相机/IMU/Velodyne 标定。各日志额外选择 0/4/8 三帧 LiDAR 供独立稀疏深度诊断，LiDAR 不输入世界模型或用作尺度拟合。

完整成员列表在 `data/kitti/collection.json`；使用公共 ZIP 的中央目录与 HTTP Range，不下载整条日志的其他相机或所有点云。每个已下载成员核验 ZIP CRC32 并保存 SHA256。源 SHA256 由下载字节计算，标识精确数据版本，不等于上游数字签名。

候选 0018 初始区间 OXTS mean vf 为约 -0.0012 m/s，因静止被排除。最终三日志 mean vf 分别约 4.17 / 11.42 / 10.85 m/s。日志中可以包含路边停放车辆，但自车任务是道路行车，没有使用泊车终点、倒车轨迹、停车位或泊车数据作为当前任务。

数据按日志隔离，绝不随机按帧切 train/test。三个日志仍可能覆盖接近的真实路线，也都是同一天、同一车辆/相机配置；不能据此声称跨城市、跨天气、跨传感器泛化。每个 split 中时间窗口重叠，窗口不是独立同分布样本；不提供伪精确统计显著性结论。

动作是相邻运动差分代理，不是 issued controls。校准用于 stereo 深度与 LiDAR 诊断；DGGT 使用预测相机参数并作当时 stereo 尺度锚定。Gaussian 监督是前端产生的伪标签，不是真实 4D 场景标注。动态 probability 来自 DGGT，不能替代动态物体真值或跟踪标注。

KITTI 许可为 CC BY-NC-SA 3.0，保留 Geiger 等作者的署名。数据、图像、权重和可能承载数据许可义务的派生产物默认不打包到源码交付。若对外分享完整演示，需要遵守 KITTI 数据许可。引用：Geiger et al., *Vision meets Robotics: The KITTI Dataset*, IJRR 2013。

未来主规模数据优先考虑 nuScenes、Waymo Open Dataset、Argoverse2，原因是 DGGT 官方提供这些数据的处理说明。它们需要各自的账户、使用条款或数据凭证，当前未伪称已经下载。见 `RESOURCE_PLAN.md` 的采集清单。
