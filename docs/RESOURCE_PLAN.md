# 资源获批后的扩展计划（当前未执行）

## 数据扩展

| 数据 | 为什么需要 | 当前状态 | 后续执行 |
|---|---|---|---|
| KITTI raw 工程子集 | 真实道路、公开文件、双目与 OXTS，低成本跑通链路 | 已采集并验证 | 扩到更多日期与不相邻道路，保持日志/地理块隔离 |
| nuScenes | 多摄像头、动态道路、标注、官方 DGGT 数据处理说明 | 未采集，需账户及许可 | 先取得 mini，再固定按 scene 的 dev splits；完整训练单独申请空间 |
| Waymo Open Dataset | DGGT 预训练域、多摄像头、丰富道路与动态参与者 | 只有已有模型权重，没有宣称采集数据 | 按官方使用条款取得凭据，选择受限日志清单和硬盘预算 |
| Argoverse2 Sensor | 城市道路、运动目标、多传感器及标注 | 未采集 | 按官方条款下载小型日志清单，建立 log-level adapter |

官方入口：
- DGGT 数据处理说明：https://github.com/xiaomi-research/dggt/tree/main/datasets
- nuScenes：https://www.nuscenes.org/nuscenes
- Waymo：https://waymo.com/open/
- Argoverse：https://www.argoverse.org/av2.html

不要把这些后续选项列为当前已完成的数据集。获取需登录的数据时，由用户按官方条款授权和配置凭据，不在仓库记录令牌。

## 计算与研究路径

1. 先用原生 518 宽度、官方预处理和冻结 DGGT 在批准的单卡上做受限推理。记录显存、吞吐、几何稳定性，不先启动大规模训练。
2. 以多个不重叠日志缓存因果表征；确保每个输入时刻只使用当时历史，不因批量重建引入未来泄漏。缓存 manifest 固定模型、数据、分辨率和索引。
3. 将当前 deterministic latent dynamics 扩为带 posterior/prior 的随机状态空间模型或时序 transformer，预测分布与不确定性。按图像预测、几何预测、自车预测分别消融。
4. 改进几何：独立标注/LiDAR 误差、动态 actor 匹配、遮挡/新显露区域、跨帧实例及多视图 consistency。真实米制需完整相机/IMU/车体标定与杆臂补偿。
5. 控制语义：取得 issued controls 或仿真干预动作后，重训 action adapter 与动力学，验证 action sensitivity 及成对干预反事实。运动差分标签不能直接替代执行器命令。
6. 仿真闭环：在明确道路、车道、交通参与者及碰撞真值的环境中对接规划器，固定路线/seed，比较成功率、碰撞、舒适度和分布外风险。当前 Gaussian 稀疏表面不能直接生成安全占据地图。

任何 GPU 训练需在资源已分配后显式新建训练入口和预算，当前 CPU 脚本不会自动触发。具体显存和耗时由 profiling 决定，不给未经实测的速度或性能保证。
