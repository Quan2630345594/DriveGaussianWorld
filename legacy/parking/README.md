# GaussianParkWorld

离线场景与专家轨迹工程。主流程为：检查场景及地图证据、生成 issued controls、逐条验证轨迹。
真实 DGGT 场景缺少可信米制尺度、地面和障碍几何时保持 `BLOCKED_GEOMETRY`，不使用 Gaussian alpha 或中心密度生成障碍真值。
本仓库当前交付的是合成地图上验证通过的专家轨迹，不是真实车辆控制日志；不自动训练或启动 GPU 作业。

## 现有工作流程与方法

```text
合成 Fixture ────────────────> 场景契约与来源检查
                                      |
                                      v
                           解析合成地图 + 起终点请求
                                      |
                                      v
                           Hybrid A* 搜索与控制跟踪
                                      |
                                      v
                           issued controls + 状态轨迹
                                      |
                                      v
                           独立 replay / 碰撞 / 终点验收
                                      |
                                      v
                           可选 Episode 渲染与数据适配

已有 DGGT 导出 -> 格式转换 -> 几何证据检查 -> BLOCKED_GEOMETRY
```

| 阶段 | 当前方法 | 输入与产物 |
| --- | --- | --- |
| 场景准备 | 创建确定性合成 Fixture，或转换已有单 gauge DGGT 导出；不执行重建推理 | 每个场景包含 `manifest.json`、`scene.npz`；真实源另有转换报告 |
| 地图与请求 | 使用明确米制、地面和边界的解析矩形障碍地图，未知区域按占据处理 | `--batch` 使用 3 种布局，每图默认 25 个请求，包含故意阻塞的起点 |
| 专家规划 | 有限预算 Hybrid A* 搜索、控制跟踪和恒定控制自行车模型积分 | 导出实际发出的速度/转角命令、时间步长及状态；失败单独记录 |
| 独立验收 | 核查源文件哈希与 gauge、地图、时间轴、动作重放、扫掠车身碰撞和停车终点 | `validate-experts` 读取已有交付，不重新规划或渲染 |
| 可选观测 | 沿轨迹生成 RGB/alpha、位姿及时间信息，按场景划分 train/val/test | Episode 数据及验证摘要；CPU 参考渲染不等于物理传感器仿真 |
| 可选适配 | 中间包转换、合成可见性参考、历史/未来窗口与 DataLoader 检查 | 数据接口验证，不包含官方模型 forward、训练或预测闭环 |

规划状态为后轴 `[x_m, y_m, yaw_rad, signed_speed_m_s]`，动作为
`[signed_speed_m_s, front_steering_rad]`，不是通过相邻位姿差分反推的动作标签。
当前模型轴距为 2.7 m，车身长宽为 4.4/1.8 m，安全裕量为 0.2 m，最大速度为 1 m/s，最大前轮转角为 0.55 rad。
终点位置/航向容差为 0.35 m / 0.12 rad，并要求最终速度为 0。
速度直接取命令值，不模拟加速度、转向速率或执行器延迟，因此通过验收也不代表可直接用于实车控制。

## 快速运行

从仓库根目录使用已有 `parkings` 环境，不安装或下载依赖。CPU 场景和规划接口需要 NumPy/SciPy。
新建产物的输出目录必须尚不存在，父目录必须已存在；已有目录拒绝覆盖，请重复运行时换一个新名称。
以下命令复用本机环境路径；其他机器需将 `P` 替换为已准备好依赖的 Python 路径。

```bash
P=/home/tiger/ProgramFiles/miniconda3/envs/parkings/bin/python
export CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 1. 已有合成场景可直接复用；也可创建新的明确标记 Fixture。
$P -B -m gaussian_park_world.data_pipeline prepare-scenes --output output/example_scenes

# 2. 三种解析合成地图布局，多起终点、前进、倒车和左右转弯。
$P -B -m gaussian_park_world.data_pipeline generate-experts \
  --scenes output/example_scenes --output output/example_experts \
  --batch --budget 6000 --retries 2

# 3. 独立验证来源、地图、状态 replay、扫掠车身碰撞和终点。
$P -B -m gaussian_park_world.data_pipeline validate-experts --experts output/example_experts
```

历史已生成并验收的轨迹位于 `output/expert_trajectory_delivery_20261017_r2/`，以下数量不是每次重跑的成功保证：
75 个请求、65 条成功轨迹、10 次已记录失败，1,894 帧状态、1,829 个 issued controls。
65 条轨迹包含 47 条前进、18 条倒车，其中 30 条有转弯；转弯与前进分类重叠。
地图是 3 个不同几何布局，而不是把旧的 3 个外观 seed 当作 3 种路径环境。
这些解析障碍与场景中的视觉 Gaussian 标记是不同的合成构造，不代表真实传感器/障碍对应关系。

目录内 `experts.json` 是索引，`map*.json` 是地图快照，逐条 NPZ 包含动作、dt、状态、时间戳和校准可用时的车辆/相机位姿。
验证命令可直接检查已有交付，不会重新规划或渲染。
轨迹 `states` 为 `[N,4]`，`actions` 为 `[N-1,2]`，`dt` 为 `[N-1]`。
验收依赖索引中的原场景路径与源文件仍然有效；仅移动专家输出目录并不构成可搬迁的完整归档。
`--budget` 支持 `0..20000`，`--retries` 支持 `0..2`；不带 `--batch` 时每个场景默认只生成布局 0 的 5 个请求。

## 真实场景门槛

已有真实源 left 转换场景：`output/offline_dggt_left_20261016_r2/`，120,000 Gaussian、16 个预测相机位姿。
它仍是未知米制和方向的独立 DGGT gauge，不能用于真实车辆碰撞规划，也不能直接融合四路输出。
真实规划至少需要同 gauge 的米制尺度与配准、地面/朝上方向、可信可通行区和障碍地图、实车 footprint/限制、相机外参及数据使用权限。
预测 optical-to-world 位姿不等于 camera-to-vehicle 标定；源帧索引也不等于同步秒级时钟。

已有场景的格式转换和证据检查入口仍保留：

```bash
$P -B -m gaussian_park_world.data_pipeline prepare-scenes \
  --dggt-source /path/to/reconstruction_single_left --output output/example_dggt
$P -B -m gaussian_park_world.scenes.readiness template \
  --scene output/example_dggt --output output/example_geometry.json
$P -B -m gaussian_park_world.scenes.readiness check \
  --scene output/example_dggt --metadata output/example_geometry.json --output output/example_readiness.json
```

证据完整性检查不是测量真实性认证，也尚未启用真实几何 importer。`USER_ASSERTED` 不会被提升为实测真值。
DGGT 转换输入至少需要 activated Gaussian `scene.npz`、`scene.json` 和含 `extrinsics`/`intrinsics` 的 `predictions.npz`。
`prepare-scenes --dggt-source` 只转换已有导出，不运行 DGGT，也不加载原始帧重新重建。
`readiness check` 当前即使证据完整仍返回 `BLOCKED_GEOMETRY`；合法的阻塞报告可以退出码 0 结束，自动化必须检查报告状态，不能只看退出码。
本次真实门控结果见 `output/expert_real_geometry_gate_20261017_r2/experts.json`，真实轨迹为 0。

## 可选 Episode 与数据适配

主流程专家验收通过后，可显式执行以下 CPU 观测生成。此步骤不是专家交付的必要条件，当前仅用于合成规划场景。

```bash
$P -B -m gaussian_park_world.data_pipeline build-episodes \
  --experts output/example_experts --output output/example_episodes \
  --renderer cpu_reference
$P -B -m gaussian_park_world.data_pipeline validate --dataset output/example_episodes
```

根目录产物为 `dataset.json`、`validation.json`，每个成功 Episode 包含 `frames.npz` 和 `manifest.json`。
划分依据排序后的 `scene_id` 循环分配 train/val/test，不按帧随机切分；失败记录保留在索引中。
`build-episodes` 不代替完整专家验收，因此应先运行 `validate-experts`。

以下为小规模合成数据适配示例，窗口批处理检查另需 PyTorch：

```bash
$P -B -m gaussian_park_world.gwm_adapter.episodes convert \
  --dataset output/example_episodes --output output/example_package --gaussian-count 32
$P -B -m gaussian_park_world.gwm_adapter.episodes validate --package output/example_package
$P -B -m gaussian_park_world.gwm_adapter.parking prepare \
  --package output/example_package --output output/example_observations
$P -B -m gaussian_park_world.gwm_adapter.parking export-windows \
  --observations output/example_observations --output output/example_windows \
  --history 2 --horizon 4
$P -B -m gaussian_park_world.gwm_adapter.parking validate --path output/example_windows
$P -B -m gaussian_park_world.gwm_adapter.parking inspect-batch \
  --windows output/example_windows --batch-size 2 --split all --all-batches
```

合成可见性 oracle 只支持显式合成来源且 Gaussian 数量不超过 32，不是 RGB-to-3D 感知方法。
数据接口限制 rollout 输入，避免暴露未来 RGB、未来位姿或 privileged reference。
官方模型运行时、动作语义和 Gaussian mask 消费方式尚未完成绑定；`official-check` 是预期退出码为 2 的阻塞检查，不是推理入口。

## 工程接口

```text
gaussian_park_world/
  scenes/                 场景契约、格式转换、几何证据门控
  planning/expert.py       现有 Hybrid A*、自行车积分、跟踪和碰撞检查
  planning/delivery.py     专家轨迹批量导出和独立验证
  dataset/offline.py       保留的 Episode 写入、渲染和校验
  rendering/              保留的生产渲染器，不在专家生成流程中启动
  world_model/            仅保留几何、Gaussian 校验和 CPU 参考渲染帮助函数
  gwm_adapter/            保留的既有数据接口，不是默认工程入口或已验证官方模型
  data_pipeline.py        场景 -> 专家轨迹 -> 校验 CLI
scripts/rendering/render.py 离线整图渲染工具
```

上述可选步骤是现有接口的用法，不代表本次文档更新运行了观测生成、GWM 接口或模型。
生产渲染另需匹配的 PyTorch/CUDA/gsplat；视频需 ffmpeg/ffprobe；FOV 图像处理需 OpenCV，归档 SfM 模式需 pycolmap。
生产 `Renderer.render()` 为推理接口，返回 NumPy RGB/Alpha，不支持训练反向传播。

## 工程测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -B -m pytest -q -p no:cacheprovider \
  tests/test_expert_delivery.py tests/test_offline_planning.py tests/test_offline_scenes.py tests/test_offline_dataset.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -B -m pytest -q -p no:cacheprovider \
  tests/test_scene_readiness.py

# 旧几何/参考渲染测试含进程级导入隔离断言，独立执行。
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -B -m pytest -q -p no:cacheprovider \
  tests/test_geometry.py tests/test_gaussian_state_contract.py tests/test_gaussian_observation_reference.py

# 可选数据适配测试，不是官方模型或 GPU 渲染验收。
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -B -m pytest -q -p no:cacheprovider \
  tests/test_gwm_gaussians.py tests/test_gwm_episode_adapter.py \
  tests/test_causal_gaussian_observations.py tests/test_parking_sequence_loader.py
```

场景 Schema 与来源约束以 `gaussian_park_world/scenes/contract.py` 为准，几何证据检查见 `gaussian_park_world/scenes/readiness.py`。
实际交付统计与失败原因可查对应输出目录内的 `experts.json`，本次文档更新不新增交付或测试通过记录。
原始数据、模型权重、场景产物、已有 output、第三方源码和许可证不属于清理目标。
