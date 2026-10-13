# 外部模型 submodule

GaussianParkWorld 保留 `third_party/dggt` 和 `third_party/splatt3r` 上游源码作为历史依赖。
源码由 `.gitmodules` 固定，模型权重、数据集和 CUDA 环境外置。
当前第一方代码仅提供渲染接口，渲染已有 NPZ 不需要运行这些模型。

~~~powershell
git clone --recurse-submodules https://github.com/Quan2630345594/GaussianPark3DGS.git GaussianParkWorld
~~~

已有检出执行：

~~~powershell
git submodule update --init --recursive
~~~

远端仓库暂时仍使用原名称，本地检出目录使用 `GaussianParkWorld`。
旧 DGGT 重建 CLI 已移除；第三方源码和其许可证未改动。
