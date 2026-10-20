# 从 GaussianParkWorld 迁移

工作目录已从 `/home/tiger/Disk/GaussianParkWorld` 改为 `/home/tiger/Disk/DriveGaussianWorld`，Python 包名为 `drive_gaussian_world`，发行包名为 `drive-gaussian-world`。

迁移前存在 README 修改、已暂存 LICENSE 删除及大量未跟踪源文件。未重置这些改动、未恢复用户删除的许可、未执行自动提交或推送。快照位于 `/home/tiger/Disk/GaussianParkWorld_backups/20261019_211454/`：

- `source_before_migration.tar.gz`：原 README、包、脚本、测试、gitignore 和 submodule 配置，含未跟踪源码。
- `git_status.txt`：迁移前工作树状态。
- `unstaged.patch` 与 `staged.patch`：含二进制补丁的原 diff。

原 README、`gaussian_park_world`、scripts 和 tests 完整搬到 `legacy/parking/`。原始泊车数据、已有 output、checkpoints 和第三方 checkout 保留。旧数据不被新 CLI 默认扫描；旧导出的绝对路径引用可能仍指向原目录，应按归档用途人工核对，不声称历史交付已整体重定位验收。

当前新源码和新增数据产物均位于 DriveGaussianWorld 目录。原 Git remote 尚未改名，远端仓库也未重命名；需要对外发布时再由项目所有者处理仓库名称和新增代码许可。新增 source archive 不包含数据、权重、旧泊车资产或 .git。
