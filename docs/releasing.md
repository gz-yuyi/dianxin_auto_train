# 版本管理与发布

## 版本约定

- 使用 [Semantic Versioning](https://semver.org/lang/zh-CN/)：应用版本写为 `MAJOR.MINOR.PATCH`，Git 标签写为 `vMAJOR.MINOR.PATCH`。
- 修复使用 PATCH，新功能使用 MINOR；1.0.0 后的不兼容变更使用 MAJOR。0.x 阶段尚不承诺稳定接口，不兼容变更至少提升 MINOR，并明确写入 Changelog。
- 正式发布使用 annotated tag。已公开的 Git tag、版本镜像标签和历史 Changelog 不移动、不覆盖；修复另发新版本。
- `pyproject.toml`、`uv.lock` 中本项目的版本与 `src/version.py` 必须一致；OpenAPI 使用 `src/version.py` 中的版本。
- `latest`、`npu`、`npu-<SDK>` 是滚动标签。正式交付使用固定版本标签，并额外记录镜像 digest。

镜像标签示例（仓库前缀均为 `crpi-lxfoqbwevmx9mc1q.cn-chengdu.personal.cr.aliyuncs.com/yuyi_tech/dianxin_auto_train`）：

| 场景 | 标签 | 架构 |
| --- | --- | --- |
| NVIDIA GPU / CPU | `v0.1.0-gpu`；兼容别名 `v0.1.0` | `linux/amd64` |
| 昇腾 CANN 8.1 / 910B | `v0.1.0-npu-cann81-910b` | `linux/arm64` |
| 开发追溯（CPU/CUDA 工作流） | `sha-<git-sha>` | `linux/amd64` |

其他 NPU SDK 需单独构建和验证；Release 只能列出确实已成功发布的镜像。镜像构建成功不等于实机验证通过。

## Changelog 维护

`CHANGELOG.md` 遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，而不是直接堆砌 Git 提交记录。

1. 日常变更放入 `## [Unreleased]` 下，描述用户可感知的影响。
2. 按需使用 `Added`、`Changed`、`Deprecated`、`Removed`、`Fixed`、`Security` 分类，不保留空分类。
3. 发布时把本次条目移入 `## [版本] - YYYY-MM-DD`，使用 UTC 发布日期，并继续保留空的 `Unreleased` 区域。
4. 明确写出不兼容变更、默认参数变化、升级操作及兼容性限制。
5. 更新底部链接：`Unreleased` 比较当前版本到 HEAD，后续版本链接比较上一个正式版本到本版本。首个版本链接指向 Release。
6. GitHub Release 的功能变更说明取自同版本 Changelog；镜像地址、digest、验证范围和已知限制可附加在 Release 中。

## 发布检查清单

1. 确认工作区干净、发布分支最新；更新项目版本、锁文件、应用版本、Changelog 和 README 中的当前版本示例。
2. 执行测试：
   ```bash
   RELEASE_TAG=v0.1.0 .venv/bin/python -m unittest discover -s tests -v
   git diff --check
   ```
   镜像内也执行同一套测试，并验证 OpenAPI 的 `info.version`。变更业务逻辑时应增加对应集成/实机验证。
3. 提交发布变更，在同一提交上创建 annotated tag：
   ```bash
   git commit -m "chore(release): prepare v0.1.0"
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin main
   git push origin v0.1.0
   ```
   不在已发布标签上重新构建并覆盖镜像；失败的构建应先排查，发布后的代码修复使用新版本。
4. Git 标签会触发现有 GPU 和默认 NPU 工作流。它们在构建前验证 Git tag、项目版本、应用版本、锁文件和 Changelog 一致。
5. 等待**标签触发的**两套构建成功，检查实际镜像标签、架构、OCI revision/version 和 digest；主分支构建成功不能替代版本标签构建成功。
6. 先创建草稿 GitHub Release，写入 Changelog 内容、已构建镜像的完整地址/digest、验证范围和已知限制。确认所有发布检查通过后再公开 Release，避免把未完成的镜像交给运维。
7. 记录实机验证所用的 digest，不把旧镜像的验证结果宣称为新构建镜像的验证结果。若没有针对某个硬件验证，应明确标记。

目前 GitHub Release 由维护者在构建通过后发布，工作流不会自动发布 Release，也不会自动更新任何服务器。

## v0.1.0 交付注意事项

- API 路径为 `/training/...` 和 `/inference/...`，没有 `/api/v1` 前缀。
- 当前接口没有鉴权；优先使用内网/回环监听，或在前置网关增加鉴权，不直接公开管理接口。
- 训练与推理需要本地基座时，分别配置请求中的 `base_model` 和 `INFERENCE_BASE_MODEL`。
- Redis 使用 `REDIS_HOST`、`REDIS_PORT`、`REDIS_DB_*` 等分项配置；不能仅依赖 Compose 中的完整 URL 环境变量。
- `/inference/status` 的显存统计是当前进程的 PyTorch 视角，并非整卡实际空闲显存；共享卡环境以 `nvidia-smi` / 设备管理工具为准。
- 镜像未包含模型、业务训练数据、宿主机 Docker、GPU/NPU 驱动；完全断网部署还需提前导出应用/Redis 镜像、准备本地模型及匹配的宿主机运行环境。
- `0.1.0` 是首次建立正式版本基线；Changelog 汇总已有能力，不表示所有功能都在发布当天新增。
