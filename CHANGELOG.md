# Changelog

本文件记录各版本面向用户的重要变更，遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)
格式和 [Semantic Versioning](https://semver.org/lang/zh-CN/) 版本规范。

未发布的变更维护在 `Unreleased` 下；发布后保留历史记录，不移动或覆盖已发布版本。

## [Unreleased]

## [0.1.0] - 2026-11-11

首个正式发布版本，汇总此前尚未打标签的训练、推理与部署能力。

### Added

- 基于 FastAPI、Celery 和 Redis 的异步文本分类训练服务，支持任务提交、查询、停止、删除及进度回调。
- Excel 训练数据上传、独立验证集、分层划分、逐 batch 进度及逐 epoch 的准确率、损失和 F1 指标。
- LoRA 和全量微调，混合精度、梯度累积、早停与标签锚点样本；支持命令行本地训练和推理。
- 共享基座的多 LoRA 推理服务，支持模型发布、加载、重载、卸载、查询和批量预测。
- 多推理节点网关、轮询/最少积压调度、上游熔断、预测超时、队列容量限制和卡死看门狗。
- CPU、NVIDIA GPU 与昇腾 NPU 运行支持，以及 Docker Compose、离线打包脚本和镜像构建工作流。
- 正式镜像版本标签、应用版本一致性检查，以及版本发布与 Changelog 维护说明。

### Changed

- 默认启用 LoRA，使用优化后的训练参数；训练产物目录统一为英文模型名，去除可选 `.pt` 后缀。
- 卸载模型后默认不调用加速器 `empty_cache`；显式启用时由独立线程处理。
- OpenAPI 显示的应用版本与项目版本统一为 `0.1.0`，替代此前的硬编码 `1.0.0`；接口路径不变。

### Fixed

- 修复 NPU 驱动调用或显存回收挂起引起的推理服务死锁，增加超时取消和容器自愈防护。
- 修复替换磁盘产物后仍复用旧 LoRA adapter 或分类头缓存的问题。
- 修复 NPU Celery 进程池兼容性问题，默认使用单并发 `solo` 池。
- 更正文档中的 API 基础地址：实际使用 `/training/...` 和 `/inference/...`，没有 `/api/v1` 前缀。

[Unreleased]: https://github.com/gz-yuyi/dianxin_auto_train/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gz-yuyi/dianxin_auto_train/releases/tag/v0.1.0
