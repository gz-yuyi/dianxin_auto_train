## 离线部署指南

本文用于指导运维在无法联网的环境中部署 `dianxin_auto_train`。

### 一、在线机器制作离线包

前置要求：
- 可访问镜像仓库的 Docker CLI
- `bash`、`tar`，以及 `uv`（或系统 Python + 已安装依赖）

操作步骤：
1. 在项目根目录执行：
   ```bash
   ./scripts/build_offline_bundle.sh --output ./offline_bundle
   ```
2. 常用参数：
   - `DX_IMAGE_TAG`：业务镜像标签（默认 `latest`）
   - `APP_IMAGE` / `REDIS_IMAGE`：镜像完整地址，可指向自有镜像仓库
   - `MODEL_NAME` / `MODEL_SOURCE` / `MODEL_DIR_NAME`：指定要预下载的模型及落盘名称（默认从 ModelScope 下载 `google-bert/bert-base-chinese`，落盘目录名 `bert-base-chinese`）
   - `-o/--output`：离线包输出目录，默认 `./offline_bundle_<timestamp>`
3. 脚本会完成：拉取镜像并 `docker save` 到 `images/`，下载模型到 `models/`，复制 `docker-compose.yml`、`.env.offline.example` 和说明文档到 `compose/`，最后生成同名 `.tar.gz` 压缩包。

### 二、离线包结构

```
offline_bundle_<timestamp>/
├── images/               # 保存好的 Docker 镜像
├── models/<model>/       # 预下载模型（默认 bert-base-chinese）
├── compose/
│   ├── docker-compose.yml
│   └── .env.example      # 复制为 .env 后修改
├── data/                 # 放训练数据
├── artifacts/            # 模型输出目录
├── README_OFFLINE.md     # 本文
└── bundle_manifest.txt   # 元数据
```

### 三、离线环境部署

1. 拷贝 `.tar.gz` 至目标机器并解压：
   ```bash
   tar -xzf offline_bundle_<timestamp>.tar.gz
   cd offline_bundle_<timestamp>
   ```
2. 依次导入镜像：
   ```bash
   for image in images/*.tar; do docker load -i "$image"; done
   ```
3. 进入 `compose/`，根据 `.env.example` 生成 `.env`，确保 `DX_IMAGE_TAG` 与镜像一致。
4. 如需训练数据，放入 `data/`；模型已挂载在 `/app/models/<model>`，提交任务时把 `base_model` 指向该路径即可。
5. 启动服务：
   ```bash
   cd compose
   docker compose up -d
   ```
6. 如果需要升级，重新在在线环境生成离线包并替换旧目录与镜像。

### 四、补充说明

- 目标机器只需安装 Docker 即可运行。
- 若需添加新的预训练模型，重新执行脚本并覆盖 `models/` 内容。
- 脚本会保留解压目录与压缩包，方便二次检查或增量更新。
