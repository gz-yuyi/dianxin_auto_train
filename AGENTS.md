#oss.tuyunai.cn AGENTS.md

## Remote Deployment

- SSH host alias: `telecom-ascend`.
- Deployment directory: `/mnt/nvme0n1/hanbing/volumes/dianxin_auto_train`.
- Compose project name: `dianxin_auto_train`.
- Remote Docker is old; use `/usr/local/bin/docker-compose`, not `docker compose -f`.
- Active compose files:
  - `docker-compose.yml`
  - `docker-compose.npu.yml`
  - `docker-compose.api-gateway.yml`
- Application image:
  - `crpi-lxfoqbwevmx9mc1q.cn-chengdu.personal.cr.aliyuncs.com/yuyi_tech/dianxin_auto_train:npu`
- Main services:
  - `api`: gateway, exposes host port `9011`
  - `api-npu0`, `api-npu6`, `api-npu7`: NPU inference upstreams
  - `worker`: Celery worker
  - `redis`: project Redis, usually do not recreate
- Runtime mounts live under the deployment directory:
  - `artifacts/` -> `/app/artifacts`
  - `data/` -> `/app/data`
  - `models/` -> `/app/models`

## Deployment Notes

- GitHub Actions builds and pushes the `:npu` image on pushes to `main`.
- When updating the server, only operate on the `dianxin_auto_train` compose project and its app services. Do not touch unrelated containers or compose projects.
- A safe app-only update target is: `api api-npu0 api-npu6 api-npu7 worker`.
- Restarting API containers clears in-memory loaded inference models; reload required models before prediction checks.
- The remote Docker daemon may use proxy `127.0.0.1:7890`; if image pulls fail, verify proxy access before changing daemon config.
