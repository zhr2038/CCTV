# NAS 无人值守家庭成长档案 V3 (`cctv-growth-v3`)

部署目录：`/volume1/docker/cctv-growth-v3`  
归档与展示根目录：`/volume2/Growth_Archive_V3`  
局域网控制台与家庭相册：`http://<NAS_LAN_IP>:18795/`

> 部署前请先在 `nas-growth-v3/` 目录复制模板：`cp config.example.toml config.toml && cp compose.example.yaml compose.yaml`，并在 `secrets/` 下配置对应 API Key（权限 `0600`）；如需通过本地部署脚本同步，可在仓库根目录复制 `.env.nas.example` 为 `.env.nas` 并填入 NAS 主机与账号信息。

## 核心策略与安全边界
1. **源录像 100% 只读与四重防误删保护锁 (`lifecycle.py`)**：`/volume2/CCTV:/cctv:ro`、`/volume2/Baby_Vault:/vault:ro` 均严格只读挂载。必须同时满足「百度首次源片备份完成 + 历史全量归档完成 + 单日产出品完整 + 超过 `source_retention_days` (默认 90 天)」且明确处于 `enforce` 模式，才允许清理超期监控原片；`Baby_Vault` 永久不受清理。项目完成后原视频不上百度云（`cctv-baidu-backup` 已设 `daily_enabled=false`），由 `ArchiveCloudSync` 自动同步 `/volume2/Growth_Archive_V3` 精编成果上云。
2. **NAS 24×7 运行 + 本地轻量开发**：日常服务全天候运行于 NAS `cctv-growth-v3` 容器；本地工作区仅保留代码与轻量工具，改动后配置好 `.env.nas`（或设置 `CCTV_NAS_HOST` / `CCTV_NAS_USER` 环境变量）并运行 `py scripts/deploy_nas_growth_v3.py` 即可增量打包同步并零丢数热重建 NAS 容器。
3. **P0 当日优先 + 多日滚动 6 路高并发**：每 15 分钟扫描当日与昨日稳定片段（闭档 >=15 分钟、最新档 >=90 分钟），始终保持至少 2 个未完成历史日期（如 18 号清空后立即拉入 16 号与 17 号交错并跑）6 路并发推进，后台异步审片不阻塞主队列，遇到当日新片立即抢占。
4. **云端专职分工 + 多账号 GCP Vertex AI 优先（优先消耗 \$300 试用赠金）+ 自领任务工作池 + 隐私门禁**：
   - 高频筛选与厨房核验（`fast`，自领任务工作池）：优先走多账号 GCP Vertex AI 的 `gemini-4-hydrogen`（`/run/secrets/gcp_vertex_sa_1.json` / `gcp_vertex_sa_2.json`），并与 `qwen4-max`、`DeepSeek-V4.1-Flash-1017`、`gpt-6.2-sol`（OpenAI 云端 CLIProxy 网关）并发抢单处理；AI Studio `gemini-4-hydrogen` 原 Key 作为末位容灾兜底，无额度/冷却自动退出并把未完成任务退回队列由有额度的模型无缝接手。
   - 重要里程碑与高光精审（`pro`）：由多账号 GCP Vertex AI 的 `gemini-4-carbon` 与 OpenAI 云端 `gpt-6.2-sol` 共同并发精审高光（优先消耗 GCP 试用赠金与订阅额度，无额度自动退出切备）。
   - 音轨全程保留在 NAS 本地，永不上传；宝宝床头夜间睡眠时段仅本地处理，永不上传云端。
5. **纯童声「妈妈/妈」「爸爸/爸」隔离核验**：本地 YAMNet + 基频 F0 >= 310 Hz 突发段隔离 + 临时短生命周期 INT8 Whisper 子进程（处理完立即释放全部内存），同时保留清晰双音节与单音节（语音优先留存）。
6. **NAS 内存与硬件保护**：预留 >= 4.0 GiB `MemAvailable` 安全线；使用 `/dev/dri/renderD128` (`h264_vaapi`) 硬件加速编码 720p15 T1 / 1fps T2 / 1080p 精华。
