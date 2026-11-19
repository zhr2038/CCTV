# NAS CCTV 百度网盘备份

部署目录：`/volume1/docker/cctv-baidu-backup`。容器：`cctv-baidu-backup`。
部署前先复制模板文件：`cp config.example.json config.json && cp compose.example.yaml compose.yaml`，并按需修改局域网 IP 与目录配置。
复用现有 OpenList 4.2.7 和百度网盘授权，备份源为 `/volume2/CCTV`，对应 Windows 的 `\\<NAS_LAN_IP>\cctv`。
百度网盘目标为 `/CCTV`，各子目录、文件名和层次保持一致。

## 当前运行方式

按要求先执行首次全量补传。**每日增量默认未启用**；预设时间为北京时间 **03:30**，首次完成后再决定是否启用。
首次任务会生成持久化文件清单，同一优先级内默认按修改时间从新到旧上传。发生网络错误会重试，连续失败后每 30 分钟重试未完成的首次任务。容器重启会恢复进度，不需要重新上传已有成功记录。
配置 `vault_root` 后，整个 `aqara_video` 和 `xiaomi_camera_videos/<BEDSIDE_CAMERA_ID>` 从只读挂载的 `/volume2/Baby_Vault` 读取，沿用原有网盘相对路径。从 `vault_priority_start`（如 `YYYY-MM-DD`）起按拍摄日从早到晚优先上传这两组录像，随后处理它们的较早录像以及其他目录，`TeslaCam` 排在最后。首次切换重新扫描，保险箱里相对路径、大小和修改时间一致的已上传文件沿用原成功时间和 SHA-256。其他大小相同但文件属性不同的录像会比对 SHA-256，确认内容相同后沿用成功记录。
版本 1.0.1 修复重启恢复：OpenList 尚未启动时每 15 秒自动重试；设备编号变化不再影响文件指纹，既有断点直接兼容。首次队列仍以原清单为准，已超过稳定等待时间的文件恢复为可上传状态。
如果录像已被源端清理，程序保留 `missing` 记录，继续上传其他文件，不把缺失文件记作上传成功；全部可读取文件上传后，页面会明确显示有源文件缺失。原始卷尚未挂载或必需目录不可用时会停止处理，避免误记为文件被清理。
队列按每批 100 条读取并释放数据库游标，允许 SQLite 日志定期回收，避免长期上传时日志持续膨胀。

扫描覆盖 CCTV 下全部六个顶层目录，包括 Aqara、小米、TeslaCam、NORMAL、USB Copy 和 XiaoBaiCam。
回收站及 NAS 系统缓存目录不上传；符号链接不跟随。文件在最后修改后至少等待 90 分钟，并在上传前后检查大小、修改时间、inode 和 ctime。首次清单里的新近文件会在后续重试中补传；首次扫描结束后新增的文件留待未来的增量任务。

源目录只读挂载。程序不删除本地文件，不删除云端多余文件。相同源路径内容发生变化时，后续运行会覆盖对应云端路径；这不是多版本归档。
百度网盘拒绝零字节文件，因此空文件的路径和修改时间保存在云端 `.cctv-backup-meta/empty-files.json`，恢复时需据此创建空文件。

## 状态和维护

局域网管理页面：`http://<NAS_LAN_IP>:8791/`；JSON 状态接口：`http://<NAS_LAN_IP>:8791/status.json`。
页面显示实际上传进度、目录统计、当前文件、最近五分钟完成速度和活动记录，提供暂停、继续及重试未完成任务。
暂停在当前文件完成后生效，暂停状态跨容器重启保留。控制接口限定 NAS/本机 Host、同源 Origin、JSON 及专用请求头，不接受跨站表单操作；管理页面面向局域网，不含账号登录。
页面不提供录像下载或凭据。
`totals` 约每五分钟刷新一次；页面的上传数量、容量和目录进度在每个文件完成后更新，页面每三秒刷新。
页面百分比按清单中的文件容量计算；速度按已完成文件的容量变化计算，初期估算剩余时间可能波动。

在 NAS SSH 中执行：

```sh
cd /volume1/docker/cctv-baidu-backup
docker compose ps
docker compose logs --tail 100
docker compose exec cctv-baidu-backup python /app/backup.py status
```

暂停或继续首次任务：

```sh
docker compose stop
docker compose start
```

确认首次完成并决定启用每日增量后，把 `config.json` 中 `daily_enabled` 改为 `true`，再执行 `docker compose restart`。目前保持 `false`。
每日任务如果跨日运行，程序只运行一个实例，不叠加上传任务；NAS 宕机后会恢复未完成任务。
锁文件防止维护命令与后台任务同时写入同一份断点库。

## 数据和权限

- `state/manifest.sqlite3`：文件指纹、上传状态、已发送内容的 SHA-256、成功时间。
- `state/status.json`：完整的本地运行状态。
- `state/scan-errors.json`：扫描时无法读取的路径，非空时不会报告首次全部完成。
- `state/control.json`：持久化暂停状态。
- `state/activity.json`：最近的操作和运行事件。
- `state/tmp/`：SQLite 大清单查询的临时空间，避免使用受限的容器 `/tmp`。
- `secrets/openlist.json`：自动生成的专用账号凭据，仅保存在 NAS，权限 0600；不要复制到代码、聊天或镜像。
- OpenList 专用账号 `cctv-backup` 仅能访问 `/百度网盘/CCTV`，具有建目录和上传权限，不具有删除、移动、重命名权限。
- 容器不挂载 Docker socket，不读取 OpenList 的数据库或百度令牌；`provision.py` 仅在安装时以只读方式读取既有配置，建立专用账号。
- 镜像基于本机已有、固定 digest 的 Python 3.12，程序无第三方 Python 依赖。构建上下文只包含 `backup.py` 和 Dockerfile。

## 验证及限制

已在 NAS Linux 容器内运行自动验证，覆盖 03:30 调度、断点恢复、增量跳过、源文件变化、符号链接、失败不标成功、异步确认不算完成、批量检查点、空文件恢复清单、大清单免排序，以及页面控制的同源限制和暂停恢复。
安装时已实际上传一段 4,085,103 字节的录像，并从百度网盘下载，SHA-256 与源内容一致；确认原录像挂载只读。

日常上传使用同步 OpenList API，只有成功响应且云端可见大小正确后才写入成功断点，保存发送内容的 SHA-256。
这不等同于对每个云端文件做完整下载校验。容器健康表示进程和心跳正常，是否有上传错误应查看 `phase`、`run_failures` 和 `totals.failed`。
网盘会员文件大小上限、容量、授权有效期、接口限流和家庭上行速度仍会影响上传；失败文件保留在待重试清单，不会记作已备份。
OpenList 上传时会使用其既有临时目录缓存单个文件，程序不修改其配置或重启 OpenList。

原有家庭 AI 索引程序及其本地处理规则保持独立；此次云备份授权仅用于这个容器。
