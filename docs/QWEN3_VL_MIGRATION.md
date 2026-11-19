# Qwen3-VL 视觉架构升级

## 结果

视觉模型已从业务代码中解耦。默认普通分析使用
`Qwen/Qwen3-VL-8B-Instruct`，高价值里程碑使用
`Qwen/Qwen3-VL-32B-Instruct` 二次确认，视觉搜索使用独立的
`Qwen/Qwen3-VL-Embedding-8B`。旧 Qwen2.5-VL 适配器和历史数据库
结果继续可用。

本次没有修改 SQLite 表、字段、CHECK 约束或事件模型。结构化 Qwen3
结果写入现有 `visual_frame_verifications.details_json`，确认的成长
里程碑写入现有 `activity_events.details_json`；搜索向量继续写入现有
`visual_embeddings`，通过 `model_name` 与 SigLIP 旧向量隔离。

## 代码边界

- `src/cctv_ai/vl_models/base.py`：模型无关接口、固定事件 JSON、安全过滤。
- `src/cctv_ai/vl_models/qwen3_vl.py`：Qwen3 本地 4bit/8bit/BF16 适配器。
- `src/cctv_ai/vl_models/qwen25_vl.py`：Qwen2.5 兼容适配器。
- `src/cctv_ai/vl_models/tiered.py`：8B 候选、32B 复核及原事件详情合并。
- `src/cctv_ai/vl_models/embedding.py`：独立 Qwen3-VL-Embedding 编码器。
- `src/cctv_ai/model_config.py`：`config/models.toml` 校验和本地路径解析。
- `src/cctv_ai/visual_embedding_backfill.py`：可恢复的向量回填。

业务调用只依赖 `VisionLanguageModel`：

```python
class VisionLanguageModel:
    def analyze_image(self, image, prompt, context=None):
        ...

    def analyze_video_frames(self, frames, prompt, context=None):
        ...

    def generate_summary(self, events):
        ...
```

## 模型路径与离线要求

默认路径：

| 用途 | 模型 | 本地目录 |
|---|---|---|
| 普通视觉理解 | Qwen3-VL-8B-Instruct | `.data/models/qwen3-vl-8b-instruct` |
| 重要里程碑复核 | Qwen3-VL-32B-Instruct | `.data/models/qwen3-vl-32b-instruct` |
| 视觉检索 | Qwen3-VL-Embedding-8B | `.data/models/qwen3-vl-embedding-8b` |

路径、设备、4bit/8bit、token 上限和温度都由
`config/models.toml` 配置。加载器要求目录内已有 `config.json` 和完整
Safetensors；所有 `from_pretrained` 均传入 `local_files_only=True`。
模型缺失时会提示运行显式下载脚本，不会访问网络。

模型下载：

```powershell
.\.venv\Scripts\python.exe .\scripts\download_vlm_model.py --role vision-language
.\.venv\Scripts\python.exe .\scripts\download_vlm_model.py --role embedding
.\.venv\Scripts\python.exe .\scripts\download_vlm_model.py --role high-value
```

## 结构化输出和防误报

模型结果必须解析成以下固定结构，数据库不保存自由格式自然语言回答：

```json
{
  "activity": "feeding",
  "description": "宝宝躺在床上，成人使用奶瓶喂食",
  "objects": ["baby", "bottle"],
  "confidence": 0.88,
  "evidence": ["奶瓶奶嘴与宝宝嘴部直接接触"],
  "uncertain": false
}
```

YOLO 内容随相机、时间、物体候选、置信度、框和运动量一起传入。提示词明确
说明检测不是事实，尤其是置信度低于 0.40 的物体。即使模型输出
`activity=feeding`，只要描述和证据没有直接嘴部接触，解析层会强制
`uncertain=true` 且置信度最多 0.45，原喝奶统计只会看到 `UNCLEAR`。

智力、性格、情绪稳定性、发育正常/异常和能力优劣等判断会被过滤，结果
降为不确定；哭泣、微笑、坐、爬、站等直接可见动作仍可描述。

第一次说话不能从嘴部动作推断。8B 候选必须同时带
`audio_transcript_confirmed=true`，才允许进入 32B 二次复核。

## 喂养统计复核修正（2026-07-30）

旧流程把原始 2880×1620 录像先缩成 640×360 关键帧，再让 Qwen 判断
嘴边的小物体。实测 8B 和 32B 都会把成人手部或普通物体补全成奶瓶；
继续加大模型不能解决信息已经在缩放中丢失的问题。新流程改为：

```text
每 5 分钟一个宽召回峰值
  → 回读 NAS 原始录像
  → 候选点 -20/-10/0/+10/+20 秒原生分辨率画面
  → Qwen3-VL-8B 四组序列并行确认
  → 直接嘴部接触且 confidence >= 0.85
  → 20 分钟会话归并
```

基础检测只负责召回。多帧 Qwen3 明确确认的接触可以直接成为事件证据；
`NOT_FEEDING`、`UNCLEAR` 和低置信结果不会进入次数。每批结果立即写入
原 `visual_frame_verifications` 表，关机后只继续尚未保存且原片读取完整
的时间窗。

真实数据复核：

- 人工否决事件 `225818` 的中心帧 `3482`：低分辨率 8B 和 32B 都曾给出
  `FEED_BOTTLE, confidence=0.95`；回读五张原片后改为
  `NOT_FEEDING`，与人工结论一致。
- 关键帧 `21722`：五张原片都能看见奶瓶接触，输出
  `FEED_BOTTLE, confidence=0.95`。
- 关键帧 `155221` 和 `42610`：原片分别显示普通抱姿、安抚奶嘴和照护
  动作，均输出 `NOT_FEEDING`。
- 两段人工否决的空床睡眠（锚点 `23795`、`23823`）均输出
  `NOT_SLEEPING`；人工否决的玩耍锚点 `24725` 输出 `NONE`。

次数只表示摄像头明确拍到的喂养会话，不保证覆盖摄像头外发生的喂养；
当前时长仍是可见证据下限，不等于完整喝奶时长。

## 真实录像测试（2026-07-30）

测试源为 NAS 上的真实床头录像：

`xiaomi_camera_videos/<BEDSIDE_CAMERA_ID>/00_20260730064235_20260730065308.mp4`

中心关键帧为 `155917`，时间 `2026-07-30 06:51:20+08:00`。测试读取
相邻三张真实关键帧（06:50:49、06:51:19、06:51:49），没有改写 NAS
原片。

RTX 5090 实测：

- Qwen3-VL-8B-Instruct：CUDA、bitsandbytes 4bit，模型内存
  `6,258,342,696` 字节，加载约 `13.85` 秒，三帧推理约 `4.40` 秒。
- 原片序列批处理实测 8 组会占用约 `32.1/32.6 GiB`，复杂输出会产生
  长尾；生产因此限制为 4 组并行，显存约 25–27 GiB，保留生成缓存余量。
- 关键帧 `155915`（2026-07-30 06:50:20）由 8B 输出
  `confidence=0.95, uncertain=false`，证据明确包含奶瓶奶嘴与宝宝嘴部
  直接接触，结果已写入现有验证表。
- YOLO 在中心帧给出 `bottle=0.2338` 的低置信候选。
- 8B 原始语义倾向于 feeding，但没有给出足够明确的嘴部接触措辞；一致性
  门把最终结果降为 `confidence=0.45, uncertain=true`。因此该帧不会被
  当作一次喝奶写入统计。这验证了基础检测不能直接支配大模型结论。
- Qwen3-VL-32B-Instruct：CUDA、bitsandbytes 4bit，模型内存
  `19,020,782,376` 字节，加载约 `60.63` 秒，单帧复核约 `13.87` 秒；
  对关键帧 `155915` 给出 `feeding, confidence=0.85,
  uncertain=false`。32B 只用于少量高价值里程碑，不与 8B 常驻显存。
- Qwen3-VL-Embedding-8B：CUDA 4bit，已为同一关键帧生成 1024 维
  `float16-le` 归一化向量；查询“宝宝躺在床上看彩色绘本”的归一化向量
  与该帧相似度为 `0.3702`。

8B 与 32B 完整测试输出分别保存在
`.data/qwen3-vl-smoke-video.out.json` 和
`.data/qwen3-vl-32b-smoke.out.json`。

## 搜索迁移

Qwen3 搜索回填不会覆盖 SigLIP：

```powershell
.\.venv\Scripts\python.exe -m cctv_ai.cli visual-embed-qwen3 --limit 1000
```

在 Qwen3 向量覆盖关键帧比例达到 95% 前，服务仍选择完整的 SigLIP
索引。达到门槛后自动改为 Qwen3-VL-Embedding 召回，再由
Qwen3-VL-8B 对前排候选做可见条件核验；生成模型从不承担向量全文搜索。
计划任务每轮以 4096 张关键帧为一个可恢复批次回填，电脑关机或 NAS
暂时不可用都不会丢失已经写入的向量。

## 验收命令

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe .\scripts\smoke_test_qwen3_vl.py `
  --keyframe-id 155917 --video-frames 3
```

数据库还应执行：

```sql
PRAGMA quick_check;
PRAGMA foreign_key_check;
```

页面没有新增必需字段，旧事件和旧向量保留，静态前端无需迁移即可继续显示。
