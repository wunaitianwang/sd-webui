---
name: sd-webui
description: 调用本地 Stable Diffusion WebUI（秋叶整合包）生成和处理图片。当用户要求"画图"、"生成图片"、"AI绘画"、"文生图"、"图生图"、"放大图片"、"反推提示词"、"换模型"、"用SD生成"，或提到 stable diffusion、SD、WebUI、anything-v5、LoRA、ControlNet、采样器、提示词生图等内容时使用本 skill。即使用户没有明确说"用SD"，只要是本地AI图片生成、二次元/动漫风格图片创作、图片高清放大、图片转提示词的需求，都应使用本 skill（注意：如果用户明确要求用速创API/在线服务生图则用 ai-image-gen，本 skill 只负责本地 SD WebUI）。WebUI 未运行时本 skill 会自动启动它。
---

# SD WebUI 本地图片生成

通过 HTTP API 驱动本地 Stable Diffusion WebUI（A1111 秋叶整合包 v4.11.1）。
所有操作都通过 `scripts/sd.py` 完成，不要直接写 curl 或临时 Python 脚本。

## 环境概况（本机实际情况）

- 安装目录：`D:\ai\sd-webui-aki-v4.11.1-cu128\sd-webui-aki-v4.11.1-cu128`
- API 地址：`http://127.0.0.1:7860`
- GPU：RTX 4060 Laptop **8GB 显存**
- 两个 checkpoint（用 `models` 命令确认当前列表，用户可能新增）：
  - `illustriousXLV20_v20Stable`（**SDXL/Illustrious 架构**，动漫）：原生分辨率 1024×1024 / 竖图 832×1216，**别用 512 出图会糊**；**不要**搭配 SD1.5 的 embedding（EasyNegative 等对它无效）；负面用纯文本 `worst quality, low quality, bad anatomy, bad hands, watermark`
  - `sd1.5\anything-v5`（SD1.5 架构，动漫）：512×768 直出，负面提示词用 `EasyNegative, badhandv4` embedding
- 生成前先 `status` 看当前加载的是哪个模型，按架构选分辨率和负面提示词；切换模型要十几秒
- **VAE 必须保持 `Automatic`**（`options --set sd_vae=Automatic`）：如果 status 显示 sd_vae 是 `sdxl_vae.safetensors` 而当前模型是 SD1.5，出图会色彩崩坏成噪点——先改回 Automatic 再生成
- 8GB 显存下 SDXL 直出别超 832×1216；脚本已用 `--medvram` 启动（没有它 SDXL 会在 VAE 解码阶段显存爆掉假死）。更大分辨率用 hires fix（SDXL 用 --hr-scale 1.5）或 upscale
- 动漫放大模型：`4x-AnimeSharp`、`R-ESRGAN 4x+ Anime6B`
- ControlNet / ADetailer 扩展已安装，但**模型文件未下载**——用户要用时先查 `raw /controlnet/model_list`，为空就告知用户需要先下载对应模型，不要硬试

## 工作流

每个命令都会在 API 离线时自动启动 WebUI 并等待就绪（首次启动约 40-90 秒，会加载模型）。
脚本输出 JSON 到 stdout，启动日志在 stderr。

```bash
SD="python C:/Users/z2269/.claude/skills/sd-webui/scripts/sd.py"

$SD status                  # API 状态 + 当前模型 + 显存
$SD start                   # 显式启动（其他命令也会自动启动）
$SD stop                    # 关闭 WebUI 释放显存
$SD models                  # 可用 checkpoint
$SD samplers / upscalers / loras / embeddings / schedulers
$SD set-model "名称"        # 切换 checkpoint（加载需十几秒）
```

### 生成图片前：先问保存位置

用户没有指定保存目录时，**先询问一次**图片保存到哪里（给出建议如 `./sd-output`），同一会话内复用该答案，不要每张图都问。

### 文生图

```bash
$SD txt2img --prompt "1girl, silver hair, ..." \
  --negative "EasyNegative, badhandv4" \
  --width 512 --height 768 --steps 24 --cfg 7 \
  --sampler "DPM++ 2M" --scheduler Karras \
  --outdir "保存目录"
```

输出 JSON 里有 `saved`（图片路径列表）和 `seed`。生成后把图片路径告诉用户；如果环境支持，用 Read 工具查看图片确认质量再交付。

高清出图（先 512×768 再 2 倍放大，8GB 显存的正确方式）：

```bash
$SD txt2img --prompt "..." --hr --hr-scale 2 --hr-upscaler 4x-AnimeSharp --denoise 0.45 ...
```

### 图生图 / 局部重绘

```bash
$SD img2img --init 输入图.png --prompt "..." --denoise 0.6 --outdir 目录
# 局部重绘加: --mask 蒙版.png  (白色区域=重画)
```

`--denoise`：0.3 轻微变化，0.5-0.6 风格化重绘，0.75+ 大改。

### 放大 / 反推 / 读取生成参数

```bash
$SD upscale --image 图.png --scale 2 --upscaler "R-ESRGAN 4x+ Anime6B" --outdir 目录
$SD interrogate --image 图.png --model deepdanbooru   # 动漫图用 deepdanbooru，照片用 clip
$SD png-info --image 图.png                            # 读取 SD 生成图的原始参数
```

### 长任务监控与中断

生成大图时可另开命令查询：`$SD progress`（百分比+ETA），`$SD interrupt` 中断。

### 任意 API（扩展功能全覆盖）

`txt2img --extra` 可以把任意 JSON 合并进 payload（如 `alwayson_scripts` 启用 ADetailer/ControlNet）。
更原始的访问用 raw：

```bash
$SD raw /controlnet/model_list
$SD raw /sdapi/v1/txt2img --json-file payload.json --save-images-to 目录
```

ControlNet、ADetailer、Ultimate Upscale、Regional Prompter 等扩展的 payload 写法见
`references/api.md`——用到这些扩展时先读它。

## 动漫模型提示词要点（两个模型通用）

- 用 Danbooru 标签风格，逗号分隔：`1girl, solo, long hair, ...`
- 质量词放开头：anything-v5 用 `masterpiece, best quality`；Illustrious 用 `masterpiece, best quality, amazing quality`
- 负面提示词按架构选（见上面环境概况），不要给 SDXL 模型用 SD1.5 embedding
- 用户给中文描述时，翻译成英文标签再生成；把最终 prompt 告诉用户方便复用
- 推荐参数：`DPM++ 2M` + Karras；anything-v5: steps 20-28 / CFG 6-8 / 512×768；Illustrious: steps 24-30 / CFG 5-7 / 832×1216

## 故障排查

- 启动超时 → 看 `D:\ai\...\tmp\sd_api_launch.log` 尾部报错（日志含 \r 进度条，用 `tr '\r' '\n'` 再 tail）
- SD1.5 模型出图全是彩色噪点/色块 → VAE 配错了（SDXL 的 VAE 用在 SD1.5 上），`options --set sd_vae=Automatic`
- 生成卡在 96% 不动、GPU 100% 占用十几分钟 → SDXL 在 8GB 显存上没开 medvram 导致显存抖动假死；本 skill 启动自带 --medvram，但若用户自己用绘世启动器启动则可能没有。处理：`interrupt` 无效就 `stop`（会自动强杀进程树），再用 `start` 重启
- 显存不足（CUDA out of memory）→ 降低分辨率/批量，或换 SD1.5 模型
- 用户用绘世启动器开的 WebUI 没勾 API → 让用户在启动器"高级选项"里开启 API，或 `stop` 后用本 skill 启动
- 端口不是 7860 → 设环境变量 `SD_WEBUI_URL`
- `raw` 命令在 Git Bash 下路径开头会被转成 Windows 路径 → 脚本已自动修正，也可不写开头的 `/`（如 `raw sdapi/v1/memory`）
