# SD WebUI API 进阶参考

本机扩展：ControlNet、ADetailer、Ultimate Upscale、Tiled Diffusion/VAE (multidiffusion)、
Regional Prompter、AnimateDiff、segment-anything、inpaint-anything、wd14-tagger、
supermerger、lora-block-weight、dynamic-thresholding、infinite-image-browsing。

扩展通过 `alwayson_scripts`（常驻脚本）或 `script_name`+`script_args`（互斥脚本）注入
txt2img/img2img payload。用 `sd.py txt2img --extra '<json>'` 合并，或 `sd.py raw` 全手写。

## 目录

1. [ControlNet](#controlnet)
2. [ADetailer（修脸修手）](#adetailer)
3. [Ultimate SD Upscale](#ultimate-sd-upscale)
4. [Tiled Diffusion / Tiled VAE](#tiled-diffusion--tiled-vae)
5. [Regional Prompter](#regional-prompter)
6. [LoRA 用法](#lora)
7. [wd14-tagger 反推](#wd14-tagger)
8. [AnimateDiff](#animatediff)
9. [常用裸 API 端点](#常用裸-api-端点)

## ControlNet

**注意：本机 ControlNet 模型目录为空。** 先确认有模型再用：

```bash
sd.py raw /controlnet/model_list        # 可用模型
sd.py raw /controlnet/module_list      # 可用预处理器
```

为空时告知用户：需将模型（如 control_v11p_sd15_openpose.pth）放入
`models/ControlNet/`，SD1.5 模型对应 v1.1 系列。

`--extra` 写法（openpose 例）：

```json
{
  "alwayson_scripts": {
    "controlnet": {
      "args": [
        {
          "enabled": true,
          "image": "<base64 of reference image>",
          "module": "openpose_full",
          "model": "control_v11p_sd15_openpose [cab727d4]",
          "weight": 1.0,
          "guidance_start": 0.0,
          "guidance_end": 1.0,
          "control_mode": 0,
          "resize_mode": 1,
          "pixel_perfect": true
        }
      ]
    }
  }
}
```

- `control_mode`: 0 均衡 / 1 提示词优先 / 2 ControlNet 优先
- `resize_mode`: 0 拉伸 / 1 裁剪 / 2 填充
- 多个 ControlNet 单元就在 `args` 里放多个对象
- 单独跑预处理器：`POST /controlnet/detect`，payload `{"controlnet_module": "canny", "controlnet_input_images": ["<b64>"]}`

base64 编码图片：`python -c "import base64;print(base64.b64encode(open('x.png','rb').read()).decode())" > b64.txt`，
然后写进 json 文件用 `--json-file` 传，避免命令行长度限制。

## ADetailer

自动检测并重绘脸/手。**模型目录当前为空**，但 adetailer 启动时会自动从 huggingface
下载默认模型（需要网络，本机有代理 127.0.0.1:7890）。失败则提示用户手动下载
face_yolov8n.pt 到 `models/adetailer/`。

```json
{
  "alwayson_scripts": {
    "ADetailer": {
      "args": [
        true,
        false,
        {
          "ad_model": "face_yolov8n.pt",
          "ad_prompt": "detailed beautiful face",
          "ad_denoising_strength": 0.4,
          "ad_inpaint_only_masked": true
        }
      ]
    }
  }
}
```

args 结构：`[启用(bool), 跳过img2img(bool), 单元1(dict), 单元2(dict)...]`。
修手用 `"ad_model": "hand_yolov8n.pt"`。

## Ultimate SD Upscale

img2img 专用互斥脚本，把大图切块逐块重绘放大，8GB 显存放大到 4K 的正确姿势。
搭配低 denoise（0.2-0.35）。

```bash
sd.py img2img --init big.png --prompt "masterpiece, best quality, detailed" \
  --denoise 0.3 --outdir out --extra '{
    "script_name": "Ultimate SD upscale",
    "script_args": [null, 512, 512, 8, 32, 64, 0.35, 32, 6, true, 0, false, 8, 0, 2, 2048, 2048, 2.0]
  }'
```

script_args 顺序（版本相关，出错时让用户在 WebUI 里手动确认一次参数）：
`[_, tile_width, tile_height, mask_blur, padding, seams_fix_width, seams_fix_denoise,
seams_fix_padding, upscaler_index, save_upscaled_image, redraw_mode, save_seams_fix_image,
seams_fix_mask_blur, seams_fix_type, target_size_type, custom_width, custom_height, custom_scale]`

`upscaler_index`: 按 `sd.py upscalers` 返回顺序（0=None, 1=Lanczos, ...）。

## Tiled Diffusion / Tiled VAE

直接生成超大图时避免爆显存：

```json
{
  "alwayson_scripts": {
    "Tiled Diffusion": {"args": [true, "MultiDiffusion", false, true, 1024, 1024, 96, 96, 48, 4, "4x-AnimeSharp", 2, false, 0, 0.0, 0]},
    "Tiled VAE": {"args": [true, 1536, 96, true, true, true, false]}
  }
}
```

参数顺序随版本变动，失败时退回 hires fix 或 Ultimate Upscale 方案。

## Regional Prompter

把画面分区域用不同提示词。prompt 用 `BREAK` 分隔各区域：

```json
{
  "alwayson_scripts": {
    "Regional Prompter": {
      "args": [true, false, "Matrix", "Columns", "Mask", "Prompt", "1,1", "0.2", false, false, false, "Attention", false, "0", "0", "0.4", null, null, false]
    }
  }
}
```

`"Columns"` + `"1,1"` = 左右两栏。prompt 写法：`公共描述 BREAK 左边内容 BREAK 右边内容`。

## LoRA

LoRA 不走单独参数，直接写进 prompt：`<lora:文件名:0.8>`。
`sd.py loras` 列出可用 LoRA（当前 Lora 目录为空，用户放入 .safetensors 后即可用）。
权重 0.6-1.0 常用；lora-block-weight 扩展支持 `<lora:name:1:LBW=...>` 分层控权。

## wd14-tagger

比 deepdanbooru 更准的动漫标签反推：

```bash
sd.py raw /tagger/v1/interrogate --json '{"image": "<b64>", "model": "wd14-vit-v2-git", "threshold": 0.35}'
sd.py raw /tagger/v1/interrogators   # 可用模型列表
```

## AnimateDiff

生成短动画（需先下载运动模块到 `extensions/sd-webui-animatediff/model/`，目录当前为空）：

```json
{
  "alwayson_scripts": {
    "AnimateDiff": {
      "args": [{
        "model": "mm_sd15_v3.safetensors",
        "enable": true, "video_length": 16, "fps": 8,
        "closed_loop": "R+P", "format": ["GIF"]
      }]
    }
  }
}
```

## 常用裸 API 端点

| 端点 | 方法 | 用途 |
|---|---|---|
| /sdapi/v1/txt2img, /img2img | POST | 生成 |
| /sdapi/v1/extra-single-image | POST | 单图后处理放大 |
| /sdapi/v1/extra-batch-images | POST | 批量放大 |
| /sdapi/v1/png-info | POST | 读图内参数 |
| /sdapi/v1/interrogate | POST | clip/deepdanbooru 反推 |
| /sdapi/v1/refresh-checkpoints | POST | 刷新模型列表（新放入文件后）|
| /sdapi/v1/refresh-loras | POST | 刷新 LoRA |
| /sdapi/v1/unload-checkpoint | POST | 卸载模型省显存 |
| /sdapi/v1/reload-checkpoint | POST | 重新加载 |
| /sdapi/v1/memory | GET | 显存占用 |
| /sdapi/v1/scripts | GET | 已注册脚本名（查 script_name 写法）|
| /sdapi/v1/script-info | GET | 脚本参数签名（args 顺序不确定时查这个）|
| /controlnet/version, /model_list, /module_list, /detect | | ControlNet |
| /tagger/v1/interrogate, /interrogators | | wd14 标签反推 |
| /sam/sam-predict, /sam/dino-predict | POST | segment-anything 抠图 |

**script_args 顺序拿不准时，先 `sd.py raw /sdapi/v1/script-info` 查参数签名，别瞎猜。**
