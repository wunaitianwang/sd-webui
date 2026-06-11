#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sd.py — CLI for local Stable Diffusion WebUI (A1111) API.

Pure stdlib (urllib), no pip dependencies. Designed for Claude Code to drive
the aki (秋叶) integration package at SD_ROOT.

Commands: status start stop models samplers upscalers loras embeddings
          options set-model txt2img img2img upscale interrogate png-info
          progress interrupt raw
Run `python sd.py <command> --help` for per-command options.
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

SD_ROOT = os.environ.get(
    "SD_WEBUI_ROOT",
    r"D:\ai\sd-webui-aki-v4.11.1-cu128\sd-webui-aki-v4.11.1-cu128",
)
BASE_URL = os.environ.get("SD_WEBUI_URL", "http://127.0.0.1:7860")
PID_FILE = os.path.join(SD_ROOT, "tmp", "sd_api_launch.pid")
LOG_FILE = os.path.join(SD_ROOT, "tmp", "sd_api_launch.log")

# Windows console may be GBK; force UTF-8 stdout so JSON prints safely.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def api(path, payload=None, method=None, timeout=600):
    """Call WebUI API. GET if payload is None else POST. Returns parsed JSON."""
    url = BASE_URL + path
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=method or ("POST" if data else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def api_ok():
    try:
        api("/sdapi/v1/progress?skip_current_image=true", timeout=5)
        return True
    except Exception:
        return False


def ensure_running(wait=300):
    """Start WebUI with --api if not already up. Returns True when ready."""
    if api_ok():
        return True
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
    # If another launch is in flight (pid file fresh + process alive), just wait.
    if not _launch_in_flight():
        python_exe = os.path.join(SD_ROOT, "python", "python.exe")
        # --medvram is essential: 8GB laptop GPU + SDXL checkpoints OOM-crash
        # the server mid-generation without it (observed 356s/it thrashing).
        cmd = [python_exe, "launch.py", "--api", "--xformers", "--medvram",
               "--api-server-stop"]
        log = open(LOG_FILE, "w", encoding="utf-8", errors="replace")
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        proc = subprocess.Popen(cmd, cwd=SD_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=flags)
        with open(PID_FILE, "w") as f:
            f.write(str(proc.pid))
        print(f"[sd.py] launching WebUI (pid {proc.pid}), log: {LOG_FILE}", file=sys.stderr)
    deadline = time.time() + wait
    while time.time() < deadline:
        if api_ok():
            print("[sd.py] WebUI API ready", file=sys.stderr)
            return True
        time.sleep(3)
    print(f"[sd.py] timed out after {wait}s; check {LOG_FILE}", file=sys.stderr)
    return False


def _pid_alive(pid):
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _launch_in_flight():
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        return _pid_alive(pid)
    except (OSError, ValueError):
        return False


def save_images(images, outdir, prefix, info_text=None):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    ts = time.strftime("%Y%m%d-%H%M%S")
    for i, b64 in enumerate(images):
        p = os.path.join(outdir, f"{prefix}-{ts}-{i}.png")
        with open(p, "wb") as f:
            f.write(base64.b64decode(b64.split(",", 1)[-1]))
        paths.append(p)
    if info_text:
        meta = os.path.join(outdir, f"{prefix}-{ts}-info.json")
        with open(meta, "w", encoding="utf-8") as f:
            f.write(info_text if isinstance(info_text, str) else json.dumps(info_text, ensure_ascii=False, indent=2))
        paths.append(meta)
    return paths


def b64_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def out(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


# ---------------- commands ----------------

def cmd_status(_):
    if not api_ok():
        out({"api": "offline", "hint": "run: python sd.py start"})
        return
    opts = api("/sdapi/v1/options")
    mem = {}
    try:
        m = api("/sdapi/v1/memory")
        cuda = m.get("cuda", {}).get("system", {})
        mem = {"vram_free_gb": round(cuda.get("free", 0) / 2**30, 2),
               "vram_total_gb": round(cuda.get("total", 0) / 2**30, 2)}
    except Exception:
        pass
    out({"api": "online", "url": BASE_URL,
         "checkpoint": opts.get("sd_model_checkpoint"),
         "vae": opts.get("sd_vae"),
         "clip_skip": opts.get("CLIP_stop_at_last_layers"), **mem})


def cmd_start(args):
    ok = ensure_running(wait=args.wait)
    cmd_status(args) if ok else sys.exit(1)


def cmd_stop(_):
    if not api_ok():
        out({"stopped": False, "reason": "already offline"})
        return
    try:
        api("/sdapi/v1/server-stop", payload={}, method="POST", timeout=10)
    except Exception:
        pass  # server dies mid-response; that's expected
    time.sleep(3)
    if api_ok():
        # server-stop unavailable (launched without --api-server-stop, e.g.
        # via the GUI launcher) — fall back to killing the recorded PID.
        try:
            with open(PID_FILE) as f:
                pid = int(f.read().strip())
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True)
            else:
                os.kill(pid, 15)
            time.sleep(2)
        except (OSError, ValueError):
            pass
    out({"stopped": not api_ok()})


def _require_api():
    if not ensure_running():
        print("ERROR: WebUI API unavailable", file=sys.stderr)
        sys.exit(1)


def cmd_models(_):
    _require_api()
    out([{"title": m["title"], "model_name": m["model_name"]}
         for m in api("/sdapi/v1/sd-models")])


def cmd_samplers(_):
    _require_api()
    out([s["name"] for s in api("/sdapi/v1/samplers")])


def cmd_schedulers(_):
    _require_api()
    out([s["name"] for s in api("/sdapi/v1/schedulers")])


def cmd_upscalers(_):
    _require_api()
    out([u["name"] for u in api("/sdapi/v1/upscalers")])


def cmd_loras(_):
    _require_api()
    out([{"name": l["name"], "alias": l.get("alias")} for l in api("/sdapi/v1/loras")])


def cmd_embeddings(_):
    _require_api()
    out(list(api("/sdapi/v1/embeddings").get("loaded", {}).keys()))


def cmd_options(args):
    _require_api()
    if args.set:
        payload = {}
        for kv in args.set:
            k, _, v = kv.partition("=")
            try:
                v = json.loads(v)
            except ValueError:
                pass  # keep as string
            payload[k] = v
        api("/sdapi/v1/options", payload)
        out({"updated": payload})
    else:
        opts = api("/sdapi/v1/options")
        keys = ["sd_model_checkpoint", "sd_vae", "CLIP_stop_at_last_layers",
                "eta_noise_seed_delta", "samples_save"]
        out({k: opts.get(k) for k in keys} if not args.all else opts)


def cmd_set_model(args):
    _require_api()
    api("/sdapi/v1/options", {"sd_model_checkpoint": args.name})
    out({"checkpoint": args.name})


def cmd_txt2img(args):
    _require_api()
    payload = {
        "prompt": args.prompt,
        "negative_prompt": args.negative,
        "width": args.width, "height": args.height,
        "steps": args.steps, "cfg_scale": args.cfg,
        "sampler_name": args.sampler, "scheduler": args.scheduler,
        "seed": args.seed, "batch_size": args.batch,
        "n_iter": args.iter,
    }
    if args.hr:
        payload.update({
            "enable_hr": True, "hr_scale": args.hr_scale,
            "hr_upscaler": args.hr_upscaler,
            "hr_second_pass_steps": args.hr_steps,
            "denoising_strength": args.denoise,
        })
    if args.extra:
        payload.update(json.loads(args.extra))
    r = api("/sdapi/v1/txt2img", payload)
    info = json.loads(r.get("info", "{}"))
    paths = save_images(r["images"], args.outdir, "txt2img", r.get("info"))
    out({"saved": paths, "seed": info.get("seed"), "model": info.get("sd_model_name")})


def cmd_img2img(args):
    _require_api()
    payload = {
        "init_images": [b64_file(args.init)],
        "prompt": args.prompt, "negative_prompt": args.negative,
        "width": args.width, "height": args.height,
        "steps": args.steps, "cfg_scale": args.cfg,
        "sampler_name": args.sampler, "scheduler": args.scheduler,
        "seed": args.seed, "denoising_strength": args.denoise,
        "batch_size": args.batch,
    }
    if args.mask:
        payload["mask"] = b64_file(args.mask)
        payload["inpainting_fill"] = args.mask_fill
        payload["inpaint_full_res"] = args.inpaint_full_res
    if args.extra:
        payload.update(json.loads(args.extra))
    r = api("/sdapi/v1/img2img", payload)
    info = json.loads(r.get("info", "{}"))
    paths = save_images(r["images"], args.outdir, "img2img", r.get("info"))
    out({"saved": paths, "seed": info.get("seed")})


def cmd_upscale(args):
    _require_api()
    payload = {
        "image": b64_file(args.image),
        "upscaling_resize": args.scale,
        "upscaler_1": args.upscaler,
    }
    r = api("/sdapi/v1/extra-single-image", payload)
    paths = save_images([r["image"]], args.outdir, "upscale")
    out({"saved": paths})


def cmd_interrogate(args):
    _require_api()
    r = api("/sdapi/v1/interrogate",
            {"image": b64_file(args.image), "model": args.model})
    out(r)


def cmd_png_info(args):
    _require_api()
    r = api("/sdapi/v1/png-info", {"image": "data:image/png;base64," + b64_file(args.image)})
    out({"info": r.get("info")})


def cmd_progress(_):
    if not api_ok():
        out({"api": "offline"})
        return
    r = api("/sdapi/v1/progress?skip_current_image=true")
    out({"progress": round(r.get("progress", 0) * 100, 1),
         "eta_seconds": round(r.get("eta_relative", 0), 1),
         "job": r.get("state", {}).get("job")})


def cmd_interrupt(_):
    _require_api()
    api("/sdapi/v1/interrupt", payload={})
    out({"interrupted": True})


def cmd_raw(args):
    _require_api()
    path = args.path
    # Git Bash mangles leading-slash args into Windows paths (MSYS path
    # conversion), e.g. /sdapi/... -> C:/Program Files/Git/sdapi/...
    # Recover the real endpoint, and accept paths without a leading slash.
    if ":/" in path[:3]:
        for root in ("/sdapi/", "/controlnet/", "/tagger/", "/sam/",
                     "/adetailer/", "/agent-scheduler/", "/infinite_image_browsing/"):
            if root in path:
                path = path[path.index(root):]
                break
    if not path.startswith("/"):
        path = "/" + path
    payload = None
    if args.json:
        payload = json.loads(args.json)
    elif args.json_file:
        with open(args.json_file, encoding="utf-8") as f:
            payload = json.load(f)
    r = api(path, payload, method=args.method)
    if args.save_images_to and isinstance(r, dict) and r.get("images"):
        paths = save_images(r["images"], args.save_images_to, "raw", r.get("info"))
        r["images"] = f"<saved to {paths}>"
    out(r)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(fn=cmd_status)
    sp = sub.add_parser("start")
    sp.add_argument("--wait", type=int, default=300)
    sp.set_defaults(fn=cmd_start)
    sub.add_parser("stop").set_defaults(fn=cmd_stop)
    sub.add_parser("models").set_defaults(fn=cmd_models)
    sub.add_parser("samplers").set_defaults(fn=cmd_samplers)
    sub.add_parser("schedulers").set_defaults(fn=cmd_schedulers)
    sub.add_parser("upscalers").set_defaults(fn=cmd_upscalers)
    sub.add_parser("loras").set_defaults(fn=cmd_loras)
    sub.add_parser("embeddings").set_defaults(fn=cmd_embeddings)

    sp = sub.add_parser("options")
    sp.add_argument("--set", nargs="*", metavar="KEY=VAL")
    sp.add_argument("--all", action="store_true")
    sp.set_defaults(fn=cmd_options)

    sp = sub.add_parser("set-model")
    sp.add_argument("name")
    sp.set_defaults(fn=cmd_set_model)

    sp = sub.add_parser("txt2img")
    sp.add_argument("--prompt", required=True)
    sp.add_argument("--negative", default="")
    sp.add_argument("--width", type=int, default=512)
    sp.add_argument("--height", type=int, default=512)
    sp.add_argument("--steps", type=int, default=24)
    sp.add_argument("--cfg", type=float, default=7.0)
    sp.add_argument("--sampler", default="DPM++ 2M")
    sp.add_argument("--scheduler", default="Karras")
    sp.add_argument("--seed", type=int, default=-1)
    sp.add_argument("--batch", type=int, default=1)
    sp.add_argument("--iter", type=int, default=1)
    sp.add_argument("--hr", action="store_true", help="enable hires fix")
    sp.add_argument("--hr-scale", type=float, default=2.0)
    sp.add_argument("--hr-upscaler", default="4x-AnimeSharp")
    sp.add_argument("--hr-steps", type=int, default=12)
    sp.add_argument("--denoise", type=float, default=0.45)
    sp.add_argument("--extra", help="JSON merged into payload (alwayson_scripts etc.)")
    sp.add_argument("--outdir", required=True)
    sp.set_defaults(fn=cmd_txt2img)

    sp = sub.add_parser("img2img")
    sp.add_argument("--init", required=True, help="input image path")
    sp.add_argument("--prompt", required=True)
    sp.add_argument("--negative", default="")
    sp.add_argument("--width", type=int, default=512)
    sp.add_argument("--height", type=int, default=512)
    sp.add_argument("--steps", type=int, default=24)
    sp.add_argument("--cfg", type=float, default=7.0)
    sp.add_argument("--sampler", default="DPM++ 2M")
    sp.add_argument("--scheduler", default="Karras")
    sp.add_argument("--seed", type=int, default=-1)
    sp.add_argument("--denoise", type=float, default=0.6)
    sp.add_argument("--batch", type=int, default=1)
    sp.add_argument("--mask", help="mask image path (white = repaint)")
    sp.add_argument("--mask-fill", type=int, default=1,
                    help="0 fill 1 original 2 latent noise 3 latent nothing")
    sp.add_argument("--inpaint-full-res", action="store_true")
    sp.add_argument("--extra")
    sp.add_argument("--outdir", required=True)
    sp.set_defaults(fn=cmd_img2img)

    sp = sub.add_parser("upscale")
    sp.add_argument("--image", required=True)
    sp.add_argument("--scale", type=float, default=2.0)
    sp.add_argument("--upscaler", default="R-ESRGAN 4x+ Anime6B")
    sp.add_argument("--outdir", required=True)
    sp.set_defaults(fn=cmd_upscale)

    sp = sub.add_parser("interrogate")
    sp.add_argument("--image", required=True)
    sp.add_argument("--model", default="deepdanbooru", choices=["clip", "deepdanbooru"])
    sp.set_defaults(fn=cmd_interrogate)

    sp = sub.add_parser("png-info")
    sp.add_argument("--image", required=True)
    sp.set_defaults(fn=cmd_png_info)

    sub.add_parser("progress").set_defaults(fn=cmd_progress)
    sub.add_parser("interrupt").set_defaults(fn=cmd_interrupt)

    sp = sub.add_parser("raw", help="arbitrary API call")
    sp.add_argument("path", help="e.g. /sdapi/v1/txt2img or /controlnet/model_list")
    sp.add_argument("--json", help="inline JSON payload (POST)")
    sp.add_argument("--json-file", help="payload from file (POST)")
    sp.add_argument("--method", help="override HTTP method")
    sp.add_argument("--save-images-to", help="decode images[] in response to this dir")
    sp.set_defaults(fn=cmd_raw)

    args = p.parse_args()
    try:
        args.fn(args)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:2000]
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
