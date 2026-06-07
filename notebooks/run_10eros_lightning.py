#!/usr/bin/env python3
"""run_10eros_lightning.py — Standalone (no Jupyter) launcher for the
Vantage 10Eros LTX-2.3 image-to-video workflow on Lightning AI.

USAGE:
    # On a Lightning AI studio (Clean ComfyUI Template recommended):
    python run_10eros_lightning.py                  # setup + Gradio UI
    python run_10eros_lightning.py --setup-only     # downloads + custom nodes, no UI
    python run_10eros_lightning.py --skip-setup     # straight to UI (Models already cached)
    python run_10eros_lightning.py --gradio-port 7860

    # CLI single-shot generation (no Gradio):
    python run_10eros_lightning.py --cli \\
        --source ./me.jpg --prompt "soft gentle smile" \\
        --mode I2V --width 768 --height 1024 --seed 42 \\
        --run-mode full --omninft 0 --photoreal \\
        --output ./result.mp4

LIGHTNING AI:
    1. Launch a Studio from the Clean ComfyUI Template
       (https://lightning.ai/mindthemath/studios/clean-comfyui-template-v0-3-15-20250221).
    2. Switch hardware to GPU (L40s or better recommended).
    3. Upload this script to /teamspace/studios/this_studio/.
    4. In Terminal: python run_10eros_lightning.py
    5. Use Lightning's Custom Port plugin (right panel) to forward port
       7860 → public URL. ComfyUI's 8188 stays internal.

DEPENDENCIES INSTALLED AUTOMATICALLY:
    - mediapipe==0.10.13 (10S-Comfy-nodes uses removed mp.solutions API)
    - 11 custom node packs into ComfyUI/custom_nodes/
    - ~24 GB of model files into ComfyUI/models/

FACE IDENTITY HONESTY:
    LTX-2.3 + 10Eros + 10S nodes preserves subject CATEGORY (Indian
    woman, brown skin) and frame-to-frame consistency well, but does NOT
    pixel-perfect-clone your source photo's face. For strict identity:
    train a 5-min character LoRA (ostris/ai-toolkit) or use Wan 2.2,
    which the LTX-2.3 community itself recommends for face consistency.
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, socket, struct, subprocess, sys, threading, time, wave
import urllib.request, urllib.error
from pathlib import Path

# ----------------------------------------------------------------------
# Constants and discovery
# ----------------------------------------------------------------------

LIGHTNING_WORKSPACE = Path("/teamspace/studios/this_studio")
BASE_PATH = LIGHTNING_WORKSPACE if LIGHTNING_WORKSPACE.exists() else Path.cwd()

COMFY_URL = "http://127.0.0.1:8188"
GEMMA_FILE = "gemma-3-12b-it-orthogonal-reflection-bounded-ablation-v4-12B-fp4_mixed.safetensors"
GEMMA_URL = ("https://huggingface.co/inflatebot/"
             "LTX23-gemma-3-12b-it-orthogonal-reflection-bounded-ablation-v4-fp4_mixed/"
             f"resolve/main/{GEMMA_FILE}")
LTX_TEXT_ENCODER_FILE = "10Eros_v1_text_encoder.safetensors"

NODE_REPOS = [
    "https://github.com/kijai/ComfyUI-KJNodes",
    "https://github.com/city96/ComfyUI-GGUF",
    "https://github.com/Lightricks/ComfyUI-LTXVideo",
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
    "https://github.com/kijai/ComfyUI-MelBandRoFormer",
    "https://github.com/TenStrip/10S-Comfy-nodes",
    "https://github.com/rgthree/rgthree-comfy",
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
    "https://github.com/ClownsharkBatwing/RES4LYF",
    "https://github.com/melMass/comfy_mtb",
    "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes",
]

MODELS = [
    ("https://huggingface.co/vantagewithai/LTX2.3-10Eros-GGUF/resolve/main/10Eros_v1-Q4_K_M.gguf",
     "unet", "10Eros_v1-Q4_K_M.gguf"),
    ("https://huggingface.co/vantagewithai/LTX2.3-10Eros-Split/resolve/main/text_encoder/10Eros_v1_text_encoder.safetensors",
     "text_encoders", "10Eros_v1_text_encoder.safetensors"),
    (GEMMA_URL, "text_encoders", GEMMA_FILE),
    ("https://huggingface.co/vantagewithai/LTX2.3-10Eros-Split/resolve/main/vae/10Eros_v1_vae.safetensors",
     "vae", "10Eros_v1_vae.safetensors"),
    ("https://huggingface.co/vantagewithai/LTX2.3-10Eros-Split/resolve/main/audio_vae/10Eros_v1_audio_vae.safetensors",
     "vae", "10Eros_v1_audio_vae.safetensors"),
    ("https://huggingface.co/VasiliyWeb/OmniNFT_ComfyUI/resolve/main/OmniNFT_converted_lora.safetensors",
     "loras", "OmniNFT_converted_lora.safetensors"),
    ("https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
     "latent_upscale_models", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
    ("https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors",
     "diffusion_models", "MelBandRoformer_fp16.safetensors"),
]

WF_DL_URL = "https://huggingface.co/vantagewithai/LTX2.3-10Eros-Split/resolve/main/Vantage-10Eros_I2V_v3.2.json"

def discover_comfy_path() -> str:
    candidates = [
        str(BASE_PATH / "ComfyUI"),
        str(BASE_PATH / "comfyui"),
        "/workspace/ComfyUI",
        "/root/ComfyUI",
        str(Path.home() / "ComfyUI"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "main.py")):
            return c
    for root in (str(BASE_PATH), "/workspace", "/root", str(Path.home())):
        if os.path.exists(root):
            for hit in glob.glob(f"{root}/**/main.py", recursive=True)[:10]:
                if "ComfyUI" in hit or "comfyui" in hit:
                    return os.path.dirname(hit)
    return None

# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------

def run(cmd, **kw):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, **kw)

def dl(url, dest, fname):
    import requests
    Path(dest).mkdir(parents=True, exist_ok=True)
    fpath = os.path.join(dest, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        print(f"  cached: {fname}")
        return
    if shutil.which("aria2c"):
        run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16",
             "-k", "1M", "-d", dest, "-o", fname, url])
        return
    print(f"  downloading {fname} (requests fallback — install aria2 for speed)...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        tmp = fpath + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk: f.write(chunk)
        os.replace(tmp, fpath)

def setup_environment(comfy_path: str, force_full: bool = False):
    """Install custom nodes, models, mediapipe pin. Idempotent."""
    print(f"[setup] ComfyUI at {comfy_path}", flush=True)

    print("[setup] Pinning mediapipe==0.10.13 (10S-Comfy-nodes uses removed mp.solutions API)...")
    run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "mediapipe==0.10.13"])

    print("[setup] Installing 11 custom node packs...")
    nodes_dir = f"{comfy_path}/custom_nodes"
    os.makedirs(nodes_dir, exist_ok=True)
    for url in NODE_REPOS:
        name = url.rstrip("/").split("/")[-1]
        path = os.path.join(nodes_dir, name)
        if not os.path.exists(path):
            run(["git", "clone", "-q", url, path])
        req = f"{path}/requirements.txt"
        if os.path.exists(req):
            run([sys.executable, "-m", "pip", "install", "-q", "-r", req])
    # Re-pin mediapipe (transitive deps may have overwritten)
    run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "mediapipe==0.10.13"])

    # Try to install aria2c for fast downloads
    if shutil.which("aria2c") is None:
        try:
            run(["apt-get", "update", "-qq"], check=False)
            run(["apt-get", "install", "-y", "-qq", "aria2"], check=False)
        except Exception: pass

    print("[setup] Downloading Vantage / 10Eros models (~24 GB total, mostly cached on rerun)...")
    B = f"{comfy_path}/models"
    for url, sub, fname in MODELS:
        dl(url, f"{B}/{sub}", fname)

    wf_path = str(BASE_PATH / "Vantage-10Eros_I2V_v3.2.json")
    dl(WF_DL_URL, str(BASE_PATH), "Vantage-10Eros_I2V_v3.2.json")

    audio_placeholder = os.path.join(comfy_path, "input", "1.wav")
    if not os.path.exists(audio_placeholder):
        os.makedirs(os.path.dirname(audio_placeholder), exist_ok=True)
        ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
        if os.path.exists(ffmpeg):
            run([ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                 "-t", "1", "-c:a", "pcm_s16le", audio_placeholder], check=False)
        else:
            with wave.open(audio_placeholder, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
                w.writeframes(struct.pack("<" + "h"*44100, *([0]*44100)))

    # Sanity check
    import importlib, mediapipe as mp
    importlib.reload(mp)
    assert hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"), \
        f"mediapipe {mp.__version__} missing mp.solutions"
    assert os.path.exists(wf_path), f"Workflow file missing"
    print(f"[setup] OK. mediapipe={mp.__version__}, workflow={wf_path}", flush=True)

# ----------------------------------------------------------------------
# UI -> API workflow converter
# ----------------------------------------------------------------------

WIDGET_TYPE_NAMES = {"COMBO","INT","FLOAT","STRING","BOOLEAN",
                     "COMFY_DYNAMICCOMBO_V3","COMFY_DYNAMICCOMBO",
                     "COMFY_MULTILINESTRING_V3","COMFY_MULTILINESTRING"}

def _is_widget(t):
    if isinstance(t, list) and t:
        f = t[0]
        if isinstance(f, list): return True
        if isinstance(f, str) and f in WIDGET_TYPE_NAMES: return True
    return False

def _map_widgets(info, vals):
    inp = info.get("input", {}) or {}
    order = info.get("input_order", {}) or {}
    if isinstance(vals, dict): return dict(vals)
    if not isinstance(vals, list): return {}
    out, idx = {}, 0
    for sec in ("required", "optional"):
        keys = order.get(sec) or list((inp.get(sec) or {}).keys())
        for k in keys:
            v = (inp.get(sec) or {}).get(k)
            if not _is_widget(v): continue
            if idx >= len(vals): return out
            out[k] = vals[idx]; idx += 1
            if isinstance(v, list) and v[0] in ("COMFY_DYNAMICCOMBO_V3","COMFY_DYNAMICCOMBO"):
                opts = v[1].get("options", []) if len(v) >= 2 and isinstance(v[1], dict) else []
                for opt in opts:
                    if isinstance(opt, dict) and opt.get("key") == vals[idx-1]:
                        for nk, nv in (opt.get("inputs") or {}).get("required", {}).items():
                            if _is_widget(nv) and idx < len(vals):
                                out[f"{k}.{nk}"] = vals[idx]; idx += 1
                        break
    return out

def ui_to_api(ui_wf, object_info):
    nodes = ui_wf.get("nodes") or []
    links = ui_wf.get("links") or []
    skipped = {str(n.get("id")) for n in nodes if n.get("mode") in (2, 4)}
    link_src = {}
    for raw in links:
        if not isinstance(raw, list) or len(raw) < 5: continue
        link_src[int(raw[0])] = (str(raw[1]), int(raw[2]))
    api = {}
    for n in nodes:
        nid = str(n.get("id")); ctype = n.get("type")
        if not ctype or ctype in ("MarkdownNote","Note","PrimitiveNode"): continue
        if n.get("mode") in (2, 4): continue
        info = object_info.get(ctype)
        if info is None:
            stub = {}
            for s in (n.get("inputs") or []):
                if s.get("link") is None: continue
                src = link_src.get(int(s["link"]))
                if src: stub[s.get("name")] = [src[0], src[1]]; break
            api[nid] = {"class_type": ctype, "inputs": stub}
            continue
        wv = n.get("widgets_values") or []
        inputs = _map_widgets(info, wv)
        # Power Lora Loader: one literal backslash strip ("\\" in Python source).
        if ctype == "Power Lora Loader (rgthree)" and isinstance(wv, list):
            slot = 0
            for x in wv:
                if isinstance(x, dict) and "lora" in x and "strength" in x:
                    lp = x.get("lora") or ""
                    if "\\" in lp: lp = lp.rsplit("\\", 1)[-1]
                    if "/" in lp: lp = lp.rsplit("/", 1)[-1]
                    slot += 1
                    inputs[f"lora_{slot}"] = {"on": bool(x.get("on", True)),
                                              "lora": lp,
                                              "strength": float(x.get("strength", 1.0)),
                                              "strengthTwo": x.get("strengthTwo")}
        for s in (n.get("inputs") or []):
            if s.get("link") is None: continue
            src = link_src.get(int(s["link"]))
            if src and src[0] not in skipped:
                inputs[s.get("name")] = [src[0], src[1]]
        api[nid] = {"class_type": ctype, "inputs": inputs}
    unknown = [nid for nid, n in api.items() if n["class_type"] not in object_info]
    pt = {}
    for nid in unknown:
        for v in api[nid]["inputs"].values():
            if isinstance(v, list) and len(v) >= 2: pt[nid] = v; break
    for n in api.values():
        for k, v in list(n["inputs"].items()):
            if isinstance(v, list) and len(v) >= 1 and str(v[0]) in pt:
                n["inputs"][k] = pt[str(v[0])]
    for nid in unknown:
        if nid in pt: api.pop(nid, None)
    return api

# ----------------------------------------------------------------------
# ComfyUI server lifecycle
# ----------------------------------------------------------------------

def is_server_running(port=8188):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def boot_server(comfy_path: str):
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_path = str(BASE_PATH / "comfyui_server.log")
    log = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(["python", "main.py", "--listen", "127.0.0.1", "--port", "8188"],
                            cwd=comfy_path, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    def stream():
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
    threading.Thread(target=stream, daemon=True).start()
    print(f"[comfy] booting (log: {log_path})...", flush=True)
    t0 = time.time()
    while not is_server_running():
        if time.time() - t0 > 300:
            raise RuntimeError(f"ComfyUI failed to start. See {log_path}")
        time.sleep(2)
    print("[comfy] ready on http://127.0.0.1:8188", flush=True)
    return proc

# ----------------------------------------------------------------------
# Workflow patcher (matches notebook Cell 3 logic)
# ----------------------------------------------------------------------

PHOTOREAL_CFG = "3,2.5,2,2,1.8,1.5,1.5,1.3,1.2,1.2,1.1,1,1"
PHOTOREAL_STG = "3,2.5,2,2,1.8,1.5,1.5,1.3,1.2,1.2,1.1,1,1"
SMOKE_SIGMAS  = "1.0,0.5,0.0"
SMOKE_CFG     = "2,1.5"
SMOKE_STG     = "2,1.5"

def load_and_convert_workflow(wf_ui_path: str):
    import requests
    ui = json.loads(Path(wf_ui_path).read_text(encoding="utf-8"))
    info = requests.get(f"{COMFY_URL}/object_info", timeout=60).json()
    return ui_to_api(ui, info)

def patch_workflow(api_wf, mode, image_filepath, prompt, neg_prompt, width, height,
                   seed, omninft_strength, photoreal_cfg=True, smoke=False,
                   comfy_path: str = ""):
    input_dir = f"{comfy_path}/input"
    os.makedirs(input_dir, exist_ok=True)

    if mode == "I2V":
        if not image_filepath: raise ValueError("--source required for I2V")
        fname = os.path.basename(image_filepath)
        shutil.copy(image_filepath, os.path.join(input_dir, fname))
        for n in api_wf.values():
            if n.get("class_type") == "LoadImage": n["inputs"]["image"] = fname
    else:
        from PIL import Image
        Image.new("RGB", (width, height), "black").save(os.path.join(input_dir, "_t2v_blank.png"))
        for n in api_wf.values():
            if n.get("class_type") == "LoadImage": n["inputs"]["image"] = "_t2v_blank.png"

    # Text encoder slots
    for n in api_wf.values():
        if n.get("class_type") == "DualCLIPLoader":
            ins = n.setdefault("inputs", {})
            ins["clip_name1"] = GEMMA_FILE
            ins["clip_name2"] = LTX_TEXT_ENCODER_FILE
        if n.get("class_type") == "CLIPLoader":
            n.setdefault("inputs", {})["clip_name"] = GEMMA_FILE

    # Dimensions
    for rnid, rn in api_wf.items():
        if rn.get("class_type") not in ("ImageResizeKJv2", "ImageResizeKJ"): continue
        for role, target in (("width", int(width)), ("height", int(height))):
            ref = rn["inputs"].get(role)
            if isinstance(ref, list) and len(ref) >= 1:
                src = api_wf.get(str(ref[0]))
                if src and src.get("class_type") in ("INTConstant", "PrimitiveInt"):
                    src.setdefault("inputs", {})["value"] = target
    for n in api_wf.values():
        if n.get("class_type") == "EmptyLTXVLatentVideo":
            n.setdefault("inputs", {})["width"] = int(width)
            n["inputs"]["height"] = int(height)

    # Prompts (trace from guider -> CLIPTextEncode)
    def trace_clip_text(start, slot, hops=8):
        seen=set(); cur=api_wf.get(start,{}).get("inputs",{}).get(slot)
        while isinstance(cur,list) and cur and hops>0:
            nid=str(cur[0])
            if nid in seen: break
            seen.add(nid)
            n=api_wf.get(nid,{})
            if n.get("class_type")=="CLIPTextEncode": return nid
            cur=next((v for v in n.get("inputs",{}).values() if isinstance(v,list)),None)
            hops-=1
        return None
    guider = next((nid for nid,n in api_wf.items()
                   if n.get("class_type") in ("STGGuiderAdvanced","CFGGuider","STGGuider")), None)
    if guider:
        p = trace_clip_text(guider, "positive"); ng = trace_clip_text(guider, "negative")
        if p:  api_wf[p]["inputs"]["text"]  = prompt
        if ng: api_wf[ng]["inputs"]["text"] = neg_prompt

    # CFG / sigmas
    for n in api_wf.values():
        if n.get("class_type") == "STGGuiderAdvanced":
            ins = n.setdefault("inputs", {})
            if smoke:
                ins["sigmas"] = SMOKE_SIGMAS
                ins["cfg_values"] = SMOKE_CFG
                ins["stg_scale_values"] = SMOKE_STG
                ins["stg_rescale_values"] = "1,1"
                ins["stg_layers_indices"] = "[9999],[9999]"
            elif photoreal_cfg:
                ins["cfg_values"] = PHOTOREAL_CFG
                ins["stg_scale_values"] = PHOTOREAL_STG

    # OmniNFT strength (Vantage's own Power Lora Loader toggle)
    for n in api_wf.values():
        if n.get("class_type") == "Power Lora Loader (rgthree)":
            for v in n.get("inputs", {}).values():
                if isinstance(v, dict) and "OmniNFT" in str(v.get("lora","")):
                    v["strength"] = float(omninft_strength)
                    v["on"] = float(omninft_strength) > 0

    # Seeds
    for n in api_wf.values():
        ins = n.get("inputs", {})
        if "noise_seed" in ins: ins["noise_seed"] = int(seed)
        if "seed" in ins and isinstance(ins["seed"], (int, float)): ins["seed"] = int(seed)
    return api_wf

# ----------------------------------------------------------------------
# Submission helpers
# ----------------------------------------------------------------------

def queue_prompt(wf):
    import requests
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": wf}, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"/prompt {r.status_code}: {r.text[:800]}")
    return r.json()

def wait_for_completion(pid: str, comfy_path: str):
    while True:
        try:
            h = json.loads(urllib.request.urlopen(
                f"{COMFY_URL}/history/{pid}", timeout=10).read())
            if str(pid) in h:
                st = h[str(pid)].get("status", {})
                if st.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI rejected: {st.get('messages',[])[:3]}")
                break
            q = json.loads(urllib.request.urlopen(
                f"{COMFY_URL}/queue", timeout=10).read())
            if not any(str(j[1]) == str(pid)
                       for j in q.get("queue_running",[])+q.get("queue_pending",[])):
                raise RuntimeError("Generation crashed — see ComfyUI log")
        except urllib.error.URLError:
            pass
        time.sleep(3)
    mp4s = (glob.glob(f"{comfy_path}/output/**/*.mp4", recursive=True)
            + glob.glob(f"{comfy_path}/output/*.mp4"))
    return max(mp4s, key=os.path.getctime) if mp4s else None

# ----------------------------------------------------------------------
# Run-mode entry points
# ----------------------------------------------------------------------

def do_dry_run(args, comfy_path: str, wf_ui_path: str):
    api = load_and_convert_workflow(wf_ui_path)
    W = max(512, round(args.width / 32) * 32)
    H = max(512, round(args.height / 32) * 32)
    api = patch_workflow(api, args.mode, args.source, args.prompt, args.negative,
                         W, H, args.seed, args.omninft, args.photoreal,
                         smoke=False, comfy_path=comfy_path)
    import requests
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": api}, timeout=30)
    if r.status_code >= 400:
        print(f"VALIDATION FAILED:\n{r.text[:800]}"); return None
    pid = r.json().get("prompt_id")
    time.sleep(1)
    try:
        h = requests.get(f"{COMFY_URL}/history/{pid}", timeout=10).json()
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                print(f"VALIDATION FAILED:\n{json.dumps(st.get('messages',[])[:5], indent=2)}")
                return None
    except Exception: pass
    requests.post(f"{COMFY_URL}/interrupt", timeout=5)
    print("VALIDATION OK")
    return "OK"

def do_generate(args, comfy_path: str, wf_ui_path: str):
    smoke = args.run_mode == "smoke"
    eff_w, eff_h = (384, 512) if smoke else (args.width, args.height)
    W = max(512, round(eff_w / 32) * 32)
    H = max(512, round(eff_h / 32) * 32)
    api = load_and_convert_workflow(wf_ui_path)
    api = patch_workflow(api, args.mode, args.source, args.prompt, args.negative,
                         W, H, args.seed, args.omninft, args.photoreal,
                         smoke=smoke, comfy_path=comfy_path)
    pid = queue_prompt(api)["prompt_id"]
    print(f"[gen] prompt_id={pid} mode={args.run_mode} dims={W}x{H}", flush=True)
    print(f"[gen] sampling (~{'1-2' if smoke else '3-6'} min on L40s+)...", flush=True)
    mp4 = wait_for_completion(pid, comfy_path)
    if mp4 and args.output:
        shutil.copy(mp4, args.output)
        print(f"[gen] saved -> {args.output}")
    else:
        print(f"[gen] result at {mp4}")
    return mp4

# ----------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------

def launch_gradio(comfy_path: str, wf_ui_path: str, port: int = 7860):
    import gradio as gr

    def gen_handler(run_mode, mode, image_filepath, prompt, neg_prompt, width, height,
                    seed, omninft_strength, photoreal_cfg, progress=gr.Progress()):
        class A: pass
        args = A()
        args.run_mode = run_mode
        args.mode = "I2V" if mode == "Image-to-Video" else "T2V"
        args.source = image_filepath
        args.prompt = prompt
        args.negative = neg_prompt
        args.width = int(width); args.height = int(height)
        args.seed = int(seed); args.omninft = float(omninft_strength)
        args.photoreal = bool(photoreal_cfg)
        args.output = None
        if run_mode == "dry-run":
            progress(0.5, desc="Validating workflow...")
            ok = do_dry_run(args, comfy_path, wf_ui_path)
            progress(1.0, desc="Done")
            raise gr.Error("VALIDATION OK — try smoke or full next" if ok else "VALIDATION FAILED — see terminal")
        progress(0.1, desc=f"Sampling ({run_mode} mode)...")
        mp4 = do_generate(args, comfy_path, wf_ui_path)
        progress(1.0, desc="Done")
        return mp4

    with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
        gr.Markdown("# 10Eros LTX-2.3 — Vantage Workflow")
        gr.Markdown("""**Modes:** dry-run (30 s validation) → smoke (1-2 min, 384×512) → full (3-6 min, requested dims).

**Honest about face identity:** LTX-2.3 preserves subject category + frame-to-frame consistency well. For exact source-photo identity, train a character LoRA or use Wan 2.2.""")
        with gr.Row():
            with gr.Column(scale=1):
                run_mode_sel = gr.Radio(["dry-run", "smoke", "full"], value="dry-run",
                                        label="Run mode")
                mode_sel = gr.Radio(["Image-to-Video", "Text-to-Video"],
                                    value="Image-to-Video", label="Mode")
                img_in = gr.Image(type="filepath", label="Source portrait")
                prompt = gr.Textbox(label="Prompt",
                    value="cinematic close-up portrait, soft gentle smile forming, subtle breathing motion, photorealistic, sharp focus, natural skin texture, studio lighting",
                    lines=3)
                neg = gr.Textbox(label="Negative prompt",
                    value="anime, cartoon, drawing, illustration, painting, 3d render, cgi, stylized, blurry, distorted, deformed",
                    lines=2)
                with gr.Row():
                    w = gr.Slider(512, 1344, step=32, value=768, label="Width (full)")
                    hh = gr.Slider(512, 1344, step=32, value=1024, label="Height (full)")
                with gr.Row():
                    seed_in = gr.Number(value=42, label="Seed", precision=0)
                    omni = gr.Slider(0.0, 1.5, step=0.05, value=0.0,
                                     label="OmniNFT (0=photoreal, 0.8=Vantage anime)")
                photoreal_chk = gr.Checkbox(value=True,
                    label="Extend CFG schedule")
                gen_btn = gr.Button("Run", variant="primary")
            with gr.Column(scale=1):
                vid_out = gr.Video(label="Result")
        mode_sel.change(lambda m: gr.update(visible=(m == "Image-to-Video")),
                        inputs=mode_sel, outputs=img_in)
        gen_btn.click(gen_handler,
                      inputs=[run_mode_sel, mode_sel, img_in, prompt, neg, w, hh, seed_in, omni, photoreal_chk],
                      outputs=vid_out)
    demo.launch(share=True, inline=False, server_name="0.0.0.0", server_port=port)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Vantage 10Eros LTX-2.3 launcher.")
    p.add_argument("--setup-only", action="store_true", help="Run setup, no UI.")
    p.add_argument("--skip-setup", action="store_true", help="Skip setup, jump to UI.")
    p.add_argument("--cli", action="store_true", help="CLI single-shot mode (no Gradio).")
    p.add_argument("--gradio-port", type=int, default=7860)
    # CLI mode args
    p.add_argument("--run-mode", choices=["dry-run","smoke","full"], default="dry-run")
    p.add_argument("--mode", choices=["I2V","T2V"], default="I2V")
    p.add_argument("--source", type=str, default=None, help="I2V source image path")
    p.add_argument("--prompt", type=str,
                   default="cinematic close-up portrait, soft gentle smile, photorealistic")
    p.add_argument("--negative", type=str,
                   default="anime, cartoon, drawing, illustration, painting, blurry, distorted")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--omninft", type=float, default=0.0,
                   help="OmniNFT LoRA strength (0=photoreal, 0.8=Vantage default anime)")
    p.add_argument("--photoreal", action="store_true",
                   help="Extend CFG schedule for stronger negative prompt steering")
    p.add_argument("--output", type=str, default=None, help="Output MP4 path")
    args = p.parse_args()

    # 1. Discover or clone ComfyUI
    comfy_path = discover_comfy_path()
    if comfy_path is None:
        comfy_path = str(BASE_PATH / "ComfyUI")
        if not os.path.exists(comfy_path):
            print(f"[init] ComfyUI not found; cloning to {comfy_path}", flush=True)
            run(["git", "clone", "-q", "https://github.com/comfyanonymous/ComfyUI", comfy_path])
            run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{comfy_path}/requirements.txt"])

    # 2. Setup (unless explicitly skipped)
    if not args.skip_setup:
        setup_environment(comfy_path)

    if args.setup_only:
        print("[done] Setup complete; rerun without --setup-only to launch.")
        return

    wf_ui_path = str(BASE_PATH / "Vantage-10Eros_I2V_v3.2.json")

    # 3. Boot ComfyUI
    if not is_server_running():
        boot_server(comfy_path)
    else:
        print("[comfy] already running on 8188", flush=True)

    # 4. CLI single-shot, or Gradio UI
    if args.cli:
        if args.run_mode == "dry-run":
            do_dry_run(args, comfy_path, wf_ui_path)
        else:
            do_generate(args, comfy_path, wf_ui_path)
    else:
        launch_gradio(comfy_path, wf_ui_path, args.gradio_port)

if __name__ == "__main__":
    main()
