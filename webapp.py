"""image2video — Flask web app for local Sulphur-2 video generation.

This is the front door. It:
  - spawns ComfyUI as a child process on :8188 at startup
  - exposes a 3-tab UI on :8080 (T2V / I2V / Extend)
  - submits Sulphur-2 workflow JSONs to ComfyUI's REST API,
    patched dynamically by class_type (not node ID)
  - streams generation progress to the browser via /status polling
  - copies the resulting MP4 into image2video_jobs/<id>/output.mp4
  - optionally enhances the user's prompt with a local Qwen-2.5-7B
    (4-bit, lazy-loaded so it doesn't compete with Sulphur-2 for VRAM)

Run with:
    conda run -n img2vid python webapp.py
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# cuDNN DLL discovery patch — MUST run before torch is imported.
# Python 3.8+ on Windows ignores PATH for native imports; the
# os.add_dll_directory cookies must stay alive in a module-level list.
# ---------------------------------------------------------------------------
import os, sys
_dll_cookies = []
if sys.platform == "win32":
    _sp = os.path.join(sys.prefix, "Lib", "site-packages")
    for _sub in ("cudnn", "cublas", "cuda_runtime", "curand", "cufft",
                 "cuda_nvrtc", "nvjitlink"):
        _bin = os.path.join(_sp, "nvidia", _sub, "bin")
        if os.path.isdir(_bin):
            _dll_cookies.append(os.add_dll_directory(_bin))
            os.environ["PATH"] = _bin + os.pathsep + os.environ["PATH"]

# ---------------------------------------------------------------------------
import atexit
import io
import json
import queue
import shutil
import signal
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import requests
import websocket  # websocket-client
from flask import (
    Flask, abort, jsonify, redirect, render_template_string,
    request, send_from_directory, url_for,
)

# ===========================================================================
# Config
# ===========================================================================
HERE          = Path(__file__).resolve().parent
COMFY_DIR     = HERE / "comfyui"
WORKFLOWS_DIR = HERE / "workflows"
JOBS_DIR      = HERE / "image2video_jobs"
MODELS_DIR    = HERE / "models"
OUT_DIR       = HERE / "out"
QWEN_PATH     = MODELS_DIR / "qwen2.5-7b-instruct"

COMFY_HOST    = os.getenv("IMG2VID_COMFY_HOST", "127.0.0.1")
COMFY_PORT    = int(os.getenv("IMG2VID_COMFY_PORT", "8188"))
COMFY_URL     = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_WS_URL  = f"ws://{COMFY_HOST}:{COMFY_PORT}/ws"

WEB_PORT      = int(os.getenv("IMG2VID_PORT", "8080"))

CLIENT_ID     = str(uuid.uuid4())  # identifies our WS connection to ComfyUI

# NOTE: per-family file names are configured below in the MODEL_FAMILY
# block. The SULPHUR_* legacy names are kept as aliases for the patcher.

# GGUF-mode toggle. When IMG2VID_USE_GGUF=1, the patcher rewrites the
# unified CheckpointLoaderSimple into UnetLoaderGGUF + VAELoader so the
# UNET gets loaded from a Q-quantized GGUF instead of the 29 GB FP8
# safetensors. Cuts Windows commit budget by 14-19 GB depending on the
# quant level, lifting the pagefile blocker.
USE_GGUF = os.getenv("IMG2VID_USE_GGUF", "1") == "1"

# Model family selector. "sulphur" uses Sulphur-2-base files; "10eros"
# uses LTX2.3-10Eros files (different fine-tune with better face
# consistency via TenStrip's forward-hook nodes).
MODEL_FAMILY = os.getenv("IMG2VID_MODEL_FAMILY", "10eros").lower()

# Per-family file name table. The patcher reads from this when rewriting
# loader inputs - the LTXVideo / KJ / 10S loaders all hardcode whatever
# filenames the original workflow author had; we overwrite them with
# what we actually ship.
SULPHUR_GGUF_QUANT = os.getenv("IMG2VID_GGUF_QUANT", "Q4_K_M")
_EROS_GGUF_QUANT = os.getenv("IMG2VID_GGUF_QUANT", "Q3_K_M")

if MODEL_FAMILY == "10eros":
    GGUF_NAME       = f"10Eros_v1-{_EROS_GGUF_QUANT}.gguf"
    VAE_NAME        = "ltx-2-3-22b-VAE.safetensors"
    AUDIO_VAE_NAME  = "ltx-2-3-22b-audio_vae.safetensors"
    CKPT_NAME       = "ltx-2-3-22b-text_encoder.safetensors"   # for the LTX loaders that scan checkpoints/
    # Default: gemma_3_12B_it_fp4_mixed.safetensors (9.45 GB Comfy-Org
    # quant). The real noise cause was the Power Lora Loader silently
    # dropping the OmniNFT LoRA, not the quant. Test FP4 first; if
    # output quality is still off, swap to gemma_3_12B_it.safetensors
    # (22.7 GB FP16) which is what Vantage names canonically.
    TEXT_ENCODER    = os.getenv(
        "IMG2VID_GEMMA_FILE", "gemma_3_12B_it_fp4_mixed.safetensors")
    UPSCALER_NAME   = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
else:
    GGUF_NAME       = f"sulphur_dev-{SULPHUR_GGUF_QUANT}.gguf"
    VAE_NAME        = "sulphur_vae.safetensors"
    AUDIO_VAE_NAME  = "sulphur_audio_vae.safetensors"
    CKPT_NAME       = "sulphur_text_encoder.safetensors"
    TEXT_ENCODER    = (
        "gemma-3-12b-it-orthogonal-reflection-bounded-ablation-v4-12B-fp4_mixed.safetensors")
    UPSCALER_NAME   = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

# Back-compat aliases used elsewhere in the file.
SULPHUR_GGUF_NAME = GGUF_NAME
SULPHUR_VAE_NAME = VAE_NAME
SULPHUR_AUDIO_VAE_NAME = AUDIO_VAE_NAME
SULPHUR_TEXT_ENCODER_NAME = TEXT_ENCODER
SULPHUR_UPSCALER_NAME = UPSCALER_NAME
SULPHUR_CKPT_NAME = CKPT_NAME

LOADER_REMAPS = {
    "CheckpointLoaderSimple":  {"ckpt_name": SULPHUR_CKPT_NAME},
    "CheckpointLoader":        {"ckpt_name": SULPHUR_CKPT_NAME},
    # The Vantage 10Eros workflow uses DualCLIPLoader with the language
    # encoder (Gemma) as clip_name1 and the LTX-2.3 text encoder as
    # clip_name2. Without this remap, the workflow's hardcoded filenames
    # stay (gemma_3_12B_it.safetensors) - which becomes invalid once we
    # switch to the FP4 quant. Always force our shipped filenames.
    "DualCLIPLoader":          {"clip_name1": SULPHUR_TEXT_ENCODER_NAME,
                                "clip_name2": "ltx-2-3-22b-text_encoder.safetensors"},
    "CLIPLoader":              {"clip_name": SULPHUR_TEXT_ENCODER_NAME},
    # LTXVAudioVAELoader + LTXAVTextEncoderLoader both scan
    # models/checkpoints/. Originally they'd pull their layers from the
    # unified 29 GB FP8 ckpt, but mmap-slicing that giant file under
    # commit-budget pressure caused a Windows access violation in
    # comfy.utils.load_torch_file. Switch them to the SPLIT files
    # (sulphur_audio_vae.safetensors, sulphur_text_encoder.safetensors)
    # which we copy from models/vae and models/text_encoders into
    # models/checkpoints during stage 4.
    "LTXVAudioVAELoader":      {"ckpt_name": SULPHUR_AUDIO_VAE_NAME},
    "LTXAVTextEncoderLoader":  {"ckpt_name": "sulphur_text_encoder.safetensors",
                                "text_encoder": SULPHUR_TEXT_ENCODER_NAME},
    "LatentUpscaleModelLoader": {"model_name": SULPHUR_UPSCALER_NAME},
}
# LoRA filename in the workflow -> file we actually have on disk
LORA_NAME_MAP = {
    "sulphur_final.safetensors": "sulphur_lora_rank_768.safetensors",
}

JOBS_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


# ===========================================================================
# ffmpeg discovery — prefer winget Gyan build, fall back to PATH, then to
# imageio-ffmpeg's bundled binary.
# ===========================================================================
def _find_ffmpeg() -> str:
    cand = os.getenv("FFMPEG_BIN")
    if cand and Path(cand).is_file():
        return cand
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return "ffmpeg"  # last resort; will fail loudly at runtime

FFMPEG = _find_ffmpeg()


# ===========================================================================
# Job state
# ===========================================================================
PHASES = ("queued", "enhancing", "submitting", "generating", "encoding",
          "done", "error")

@dataclass
class Job:
    id: str
    mode: str                     # "t2v" | "i2v" | "extend"
    prompt: str = ""
    enhanced_prompt: str = ""
    negative_prompt: str = ""
    width: int = 768
    height: int = 512
    frames: int = 97              # ~4s at 24fps; LTX likes multiples of 8 + 1
    steps: int = 30
    seed: int = 0                 # 0 = random
    source_image: Optional[str] = None    # path inside job dir (I2V / extend)
    parent_job_id: Optional[str] = None
    parent_output: Optional[str] = None   # path to parent MP4 (extend only)
    phase: str = "queued"
    message: str = ""
    current_step: int = 0
    total_steps: int = 0
    progress: float = 0.0
    error: str = ""
    output_path: str = ""         # absolute path to image2video_jobs/<id>/output.mp4
    created_at: float = field(default_factory=time.time)

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()

def _new_job(mode: str, **kw) -> Job:
    jid = uuid.uuid4().hex[:12]
    job = Job(id=jid, mode=mode, **kw)
    job.dir.mkdir(parents=True, exist_ok=True)
    with JOBS_LOCK:
        JOBS[jid] = job
    return job

def _set(job: Job, **kw):
    with JOBS_LOCK:
        for k, v in kw.items():
            setattr(job, k, v)
        # Persist meta.json so we survive a webapp restart (read-only).
        try:
            (job.dir / "meta.json").write_text(
                json.dumps(asdict(job), indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass


# ===========================================================================
# Workflow loading + patching — by class_type, never by node ID
# ===========================================================================
# Node-class taxonomy for the LTX-2.3 / Sulphur-2 graphs we patch.
# The actual API-format workflow shipped by Sulphur uses a two-stage
# SamplerCustomAdvanced pipeline; prompts and seeds don't live on the
# sampler directly. See SKILL.md gotcha #6 for the graph shape.
SAMPLER_CLASSES = (
    "SamplerCustomAdvanced", "SamplerCustom",
    "KSamplerAdvanced", "KSampler",
)
GUIDER_CLASSES   = (
    "CFGGuider", "BasicGuider", "DualCFGGuider",
    # 10Eros / Vantage workflow uses Skipped-Token Guider (STGGuiderAdvanced)
    # which has the same positive/negative input shape but a different
    # class name. WITHOUT this, the patcher walks zero guiders and the
    # user's prompt never gets written into CLIPTextEncode - the model
    # samples with empty conditioning, which looks like pure noise.
    "STGGuiderAdvanced", "STGGuider",
)
NOISE_CLASSES    = ("RandomNoise",)
SCHEDULER_CLASSES = ("LTXVScheduler", "BasicScheduler", "KarrasScheduler")
LOAD_IMAGE_CLASSES = ("LoadImage", "LoadImageMask")
TEXT_ENCODE_CLASSES = (
    "CLIPTextEncode", "BNK_CLIPTextEncodeAdvanced",
    "LTXAVTextEncoderEncode",
)
STRING_PRIMITIVE_CLASSES = (
    "PrimitiveStringMultiline", "PrimitiveString", "String",
    "CLIPTextEncode",  # patched directly; not really a primitive
)
VIDEO_OUT_CLASSES = (
    "SaveVideo", "CreateVideo",
    "VHS_VideoCombine", "SaveAnimatedWEBP", "SaveAnimatedPNG",
)
LATENT_CLASSES = (
    "EmptyLTXVLatentVideo", "EmptyLatentVideo",
)
I2V_INJECT_CLASSES = ("LTXVImgToVideoInplace", "LTXVImgToVideo")

def _nodes_by_class(wf: dict, classes: tuple) -> list[str]:
    return [nid for nid, n in wf.items() if n.get("class_type") in classes]

def _upstream(wf: dict, node_id: str, input_name: str) -> Optional[str]:
    """Node id that feeds `input_name` of `node_id`, or None."""
    inp = wf.get(node_id, {}).get("inputs", {}).get(input_name)
    if isinstance(inp, list) and len(inp) >= 1:
        return str(inp[0])
    return None

def _trace_back(wf: dict, start: str, target_classes: tuple, max_hops: int = 8) -> Optional[str]:
    """BFS backwards through every input of `start` looking for a node whose
    class_type is in target_classes. Returns the first match's node id."""
    seen, q = {start}, [start]
    for _ in range(max_hops):
        nxt = []
        for nid in q:
            node = wf.get(nid, {})
            if node.get("class_type") in target_classes and nid != start:
                return nid
            for v in node.get("inputs", {}).values():
                if isinstance(v, list) and len(v) >= 1:
                    u = str(v[0])
                    if u not in seen:
                        seen.add(u); nxt.append(u)
        q = nxt
        if not q:
            break
    return None

# Compat: a couple of node class_types in the shipped Sulphur workflow
# don't exist in any ComfyUI bundle we install (Sulphur was authored
# against a slightly different node set). Map them to equivalents from
# the stock ComfyUI core; preserve semantics by renaming inputs and
# injecting any required new ones.
#   ImageScaleDownBy(scale_by, images) -> ImageScaleBy(scale_by, image, upscale_method)
#   ResizeImageResolution(resolution, method, image)
#       -> ImageScaleToMaxDimension(largest_size, upscale_method, image)
NODE_COMPAT: dict[str, dict] = {
    "ImageScaleDownBy": {
        "to": "ImageScaleBy",
        "rename": {"images": "image"},
        "drop":   [],
        "inject": {"upscale_method": "nearest-exact"},
    },
    "ResizeImageResolution": {
        "to": "ImageScaleToMaxDimension",
        "rename": {"resolution": "largest_size"},
        "drop":   ["method"],
        "inject": {"upscale_method": "nearest-exact"},
    },
}

def _apply_compat(wf: dict) -> dict:
    """Mutate the workflow in place: rewrite class_types listed in
    NODE_COMPAT to their stock-Comfy equivalents."""
    for nid, node in wf.items():
        ct = node.get("class_type")
        if ct not in NODE_COMPAT:
            continue
        spec = NODE_COMPAT[ct]
        node["class_type"] = spec["to"]
        ins = node.setdefault("inputs", {})
        for old, new in spec["rename"].items():
            if old in ins:
                ins[new] = ins.pop(old)
        for k in spec["drop"]:
            ins.pop(k, None)
        for k, v in spec["inject"].items():
            ins.setdefault(k, v)
    return wf

def _apply_gguf_swap(wf: dict) -> dict:
    """Replace every CheckpointLoaderSimple with UnetLoaderGGUF +
    VAELoader and rewire every consumer.

    CheckpointLoaderSimple exposes 3 outputs (MODEL=slot 0, CLIP=slot 1,
    VAE=slot 2). GGUF only carries the UNET, so we split it:
      old [ckpt_id, 0]  (MODEL) ->  [ckpt_id + '_unet', 0]
      old [ckpt_id, 2]  (VAE)   ->  [ckpt_id + '_vae',  0]
    Sulphur's workflow doesn't read CLIP from the checkpoint (text
    encoding goes through LTXAVTextEncoderLoader instead), so slot 1
    has no consumers and we ignore it.

    Cuts mmap commit budget from ~29 GB (FP8 ckpt) to ~14-22 GB (GGUF
    quant + small VAE) and lifts the Windows pagefile blocker.
    """
    ckpt_ids = [nid for nid, n in wf.items()
                if n.get("class_type") == "CheckpointLoaderSimple"]
    if not ckpt_ids:
        return wf
    new_nodes: dict[str, dict] = {}
    remap: dict[str, dict[int, list]] = {}   # ckpt_id -> {old_slot: new_ref}
    for ckpt_id in ckpt_ids:
        unet_id = f"{ckpt_id}_unet"
        vae_id  = f"{ckpt_id}_vae"
        new_nodes[unet_id] = {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": SULPHUR_GGUF_NAME},
        }
        new_nodes[vae_id] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": SULPHUR_VAE_NAME},
        }
        remap[ckpt_id] = {
            0: [unet_id, 0],   # MODEL
            2: [vae_id, 0],    # VAE
        }
    # Rewire every consumer of [ckpt_id, slot]
    for n in wf.values():
        ins = n.get("inputs", {})
        for k, v in list(ins.items()):
            if isinstance(v, list) and len(v) >= 2 and str(v[0]) in remap:
                slot = int(v[1])
                if slot in remap[str(v[0])]:
                    ins[k] = remap[str(v[0])][slot]
    # Insert the new nodes and remove the old CheckpointLoaderSimple
    wf.update(new_nodes)
    for nid in ckpt_ids:
        wf.pop(nid, None)
    return wf

# Workflows keyed by mode ("t2v" / "i2v"). We use the SAME underlying
# Sulphur JSON for both modes - the LTXVImgToVideoInplace nodes have a
# bypass flag the patcher toggles. See _patch_workflow.
WORKFLOWS: dict[str, dict] = {}

def _is_api_format(wf: dict) -> bool:
    """API format: top-level keys are node ids -> {class_type, inputs}.
    UI editor format wraps in {"nodes":[...], "links":[...]} - reject."""
    if not isinstance(wf, dict):
        return False
    if "nodes" in wf or "links" in wf:
        return False
    # Heuristic: at least one value has class_type
    return any(isinstance(v, dict) and "class_type" in v for v in wf.values())

def _convert_ui_workflow(ui_wf: dict) -> Optional[dict]:
    """Use tools/ui_to_api.py to convert a UI-editor workflow into API
    format. Requires ComfyUI to be reachable (we read /object_info)."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ui_to_api", str(HERE / "tools" / "ui_to_api.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        info = m.fetch_object_info(COMFY_URL)
        return m.convert(ui_wf, info)
    except Exception as e:
        print(f"[webapp] UI->API conversion failed: {e}")
        return None

def _load_workflows() -> None:
    """Scan ./workflows/ for usable workflows.

    Workflow selection depends on MODEL_FAMILY:
      - 10eros -> prefer Vantage-10Eros_I2V_v3.2.json (UI format,
        we convert it at boot via ComfyUI's /object_info schema)
      - sulphur -> use ltx23_i2v distilled.json (API format)

    For both families we register the chosen workflow as BOTH t2v and
    i2v - the patcher flips LTXVImgToVideoInplace.bypass based on
    whether the job has a source image.
    """
    WORKFLOWS.clear()
    if not WORKFLOWS_DIR.exists():
        print(f"[webapp] WARNING: {WORKFLOWS_DIR} missing - run setup.ps1 first")
        return
    candidates = sorted(WORKFLOWS_DIR.rglob("*.json"))

    # Preferred filename per model family. First match wins.
    if MODEL_FAMILY == "10eros":
        preferred = ["vantage-10eros_i2v_v3.2", "10eros_i2v"]
    else:
        preferred = ["ltx23_i2v distilled"]

    def _prio(p: Path) -> int:
        n = p.name.lower()
        for i, sub in enumerate(preferred):
            if sub in n:
                return i
        return 99

    # For 10eros we ALWAYS prefer re-converting the UI source file each
    # boot, because our converter is still evolving and a cached
    # .api.json beside the source may be stale. Filter out .api.json
    # files when the matching UI source exists.
    if MODEL_FAMILY == "10eros":
        ui_stems = {p.stem for p in candidates
                    if not p.name.endswith(".api.json")}
        candidates = [p for p in candidates
                      if not (p.name.endswith(".api.json")
                              and p.stem.replace(".api", "") in ui_stems)]

    sorted_paths = sorted(candidates, key=_prio)
    chosen: Optional[tuple[Path, dict]] = None
    for p in sorted_paths:
        try:
            wf = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[webapp] skipping {p.name}: {e}")
            continue
        # API-format: use directly. UI-format: convert via ComfyUI schema.
        if _is_api_format(wf):
            print(f"[webapp] loaded API workflow: {p.name} ({len(wf)} nodes)")
            chosen = (p, wf)
            break
        else:
            # UI-format. Convert if it's a preferred file; skip otherwise.
            if _prio(p) >= 99:
                print(f"[webapp] skipping {p.name} (UI editor format)")
                continue
            print(f"[webapp] converting {p.name} (UI format) ...")
            api = _convert_ui_workflow(wf)
            if api:
                print(f"[webapp] converted to {len(api)} nodes from {p.name}")
                chosen = (p, api)
                break
            else:
                print(f"[webapp] conversion failed for {p.name}")

    if chosen is None:
        print(f"[webapp] WARNING: no usable workflow for family={MODEL_FAMILY}")
        return
    WORKFLOWS["t2v"] = chosen[1]
    WORKFLOWS["i2v"] = chosen[1]
    print(f"[webapp] family={MODEL_FAMILY} using {chosen[0].name} for both T2V and I2V")

def _resolve_text_target(wf: dict, start_ref: list, max_hops: int = 8) -> Optional[str]:
    """Walk back from a [node_id, slot] reference through combiner nodes
    (LTXVCropGuides / LTXVConditioning) until we hit a CLIPTextEncode or
    string primitive.

    Slot semantics: combiner nodes preserve slot identity - output 0 is
    the "positive" output, output 1 is the "negative" output. So when
    following back, slot 0 takes the upstream's `positive` input and
    slot 1 takes its `negative` input. Without this, positive and
    negative end up resolving to the same CLIPTextEncode (the one on
    the positive chain) and overwriting each other.
    """
    if not isinstance(start_ref, list) or len(start_ref) < 1:
        return None
    cur_id, cur_slot = str(start_ref[0]), (int(start_ref[1]) if len(start_ref) > 1 else 0)
    seen = set()
    for _ in range(max_hops):
        key = (cur_id, cur_slot)
        if key in seen: break
        seen.add(key)
        node = wf.get(cur_id)
        if not node: break
        ct = node.get("class_type", "")
        if ct in TEXT_ENCODE_CLASSES:
            return cur_id
        if ct in ("PrimitiveStringMultiline", "PrimitiveString", "String"):
            return cur_id
        inputs = node.get("inputs", {})
        # Slot 0 -> "positive", slot 1 -> "negative" (combiner convention)
        slot_name = "positive" if cur_slot == 0 else ("negative" if cur_slot == 1 else None)
        nxt_ref = inputs.get(slot_name) if slot_name else None
        # If the combiner doesn't have that named input, fall back to the
        # first connection-shaped input (handles non-combiner pass-throughs)
        if not isinstance(nxt_ref, list):
            nxt_ref = next((v for v in inputs.values()
                            if isinstance(v, list) and len(v) >= 1), None)
        if not isinstance(nxt_ref, list):
            break
        cur_id = str(nxt_ref[0])
        cur_slot = int(nxt_ref[1]) if len(nxt_ref) > 1 else 0
    return None

def _set_text(wf: dict, node_id: str, text: str) -> None:
    """Patch the right key on a CLIPTextEncode / string primitive."""
    node = wf.get(node_id)
    if not node: return
    inp = node.setdefault("inputs", {})
    for key in ("text", "value", "string"):
        if key in inp or node.get("class_type") in TEXT_ENCODE_CLASSES:
            inp[key] = text
            return
    inp["text"] = text  # last-resort

def _patch_workflow(wf: dict, job: Job) -> dict:
    """Deep-copy + patch the LTX-2.3 / Sulphur-2 workflow for this job.

    LTX uses a custom-sampler graph (SamplerCustomAdvanced + CFGGuider +
    RandomNoise + LTXVScheduler) rather than a classic KSampler. Prompts
    live on CFGGuider.positive/negative which themselves point at
    CLIPTextEncode (possibly through a PrimitiveStringMultiline). Seed
    lives on RandomNoise. Steps on LTXVScheduler. Workflows often have
    TWO of each (two-stage low-res-then-refine sampling), so we patch
    every match, not just the first.

    The same workflow handles T2V + I2V: flip
    LTXVImgToVideoInplace.bypass = True for T2V (no source image), else
    False and patch LoadImage.image.
    """
    wf = json.loads(json.dumps(wf))  # deep copy
    _apply_compat(wf)                # rewrite class_types we don't have
    if USE_GGUF:
        _apply_gguf_swap(wf)         # CheckpointLoaderSimple -> UnetGGUF + VAELoader

    pos_text = job.enhanced_prompt or job.prompt or ""
    neg_text = job.negative_prompt or ""

    # 1. Prompts via every CFGGuider in the graph.
    guiders = _nodes_by_class(wf, GUIDER_CLASSES)
    if guiders:
        for gid in guiders:
            gi = wf[gid].setdefault("inputs", {})
            for role, text in (("positive", pos_text), ("negative", neg_text)):
                ref = gi.get(role)
                if not isinstance(ref, list) or len(ref) < 1 or not text:
                    continue
                # Pass full ref (incl. slot) so the resolver can route
                # positive/negative correctly through combiner nodes.
                target = _resolve_text_target(wf, ref)
                if target:
                    _set_text(wf, target, text)
    else:
        # Fallback: no guiders found — patch every CLIPTextEncode with
        # the positive prompt (best-effort; loses negative).
        for nid in _nodes_by_class(wf, TEXT_ENCODE_CLASSES):
            _set_text(wf, nid, pos_text)

    # 2. Seed on every RandomNoise.
    seed = int(job.seed) if job.seed else int(uuid.uuid4().int & 0xFFFFFFFF)
    for nid in _nodes_by_class(wf, NOISE_CLASSES):
        wf[nid].setdefault("inputs", {})["noise_seed"] = seed

    # 3. Steps on LTXVScheduler / generic schedulers, plus any classic
    # samplers that take steps directly.
    for nid in _nodes_by_class(wf, SCHEDULER_CLASSES):
        si = wf[nid].setdefault("inputs", {})
        if "steps" in si:
            si["steps"] = int(job.steps)
    for nid in _nodes_by_class(wf, ("KSampler", "KSamplerAdvanced")):
        si = wf[nid].setdefault("inputs", {})
        if "steps" in si:
            si["steps"] = int(job.steps)

    # 4. Width / height / frame count.
    # The Vantage 10Eros workflow drives dims from INTConstants that feed
    # an ImageResizeKJv2 - that's the PRIMARY dim setting. The latent's
    # width/height inputs are CONNECTIONS to GetImageSize -> Resize, so
    # they auto-derive once we patch the resize target. Patching latents
    # with literal width/height (the old behaviour) overrides those
    # connections with literals that don't match the resized image's
    # actual dims -> sampler operates on a latent shape that has nothing
    # to do with the conditioning image, and output is colored noise.
    #
    # Correct pipeline: patch upstream INTConstants of ImageResizeKJv2.
    # If no such resize node exists (simpler workflows) fall back to
    # patching the latent directly.
    resize_nodes = _nodes_by_class(wf, ("ImageResizeKJv2", "ImageResizeKJ",
                                        "ImageScale", "ImageScaleBy"))
    patched_via_resize = False
    for nid in resize_nodes:
        ins = wf[nid].setdefault("inputs", {})
        for role, target in (("width", int(job.width)), ("height", int(job.height))):
            ref = ins.get(role)
            if isinstance(ref, list) and len(ref) >= 1:
                src_id = str(ref[0])
                src = wf.get(src_id)
                if src and src.get("class_type") in ("INTConstant", "PrimitiveInt"):
                    src.setdefault("inputs", {})["value"] = target
                    patched_via_resize = True
                else:
                    # Direct connection to something else - just overwrite
                    # the ref with a literal value.
                    ins[role] = target
                    patched_via_resize = True
            elif role in ins:
                ins[role] = target
                patched_via_resize = True

    # Latent dims: only force-write when the workflow has NO resize chain
    # (otherwise the latent's connections will derive the right values).
    if not patched_via_resize:
        for nid in _nodes_by_class(wf, LATENT_CLASSES):
            li = wf[nid].setdefault("inputs", {})
            if "width" in li and not isinstance(li["width"], list):
                li["width"]  = int(job.width)
            if "height" in li and not isinstance(li["height"], list):
                li["height"] = int(job.height)

    # Frame count is always safe to set on latents (connection-shaped
    # length inputs are rare; usually it's a literal or derived from a
    # primitive node).
    for nid in _nodes_by_class(wf, LATENT_CLASSES):
        li = wf[nid].setdefault("inputs", {})
        for k in ("length", "num_frames", "frame_count", "video_length"):
            if k in li and not isinstance(li[k], list):
                li[k] = int(job.frames)

    # 5. I2V image injection. Sulphur's workflow has
    # LTXVImgToVideoInplace nodes with a `bypass` flag - flip it to True
    # for T2V (no image), False for I2V (patch LoadImage.image).
    i2v_nodes = _nodes_by_class(wf, I2V_INJECT_CLASSES)
    load_image_nodes = _nodes_by_class(wf, LOAD_IMAGE_CLASSES)

    if job.source_image:
        # Copy upload into comfyui/input/ with a job-prefixed name
        inp_dir = COMFY_DIR / "input"
        inp_dir.mkdir(parents=True, exist_ok=True)
        src = Path(job.source_image)
        dst_name = f"{job.id}_{src.name}"
        try:
            shutil.copy2(src, inp_dir / dst_name)
        except Exception as e:
            print(f"[patch] copying source image failed: {e}")
        for nid in load_image_nodes:
            wf[nid].setdefault("inputs", {})["image"] = dst_name
        for nid in i2v_nodes:
            wf[nid].setdefault("inputs", {})["bypass"] = False
    else:
        # T2V: bypass the image-injection nodes so the latent passes
        # through unchanged. BUT - ComfyUI validates LoadImage's file
        # before executing, so we must point it at an existing file even
        # when bypassed. img2vid_placeholder.png is created by
        # out/stage2_assets.py and shipped in comfyui/input/.
        for nid in load_image_nodes:
            wf[nid].setdefault("inputs", {})["image"] = "img2vid_placeholder.png"
        for nid in i2v_nodes:
            wf[nid].setdefault("inputs", {})["bypass"] = True

    # 6. Loader filenames. Sulphur's workflow hardcodes filenames that
    # don't match what we actually ship. Override each loader's filename
    # input with what's on disk (see LOADER_REMAPS) and rewrite stale
    # lora_name references via LORA_NAME_MAP.
    for nid, node in wf.items():
        ct = node.get("class_type")
        if ct in LOADER_REMAPS:
            ins = node.setdefault("inputs", {})
            for input_name, our_file in LOADER_REMAPS[ct].items():
                ins[input_name] = our_file
        if ct in ("LoraLoaderModelOnly", "LoraLoader"):
            ins = node.setdefault("inputs", {})
            old = ins.get("lora_name")
            if old in LORA_NAME_MAP:
                ins["lora_name"] = LORA_NAME_MAP[old]

    # 6b. UNETLoader -> UnetLoaderGGUF swap (when USE_GGUF is on).
    # The Vantage 10Eros workflow ships with a stock UNETLoader pointing
    # at a BF16 safetensors AND a UnetLoaderGGUF that's bypassed by
    # default. The converter drops the bypassed node, leaving only the
    # BF16 path - which would load a 46 GB file we don't have. Swap the
    # class to GGUF and point at our quant.
    if USE_GGUF:
        for nid, node in wf.items():
            if node.get("class_type") == "UNETLoader":
                node["class_type"] = "UnetLoaderGGUF"
                ins = node.setdefault("inputs", {})
                ins["unet_name"] = GGUF_NAME
                # UNETLoader had a weight_dtype input; UnetLoaderGGUF
                # doesn't. Drop it so it doesn't leak through.
                ins.pop("weight_dtype", None)

    return wf


# ===========================================================================
# ComfyUI subprocess + client
# ===========================================================================
class ComfyClient:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()

    def start(self) -> None:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return
            if not (COMFY_DIR / "main.py").is_file():
                raise RuntimeError(
                    f"ComfyUI not found at {COMFY_DIR}. Run .\\setup.ps1 first."
                )
            log_path = OUT_DIR / "comfy.log"
            err_path = OUT_DIR / "comfy.err"
            print(f"[webapp] spawning ComfyUI on {COMFY_HOST}:{COMFY_PORT}")
            creationflags = 0
            if sys.platform == "win32":
                # CREATE_NEW_PROCESS_GROUP so taskkill /T can find the tree
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            # 16 GB VRAM tuning per the LTX-10Eros Vantage tutorial:
            #  --force-fp16  keeps sampling activations at FP16 (vs FP32
            #                default), important alongside GGUF Q3/Q4
            #                UNET weights so attention buffers fit.
            #  --lowvram     aggressively offloads models to CPU between
            #                ops. Required because the canonical
            #                gemma_3_12B_it.safetensors text encoder is
            #                22.7 GB and the Q3_K_M UNET is 11 GB; sum
            #                exceeds the 16 GB VRAM card, so they have
            #                to swap on-demand.
            self.proc = subprocess.Popen(
                [sys.executable, str(COMFY_DIR / "main.py"),
                 "--listen", COMFY_HOST, "--port", str(COMFY_PORT),
                 "--force-fp16", "--lowvram"],
                cwd=str(COMFY_DIR),
                stdout=open(log_path, "wb"),
                stderr=open(err_path, "wb"),
                creationflags=creationflags,
            )
        # Wait for ready
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                r = requests.get(f"{COMFY_URL}/system_stats", timeout=2)
                if r.status_code == 200:
                    print("[webapp] ComfyUI ready")
                    return
            except Exception:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"ComfyUI exited with code {self.proc.returncode}; "
                    f"see {OUT_DIR / 'comfy.err'}"
                )
            time.sleep(1.5)
        raise RuntimeError("ComfyUI did not become ready within 180 s")

    def stop(self) -> None:
        with self.lock:
            if not self.proc:
                return
            pid = self.proc.pid
            print(f"[webapp] stopping ComfyUI (pid={pid})")
            try:
                if sys.platform == "win32":
                    # /T kills the whole process tree, /F = force
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                   capture_output=True)
                else:
                    self.proc.terminate()
                    try: self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired: self.proc.kill()
            except Exception as e:
                print(f"[webapp] error stopping ComfyUI: {e}")
            self.proc = None

    def is_alive(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    def submit(self, workflow: dict) -> str:
        # Generous timeout because ComfyUI's /prompt endpoint may block
        # while it validates + warms its node graph on the first call.
        r = requests.post(
            f"{COMFY_URL}/prompt",
            json={"prompt": workflow, "client_id": CLIENT_ID},
            timeout=180,
        )
        if r.status_code >= 400:
            # Surface ComfyUI's validation messages instead of a bare 400.
            try:
                err = r.json()
            except Exception:
                err = r.text
            raise RuntimeError(
                f"ComfyUI /prompt {r.status_code}: {json.dumps(err)[:1500]}"
            )
        return r.json()["prompt_id"]

    def history(self, prompt_id: str) -> Optional[dict]:
        r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
        r.raise_for_status()
        return r.json().get(prompt_id)

    def get_image(self, filename: str, subfolder: str, type_: str) -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": type_}
        r = requests.get(f"{COMFY_URL}/view", params=params, timeout=60)
        r.raise_for_status()
        return r.content

COMFY = ComfyClient()


# ===========================================================================
# Qwen prompt enhancer — lazy load, unload after each call
# ===========================================================================
ENHANCE_SYSTEM = (
    "You are an expert cinematic prompt engineer for AI video generation. "
    "Given a short user idea, rewrite it as a vivid 1-2 sentence prompt "
    "(under 80 words). Include subject, action, camera movement, lighting, "
    "mood. Output ONLY the rewritten prompt — no preamble, no quotes, no "
    "explanation."
)
ENHANCE_CONT_SYSTEM = (
    "You are an expert cinematic prompt engineer. Given the previous scene "
    "description and a user's continuation idea, write a 1-2 sentence prompt "
    "(under 80 words) describing what happens next. Emphasize 'seamless "
    "continuation of the previous motion', consistent lighting and "
    "character appearance, smooth camera continuation. Output ONLY the "
    "rewritten prompt — no preamble."
)

class Enhancer:
    """Lazy wrapper around Qwen2.5-7B-Instruct (4-bit bnb).

    Loads on demand, runs the request, then frees the model and clears the
    CUDA cache so Sulphur-2 gets its VRAM back before the diffusion run.
    """
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def enhance(self, prompt: str, *, continuation: bool = False,
                previous: str = "") -> str:
        if not self.path.is_dir():
            # Graceful fallback — return the raw prompt unchanged.
            print(f"[enhance] Qwen not found at {self.path}; pass-through")
            return prompt
        with self.lock:
            import torch
            from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                      BitsAndBytesConfig)
            print("[enhance] loading Qwen…")
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            tok = AutoTokenizer.from_pretrained(str(self.path))
            model = AutoModelForCausalLM.from_pretrained(
                str(self.path),
                quantization_config=bnb,
                device_map="auto",
            )
            try:
                sys_msg = ENHANCE_CONT_SYSTEM if continuation else ENHANCE_SYSTEM
                user = (f"Previous scene: {previous}\n\nContinuation: {prompt}"
                        if continuation else f"Idea: {prompt}")
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user},
                ]
                txt = tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
                inputs = tok(txt, return_tensors="pt").to(model.device)
                out = model.generate(
                    **inputs, max_new_tokens=180, do_sample=True,
                    temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id,
                )
                gen = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
                # Strip stray quotes / "Enhanced prompt:" prefixes that LLMs add
                for prefix in ("Enhanced prompt:", "Prompt:", "Output:"):
                    if gen.lower().startswith(prefix.lower()):
                        gen = gen[len(prefix):].strip()
                return gen.strip('"').strip("'") or prompt
            finally:
                del model, tok
                import gc; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                print("[enhance] Qwen unloaded")

ENHANCER = Enhancer(QWEN_PATH)


# ===========================================================================
# Worker — single-threaded; Sulphur-2 owns the GPU exclusively
# ===========================================================================
def _ws_progress_loop(job: Job, prompt_id: str):
    """Listen on the ComfyUI WebSocket for this prompt's progress events
    and update the job. Returns when execution_success / execution_error
    arrives or the WS closes."""
    try:
        ws = websocket.create_connection(
            f"{COMFY_WS_URL}?clientId={CLIENT_ID}", timeout=10)
    except Exception as e:
        print(f"[ws] connect failed: {e}")
        return
    try:
        ws.settimeout(2.0)
        while True:
            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                # Periodic safety: also poll history in case WS missed the end
                h = COMFY.history(prompt_id)
                if h and h.get("status", {}).get("completed"):
                    return
                continue
            except Exception:
                return
            if not isinstance(msg, str):
                continue
            try:
                data = json.loads(msg)
            except Exception:
                continue
            t = data.get("type")
            d = data.get("data") or {}
            if t == "progress" and d.get("prompt_id") == prompt_id:
                _set(job,
                     current_step=int(d.get("value", 0)),
                     total_steps=int(d.get("max", job.total_steps or 1)),
                     progress=float(d.get("value", 0)) /
                              float(max(1, d.get("max", 1))))
            elif t == "executing" and d.get("prompt_id") == prompt_id:
                if d.get("node") is None:
                    return
            elif t == "execution_success" and d.get("prompt_id") == prompt_id:
                return
            elif t == "execution_error" and d.get("prompt_id") == prompt_id:
                _set(job, error=str(d.get("exception_message", "ComfyUI error")))
                return
    finally:
        try: ws.close()
        except Exception: pass

def _collect_output(job: Job, prompt_id: str) -> Optional[Path]:
    """Pull the produced video from ComfyUI's history into job.dir/output.mp4."""
    h = COMFY.history(prompt_id)
    if not h:
        return None
    outputs = h.get("outputs", {}) or {}
    # Look for video outputs first
    for node_outs in outputs.values():
        for key in ("gifs", "videos"):
            for v in node_outs.get(key, []) or []:
                data = COMFY.get_image(v["filename"], v.get("subfolder", ""),
                                       v.get("type", "output"))
                dst = job.dir / "output.mp4"
                # If the file is .webp, re-encode to mp4 below; otherwise write as-is
                if v["filename"].lower().endswith((".mp4", ".mov", ".mkv")):
                    dst.write_bytes(data)
                    return dst
                else:
                    tmp = job.dir / v["filename"]
                    tmp.write_bytes(data)
                    # Re-encode to mp4 with h264 + faststart
                    subprocess.run(
                        [FFMPEG, "-y", "-i", str(tmp),
                         "-c:v", "libx264", "-pix_fmt", "yuv420p",
                         "-movflags", "+faststart", str(dst)],
                        check=True, capture_output=True,
                    )
                    tmp.unlink(missing_ok=True)
                    return dst
        # Some output nodes (e.g. SaveVideo) report under `images` with
        # animated=True and a .mp4 filename. Handle .mp4/.mov/.mkv as
        # native videos; re-encode .webp/.gif/.png animations to MP4.
        for v in node_outs.get("images", []) or []:
            fn = v["filename"].lower()
            data = COMFY.get_image(v["filename"], v.get("subfolder", ""),
                                   v.get("type", "output"))
            if fn.endswith((".mp4", ".mov", ".mkv")):
                dst = job.dir / "output.mp4"
                dst.write_bytes(data)
                return dst
            if fn.endswith((".webp", ".gif", ".png")):
                tmp = job.dir / v["filename"]
                tmp.write_bytes(data)
                dst = job.dir / "output.mp4"
                subprocess.run(
                    [FFMPEG, "-y", "-i", str(tmp),
                     "-c:v", "libx264", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart", str(dst)],
                    check=True, capture_output=True,
                )
                tmp.unlink(missing_ok=True)
                return dst
    return None

def _concat_with_parent(job: Job, new_clip: Path) -> Path:
    """Stitch parent's MP4 with the just-generated continuation, in-place."""
    if not job.parent_output or not Path(job.parent_output).is_file():
        return new_clip
    concat_list = job.dir / "concat.txt"
    concat_list.write_text(
        f"file '{Path(job.parent_output).as_posix()}'\n"
        f"file '{new_clip.as_posix()}'\n",
        encoding="utf-8",
    )
    final = job.dir / "extended.mp4"
    # Concat-copy works when both clips share codec/timebase. ComfyUI's
    # outputs are consistent, and we re-encode .webp -> h264 the same way.
    rc = subprocess.run(
        [FFMPEG, "-y", "-f", "concat", "-safe", "0",
         "-i", str(concat_list), "-c", "copy",
         "-movflags", "+faststart", str(final)],
        capture_output=True,
    )
    if rc.returncode != 0:
        # Fall back to re-encode if concat-copy fails (codec mismatch).
        rc = subprocess.run(
            [FFMPEG, "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(final)],
            capture_output=True,
        )
        if rc.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {rc.stderr.decode(errors='replace')[:400]}")
    # Replace output.mp4 with the concatenated version
    final.replace(job.dir / "output.mp4")
    return job.dir / "output.mp4"

def _run_job(job: Job) -> None:
    try:
        # 1. Optional prompt enhancement
        if job.prompt and not job.enhanced_prompt and job.mode != "extend" \
                and getattr(job, "_do_enhance", False):
            _set(job, phase="enhancing", message="Rewriting your prompt with Qwen…")
            job.enhanced_prompt = ENHANCER.enhance(job.prompt)
        elif job.mode == "extend" and job.prompt and not job.enhanced_prompt \
                and getattr(job, "_do_enhance", False):
            _set(job, phase="enhancing", message="Rewriting continuation prompt…")
            parent = JOBS.get(job.parent_job_id) if job.parent_job_id else None
            prev = (parent.enhanced_prompt or parent.prompt) if parent else ""
            job.enhanced_prompt = ENHANCER.enhance(
                job.prompt, continuation=True, previous=prev)

        # 2. Pick + patch workflow
        _set(job, phase="submitting", message="Sending workflow to ComfyUI…")
        mode_key = "i2v" if (job.source_image or job.mode == "extend") else "t2v"
        wf = WORKFLOWS.get(mode_key)
        if not wf:
            raise RuntimeError(f"no workflow registered for mode={mode_key}")
        patched = _patch_workflow(wf, job)

        # 3. Submit + watch progress
        if not COMFY.is_alive():
            COMFY.start()
        prompt_id = COMFY.submit(patched)
        _set(job, phase="generating",
             message=f"Sampling {job.frames} frames on Sulphur-2…",
             total_steps=job.steps)

        # Run WS listener in this thread; it returns when done/error
        _ws_progress_loop(job, prompt_id)
        if job.error:
            _set(job, phase="error")
            return

        # 4. Collect output + optional stitch with parent
        _set(job, phase="encoding", message="Saving MP4…")
        out = _collect_output(job, prompt_id)
        if out is None:
            raise RuntimeError("ComfyUI finished but no video output found "
                               "in its history — check the workflow's save node")
        if job.mode == "extend" and job.parent_output:
            out = _concat_with_parent(job, out)
        _set(job, phase="done", message="Done", output_path=str(out),
             progress=1.0)
    except Exception as e:
        traceback.print_exc()
        _set(job, phase="error", error=str(e))

def _worker_loop():
    while True:
        jid = JOB_QUEUE.get()
        job = JOBS.get(jid)
        if not job:
            continue
        _run_job(job)


# ===========================================================================
# Flask app + routes
# ===========================================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 256 MB

# ---------- pages ----------
@app.get("/")
def index():
    return render_template_string(INDEX_HTML)

@app.get("/job/<job_id>")
def view_job(job_id):
    if job_id not in JOBS:
        abort(404)
    return render_template_string(VIEW_HTML, job_id=job_id)

# ---------- generate endpoints ----------
def _form_params() -> dict:
    """Pull common video params from the form (with sane defaults)."""
    f = request.form
    def _i(k, d):
        try: return int(f.get(k, d))
        except Exception: return d
    return dict(
        width  = _i("width", 768),
        height = _i("height", 512),
        frames = _i("frames", 97),
        steps  = _i("steps", 30),
        seed   = _i("seed", 0),
        negative_prompt = f.get("negative", "").strip(),
    )

@app.post("/generate/t2v")
def gen_t2v():
    prompt = request.form.get("prompt", "").strip()
    if not prompt: return ("prompt required", 400)
    job = _new_job("t2v", prompt=prompt, **_form_params())
    job._do_enhance = request.form.get("enhance") == "1"  # type: ignore
    JOB_QUEUE.put(job.id)
    return redirect(url_for("view_job", job_id=job.id))

@app.post("/generate/i2v")
def gen_i2v():
    if "source" not in request.files:
        return ("source image required", 400)
    src = request.files["source"]
    if not src or not src.filename:
        return ("source image required", 400)
    prompt = request.form.get("prompt", "").strip()
    job = _new_job("i2v", prompt=prompt, **_form_params())
    # Save the upload into the job dir; convert all uploads to PNG for
    # consistent downstream behaviour.
    src_path = job.dir / "source.png"
    raw = src.read()
    try:
        # Round-trip through OpenCV to normalize colorspace + format
        import numpy as np
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("decode failed")
        cv2.imwrite(str(src_path), img)
    except Exception:
        src_path.write_bytes(raw)
    job.source_image = str(src_path)
    job._do_enhance = request.form.get("enhance") == "1"  # type: ignore
    JOB_QUEUE.put(job.id)
    return redirect(url_for("view_job", job_id=job.id))

@app.post("/extend/<parent_id>")
def extend_job(parent_id):
    parent = JOBS.get(parent_id)
    if not parent or parent.phase != "done":
        return ("parent job not complete", 400)
    parent_out = Path(parent.output_path)
    if not parent_out.is_file():
        return ("parent output not found", 400)
    prompt = request.form.get("prompt", "").strip()

    # Extract last frame of parent into a new job dir as the I2V seed.
    job = _new_job("extend",
                   prompt=prompt,
                   parent_job_id=parent_id,
                   parent_output=str(parent_out),
                   width=parent.width, height=parent.height,
                   frames=int(request.form.get("frames", parent.frames)),
                   steps=int(request.form.get("steps", parent.steps)))
    last = job.dir / "start_frame.png"
    cap = cv2.VideoCapture(str(parent_out))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
        ok, frame = cap.read()
        if not ok or frame is None:
            return ("could not extract last frame of parent", 500)
        cv2.imwrite(str(last), frame)
    finally:
        cap.release()
    job.source_image = str(last)
    job._do_enhance = request.form.get("enhance") == "1"  # type: ignore
    JOB_QUEUE.put(job.id)
    return redirect(url_for("view_job", job_id=job.id))

# ---------- prompt enhancement (sync) ----------
@app.post("/enhance")
def enhance():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify(error="prompt required"), 400
    out = ENHANCER.enhance(prompt)
    return jsonify(enhanced=out)

# ---------- status / files ----------
@app.get("/job/<job_id>/status")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job: abort(404)
    return jsonify(
        id=job.id, mode=job.mode, phase=job.phase, message=job.message,
        current_step=job.current_step, total_steps=job.total_steps,
        progress=job.progress, error=job.error,
        has_output=Path(job.output_path).is_file() if job.output_path else False,
        prompt=job.prompt, enhanced_prompt=job.enhanced_prompt,
        parent_job_id=job.parent_job_id,
    )

@app.get("/job/<job_id>/output")
def job_output(job_id):
    job = JOBS.get(job_id)
    if not job or not job.output_path:
        abort(404)
    p = Path(job.output_path)
    if not p.is_file(): abort(404)
    return send_from_directory(str(p.parent), p.name,
                                mimetype="video/mp4", conditional=True)

@app.get("/job/<job_id>/download")
def job_download(job_id):
    job = JOBS.get(job_id)
    if not job or not job.output_path: abort(404)
    p = Path(job.output_path)
    if not p.is_file(): abort(404)
    return send_from_directory(str(p.parent), p.name,
                                mimetype="video/mp4", as_attachment=True,
                                download_name=f"{job_id}.mp4")

@app.get("/comfy/healthz")
def comfy_healthz():
    try:
        r = requests.get(f"{COMFY_URL}/system_stats", timeout=2)
        return (r.text, r.status_code, {"Content-Type": "application/json"})
    except Exception as e:
        return (str(e), 503)


# ===========================================================================
# HTML templates (inline, faceswap-style)
# ===========================================================================
INDEX_HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8"><title>image2video</title>
<style>
  :root { color-scheme: dark; }
  body { font: 15px/1.45 system-ui, sans-serif; background: #0c0f14; color: #e6e8ec;
         margin: 0; padding: 32px; }
  h1 { font-size: 24px; margin: 0 0 24px; }
  .tabs { display: flex; gap: 4px; border-bottom: 1px solid #2a2f3a; margin-bottom: 24px; }
  .tab { padding: 10px 16px; cursor: pointer; border-bottom: 2px solid transparent; }
  .tab.active { border-bottom-color: #6aa6ff; color: #fff; }
  .panel { display: none; max-width: 720px; }
  .panel.active { display: block; }
  label { display: block; margin: 12px 0 4px; color: #aab1bf; font-size: 13px; }
  textarea, input[type=text], input[type=number], input[type=file] {
    width: 100%; padding: 9px 11px; background: #161a23; color: #e6e8ec;
    border: 1px solid #2a2f3a; border-radius: 6px; font: inherit;
  }
  textarea { min-height: 90px; resize: vertical; }
  .row { display: flex; gap: 12px; }
  .row > * { flex: 1; }
  button { padding: 10px 18px; background: #6aa6ff; color: #0c0f14;
           border: 0; border-radius: 6px; font-weight: 600; cursor: pointer;
           margin-top: 18px; }
  button:hover { background: #82b6ff; }
  .checkbox { display: flex; gap: 6px; align-items: center; margin: 12px 0; }
  .checkbox input { width: auto; }
  .hint { color: #6b7280; font-size: 12px; margin-top: 4px; }
</style>
</head><body>
<h1>image2video</h1>
<div class="tabs">
  <div class="tab active" data-tab="t2v">Text → Video</div>
  <div class="tab" data-tab="i2v">Image → Video</div>
  <div class="tab" data-tab="extend">Extend Existing</div>
</div>

<form id="f-t2v" class="panel active" method="post" action="/generate/t2v">
  <label>Prompt</label>
  <textarea name="prompt" placeholder="A cinematic shot of a fox running through a foggy autumn forest at dawn, golden light, slow tracking camera"></textarea>
  <label>Negative prompt (optional)</label>
  <textarea name="negative" placeholder="blurry, low quality, watermark, distorted"></textarea>
  <div class="row">
    <div><label>Width</label><input name="width" type="number" value="768"></div>
    <div><label>Height</label><input name="height" type="number" value="512"></div>
    <div><label>Frames</label><input name="frames" type="number" value="97"></div>
    <div><label>Steps</label><input name="steps" type="number" value="30"></div>
    <div><label>Seed (0 = random)</label><input name="seed" type="number" value="0"></div>
  </div>
  <div class="checkbox">
    <input type="checkbox" id="enh-t2v" name="enhance" value="1" checked>
    <label for="enh-t2v" style="margin:0">Enhance prompt with Qwen first</label>
  </div>
  <button type="submit">Generate</button>
</form>

<form id="f-i2v" class="panel" method="post" action="/generate/i2v" enctype="multipart/form-data">
  <label>Source image (first frame)</label>
  <input type="file" name="source" accept="image/*" required>
  <label>Prompt (describes motion)</label>
  <textarea name="prompt" placeholder="The character begins walking forward, camera slowly pans right, atmospheric"></textarea>
  <label>Negative prompt (optional)</label>
  <textarea name="negative"></textarea>
  <div class="row">
    <div><label>Width</label><input name="width" type="number" value="768"></div>
    <div><label>Height</label><input name="height" type="number" value="512"></div>
    <div><label>Frames</label><input name="frames" type="number" value="97"></div>
    <div><label>Steps</label><input name="steps" type="number" value="30"></div>
  </div>
  <div class="checkbox">
    <input type="checkbox" id="enh-i2v" name="enhance" value="1" checked>
    <label for="enh-i2v" style="margin:0">Enhance prompt with Qwen first</label>
  </div>
  <button type="submit">Generate</button>
</form>

<div id="f-extend" class="panel">
  <p>To extend a video, generate one in the T2V or I2V tab first, then use
  the <strong>Extend</strong> button on its viewer page.</p>
  <p>Recent jobs:</p>
  <ul id="recent"></ul>
</div>

<script>
  document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('f-' + t.dataset.tab).classList.add('active');
  });
  // Lazy: surface recent jobs on the Extend tab if we have them in memory.
  // For MVP we just hint the user to navigate from a viewer page.
</script>
</body></html>
"""

VIEW_HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8"><title>job {{ job_id }}</title>
<style>
  :root { color-scheme: dark; }
  body { font: 15px/1.45 system-ui, sans-serif; background: #0c0f14; color: #e6e8ec;
         margin: 0; padding: 32px; max-width: 900px; }
  h1 { font-size: 20px; margin: 0 0 18px; }
  .pill { display: inline-block; padding: 2px 10px; border-radius: 999px;
          background: #1f2630; color: #aab1bf; font-size: 12px; margin-left: 8px;
          vertical-align: middle; }
  .pill.done { background: #163b1f; color: #7be38c; }
  .pill.error { background: #3b1616; color: #ff8585; }
  .progress { height: 10px; background: #161a23; border-radius: 5px; overflow: hidden;
              margin: 14px 0; }
  .progress > div { height: 100%; background: #6aa6ff;
                    transition: width .25s; width: 0%; }
  .msg { color: #aab1bf; min-height: 20px; }
  .err { color: #ff8585; }
  video { width: 100%; margin-top: 20px; border-radius: 8px; background: #000; }
  .actions { display: flex; gap: 10px; margin-top: 16px; }
  .actions a, .actions button { padding: 9px 14px; background: #6aa6ff;
                                color: #0c0f14; border: 0; border-radius: 6px;
                                font-weight: 600; cursor: pointer;
                                text-decoration: none; }
  .actions .ghost { background: #1f2630; color: #e6e8ec; }
  details { margin-top: 20px; }
  summary { cursor: pointer; color: #aab1bf; }
  textarea, input[type=text], input[type=number] {
    width: 100%; padding: 9px 11px; background: #161a23; color: #e6e8ec;
    border: 1px solid #2a2f3a; border-radius: 6px; font: inherit;
    margin-top: 6px;
  }
  .checkbox { display: flex; gap: 6px; align-items: center; margin: 10px 0; }
  .checkbox input { width: auto; }
</style>
</head><body>
<h1>Job <code id="jid">{{ job_id }}</code><span class="pill" id="phase">queued</span></h1>
<div class="progress"><div id="bar"></div></div>
<div class="msg" id="msg"></div>
<div class="err" id="err"></div>
<div id="player-wrap" style="display:none">
  <video id="player" controls playsinline></video>
  <div class="actions">
    <a id="dl" href="">Download MP4</a>
    <a class="ghost" href="/">New job</a>
  </div>
  <details>
    <summary>Extend this video</summary>
    <form method="post" id="ext-form">
      <label>Continuation prompt</label>
      <textarea name="prompt" placeholder="The next scene continues with the same motion and lighting…"></textarea>
      <label>Frames</label><input name="frames" type="number" value="97">
      <label>Steps</label><input name="steps" type="number" value="30">
      <div class="checkbox">
        <input type="checkbox" id="enh-ext" name="enhance" value="1" checked>
        <label for="enh-ext" style="margin:0">Enhance continuation with Qwen</label>
      </div>
      <div class="actions"><button type="submit">Extend</button></div>
    </form>
  </details>
  <details>
    <summary>Prompt details</summary>
    <p><b>Original:</b> <span id="p-orig"></span></p>
    <p><b>Enhanced:</b> <span id="p-enh"></span></p>
  </details>
</div>

<script>
  const JOB = "{{ job_id }}";
  const $ = id => document.getElementById(id);
  $('ext-form').action = `/extend/${JOB}`;

  async function tick() {
    let r;
    try { r = await fetch(`/job/${JOB}/status`).then(r => r.json()); }
    catch (e) { setTimeout(tick, 1500); return; }
    $('phase').textContent = r.phase;
    $('phase').className = 'pill' + (r.phase === 'done' ? ' done' :
                                     r.phase === 'error' ? ' error' : '');
    $('msg').textContent = r.message || '';
    $('err').textContent = r.error || '';
    const pct = Math.round((r.progress || 0) * 100);
    $('bar').style.width = pct + '%';
    if (r.phase === 'done' && r.has_output) {
      const v = $('player');
      if (!v.src) v.src = `/job/${JOB}/output`;
      $('dl').href = `/job/${JOB}/download`;
      $('player-wrap').style.display = 'block';
      $('p-orig').textContent = r.prompt || '';
      $('p-enh').textContent = r.enhanced_prompt || '(not enhanced)';
      return;  // stop polling once done
    }
    if (r.phase === 'error') return;
    setTimeout(tick, 1500);
  }
  tick();
</script>
</body></html>
"""


# ===========================================================================
# Lifecycle: start ComfyUI + worker thread, register cleanup, run Flask
# ===========================================================================
def _cleanup():
    try: COMFY.stop()
    except Exception: pass

def main():
    atexit.register(_cleanup)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), sys.exit(0)))

    # Spawn ComfyUI in a background thread, then load workflows once it's
    # ready (the UI->API converter needs /object_info from ComfyUI).
    def _boot_then_load():
        try:
            COMFY.start()
        except Exception as e:
            print(f"[webapp] ComfyUI failed to start: {e}")
            return
        try:
            _load_workflows()
        except Exception as e:
            print(f"[webapp] workflow loading failed: {e}")
    threading.Thread(target=_boot_then_load, daemon=True).start()

    # Worker thread
    threading.Thread(target=_worker_loop, daemon=True).start()

    print(f"[webapp] starting on http://127.0.0.1:{WEB_PORT}/")
    app.run(host="0.0.0.0", port=WEB_PORT, debug=False, use_reloader=False,
            threaded=True)

if __name__ == "__main__":
    main()
