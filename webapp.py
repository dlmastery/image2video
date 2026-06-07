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
            # 16 GB VRAM tuning: NO --lowvram, NO --force-fp16.
            # Tried --lowvram + --force-fp16 during 10Eros debugging — sampler
            # ran at 5+ min/step because lowvram forced UNET swap on every
            # step. With text encoder auto-offloaded to CPU (~22 GB regular
            # RAM), the Q3_K_M UNET (10.4 GB) + VAE (1.4 GB) fits in 16 GB
            # cleanly and stays resident → ~5–20 s/step.
            # expandable_segments allocator prevents VRAM fragmentation
            # across the dual-stage sampler (matches the working Lightning
            # AI Colab reference for LTX-2.3).
            env = os.environ.copy()
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            self.proc = subprocess.Popen(
                [sys.executable, str(COMFY_DIR / "main.py"),
                 "--listen", COMFY_HOST, "--port", str(COMFY_PORT)],
                cwd=str(COMFY_DIR),
                stdout=open(log_path, "wb"),
                stderr=open(err_path, "wb"),
                creationflags=creationflags,
                env=env,
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

@app.get("/jobs.json")
def jobs_json():
    """Recent completed jobs for the homepage gallery."""
    items = []
    for jid, j in JOBS.items():
        if j.phase != "done":
            continue
        items.append({
            "id": jid,
            "mode": j.mode,
            "prompt": (j.prompt or "")[:80],
            "thumb": f"/job/{jid}/output",     # served as <video poster> source
            "url": f"/job/{jid}",
        })
    items.sort(key=lambda x: x["id"], reverse=True)
    from flask import jsonify
    return jsonify(items[:12])

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

@app.get("/job/<job_id>/logs.json")
def job_logs(job_id):
    """Real-time log tail of the most recent ComfyUI output lines.
    Frontend polls this to show what's happening at every step
    (model loads, text encoding, sampler step %, VAE decode, mux).
    Returns: {lines:[...], sampler:'8/13 [02:14<03:30, 26.7s/it]'}"""
    job = JOBS.get(job_id)
    if not job: abort(404)
    log_path = OUT_DIR / "comfy.err"
    lines, sampler = [], None
    if log_path.is_file():
        # Read last ~16 KB — enough for several minutes of log
        sz = log_path.stat().st_size
        with open(log_path, "rb") as f:
            if sz > 16384:
                f.seek(-16384, 2)
                f.readline()  # discard partial line
            tail = f.read().decode("utf-8", errors="replace").splitlines()
        # ANSI strip
        import re
        ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        cleaned = []
        for ln in tail:
            ln = ansi.sub("", ln).rstrip("\r").rstrip()
            if not ln: continue
            cleaned.append(ln)
        lines = cleaned[-80:]
        # Pull the most recent sampler progress line
        for ln in reversed(lines):
            m = re.search(r"(\d+/\d+ \[[0-9:]+<[0-9:]+, [0-9.]+s/it\])", ln)
            if m:
                sampler = m.group(1); break
    return jsonify(lines=lines, sampler=sampler,
                   phase=job.phase, current_step=job.current_step,
                   total_steps=job.total_steps)

@app.get("/job/<job_id>/artifacts.json")
def job_artifacts(job_id):
    """List recent generated images (intermediate frames, debug probes,
    VAE-decoded outputs) for the artifact scroller."""
    job = JOBS.get(job_id)
    if not job: abort(404)
    items = []
    try:
        # Scan ComfyUI's output PROBE/ dir + the job's own dir.
        from pathlib import Path as _P
        comfy_out = _P(__file__).parent / "comfyui" / "output"
        for sub in ("PROBE", "Eros"):
            d = comfy_out / sub
            if not d.is_dir(): continue
            for p in sorted(d.glob("*.png"),
                            key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
                items.append({
                    "name": p.name,
                    "url":  f"/artifact/{sub}/{p.name}",
                    "mtime": int(p.stat().st_mtime),
                    "size": p.stat().st_size,
                })
    except Exception as e:
        return jsonify(items=[], error=str(e))
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify(items=items[:30])

@app.get("/artifact/<sub>/<name>")
def artifact_serve(sub, name):
    """Serve a generated probe/preview image."""
    if sub not in ("PROBE", "Eros") or "/" in name or "\\" in name:
        abort(404)
    from pathlib import Path as _P
    d = _P(__file__).parent / "comfyui" / "output" / sub
    return send_from_directory(str(d), name, mimetype="image/png", conditional=True)

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
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>image2video — bring stills to life</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root { color-scheme: dark;
    --bg-0:#05060c; --bg-1:#0c0f1c; --bg-2:#13182a;
    --ink-0:#f6f8fc; --ink-1:#c5cce0; --ink-2:#8c95b0; --ink-3:#5b657d;
    --accent-1:#7a5cff; --accent-2:#3aa1ff; --accent-3:#ff5cb1;
    --good:#52d6a3; --warn:#fbbf24;
    --line:rgba(255,255,255,.07); --line-hot:rgba(122,92,255,.4);
    --panel:rgba(20,26,42,.6); --panel-solid:#13182a;
    --shadow-card: 0 30px 80px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.06);
    --shadow-glow: 0 0 40px rgba(122,92,255,.35);
    --grad-accent: linear-gradient(135deg, #7a5cff 0%, #3aa1ff 50%, #ff5cb1 100%);
    --grad-text: linear-gradient(135deg, #fff 0%, #c5cce0 50%, #7a5cff 100%);
  }
  *, *::before, *::after { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body { font-family: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
         color: var(--ink-0); background: var(--bg-0); min-height: 100vh;
         overflow-x: hidden; -webkit-font-smoothing: antialiased; }
  /* ============ Aurora animated background ============ */
  .aurora { position: fixed; inset: 0; z-index: -2; overflow: hidden; background: var(--bg-0); }
  .aurora::before, .aurora::after, .aurora .blob {
    content: ""; position: absolute; border-radius: 50%; filter: blur(80px);
    opacity: 0.55; will-change: transform; }
  .aurora::before {
    width: 600px; height: 600px; left: -150px; top: -150px;
    background: radial-gradient(circle, var(--accent-1), transparent 60%);
    animation: float1 24s ease-in-out infinite; }
  .aurora::after {
    width: 700px; height: 700px; right: -200px; top: 5%;
    background: radial-gradient(circle, var(--accent-2), transparent 60%);
    animation: float2 30s ease-in-out infinite; }
  .aurora .blob {
    width: 550px; height: 550px; left: 30%; bottom: -200px;
    background: radial-gradient(circle, var(--accent-3), transparent 60%);
    animation: float3 36s ease-in-out infinite; }
  @keyframes float1 { 0%,100% { transform: translate(0,0) scale(1); }
                      50% { transform: translate(140px,80px) scale(1.1); } }
  @keyframes float2 { 0%,100% { transform: translate(0,0) scale(1); }
                      50% { transform: translate(-120px,140px) scale(1.05); } }
  @keyframes float3 { 0%,100% { transform: translate(0,0) scale(1); }
                      50% { transform: translate(80px,-100px) scale(1.15); } }
  /* film-grain overlay */
  .grain { position: fixed; inset: 0; z-index: -1; pointer-events: none;
           opacity: 0.15; mix-blend-mode: overlay;
           background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9'/></filter><rect width='200' height='200' filter='url(%23n)' opacity='.5'/></svg>"); }
  /* ============ Spinners ============ */
  .ring { width: 18px; height: 18px; position: relative; display: inline-block; }
  .ring::before, .ring::after {
    content: ""; position: absolute; inset: 0; border-radius: 50%;
    border: 2px solid transparent; }
  .ring::before { border-top-color: var(--accent-1);
    animation: spin 1.1s cubic-bezier(.5,.05,.95,.5) infinite; }
  .ring::after { border-top-color: var(--accent-2); inset: 4px;
    animation: spin 1.6s cubic-bezier(.5,.05,.95,.5) infinite reverse; }
  .ring-lg { width: 64px; height: 64px; }
  .ring-lg::after { inset: 8px; border-width: 3px; }
  .ring-lg::before { border-width: 3px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  /* Pulse-glow for active dots */
  @keyframes pulse-glow {
    0%,100% { box-shadow: 0 0 12px rgba(122,92,255,.4); }
    50%     { box-shadow: 0 0 24px rgba(122,92,255,.85); }
  }
  @keyframes blink { 50% { opacity: 0.35; } }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 48px 28px 80px; position: relative; }
  /* ============ Hero / Header ============ */
  .top { display: flex; align-items: center; justify-content: space-between;
         margin-bottom: 56px; }
  .logo { display: flex; align-items: center; gap: 12px; font-weight: 700;
          font-size: 17px; letter-spacing: -0.3px; }
  .logo .mark { width: 32px; height: 32px; border-radius: 9px;
                background: var(--grad-accent);
                display: grid; place-items: center;
                box-shadow: var(--shadow-glow); }
  .logo .mark::after { content: '▷'; color: rgba(7,10,18,0.85); font-size: 14px;
                       margin-left: 2px; transform: translateY(0px); }
  .nav-r { display: flex; gap: 24px; align-items: center; }
  .nav-r a { color: var(--ink-1); text-decoration: none; font-size: 14px;
             transition: color 0.15s; }
  .nav-r a:hover { color: var(--ink-0); }
  .hero { text-align: center; max-width: 760px; margin: 0 auto 48px; }
  .hero .pill { display: inline-flex; gap: 8px; align-items: center;
                padding: 6px 14px; border: 1px solid var(--line);
                border-radius: 999px; font-size: 12px; color: var(--ink-1);
                margin-bottom: 22px;
                background: rgba(22,27,42,0.4); backdrop-filter: blur(8px); }
  .hero .pill .dot { width: 6px; height: 6px; border-radius: 50%;
                     background: var(--good); box-shadow: 0 0 8px var(--good); }
  h1 { font-size: clamp(36px, 5.5vw, 56px); line-height: 1.05; margin: 0 0 18px;
       font-weight: 700; letter-spacing: -1.5px;
       background: var(--grad-text); -webkit-background-clip: text;
       background-clip: text; color: transparent; }
  h1 .acc { background: var(--grad-accent); -webkit-background-clip: text;
            background-clip: text; color: transparent; }
  .hero p { font-size: 17px; color: var(--ink-1); margin: 0 auto;
            max-width: 540px; line-height: 1.55; }
  /* ============ Stepper ============ */
  .stepper { display: flex; gap: 14px; margin-bottom: 36px; max-width: 720px;
             margin-left: auto; margin-right: auto; }
  .stepper .s { flex: 1; padding: 14px 16px; background: var(--panel);
                backdrop-filter: blur(12px); border: 1px solid var(--line);
                border-radius: 14px; display: flex; gap: 12px; align-items: center;
                transition: all 0.25s cubic-bezier(.4,0,.2,1); position: relative;
                overflow: hidden; }
  .stepper .s::before {
    content: ''; position: absolute; inset: 0; border-radius: 14px;
    background: var(--grad-accent); opacity: 0; transition: opacity 0.25s;
    z-index: -1; }
  .stepper .s.active { border-color: var(--line-hot);
                       box-shadow: 0 8px 32px -8px rgba(106,166,255,0.25); }
  .stepper .s.done { border-color: rgba(52,211,153,0.35); }
  .stepper .num { width: 30px; height: 30px; border-radius: 9px;
                  background: var(--line); display: grid; place-items: center;
                  font-weight: 700; font-size: 13px; color: var(--ink-3);
                  transition: all 0.25s; }
  .stepper .active .num { background: var(--grad-accent); color: #0a0e1a;
                           box-shadow: 0 0 16px rgba(106,166,255,0.4); }
  .stepper .done .num { background: var(--good); color: #052016; }
  .stepper .done .num::after { content: '✓'; }
  .stepper .done .num span { display: none; }
  .stepper .lbl { font-size: 13px; color: var(--ink-3); font-weight: 500;
                  transition: color 0.25s; }
  .stepper .active .lbl, .stepper .done .lbl { color: var(--ink-0); }
  /* ============ Honesty notice ============ */
  .notice { background: rgba(251,191,36,0.06); border: 1px solid rgba(251,191,36,0.18);
            padding: 14px 18px; border-radius: 12px; font-size: 13px;
            color: #d4c08a; margin: 0 auto 32px; max-width: 720px; line-height: 1.6;
            backdrop-filter: blur(8px); }
  .notice strong { color: var(--warn); font-weight: 600; }
  /* ============ Step pages ============ */
  .step { display: none; max-width: 880px; margin: 0 auto;
          animation: rise 0.45s cubic-bezier(.4,0,.2,1); }
  .step.active { display: block; }
  @keyframes rise { from { opacity: 0; transform: translateY(12px); }
                    to { opacity: 1; transform: translateY(0); } }
  h2 { font-size: 24px; margin: 0 0 8px; font-weight: 600; letter-spacing: -0.4px;
       text-align: center; }
  .step-sub { color: var(--ink-1); margin: 0 0 32px; text-align: center;
              font-size: 14px; }
  /* ============ Cards ============ */
  .grid { display: grid; gap: 14px;
          grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
  .card { background: var(--panel); backdrop-filter: blur(12px);
          border: 1px solid var(--line); border-radius: 16px;
          padding: 22px 20px; cursor: pointer; position: relative;
          transition: all 0.25s cubic-bezier(.4,0,.2,1);
          overflow: hidden; }
  .card::before {
    content: ''; position: absolute; inset: 0; border-radius: 16px; padding: 1px;
    background: var(--grad-accent);
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    opacity: 0; transition: opacity 0.25s; pointer-events: none; }
  .card:hover { transform: translateY(-3px); border-color: transparent;
                box-shadow: 0 20px 40px -16px rgba(0,0,0,0.4); }
  .card:hover::before { opacity: 1; }
  .card.sel { transform: translateY(-3px); border-color: transparent;
              box-shadow: var(--shadow-glow), 0 20px 40px -16px rgba(0,0,0,0.4);
              background: linear-gradient(180deg, rgba(106,166,255,0.08), rgba(167,139,250,0.04)); }
  .card.sel::before { opacity: 1; }
  .card .icon { font-size: 28px; margin-bottom: 12px; display: inline-block;
                line-height: 1; }
  .card .name { font-weight: 600; font-size: 16px; margin-bottom: 6px;
                letter-spacing: -0.2px; }
  .card .desc { color: var(--ink-1); font-size: 13px; line-height: 1.55; }
  /* ============ Form ============ */
  .form-card { background: var(--panel); backdrop-filter: blur(12px);
               border: 1px solid var(--line); border-radius: 18px;
               padding: 28px; box-shadow: var(--shadow-card); }
  label { display: block; margin: 18px 0 7px; color: var(--ink-1);
          font-size: 13px; font-weight: 500; letter-spacing: -0.1px; }
  label:first-child { margin-top: 0; }
  textarea, input[type=text], input[type=number], input[type=file] {
    width: 100%; padding: 11px 14px; background: rgba(7,10,18,0.5);
    color: var(--ink-0); border: 1px solid var(--line); border-radius: 10px;
    font: 14px/1.5 inherit; font-family: inherit;
    transition: border-color 0.15s, background 0.15s; }
  textarea:focus, input:focus { outline: none; border-color: var(--accent-1);
                                 background: rgba(7,10,18,0.7); }
  textarea { min-height: 96px; resize: vertical; }
  input[type=file] { padding: 9px; cursor: pointer; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; }
  .row > * { flex: 1; min-width: 140px; }
  .ck { display: flex; gap: 10px; align-items: center; margin: 18px 0; }
  .ck input { width: auto; accent-color: var(--accent-1); }
  .ck label { margin: 0; cursor: pointer; }
  details { margin-top: 22px; }
  summary { cursor: pointer; color: var(--ink-1); padding: 10px 14px;
            font-size: 13px; user-select: none; font-weight: 500;
            border: 1px solid var(--line); border-radius: 10px;
            transition: all 0.15s; list-style: none; display: flex;
            align-items: center; gap: 8px; }
  summary::-webkit-details-marker { display: none; }
  summary::before { content: '⚙'; font-size: 13px; opacity: 0.7; }
  summary:hover { color: var(--ink-0); border-color: var(--accent-1); }
  /* ============ Nav buttons ============ */
  .nav { display: flex; justify-content: space-between; margin-top: 32px;
         gap: 12px; }
  button { padding: 12px 26px; background: var(--grad-accent); color: #0a0e1a;
           border: 0; border-radius: 10px; font: 600 14px inherit; cursor: pointer;
           font-family: inherit; letter-spacing: -0.1px;
           transition: all 0.2s cubic-bezier(.4,0,.2,1);
           box-shadow: 0 8px 24px -8px rgba(106,166,255,0.4); }
  button:hover { transform: translateY(-1px);
                 box-shadow: 0 12px 28px -8px rgba(106,166,255,0.55); }
  button:active { transform: translateY(0); }
  button.ghost { background: transparent; color: var(--ink-1);
                 border: 1px solid var(--line); box-shadow: none; }
  button.ghost:hover { color: var(--ink-0); border-color: var(--accent-1);
                       background: rgba(106,166,255,0.04);
                       transform: translateY(-1px); box-shadow: none; }
  button:disabled { opacity: 0.35; cursor: not-allowed;
                    transform: none; box-shadow: none; }
  button:disabled:hover { transform: none; }
  /* Submit button glow pulse + spinner state */
  button.glow { position: relative; padding: 12px 30px; overflow: hidden;
                animation: btn-pulse 3.5s ease-in-out infinite; }
  @keyframes btn-pulse {
    0%, 100% { box-shadow: 0 8px 24px -8px rgba(122,92,255,0.4); }
    50%      { box-shadow: 0 12px 32px -8px rgba(122,92,255,0.7),
                            0 0 0 4px rgba(122,92,255,0.12); }
  }
  button .btn-spin { display: none; align-items: center; gap: 10px; }
  button.loading .btn-label { display: none; }
  button.loading .btn-spin { display: inline-flex; }
  button.loading { animation: none; }
  button.loading .ring::before { border-top-color: #0a0e1a; }
  button.loading .ring::after  { border-top-color: rgba(10,14,26,0.6); }
  /* ============ Summary chips on step 3 ============ */
  .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 24px;
           justify-content: center; }
  .chip { background: rgba(22,27,42,0.6); padding: 7px 14px; border-radius: 999px;
          font-size: 12px; color: var(--ink-1); border: 1px solid var(--line);
          backdrop-filter: blur(8px); letter-spacing: 0.1px; }
  .chip strong { color: var(--ink-0); font-weight: 600; }
  /* ============ Gallery (footer) ============ */
  .gallery { margin-top: 80px; padding-top: 48px;
             border-top: 1px solid var(--line); }
  .gallery h3 { font-size: 14px; color: var(--ink-1); text-transform: uppercase;
                letter-spacing: 1.5px; font-weight: 600; margin: 0 0 24px;
                text-align: center; }
  .gallery-grid { display: grid; gap: 12px;
                  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); }
  .gallery-grid .slot { aspect-ratio: 9/14; background: var(--panel);
                        border-radius: 12px; border: 1px solid var(--line);
                        display: grid; place-items: center;
                        color: var(--ink-3); font-size: 12px;
                        font-family: 'Geist Mono', monospace; }
  footer { margin-top: 56px; text-align: center; color: var(--ink-3);
           font-size: 12px; line-height: 1.6; }
  footer a { color: var(--ink-1); text-decoration: none; }
  footer a:hover { color: var(--ink-0); }
  /* ============ Mobile ============ */
  @media (max-width: 720px) {
    .wrap { padding: 28px 18px 60px; }
    .top { margin-bottom: 36px; }
    .hero { margin-bottom: 36px; }
    h1 { font-size: 36px; letter-spacing: -1px; }
    .hero p { font-size: 15px; }
    .stepper { flex-direction: column; gap: 8px; }
    .stepper .s { padding: 12px; }
    .form-card { padding: 20px; }
    .nav { flex-direction: column-reverse; }
    .nav button { width: 100%; }
  }
</style>
</head><body>
<div class="aurora"><div class="blob"></div></div>
<div class="grain"></div>
<div class="wrap">
  <div class="top">
    <div class="logo">
      <span class="mark"></span>
      <span>image2video</span>
    </div>
    <div class="nav-r">
      <a href="#recent">Recent jobs</a>
      <a href="https://github.com/dlmastery" target="_blank">GitHub</a>
    </div>
  </div>

  <div class="hero">
    <div class="pill"><span class="dot"></span> Local · GPU ready</div>
    <h1>Bring stills to <span class="acc">life.</span></h1>
    <p>Drop in a portrait, pick a style, describe the motion. Cinematic short clips
       with synced audio — generated on your own GPU in minutes.</p>
  </div>

  <div class="notice">
    <strong>Heads up</strong> — LTX-2.3 preserves the <em>vibe</em> of your source
    (skin tone, hair color, jewelry, clothing, mood) and gives you beautiful
    motion, but it doesn't pixel-clone your exact face. Expect "this style of
    person doing what you described," not a strict photographic copy.
  </div>

  <div class="stepper">
    <div class="s active" data-step="1"><div class="num"><span>1</span></div><div class="lbl">Pick mode</div></div>
    <div class="s" data-step="2"><div class="num"><span>2</span></div><div class="lbl">Pick style</div></div>
    <div class="s" data-step="3"><div class="num"><span>3</span></div><div class="lbl">Describe &amp; go</div></div>
  </div>

  <!-- ============ Step 1: pick mode ============ -->
  <div class="step active" data-step="1">
    <h2>What are you making?</h2>
    <p class="step-sub">Three modes. Pick the one that fits your input.</p>
    <div class="grid">
      <div class="card" data-mode="i2v">
        <div class="icon">🎞️</div>
        <div class="name">Image → Video</div>
        <div class="desc">Animate a still photo. Best for portraits and scenes you already have.</div>
      </div>
      <div class="card" data-mode="t2v">
        <div class="icon">✨</div>
        <div class="name">Text → Video</div>
        <div class="desc">Generate from a prompt alone. Best for scenes with no reference image.</div>
      </div>
      <div class="card" data-mode="extend">
        <div class="icon">↪️</div>
        <div class="name">Extend a video</div>
        <div class="desc">Continue from a previously generated clip's last frame.</div>
      </div>
    </div>
    <div class="nav">
      <span></span>
      <button id="b1-next" disabled>Continue →</button>
    </div>
  </div>

  <!-- ============ Step 2: pick preset ============ -->
  <div class="step" data-step="2">
    <h2>Pick a style</h2>
    <p class="step-sub">Each preset sets prompts, dimensions, frames and sampling for you.</p>
    <div class="grid" id="preset-grid"></div>
    <div class="nav">
      <button class="ghost" data-back="1">← Back</button>
      <button id="b2-next" disabled>Continue →</button>
    </div>
  </div>

  <!-- ============ Step 3: prompt + image + go ============ -->
  <div class="step" data-step="3">
    <h2>Describe the scene</h2>
    <p class="step-sub">We pre-filled the prompt. Tweak it to taste — then go.</p>
    <div class="chips" id="summary-chips"></div>
    <div class="form-card">
      <form id="genForm" method="post" enctype="multipart/form-data">
        <div id="image-field-wrap">
          <label>Source image (the first frame of your video)</label>
          <input type="file" name="source" id="source-input" accept="image/*">
        </div>
        <label>Prompt — describe what you want to happen</label>
        <textarea name="prompt" id="prompt-input" placeholder=""></textarea>
        <div class="row" style="margin-top:14px;">
          <div>
            <label>Duration (seconds)</label>
            <input name="duration" id="duration-input" type="number"
                   value="2" min="1" max="10" step="1">
          </div>
          <div>
            <label>Resolution preset</label>
            <select id="res-preset" name="res_preset"
                    style="width:100%;padding:11px 14px;background:rgba(7,10,18,0.5);
                           color:var(--ink-0);border:1px solid var(--line);
                           border-radius:10px;font:14px inherit;">
              <option value="portrait">Portrait 768×1024 (fast)</option>
              <option value="square">Square 768×768</option>
              <option value="landscape">Landscape 1024×768</option>
              <option value="wide">Wide 1280×768 (slow)</option>
              <option value="vertical">Vertical 512×768 (smoke)</option>
            </select>
          </div>
        </div>
        <details>
          <summary>Advanced settings (auto-set by Duration + Resolution)</summary>
          <label>Negative prompt</label>
          <textarea name="negative" id="negative-input"></textarea>
          <div class="row">
            <div><label>Width</label><input name="width" id="width-input" type="number" value="768" readonly></div>
            <div><label>Height</label><input name="height" id="height-input" type="number" value="1024" readonly></div>
            <div><label>Frames (auto-calc'd from duration)</label><input name="frames" id="frames-input" type="number" value="49" readonly></div>
            <div><label>Steps</label><input name="steps" id="steps-input" type="number" value="13"></div>
            <div><label>Seed (0 = random)</label><input name="seed" type="number" value="0"></div>
          </div>
          <div class="ck">
            <input type="checkbox" id="enh" name="enhance" value="1" checked>
            <label for="enh">Let Qwen polish my prompt first</label>
          </div>
        </details>
        <div class="nav">
          <button type="button" class="ghost" data-back="2">← Back</button>
          <button type="submit" id="gen-btn" class="glow">
            <span class="btn-label">Generate clip →</span>
            <span class="btn-spin"><span class="ring"></span> Submitting…</span>
          </button>
        </div>
      </form>
    </div>
  </div>

  <!-- ============ Gallery / footer ============ -->
  <div class="gallery">
    <h3>Recent generations</h3>
    <div class="gallery-grid" id="recent-grid">
      <div class="slot">empty</div>
      <div class="slot">empty</div>
      <div class="slot">empty</div>
      <div class="slot">empty</div>
      <div class="slot">empty</div>
      <div class="slot">empty</div>
    </div>
  </div>

  <footer>
    <p>Running locally on your own GPU. No data leaves your machine.</p>
  </footer>
</div>

<script>
  // ============ Preset definitions ============
  // Each preset bundles prompt template, negative prompt, dims, frames, steps.
  // Pulls from the lessons we learned: 17-frame minimum for I2V due to audio
  // sync chain, photoreal CFG works better when frames are ≥49.
  const PRESETS = {
    // Default prompts are intentionally SHORT and natural — Qwen
    // enhancement (if enabled) and the negative prompt do the heavy
    // lifting. Easier for new users to edit one sentence than a paragraph.
    i2v: [
      { id: 'smile', icon: '😊', name: 'Smile',
        desc: 'Make the person smile gently.',
        prompt: 'make the person smile gently',
        negative: 'anime, cartoon, blurry, distorted',
        width: 768, height: 1024, frames: 49, steps: 13 },
      { id: 'talking', icon: '🗣️', name: 'Talking',
        desc: 'Subject talks naturally to camera.',
        prompt: 'the person speaks naturally to the camera',
        negative: 'anime, cartoon, frozen, motionless',
        width: 768, height: 1024, frames: 97, steps: 20 },
      { id: 'walk_toward', icon: '🚶', name: 'Walk towards camera',
        desc: 'Subject walks forward into frame.',
        prompt: 'the person walks towards the camera',
        negative: 'anime, cartoon, distorted, deformed',
        width: 768, height: 1024, frames: 97, steps: 20 },
      { id: 'look_around', icon: '👀', name: 'Look around',
        desc: 'Subject glances around the scene.',
        prompt: 'the person looks around the room',
        negative: 'anime, cartoon, frozen, motionless',
        width: 768, height: 1024, frames: 49, steps: 13 },
      { id: 'anime_motion', icon: '🌸', name: 'Anime Motion',
        desc: 'Stylized motion (enables OmniNFT).',
        prompt: 'anime style smooth motion',
        negative: 'blurry, distorted, photorealistic',
        width: 768, height: 1024, frames: 49, steps: 13 },
      { id: 'custom', icon: '⚙️', name: 'Custom',
        desc: 'Empty — you write the prompt.',
        prompt: '', negative: 'anime, cartoon, blurry, distorted',
        width: 768, height: 1024, frames: 49, steps: 13 },
    ],
    t2v: [
      { id: 'scene', icon: '🎬', name: 'Cinematic Scene',
        desc: 'Describe a scene; we render it cinematically.',
        prompt: 'a cinematic scene of ',
        negative: 'anime, cartoon, blurry, low quality',
        width: 1024, height: 768, frames: 97, steps: 20 },
      { id: 'nature', icon: '🌿', name: 'Nature Doc',
        desc: 'BBC Earth-style nature shot.',
        prompt: 'a slow nature documentary shot of ',
        negative: 'anime, cartoon, fast motion',
        width: 1024, height: 768, frames: 97, steps: 20 },
      { id: 'custom', icon: '⚙️', name: 'Custom',
        desc: 'Empty — you write the prompt.',
        prompt: '', negative: 'anime, cartoon, blurry, distorted',
        width: 1024, height: 768, frames: 97, steps: 20 },
    ],
    // Extend mode is special: presets come from /jobs.json (your prior
    // completed renders). Populated dynamically when step 2 opens.
    extend: [],
  };

  // ============ State ============
  let mode = null, preset = null, extendParentId = null;

  // ============ Step navigation ============
  function showStep(n) {
    document.querySelectorAll('.step').forEach(s =>
      s.classList.toggle('active', +s.dataset.step === n));
    document.querySelectorAll('.stepper .s').forEach(s => {
      const sn = +s.dataset.step;
      s.classList.toggle('active', sn === n);
      s.classList.toggle('done', sn < n);
    });
  }
  document.querySelectorAll('[data-back]').forEach(b =>
    b.onclick = () => showStep(+b.dataset.back));

  // ============ Step 1: mode select ============
  document.querySelectorAll('[data-mode]').forEach(c => c.onclick = () => {
    document.querySelectorAll('[data-mode]').forEach(x => x.classList.remove('sel'));
    c.classList.add('sel');
    mode = c.dataset.mode;
    document.getElementById('b1-next').disabled = false;
  });
  document.getElementById('b1-next').onclick = () => {
    renderPresets(); showStep(2);
  };

  // ============ Step 2: preset select (or recent-job select if Extend) ============
  function renderPresets() {
    const grid = document.getElementById('preset-grid');
    grid.innerHTML = '<div class="card" style="grid-column:1/-1;justify-self:center;border:none;background:none;cursor:default;"><span class="ring"></span></div>';
    preset = null; extendParentId = null;
    document.getElementById('b2-next').disabled = true;
    if (mode === 'extend') {
      // Fetch recent completed jobs; user picks which to extend.
      fetch('/jobs.json').then(r => r.json()).then(items => {
        grid.innerHTML = '';
        if (!items || !items.length) {
          grid.innerHTML = '<div class="card" style="grid-column:1/-1;cursor:default;">' +
            '<div class="icon">📭</div><div class="name">No completed jobs yet</div>' +
            '<div class="desc">Generate a video first (T2V or I2V) — it will appear here for extending.</div></div>';
          return;
        }
        items.forEach(it => {
          const div = document.createElement('div');
          div.className = 'card'; div.dataset.preset = it.id;
          div.style.cssText = 'padding:0;overflow:hidden;';
          div.innerHTML = `
            <video src="${it.thumb}" muted playsinline preload="metadata"
                   style="width:100%;aspect-ratio:9/14;object-fit:cover;display:block;"
                   onmouseover="this.play()" onmouseout="this.pause();this.currentTime=0"></video>
            <div style="padding:14px 16px;">
              <div class="name">${it.mode.toUpperCase()} · ${it.id.slice(0,8)}</div>
              <div class="desc">${it.prompt}…</div>
            </div>`;
          div.onclick = () => {
            document.querySelectorAll('[data-preset]').forEach(x => x.classList.remove('sel'));
            div.classList.add('sel');
            extendParentId = it.id;
            preset = { id: 'extend_'+it.id, prompt: '', negative: '',
                       width: 768, height: 1024, frames: 49, steps: 13,
                       name: 'Extend ' + it.id.slice(0,8) };
            document.getElementById('b2-next').disabled = false;
          };
          grid.appendChild(div);
        });
      }).catch(() => {
        grid.innerHTML = '<div class="card" style="cursor:default;grid-column:1/-1;">' +
          'Could not load recent jobs.</div>';
      });
      return;
    }
    // Regular style preset picker — show default prompt right on the card,
    // editable inline so users can tweak before continuing to step 3.
    grid.innerHTML = '';
    (PRESETS[mode] || []).forEach(p => {
      // Work on a copy so editing one preset doesn't mutate the original
      const pc = JSON.parse(JSON.stringify(p));
      const div = document.createElement('div');
      div.className = 'card'; div.dataset.preset = pc.id;
      div.innerHTML = `
        <div class="icon">${pc.icon}</div>
        <div class="name">${pc.name}</div>
        <div class="desc">${pc.desc}</div>
        <label style="margin:14px 0 5px;display:block;color:var(--ink-2);font-size:12px;">
          Default prompt — edit below
        </label>
        <textarea class="preset-prompt"
          style="width:100%;padding:8px 10px;background:rgba(7,10,18,0.55);
                 color:var(--ink-0);border:1px solid var(--line);border-radius:8px;
                 font:13px/1.45 inherit;font-family:inherit;min-height:60px;resize:vertical;"
          placeholder="${pc.prompt ? '' : 'Write your prompt here…'}"
          onclick="event.stopPropagation()"
          oninput="event.stopPropagation()">${pc.prompt || ''}</textarea>`;
      const textarea = div.querySelector('.preset-prompt');
      textarea.addEventListener('input', () => { pc.prompt = textarea.value; });
      div.onclick = (e) => {
        if (e.target === textarea) return;  // clicking the textarea shouldn't toggle
        document.querySelectorAll('[data-preset]').forEach(x => x.classList.remove('sel'));
        div.classList.add('sel');
        preset = pc;  // selected preset carries the (possibly edited) prompt
        document.getElementById('b2-next').disabled = false;
      };
      grid.appendChild(div);
    });
  }
  document.getElementById('b2-next').onclick = () => {
    applyPreset(); showStep(3);
  };

  // ============ Resolution presets (step 3) ============
  const RES = {
    portrait:  { w: 768,  h: 1024 },
    square:    { w: 768,  h: 768  },
    landscape: { w: 1024, h: 768  },
    wide:      { w: 1280, h: 768  },
    vertical:  { w: 512,  h: 768  },
  };
  // Vantage MathExpression: frames = ceil((ceil(s)*24)/8)*8 + 1
  function framesFromSeconds(s) {
    return Math.ceil((Math.ceil(Math.max(1, parseInt(s||'2',10))) * 24) / 8) * 8 + 1;
  }

  // ============ Step 3: apply preset to form, configure submission ============
  function applyPreset() {
    // Summary chips
    const sc = document.getElementById('summary-chips');
    sc.innerHTML =
      `<span class="chip">Mode: <strong>${mode.toUpperCase()}</strong></span>
       <span class="chip">Preset: <strong>${preset.name}</strong></span>
       <span class="chip">Dims: <strong>${preset.width}×${preset.height}</strong></span>
       <span class="chip">Frames: <strong>${preset.frames}</strong></span>`;

    // Hide/show source image based on mode (Extend uses parent's last frame).
    const wrap = document.getElementById('image-field-wrap');
    const src = document.getElementById('source-input');
    if (mode === 'i2v') { wrap.style.display = 'block'; src.required = true; }
    else                { wrap.style.display = 'none';  src.required = false; }

    // Fill form fields
    document.getElementById('prompt-input').value = preset.prompt;
    document.getElementById('prompt-input').placeholder =
      mode === 'extend'
        ? "Describe what happens next (the previous clip's last frame is the seed)…"
        : (preset.prompt ? '' : 'Describe what should happen in the video…');
    document.getElementById('negative-input').value = preset.negative;
    document.getElementById('width-input').value = preset.width;
    document.getElementById('height-input').value = preset.height;
    document.getElementById('frames-input').value = preset.frames;
    document.getElementById('steps-input').value = preset.steps;

    // Submission action routes to backend by mode
    const f = document.getElementById('genForm');
    f.action = mode === 'extend'
      ? '/extend/' + extendParentId
      : '/generate/' + mode;
  }

  // Live Duration → Frames + Resolution → Width/Height (Step 3)
  const durInput = document.getElementById('duration-input');
  const framesIn = document.getElementById('frames-input');
  const widthIn  = document.getElementById('width-input');
  const heightIn = document.getElementById('height-input');
  const resSel   = document.getElementById('res-preset');
  function syncDims() {
    if (durInput && framesIn) framesIn.value = framesFromSeconds(durInput.value);
    if (resSel && widthIn && heightIn) {
      const r = RES[resSel.value] || RES.portrait;
      widthIn.value = r.w; heightIn.value = r.h;
    }
  }
  if (durInput) durInput.addEventListener('input', syncDims);
  if (resSel)   resSel.addEventListener('change', syncDims);

  // Spinner on submit
  document.getElementById('genForm').addEventListener('submit', () => {
    syncDims();  // make sure latest values are in the hidden frames/width/height inputs
    const btn = document.getElementById('gen-btn');
    if (btn) { btn.classList.add('loading'); btn.disabled = true; }
  });

  // ============ Recent jobs gallery ============
  fetch('/jobs.json').then(r => r.json()).then(items => {
    const grid = document.getElementById('recent-grid');
    if (!items || !items.length) {
      grid.innerHTML = '<div class="slot" style="grid-column:1/-1;">' +
        'No generations yet — make one above ↑</div>';
      return;
    }
    grid.innerHTML = '';
    items.forEach(it => {
      const a = document.createElement('a');
      a.href = it.url; a.className = 'slot';
      a.style.cssText = 'overflow:hidden;position:relative;text-decoration:none;padding:0;';
      a.innerHTML = `
        <video src="${it.thumb}" muted playsinline preload="metadata"
               style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;"
               onmouseover="this.play()" onmouseout="this.pause();this.currentTime=0"></video>
        <div style="position:absolute;inset:auto 0 0 0;padding:8px 10px;
                    background:linear-gradient(180deg,transparent,rgba(7,10,18,0.85));
                    font-size:11px;color:#cdd5e6;letter-spacing:0.2px;">
          <strong style="text-transform:uppercase;font-size:10px;letter-spacing:1px;
                         color:#82b6ff;">${it.mode}</strong><br>${it.prompt}…
        </div>`;
      grid.appendChild(a);
    });
  }).catch(() => { /* gallery is optional */ });
</script>
</body></html>
"""

VIEW_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>generating · {{ job_id }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root { color-scheme: dark;
    --bg-0:#05060c; --bg-1:#0c0f1c;
    --ink-0:#f6f8fc; --ink-1:#c5cce0; --ink-2:#8c95b0; --ink-3:#5b657d;
    --accent-1:#7a5cff; --accent-2:#3aa1ff; --accent-3:#ff5cb1;
    --good:#52d6a3; --err:#ff5c87;
    --line:rgba(255,255,255,.07);
    --panel:rgba(20,26,42,.6); --shadow-card:0 30px 80px rgba(0,0,0,.45);
    --grad-accent:linear-gradient(135deg,#7a5cff 0%,#3aa1ff 50%,#ff5cb1 100%);
    --grad-text:linear-gradient(135deg,#fff 0%,#c5cce0 50%,#7a5cff 100%); }
  *,*::before,*::after { box-sizing:border-box; }
  html,body { margin:0; padding:0; }
  body { font-family:"Inter",ui-sans-serif,system-ui,sans-serif; color:var(--ink-0);
         background:var(--bg-0); min-height:100vh; -webkit-font-smoothing:antialiased; }
  /* aurora */
  .aurora { position:fixed; inset:0; z-index:-2; overflow:hidden; background:var(--bg-0); }
  .aurora::before,.aurora::after,.aurora .blob {
    content:""; position:absolute; border-radius:50%; filter:blur(80px); opacity:.5; will-change:transform; }
  .aurora::before { width:600px;height:600px;left:-150px;top:-150px;
    background:radial-gradient(circle,var(--accent-1),transparent 60%);
    animation:float1 24s ease-in-out infinite; }
  .aurora::after { width:700px;height:700px;right:-200px;top:5%;
    background:radial-gradient(circle,var(--accent-2),transparent 60%);
    animation:float2 30s ease-in-out infinite; }
  .aurora .blob { width:550px;height:550px;left:30%;bottom:-200px;
    background:radial-gradient(circle,var(--accent-3),transparent 60%);
    animation:float3 36s ease-in-out infinite; }
  @keyframes float1 { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(140px,80px) scale(1.1)} }
  @keyframes float2 { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(-120px,140px) scale(1.05)} }
  @keyframes float3 { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(80px,-100px) scale(1.15)} }
  .grain { position:fixed; inset:0; z-index:-1; pointer-events:none; opacity:.15;
    mix-blend-mode:overlay;
    background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9'/></filter><rect width='200' height='200' filter='url(%23n)' opacity='.5'/></svg>"); }

  /* header */
  header.top { position:sticky; top:0; z-index:10; padding:1rem 1.5rem;
    display:flex; align-items:center; justify-content:space-between;
    border-bottom:1px solid var(--line); backdrop-filter:blur(14px);
    background:rgba(5,6,12,0.65); }
  .brand { display:flex; align-items:center; gap:.6rem; font-weight:700;
           text-decoration:none; color:var(--ink-0); }
  .brand .dot { width:10px; height:10px; border-radius:50%;
    background:linear-gradient(135deg,var(--accent-1),var(--accent-3));
    box-shadow:0 0 18px var(--accent-1); }
  .top a { color:var(--ink-1); text-decoration:none; font-size:.9rem; }
  .top a:hover { color:var(--accent-2); }
  .jobid { font-family:"JetBrains Mono",ui-monospace,monospace; font-size:.85rem;
           color:var(--ink-2); letter-spacing:.05em; }

  main { max-width:920px; margin:0 auto; padding:3rem 1.5rem 5rem; }

  /* Stage / video container */
  .stage { width:100%; aspect-ratio:9/14; max-height:600px; background:#000;
    border-radius:24px; overflow:hidden; position:relative;
    box-shadow:var(--shadow-card); border:1px solid var(--line); }
  .stage video { width:100%; height:100%; object-fit:contain; display:none; background:#000; }
  .stage video.live { display:block; }

  /* In-progress overlay */
  .prep { position:absolute; inset:0; display:flex; flex-direction:column;
    align-items:center; justify-content:center; padding:2rem; text-align:center;
    background:radial-gradient(800px 500px at 50% 30%,rgba(122,92,255,0.18) 0%,transparent 60%); }
  .ring { width:92px; height:92px; margin-bottom:1.4rem; position:relative; }
  .ring::before,.ring::after { content:""; position:absolute; inset:0; border-radius:50%;
    border:3px solid transparent; }
  .ring::before { border-top-color:var(--accent-1);
    animation:spin 1.1s cubic-bezier(.5,.05,.95,.5) infinite; }
  .ring::after { border-top-color:var(--accent-2); inset:12px;
    animation:spin 1.6s cubic-bezier(.5,.05,.95,.5) infinite reverse; }
  @keyframes spin { to { transform:rotate(360deg); } }

  .phase-name { font-size:1.5rem; font-weight:700; letter-spacing:-.02em;
    background:var(--grad-text); -webkit-background-clip:text; background-clip:text;
    color:transparent; margin-bottom:.6rem; }
  .phase-msg { color:var(--ink-1); font-size:.95rem; margin-top:.2rem;
    max-width:520px; line-height:1.55; }
  .phase-msg.err { color:var(--err); }
  .elapsed { font-family:"JetBrains Mono",ui-monospace,monospace; font-size:.8rem;
    color:var(--ink-3); margin-top:1.2rem; letter-spacing:.06em; }

  /* Phase step pills */
  .steps { margin-top:1.6rem; display:flex; gap:.4rem; justify-content:center; flex-wrap:wrap; }
  .step { padding:.35rem .8rem; border-radius:999px;
    background:rgba(255,255,255,.04); color:var(--ink-2); font-size:.72rem;
    border:1px solid transparent; font-family:"JetBrains Mono",ui-monospace,monospace;
    transition:all .2s; letter-spacing:.04em; }
  .step.done { color:var(--good); border-color:rgba(82,214,163,.3);
    background:rgba(82,214,163,.08); }
  .step.active { color:var(--ink-0); border-color:var(--accent-1);
    background:rgba(122,92,255,.15); box-shadow:0 0 24px rgba(122,92,255,.25); }

  /* Progress bar */
  .progress-wrap { width:100%; padding:1.6rem 1.4rem 0; }
  .progress { width:100%; height:6px; background:rgba(255,255,255,.06);
    border-radius:3px; overflow:hidden; }
  .progress > div { height:100%; background:var(--grad-accent); width:0;
    transition:width .3s; border-radius:3px;
    box-shadow:0 0 12px rgba(122,92,255,.5); }
  .progress-meta { display:flex; justify-content:space-between; margin-top:.5rem;
    color:var(--ink-3); font-size:.78rem;
    font-family:"JetBrains Mono",ui-monospace,monospace; }

  /* Done state */
  #player-wrap { display:none; }
  #player-wrap.show { display:block; }
  .actions { display:flex; gap:.7rem; margin-top:1.4rem; flex-wrap:wrap; }
  .btn { padding:.7rem 1.4rem; background:var(--grad-accent); color:#0a0e1a;
    border:0; border-radius:10px; font:600 .9rem inherit; cursor:pointer;
    text-decoration:none; display:inline-flex; align-items:center; gap:.5rem;
    transition:all .15s;
    box-shadow:0 8px 24px -8px rgba(122,92,255,.45); }
  .btn:hover { transform:translateY(-1px);
    box-shadow:0 12px 32px -8px rgba(122,92,255,.6); }
  .btn.ghost { background:transparent; color:var(--ink-1);
    border:1px solid var(--line); box-shadow:none; }
  .btn.ghost:hover { color:var(--ink-0); border-color:var(--accent-1);
    background:rgba(122,92,255,.04); }

  details { margin-top:1.8rem; }
  summary { cursor:pointer; color:var(--ink-1); padding:.7rem 1rem;
    border:1px solid var(--line); border-radius:10px; font-size:.85rem;
    user-select:none; list-style:none; }
  summary::-webkit-details-marker { display:none; }
  summary:hover { color:var(--ink-0); border-color:var(--accent-1); }
  .panel { background:var(--panel); border:1px solid var(--line);
    border-radius:14px; padding:1.4rem; margin-top:.7rem;
    backdrop-filter:blur(10px); }
  .panel label { display:block; margin:.7rem 0 .3rem; color:var(--ink-2); font-size:.8rem; }
  .panel label:first-child { margin-top:0; }
  textarea, .panel input[type=number] {
    width:100%; padding:.7rem .9rem; background:rgba(7,10,18,.5); color:var(--ink-0);
    border:1px solid var(--line); border-radius:8px; font:14px/1.5 inherit;
    font-family:inherit; }
  textarea:focus, input:focus { outline:none; border-color:var(--accent-1); }
  textarea { min-height:80px; resize:vertical; }
  .row { display:flex; gap:.7rem; flex-wrap:wrap; }
  .row > * { flex:1; min-width:120px; }
  .ck { display:flex; gap:.5rem; align-items:center; margin:.7rem 0; }
  .ck input { width:auto; accent-color:var(--accent-1); }
  .ck label { margin:0; cursor:pointer; }

  /* Prompt details box */
  .prompts p { color:var(--ink-1); font-size:.85rem; line-height:1.5;
    margin:.6rem 0; }
  .prompts b { color:var(--accent-2); font-weight:500; margin-right:.4rem; }
</style>
</head><body>
<div class="aurora"><div class="blob"></div></div>
<div class="grain"></div>

<header class="top">
  <a href="/" class="brand"><span class="dot"></span> image2video</a>
  <span class="jobid">{{ job_id }}</span>
</header>

<main>
  <div class="stage" id="stage">
    <video id="player" controls playsinline></video>
    <div class="prep" id="prep">
      <div class="ring"></div>
      <div class="phase-name" id="phase-name">Queued</div>
      <div class="phase-msg" id="phase-msg">Waiting for an available worker…</div>
      <div class="steps" id="steps">
        <span class="step" data-phase="queued">queued</span>
        <span class="step" data-phase="enhancing">enhance prompt</span>
        <span class="step" data-phase="loading_models">load models</span>
        <span class="step" data-phase="encoding_text">encode text</span>
        <span class="step" data-phase="sampling">sample</span>
        <span class="step" data-phase="decoding">decode</span>
        <span class="step" data-phase="muxing">mux audio</span>
        <span class="step" data-phase="done">done</span>
      </div>
      <div class="elapsed" id="elapsed">0:00 elapsed</div>
    </div>
  </div>

  <div class="progress-wrap">
    <div class="progress"><div id="bar"></div></div>
    <div class="progress-meta">
      <span id="pct-label">0%</span>
      <span id="step-label">step 0 / 0</span>
    </div>
  </div>

  <details id="log-details" open style="margin-top:1.4rem;">
    <summary style="display:flex;justify-content:space-between;align-items:center;">
      <span>Live log — what ComfyUI is doing right now</span>
      <span class="ck" style="margin:0;gap:.4rem;" onclick="event.stopPropagation()">
        <input type="checkbox" id="log-on" checked
               onchange="event.stopPropagation();toggleLogs(this.checked)">
        <label for="log-on" style="font-size:.78rem;color:var(--ink-2);
          font-family:'JetBrains Mono',ui-monospace,monospace;">stream</label>
      </span>
    </summary>
    <div class="panel" style="padding:0;background:rgba(0,0,0,.45);">
      <pre id="loglines" style="margin:0;padding:1rem;max-height:260px;overflow:auto;
            font:12px/1.55 'JetBrains Mono',ui-monospace,monospace;color:var(--ink-1);
            white-space:pre-wrap;word-break:break-word;"></pre>
    </div>
  </details>

  <details id="art-details" open style="margin-top:.7rem;">
    <summary>Generated artifacts — frames + probes as they appear</summary>
    <div class="panel" style="padding:.8rem;">
      <div id="art-grid" style="display:grid;gap:.5rem;
            grid-template-columns:repeat(auto-fill,minmax(120px,1fr));
            max-height:260px;overflow-y:auto;">
        <div style="grid-column:1/-1;color:var(--ink-3);font-size:.8rem;text-align:center;padding:1rem;">
          Waiting for first artifact…
        </div>
      </div>
    </div>
  </details>

  <div id="player-wrap">
    <div class="actions">
      <a class="btn" id="dl" href="" download>↓ Download MP4</a>
      <a class="btn ghost" href="/">+ New generation</a>
    </div>

    <details open>
      <summary>Extend this video — continue from the last frame</summary>
      <div class="panel">
        <form method="post" id="ext-form">
          <label>What happens next?</label>
          <textarea name="prompt" placeholder="The next scene continues with the same motion and lighting…"></textarea>
          <div class="row">
            <div><label>Duration (sec)</label><input name="duration" type="number" value="2" min="1" max="10" id="ext-dur"></div>
            <div><label>Frames (auto)</label><input name="frames" type="number" value="49" id="ext-frames" readonly></div>
            <div><label>Steps</label><input name="steps" type="number" value="13"></div>
          </div>
          <div class="ck">
            <input type="checkbox" id="enh-ext" name="enhance" value="1" checked>
            <label for="enh-ext">Enhance continuation with Qwen</label>
          </div>
          <div class="actions"><button class="btn" type="submit">Extend ↪</button></div>
        </form>
      </div>
    </details>

    <details>
      <summary>Prompt details</summary>
      <div class="panel prompts">
        <p><b>Original:</b> <span id="p-orig"></span></p>
        <p><b>Enhanced:</b> <span id="p-enh"></span></p>
      </div>
    </details>
  </div>
</main>

<script>
  const JOB = "{{ job_id }}";
  const $ = id => document.getElementById(id);
  $('ext-form').action = `/extend/${JOB}`;

  // Auto-update Extend's frames from duration (24 fps -> multiples of 8 + 1)
  const extDur = $('ext-dur'), extFrames = $('ext-frames');
  function syncExtFrames() {
    const sec = Math.max(1, parseInt(extDur.value || '2', 10));
    // Vantage MathExpression: frames = ceil((ceil(s)*24)/8)*8 + 1
    extFrames.value = Math.ceil((Math.ceil(sec) * 24) / 8) * 8 + 1;
  }
  if (extDur) { extDur.addEventListener('input', syncExtFrames); syncExtFrames(); }

  // Human-friendly phase names + descriptions
  const PHASE_INFO = {
    queued:          { name: 'Queued',           msg: 'Waiting for an available worker…' },
    enhancing:       { name: 'Enhancing prompt', msg: 'Qwen is polishing your prompt into a richer description…' },
    loading_models:  { name: 'Loading models',   msg: 'Pulling UNET + text encoders + VAE into VRAM (~30 s)…' },
    encoding_text:   { name: 'Encoding prompt',  msg: 'Running Gemma + LTX text encoders on your prompt…' },
    encoding_image:  { name: 'Encoding image',   msg: 'VAE-encoding your source image into the video latent…' },
    sampling:        { name: 'Sampling',         msg: 'Denoising the video latent through the diffusion sampler…' },
    upscaling:       { name: 'Upscaling',        msg: 'Latent 2× spatial upscale before refinement pass…' },
    decoding:        { name: 'Decoding',         msg: 'VAE-decoding the latent into pixel frames…' },
    muxing:          { name: 'Muxing audio',     msg: 'Combining frames + audio into the final MP4…' },
    done:            { name: 'Done',             msg: 'Your clip is ready below.' },
    error:           { name: 'Error',            msg: 'Something went wrong.' },
  };

  const t0 = Date.now();
  function fmtElapsed(s) {
    const m = Math.floor(s / 60); const r = s % 60;
    return `${m}:${String(r).padStart(2,'0')} elapsed`;
  }

  let pollMs = 1200;
  async function tick() {
    let r;
    try { r = await fetch(`/job/${JOB}/status`).then(r => r.json()); }
    catch (e) { setTimeout(tick, pollMs); return; }

    const ph = r.phase || 'queued';
    const info = PHASE_INFO[ph] || { name: ph, msg: r.message || '' };
    $('phase-name').textContent = info.name;
    $('phase-msg').textContent = r.message || info.msg;
    $('phase-msg').classList.toggle('err', ph === 'error');

    // Step pills: mark done up to current, active is current
    const order = Object.keys(PHASE_INFO).filter(k => k !== 'error');
    const curIdx = order.indexOf(ph);
    document.querySelectorAll('.step').forEach(s => {
      const idx = order.indexOf(s.dataset.phase);
      s.classList.remove('active', 'done');
      if (idx >= 0 && curIdx >= 0) {
        if (idx < curIdx) s.classList.add('done');
        else if (idx === curIdx) s.classList.add('active');
      }
    });

    const pct = Math.round((r.progress || 0) * 100);
    $('bar').style.width = pct + '%';
    $('pct-label').textContent = pct + '%';
    if (r.total_steps) {
      $('step-label').textContent =
        `step ${r.current_step || 0} / ${r.total_steps}`;
    } else {
      $('step-label').textContent = '';
    }
    $('elapsed').textContent = fmtElapsed(Math.floor((Date.now() - t0) / 1000));

    if (ph === 'done' && r.has_output) {
      const v = $('player');
      if (!v.src) v.src = `/job/${JOB}/output`;
      v.classList.add('live');
      $('prep').style.display = 'none';
      $('dl').href = `/job/${JOB}/download`;
      $('player-wrap').classList.add('show');
      $('p-orig').textContent = r.prompt || '(empty)';
      $('p-enh').textContent = r.enhanced_prompt || '(not enhanced)';
      return;  // stop polling
    }
    if (ph === 'error') {
      $('phase-msg').textContent = r.error || r.message || 'Generation failed.';
      // keep the spinner spinning but in red mood; user can read the error
      return;
    }
    // Adaptive poll — slow down when sampling (long-running phase)
    pollMs = (ph === 'sampling' || ph === 'upscaling') ? 2500 : 1200;
    setTimeout(tick, pollMs);
  }

  // ============ Live log tail (toggleable) ============
  const logEl = $('loglines');
  let lastLogJoin = '';
  let logsOn = (localStorage.getItem('logsOn') !== '0');  // default ON
  $('log-on').checked = logsOn;
  function toggleLogs(on) {
    logsOn = on;
    localStorage.setItem('logsOn', on ? '1' : '0');
    if (!on) logEl.innerHTML = '<span style="color:var(--ink-3)">' +
      'log streaming paused — toggle the checkbox above to re-enable</span>';
  }
  window.toggleLogs = toggleLogs;
  async function tickLogs() {
    if (!logsOn) { setTimeout(tickLogs, 2000); return; }
    try {
      const r = await fetch(`/job/${JOB}/logs.json`).then(r => r.json());
      const lines = (r.lines || []).map(ln => {
        if (/error|RuntimeError|OOM|out of memory/i.test(ln))
          return `<span style="color:var(--err)">${ln}</span>`;
        if (/\d+\/\d+ \[\d+:\d+&lt;\d+:\d+/.test(ln) || /sampler|sampling/i.test(ln))
          return `<span style="color:var(--accent-2)">${ln}</span>`;
        if (/INFO|loaded|warmup/i.test(ln))
          return `<span style="color:var(--ink-2)">${ln}</span>`;
        return ln;
      }).join('\n');
      if (lines !== lastLogJoin) {
        const wasAtBottom = (logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 20);
        logEl.innerHTML = lines;
        if (wasAtBottom) logEl.scrollTop = logEl.scrollHeight;
        lastLogJoin = lines;
      }
    } catch (e) { /* ignore */ }
    setTimeout(tickLogs, 2000);
  }

  // ============ Artifact gallery (frames/probes as they appear) ============
  const artEl = $('art-grid');
  let lastArtKey = '';
  async function tickArtifacts() {
    try {
      const r = await fetch(`/job/${JOB}/artifacts.json`).then(r => r.json());
      const items = r.items || [];
      const key = items.map(it => it.name + ':' + it.mtime).join('|');
      if (key !== lastArtKey) {
        if (!items.length) {
          artEl.innerHTML = '<div style="grid-column:1/-1;color:var(--ink-3);' +
            'font-size:.8rem;text-align:center;padding:1rem;">Waiting for first artifact…</div>';
        } else {
          artEl.innerHTML = items.map(it => {
            const t = new Date(it.mtime * 1000).toLocaleTimeString();
            return `<a href="${it.url}" target="_blank"
              style="display:block;border:1px solid var(--line);border-radius:8px;
                     overflow:hidden;background:#000;text-decoration:none;
                     transition:transform .15s;"
              onmouseover="this.style.transform='translateY(-2px) scale(1.02)';
                           this.style.borderColor='var(--accent-1)';"
              onmouseout="this.style.transform='none';
                          this.style.borderColor='var(--line)';">
              <img src="${it.url}" style="width:100%;display:block;aspect-ratio:3/4;object-fit:cover;">
              <div style="padding:4px 6px;background:rgba(0,0,0,.5);">
                <div style="font:9px 'JetBrains Mono',monospace;color:var(--ink-2);
                            overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                  ${it.name}</div>
                <div style="font:9px 'JetBrains Mono',monospace;color:var(--ink-3);">${t}</div>
              </div></a>`;
          }).join('');
        }
        lastArtKey = key;
      }
    } catch (e) { /* ignore */ }
    setTimeout(tickArtifacts, 3000);
  }

  tick();
  tickLogs();
  tickArtifacts();
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
