# Install Guide

Two supported install paths:

1. **Local Windows + RTX 4090** — best for iteration, no cloud cost
2. **Lightning AI Studio (cloud GPU)** — best for L40s / A100 / H100 access

Both paths use the same Vantage 10Eros workflow and produce the same outputs.

---

## Path A — Local Windows (RTX 4090, 16 GB+ VRAM)

### Prerequisites

| Tool | Verify | If missing |
|---|---|---|
| Windows 10 build 19044+ or Windows 11 | `winver` | Update Windows |
| NVIDIA driver R535+ (CUDA 12 compatible) | `nvidia-smi` | https://www.nvidia.com/Download/index.aspx |
| Anaconda or Miniconda | `conda --version` | https://www.anaconda.com/download |
| Git | `git --version` | https://git-scm.com/download/win |
| ffmpeg (Gyan.dev build, not Anaconda's) | `ffmpeg -version` | `winget install Gyan.FFmpeg` |
| GitHub CLI (only for pushing) | `gh auth status` | https://cli.github.com/ |

Storage needed: ~35 GB (ComfyUI clone + ~24 GB of model weights + scratch space).

### One-shot install

```powershell
git clone https://github.com/dlmastery/image2video.git
cd image2video
.\setup.ps1
# expect: ~15-20 min, downloads ComfyUI + custom node packs + 10Eros models + Gemma + LoRA
```

`setup.ps1` provisions:

- conda env `img2vid` (Python 3.11)
- `comfyui/` cloned from `comfyanonymous/ComfyUI`
- 11 custom node packs under `comfyui/custom_nodes/` (10S-Comfy-nodes, ComfyUI-GGUF, ComfyUI-LTXVideo, ComfyUI-KJNodes, rgthree-comfy, ComfyUI-Custom-Scripts, ComfyUI-VideoHelperSuite, ComfyUI-MelBandRoFormer, comfy_mtb, RES4LYF, ComfyUI_Comfyroll_CustomNodes)
- All model weights into `comfyui/models/*/`:
  - 10Eros UNET Q3_K_M GGUF (~10.4 GB)
  - Gemma 3 12B text encoder (~9 GB safetensors)
  - LTX-2.3 text encoder (~2.3 GB)
  - 10Eros / LTX-2.3 video VAE + audio VAE
  - OmniNFT LoRA (anime stylizer — keep off for photoreal)
  - Distilled LoRA
  - Spatial upscaler v1.1
  - MelBand RoFormer (audio)
- `mediapipe==0.10.13` pin (critical: 10S-Comfy-nodes' `LTXFaceDetector` uses removed `mp.solutions` API)

### Run it

In one PowerShell tab — boot ComfyUI:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
conda run -n img2vid --no-capture-output python comfyui/main.py --listen 127.0.0.1 --port 8188
```

Wait ~70 s for the custom nodes to load. Open <http://127.0.0.1:8188> in a browser to verify the ComfyUI canvas appears (this also has the **🚀 Manager** button for installing more nodes).

In a second PowerShell tab — boot the webapp:

```powershell
conda run -n img2vid --no-capture-output python webapp.py
```

Then open <http://localhost:8080/>. You'll see the 3-step wizard:

1. **Pick mode** — Image-to-Video, Text-to-Video, or Extend
2. **Pick style** — Cinematic Portrait, Talking Head, Landscape Pan, Anime Motion, Custom; for Extend mode this shows your recent completed jobs instead
3. **Describe & go** — prompt is pre-filled by the preset; tweak and click Generate

### Smoke test

```powershell
# Verify CUDA + ComfyUI ready
conda run -n img2vid python test-cuda.py    # should print VERDICT: CUDA works
curl http://127.0.0.1:8188/system_stats     # should show GPU info
curl http://localhost:8080/jobs.json        # should return []
```

If all three pass, drop a portrait JPG into the I2V flow with the "Cinematic Portrait" preset. First generation pays a ~30 s model warmup; subsequent ones are immediate.

---

## Path B — Lightning AI Studio (cloud GPU)

Notebook: `notebooks/10eros_lightning.ipynb`.

### Setup

1. Sign in to https://lightning.ai/
2. **New Studio** → search **Clean ComfyUI Template** (by mindthemath) → Create
3. Switch hardware to a GPU (L40s 48 GB recommended; H100 80 GB best perf)
4. In Lightning's file browser, upload `10eros_lightning.ipynb` from this repo's `notebooks/` directory
5. Open the notebook in JupyterLab

### Run it

| Cell | What it does | Time on first run |
|---|---|---|
| 1 | Detects template's pre-installed ComfyUI, installs 11 custom node packs, pins `mediapipe==0.10.13`, downloads ~24 GB of model weights, generates 1-sec silent `1.wav` placeholder | ~12 min |
| 2 | Loads inline UI→API workflow converter | instant |
| 3 | Boots ComfyUI on port 8188, launches Gradio UI on port 7860 | ~70 s |

After Cell 3 finishes, click Lightning's **Custom Port plugin** (right-hand panel) → forward port `7860` → you get a public URL for the Gradio UI. Port 8188 (ComfyUI) stays internal.

The Gradio UI has 3 run modes — **always start with `dry-run` after any change** to validate config without paying for sampling:

| Mode | Time on L40s | Use |
|---|---|---|
| `dry-run` | ~30 s | Validates against ComfyUI schema, NO sampling. Catches strength-out-of-range, missing file, class-not-found. |
| `smoke` | ~1-2 min | 2 stage-1 steps at 384×512. End-to-end check before quality. |
| `full` | ~3-6 min | Vantage default 13+3 steps at requested resolution. |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Step time ~5+ min per sampler step | VRAM thrashing from another GPU process (Norton, Chrome, etc.) | Close other GPU consumers. `nvidia-smi` should show ComfyUI as the only major user. |
| `RuntimeError: CUDA error: out of memory` mid-run | Cumulative model load + workflow exceeded VRAM | Restart ComfyUI to free the pool: kill the process, rerun `python comfyui/main.py --listen 127.0.0.1 --port 8188` |
| `module 'mediapipe' has no attribute 'solutions'` | mediapipe >= 0.10.14 removed `mp.solutions` (10S-Comfy-nodes still uses it) | `conda run -n img2vid pip install --force-reinstall mediapipe==0.10.13` |
| Output is generic anime / not photoreal | OmniNFT LoRA is on at default 0.8 | In webapp UI, set OmniNFT slider to 0 |
| Output doesn't match source person | Architectural limit of LTX-2.3 (see `docs/face-consistency.md`) | Use the new `placement_mode='keyframe'` + `reference_mask_mode='whole_frame'` probe — see `out/instrument_keyframe_consistency.py` |
| `ComfyUI not found` in notebook | Auto-detect failed | Run setup cell first OR launch a Studio from the Clean ComfyUI Template |
| Notebook Cell 1 OOMs while downloading | Lightning storage cap reached | Free space by `rm -rf ~/.cache/huggingface` then re-run cell |

---

## What's in this repo

```
image2video/
├── webapp.py                              # Flask app (3-step wizard, port 8080)
├── notebooks/
│   └── 10eros_lightning.ipynb             # Lightning AI notebook
├── workflows/
│   ├── Vantage-10Eros_I2V_v3.2.json       # Vantage UI workflow (97 nodes)
│   ├── Vantage-10Eros_I2V_v3.2.api.json   # API-converted (67 nodes)
│   ├── AICHUCKY_Ltx2.3.json               # Reference vanilla LTX-2.3 I2V
│   └── ltx23face.json                     # TenStrip face-consistency reference
├── tools/
│   └── ui_to_api.py                       # UI → API workflow converter
├── setup.ps1                              # One-shot Windows installer
├── INSTALL.md                             # This file
└── README.md                              # Project overview
```

`comfyui/`, `models/`, `out/`, `image2video_jobs/` are all gitignored — they're per-user state that gets regenerated by `setup.ps1` or at runtime.
