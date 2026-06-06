# image2video

Local web app that turns text prompts and still images into short videos
on your own GPU, with optional clip-by-clip extension.

The Flask server on `:8080` owns the UI, the prompt enhancer (a local
Qwen LLM), and an embedded ComfyUI runtime on `:8188` that drives the
Sulphur-2 video model. From your browser you just submit a prompt or
upload an image; the server does the rest and hands you back an MP4.

Modalities in v1:

| Mode | Inputs | Output |
|---|---|---|
| **Text → video** (T2V) | prompt | MP4 (5–10 s) |
| **Image → video** (I2V) | image + optional prompt | MP4 (5–10 s) |
| **Extend video** | a previously-generated MP4 + continuation prompt | longer MP4, last-frame chained |

## Requirements

- Windows 10/11
- NVIDIA GPU, ≥ 16 GB VRAM (24 GB recommended — built on RTX 4090 Laptop)
- ~80 GB free disk (Sulphur-2 FP8 ≈ 29 GB + Qwen ≈ 15 GB + ComfyUI + buffer)
- Anaconda / Miniconda on PATH
- Git on PATH

## Quickstart

```powershell
git clone https://github.com/dlmastery/image2video.git
cd image2video
.\setup.ps1                                # one-time, ~30 min, large download
conda run -n img2vid python webapp.py      # start the web app
# open http://localhost:8080/
```

See [CLAUDE.md](CLAUDE.md) for the operator manual (the same one used by
agents working on this repo).
