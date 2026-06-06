# CLAUDE.md

Operator's manual for AI agents and humans working on this repo. Read
[README.md](README.md) for the user pitch. This file tells you **how
to build, run, and extend the project without breaking it**.

If you only have time for one thing, read the [TL;DR](#tldr) then
[Things that broke before](#things-that-broke-before).

---

## Table of contents

1. [TL;DR](#tldr)
2. [What this is](#what-this-is)
3. [Architecture in one diagram](#architecture-in-one-diagram)
4. [Two processes, one GPU](#two-processes-one-gpu)
5. [Code map](#code-map)
6. [Common tasks](#common-tasks)
7. [Editing workflow](#editing-workflow)
8. [HTTP API surface](#http-api-surface)
9. [Workflow JSON patching](#workflow-json-patching)
10. [Things that broke before](#things-that-broke-before)
11. [Troubleshooting matrix](#troubleshooting-matrix)
12. [VRAM budget](#vram-budget)
13. [Things NOT to commit](#things-not-to-commit)

---

## TL;DR

```powershell
git clone https://github.com/dlmastery/image2video.git
cd image2video
.\setup.ps1                                # one-time, ~30 min, ~80 GB
conda run -n img2vid python webapp.py      # start the web app
# open http://localhost:8080/
```

The Flask server on `:8080` is the front door. It owns the UI, the
Qwen prompt enhancer, and an embedded ComfyUI subprocess on `:8188`.
You never touch the ComfyUI GUI.

If you're modifying `webapp.py`, you only need to restart it — ComfyUI
shuts down with its parent.

---

## What this is

A self-contained Windows desktop web app for short-form video
generation, in three modes:

| Mode | Input | Output |
|---|---|---|
| **T2V** | prompt | 5–10 s MP4 |
| **I2V** | image + optional prompt | 5–10 s MP4 |
| **Extend** | a prior job's MP4 + continuation prompt | longer MP4 via last-frame chaining |

Same operator model as `face-swap-streamer`: a Flask server, vanilla-JS
viewer, inline HTML templates, per-job working dirs, no database.

---

## Architecture in one diagram

```
browser ── /:8080 ──▶  webapp.py (Flask)
                              │
                              │  child process (spawned at boot)
                              ▼
                       ComfyUI on 127.0.0.1:8188   ── loads Sulphur-2 FP8
                              │                       (~12 GB VRAM resident)
                              │  HTTP /prompt
                              │  WebSocket /ws  (progress events)
                              │
                              ▼
                       ComfyUI/output/<file>.mp4
                              │
                              ▼
                       copied into image2video_jobs/<id>/output.mp4
                              │
                              ▼
                       served by webapp.py to the browser

  Separately, on-demand:
  webapp.py ── transformers + bitsandbytes ──▶  Qwen2.5-7B 4-bit (~5 GB)
                                                lazy-load before enhance,
                                                unload after.
```

---

## Two processes, one GPU

| Process | Owns | Why |
|---|---|---|
| `webapp.py` (Flask) | UI, job queue, Qwen prompt enhancer, ComfyUI lifecycle | Single front door for the user |
| ComfyUI on `:8188`  | Sulphur-2 weights, LTX VAE / text encoder, video sampling, frame→MP4 encode | Sulphur-2 has no diffusers-compatible repo layout (no `model_index.json`), so we have to drive it through ComfyUI; we just use ComfyUI's API and hide its GUI |

Both run in the **same conda env** (`img2vid`, Python 3.11) so they
share the torch + cu121 + xformers + cudnn install. ComfyUI is spawned
via `subprocess.Popen` inside the same env.

Lifecycle:
- `webapp.py` calls `comfy_client.start()` at boot → spawns
  `python comfyui/main.py --listen 127.0.0.1 --port 8188` and waits
  up to 120 s for `/system_stats` to return 200.
- `atexit` + SIGINT handlers kill the subprocess on shutdown
  (`taskkill /T /F` on Windows so the whole process tree dies).

---

## Code map

| File | Purpose |
|---|---|
| `webapp.py` | The whole app: Flask routes, job state, ComfyUI client, Qwen enhancer, inline HTML/JS for the UI. Mirrors `face-swap-streamer/webapp.py`'s single-file pattern. |
| `setup.ps1` | One-shot installer: conda env, ComfyUI clone, custom nodes, model + Qwen download, cuDNN patch, smoke test. |
| `workflows/` | T2V / I2V / extension workflow JSONs copied here by `setup.ps1` from the Sulphur HF repo. **DO NOT** check huge models in here; only the small JSONs. |
| `comfyui/` | Cloned ComfyUI + custom nodes. Has its own `.git`; we never touch its source except for the cuDNN patch (which `setup.ps1` re-applies idempotently). |
| `models/` | `models/qwen2.5-7b-instruct/` and any other side-loaded models. ComfyUI's own model store lives at `comfyui/models/`. |
| `image2video_jobs/<id>/` | Per-job working dir: uploaded source image (I2V), parent MP4 (Extend), final `output.mp4`, `meta.json`. |
| `requirements.txt` | Pip deps **on top of** ComfyUI's own requirements. transformers + bnb + flask + opencv. |

---

## Common tasks

### Run the webapp

```powershell
conda run -n img2vid python webapp.py
```

ComfyUI starts as a child process. First boot pays a ~30 s model load.
Subsequent jobs reuse the same loaded model.

### Run in the background + tail logs

```powershell
Start-Process -WindowStyle Hidden -FilePath conda `
  -ArgumentList @('run','-n','img2vid','--no-capture-output','python','webapp.py') `
  -RedirectStandardOutput out\webapp.log -RedirectStandardError out\webapp.err

Get-Content out\webapp.log -Tail 50 -Wait
```

ComfyUI's own stderr goes into `out\comfy.log`.

### Stop everything

Ctrl+C in the webapp terminal kills both processes via the atexit hook.
If something gets stuck:

```powershell
Get-Process python | Where-Object { $_.MainModule.FileName -match 'envs\\img2vid' } | Stop-Process -Force
```

### Re-apply install

```powershell
.\setup.ps1 -Force
```

Pulls latest ComfyUI / custom nodes (CAREFUL — upstream changes can
break the workflow JSON patcher; see [Things that broke before](#things-that-broke-before)).

---

## Editing workflow

1. Edit `webapp.py`.
2. Ctrl+C the running webapp (kills ComfyUI too).
3. `conda run -n img2vid python webapp.py`.
4. Wait ~30 s for ComfyUI to re-bind and Sulphur-2 to re-load.
5. Test in browser at <http://localhost:8080/>.
6. If a job fails, read **both** `image2video_jobs/<id>/meta.json` and
   `out\comfy.log` — the failure could be in either layer.

For HTML/JS-only changes inside the inlined templates, hard-refresh the
browser (Ctrl+F5) — no server restart needed.

---

## HTTP API surface

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/` | Main page with three tabs: T2V, I2V, Extend |
| `POST` | `/generate/t2v` | Form (`prompt`, `steps`, `frames`, `width`, `height`, `enhance`) → spawns job, redirects to `/job/<id>` |
| `POST` | `/generate/i2v` | Multipart form (`source` image + same params) → I2V job, redirects to `/job/<id>` |
| `POST` | `/extend/<parent_id>` | Form (`prompt`, `frames`, `enhance`) → reads parent's `output.mp4`, last-frame I2V, ffmpeg-concat with parent, redirects to new `/job/<id>` |
| `POST` | `/enhance` | JSON `{prompt}` → loads Qwen, returns `{enhanced: "..."}`, unloads Qwen |
| `GET`  | `/job/<id>` | Viewer page: progress bar, video player when done, Extend button |
| `GET`  | `/job/<id>/status` | JSON: `phase`, `message`, `current_step`, `total_steps`, `progress`, `error` |
| `GET`  | `/job/<id>/output` | Serves `output.mp4` inline (Range-aware) for the `<video>` element |
| `GET`  | `/job/<id>/download` | Same MP4 with `Content-Disposition: attachment` |
| `GET`  | `/comfy/healthz` | Pass-through to ComfyUI's `/system_stats`. Used by the viewer to know when the backend is ready. |

Job phases (in order):
`queued → enhancing → submitting → generating → encoding → done`
plus `error` (terminal).

---

## Workflow JSON patching

ComfyUI workflows are dicts of node-id → `{class_type, inputs, ...}`. The
exported API format uses string node IDs like `"6"`, `"10"` — **these
IDs change every export**. Don't hardcode them.

Resolve nodes by `class_type`, then walk the graph from the sampler
backwards to find the actual positive-prompt CLIPTextEncode, the
LoadImage feeding I2V, etc.:

```python
def find_sampler(wf):
    for cls in ("LTXVideoSampler","KSamplerAdvanced","KSampler"):
        ids = [nid for nid,n in wf.items() if n.get("class_type")==cls]
        if ids: return ids[0]
    raise RuntimeError("no sampler node in workflow")

def upstream(wf, node_id, input_name):
    """Return the node_id that feeds `input_name` of node_id."""
    inp = wf[node_id]["inputs"].get(input_name)
    if isinstance(inp, list) and len(inp)==2: return inp[0]
    return None

sampler = find_sampler(wf)
pos_node = upstream(wf, sampler, "positive")
wf[pos_node]["inputs"]["text"] = user_prompt
```

The same trick finds the I2V image input (`upstream(sampler, "conditioning")`
→ a chain ending in a `LoadImage` node).

If the workflow JSON shape changes upstream, the resolver needs to
learn the new node class names — single point of change.

---

## Things that broke before

### 1. cuDNN DLL discovery on Windows

Same gotcha as faceswap: torch + onnxruntime ignore PATH for native
imports on Python 3.8+. `setup.ps1` prepends an
`os.add_dll_directory(...)` block to `comfyui/main.py` and keeps the
returned cookies alive in a module-level list. **Don't `[os.add_dll_directory(p) for p in dirs]`** — the cookies are GC'd and DLLs vanish from the search path.

### 2. ComfyUI subprocess won't die on Windows

`Popen(...).terminate()` only kills the python.exe wrapper; the
child python process holding the GPU context survives, port 8188 stays
bound, and the next webapp restart fails to bind 8188.

Fix: spawn with `creationflags=CREATE_NEW_PROCESS_GROUP`, kill via
`taskkill /T /F /PID <pid>` (`/T` = whole tree). `webapp.py`'s
`comfy_stop()` does this.

### 3. Workflow JSON node IDs are NOT stable

Re-exporting the same workflow renumbers nodes. Patching by string ID
breaks on re-export. **Always resolve by `class_type` + graph traversal.**
See [Workflow JSON patching](#workflow-json-patching).

### 4. Qwen + Sulphur-2 share 24 GB

Qwen 7B in 4-bit is ~5 GB; Sulphur-2 FP8 working set is ~15 GB. Both
resident at once = OOM during sampling. Qwen must be loaded → used →
**unloaded with `del model; torch.cuda.empty_cache()`** before each
generation request. The `Enhancer.enhance(prompt)` method does this.

### 5. ffmpeg concat-copy needs matching timebases

`ffmpeg -f concat -c copy` only works when all input files have the
same codec, pixel format, and timebase. ComfyUI's MP4 outputs are
consistent across runs — fine. **Do NOT** mix outputs from different
ComfyUI versions in one extend chain.

### 6. Sulphur-2-base ships flat .safetensors, no diffusers layout

`SulphurAI/Sulphur-2-base` does NOT contain `model_index.json`,
`transformer/`, `vae/`, etc. — only flat blobs and ComfyUI workflows.
`DiffusionPipeline.from_pretrained()` will not work. This is why we
embed ComfyUI instead of using `diffusers` directly. Re-check this
periodically — Lightricks has said diffusers support is "coming soon."

---

## Troubleshooting matrix

| Symptom | First place to look | Likely fix |
|---|---|---|
| Webapp on `:8080` but `/comfy/healthz` is 503 | `out\comfy.log` | ComfyUI failed to start. Common: wrong checkpoint name, missing custom node, CUDA OOM at load |
| `setup.ps1` step 6 times out at 120 s | `out\comfy_smoke.log` | Model load takes >120 s on slow disks. Re-run smoke test by hand: `conda run -n img2vid python comfyui/main.py --listen 127.0.0.1 --port 8188` |
| Generation stuck on "generating 0%" forever | `out\comfy.log` last lines | CUDA died silently — check `nvidia-smi`. Restart webapp |
| Output MP4 looks black | ComfyUI workflow VAE node | Wrong VAE for Sulphur — check `comfyui/models/vae/` and the workflow's VAE loader |
| "No video output found in history" | `/job/<id>/status` JSON | Workflow's output node didn't write a video. Check `class_type` of save node in workflow JSON; webapp scans for `VHS_VideoCombine`, `SaveAnimatedWEBP`, `SaveVideo` |
| Qwen OOM during enhance | `out\webapp.log` | Sulphur-2 still in VRAM. Either (a) drop Qwen to 3-bit, (b) skip enhance for that job, (c) let Sulphur-2 unload first (close all jobs) |

---

## VRAM budget

Target hardware: RTX 4090 Laptop, 24 GB.

| Component | Resident VRAM | When |
|---|---|---|
| Sulphur-2 FP8 transformer | ~9 GB | Always (loaded by ComfyUI at boot) |
| LTX VAE | ~1 GB | Always |
| Text encoder | ~1 GB | Always |
| Activations / sampler scratch | ~3–5 GB | During generation |
| Qwen2.5-7B 4-bit | ~5 GB | Only during `/enhance` (lazy-load + unload) |

Steady state during generation: ~14–16 GB. Peak with Qwen: would be
~21 GB, which is why Qwen unloads first.

---

## Things NOT to commit

The `.gitignore` excludes them; be deliberate.

| Path | Why |
|---|---|
| `comfyui/` | Cloned with its own git history |
| `models/` and any `*.safetensors`, `*.gguf`, `*.pth`, `*.onnx`, `*.bin` | Multi-GB; downloaded by `setup.ps1` |
| `image2video_jobs/` | User uploads + generated content |
| `out/`, `*.log` | Build / runtime artefacts |
| `.claude/`, `.playwright-mcp/` | Agent runtime state |
| `.commit-msg.tmp` | PowerShell heredoc workaround |

If you add a new file type that should be excluded, update `.gitignore`
in the same commit.
