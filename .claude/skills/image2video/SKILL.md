---
name: image2video
description: How to work on the image2video web app — a Flask server that drives an embedded ComfyUI runtime for Sulphur-2 / LTX-2.3 video generation (T2V + I2V + Extend) with a Qwen prompt enhancer. Use when editing webapp.py, setup.ps1, the workflow patcher, or extending modalities. Includes the architecture, file map, gotchas already paid for, and a live progress tracker.
---

# image2video — operator skill

A Flask server on `:8080` is the front door. It spawns ComfyUI on
`:8188` as a child process, drives it via REST/WS, runs a lazy Qwen
prompt enhancer in-process, and serves the resulting MP4 back.

This skill is the durable working memory for the project — what was
built, what burned us, what's next. Update the **Progress** section as
work lands. Update the **Gotchas** section the moment a new failure
mode is diagnosed (don't make the next agent rediscover it).

---

## Progress (LIVE — update on every change)

### Done

- **Scaffold** — repo at `C:\Users\evija\image2video`, files:
  `webapp.py` (1076 lines), `setup.ps1`, `CLAUDE.md`, `README.md`,
  `requirements.txt`, `.gitignore`, `workflows/`, `docs/`.
- **GitHub repo created**: `dlmastery/image2video` (PRIVATE; flip with
  `gh repo edit dlmastery/image2video --visibility public` if you want
  it public later). Branch `main` tracks origin. Initial 7 commits
  pushed (scaffold → em-dash strip → SSL bypass + 2>&1 drop →
  truststore → CUDA torch → workflows + skill).
  **Sync policy: push at every meaningful checkpoint** (patcher
  passes, converter works, smoke test green, etc.). Use
  `$env:GIT_SSL_NO_VERIFY='1'; git push` since this box has the same
  corporate-cert issue we worked around in setup.ps1.
- **Install (setup.ps1) succeeds end-to-end.**
  - Conda env `img2vid` (Python 3.11) created
  - ComfyUI + ComfyUI-Manager + ComfyUI-LTXVideo + ComfyUI-GGUF +
    ComfyUI-PromptRelay cloned into `comfyui/`
  - Sulphur-2 FP8 (`sulphur_dev_fp8mixed.safetensors`, ~29 GB) +
    Qwen2.5-7B-Instruct (~15 GB) + Sulphur's bundled workflows
    downloaded
  - CUDA-12.1 torch wheels force-installed (replaces CPU-only default)
  - cuDNN DLL discovery patch applied to `comfyui/main.py`
  - Smoke test passed: ComfyUI binds `:8188` on the 4090

### In progress

- **Webapp + ComfyUI run end-to-end up to model load.** `/prompt` accepts
  the patched workflow with zero validation errors. ComfyUI begins
  sampling on the 4090 but bails with Windows error 1455 ("paging
  file too small") during model load.
- **BLOCKER: Windows pagefile.** Combined commit budget needed: ~50 GB
  (Sulphur FP8 29 GB + Sulphur LoRA 10 GB + distilled LoRA 3-4 GB +
  Gemma encoder 7 GB + ComfyUI Python overhead). User has 32 GB RAM
  so default pagefile of ~32 GB caps total commit ~64 GB, but other
  resident processes eat that headroom. **Fix:** raise Windows
  pagefile to 96-128 GB via System Properties -> Advanced -> Performance
  Settings -> Advanced -> Virtual Memory.

### Next (in order)

1. **Start `webapp.py`** — confirm Flask binds `:8080`, ComfyUI child
   binds `:8188`, workflow loader prints
   `using 'ltx23_i2v distilled.json' for both T2V and I2V`.
2. **Smoke-test T2V** with cheap params (256x256, 25 frames, 4 steps,
   enhance OFF). Confirm: enhancing skipped, submitting, generating
   ticks the step bar, encoding, done. Output MP4 plays in VLC.
3. **Smoke-test I2V** with an uploaded image. Confirm
   `comfyui/input/<jobid>_<filename>` exists, bypass=False, output
   animates the input image.
4. **Smoke-test Extend** on a T2V result. Confirm `start_frame.png`
   in the new job dir, concat MP4 plays continuously.
5. **UI→API converter** for the 3 UI-format workflows (optional —
   only worth doing if v1 quality is insufficient and we want the
   non-distilled base workflow).
6. **Fix whatever breaks**.

### Done (workflow patcher)

- `_patch_workflow` rewritten for the real LTX-2.3 graph (see commit
  history). Key behaviours:
  - Walks every `CFGGuider`, follows `positive`/`negative` slot
    semantics through combiner nodes (`LTXVCropGuides`,
    `LTXVConditioning`) to the terminal `CLIPTextEncode`.
  - **Slot-aware resolver**: slot 0 -> "positive" input, slot 1 ->
    "negative". Without this, both sides routed to the same encoder
    and overwrote each other.
  - Patches `noise_seed` on every `RandomNoise`, `steps` on every
    `LTXVScheduler`, dims/length on `EmptyLTXVLatentVideo`.
  - For T2V: sets `LTXVImgToVideoInplace.bypass = True` on every
    such node, so the same workflow works for both modes.
  - For I2V: copies the upload into `comfyui/input/`, sets bypass
    False, patches `LoadImage.image` to the new filename.
- `_load_workflows` now picks the single API-format workflow
  (`ltx23_i2v distilled.json`) and registers it as BOTH `t2v` and
  `i2v` in WORKFLOWS (same dict, deep-copied per job).
- **Node-compat layer** (`_apply_compat`): renames
  `ImageScaleDownBy` -> `ImageScaleBy` (rename `images`->`image`, add
  `upscale_method`) and `ResizeImageResolution` ->
  `ImageScaleToMaxDimension` (rename `resolution` -> `largest_size`).
  Sulphur authored against an older node set; stock ComfyUI provides
  equivalents that we map at patch time.
- **Loader filename remaps**: workflow hardcodes filenames Sulphur
  never shipped. `LOADER_REMAPS` table rewrites every
  `CheckpointLoaderSimple.ckpt_name`, `LTXVAudioVAELoader.ckpt_name`
  (which actually scans `models/checkpoints/`, not `models/vae/` -
  the audio VAE is baked into the unified Sulphur checkpoint),
  `LTXAVTextEncoderLoader.ckpt_name`+`text_encoder`, and
  `LatentUpscaleModelLoader.model_name`. `LORA_NAME_MAP` rewrites
  `sulphur_final.safetensors` -> `sulphur_lora_rank_768.safetensors`
  (the actual filename Sulphur shipped under).
- **T2V placeholder image** (`comfyui/input/img2vid_placeholder.png`,
  1x1 black PNG written by `out/stage2_assets.py`): the patcher
  points `LoadImage.image` at this so ComfyUI's pre-execution
  file-existence check passes for T2V jobs, even though
  `LTXVImgToVideoInplace.bypass=True` means the image is never used.

### Done (model assets)

`out/stage2_assets.py` + `out/stage3_assets.py` (run once at the bottom
of `setup.ps1` in a future pass) download:
- `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` from `Lightricks/LTX-2.3`
  into `comfyui/models/upscale_models/`
- `gemma-3-12b-it-orthogonal-reflection-bounded-ablation-v4-12B-fp4_mixed.safetensors`
  from `inflatebot/LTX23-gemma-3-12b-it-orthogonal-reflection-bounded-ablation-v4-fp4_mixed`
  into `comfyui/models/text_encoders/`
- `sulphur_audio_vae.safetensors` from
  `vantagewithai/Sulphur-2-Base-Split` into `comfyui/models/vae/`
  (turned out unused - audio VAE is in the main ckpt - but harmless)
- `distill_loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors`
  + `sulphur_lora_rank_768.safetensors` from `SulphurAI/Sulphur-2-base`
  into `comfyui/models/loras/`
- `comfyui/input/img2vid_placeholder.png` (1x1 black PNG)

These ARE NOT yet wired into `setup.ps1`. Either rerun `out/stage2_assets.py`
+ `out/stage3_assets.py` manually after `setup.ps1`, or fold them in.

---

## Architecture in one diagram

```
browser ── /:8080 ──▶  webapp.py (Flask)
                            │
                            │ spawns at boot via subprocess.Popen
                            ▼
                       ComfyUI on 127.0.0.1:8188
                            │
                            │ Sulphur-2 FP8 + LTX VAE/text encoder
                            │ (loaded lazily on first /prompt)
                            ▼
                       ComfyUI/output/<file>.mp4 (or .webp)
                            │
                            ▼
                       copied + remuxed into
                       image2video_jobs/<id>/output.mp4
                            │
                            ▼
                       served back by Flask

  Separately, on /enhance:
  webapp.py ── transformers + bitsandbytes ──▶ Qwen2.5-7B 4-bit (~5 GB)
                                                lazy-load → enhance →
                                                del + cuda.empty_cache
```

Two processes, one GPU. Both run in conda env `img2vid` (Python 3.11)
so they share torch + cu121 + cudnn.

---

## File map

| Path | Purpose |
|---|---|
| `webapp.py` (~1076 lines) | Whole app: Flask routes, job state, ComfyUI client, Qwen enhancer, workflow patcher, inline HTML templates. Single file, faceswap-style. |
| `setup.ps1` | One-shot installer (conda env, ComfyUI clone, model downloads, cuDNN patch, smoke test) |
| `workflows/` | 4 Sulphur LTX-2.3 JSONs. **Only `ltx23_i2v distilled.json` is API-format.** Other 3 are UI-format — need converter. |
| `comfyui/` | Cloned ComfyUI + custom nodes (has its own .git; don't add to ours) |
| `models/qwen2.5-7b-instruct/` | Qwen weights for prompt enhancement |
| `comfyui/models/checkpoints/sulphur_dev_fp8mixed.safetensors` | The 29 GB video model |
| `image2video_jobs/<id>/` | Per-job dir: uploaded source, parent MP4 (extend), `output.mp4`, `meta.json` |
| `out/` | Runtime logs: `comfy.log`, `setup.log`, etc. |
| `.claude/skills/image2video/SKILL.md` | This file — living working memory |
| `CLAUDE.md` | Operator manual for humans (also useful for agents but more verbose) |

---

## Common commands

```powershell
# Start the webapp (foreground; Ctrl+C also kills ComfyUI)
conda run -n img2vid python webapp.py

# Background + tail log
Start-Process -WindowStyle Hidden -FilePath conda `
  -ArgumentList @('run','-n','img2vid','--no-capture-output','python','webapp.py') `
  -RedirectStandardOutput out\webapp.log -RedirectStandardError out\webapp.err
Get-Content out\webapp.log -Tail 30 -Wait

# Stop everything (kills ComfyUI subprocess tree too)
Get-Process python | Where-Object { $_.MainModule.FileName -match 'envs\\img2vid' } | Stop-Process -Force

# Re-apply install (idempotent; -Force re-pulls upstream nodes)
.\setup.ps1
```

For a deeper command reference see [CLAUDE.md](../../../CLAUDE.md).

---

## Gotchas — these already cost us time, don't relearn

### 1. PowerShell 5.1 chokes on em-dashes

PS5 reads `.ps1` files as Windows-1252 if there's no BOM. Em-dash `—`
(U+2014) breaks the parser somewhere downstream, producing
"missing terminator" errors at unrelated line numbers. **All `.ps1`
files in this repo must be ASCII-only.** Use `-` not `—`.

### 2. `2>&1` on native exes + `$ErrorActionPreference=Stop`

In `Run-Conda`, `& conda @argList 2>&1` wrapped conda's first stderr
warning as a `NativeCommandError` ErrorRecord, which combined with
`Stop` threw on conda's `Retrying` warning even when conda's actual
exit code was 0. **Don't redirect stderr from native exes in this
script.** The outer `Tee-Object` on the caller side still captures
both streams.

### 3. Corporate-cert SSL on conda/pip/HF

This box has a corporate-issued root cert that Python's bundled
`certifi` doesn't trust. Three layers of bypass needed:

- conda: `$env:CONDA_SSL_VERIFY = "false"`
- pip: `--trusted-host pypi.org --trusted-host files.pythonhosted.org
  --trusted-host pypi.python.org --trusted-host download.pytorch.org`
- huggingface_hub: `truststore.inject_into_ssl()` BEFORE the
  `from huggingface_hub import ...` line. Truststore reads the
  Windows trust store (which DOES trust the corporate root).
  Fallback to `requests.adapters.HTTPAdapter.send` monkey-patch with
  `verify=False` if truststore unavailable.

`ssl._create_default_https_context = ssl._create_unverified_context`
only affects stdlib `urllib`, NOT the `requests`/`urllib3` chain that
`huggingface_hub` uses. Don't waste time on that path.

### 4. PyPI default `torch` on Windows is CPU-only

ComfyUI's `requirements.txt` + custom-node requirements all spec plain
`torch`, which on Windows resolves to the CPU wheel. `setup.ps1` step
3e force-reinstalls from `https://download.pytorch.org/whl/cu121`
AFTER the other pip steps so nothing downgrades us back. Failure
signature: `AssertionError: Torch not compiled with CUDA enabled` in
ComfyUI's stderr, plus its `comfy_kitchen backend cuda: 'CUDA not
available on this system'` diagnostic.

### 5. Sulphur ships ComfyUI workflows in TWO formats

`SulphurAI/Sulphur-2-base/workflows/` ships:

- `ltx23_i2v distilled.json` — **API format** (`{"<nodeid>":{...}}`),
  ready to POST to `/prompt`
- `ltx23_i2v base.json`, `ltx23_t2v base.json`,
  `ltx23_t2v distilled.json` — **UI editor format**
  (`{"nodes":[...], "links":[...], ...}`), `/prompt` will reject

We need a UI→API converter to use the other three (which are the
higher-quality / non-distilled variants).

### 6. LTX-2.3 sampler graph is NOT classic KSampler

The API workflow uses `SamplerCustomAdvanced` whose inputs are
`noise`, `guider`, `sampler`, `sigmas`, `latent_image` — NOT the
classic `positive`/`negative`/`steps`/`seed` pattern. Patcher must
trace:

- `SamplerCustomAdvanced.guider` → `CFGGuider` → `.positive` /
  `.negative` (both ref `CLIPTextEncode`)
- `SamplerCustomAdvanced.noise` → `RandomNoise` → patch `.noise_seed`
- `LTXVScheduler` holds the step count
- I2V image input goes through
  `LoadImage` → `ResizeImageResolution` → `ImageScaleDownBy` →
  `LTXVImgToVideoInplace` → latent chain
- T2V uses `EmptyLTXVLatentVideo` instead of the image chain

### 7. ComfyUI subprocess won't die without `/T /F`

`Popen.terminate()` only kills the python wrapper; the child process
holding the CUDA context survives and `:8188` stays bound. Use
`taskkill /PID <pid> /T /F` on Windows. `webapp.py::ComfyClient.stop`
already does this.

### 8. Node IDs in workflow JSON are NOT stable

Every UI export renumbers nodes. **Never index workflow dicts by
literal string IDs.** Walk by `class_type` + graph traversal.

---

## VRAM budget on RTX 4090 24 GB

| Component | Resident | When |
|---|---|---|
| Sulphur-2 FP8 transformer | ~9 GB | After first /prompt |
| LTX VAE + text encoder | ~2 GB | Always |
| Activations / sampler scratch | ~3-5 GB | During generation |
| Qwen2.5-7B 4-bit (bnb) | ~5 GB | **Only during /enhance** — lazy load + unload |

Steady-state generation: ~14-16 GB. Peak with Qwen co-resident: ~21
GB — close. That's why Qwen unloads with `del model;
torch.cuda.empty_cache(); torch.cuda.ipc_collect()` between uses.

---

## Open questions

- **Repo visibility** — push `dlmastery/image2video` as public (like
  faceswap) or private?
- **First-class T2V workflow** — derive from i2v distilled (strip
  image nodes, swap to EmptyLTXVLatentVideo) OR convert one of the
  UI-format T2V JSONs? Converter is more general; derivation is
  faster to ship.
- **VRAM contention** — confirmed on paper but not measured in
  practice yet. Once we run an end-to-end T2V, check `nvidia-smi`
  during a generation that follows an /enhance call.

---

## Smoke-test recipe (once webapp is up)

1. `Invoke-WebRequest http://localhost:8080/` should return 200 + the
   form HTML.
2. Submit a T2V job: prompt "a fox running in autumn leaves, slow
   tracking camera", 25 frames, 512×384, 20 steps, enhance ON.
3. Watch `/job/<id>/status` go through phases:
   `queued → enhancing → submitting → generating → encoding → done`.
4. Confirm `image2video_jobs/<id>/output.mp4` exists and plays in VLC.
5. Use the Extend button — confirm `start_frame.png` is in the new
   job dir, the concatenated MP4 plays continuously.

If any step hangs > 5 min beyond expected, read **both**
`out\webapp.log` AND `out\comfy.log` — the failure could be on either
side of the `/prompt` boundary.
