"""Stage 4: Download GGUF UNET + split VAE/text encoder so we can cut the
commit budget from ~57 GB (FP8) to ~22 GB (GGUF Q4_K_M).

Why: Sulphur's unified FP8 safetensors is 29 GB. Windows mmap commits
the whole file even when only a fraction is in VRAM at any time. With
LoRAs (13 GB) + Gemma (7 GB) + ComfyUI overhead the commit ceiling is
57 GB, which busts the default 32 GB RAM + 32 GB pagefile = 64 GB
limit once other resident processes are added.

Replacing the unified FP8 with:
  - GGUF Q4_K_M UNET            14 GB  (vantagewithai/Sulphur-2-Base-GGUF)
  - sulphur_vae.safetensors      small (vantagewithai/Sulphur-2-Base-Split)
  - sulphur_text_encoder        small (vantagewithai/Sulphur-2-Base-Split)
  - audio_vae already in /vae/  (from stage 3)

drops total commit to ~22 GB and lifts the pagefile blocker entirely.

Variant is configurable via $env:IMG2VID_GGUF_QUANT (default Q4_K_M).
Options: Q3_K_S (10 GB) ... Q4_K_M (14 GB) ... Q5_K_M (16 GB) ... Q8_0 (23 GB).
"""
import os, sys
import truststore; truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
COMFY_MODELS = HERE / "comfyui" / "models"

QUANT = os.environ.get("IMG2VID_GGUF_QUANT", "Q4_K_M")
GGUF_FILE = f"sulphur_dev-{QUANT}.gguf"

DOWNLOADS = [
    ("vantagewithai/Sulphur-2-Base-GGUF", GGUF_FILE, "unet",
     f"Sulphur UNET ({QUANT}) for UnetLoaderGGUF"),
    ("vantagewithai/Sulphur-2-Base-Split", "vae/sulphur_vae.safetensors", "vae",
     "Visual VAE (split from unified ckpt)"),
    ("vantagewithai/Sulphur-2-Base-Split",
     "text_encoder/sulphur_text_encoder.safetensors", "text_encoders",
     "Sulphur text encoder layers (split)"),
]

for repo, fn, sub, note in DOWNLOADS:
    target_dir = COMFY_MODELS / sub
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / Path(fn).name
    if target_path.exists():
        print(f"[skip] {target_path.name} ({target_path.stat().st_size / 1e9:.1f} GB)")
        continue
    print(f"[get ] {note}")
    print(f"       repo={repo} file={fn}")
    p = hf_hub_download(repo_id=repo, filename=fn, local_dir=str(target_dir))
    # Flatten subdirs that hf_hub_download preserves
    if Path(p) != target_path:
        import shutil
        shutil.move(p, target_path)
        # clean empty subdirs
        for d in sorted(target_dir.glob("**/"), reverse=True):
            if d != target_dir and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    print(f"       saved -> {target_path}  ({target_path.stat().st_size / 1e9:.1f} GB)")

# Both LTXVAudioVAELoader and LTXAVTextEncoderLoader scan
# models/checkpoints/ for their layers. Originally they'd pull from the
# unified Sulphur FP8 ckpt, but mmap-slicing that 29 GB file under
# Windows commit-budget pressure crashes ComfyUI with an access
# violation. Mirror the small split files into checkpoints/ so the
# loaders read from them instead.
import shutil
CKPT_DIR = COMFY_MODELS / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
for src in [COMFY_MODELS / "vae" / "sulphur_audio_vae.safetensors",
            COMFY_MODELS / "text_encoders" / "sulphur_text_encoder.safetensors"]:
    if src.exists():
        dst = CKPT_DIR / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  mirrored {src.name} into checkpoints/")
        else:
            print(f"  [skip] {dst.name} already in checkpoints/")

print("\nAll stage-4 GGUF assets ready.")
print(f"Active quant: {QUANT} (override via IMG2VID_GGUF_QUANT env var)")
