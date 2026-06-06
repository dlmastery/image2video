"""Download all remaining model files needed by the Sulphur i2v distilled workflow.

Mapping from workflow-expected filename -> our actual on-disk filename
gets resolved by the patcher (webapp.py). Here we just download the
files we need to a known location.
"""
import os, sys, shutil
import truststore; truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
COMFY_MODELS = HERE / "comfyui" / "models"

DOWNLOADS = [
    # (repo_id, file_in_repo, comfy_subdir, comment)
    ("inflatebot/LTX23-gemma-3-12b-it-orthogonal-reflection-bounded-ablation-v4-fp4_mixed",
     "gemma-3-12b-it-orthogonal-reflection-bounded-ablation-v4-12B-fp4_mixed.safetensors",
     "text_encoders",
     "Gemma text encoder for LTXAVTextEncoderLoader (~7 GB)"),

    ("vantagewithai/Sulphur-2-Base-Split",
     "audio_vae/sulphur_audio_vae.safetensors",
     "vae",
     "Audio VAE for LTXVAudioVAELoader (~few hundred MB)"),

    ("SulphurAI/Sulphur-2-base",
     "distill_loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
     "loras",
     "Distilled LoRA (~3-4 GB)"),

    ("SulphurAI/Sulphur-2-base",
     "sulphur_lora_rank_768.safetensors",
     "loras",
     "Sulphur LoRA - workflow expects 'sulphur_final' but we patch it (~10 GB)"),
]

for repo, fn, sub, note in DOWNLOADS:
    target_dir = COMFY_MODELS / sub
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / Path(fn).name
    if target_path.exists():
        print(f"[skip] {target_path.name} already on disk ({target_path.stat().st_size / 1e9:.1f} GB)")
        continue
    print(f"[get ] {note}")
    print(f"       repo={repo} file={fn}")
    p = hf_hub_download(repo_id=repo, filename=fn, local_dir=str(target_dir))
    # hf_hub_download preserves subdirs - flatten if needed
    if Path(p) != target_path and Path(p).exists():
        shutil.move(p, target_path)
        # clean empty subdirs
        for d in sorted(target_dir.glob("**/"), reverse=True):
            if d != target_dir and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    print(f"       saved -> {target_path}")

# Verify spatial upscaler is in the right dir for LatentUpscaleModelLoader.
# That loader pulls from comfyui/models/upscale_models/ - confirm it's there.
upsc = COMFY_MODELS / "upscale_models" / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
print(f"\nUpscaler at {upsc}: {'OK' if upsc.exists() else 'MISSING'}")

print("\nAll stage-3 downloads done.")
