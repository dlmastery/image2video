"""Download the canonical 10Eros asset set, per the Vantage tutorial.

Storage map (from the video description):
  unet/                 10Eros_v1-{QUANT}.gguf                        ~14 GB
  text_encoders/        gemma_3_12B_it.safetensors                    ~7 GB
                        ltx-2-3-22b-text_encoder.safetensors          ~2.3 GB
  vae/                  ltx-2-3-22b-VAE.safetensors                   ~1.4 GB
                        ltx-2-3-22b-audio_vae.safetensors             ~0.36 GB
  loras/                ltx-2.3-22b-distilled-lora-1.1_...safetensors ~3.5 GB
                        OmniNFT_converted_lora.safetensors            ~1 GB
  upscale_models/       ltx-2.3-spatial-upscaler-x2-1.1.safetensors   ~1 GB

The text encoder + VAE come from LTX-2.3 base (NOT 10Eros-Split), which
both Sulphur and 10Eros share. The distilled LoRA still ships from the
Sulphur-2-Base-Split repo (Vantage just re-uses it for 10Eros too).
Quant via env IMG2VID_GGUF_QUANT (Q4_K_M default = best for 16 GB VRAM
laptops, Q5_K_M better for 24 GB desktop).
"""
import os, sys, shutil
import truststore; truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
COMFY_MODELS = HERE / "comfyui" / "models"
QUANT = os.environ.get("IMG2VID_GGUF_QUANT", "Q4_K_M")

DOWNLOADS = [
    # (repo_id, file_in_repo, comfy_subdir, note, ~size GB)
    ("vantagewithai/LTX2.3-10Eros-GGUF",
     f"10Eros_v1-{QUANT}.gguf",
     "unet",
     f"10Eros UNET ({QUANT}) for UnetLoaderGGUF", 14.3),

    ("Comfy-Org/ltx-2",
     "split_files/text_encoders/gemma_3_12B_it.safetensors",
     "text_encoders",
     "Gemma 3 12B text encoder (ComfyOrg official)", 7.0),

    ("vantagewithai/LTX-2.3-Split",
     "text_encoder/ltx-2-3-22b-text_encoder.safetensors",
     "text_encoders",
     "LTX-2.3 base text encoder layers", 2.3),

    ("vantagewithai/LTX-2.3-Split",
     "vae/ltx-2-3-22b-VAE.safetensors",
     "vae",
     "LTX-2.3 visual VAE", 1.4),

    ("vantagewithai/LTX-2.3-Split",
     "audio_vae/ltx-2-3-22b-audio_vae.safetensors",
     "vae",
     "LTX-2.3 audio VAE", 0.36),

    ("vantagewithai/Sulphur-2-Base-Split",
     "lora/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
     "loras",
     "Distilled LoRA (shared between Sulphur + 10Eros)", 3.5),

    ("VasiliyWeb/OmniNFT_ComfyUI",
     "OmniNFT_converted_lora.safetensors",
     "loras",
     "OmniNFT identity LoRA (optional but recommended)", 1.0),

    ("Lightricks/LTX-2.3",
     "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
     "upscale_models",
     "Spatial upscaler v1.1 (newer than the v1.0 we had)", 1.0),
]

total_need = sum(g for _,_,_,_,g in DOWNLOADS)
total_have = sum((COMFY_MODELS / sub / Path(fn).name).stat().st_size / 1e9
                 for _, fn, sub, _, _ in DOWNLOADS
                 if (COMFY_MODELS / sub / Path(fn).name).exists())
print(f"Need ~{total_need:.1f} GB; have ~{total_have:.1f} GB already.")

for repo, fn, sub, note, _ in DOWNLOADS:
    target_dir = COMFY_MODELS / sub
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / Path(fn).name
    if target_path.exists():
        print(f"[skip] {target_path.name} ({target_path.stat().st_size / 1e9:.1f} GB)")
        continue
    print(f"[get ] {note}")
    print(f"       repo={repo} file={fn}")
    p = hf_hub_download(repo_id=repo, filename=fn, local_dir=str(target_dir))
    if Path(p) != target_path:
        shutil.move(p, target_path)
        for d in sorted(target_dir.glob("**/"), reverse=True):
            if d != target_dir and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    print(f"       saved -> {target_path}  ({target_path.stat().st_size / 1e9:.1f} GB)")

# Mirror audio VAE + text encoder layers into models/checkpoints/
# because LTXVAudioVAELoader + LTXAVTextEncoderLoader insist on
# scanning that dir for their ckpt_name options.
CKPT_DIR = COMFY_MODELS / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
for src in [COMFY_MODELS / "vae" / "ltx-2-3-22b-audio_vae.safetensors",
            COMFY_MODELS / "text_encoders" / "ltx-2-3-22b-text_encoder.safetensors"]:
    if src.exists():
        dst = CKPT_DIR / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  mirrored {src.name} into checkpoints/")
        else:
            print(f"  [skip] {dst.name} already in checkpoints/")

print(f"\n10Eros assets ready. Active quant: {QUANT}.")
