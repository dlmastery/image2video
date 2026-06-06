"""Download the LTX spatial upscaler and create a placeholder PNG for T2V."""
import os, sys
import truststore; truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
UPSCALE_DIR = HERE / "comfyui" / "models" / "upscale_models"
INPUT_DIR = HERE / "comfyui" / "input"
UPSCALE_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Spatial upscaler. LatentUpscaleModelLoader reads from
# models/upscale_models/ (NOT latent_upscale_models/ despite the name).
print("Downloading LTX spatial upscaler...")
p = hf_hub_download(
    repo_id="Lightricks/LTX-2.3",
    filename="ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    local_dir=str(UPSCALE_DIR),
    local_dir_use_symlinks=False,
)
print(f"  saved: {p}")

# 2. Placeholder PNG (1x1 black). T2V workflows still need LoadImage to
# point at a real file even though LTXVImgToVideoInplace is bypassed.
import cv2, numpy as np
ph = INPUT_DIR / "img2vid_placeholder.png"
cv2.imwrite(str(ph), np.zeros((512, 512, 3), dtype=np.uint8))
print(f"  saved: {ph}")
print("Done.")
