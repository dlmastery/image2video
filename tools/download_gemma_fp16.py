"""Download canonical Comfy-Org Gemma 3 12B FP16 (22.7 GB).

Vantage tutorial mandates this exact file. The FP4 mixed quant we tried
seems incompatible with DualCLIPLoader type='ltxv' and produces noise.
"""
import truststore; truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil
HERE = Path(__file__).resolve().parent.parent
TE_DIR = HERE / "comfyui" / "models" / "text_encoders"
TE_DIR.mkdir(parents=True, exist_ok=True)
target = TE_DIR / "gemma_3_12B_it.safetensors"
if target.exists():
    print(f"[skip] {target.name} already on disk ({target.stat().st_size/1e9:.1f} GB)")
else:
    print("downloading Comfy-Org/ltx-2/split_files/text_encoders/gemma_3_12B_it.safetensors (~22.7 GB)")
    p = hf_hub_download(
        repo_id="Comfy-Org/ltx-2",
        filename="split_files/text_encoders/gemma_3_12B_it.safetensors",
        local_dir=str(TE_DIR),
    )
    if Path(p) != target:
        shutil.move(p, target)
        for d in sorted(TE_DIR.glob("**/"), reverse=True):
            if d != TE_DIR and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    print(f"saved -> {target} ({target.stat().st_size/1e9:.1f} GB)")
