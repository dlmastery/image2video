"""Step-by-step audit of the Vantage 10Eros tutorial vs our install."""
import json, os, requests
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
COMFY = HERE / "comfyui"
GREEN, RED, AMBER = "[OK]", "[XX]", "[~~]"

def have(p):  return p.exists()
def status(b): return GREEN if b else RED

print("=" * 78)
print(" VANTAGE 10EROS WORKFLOW - STEP-BY-STEP AUDIT")
print("=" * 78)

# ----- 1. ComfyUI setup & flags -----
print("\n## 1. ComfyUI runtime flags")
webapp = (HERE / "webapp.py").read_text(encoding="utf-8")
for flag, label in [("--lowvram", "--lowvram (canonical Vantage low-VRAM tip)"),
                    ("--force-fp16", "--force-fp16 (FP16 activations alongside GGUF)")]:
    print(f"  {status(flag in webapp)}  {label}")

# ----- 2. Required custom nodes -----
print("\n## 2. Custom node packs cloned + loading")
required_packs = [
    "ComfyUI-GGUF",
    "ComfyUI-VideoHelperSuite",
    "ComfyUI-KJNodes",
    "10S-Comfy-nodes",
    "ComfyUI-LTXVideo",
    "rgthree-comfy",
    "comfy_mtb",
    "ComfyUI-Custom-Scripts",
    "ComfyUI_Comfyroll_CustomNodes",
    "RES4LYF",
    "Nvidia_RTX_Nodes_ComfyUI",
]
cn_dir = COMFY / "custom_nodes"
cloned = {p.name for p in cn_dir.iterdir() if p.is_dir()} if cn_dir.exists() else set()
for p in required_packs:
    s = GREEN if p in cloned else RED
    note = ""
    if p == "Nvidia_RTX_Nodes_ComfyUI":
        note = "  (IMPORT FAILS - needs nvvfx SDK; workflow bypasses RTXVideoSuperResolution)"
    print(f"  {s}  {p}{note}")

# ----- 3. Required Python deps -----
print("\n## 3. Required Python deps")
for mod, label in [("mediapipe", "mediapipe (10S face detector)"),
                   ("gguf", "gguf (GGUF format loader)"),
                   ("pywt", "PyWavelets (RES4LYF Sigmas Easing)"),
                   ("sageattention", "sageattention (KJ PathchSageAttentionKJ)"),
                   ("triton", "triton-windows (sage CUDA kernels)")]:
    try:
        __import__(mod)
        print(f"  {GREEN}  {label}")
    except ImportError:
        print(f"  {RED}  {label}")

# ----- 4. Model files in correct dirs -----
print("\n## 4. Model files (Vantage's exact paths)")
expected = {
    "unet/10Eros_v1-*.gguf":             "10Eros UNET (any quant)",
    "text_encoders/gemma_3_12B_it*.safetensors":  "Gemma 3 12B text encoder",
    "text_encoders/ltx-2-3-22b-text_encoder.safetensors": "LTX-2.3 base text encoder",
    "vae/ltx-2-3-22b-VAE.safetensors":    "LTX-2.3 visual VAE",
    "vae/ltx-2-3-22b-audio_vae.safetensors": "LTX-2.3 audio VAE",
    "loras/OmniNFT_converted_lora.safetensors": "OmniNFT LoRA",
    "loras/ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors": "distilled LoRA",
    "upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors": "spatial upscaler x2 v1.1",
    "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors": "spatial upscaler (legacy dir mirror)",
}
for rel, label in expected.items():
    base = COMFY / "models" / rel.replace("*", "").rsplit("/", 1)[0]
    pattern = rel.rsplit("/", 1)[1]
    if "*" in pattern:
        matches = list(base.glob(pattern))
    else:
        matches = [base / pattern] if (base / pattern).exists() else []
    if matches:
        sz = matches[0].stat().st_size / 1e9
        print(f"  {GREEN}  {rel}  ({sz:.1f} GB)")
    else:
        print(f"  {RED}  {rel}  - MISSING ({label})")

# ----- 5. Workflow file + converter sanity -----
print("\n## 5. Vantage workflow JSON")
wf_ui = HERE / "workflows" / "Vantage-10Eros_I2V_v3.2.json"
wf_api = HERE / "workflows" / "Vantage-10Eros_I2V_v3.2.api.json"
print(f"  {status(wf_ui.exists())}  source: workflows/Vantage-10Eros_I2V_v3.2.json")
print(f"  {status(wf_api.exists())}  converted: workflows/Vantage-10Eros_I2V_v3.2.api.json")
if wf_api.exists():
    api = json.loads(wf_api.read_text(encoding="utf-8"))
    print(f"        ({len(api)} nodes after UI->API conversion + bypass)")

# ----- 6. Live ComfyUI registration sanity for the key 10Eros nodes -----
print("\n## 6. Critical node classes (live ComfyUI /object_info)")
try:
    info = requests.get('http://127.0.0.1:8188/object_info', timeout=15).json()
    reg = set(info.keys())
    print(f"  ComfyUI is up. Total registered classes: {len(reg)}")
    # 10S forward-hook face consistency
    critical = [
        ("UnetLoaderGGUF",       "GGUF UNET loader"),
        ("VAELoader",            "split VAE"),
        ("DualCLIPLoader",       "dual CLIP for Gemma + LTX text encoder"),
        ("LTXFaceDetector",      "10S face detector"),
        ("LTXLikenessAnchor",    "10S identity anchor"),
        ("LTXLikenessGuide",     "10S identity guide"),
        ("LTXLatentAnchorAware", "10S long-form consistency"),
        ("LTXTiledSampler",      "10S tiled sampler"),
        ("LTXVTiledSampler",     "LTX v-tiled sampler"),
        ("LTXVLatentUpsamplerTiled", "10S latent upsampler"),
        ("LTX2LoraLoaderAdvanced","10S LoRA loader for LTX-2"),
        ("Power Lora Loader (rgthree)", "rgthree power LoRA"),
        ("STGGuiderAdvanced",    "STG guider"),
        ("LTXVAudioVAEEncode",   "audio VAE encode"),
        ("LTXVAudioVAEDecode",   "audio VAE decode"),
    ]
    for cls, label in critical:
        print(f"  {status(cls in reg)}  {cls}  -  {label}")
except Exception as e:
    print(f"  ComfyUI not up: {e}")

# ----- 7. Per-node loader filename sanity in the converted workflow -----
print("\n## 7. Loader filenames in converted workflow (should match disk)")
if wf_api.exists():
    api = json.loads(wf_api.read_text(encoding="utf-8"))
    for nid, n in api.items():
        ct = n.get("class_type", "")
        ins = n.get("inputs", {})
        if ct in ("UnetLoaderGGUF","CheckpointLoaderSimple","UNETLoader",
                  "VAELoader","LatentUpscaleModelLoader",
                  "LTXVAudioVAELoader","LTXAVTextEncoderLoader",
                  "DualCLIPLoader","CLIPLoader","LoraLoaderModelOnly"):
            files = []
            for k in ("unet_name","ckpt_name","vae_name","model_name",
                      "clip_name1","clip_name2","clip_name","text_encoder",
                      "lora_name"):
                if k in ins and isinstance(ins[k], str):
                    files.append(f"{k}={ins[k]}")
            print(f"  node {nid:>4} {ct}:  {', '.join(files)}")

# ----- 8. ComfyUI flags actually present in the running process? -----
print("\n## 8. Vantage recommended settings (defaults in our patcher)")
print(f"  resolution    -> patcher writes whatever the form/job asks for")
print(f"  steps         -> overrides LTXVScheduler.steps (note: workflow may use ManualSigmas instead)")
print(f"  CFG           -> NOT overridden (left at workflow defaults)")
print(f"  sampler       -> NOT overridden (left at workflow defaults)")
print(f"  tiled sampler -> uses workflow defaults (LTXTiledSampler / LTXVTiledSampler nodes)")
print(f"  --lowvram     -> {'enabled' if '--lowvram' in webapp else 'NOT ENABLED'}")
print(f"  --force-fp16  -> {'enabled' if '--force-fp16' in webapp else 'NOT ENABLED'}")

print("\n" + "=" * 78)
print(" AUDIT COMPLETE")
print("=" * 78)
