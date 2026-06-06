"""Comprehensive end-to-end audit. Checks every layer:
  1. Vantage canonical file paths
  2. ComfyUI running on the 4090
  3. Required node classes registered
  4. Patcher writes prompt correctly
  5. Patcher rewrites loader filenames
  6. Power Lora Loader emits OmniNFT
  7. Converted workflow POSTs to /prompt with no errors
"""
import json, os, sys, requests, importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
COMFY = HERE / "comfyui"
OK, BAD, WARN = "[OK]", "[XX]", "[!!]"

issues: list[str] = []
def fail(msg): issues.append(msg); print(f"  {BAD}  {msg}")
def ok(msg):   print(f"  {OK}  {msg}")
def warn(msg): print(f"  {WARN}  {msg}")

print("=" * 78)
print(" image2video FULL AUDIT (10Eros / Vantage)")
print("=" * 78)

# 1. Vantage canonical files
print("\n## 1. Canonical files (Vantage tutorial exact paths)")
expected = [
    ("unet",              "10Eros_v1-*.gguf",                                       "10Eros UNET GGUF"),
    ("text_encoders",     "gemma_3_12B_it*.safetensors",                            "Gemma 3 12B text encoder"),
    ("text_encoders",     "ltx-2-3-22b-text_encoder.safetensors",                   "LTX-2.3 base text encoder"),
    ("vae",               "ltx-2-3-22b-VAE.safetensors",                            "LTX-2.3 visual VAE"),
    ("vae",               "ltx-2-3-22b-audio_vae.safetensors",                      "LTX-2.3 audio VAE"),
    ("loras",             "OmniNFT_converted_lora.safetensors",                     "OmniNFT identity LoRA"),
    ("loras",             "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
                                                                                    "distilled LoRA"),
    ("latent_upscale_models", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",        "spatial upscaler v1.1"),
]
for sub, pat, label in expected:
    base = COMFY / "models" / sub
    if "*" in pat:
        matches = list(base.glob(pat))
    else:
        matches = [base / pat] if (base / pat).exists() else []
    if matches:
        sz = matches[0].stat().st_size / 1e9
        ok(f"models/{sub}/{matches[0].name}  ({sz:.1f} GB)")
    else:
        fail(f"models/{sub}/{pat}  MISSING ({label})")

# 2. Webapp + ComfyUI alive
print("\n## 2. ComfyUI runtime")
try:
    sys_stats = requests.get("http://127.0.0.1:8188/system_stats", timeout=5).json()
    devs = sys_stats.get("devices") or []
    if devs:
        first = devs[0]
        name = first.get("name") or first.get("type") or "?"
        if "4090" in name:
            ok(f"CUDA device = {name}")
        else:
            fail(f"CUDA device = {name} (expected NVIDIA RTX 4090)")
        vram_gb = (first.get("vram_total") or 0) / 1e9
        ok(f"VRAM available = {vram_gb:.1f} GB")
    else:
        fail("ComfyUI has no devices registered")
except Exception as e:
    fail(f"ComfyUI not reachable: {e}")
    print("\n  (cannot continue without ComfyUI; restart webapp.py)")
    sys.exit(1)

# 3. Required node classes registered
print("\n## 3. Critical node classes (live /object_info)")
info = requests.get("http://127.0.0.1:8188/object_info", timeout=15).json()
reg = set(info.keys())
critical = [
    "UnetLoaderGGUF", "VAELoader", "DualCLIPLoader",
    "LTXFaceDetector", "LTXLikenessAnchor", "LTXLikenessGuide", "LTXLatentAnchorAware",
    "LTXTiledSampler", "LTXVTiledSampler", "LTXVLatentUpsamplerTiled",
    "LTX2LoraLoaderAdvanced", "Power Lora Loader (rgthree)",
    "STGGuiderAdvanced", "LTXVAudioVAEDecode", "LTXVImgToVideoInplaceKJ",
]
for cls in critical:
    if cls in reg:
        ok(cls)
    else:
        fail(f"{cls} NOT REGISTERED")

# 4. Patcher writes prompt correctly
print("\n## 4. Patcher unit test (prompt landing in CLIPTextEncode)")
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("wapp", str(HERE / "webapp.py"))
m = importlib.util.module_from_spec(spec); sys.modules["wapp"] = m; spec.loader.exec_module(m)
wf = json.load(open(HERE / "workflows" / "Vantage-10Eros_I2V_v3.2.api.json", encoding="utf-8"))
job = m.Job(id="audit", mode="i2v",
            prompt="AUDIT_POSITIVE", negative_prompt="AUDIT_NEGATIVE",
            width=256, height=384, frames=17, steps=4, seed=1)
patched = m._patch_workflow(wf, job)

clip_nodes = [(nid, n) for nid, n in patched.items() if n.get("class_type") == "CLIPTextEncode"]
pos_found = any("AUDIT_POSITIVE" in str(n["inputs"].get("text","")) for _, n in clip_nodes)
neg_found = any("AUDIT_NEGATIVE" in str(n["inputs"].get("text","")) for _, n in clip_nodes)
if pos_found:  ok(f"positive prompt 'AUDIT_POSITIVE' reaches a CLIPTextEncode")
else:          fail("positive prompt was DROPPED by patcher (model gets empty conditioning)")
if neg_found:  ok(f"negative prompt 'AUDIT_NEGATIVE' reaches a CLIPTextEncode")
else:          warn("negative prompt was DROPPED (less critical, just less guidance)")

# 5. Patcher rewrites loader filenames
print("\n## 5. Loader filename remaps")
def find_class(p, ct):
    return [(nid, n) for nid, n in p.items() if n.get("class_type") == ct]
checks = [
    ("UnetLoaderGGUF",        "unet_name",   m.GGUF_NAME),
    ("DualCLIPLoader",        "clip_name1",  m.TEXT_ENCODER),
    ("DualCLIPLoader",        "clip_name2",  "ltx-2-3-22b-text_encoder.safetensors"),
    ("VAELoader",             "vae_name",    m.VAE_NAME),
    ("LatentUpscaleModelLoader", "model_name", m.UPSCALER_NAME),
]
for cls, key, want in checks:
    nodes = find_class(patched, cls)
    if not nodes:
        warn(f"no {cls} node in patched workflow")
        continue
    actuals = [n["inputs"].get(key) for _, n in nodes]
    if want in actuals:
        ok(f"{cls}.{key} = {want!r}")
    else:
        fail(f"{cls}.{key} actuals={actuals!r}, expected {want!r}")

# 6. Power Lora Loader emits OmniNFT
print("\n## 6. Power Lora Loader (OmniNFT identity LoRA)")
plls = find_class(patched, "Power Lora Loader (rgthree)")
if not plls:
    fail("no Power Lora Loader in patched workflow")
else:
    for nid, n in plls:
        loras_found = []
        for k, v in n["inputs"].items():
            if k.startswith("lora_") and isinstance(v, dict):
                loras_found.append((k, v.get("lora"), v.get("strength"), v.get("on")))
        if loras_found:
            for k, name, strength, on in loras_found:
                ok(f"node {nid} {k}: {name!r} strength={strength} on={on}")
        else:
            fail(f"node {nid} Power Lora Loader has ZERO lora_X slots (OmniNFT not loaded)")

# 7. Submit patched workflow to /prompt
print("\n## 7. Live submission to /prompt")
try:
    r = requests.post("http://127.0.0.1:8188/prompt",
                      json={"prompt": patched, "client_id": "audit"}, timeout=30)
    if r.status_code < 400:
        j = r.json()
        ok(f"queued: prompt_id={j.get('prompt_id')}")
        if j.get("node_errors"):
            for nid, e in j["node_errors"].items():
                fail(f"node {nid}: {e}")
        # Cancel so we don't actually run an audit job
        try:
            requests.post("http://127.0.0.1:8188/queue",
                          json={"delete": [j["prompt_id"]]}, timeout=5)
        except Exception: pass
    else:
        fail(f"/prompt {r.status_code}: {r.text[:400]}")
except Exception as e:
    fail(f"submit failed: {e}")

# Summary
print("\n" + "=" * 78)
if not issues:
    print(" AUDIT PASSED - everything wired correctly.")
else:
    print(f" AUDIT FOUND {len(issues)} ISSUE(S):")
    for i in issues: print(f"   * {i}")
print("=" * 78)
