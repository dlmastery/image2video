"""Add SaveImage probes at each major stage of the Vantage 10Eros workflow.

We don't modify the source JSON - we deep-copy the patched workflow, add
SaveImage nodes wired to key intermediate outputs, and submit that.

Probes added:
  - load_image:   after LoadImage (the source frame)
  - resized:      after ImageResizeKJv2 (model input dims)
  - vae_decode:   after VAEDecode (final pixel-space frames from refined latent)

For latents we use VAEDecodeTiled + SaveImage to get pixel-space.
"""
import json, requests, copy, sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("wapp", os.path.join(HERE, "webapp.py"))
m = importlib.util.module_from_spec(spec); sys.modules["wapp"]=m; spec.loader.exec_module(m)

import cv2, numpy as np
img = cv2.imread(os.path.join(HERE, "comfyui/input/woman_test.png"))
cv2.imwrite(os.path.join(HERE, "comfyui/input/woman_test_tiny.png"),
            cv2.resize(img, (256, 384)))  # match smoke dims
print(f"smoke input: comfyui/input/woman_test_tiny.png  size=(256,384)")

# Override source image in the job
import shutil
shutil.copy2(os.path.join(HERE, "comfyui/input/woman_test_tiny.png"),
             os.path.join(HERE, "comfyui/input/smoke_start.png"))

wf = json.load(open(os.path.join(HERE,
    "workflows/Vantage-10Eros_I2V_v3.2.api.json"), encoding="utf-8"))
job = m.Job(id="probe", mode="i2v",
            prompt="slow gentle smile forming, breathing motion",
            negative_prompt="blurry, distorted",
            width=256, height=384, frames=17, steps=4, seed=42,
            source_image=os.path.join(HERE, "comfyui/input/smoke_start.png"))
patched = m._patch_workflow(wf, job)

# Find key nodes by class
load_image = next((nid for nid, n in patched.items()
                   if n.get("class_type") == "LoadImage"), None)
resize    = next((nid for nid, n in patched.items()
                   if n.get("class_type") == "ImageResizeKJv2"), None)
vaedecode = next((nid for nid, n in patched.items()
                   if n.get("class_type") in ("VAEDecode","LTXVTiledVAEDecode",
                                              "LTXVSpatioTemporalTiledVAEDecode")), None)
print(f"key nodes: LoadImage={load_image}, ImageResizeKJv2={resize}, "
      f"VAEDecode={vaedecode}")

# Insert SaveImage probes
def add_save(api, src_node, src_slot, prefix):
    nid = f"PROBE_{prefix}"
    api[nid] = {"class_type": "SaveImage",
                "inputs": {"images": [src_node, src_slot],
                           "filename_prefix": f"PROBE/{prefix}"}}
    return nid

if load_image:
    add_save(patched, load_image, 0, "01_load")
if resize:
    add_save(patched, resize, 0, "02_resize")
if vaedecode:
    add_save(patched, vaedecode, 0, "03_vaedecode")

# Submit
r = requests.post("http://127.0.0.1:8188/prompt",
                  json={"prompt": patched, "client_id": "probe"}, timeout=30)
print(f"submit: {r.status_code}")
if r.status_code < 400:
    pid = r.json().get("prompt_id")
    print(f"prompt_id={pid}")
    # Poll history until done
    import time
    for i in range(150):
        time.sleep(8)
        h = requests.get(f"http://127.0.0.1:8188/history/{pid}", timeout=10).json()
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed"):
                print(f"[done] after {i*8+8}s")
                # Find probe outputs
                outs = h[pid].get("outputs", {})
                for nid, o in outs.items():
                    if nid.startswith("PROBE_"):
                        imgs = o.get("images", [])
                        for im in imgs:
                            print(f"  {nid}: {im['filename']}")
                break
            if st.get("status_str") == "error":
                print(f"[error] {st}")
                break
        # Also show queue progress
        q = requests.get("http://127.0.0.1:8188/queue", timeout=5).json()
        running = q.get("queue_running") or []
        if running:
            print(f"  [{i*8+8}s] running...")
else:
    print(f"submit failed: {r.text[:500]}")
