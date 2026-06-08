"""Drive image2video's /generate/i2v endpoint with a prompt built from
the ai-filmmaking skill (Variant A — text-driven Seedance shot prompt
with optional character-sheet reference image).

Why this script exists: the skill ships paste-ready prompt templates
that are tuned to lock subject identity, lighting, and pacing across
short cinematic clips. image2video produces ~1–4 sec clips, so we
adapt Variant A's TIMELINE to that runtime and submit the rendered
prompt through the webapp like a normal user would. Useful as both a
smoke test and a way to verify the skill's templates translate
cleanly to LTX-2.3 + 10Eros (not just Seedance).

Run:
    conda run -n img2vid python tools/test_prompt_from_skill.py \
        --source comfyui/input/woman_test.png

Default subject = the same Indian-woman portrait used by the existing
probe scripts in out/, so this slots into the established test suite.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests


# -- Variant A skeleton, lifted from .claude/skills/ai-filmmaking/SKILL.md.
# We keep the field names identical to the skill so the rendered prompt
# is a one-to-one realisation of the template — easy to diff against
# the source of truth when the skill evolves.
VARIANT_A_TEMPLATE = """\
FORMAT: {duration_s} seconds / 1 CUT / {genre_tone} / {audio_instruction}

SUBJECT 1: {subject_1}

ENVIRONMENT: {environment}

AUDIO / MOOD: {audio_mood}

TIMELINE (must cover full 0:00–0:{duration_s_padded}):
0:00–0:{half_s}: {beat_a}
0:{half_s}–0:{duration_s_padded}: {beat_b}
"""


def build_test_prompt(duration_s: int = 4) -> str:
    """Fill Variant A for the default test subject (mid-twenties woman,
    soft smile beat). Keep subject description tight — ~30 words per
    the skill's character-lock rule."""
    half = max(1, duration_s // 2)
    return VARIANT_A_TEMPLATE.format(
        duration_s=duration_s,
        duration_s_padded=f"{duration_s:02d}",
        half_s=f"{half:02d}",
        genre_tone="cinematic portrait / intimate / photoreal",
        audio_instruction="NO MUSIC",
        subject_1=(
            "a woman in her mid-twenties, warm brown skin, dark-brown almond eyes, "
            "long dark-brown hair, soft natural-tone makeup, plain dark top. "
            "Identity locked to the attached source image."
        ),
        environment=(
            "soft window-side daylight, neutral cream backdrop, "
            "shallow depth of field, 35mm portrait lens"
        ),
        audio_mood="silent ambient — room tone only",
        beat_a=(
            "Medium close-up, locked frame — face neutral, eyes meet camera, "
            "gentle inhale lifting the shoulders by a fraction"
        ),
        beat_b=(
            "Hold framing — corners of the mouth lift into a soft, warm smile "
            "(no teeth), eyes soften, a slow blink lands on the smile"
        ),
    )


def submit_i2v(base_url: str, source_path: Path, prompt: str,
               width: int, height: int, frames: int, steps: int,
               seed: int) -> str:
    """POST to /generate/i2v, return the new job id (parsed from the
    302 redirect Flask sends back: /job/<id>)."""
    url = f"{base_url.rstrip('/')}/generate/i2v"
    with source_path.open("rb") as f:
        files = {"source": (source_path.name, f, "image/png")}
        data = {
            "prompt": prompt,
            "width": str(width),
            "height": str(height),
            "frames": str(frames),
            "steps": str(steps),
            "seed": str(seed),
        }
        r = requests.post(url, files=files, data=data,
                          allow_redirects=False, timeout=30)
    if r.status_code != 302:
        sys.exit(f"submit failed: HTTP {r.status_code} — {r.text[:400]}")
    job_id = r.headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    return job_id


def poll(base_url: str, job_id: str, timeout_s: int = 1200) -> dict:
    """Poll /status until phase == done or error, or timeout."""
    url = f"{base_url.rstrip('/')}/job/{job_id}/status"
    started = time.time()
    last_phase = ""
    while time.time() - started < timeout_s:
        try:
            j = requests.get(url, timeout=10).json()
        except Exception as e:
            print(f"  poll error: {e}", flush=True)
            time.sleep(5)
            continue
        phase = j.get("phase", "?")
        if phase != last_phase:
            print(f"  [{int(time.time()-started):4d}s] {phase}: {j.get('message','')}", flush=True)
            last_phase = phase
        if phase in ("done", "error"):
            return j
        time.sleep(3)
    sys.exit(f"timeout after {timeout_s}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--source", type=Path, required=True,
                    help="portrait image to animate (PNG/JPG)")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--frames", type=int, default=97,
                    help="output frames (~24fps; 97 ≈ 4s)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duration-s", type=int, default=4,
                    help="seconds of timeline to write into the prompt")
    ap.add_argument("--print-only", action="store_true",
                    help="just emit the rendered prompt, do not submit")
    args = ap.parse_args()

    prompt = build_test_prompt(duration_s=args.duration_s)
    print("=" * 72)
    print("Prompt (Variant A from ai-filmmaking skill):")
    print("=" * 72)
    print(prompt)
    print("=" * 72)

    if args.print_only:
        return

    if not args.source.is_file():
        sys.exit(f"source not found: {args.source}")

    print(f"\nSubmitting to {args.base_url}/generate/i2v ...")
    job_id = submit_i2v(args.base_url, args.source, prompt,
                        args.width, args.height, args.frames,
                        args.steps, args.seed)
    print(f"job_id = {job_id}")
    print(f"viewer = {args.base_url}/job/{job_id}\n")

    final = poll(args.base_url, job_id)
    print()
    print(json.dumps({
        "job_id": job_id,
        "phase": final.get("phase"),
        "error": final.get("error"),
        "has_output": final.get("has_output"),
        "download": f"{args.base_url}/job/{job_id}/download"
                    if final.get("has_output") else None,
    }, indent=2))


if __name__ == "__main__":
    main()
