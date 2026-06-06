"""Convert a ComfyUI UI-editor JSON workflow to API format.

UI-editor format (saved by the desktop ComfyUI canvas):
    {
      "nodes": [
        {"id": 6, "type": "ClassName", "widgets_values": [...],
         "inputs": [{"name": "clip", "link": 12, "type": "CLIP"}, ...],
         "outputs": [{"name": "CONDITIONING", "links": [...]}],
         ...},
        ...
      ],
      "links": [[link_id, src_node, src_slot, dst_node, dst_slot, "TYPE"], ...]
    }

API format (accepted by ComfyUI's /prompt endpoint):
    {
      "<node_id>": {"class_type": "ClassName",
                    "inputs": {"text": "...", "clip": ["7", 0]}},
      ...
    }

The conversion needs ComfyUI's node-class schema to map widget_values
to named input keys. Schema comes from /object_info: each class lists
its `input.required` + `input.optional` in declaration order, and the
fields whose declared type is a list-of-options are widgets. We zip
the node's widgets_values against that ordered widget-input list.

Tested against `vantagewithai/LTX2.3-10Eros-Split/Vantage-10Eros_I2V_v3.2.json`.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from typing import Any, Optional
import requests

COMFY_URL_DEFAULT = "http://127.0.0.1:8188"


def fetch_object_info(comfy_url: str = COMFY_URL_DEFAULT) -> dict:
    """Pull the live ComfyUI node schema."""
    r = requests.get(f"{comfy_url}/object_info", timeout=60)
    r.raise_for_status()
    return r.json()


WIDGET_TYPE_NAMES = {
    "COMBO", "INT", "FLOAT", "STRING", "BOOLEAN",
    # ComfyUI v3 introduced these for dynamic/composite widgets that
    # appear in widgets_values just like the classic ones.
    "COMFY_DYNAMICCOMBO_V3", "COMFY_DYNAMICCOMBO",
    "COMFY_MULTILINESTRING_V3", "COMFY_MULTILINESTRING",
}

def _is_widget_input(input_type: Any) -> bool:
    """Widget inputs in object_info are either:
      - ['<TYPENAME>', {...}]            where TYPENAME is in WIDGET_TYPE_NAMES
      - [<list of options>, {...}]       (older-style inline COMBO)
    Connection inputs use single-string declared types ("CLIP", "MODEL",
    "IMAGE", etc.) - those are not widgets.
    """
    if isinstance(input_type, list) and input_type:
        first = input_type[0]
        if isinstance(first, list):
            return True   # inline list-of-options combo
        if isinstance(first, str) and first in WIDGET_TYPE_NAMES:
            return True
    return False


def _widget_input_names(class_info: dict) -> list[str]:
    """Ordered list of input names that correspond to widget_values.

    NOTE: this is *static* - it can't account for dynamic combos
    (COMFY_DYNAMICCOMBO_V3) where the selected option introduces
    further widgets. _map_widgets handles that case properly.
    """
    inp = class_info.get("input", {}) or {}
    order = class_info.get("input_order", {}) or {}
    names: list[str] = []
    for section in ("required", "optional"):
        keys = order.get(section)
        if not keys:
            keys = list((inp.get(section) or {}).keys())
        for k in keys:
            v = (inp.get(section) or {}).get(k)
            if _is_widget_input(v):
                names.append(k)
    return names


def _is_dynamic_combo(spec: Any) -> bool:
    return (isinstance(spec, list) and len(spec) >= 1
            and spec[0] in ("COMFY_DYNAMICCOMBO_V3", "COMFY_DYNAMICCOMBO"))


def _map_widgets(class_info: dict, widget_vals: Any) -> dict[str, Any]:
    """Map a node's widgets_values to {input_name: value}.

    Two shapes are observed in ComfyUI UI exports:
      - list of values, ordered to match the schema's widget inputs
      - dict {input_name: value} (some newer nodes save this way)
    For dict, we just pass through. For list, we walk the schema in
    declaration order, accounting for dynamic combos (where the
    selected option introduces additional widget values).
    """
    inp = class_info.get("input", {}) or {}
    order = class_info.get("input_order", {}) or {}
    # If the workflow already stored dict-shaped widgets, just trust it.
    if isinstance(widget_vals, dict):
        return dict(widget_vals)
    if not isinstance(widget_vals, list):
        return {}
    out: dict[str, Any] = {}
    idx = 0
    for section in ("required", "optional"):
        keys = order.get(section)
        if not keys:
            keys = list((inp.get(section) or {}).keys())
        for k in keys:
            v = (inp.get(section) or {}).get(k)
            if not _is_widget_input(v):
                continue
            if idx >= len(widget_vals):
                return out
            val = widget_vals[idx]
            out[k] = val
            idx += 1
            # COMFY_DYNAMICCOMBO_V3: the selector value stays as-is
            # at the parent key, and each nested input is emitted as a
            # sibling under the DOTTED name `{parent}.{nested_name}`.
            # (Verified by ComfyUI's own validation error message
            # 'Required input is missing: multiplier, input_name:
            # resize_type.multiplier'.)
            if _is_dynamic_combo(v):
                opts = []
                if len(v) >= 2 and isinstance(v[1], dict):
                    opts = v[1].get("options", []) or []
                for opt in opts:
                    if isinstance(opt, dict) and opt.get("key") == val:
                        nested = (opt.get("inputs") or {}).get("required", {}) or {}
                        for nk, nv in nested.items():
                            if _is_widget_input(nv) and idx < len(widget_vals):
                                out[f"{k}.{nk}"] = widget_vals[idx]
                                idx += 1
                        break
    return out


def convert(ui_workflow: dict, object_info: dict) -> dict:
    """Convert one UI-editor workflow into API format."""
    nodes = ui_workflow.get("nodes") or []
    links = ui_workflow.get("links") or []

    # Set of node IDs the user has muted/bypassed in the UI. Any
    # connection input pointing at one of these gets dropped (we never
    # emit the node, so dangling refs would otherwise cause
    # NodeNotFoundError at execution time).
    skipped_node_ids: set[str] = {
        str(n.get("id")) for n in nodes if n.get("mode") in (2, 4)
    }

    # Build link_id -> (src_node_id, src_slot)
    link_src: dict[int, tuple[str, int]] = {}
    for raw in links:
        # Format: [link_id, src_node, src_slot, dst_node, dst_slot, type]
        if not isinstance(raw, list) or len(raw) < 5:
            continue
        link_id, src_node, src_slot, _dst_node, _dst_slot = (
            raw[0], raw[1], raw[2], raw[3], raw[4]
        )
        link_src[int(link_id)] = (str(src_node), int(src_slot))

    api: dict[str, dict] = {}

    for n in nodes:
        nid = str(n.get("id"))
        ctype = n.get("type")
        if not ctype:
            continue
        # Skip pure UI/decorative nodes
        if ctype in ("MarkdownNote", "Note", "PrimitiveNode"):
            continue
        # Skip nodes the user has bypassed in the UI
        if n.get("mode") in (2, 4):  # 2=mute, 4=bypass
            continue

        info = object_info.get(ctype)
        if info is None:
            # Unknown node class - missing custom node. Emit a stub with
            # its FIRST connection-shaped input populated, so the
            # post-conversion bypass pass can find a passthrough source.
            stub_inputs: dict[str, Any] = {}
            for slot in (n.get("inputs") or []):
                name = slot.get("name")
                link = slot.get("link")
                if name is None or link is None:
                    continue
                src = link_src.get(int(link))
                if src is None:
                    continue
                stub_inputs[name] = [src[0], src[1]]
                break  # only the first input is enough for bypass
            api[nid] = {"class_type": ctype, "inputs": stub_inputs}
            print(f"  [warn] unknown class_type {ctype!r} (node {nid}) "
                  f"- queued for bypass via {next(iter(stub_inputs)) if stub_inputs else '(no inputs)'}")
            continue

        # 1. Widget values mapped to named inputs (dynamic-combo aware)
        widget_vals = n.get("widgets_values") or []
        inputs: dict[str, Any] = _map_widgets(info, widget_vals)

        # Special case: rgthree Power Lora Loader. Its widgets_values is
        # a list whose entries are dicts of shape
        #   {"on": bool, "lora": "<path>", "strength": float,
        #    "strengthTwo": float|null}
        # plus some header / placeholder entries we ignore. The API
        # form takes these as lora_1, lora_2, ... slots. Without this
        # the entire LoRA stack (including OmniNFT) gets silently
        # dropped during conversion - which the workflow needs.
        if ctype == "Power Lora Loader (rgthree)" and isinstance(widget_vals, list):
            slot = 0
            for wv in widget_vals:
                if isinstance(wv, dict) and "lora" in wv and "strength" in wv:
                    lora_path = wv.get("lora") or ""
                    # Strip ltx23\ or other subdir prefixes - we keep
                    # all LoRAs flat in models/loras/.
                    if "\\" in lora_path:
                        lora_path = lora_path.rsplit("\\", 1)[-1]
                    if "/" in lora_path:
                        lora_path = lora_path.rsplit("/", 1)[-1]
                    slot += 1
                    inputs[f"lora_{slot}"] = {
                        "on": bool(wv.get("on", True)),
                        "lora": lora_path,
                        "strength": float(wv.get("strength", 1.0)),
                        "strengthTwo": wv.get("strengthTwo"),
                    }

        # 2. Connected inputs (from `n.inputs[].link`). Drop any whose
        # source is a muted/bypassed node - those nodes were never
        # emitted, so a reference to them would NodeNotFoundError at
        # execution. Switch-like consumers (Any Switch rgthree) handle
        # the missing-input case fine; nodes that strictly require an
        # input will surface a clearer validation error.
        for slot in (n.get("inputs") or []):
            name = slot.get("name")
            link = slot.get("link")
            if name is None or link is None:
                continue
            src = link_src.get(int(link))
            if src is None:
                continue
            if src[0] in skipped_node_ids:
                continue
            inputs[name] = [src[0], src[1]]

        api[nid] = {"class_type": ctype, "inputs": inputs}

    # Post-pass: bypass nodes whose class_type isn't registered in ComfyUI.
    # For each unknown node, find its first connection-shaped input
    # (assumed to be its primary passthrough) and rewrite every
    # downstream reference to point directly at that source. Then drop
    # the unknown node. Lets us tolerate missing custom node packs as
    # long as the missing nodes are passthrough-shaped (single
    # significant input/output, like RTXVideoSuperResolution which is
    # a SR pass that we can simply skip).
    unknown = [nid for nid, n in api.items() if n["class_type"] not in object_info]
    if unknown:
        passthrough_src: dict[str, list] = {}
        for nid in unknown:
            for v in api[nid]["inputs"].values():
                if isinstance(v, list) and len(v) >= 2:
                    passthrough_src[nid] = v
                    break
        for n in api.values():
            for k, v in list(n["inputs"].items()):
                if isinstance(v, list) and len(v) >= 1:
                    src = str(v[0])
                    if src in passthrough_src:
                        n["inputs"][k] = passthrough_src[src]
        # Compute which nodes are still consumed by anyone.
        def _consumers(target: str) -> int:
            cnt = 0
            for n in api.values():
                for v in n.get("inputs", {}).values():
                    if isinstance(v, list) and len(v) >= 1 and str(v[0]) == target:
                        cnt += 1
            return cnt
        for nid in unknown:
            if nid in passthrough_src:
                api.pop(nid, None)
                print(f"  [bypass] dropped unknown {nid} and rewired downstream")
            elif _consumers(nid) == 0:
                # Orphan UI helper (e.g. rgthree Fast Groups Bypasser).
                # Nothing consumes its output; safe to drop entirely.
                api.pop(nid, None)
                print(f"  [drop ] orphan unknown {nid} (no consumers, no inputs)")
            else:
                # Unknown class WITH consumers but no bypassable input.
                # Will fail at submit; leave a stub for debugging.
                print(f"  [warn ] unknown {nid} has consumers but no input to bypass through")

    return api


def convert_file(in_path: str, out_path: Optional[str] = None,
                 comfy_url: str = COMFY_URL_DEFAULT) -> dict:
    """Read a UI workflow file, fetch object_info, write API format."""
    ui = json.loads(Path(in_path).read_text(encoding="utf-8"))
    if "nodes" not in ui:
        # Already API format - pass through.
        if out_path:
            Path(out_path).write_text(json.dumps(ui, indent=2), encoding="utf-8")
        return ui
    info = fetch_object_info(comfy_url)
    api = convert(ui, info)
    if out_path:
        Path(out_path).write_text(json.dumps(api, indent=2), encoding="utf-8")
    return api


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools/ui_to_api.py <ui_workflow.json> [out.json]")
        sys.exit(2)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".json", ".api.json")
    api = convert_file(inp, out)
    print(f"converted {len(api)} nodes -> {out}")
