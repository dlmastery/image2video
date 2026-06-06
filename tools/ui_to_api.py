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


def _is_widget_input(input_type: Any) -> bool:
    """Widget inputs in object_info are either:
      - ['COMBO', {...}]                 (dropdown)
      - ['INT', {...}]                   (number)
      - ['FLOAT', {...}]                 (number)
      - ['STRING', {...}]                (text box)
      - ['BOOLEAN', {...}]               (toggle)
      - [<list of options>, {...}]       (older-style combo)
    Connection inputs use single string types ("CLIP", "MODEL", etc.)
    or list-of-strings without an opts dict.
    """
    if isinstance(input_type, list) and input_type:
        first = input_type[0]
        if isinstance(first, list):
            return True   # list-of-options combo
        if isinstance(first, str) and first in (
            "COMBO", "INT", "FLOAT", "STRING", "BOOLEAN",
        ):
            return True
    return False


def _widget_input_names(class_info: dict) -> list[str]:
    """Ordered list of input names that correspond to widget_values."""
    inp = class_info.get("input", {}) or {}
    order = class_info.get("input_order", {}) or {}
    names: list[str] = []
    # Iterate required then optional, preserving the declared order.
    for section in ("required", "optional"):
        keys = order.get(section)
        if not keys:
            keys = list((inp.get(section) or {}).keys())
        for k in keys:
            v = (inp.get(section) or {}).get(k)
            if _is_widget_input(v):
                names.append(k)
    return names


def convert(ui_workflow: dict, object_info: dict) -> dict:
    """Convert one UI-editor workflow into API format."""
    nodes = ui_workflow.get("nodes") or []
    links = ui_workflow.get("links") or []

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
            # Unknown node class - either a bypassed dependency or a
            # missing custom node. Emit a stub so the rest of the graph
            # references resolve, but log it.
            api[nid] = {"class_type": ctype, "inputs": {}}
            print(f"  [warn] unknown class_type {ctype!r} (node {nid}) "
                  f"- emitting empty inputs; check missing node packs")
            continue

        inputs: dict[str, Any] = {}

        # 1. Widget values (per-node literal config) mapped to named inputs
        widget_vals = n.get("widgets_values") or []
        widget_names = _widget_input_names(info)
        # Some nodes have hidden widgets (e.g. seed control_after_generate)
        # which appear in widgets_values but not in input schema. Truncate.
        for name, val in zip(widget_names, widget_vals):
            inputs[name] = val

        # 2. Connected inputs (from `n.inputs[].link`)
        for slot in (n.get("inputs") or []):
            name = slot.get("name")
            link = slot.get("link")
            if name is None or link is None:
                continue
            src = link_src.get(int(link))
            if src is None:
                continue
            inputs[name] = [src[0], src[1]]

        api[nid] = {"class_type": ctype, "inputs": inputs}

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
