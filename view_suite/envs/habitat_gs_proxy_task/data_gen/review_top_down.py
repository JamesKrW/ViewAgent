"""Browser UI for human review of top-down references in three corpora.

The collector scans AI2-THOR, ViewSuite/ScanNet, and Habitat-GS. Within each
corpus it reads every JSONL belonging to path_to_view, view_to_path, and
interactive_view_planning, merges repeated sample references into one item per
scene/image content, and optionally includes scene-level top-down files that are
no longer referenced after filtering.

Selections are persisted immediately to a self-contained JSON file and a flat
CSV companion. The server has no external dependencies beyond Pillow and uses
only Python's standard-library HTTP server.

Example::

    python -m view_suite.envs.habitat_gs_proxy_task.data_gen.review_top_down

For a remote machine, keep the default localhost binding and forward the port::

    ssh -L 8769:127.0.0.1:8769 <host>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import math
import mimetypes
import os
import re
import secrets
import socket
import threading
import webbrowser
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import quote, unquote, urlparse

from PIL import Image, ImageOps

TASKS: tuple[str, ...] = (
    "path_to_view",
    "view_to_path",
    "interactive_view_planning",
)
VERDICTS = {"keep", "reject", "unsure"}
DEFAULT_HTTP_PORT = 8769


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _natural_key(text: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Replace non-standard floating-point JSON values with ``null``."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@dataclass
class SourceReference:
    task: str
    split: str
    jsonl: str
    rows: int = 0


@dataclass
class TopDownItem:
    id: str
    corpus: str
    scene_id: str
    image_path: str
    image_rel: str
    sha256: str | None
    missing: bool
    variant_index: int = 1
    variant_count: int = 1
    equivalent_paths: list[str] = field(default_factory=list)
    sources: list[SourceReference] = field(default_factory=list)
    lineage: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("image_path", None)
        encoded_id = quote(self.id, safe="")
        result["image_url"] = f"/image/{encoded_id}"
        result["thumb_url"] = f"/thumb/{encoded_id}"
        return result


@dataclass
class Inventory:
    data_roots: dict[str, Path]
    items: list[TopDownItem]
    jsonls_scanned: list[str]
    parse_errors: list[str]


def _task_and_split(path: Path) -> tuple[str, str] | None:
    stem = path.stem
    for task in TASKS:
        if stem == task:
            return task, "all"
        prefix = task + "_"
        if stem.startswith(prefix):
            return task, stem[len(prefix) :]
    return None


def collect_top_downs(
    data_root: Path,
    *,
    corpus: str = "habitat_gs",
    include_orphans: bool = True,
) -> Inventory:
    """Collect and deduplicate top-down images across all three task JSONLs."""

    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found for {corpus}: {data_root}")

    # scene -> resolved image path -> source-key -> row count
    candidates: dict[str, dict[Path, dict[tuple[str, str, str], int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    jsonls: list[Path] = []
    parse_errors: list[str] = []

    for task in TASKS:
        jsonls.extend(data_root.glob(f"{task}*.jsonl"))
    jsonls = sorted(
        {path.resolve() for path in jsonls}, key=lambda p: _natural_key(p.name)
    )

    for jsonl_path in jsonls:
        task_split = _task_and_split(jsonl_path)
        if task_split is None:
            continue
        task, split = task_split
        with jsonl_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    pack = (row.get("image_detail") or {}).get("top_down_view") or {}
                    relative = pack.get("path")
                    if not relative:
                        raise KeyError("image_detail.top_down_view.path")
                    image_path = Path(relative)
                    if not image_path.is_absolute():
                        image_path = jsonl_path.parent / image_path
                    image_path = image_path.resolve()
                    scene_id = str(row.get("scene_id") or Path(relative).parts[0])
                    source_key = (task, split, str(jsonl_path.relative_to(data_root)))
                    candidates[scene_id][image_path][source_key] += 1
                except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
                    parse_errors.append(
                        f"{corpus}/{jsonl_path.name}:{line_number}: {exc}"
                    )

    if include_orphans:
        orphan_paths = set(data_root.glob("*/top_down.png"))
        orphan_paths.update(data_root.glob("*/top_down_view.png"))
        for image_path in sorted(orphan_paths):
            candidates[image_path.parent.name][image_path.resolve()]

    items: list[TopDownItem] = []
    for scene_id in sorted(candidates, key=_natural_key):
        # Merge multiple paths only when their bytes are identical. A real conflict
        # becomes a separately reviewable variant instead of being silently dropped.
        content_groups: dict[str, list[Path]] = defaultdict(list)
        path_sources = candidates[scene_id]
        for path in path_sources:
            content_key = _sha256(path) if path.is_file() else f"missing:{path}"
            content_groups[content_key].append(path)

        sorted_groups = sorted(
            content_groups.items(),
            key=lambda pair: _natural_key(min(str(path) for path in pair[1])),
        )
        variant_count = len(sorted_groups)
        for variant_index, (content_key, equivalent_paths) in enumerate(
            sorted_groups, 1
        ):
            equivalent_paths = sorted(equivalent_paths)
            primary = equivalent_paths[0]
            missing = content_key.startswith("missing:")
            digest = None if missing else content_key
            base_id = f"{corpus}::{scene_id}"
            item_id = base_id if variant_count == 1 else f"{base_id}::v{variant_index}"

            source_counts: dict[tuple[str, str, str], int] = defaultdict(int)
            for path in equivalent_paths:
                for source_key, count in path_sources[path].items():
                    source_counts[source_key] += count
            sources = [
                SourceReference(task=task, split=split, jsonl=jsonl, rows=rows)
                for (task, split, jsonl), rows in sorted(source_counts.items())
            ]

            def relative_display(path: Path) -> str:
                try:
                    return str(path.relative_to(data_root))
                except ValueError:
                    return str(path)

            items.append(
                TopDownItem(
                    id=item_id,
                    corpus=corpus,
                    scene_id=scene_id,
                    image_path=str(primary),
                    image_rel=relative_display(primary),
                    sha256=digest,
                    missing=missing,
                    variant_index=variant_index,
                    variant_count=variant_count,
                    equivalent_paths=[
                        relative_display(path) for path in equivalent_paths
                    ],
                    sources=sources,
                    lineage={"round": 1, "kind": "original"},
                )
            )

    return Inventory(
        data_roots={corpus: data_root},
        items=items,
        jsonls_scanned=[f"{corpus}/{path.relative_to(data_root)}" for path in jsonls],
        parse_errors=parse_errors,
    )


def collect_corpora(
    data_roots: dict[str, Path], *, include_orphans: bool = True
) -> Inventory:
    """Combine inventories without merging identically named cross-corpus scenes."""

    combined_items: list[TopDownItem] = []
    combined_jsonls: list[str] = []
    combined_errors: list[str] = []
    resolved_roots: dict[str, Path] = {}
    for corpus, root in data_roots.items():
        inventory = collect_top_downs(
            root, corpus=corpus, include_orphans=include_orphans
        )
        resolved_roots.update(inventory.data_roots)
        combined_items.extend(inventory.items)
        combined_jsonls.extend(inventory.jsonls_scanned)
        combined_errors.extend(inventory.parse_errors)
    return Inventory(
        data_roots=resolved_roots,
        items=combined_items,
        jsonls_scanned=combined_jsonls,
        parse_errors=combined_errors,
    )


def load_candidate_manifest(path: Path, *, expected_round: int) -> Inventory:
    """Load regenerated candidates for review rounds two and three."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Round {expected_round} candidate manifest not found: {path}. "
            "Generate replacements for the previous round's rejects first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_round = int(payload.get("round", expected_round))
    if manifest_round != expected_round:
        raise ValueError(
            f"Manifest says round={manifest_round}, expected round={expected_round}"
        )

    items: list[TopDownItem] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(payload.get("items", []), 1):
        corpus = str(entry["corpus"])
        scene_id = str(entry["scene_id"])
        item_id = str(entry.get("id") or f"{corpus}::{scene_id}")
        if item_id in seen_ids:
            raise ValueError(f"Duplicate item id in {path}: {item_id}")
        seen_ids.add(item_id)
        image_path = Path(entry["image_path"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        image_path = image_path.resolve()
        missing = not image_path.is_file()
        digest = None if missing else _sha256(image_path)
        try:
            image_rel = str(image_path.relative_to(path.parent))
        except ValueError:
            image_rel = str(image_path)
        sources = [
            SourceReference(
                task=str(source.get("task", "regenerated")),
                split=str(source.get("split", f"round_{expected_round}")),
                jsonl=str(source.get("jsonl", path.name)),
                rows=int(source.get("rows", 1)),
            )
            for source in entry.get("sources", [])
        ]
        lineage = dict(entry.get("lineage") or {})
        lineage.setdefault("round", expected_round)
        lineage.setdefault("kind", "regenerated")
        items.append(
            TopDownItem(
                id=item_id,
                corpus=corpus,
                scene_id=scene_id,
                image_path=str(image_path),
                image_rel=image_rel,
                sha256=digest,
                missing=missing,
                variant_index=int(entry.get("variant_index", 1)),
                variant_count=int(entry.get("variant_count", 1)),
                equivalent_paths=[image_rel],
                sources=sources,
                lineage=lineage,
            )
        )

    data_roots = {
        str(corpus): Path(root).expanduser().resolve()
        for corpus, root in (payload.get("data_roots") or {}).items()
    }
    if not data_roots:
        data_roots = {"round_candidates": path.parent}
    items.sort(
        key=lambda item: (_natural_key(item.corpus), _natural_key(item.scene_id))
    )
    return Inventory(
        data_roots=data_roots,
        items=items,
        jsonls_scanned=[str(path)],
        parse_errors=[],
    )


class ReviewStore:
    """Thread-safe, immediately persisted human-review state."""

    def __init__(
        self,
        inventory: Inventory,
        output_path: Path,
        *,
        review_round: int = 1,
        total_rounds: int = 3,
        source_manifest: Path | None = None,
        parent_labels: Path | None = None,
    ):
        self.inventory = inventory
        self.output_path = output_path.expanduser().resolve()
        self.csv_path = self.output_path.with_suffix(".csv")
        self.review_round = int(review_round)
        self.total_rounds = int(total_rounds)
        self.source_manifest = source_manifest.resolve() if source_manifest else None
        self.parent_labels = parent_labels.resolve() if parent_labels else None
        self._lock = threading.RLock()
        self._items = {item.id: item for item in inventory.items}
        self._labels: dict[str, dict[str, Any]] = {}
        self._load()
        self.persist()

    def _load(self) -> None:
        if not self.output_path.is_file():
            return
        try:
            payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read existing review file {self.output_path}: {exc}"
            ) from exc

        for saved in payload.get("items", []):
            item_id = saved.get("id")
            current = self._items.get(item_id)
            if current is None:
                continue
            # An old verdict must not silently follow a regenerated image.
            if saved.get("sha256") != current.sha256:
                self._labels[item_id] = {
                    "verdict": None,
                    "note": saved.get("note", ""),
                    "updated_at": None,
                    "stale_previous_verdict": saved.get("verdict"),
                    "stale_previous_sha256": saved.get("sha256"),
                }
                continue
            self._labels[item_id] = {
                "verdict": saved.get("verdict"),
                "note": saved.get("note", ""),
                "updated_at": saved.get("updated_at"),
            }

    def update(self, item_id: str, verdict: str | None, note: str) -> dict[str, Any]:
        if item_id not in self._items:
            raise KeyError(f"Unknown review item: {item_id}")
        if verdict not in VERDICTS and verdict is not None:
            raise ValueError(f"Invalid verdict: {verdict!r}")
        if len(note) > 4000:
            raise ValueError("note is limited to 4000 characters")
        with self._lock:
            self._labels[item_id] = {
                "verdict": verdict,
                "note": note,
                "updated_at": _utc_now(),
            }
            self.persist()
            return self.item_dict(item_id)

    def item_dict(self, item_id: str) -> dict[str, Any]:
        item = self._items[item_id]
        result = item.public_dict()
        result.update(self._labels.get(item_id, {}))
        result.setdefault("verdict", None)
        result.setdefault("note", "")
        result.setdefault("updated_at", None)
        return result

    def summary(self) -> dict[str, int]:
        counts = {
            "total": len(self._items),
            "keep": 0,
            "reject": 0,
            "unsure": 0,
            "unlabeled": 0,
        }
        for item_id in self._items:
            verdict = self._labels.get(item_id, {}).get("verdict")
            if verdict in VERDICTS:
                counts[verdict] += 1
            else:
                counts["unlabeled"] += 1
        return counts

    def payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "review_round": self.review_round,
                "total_rounds": self.total_rounds,
                "source_manifest": str(self.source_manifest)
                if self.source_manifest
                else None,
                "parent_labels": str(self.parent_labels)
                if self.parent_labels
                else None,
                "data_roots": {
                    corpus: str(root)
                    for corpus, root in self.inventory.data_roots.items()
                },
                "corpora": sorted({item.corpus for item in self.inventory.items}),
                "updated_at": _utc_now(),
                "summary": self.summary(),
                "jsonls_scanned": self.inventory.jsonls_scanned,
                "parse_errors": self.inventory.parse_errors,
                "items": [self.item_dict(item.id) for item in self.inventory.items],
            }

    def persist(self) -> None:
        with self._lock:
            payload = self.payload()
            _atomic_json(self.output_path, payload)
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.csv_path.with_name(self.csv_path.name + ".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "id",
                        "corpus",
                        "scene_id",
                        "verdict",
                        "note",
                        "updated_at",
                        "image_rel",
                        "sha256",
                        "variant_index",
                        "variant_count",
                        "lineage",
                    ),
                )
                writer.writeheader()
                for item in payload["items"]:
                    row = {key: item.get(key) for key in writer.fieldnames}
                    row["lineage"] = json.dumps(
                        _json_safe(item.get("lineage") or {}),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.csv_path)


_REVIEW_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Habitat-GS Top-down Review</title>
<style>
:root { color-scheme: dark; --bg:#101216; --panel:#181b21; --muted:#99a1ad;
  --line:#303641; --keep:#35c46a; --reject:#ff5c64; --unsure:#f0b84b; --accent:#63a6ff; }
* { box-sizing:border-box; }
body { margin:0; height:100vh; overflow:hidden; background:var(--bg); color:#edf1f7;
  font:14px/1.4 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
header { height:74px; display:grid; grid-template-columns:minmax(260px,1fr) auto auto;
  gap:20px; align-items:center; padding:10px 18px; border-bottom:1px solid var(--line); background:#14171c; }
h1 { margin:0; font-size:19px; } .sub { color:var(--muted); font-size:12px; margin-top:3px; }
.progress { min-width:270px; } .bar { height:8px; border-radius:5px; background:#292e37; overflow:hidden; }
.bar > div { height:100%; background:var(--accent); width:0; transition:width .2s; }
.counts { display:flex; gap:10px; margin-top:5px; font-variant-numeric:tabular-nums; font-size:12px; }
.counts .keep{color:var(--keep)} .counts .reject{color:var(--reject)} .counts .unsure{color:var(--unsure)}
.save { min-width:110px; text-align:right; color:var(--muted); }
.layout { height:calc(100vh - 74px); display:grid; grid-template-columns:300px minmax(0,1fr); }
aside { border-right:1px solid var(--line); background:#14171c; display:flex; flex-direction:column; min-height:0; }
.tools { padding:10px; border-bottom:1px solid var(--line); display:grid; gap:8px; }
input, select, textarea { width:100%; color:#edf1f7; background:#20242c; border:1px solid #3b424f;
  border-radius:7px; padding:8px; outline:none; } input:focus,select:focus,textarea:focus{border-color:var(--accent)}
#list { overflow:auto; padding:7px; display:grid; grid-template-columns:1fr 1fr; gap:7px; align-content:start; }
.card { position:relative; border:2px solid transparent; background:#20242a; border-radius:8px;
  padding:4px; cursor:pointer; color:inherit; text-align:left; min-width:0; }
.card:hover { border-color:#687181; } .card.current { border-color:var(--accent); }
.card.keep { box-shadow:inset 0 0 0 2px var(--keep); }.card.reject{box-shadow:inset 0 0 0 2px var(--reject)}
.card.unsure { box-shadow:inset 0 0 0 2px var(--unsure); }
.card img { display:block; width:100%; aspect-ratio:1/1; object-fit:cover; border-radius:5px; background:#08090b; }
.scene { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; padding:4px 2px 1px; font-size:11px; }
.badge { position:absolute; right:7px; top:7px; padding:2px 5px; border-radius:5px; background:#111b; font-size:10px; }
main { min-width:0; min-height:0; display:grid; grid-template-rows:minmax(0,1fr) auto; }
.viewer { min-height:0; position:relative; padding:14px; display:flex; align-items:center; justify-content:center;
  background-color:#0b0d10; background-image:linear-gradient(45deg,#151920 25%,transparent 25%),
  linear-gradient(-45deg,#151920 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#151920 75%),
  linear-gradient(-45deg,transparent 75%,#151920 75%); background-size:28px 28px;
  background-position:0 0,0 14px,14px -14px,-14px 0; }
#hero { max-width:100%; max-height:100%; object-fit:contain; box-shadow:0 8px 35px #000b; image-rendering:auto; }
.missing { font-size:22px; color:var(--reject); }
.bottom { border-top:1px solid var(--line); background:var(--panel); padding:12px 16px 14px; }
.identity { display:flex; justify-content:space-between; gap:18px; align-items:baseline; }
#sceneTitle { font-size:19px; font-weight:700; } #position { color:var(--muted); font-variant-numeric:tabular-nums; }
.meta { color:var(--muted); font-size:12px; margin:4px 0 9px; word-break:break-all; }
.decisions { display:grid; grid-template-columns:repeat(3,minmax(110px,180px)) 90px 90px 1fr; gap:9px; align-items:center; }
button, .download { border:1px solid #414957; border-radius:8px; padding:10px 14px; color:#f4f6fa;
  background:#262b34; cursor:pointer; font-weight:650; text-decoration:none; text-align:center; }
button:hover,.download:hover { filter:brightness(1.15); } button.keep{background:#174f2c;border-color:#2c9a53}
button.reject{background:#61262a;border-color:#b8444b} button.unsure{background:#5b451d;border-color:#a97c28}
button.clear { color:var(--muted); } button:disabled{opacity:.4;cursor:not-allowed}
.keys { color:var(--muted); font-size:11px; margin-top:8px; }
.noteRow { display:grid; grid-template-columns:1fr 290px; gap:12px; margin-top:9px; align-items:start; }
textarea { height:52px; resize:vertical; } details { color:var(--muted); font-size:11px; max-height:68px; overflow:auto; }
@media(max-width:900px){header{grid-template-columns:1fr}.progress,.save{display:none}.layout{grid-template-columns:190px 1fr}
  .decisions{grid-template-columns:repeat(3,1fr)}.decisions .nav,.download{display:none}.noteRow{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <div><h1>Top-down Review · <span id="roundTitle">Round —</span></h1><div class="sub" id="rootText"></div></div>
  <div class="progress"><div class="bar"><div id="barFill"></div></div><div class="counts" id="counts"></div></div>
  <div class="save" id="saveState">Loading…</div>
</header>
<div class="layout">
  <aside>
    <div class="tools">
      <input id="search" placeholder="Search scene…">
      <select id="corpusFilter"><option value="all">All corpora</option></select>
      <select id="filter"><option value="all">All</option><option value="unlabeled">Unlabeled</option>
        <option value="keep">Keep</option><option value="reject">Reject</option><option value="unsure">Unsure</option></select>
      <label><input id="autoAdvance" type="checkbox" checked style="width:auto"> Auto-advance after marking</label>
    </div>
    <div id="list"></div>
  </aside>
  <main>
    <div class="viewer"><img id="hero" alt="top-down"><div class="missing" id="missing" hidden>IMAGE MISSING</div></div>
    <div class="bottom">
      <div class="identity"><div id="sceneTitle">—</div><div id="position"></div></div>
      <div class="meta" id="pathText"></div>
      <div class="decisions">
        <button class="keep" id="keepBtn">K · KEEP</button><button class="reject" id="rejectBtn">R · REJECT</button>
        <button class="unsure" id="unsureBtn">U · UNSURE</button><button class="nav" id="prevBtn">← Prev</button>
        <button class="nav" id="nextBtn">Next →</button><div style="display:flex;gap:8px;justify-content:flex-end">
        <button class="clear" id="clearBtn">X · Clear</button><button id="undoBtn" disabled>Undo</button>
        <a class="download" href="/api/export.csv">CSV</a><a class="download" href="/api/export.json">JSON</a></div>
      </div>
      <div class="noteRow"><textarea id="note" placeholder="Optional note (auto-saved)…"></textarea>
        <details><summary>Sources / hash</summary><pre id="detailsText"></pre></details></div>
      <div class="keys">Hotkeys: K keep · R reject · U unsure · X clear · ←/→ navigate. Decisions save immediately.</div>
    </div>
  </main>
</div>
<script>
let state=null, visible=[], currentId=null, undoStack=[], noteTimer=null, csrfToken=null;
const $=id=>document.getElementById(id);
async function load(){const r=await fetch('/api/state',{cache:'no-store'});state=await r.json();
  csrfToken=state.csrf_token;delete state.csrf_token;
  $('roundTitle').textContent=`Round ${state.review_round} / ${state.total_rounds}`;
  $('rootText').textContent=Object.entries(state.data_roots).map(([k,v])=>`${k}: ${v}`).join(' · ');
  for(const corpus of state.corpora){const o=document.createElement('option');o.value=corpus;o.textContent=corpus;$('corpusFilter').appendChild(o);}
  applyFilter();const hash=decodeURIComponent(location.hash.slice(1));const preferred=state.items.find(x=>x.id===hash)||state.items.find(x=>!x.verdict)||state.items[0];
  if(preferred) selectItem(preferred.id);updateSummary();$('saveState').textContent='Saved';}
function updateSummary(){const s=state.summary,labeled=s.total-s.unlabeled;$('barFill').style.width=(s.total?100*labeled/s.total:0)+'%';
  $('counts').innerHTML=`<span>${labeled}/${s.total}</span><span class="keep">Keep ${s.keep}</span><span class="reject">Reject ${s.reject}</span><span class="unsure">Unsure ${s.unsure}</span><span>Left ${s.unlabeled}</span>`;}
function applyFilter(){const q=$('search').value.trim().toLowerCase(),f=$('filter').value,c=$('corpusFilter').value;
  visible=state.items.filter(x=>(c==='all'||x.corpus===c)&&(!q||(x.corpus+' '+x.scene_id+' '+x.image_rel).toLowerCase().includes(q))&&(f==='all'?(true):f==='unlabeled'?!x.verdict:x.verdict===f));
  const list=$('list');list.innerHTML='';for(const item of visible){const b=document.createElement('button');b.className='card '+(item.verdict||'');b.dataset.id=item.id;
    b.innerHTML=`<img loading="lazy" src="${item.thumb_url}" alt=""><div class="scene" title="${escapeHtml(item.corpus+' · '+item.scene_id)}">${escapeHtml(item.corpus+' · '+item.scene_id)}</div><span class="badge">${item.verdict||'—'}</span>`;
    b.onclick=()=>selectItem(item.id);list.appendChild(b);}refreshCurrentCard();}
function escapeHtml(s){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function current(){return state.items.find(x=>x.id===currentId);}
function selectItem(id){const item=state.items.find(x=>x.id===id);if(!item)return;currentId=id;location.hash=encodeURIComponent(id);$('sceneTitle').textContent=item.corpus+' · '+item.scene_id+(item.variant_count>1?` · variant ${item.variant_index}/${item.variant_count}`:'');
  $('position').textContent=`${state.items.indexOf(item)+1} / ${state.items.length}`;$('pathText').textContent=item.image_rel;
  $('hero').hidden=item.missing;$('missing').hidden=!item.missing;if(!item.missing)$('hero').src=item.image_url;$('note').value=item.note||'';
  $('detailsText').textContent=`corpus: ${item.corpus}\nsha256: ${item.sha256||'MISSING'}\nvariants: ${item.variant_index}/${item.variant_count}\n`+
    item.sources.map(s=>`${s.task}/${s.split}: ${s.rows} rows (${s.jsonl})`).join('\n');refreshCurrentCard();}
function refreshCurrentCard(){document.querySelectorAll('.card').forEach(b=>b.classList.toggle('current',b.dataset.id===currentId));const c=document.querySelector('.card.current');if(c)c.scrollIntoView({block:'nearest'});}
function recalcSummary(){const s={total:state.items.length,keep:0,reject:0,unsure:0,unlabeled:0};for(const x of state.items){x.verdict?s[x.verdict]++:s.unlabeled++;}state.summary=s;updateSummary();}
async function save(id,verdict,note,autoAdvance=false){const item=state.items.find(x=>x.id===id);if(!item)return;$('saveState').textContent='Saving…';
  const r=await fetch('/api/label',{method:'POST',headers:{'content-type':'application/json','x-csrf-token':csrfToken},body:JSON.stringify({id,verdict,note})});
  if(!r.ok){$('saveState').textContent='SAVE ERROR';alert(await r.text());return;}const data=await r.json();Object.assign(item,data.item);state.summary=data.summary;
  $('saveState').textContent='Saved';applyFilter();updateSummary();if(autoAdvance&&$('autoAdvance').checked)nextUnlabeled(id);}
function mark(verdict){const item=current();if(!item)return;undoStack.push({id:item.id,verdict:item.verdict||null,note:item.note||''});$('undoBtn').disabled=false;
  save(item.id,verdict,$('note').value,true);}
function move(delta){if(!visible.length)return;let i=visible.findIndex(x=>x.id===currentId);if(i<0)i=0;selectItem(visible[(i+delta+visible.length)%visible.length].id);}
function nextUnlabeled(afterId){const start=state.items.findIndex(x=>x.id===afterId);for(let n=1;n<=state.items.length;n++){const item=state.items[(start+n)%state.items.length];
  if(!item.verdict&&visible.some(x=>x.id===item.id)){selectItem(item.id);return;}}move(1);}
function scheduleNote(){clearTimeout(noteTimer);const id=currentId,note=$('note').value;noteTimer=setTimeout(()=>{const item=state.items.find(x=>x.id===id);if(item)save(id,item.verdict||null,note,false);},450);}
$('search').oninput=applyFilter;$('filter').onchange=applyFilter;$('corpusFilter').onchange=applyFilter;$('keepBtn').onclick=()=>mark('keep');$('rejectBtn').onclick=()=>mark('reject');
$('unsureBtn').onclick=()=>mark('unsure');$('clearBtn').onclick=()=>mark(null);$('prevBtn').onclick=()=>move(-1);$('nextBtn').onclick=()=>move(1);$('note').oninput=scheduleNote;
$('undoBtn').onclick=()=>{const old=undoStack.pop();if(!old)return;save(old.id,old.verdict,old.note,false);selectItem(old.id);$('undoBtn').disabled=!undoStack.length;};
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;
  if(e.key==='k'||e.key==='K')mark('keep');else if(e.key==='r'||e.key==='R')mark('reject');else if(e.key==='u'||e.key==='U')mark('unsure');
  else if(e.key==='x'||e.key==='X')mark(null);else if(e.key==='ArrowRight')move(1);else if(e.key==='ArrowLeft')move(-1);});
load().catch(e=>{$('saveState').textContent='LOAD ERROR';console.error(e)});
</script>
</body></html>"""


class ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], store: ReviewStore):
        self.store = store
        self.csrf_token = secrets.token_urlsafe(32)
        self.item_paths = {
            item.id: Path(item.image_path) for item in store.inventory.items
        }
        self.thumbnail_cache: dict[str, bytes] = {}
        self.thumbnail_lock = threading.Lock()
        super().__init__(address, ReviewRequestHandler)


class ReviewHTTPServerV6(ReviewHTTPServer):
    address_family = socket.AF_INET6


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server: ReviewHTTPServer
    server_version = "HabitatGSTopDownReviewer/1.0"
    MAX_BODY: ClassVar[int] = 1024 * 1024

    def log_message(self, format_string: str, *args: Any) -> None:
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(format_string, *args)

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        *,
        cache: bool = False,
        disposition: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Cache-Control", "public, max-age=3600" if cache else "no-store"
        )
        if disposition:
            self.send_header("Content-Disposition", disposition)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(
        self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self._send_bytes(
            json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            ),
            "application/json; charset=utf-8",
            status,
        )

    def _item_id(self, prefix: str) -> str:
        return unquote(urlparse(self.path).path[len(prefix) :])

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self._send_bytes(_REVIEW_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if route == "/api/state":
            payload = self.server.store.payload()
            payload["csrf_token"] = self.server.csrf_token
            self._send_json(payload)
            return
        if route == "/api/export.json":
            body = json.dumps(
                _json_safe(self.server.store.payload()),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self._send_bytes(
                body,
                "application/json; charset=utf-8",
                disposition='attachment; filename="topdown_labels.json"',
            )
            return
        if route == "/api/export.csv":
            body = self.server.store.csv_path.read_bytes()
            self._send_bytes(
                body,
                "text/csv; charset=utf-8",
                disposition='attachment; filename="topdown_labels.csv"',
            )
            return
        if route == "/health":
            self._send_json({"ok": True})
            return
        if route.startswith("/image/"):
            self._serve_image(self._item_id("/image/"), thumbnail=False)
            return
        if route.startswith("/thumb/"):
            self._serve_image(self._item_id("/thumb/"), thumbnail=True)
            return
        if route == "/favicon.ico":
            self._send_bytes(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _serve_image(self, item_id: str, *, thumbnail: bool) -> None:
        path = self.server.item_paths.get(item_id)
        if path is None or not path.is_file():
            self._send_json({"error": "image not found"}, HTTPStatus.NOT_FOUND)
            return
        if not thumbnail:
            content_type = (
                mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            )
            self._send_bytes(path.read_bytes(), content_type, cache=True)
            return
        with self.server.thumbnail_lock:
            body = self.server.thumbnail_cache.get(item_id)
            if body is None:
                with Image.open(path) as image:
                    thumb = ImageOps.fit(
                        image.convert("RGB"),
                        (180, 180),
                        method=Image.Resampling.LANCZOS,
                    )
                    buffer = io.BytesIO()
                    thumb.save(buffer, format="JPEG", quality=82, optimize=True)
                    body = buffer.getvalue()
                self.server.thumbnail_cache[item_id] = body
        self._send_bytes(body, "image/jpeg", cache=True)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/label":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            supplied_token = self.headers.get("X-CSRF-Token", "")
            if not hmac.compare_digest(supplied_token, self.server.csrf_token):
                self._send_json({"error": "invalid CSRF token"}, HTTPStatus.FORBIDDEN)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.MAX_BODY:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            item_id = str(payload["id"])
            verdict = payload.get("verdict")
            note = str(payload.get("note", ""))
            item = self.server.store.update(item_id, verdict, note)
            self._send_json({"item": item, "summary": self.server.store.summary()})
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review unique AI2-THOR, ViewSuite, and Habitat-GS top-down images."
    )
    repo_root = _repo_root()
    parser.add_argument("--ai2thor-root", type=Path, default=repo_root / "data/viewagent_ai2thor")
    parser.add_argument(
        "--viewsuite-root", type=Path, default=repo_root / "data/viewagent15k_scannet_open3d"
    )
    parser.add_argument(
        "--habitat-gs-root", type=Path, default=repo_root / "data/viewagent_habitat_gs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <repo>/data/topdown_review/round_<N>_labels.json",
    )
    parser.add_argument("--round", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Regenerated-candidate manifest; required for rounds 2 and 3.",
    )
    parser.add_argument(
        "--parent-labels",
        type=Path,
        default=None,
        help="Previous-round label file recorded as lineage metadata.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_HTTP_PORT,
        help=f"HTTP port (default: {DEFAULT_HTTP_PORT}).",
    )
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--no-orphans",
        action="store_true",
        help="Do not include scene top_down.png files absent from JSONLs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print inventory statistics without starting the server.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    review_root = _repo_root() / "data/topdown_review"
    manifest: Path | None = None
    parent_labels: Path | None = args.parent_labels
    if args.round == 1:
        data_roots = {
            "ai2thor": args.ai2thor_root,
            "viewsuite": args.viewsuite_root,
            "habitat_gs": args.habitat_gs_root,
        }
        inventory = collect_corpora(data_roots, include_orphans=not args.no_orphans)
    else:
        manifest = (
            (args.manifest or review_root / f"round_{args.round}_manifest.json")
            .expanduser()
            .resolve()
        )
        inventory = load_candidate_manifest(manifest, expected_round=args.round)
        if parent_labels is None:
            parent_labels = review_root / f"round_{args.round - 1}_labels.json"
    output = (
        (args.output or review_root / f"round_{args.round}_labels.json")
        .expanduser()
        .resolve()
    )

    scene_count = len({(item.corpus, item.scene_id) for item in inventory.items})
    variants = sum(1 for item in inventory.items if item.variant_count > 1)
    missing = sum(1 for item in inventory.items if item.missing)
    print(f"Top-down inventory: {len(inventory.items)} images, {scene_count} scenes")
    corpus_names = sorted({item.corpus for item in inventory.items})
    for corpus in corpus_names:
        corpus_items = [item for item in inventory.items if item.corpus == corpus]
        corpus_scenes = {(item.corpus, item.scene_id) for item in corpus_items}
        print(f"  {corpus}: {len(corpus_items)} images, {len(corpus_scenes)} scenes")
    if args.round == 1:
        print(
            f"JSONLs scanned: {len(inventory.jsonls_scanned)} across "
            f"{len(corpus_names)} corpora x {len(TASKS)} tasks"
        )
    else:
        print(f"Candidate manifest: {manifest}")
    print(
        f"Conflicting variants: {variants}; missing images: {missing}; parse errors: {len(inventory.parse_errors)}"
    )
    print(f"Auto-save JSON: {output}")
    print(f"Auto-save CSV:  {output.with_suffix('.csv')}")
    if inventory.parse_errors:
        for error in inventory.parse_errors[:10]:
            print(f"  [parse warning] {error}")
        if len(inventory.parse_errors) > 10:
            print(f"  ... and {len(inventory.parse_errors) - 10} more")
    if args.dry_run:
        return 0

    store = ReviewStore(
        inventory,
        output,
        review_round=args.round,
        total_rounds=3,
        source_manifest=manifest,
        parent_labels=parent_labels,
    )
    server_class = ReviewHTTPServerV6 if ":" in args.host else ReviewHTTPServer
    server = server_class((args.host, args.port), store)
    bound_host, bound_port = server.server_address[:2]
    browser_host = "127.0.0.1" if bound_host in {"0.0.0.0", "::"} else bound_host
    if ":" in browser_host:
        browser_host = f"[{browser_host}]"
    url = f"http://{browser_host}:{bound_port}/"
    print(f"Review UI: {url}")
    print("Hotkeys: K keep, R reject, U unsure, X clear, Left/Right navigate")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping reviewer.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
