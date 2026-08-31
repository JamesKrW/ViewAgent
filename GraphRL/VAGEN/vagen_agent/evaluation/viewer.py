"""Offline browser viewer for standalone VAGEN evaluation trajectories."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

_PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VAGEN evaluation</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0d1117;color:#d7e0e8;font:14px ui-monospace,monospace}
header{padding:13px 18px;border-bottom:1px solid #30363d}main{display:grid;grid-template-columns:360px 1fr;min-height:calc(100vh - 48px)}
aside{padding:12px;border-right:1px solid #30363d;overflow:auto}.item{padding:8px;margin:5px 0;border:1px solid #30363d;background:#161b22;cursor:pointer}
.item:hover{border-color:#58a6ff}.ok{color:#3fb950}.bad{color:#f85149}.muted{color:#8b949e}#detail{padding:18px;overflow:auto}
.turn{border:1px solid #30363d;margin:14px 0;padding:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pane{min-width:0;background:#161b22;padding:12px}
pre{white-space:pre-wrap;word-break:break-word;background:#090c10;padding:9px;max-height:430px;overflow:auto}.frames{display:flex;gap:8px;flex-wrap:wrap}.frames img{max-width:100%;max-height:50vh}
@media(max-width:1000px){main{grid-template-columns:1fr}.grid{grid-template-columns:1fr}}</style></head><body>
<header><b>VAGEN evaluation trajectories</b> <span class=muted id=root></span></header><main><aside><h3>Runs</h3><div id=runs></div><h3>Episodes</h3><div id=jobs></div></aside><section id=detail>Select a run.</section></main>
<script>
const esc=x=>{const d=document.createElement('div');d.textContent=String(x??'');return d.innerHTML};const fmt=x=>typeof x==='number'?x.toFixed(4):'?';let runs=[];
async function get(p){const r=await fetch(p);const x=await r.json();if(!r.ok)throw Error(x.error);return x}
async function boot(){const x=await get('/api/runs');runs=x.runs;document.getElementById('root').textContent=x.root;document.getElementById('runs').innerHTML=runs.map((r,i)=>`<div class=item onclick="openRun(${i})"><b>${esc(r.id)}</b><br><span class=muted>${esc(r.recorded)} recorded</span></div>`).join('')}
async function openRun(i){const r=runs[i],x=await get('/api/run?id='+encodeURIComponent(r.id));document.getElementById('jobs').innerHTML=x.jobs.map(j=>`<div class=item onclick="openJob('${encodeURIComponent(r.id)}','${encodeURIComponent(j.path)}')"><b>${esc(j.model)} · ${esc(j.tag)} · seed ${esc(j.seed)}</b><br><span class=${j.status==='completed'?'ok':'bad'}>${esc(j.status)}</span> · return ${fmt(j.return)} · ${esc(j.harness)}</div>`).join('');document.getElementById('detail').innerHTML=`<h2>${esc(r.id)}</h2><pre>${esc(JSON.stringify(x.summary,null,2))}</pre>`}
function frames(m,run,job){return ((m||{}).images||[]).filter(x=>x.path).map(x=>`<img src="/blob?run=${encodeURIComponent(run)}&job=${encodeURIComponent(job)}&path=${encodeURIComponent(x.path)}">`).join('')}
function message(m,run,job){return `<div><b>${esc(m.role)}</b><div class=frames>${frames(m,run,job)}</div><pre>${esc(JSON.stringify(m.content,null,2))}</pre></div>`}
async function openJob(run,job){run=decodeURIComponent(run);job=decodeURIComponent(job);const x=await get('/api/job?run='+encodeURIComponent(run)+'&job='+encodeURIComponent(job));document.getElementById('detail').innerHTML=`<h2>${esc(x.model)} · ${esc(x.tag)} · seed ${esc(x.seed)}</h2><p>Status ${esc(x.status)} · finish ${esc(x.finish_reason)} · return ${fmt(x.return)} · success ${esc(x.success)} · format ${esc(x.format_compliance)}</p>`+(x.trajectory||[]).map(c=>{const t=c.transition||{};const r=c.response||{};return `<article class=turn><h3>Call ${esc(c.call_id)} [${esc(c.model_role||'default')}] · ${esc(r.finish_reason)}</h3><div class=grid><section class=pane><h4>Model input</h4>${(c.messages||[]).map(m=>message(m,run,job)).join('')}<h4>Reasoning</h4><pre>${esc(r.reasoning_content||'')}</pre><h4>Final content</h4><pre>${esc(r.content||'')}</pre><details><summary>Model metadata</summary><pre>${esc(JSON.stringify(r,null,2))}</pre></details></section><section class=pane><h4>Environment before</h4><div class=frames>${frames(t.observation,run,job)}</div><pre>${esc(JSON.stringify((t.observation||{}).content,null,2))}</pre><p>reward ${esc(t.reward)} · terminated ${esc(t.terminated)} · truncated ${esc(t.truncated)}</p><h4>Parsed/native action</h4><pre>${esc(JSON.stringify({parsed:t.parsed_action,native:t.native_action},null,2))}</pre><details><summary>Official info</summary><pre>${esc(JSON.stringify(t.info||{},null,2))}</pre></details><h4>Environment after</h4><div class=frames>${frames(t.next_observation,run,job)}</div></section></div></article>`}).join('')}
boot().catch(e=>document.getElementById('detail').textContent=String(e));
</script></body></html>"""


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


class Index:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def runs(self) -> list[dict[str, Any]]:
        values = []
        for manifest_path in sorted(self.root.glob("*/evaluation_manifest.json")):
            run_root = manifest_path.parent
            summary = _load(run_root / "summary.json", {})
            values.append({
                "id": run_root.name,
                "recorded": summary.get("recorded", 0),
            })
        return values

    def run(self, run_id: str) -> dict[str, Any]:
        root = self._inside(self.root / run_id)
        jobs = []
        for path in sorted(root.glob("*/tag_*/seed_*/result.json")):
            result = _load(path, {})
            jobs.append({
                "path": str(path.parent.relative_to(root)),
                "model": result.get("model"),
                "tag": result.get("tag"),
                "seed": result.get("seed"),
                "harness": result.get("harness"),
                "status": result.get("status"),
                "return": result.get("return"),
            })
        return {"summary": _load(root / "summary.json", {}), "jobs": jobs}

    def job(self, run_id: str, relative: str) -> dict[str, Any]:
        root = self._inside(self.root / run_id)
        return _load(self._inside(root / relative) / "result.json", {})

    def blob(self, run_id: str, job: str, relative: str) -> Path:
        root = self._inside(self.root / run_id)
        return self._inside(root / job / relative)

    def _inside(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("path escapes viewer root")
        return resolved


def make_handler(index: Index):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, value: Any, status: int = 200) -> None:
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    body = _PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif parsed.path == "/api/runs":
                    self._json({"root": str(index.root), "runs": index.runs()})
                elif parsed.path == "/api/run":
                    self._json(index.run(query["id"][0]))
                elif parsed.path == "/api/job":
                    self._json(index.job(query["run"][0], query["job"][0]))
                elif parsed.path == "/blob":
                    path = index.blob(query["run"][0], query["job"][0], query["path"][0])
                    body = path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)}, 400)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/eval")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8910)
    args = parser.parse_args(argv)
    index = Index(Path(args.root))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(index))
    print(f"VAGEN evaluation viewer: http://{args.host}:{args.port}/?root={quote(str(index.root))}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()


__all__ = ["Index", "main", "make_handler"]
