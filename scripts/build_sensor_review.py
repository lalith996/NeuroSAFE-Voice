"""Build a visual review page for high-priority TORGO sensor candidates."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/processed/torgo_sensor_audit.csv"
PAIRS = ROOT / "data/processed/torgo_aai_verified_pairs.csv"
OUTPUT = ROOT / "reports/torgo_sensor_review.html"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sparkline(path: Path, max_points: int = 320) -> str:
    track = np.memmap(path, dtype="<f4", mode="r").reshape(-1, 12, 7)
    steps = np.linalg.norm(np.diff(track[:, :, :3], axis=0), axis=2)
    peak = np.max(steps, axis=1)
    stride = max(1, math.ceil(len(peak) / max_points))
    padded = np.pad(peak, (0, (-len(peak)) % stride))
    bins = padded.reshape(-1, stride).max(axis=1)
    width, height = 600, 110
    cap = max(20.0, float(np.max(bins)))
    scale = math.log1p(cap)
    points = []
    for i, value in enumerate(bins):
        x = i * (width - 1) / max(1, len(bins) - 1)
        y = height - 1 - math.log1p(float(value)) / scale * (height - 1)
        points.append(f"{x:.1f},{y:.1f}")
    threshold = height - 1 - math.log1p(20.0) / scale * (height - 1)
    return (f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Peak sensor position change per five millisecond frame; dashed line is 20 recorded units">'
            f'<line x1="0" y1="{threshold:.1f}" x2="{width}" y2="{threshold:.1f}" '
            f'stroke="#d44" stroke-dasharray="5 4"/>'
            f'<polyline fill="none" stroke="#236d93" stroke-width="1.5" points="{" ".join(points)}"/>'
            '</svg>')


def main() -> None:
    audit = [r for r in read_csv(AUDIT) if r["review_priority"] == "high"]
    audit.sort(key=lambda r: (-int(r["steps_over_20"]), r["speaker"], r["session"], r["utterance_id"]))
    pairs = {(r["speaker"], r["session"], r["utterance_id"]): r for r in read_csv(PAIRS)}
    cards = []
    export_rows = []
    for index, row in enumerate(audit):
        pair = pairs[(row["speaker"], row["session"], row["utterance_id"])]
        audio = "../" + pair["head_audio_path"]
        identifier = f'{row["speaker"]} / {row["session"]} / {row["utterance_id"]}'
        cards.append(
            f'<article class="card" data-speaker="{html.escape(row["speaker"])}">'
            f'<h2>{html.escape(identifier)}</h2>'
            f'<p>{html.escape(row["condition"])} · {row["frames"]} frames · '
            f'{row["steps_over_20"]} steps over 20 units · max {row["max_step_units"]} units</p>'
            f'<p>Flat channels: {html.escape(row["flat_sensors"] or "none")}; '
            f'normally active but flat: {html.escape(row["normally_active_flat_sensors"] or "none")}; '
            f'large-step channels: {html.escape(row["large_step_sensors"] or "none")}</p>'
            f'{sparkline(ROOT / row["pos_path"])}'
            f'<p class="hint">Peak change across 12 channels by time, on a logarithmic scale. '
            f'The dashed line marks 20 recorded units per 5 ms. Downsampling retains peaks.</p>'
            f'<audio controls preload="none" src="{html.escape(audio, quote=True)}"></audio>'
            f'<label>Decision <select data-i="{index}" data-field="decision">'
            '<option value="unreviewed">Unreviewed</option><option value="retain">Retain</option>'
            '<option value="repair">Needs repair</option><option value="exclude">Exclude</option>'
            '<option value="inspect_full_trace">Inspect full trace</option></select></label>'
            f'<label>Reason <textarea data-i="{index}" data-field="reason" '
            'placeholder="Record evidence for the decision"></textarea></label>'
            '</article>'
        )
        export_rows.append({
            "speaker": row["speaker"], "session": row["session"],
            "utterance_id": row["utterance_id"], "pos_path": row["pos_path"],
            "review_flag": row["review_flag"], "steps_over_20": row["steps_over_20"],
            "decision": "unreviewed", "reason": "",
        })
    script_data = json.dumps(export_rows, ensure_ascii=True).replace("</", "<\\/")
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TORGO high-priority sensor review</title><style>
body{font:16px system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 18px;color:#17202a;background:#f4f7f9}
p{line-height:1.4}.card{background:white;border:1px solid #d6e0e6;border-radius:10px;padding:18px;margin:16px 0}.card h2{font-size:18px;margin:0}
svg{width:100%;height:auto;background:#f9fbfc;border:1px solid #ddd}audio{width:100%}.hint{font-size:13px;color:#526272}
label{display:block;margin-top:12px;font-weight:600}textarea{display:block;width:100%;min-height:48px;box-sizing:border-box}
select,textarea,button{font:inherit;padding:8px;border:1px solid #bdc8d2;border-radius:6px}button{background:#145b78;color:white;border:0;cursor:pointer}
</style></head><body><h1>TORGO high-priority sensor review</h1>
<p>These are candidate artifacts, not rejected recordings. Inspect the plot and, for a final keep/repair/exclude decision, examine the full sensor trajectories and source context. Export decisions to preserve them.</p>
<button id="export">Export decisions CSV</button> <span id="count"></span>
CARDS_PLACEHOLDER
<script>
const rows=DATA_PLACEHOLDER;
const key='torgo-sensor-review-v1';let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
for(const r of rows){const v=saved[r.speaker+'|'+r.session+'|'+r.utterance_id];if(v)Object.assign(r,v)}
function count(){document.querySelector('#count').textContent=rows.filter(r=>r.decision!=='unreviewed').length+'/'+rows.length+' decided'}
for(const el of document.querySelectorAll('[data-field]'))el.value=rows[Number(el.dataset.i)][el.dataset.field];
function update(e){const t=e.target;if(!t.dataset.field)return;rows[Number(t.dataset.i)][t.dataset.field]=t.value;const o={};for(const r of rows)o[r.speaker+'|'+r.session+'|'+r.utterance_id]={decision:r.decision,reason:r.reason};try{localStorage.setItem(key,JSON.stringify(o))}catch(e){}count()}
document.addEventListener('input',update);document.addEventListener('change',update);
function cell(v){return '"'+String(v??'').replaceAll('"','""')+'"'}
document.querySelector('#export').addEventListener('click',()=>{const fields=['speaker','session','utterance_id','pos_path','review_flag','steps_over_20','decision','reason'];const out=[fields.map(cell).join(',')];for(const r of rows)out.push(fields.map(f=>cell(r[f])).join(','));const eol=String.fromCharCode(13,10);const blob=new Blob([out.join(eol)+eol],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='torgo_sensor_review_completed.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});count();
</script></body></html>"""
    page = page.replace("CARDS_PLACEHOLDER", "\n".join(cards)).replace("DATA_PLACEHOLDER", script_data)
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(cards)} high-priority traces")


if __name__ == "__main__":
    main()
