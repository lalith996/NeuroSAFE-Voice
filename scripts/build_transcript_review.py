"""Create a balanced, listen-and-correct TORGO transcript review queue.

Whisper hypotheses and TORGO prompts are aids for review, not accepted labels.
The HTML page plays project-local audio and exports reviewer edits to CSV.
"""

from __future__ import annotations

import csv
import html
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "data/results/whisper_small_en_zero_shot/utterances.csv"
MANIFEST = ROOT / "data/processed/utterance_manifest.csv"
SENSOR = ROOT / "data/processed/torgo_sensor_audit.csv"
AUDIO = ROOT / "data/processed/torgo_audio_audit.csv"
OUTPUT_CSV = ROOT / "reports/transcript_review_queue.csv"
OUTPUT_HTML = ROOT / "reports/transcript_review.html"
FIELDS = [
    "speaker", "condition", "session", "utterance_id", "audio_path",
    "microphone", "duration_seconds", "reference_prompt", "whisper_hypothesis",
    "prompt_reference_wer", "selection_reason", "audio_rms_dbfs", "audio_clipped_sample_fraction",
    "audio_review_flag", "sensor_review_flag", "review_status",
    "verified_transcript", "review_notes", "reviewer",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select(rows: list[dict], seed: int) -> list[tuple[dict, str]]:
    """Take high-disagreement and random cases in each prompt-length stratum."""
    rng = random.Random(seed)
    chosen = []
    for is_short in (True, False):
        pool = [r for r in rows if (int(r["reference_words"]) <= 3) == is_short]
        if not pool:
            continue
        ranked = sorted(pool, key=lambda r: (-float(r["wer"]), r["session"], r["utterance_id"]))
        high = ranked[: min(3, len(ranked))]
        chosen.extend((r, "high_disagreement_short" if is_short else "high_disagreement_long") for r in high)
        rest = [r for r in pool if r not in high]
        chosen.extend((r, "random_short" if is_short else "random_long") for r in rng.sample(rest, min(3, len(rest))))
    if len(chosen) < 12:
        rest = [r for r in rows if r not in [item[0] for item in chosen]]
        chosen.extend((r, "random_fill") for r in rng.sample(rest, min(12 - len(chosen), len(rest))))
    return chosen[:12]


def html_page(rows: list[dict]) -> str:
    embedded = json.dumps(rows, ensure_ascii=True).replace("</", "<\\/")
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TORGO transcript review</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 18px;color:#17202a;background:#f4f7f9}
h1{margin-bottom:6px}p{line-height:1.45}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:18px 0}
select,input,textarea,button{font:inherit;padding:8px;border:1px solid #bdc8d2;border-radius:6px;background:white}
button{background:#145b78;color:white;border:0;cursor:pointer}.card{background:white;border:1px solid #d6e0e6;border-radius:10px;padding:18px;margin:16px 0}
.meta{color:#425466;font-size:14px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pair div{background:#f6f8fa;padding:10px;border-radius:6px}
label{display:block;font-weight:600;margin:10px 0 4px}textarea{width:100%;box-sizing:border-box;min-height:58px}audio{width:100%;margin-top:12px}
@media(max-width:650px){.pair{grid-template-columns:1fr}}
</style></head><body>
<h1>TORGO transcript review</h1>
<p>Listen before marking a transcript. The written prompt and Whisper output are clues, not verified speech. This queue mixes randomly chosen and high-disagreement clips; the CSV records which is which. Use <strong>Export review CSV</strong> after edits; this page also saves drafts in this browser when storage is available. Review flags on sensor motion are for inspection only.</p>
<div class="toolbar"><label for="speaker">Speaker</label><select id="speaker"><option value="">All speakers</option></select>
<label for="statusfilter">Status</label><select id="statusfilter"><option value="">All</option><option value="unreviewed">Unreviewed</option><option value="prompt_matches">Prompt matches</option><option value="corrected">Corrected</option><option value="unintelligible">Unintelligible</option><option value="audio_issue">Audio issue</option></select>
<button id="export">Export review CSV</button><span id="count"></span></div>
<div id="cards"></div>
<script>
const rows=DATA_PLACEHOLDER;
const key='torgo-review-v1';
let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
for(const row of rows){const v=saved[row.speaker+'|'+row.session+'|'+row.utterance_id];if(v)Object.assign(row,v)}
const speakers=[...new Set(rows.map(r=>r.speaker))].sort();
for(const s of speakers){const o=document.createElement('option');o.value=s;o.textContent=s;document.querySelector('#speaker').append(o)}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function save(){const data={};for(const r of rows)data[r.speaker+'|'+r.session+'|'+r.utterance_id]={review_status:r.review_status,verified_transcript:r.verified_transcript,review_notes:r.review_notes,reviewer:r.reviewer};try{localStorage.setItem(key,JSON.stringify(data))}catch(e){}updateCount()}
function updateCount(){const n=rows.filter(r=>r.review_status&&r.review_status!=='unreviewed').length;document.querySelector('#count').textContent=n+'/'+rows.length+' reviewed'}
function render(){const sp=document.querySelector('#speaker').value,st=document.querySelector('#statusfilter').value;
const selected=rows.filter(r=>(!sp||r.speaker===sp)&&(!st||(r.review_status||'unreviewed')===st));
document.querySelector('#cards').innerHTML=selected.map(r=>{const i=rows.indexOf(r);return `<article class="card"><strong>${esc(r.speaker)} / ${esc(r.session)} / ${esc(r.utterance_id)}</strong><div class="meta">${esc(r.condition)} · ${esc(r.microphone)} mic · ${esc(r.duration_seconds)} s · prompt-reference WER ${Math.round(100*Number(r.prompt_reference_wer))}% · RMS ${esc(r.audio_rms_dbfs)} dBFS ${r.audio_review_flag?'· audio review: '+esc(r.audio_review_flag):''} ${r.sensor_review_flag?'· sensor review: '+esc(r.sensor_review_flag):''}</div><audio controls preload="none" src="../${esc(r.audio_path)}"></audio><div class="pair"><div><b>Written prompt</b><br>${esc(r.reference_prompt)}</div><div><b>Whisper output</b><br>${esc(r.whisper_hypothesis)}</div></div><label for="status${i}">Review status</label><select id="status${i}" data-i="${i}" data-field="review_status"><option value="unreviewed">Unreviewed</option><option value="prompt_matches">Prompt matches exactly</option><option value="corrected">Corrected transcript</option><option value="unintelligible">Unintelligible</option><option value="audio_issue">Audio issue</option></select><label for="transcript${i}">Verified words actually spoken</label><textarea id="transcript${i}" data-i="${i}" data-field="verified_transcript" placeholder="Enter what you hear; leave blank until reviewed">${esc(r.verified_transcript)}</textarea><label for="note${i}">Notes</label><textarea id="note${i}" data-i="${i}" data-field="review_notes">${esc(r.review_notes)}</textarea><label for="reviewer${i}">Reviewer</label><input id="reviewer${i}" data-i="${i}" data-field="reviewer" value="${esc(r.reviewer)}"></article>`}).join('');
for(const el of document.querySelectorAll('[data-field="review_status"]'))el.value=rows[Number(el.dataset.i)].review_status||'unreviewed';
updateCount()}
document.querySelector('#cards').addEventListener('input',e=>{const t=e.target;if(!t.dataset.field)return;rows[Number(t.dataset.i)][t.dataset.field]=t.value;save()});
document.querySelector('#cards').addEventListener('change',e=>{const t=e.target;if(!t.dataset.field)return;rows[Number(t.dataset.i)][t.dataset.field]=t.value;save()});
document.querySelector('#speaker').addEventListener('change',render);document.querySelector('#statusfilter').addEventListener('change',render);
function csvCell(v){return '"'+String(v??'').replaceAll('"','""')+'"'}
document.querySelector('#export').addEventListener('click',()=>{const fields=FIELDS_PLACEHOLDER;const out=[fields.map(csvCell).join(',')];for(const r of rows)out.push(fields.map(f=>csvCell(r[f])).join(','));const lineEnd=String.fromCharCode(13,10);const blob=new Blob([out.join(lineEnd)+lineEnd],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='torgo_transcript_review_completed.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});
render();
</script></body></html>""".replace("DATA_PLACEHOLDER", embedded).replace("FIELDS_PLACEHOLDER", json.dumps(FIELDS))


def main() -> None:
    baseline = read_csv(BASELINE)
    manifest = {(r["speaker"], r["session"], r["utterance_id"]): r for r in read_csv(MANIFEST) if r["dataset"] == "TORGO"}
    sensor = {(r["speaker"], r["session"], r["utterance_id"]): r for r in read_csv(SENSOR)}
    audio = {(r["speaker"], r["session"], r["utterance_id"]): r for r in read_csv(AUDIO)}
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in baseline:
        by_speaker[row["speaker"]].append(row)
    selected = []
    for speaker, rows in sorted(by_speaker.items()):
        selected.extend(select(rows, 2026 + sum(map(ord, speaker))))
    output = []
    for row, selection_reason in selected:
        key = (row["speaker"], row["session"], row["utterance_id"])
        source = manifest[key]
        output.append({
            "speaker": row["speaker"], "condition": source["condition"],
            "session": row["session"], "utterance_id": row["utterance_id"],
            "audio_path": row["audio_path"], "microphone": row["microphone"],
            "duration_seconds": row["duration_seconds"],
            "reference_prompt": row["reference"],
            "whisper_hypothesis": row["hypothesis"],
            "prompt_reference_wer": row["wer"],
            "selection_reason": selection_reason,
            "audio_rms_dbfs": audio[key]["rms_dbfs"],
            "audio_clipped_sample_fraction": audio[key]["clipped_sample_fraction"],
            "audio_review_flag": audio[key]["review_flag"],
            "sensor_review_flag": sensor.get(key, {}).get("review_flag", ""),
            "review_status": "unreviewed", "verified_transcript": "",
            "review_notes": "", "reviewer": "",
        })
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    OUTPUT_HTML.write_text(html_page(output), encoding="utf-8")
    print(f"Prepared {len(output)} recordings from {len(by_speaker)} speakers")
    print(OUTPUT_CSV)
    print(OUTPUT_HTML)


if __name__ == "__main__":
    main()
