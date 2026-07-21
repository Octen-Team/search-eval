"""Human blind-label calibration for the pairwise judge.

Protocol (prompts/pairwise_judge.md §Calibration): sample pairs, have ≥2 humans label them
blind and independently, then compare the judge's verdicts against the human majority —
target ≥85% agreement after excluding pairs where the humans themselves disagree.

export — sample N pairs from a run and emit ONE self-contained HTML page (no server, no
         dependencies; open the file, label, download the JSONL). Blinding: system names are
         never shown, the judge's verdict is not embedded at all, and left/right placement is
         randomized PER ANNOTATOR (seeded by annotator name + qid) so human position bias
         cancels across annotators.
score  — compute agreement from 2+ annotators' downloaded label files.

Usage:
  python -m scripts.calibration export --run results/run_20260708_v4 \\
      --queries data/main_queries.jsonl data/realtime_20260708.jsonl --n 80
  # → results/run_20260708_v4/calibration.html  (send to annotators)
  python -m scripts.calibration score --run results/run_20260708_v4 \\
      --labels labels_alice.jsonl labels_bob.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.common import load_jsonl, require_pairwise_meta
from src.rubric_gen import load_rubrics, render_for_judge


def sample_pairs(pairs: list[dict], n: int, ours: str, seed: int = 42) -> list[dict]:
    """Stratified sample over outcome classes so wins/losses/ties are all represented:
    ties get up to 20% of the budget, the rest splits proportionally between wins and losses."""
    rng = random.Random(seed)
    wins = [p for p in pairs if p["winner"] == ours]
    ties = [p for p in pairs if p["winner"] == "tie"]
    losses = [p for p in pairs if p["winner"] not in (ours, "tie")]
    for bucket in (wins, ties, losses):
        rng.shuffle(bucket)
    n_tie = min(len(ties), max(1, n // 5))
    rest = n - n_tie
    decisive = len(wins) + len(losses)
    n_win = min(len(wins), round(rest * (len(wins) / max(decisive, 1))))
    n_loss = min(len(losses), rest - n_win)
    picked = wins[:n_win] + ties[:n_tie] + losses[:n_loss]
    # top up from whatever has leftovers if a bucket ran short
    if len(picked) < n:
        leftovers = wins[n_win:] + losses[n_loss:] + ties[n_tie:]
        picked += leftovers[: n - len(picked)]
    rng.shuffle(picked)
    return picked[:n]


def human_majority(choices: list[str]) -> str | None:
    """Unanimous-or-nothing for 2 annotators; strict majority for 3+. None = humans disagree."""
    counts = Counter(choices)
    top, top_n = counts.most_common(1)[0]
    if top_n * 2 > len(choices):
        return top
    return None


def score_labels(pairs_by_key: dict, label_files: list[list[dict]]) -> dict:
    """Judge↔human agreement per the calibration protocol."""
    by_key: dict[tuple, dict[str, str]] = defaultdict(dict)
    for labels in label_files:
        for rec in labels:
            by_key[(rec["qid"], rec["system_y"])][rec["annotator"]] = rec["choice"]

    complete = {k: v for k, v in by_key.items()
                if len(v) >= len(label_files) and k in pairs_by_key}
    consensus, humans_disagree = {}, 0
    for k, votes in complete.items():
        maj = human_majority(list(votes.values()))
        if maj is None:
            humans_disagree += 1
        else:
            consensus[k] = maj

    agree = sum(1 for k, maj in consensus.items() if pairs_by_key[k]["winner"] == maj)
    # near-misses worth surfacing: judge tie vs human decisive (softer than a flip)
    judge_tie_human_decisive = sum(
        1 for k, maj in consensus.items()
        if pairs_by_key[k]["winner"] == "tie" and maj != "tie")
    flips = sum(1 for k, maj in consensus.items()
                if pairs_by_key[k]["winner"] not in ("tie", maj) and maj != "tie")
    return {
        "pairs_labeled_by_all": len(complete),
        "human_disagreement": humans_disagree,
        "consensus_pairs": len(consensus),
        "judge_agreement": agree,
        "judge_agreement_rate": agree / max(len(consensus), 1),
        "judge_tie_human_decisive": judge_tie_human_decisive,
        "hard_flips": flips,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Blind pairwise labeling</title>
<style>
 body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;background:#f5f6f8;color:#1a1a1a}
 header{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid #ddd;padding:8px 20px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 header b{font-size:15px} .prog{color:#666;font-size:13px}
 #annotator{padding:6px 8px;border:1px solid #ccc;border-radius:6px}
 #export{background:#059669;color:#fff;border:none;padding:7px 14px;border-radius:8px;cursor:pointer}
 .strip{display:flex;flex-wrap:wrap;gap:3px;padding:6px 20px;background:#fff;border-bottom:1px solid #eee}
 .cell{width:22px;height:20px;font-size:10px;border:1px solid #ccc;border-radius:4px;background:#fff;cursor:pointer;padding:0;color:#888}
 .cell.done{background:#bbf7d0;border-color:#86efac;color:#166534}
 .cell.cur{outline:2px solid #2563eb;color:#1e40af;font-weight:700}
 main{max-width:1280px;margin:12px auto;padding:0 16px 130px}
 .qbox{background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:12px 18px;margin-bottom:10px}
 .qbox h2{margin:0 0 4px;font-size:17px} .labels{color:#777;font-size:12.5px}
 details{margin-top:6px;font-size:13px;color:#444} details pre{white-space:pre-wrap;background:#fafafa;padding:8px;border-radius:6px}
 .cols{display:flex;gap:12px} .col{flex:1;background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:10px 14px;min-width:0}
 .col h3{margin:0 0 6px;font-size:14px;color:#555}
 .res{border-bottom:1px solid #f0f0f0;padding:7px 0;font-size:13.5px}
 .res .t{font-weight:600} .res .t a{color:#1a56db;text-decoration:none} .res .t a:hover{text-decoration:underline}
 .res .u{color:#0a7d33;font-size:11.5px;word-break:break-all}
 .res .s{color:#444;margin-top:2px;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;cursor:pointer}
 .res .s.open{display:block;overflow:visible}
 body.compact .res .s{display:none}
 .shared{background:#fef3c7;color:#92400e;font-size:10.5px;border-radius:4px;padding:1px 5px;margin-left:6px;white-space:nowrap}
 footer{position:fixed;bottom:0;left:0;right:0;z-index:10;background:#fff;border-top:1px solid #ddd;padding:10px 16px;display:flex;gap:10px;align-items:center;justify-content:center;flex-wrap:wrap}
 button.vote{font-size:15px;padding:11px 24px;border-radius:10px;border:1px solid #ccc;background:#fff;cursor:pointer}
 button.vote:hover{background:#eef4ff} button.vote.sel{background:#2563eb;color:#fff;border-color:#2563eb}
 footer .nav{padding:9px 16px;border-radius:8px;border:1px solid #ccc;background:#fff;cursor:pointer}
 #note{width:260px;border:1px solid #ddd;border-radius:6px;padding:7px;font-size:12.5px}
 .kbd{color:#999;font-size:11.5px;width:100%;text-align:center}
</style></head><body>
<header>
 <b>Blind pairwise labeling</b>
 <label>Annotator: <input id="annotator" placeholder="your name" /></label>
 <span class="prog" id="prog"></span>
 <label style="font-size:12.5px;color:#666"><input type="checkbox" id="compact"> hide snippets</label>
 <button id="export">Download labels (.jsonl)</button>
</header>
<div class="strip" id="strip"></div>
<main>
 <div class="qbox">
  <h2 id="query"></h2>
  <div class="labels" id="meta"></div>
  <details><summary>Per-query rubric (same criteria the LLM judge saw)</summary><pre id="rubric"></pre></details>
 </div>
 <div class="cols">
  <div class="col"><h3>System A</h3><div id="listA"></div></div>
  <div class="col"><h3>System B</h3><div id="listB"></div></div>
 </div>
</main>
<footer>
 <button class="nav" id="prev">&larr; Prev</button>
 <button class="vote" id="voteA">A is better <small>(1)</small></button>
 <button class="vote" id="voteT">Tie <small>(0)</small></button>
 <button class="vote" id="voteB">B is better <small>(2)</small></button>
 <button class="nav" id="next">Next &rarr;</button>
 <input id="note" placeholder="optional note (why?)" />
 <div class="kbd">Keys: 1 = A better · 0 = tie · 2 = B better · &larr;/&rarr; navigate. Titles open in a new tab.
 Yellow badge = same URL appears on both sides (skim those; read the unique ones). Labels autosave locally per annotator.</div>
</footer>
<script>
const DATA = __DATA__;
let idx = 0;
const $ = id => document.getElementById(id);
function annName(){ return $("annotator").value.trim(); }
function storeKey(){ return "callab:" + annName(); }
function loadLabels(){ try { return JSON.parse(localStorage.getItem(storeKey()) || "{}"); } catch(e){ return {}; } }
function saveLabels(l){ localStorage.setItem(storeKey(), JSON.stringify(l)); }
// deterministic per-annotator side flip: annotators see independent randomizations
function hash(s){ let h = 0; for (const c of s) { h = ((h<<5)-h + c.charCodeAt(0))|0; } return Math.abs(h); }
function flipped(p){ return hash(annName() + "|" + p.qid + "|" + p.system_y) % 2 === 1; }
function pairKey(p){ return p.qid + "||" + p.system_y; }
function esc(parent, tag, cls, text){ const el = document.createElement(tag); if (cls) el.className = cls; el.textContent = text; parent.appendChild(el); return el; }
function safeHref(u){ return /^https?:\\/\\//i.test(u) ? u : null; }
function renderList(el, results, otherRankByUrl, otherLabel){
  el.innerHTML = "";
  for (const r of results){
    const d = document.createElement("div"); d.className = "res";
    const t = esc(d, "div", "t", "[" + r.rank + "] ");
    const href = safeHref(r.url);
    if (href){
      const a = document.createElement("a");
      a.href = href; a.target = "_blank"; a.rel = "noopener noreferrer";
      a.textContent = r.title || r.url;
      t.appendChild(a);
    } else {
      t.appendChild(document.createTextNode(r.title || r.url));
    }
    const other = otherRankByUrl.get(r.url);
    if (other !== undefined) esc(t, "span", "shared", "also " + otherLabel + " #" + other);
    esc(d, "div", "u", r.url + (r.published_date ? "  ·  " + r.published_date : ""));
    const s = esc(d, "div", "s", r.snippet);
    s.title = "click to expand/collapse";
    s.onclick = () => s.classList.toggle("open");
    el.appendChild(d);
  }
  if (!results.length) esc(el, "div", "s", "(no results)");
}
function labeledCount(labels){ return Object.keys(labels).filter(k => DATA.some(q => pairKey(q) === k)).length; }
function renderStrip(labels){
  const st = $("strip"); st.innerHTML = "";
  DATA.forEach((p, i) => {
    const b = document.createElement("button");
    b.className = "cell" + (labels[pairKey(p)] ? " done" : "") + (i === idx ? " cur" : "");
    b.textContent = i + 1;
    b.onclick = () => { idx = i; render(); };
    st.appendChild(b);
  });
}
function rankMap(results){ const m = new Map(); for (const r of results) if (!m.has(r.url)) m.set(r.url, r.rank); return m; }
function render(){
  const p = DATA[idx], f = flipped(p);
  $("query").textContent = (idx+1) + ". " + p.query;
  $("meta").textContent = "vertical: " + p.vertical + " · intent: " + p.intent + " · freshness: " + p.freshness
    + " — judge on: relevance / authority / freshness / diversity / snippet quality";
  $("rubric").textContent = p.rubric || "(generic rubric)";
  const left = f ? p.results_y : p.results_x, right = f ? p.results_x : p.results_y;
  renderList($("listA"), left, rankMap(right), "B");
  renderList($("listB"), right, rankMap(left), "A");
  const labels = loadLabels(), cur = labels[pairKey(p)];
  for (const b of ["voteA","voteT","voteB"]) $(b).classList.remove("sel");
  $("note").value = cur && cur.note || "";
  if (cur){
    const disp = cur.choice === "tie" ? "T" : ((cur.choice === p.system_x) !== f ? "A" : "B");
    $("vote" + disp).classList.add("sel");
  }
  $("prog").textContent = "pair " + (idx+1) + "/" + DATA.length + " · labeled " + labeledCount(labels) + "/" + DATA.length;
  renderStrip(labels);
  window.scrollTo(0, 0);
}
function vote(disp){
  if (!annName()){ alert("Enter your annotator name first."); return; }
  const p = DATA[idx], f = flipped(p);
  let choice;
  if (disp === "T") choice = "tie";
  else if (disp === "A") choice = f ? p.system_y : p.system_x;
  else choice = f ? p.system_x : p.system_y;
  const labels = loadLabels();
  labels[pairKey(p)] = { qid: p.qid, system_y: p.system_y, choice: choice,
                         note: $("note").value, ts: new Date().toISOString() };
  saveLabels(labels);
  if (idx < DATA.length - 1) idx++;
  render();
}
$("voteA").onclick = () => vote("A");
$("voteT").onclick = () => vote("T");
$("voteB").onclick = () => vote("B");
$("prev").onclick = () => { if (idx > 0) idx--; render(); };
$("next").onclick = () => { if (idx < DATA.length - 1) idx++; render(); };
$("annotator").onchange = render;
$("compact").onchange = () => document.body.classList.toggle("compact", $("compact").checked);
$("note").onchange = () => { const p = DATA[idx], labels = loadLabels(), k = pairKey(p);
  if (labels[k]){ labels[k].note = $("note").value; saveLabels(labels); } };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === "1") vote("A"); else if (e.key === "2") vote("B"); else if (e.key === "0") vote("T");
  else if (e.key === "ArrowLeft") $("prev").onclick(); else if (e.key === "ArrowRight") $("next").onclick();
});
$("export").onclick = () => {
  if (!annName()){ alert("Enter your annotator name first."); return; }
  const labels = loadLabels();
  const rows = DATA.map(p => labels[pairKey(p)]).filter(Boolean)
    .map(l => JSON.stringify({ ...l, annotator: annName() }));
  if (!rows.length){ alert("No labels yet."); return; }
  const blob = new Blob([rows.join("\\n") + "\\n"], { type: "application/jsonl" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels_" + annName().replace(/\\W+/g, "_") + ".jsonl";
  a.click();
};
render();
</script></body></html>
"""


def cmd_export(args) -> None:
    run = Path(args.run)
    meta = require_pairwise_meta(run)
    ours = meta["ours"]
    pairs = [p for p in load_jsonl(run / "pairwise.jsonl") if "winner" in p]
    responses = {(r["qid"], r["backend"]): r for r in load_jsonl(run / "responses.jsonl")}
    qmeta = {q["qid"]: q for f in args.queries for q in load_jsonl(f)}
    rubrics = load_rubrics(args.rubrics)

    usable = [p for p in pairs
              if (p["qid"], p["system_x"]) in responses and (p["qid"], p["system_y"]) in responses
              and p["qid"] in qmeta]
    picked = sample_pairs(usable, args.n, ours, seed=args.seed)

    def results_of(qid: str, backend: str) -> list[dict]:
        return [{"rank": r["rank"], "title": r["title"], "url": r["url"],
                 "snippet": r["snippet"], "published_date": r.get("published_date")}
                for r in responses[(qid, backend)].get("results", [])]

    data = []
    for p in picked:
        q = qmeta[p["qid"]]
        rub = rubrics.get(p["qid"])
        data.append({
            "qid": p["qid"], "query": q["query"],
            "vertical": q.get("vertical"), "intent": q.get("intent"), "freshness": q.get("freshness"),
            "rubric": render_for_judge(rub) if rub else None,
            # real system identities ride along ONLY for label mapping; never displayed
            "system_x": p["system_x"], "system_y": p["system_y"],
            "results_x": results_of(p["qid"], p["system_x"]),
            "results_y": results_of(p["qid"], p["system_y"]),
        })

    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = run / "calibration.html"
    out.write_text(HTML_TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    dist = Counter("tie" if p["winner"] == "tie" else ("win" if p["winner"] == ours else "loss")
                   for p in picked)
    print(f"→ {out}  ({len(data)} pairs; outcome mix {dict(dist)})")
    print("Send the file to ≥2 annotators. Each opens it locally, enters their name, labels, and "
          "downloads labels_<name>.jsonl. The judge's verdicts are NOT embedded; sides are "
          "randomized per annotator.")


def cmd_score(args) -> None:
    run = Path(args.run)
    pairs = [p for p in load_jsonl(run / "pairwise.jsonl") if "winner" in p]
    pairs_by_key = {(p["qid"], p["system_y"]): p for p in pairs}
    label_files = [load_jsonl(f) for f in args.labels]
    for f, labels in zip(args.labels, label_files):
        anns = {l["annotator"] for l in labels}
        print(f"  {f}: {len(labels)} labels by {sorted(anns)}")
    s = score_labels(pairs_by_key, label_files)
    print(f"\npairs labeled by all annotators : {s['pairs_labeled_by_all']}")
    print(f"human↔human disagreement        : {s['human_disagreement']} (excluded per protocol)")
    print(f"consensus pairs                 : {s['consensus_pairs']}")
    print(f"judge agreement on consensus    : {s['judge_agreement']}/{s['consensus_pairs']} "
          f"= {s['judge_agreement_rate']:.0%}   [target ≥85%]")
    print(f"  soft misses (judge tie, humans decisive): {s['judge_tie_human_decisive']}")
    print(f"  hard flips  (opposite winners)          : {s['hard_flips']}")
    if s["judge_agreement_rate"] < 0.85 and s["consensus_pairs"] >= 20:
        print("BELOW TARGET — analyze the disagreement cases by dimension and revise the rubric "
              "wording (prompts/pairwise_judge.md §Calibration step 3), then re-run.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="sample pairs and emit the blind-labeling HTML page")
    ex.add_argument("--run", required=True)
    ex.add_argument("--queries", required=True, nargs="+")
    ex.add_argument("--rubrics", default="data/rubrics.jsonl")
    ex.add_argument("--n", type=int, default=80)
    ex.add_argument("--seed", type=int, default=42)
    sc = sub.add_parser("score", help="compute judge↔human agreement from label files")
    sc.add_argument("--run", required=True)
    sc.add_argument("--labels", required=True, nargs="+")
    args = ap.parse_args()
    if args.cmd == "export":
        cmd_export(args)
    else:
        cmd_score(args)


if __name__ == "__main__":
    main()
