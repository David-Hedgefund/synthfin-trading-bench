#!/usr/bin/env python3
"""Generate the static benchmark website (site/index.html) from a results directory.

The leaderboard is injected from results/<run>/leaderboard.json + run_meta.json, so the site
is a build artifact of the benchmark, not hand-maintained. Deploy site/ via GitHub Pages.

    python scripts/build_site.py --run validation
    python scripts/build_site.py --results results/v1 --out site
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GITHUB_URL = "https://github.com/David-Hedgefund/synthfin-trading-bench"


def _fmt_pct(x, digits=1):
    return f"{x*100:+.{digits}f}%" if isinstance(x, (int, float)) else "-"


def _fmt_num(x, digits=2):
    return f"{x:+.{digits}f}" if isinstance(x, (int, float)) else "-"


def _rows(leaderboard):
    rows = []
    for i, r in enumerate(leaderboard, 1):
        name = r.get("agent_name", "?")
        is_baseline = name.startswith("baseline") or name.startswith("mock")
        rows.append(
            {
                "rank": i,
                "name": name,
                "kind": "baseline" if is_baseline else "model",
                "ret": r.get("mean_total_return"),
                "sharpe": r.get("mean_sharpe"),
                "alpha": r.get("mean_alpha_ann"),
                "appraisal": r.get("mean_appraisal_ratio"),
                "appraisal_std": r.get("std_appraisal_ratio"),
                "beta": r.get("mean_beta_market"),
                "mdd": r.get("mean_max_drawdown"),
                "win": r.get("benchmark_win_rate"),
                "n": r.get("n_scenarios"),
            }
        )
    return rows


SEED_NAMES = {
    80000: ("Soft Landing", "calm"),
    80011: ("AI Capex Pays Off", "tech bull"),
    80013: ("Energy Abundance", "energy bull"),
    80014: ("Manufacturing Reshoring", "industrials bull"),
    80017: ("AI Bubble Deflation", "tech bear"),
    80018: ("Inflation Re-Acceleration", "macro shock"),
    80019: ("Credit Intermediation Stress", "financial crisis"),
    80021: ("Taiwan Strait Escalation", "geopolitical"),
    80022: ("Middle East Energy Disruption", "energy shock"),
    80023: ("2013 China Hard Landing", "global bear"),
}


def _is_model(name: str) -> bool:
    return bool(name) and not name.startswith("baseline") and not name.startswith("mock")


def _model_scenarios(results_dir: Path, leaderboard):
    """Per-scenario rows for the spotlight model: the top-ranked (best-Sharpe) LLM on the
    board. Featuring the strongest model - not an alphabetical accident - makes the point
    that even the best agent trails a passive basket."""
    import re

    scores_dir = results_dir / "scores"
    if not scores_dir.exists():
        return None, []
    by_agent: dict = {}
    for f in sorted(scores_dir.glob("*.json")):
        s = json.loads(f.read_text())
        by_agent.setdefault(s.get("agent_name", ""), []).append(s)
    # leaderboard is Sharpe-sorted; pick the first model in that order.
    model = next(
        (r.get("agent_name", "") for r in leaderboard
         if _is_model(r.get("agent_name", "")) and r.get("agent_name", "") in by_agent),
        None,
    )
    if not model:  # fall back to any model present
        model = next((a for a in by_agent if _is_model(a)), None)
    if not model:
        return None, []
    rows = []
    for s in by_agent[model]:
        m = re.search(r"seed_(\d+)", s.get("scenario_slug", ""))
        seed = int(m.group(1)) if m else 0
        name, bucket = SEED_NAMES.get(seed, (s.get("scenario_slug", "?"), ""))
        rows.append(
            {
                "seed": seed, "name": name, "bucket": bucket,
                "ret": s.get("total_return"), "alpha": s.get("alpha_ann"),
                "appraisal": s.get("appraisal_ratio"), "excess": s.get("excess_return"),
            }
        )
    rows.sort(key=lambda r: r["seed"])
    return model, rows


def build(results_dir: Path, out_dir: Path, models_only: bool = False) -> Path:
    lb = json.loads((results_dir / "leaderboard.json").read_text())
    meta = json.loads((results_dir / "run_meta.json").read_text())
    rows = _rows(lb)
    if models_only:
        rows = [r for r in rows if r["kind"] == "model"]
        for i, r in enumerate(rows, 1):  # re-rank the models-only view
            r["rank"] = i
    has_models = any(r["kind"] == "model" for r in rows)

    # scale appraisal bars across the board
    apps = [abs(r["appraisal"]) for r in rows if isinstance(r["appraisal"], (int, float))]
    app_max = max(apps) if apps else 1.0

    body_rows = []
    for r in rows:
        app = r["appraisal"]
        barw = (abs(app) / app_max * 100) if isinstance(app, (int, float)) and app_max else 0
        sign = "up" if isinstance(app, (int, float)) and app >= 0 else "down"
        ret_cls = "up" if isinstance(r["ret"], (int, float)) and r["ret"] >= 0 else "down"
        tag = (
            '<span class="tag tag-base">baseline</span>'
            if r["kind"] == "baseline"
            else '<span class="tag tag-model">model</span>'
        )
        body_rows.append(f"""
        <tr class="row-{r['kind']}">
          <td class="rank">{r['rank']}</td>
          <td class="agent"><span class="agent-name">{html.escape(r['name'])}</span>{tag}</td>
          <td class="num {ret_cls}">{_fmt_pct(r['ret'])}</td>
          <td class="num">{_fmt_num(r['sharpe'])}</td>
          <td class="num">{_fmt_pct(r['alpha'])}</td>
          <td class="num appraisal">
            <span class="appraisal-val">{_fmt_num(r['appraisal'])}</span>
            <span class="bar"><span class="bar-fill {sign}" style="width:{barw:.0f}%"></span></span>
          </td>
          <td class="num dim">{_fmt_num(r['beta'])}</td>
          <td class="num down">{_fmt_pct(r['mdd'])}</td>
          <td class="num dim">{_fmt_pct(r['win'],0)}</td>
        </tr>""")

    n_tickers = meta.get("n_tickers", "-")
    n_scen = len(meta.get("scenarios", []))
    chash = (meta.get("corpus_hash") or "")[:12]
    profile = meta.get("corpus_profile", "-")
    reb = meta.get("run_config", {}).get("rebalance_every_days", "-")

    status_banner = (
        ""
        if has_models
        else '<div class="banner">Preview leaderboard - baseline strategies only. '
        "Model submissions are open; see <a href=\"#submit\">Submit a model</a>.</div>"
    )

    # ---- per-scenario spotlight for the leading model (drives the "beat it" hook) ----
    model_name, mscn = _model_scenarios(results_dir, lb)
    n_models = sum(1 for r in rows if r["kind"] == "model")
    spotlight = ""
    if mscn:
        n = len(mscn)
        lost = sum(1 for r in mscn if isinstance(r["excess"], (int, float)) and r["excess"] < 0)
        neg_ret = sum(1 for r in mscn if isinstance(r["ret"], (int, float)) and r["ret"] < 0)
        best = n_models > 1  # is the featured model the best of several?
        if lost == n:
            head = f"{html.escape(model_name)}: beaten by the market in all {n} regimes"
            note = (
                (f"The strongest LLM on the board " if best else "")
                + f"lost to a plain hold of the universe in <b>every one of {n}</b> scenarios "
                "- calm, bull, bear and crisis alike. The bar to clear is low."
            )
        else:
            head = (
                f"{html.escape(model_name)}: "
                + ("the top LLM here, still underwater" if best else "still underwater")
            )
            note = (
                (f"Best model on the board, yet it " if best else "It ")
                + f"lost money in <b>{neg_ret} of {n}</b> regimes and trailed a passive basket in "
                f"<b>{lost}</b>. Negative alpha on average - in calm, bull, bear and crisis alike."
            )
        amax = max((abs(r["alpha"]) for r in mscn if isinstance(r["alpha"], (int, float))), default=1)
        def _sc(x):  # sign class - green for the rare regime an LLM actually wins
            return "up" if isinstance(x, (int, float)) and x >= 0 else "down"

        srows = []
        for r in mscn:
            aw = (abs(r["alpha"]) / amax * 100) if isinstance(r["alpha"], (int, float)) and amax else 0
            srows.append(
                f'<tr><td class="agent"><span class="agent-name">{html.escape(r["name"])}</span>'
                f'<span class="tag tag-base">{html.escape(r["bucket"])}</span></td>'
                f'<td class="num {_sc(r["ret"])}">{_fmt_pct(r["ret"])}</td>'
                f'<td class="num appraisal"><span class="{_sc(r["alpha"])}">{_fmt_pct(r["alpha"])}</span>'
                f'<span class="bar"><span class="bar-fill {_sc(r["alpha"])}" style="width:{aw:.0f}%"></span></span></td>'
                f'<td class="num {_sc(r["appraisal"])}">{_fmt_num(r["appraisal"])}</td>'
                f'<td class="num {_sc(r["excess"])}">{_fmt_pct(r["excess"])}</td></tr>'
            )
        spotlight = (
            '<section id="spotlight"><div class="wrap">'
            f'<div class="sec-head"><h2>{head}</h2>'
            f'<p class="sec-note">{note}</p></div>'
            '<div class="board-wrap"><div class="board-scroll"><table>'
            '<thead><tr><th>Scenario</th><th>Return</th><th>Alpha (ann)</th><th>Appraisal</th>'
            '<th>vs market</th></tr></thead><tbody>' + "".join(srows) + '</tbody></table></div>'
            f'<div class="board-foot"><span>{lost}/{len(mscn)} regimes lost to the market</span>'
            '<span>proxy: equal-weight universe</span></div></div>'
            '<p class="spotlight-cta">Think your model actually beats this? '
            '<a href="#submit">Submit it &rarr;</a></p></div></section>'
        )

    tmpl = _TEMPLATE
    repl = {
        "@@ROWS@@": "".join(body_rows),
        "@@SPOTLIGHT@@": spotlight,
        "@@STATUS@@": status_banner,
        "@@NTICKERS@@": f"{n_tickers:,}" if isinstance(n_tickers, int) else str(n_tickers),
        "@@NSCEN@@": str(n_scen),
        "@@PROFILE@@": html.escape(str(profile)),
        "@@HASH@@": html.escape(chash),
        "@@REBAL@@": str(reb),
        "@@GITHUB@@": GITHUB_URL,
    }
    for k, v in repl.items():
        tmpl = tmpl.replace(k, v)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "index.html"
    out.write_text(tmpl)
    (out_dir / ".nojekyll").write_text("")  # serve _underscore paths, skip Jekyll
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="validation", help="results/<run> to render")
    ap.add_argument("--results", default="", help="explicit results dir (overrides --run)")
    ap.add_argument("--out", default="site", help="output site directory")
    ap.add_argument("--models-only", action="store_true", help="hide baseline rows from the table")
    args = ap.parse_args()
    results_dir = Path(args.results) if args.results else ROOT / "results" / args.run
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out = build(results_dir, out_dir, models_only=args.models_only)
    print(f"wrote {out}")


# --------------------------------------------------------------------------------------
# The page. Self-contained (no external assets) so it works identically on GitHub Pages
# and in a sandboxed preview. Data numbers use ui-monospace tabular figures - the site's
# core identity: synthetic-market data rendered like a trading terminal.
# --------------------------------------------------------------------------------------
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SynthFin Trading Bench - a contamination-free LLM trading benchmark</title>
<meta name="description" content="A contamination-free sequential trading benchmark for LLM agents, run on fully synthetic markets. Models manage a portfolio through curated macro regimes and are scored on risk-adjusted return and stock-selection skill." />
<style>
  :root{
    --bg:#f4f7f9; --surface:#ffffff; --surface-2:#eef3f7; --border:#d8e1e9;
    --text:#0d1826; --muted:#5a6b7c; --faint:#8a99a8;
    --accent:#0e9c8c; --accent-weak:rgba(14,156,140,.12);
    --up:#0e8a4f; --down:#cf3b3b;
    --shadow:0 1px 2px rgba(13,24,38,.05),0 8px 30px rgba(13,24,38,.06);
    --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  @media (prefers-color-scheme:dark){
    :root{
      --bg:#080d17; --surface:#0e1523; --surface-2:#131d2e; --border:#1f2c40;
      --text:#e7eef7; --muted:#8296ad; --faint:#5d7089;
      --accent:#2ad3bf; --accent-weak:rgba(42,211,191,.14);
      --up:#3ece84; --down:#ff6a6a;
      --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 40px rgba(0,0,0,.35);
    }
  }
  :root[data-theme="light"]{
    --bg:#f4f7f9; --surface:#ffffff; --surface-2:#eef3f7; --border:#d8e1e9;
    --text:#0d1826; --muted:#5a6b7c; --faint:#8a99a8;
    --accent:#0e9c8c; --accent-weak:rgba(14,156,140,.12); --up:#0e8a4f; --down:#cf3b3b;
    --shadow:0 1px 2px rgba(13,24,38,.05),0 8px 30px rgba(13,24,38,.06);
  }
  :root[data-theme="dark"]{
    --bg:#080d17; --surface:#0e1523; --surface-2:#131d2e; --border:#1f2c40;
    --text:#e7eef7; --muted:#8296ad; --faint:#5d7089;
    --accent:#2ad3bf; --accent-weak:rgba(42,211,191,.14); --up:#3ece84; --down:#ff6a6a;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 40px rgba(0,0,0,.35);
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  @media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
  body{
    margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);
    line-height:1.55;-webkit-font-smoothing:antialiased;
  }
  a{color:inherit}
  .wrap{max-width:1080px;margin:0 auto;padding:0 24px}
  .mono{font-family:var(--mono);font-variant-numeric:tabular-nums}

  /* nav */
  nav{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--bg) 88%,transparent);
    backdrop-filter:blur(8px);border-bottom:1px solid var(--border)}
  .nav-in{display:flex;align-items:center;gap:22px;height:58px}
  .brand{display:flex;align-items:center;gap:9px;font-weight:700;letter-spacing:-.02em}
  .brand .mk{width:22px;height:22px;border-radius:5px;background:var(--accent);
    display:inline-grid;place-items:center;color:#fff;font-family:var(--mono);font-size:13px;font-weight:700}
  :root[data-theme="dark"] .brand .mk,@media (prefers-color-scheme:dark){}
  .nav-links{margin-left:auto;display:flex;gap:20px;align-items:center;font-size:14px}
  .nav-links a{color:var(--muted);text-decoration:none}
  .nav-links a:hover{color:var(--text)}
  .nav-cta{border:1px solid var(--border);padding:7px 13px;border-radius:8px;background:var(--surface)}
  .theme-btn{background:none;border:1px solid var(--border);color:var(--muted);width:34px;height:34px;
    border-radius:8px;cursor:pointer;font-size:15px}
  @media(max-width:720px){.nav-hide{display:none}}

  /* hero */
  header.hero{position:relative;overflow:hidden;border-bottom:1px solid var(--border)}
  .tape{position:absolute;inset:0;width:100%;height:100%;opacity:.5;pointer-events:none}
  .hero-in{position:relative;padding:76px 0 56px}
  .eyebrow{font-family:var(--mono);font-size:12.5px;letter-spacing:.16em;text-transform:uppercase;
    color:var(--accent);margin:0 0 18px}
  h1{font-size:clamp(2.3rem,5.4vw,3.9rem);line-height:1.02;letter-spacing:-.035em;margin:0;
    font-weight:800;text-wrap:balance;max-width:16ch}
  .lede{margin:20px 0 0;font-size:clamp(1.02rem,2.2vw,1.22rem);color:var(--muted);max-width:60ch}
  .lede b{color:var(--text);font-weight:600}
  .cta-row{display:flex;flex-wrap:wrap;gap:12px;margin-top:30px}
  .btn{display:inline-flex;align-items:center;gap:8px;padding:11px 18px;border-radius:10px;
    font-weight:600;font-size:14.5px;text-decoration:none;border:1px solid transparent;cursor:pointer}
  .btn-primary{background:var(--accent);color:#04201d}
  :root[data-theme="dark"] .btn-primary{color:#04201d}
  .btn-ghost{background:var(--surface);border-color:var(--border);color:var(--text)}
  .btn:hover{transform:translateY(-1px)}
  @media (prefers-reduced-motion:reduce){.btn:hover{transform:none}}

  /* stat strip */
  .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;margin-top:46px;
    background:var(--border);border:1px solid var(--border);border-radius:12px;overflow:hidden}
  .stat{background:var(--surface);padding:18px 22px;display:flex;align-items:center;
    justify-content:space-between;gap:16px}
  .stat .v{font-family:var(--mono);font-size:2.1rem;font-weight:700;letter-spacing:-.02em;
    white-space:nowrap;text-align:left}
  .stat .k{font-size:12.5px;color:var(--muted);text-align:right;line-height:1.3}
  .stat .v.accent{color:var(--accent)}
  @media(max-width:720px){.stats{grid-template-columns:repeat(2,1fr)}}

  section{padding:64px 0}
  .sec-head{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin-bottom:22px}
  h2{font-size:clamp(1.4rem,3vw,1.9rem);letter-spacing:-.03em;margin:0;font-weight:750;text-wrap:balance}
  .sec-note{color:var(--muted);font-size:14px;max-width:46ch}

  .banner{background:var(--accent-weak);border:1px solid color-mix(in srgb,var(--accent) 35%,transparent);
    color:var(--text);padding:11px 16px;border-radius:10px;font-size:14px;margin-bottom:20px}
  .banner a{color:var(--accent);font-weight:600;text-decoration:none}

  /* leaderboard */
  .board-wrap{border:1px solid var(--border);border-radius:14px;overflow:hidden;box-shadow:var(--shadow);
    background:var(--surface)}
  .board-scroll{overflow-x:auto}
  table{width:100%;border-collapse:collapse;font-size:14px;min-width:720px}
  thead th{position:relative;text-align:right;padding:13px 14px;font-size:11.5px;letter-spacing:.06em;
    text-transform:uppercase;color:var(--muted);font-weight:600;border-bottom:1px solid var(--border);
    background:var(--surface-2);white-space:nowrap;cursor:pointer;user-select:none}
  thead th:first-child,thead th:nth-child(2){text-align:left}
  thead th[data-active]::after{content:"";position:absolute}
  thead th .caret{color:var(--accent);font-size:9px}
  tbody td{padding:13px 14px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap}
  tbody tr:last-child td{border-bottom:none}
  tbody tr:hover td{background:color-mix(in srgb,var(--accent) 5%,transparent)}
  td.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
  td.rank{color:var(--faint);font-family:var(--mono);width:44px;text-align:left}
  td.agent{text-align:left}
  .agent-name{font-weight:600;font-family:var(--mono);letter-spacing:-.01em}
  .tag{margin-left:9px;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;padding:2px 7px;
    border-radius:20px;vertical-align:middle;font-weight:600}
  .tag-base{background:var(--surface-2);color:var(--faint);border:1px solid var(--border)}
  .tag-model{background:var(--accent-weak);color:var(--accent)}
  .up{color:var(--up)} .down{color:var(--down)} .dim{color:var(--muted)}
  td.appraisal{min-width:130px}
  .appraisal-val{font-weight:700}
  .bar{display:block;height:4px;border-radius:3px;background:var(--surface-2);margin-top:6px;overflow:hidden}
  .bar-fill{display:block;height:100%;border-radius:3px}
  .bar-fill.up{background:var(--accent)} .bar-fill.down{background:var(--down)}
  .row-baseline .agent-name{color:var(--muted);font-weight:500}
  #spotlight{padding:64px 0}
  .spotlight-cta{margin:22px 0 0;text-align:center;font-size:1.05rem;color:var(--muted)}
  .spotlight-cta a{color:var(--accent);font-weight:650;text-decoration:none}
  .spotlight-cta a:hover{text-decoration:underline}
  .board-foot{display:flex;flex-wrap:wrap;gap:8px 18px;padding:13px 16px;font-size:12.5px;color:var(--muted);
    border-top:1px solid var(--border);font-family:var(--mono)}

  /* pillars */
  .pillars{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
  @media(max-width:720px){.pillars{grid-template-columns:1fr}}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:22px}
  .card h3{margin:0 0 8px;font-size:1.06rem;letter-spacing:-.02em}
  .card p{margin:0;color:var(--muted);font-size:14.5px}
  .card .ic{width:34px;height:34px;border-radius:9px;background:var(--accent-weak);color:var(--accent);
    display:grid;place-items:center;margin-bottom:14px;font-family:var(--mono);font-weight:700}

  /* steps */
  .steps{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;counter-reset:s}
  @media(max-width:720px){.steps{grid-template-columns:1fr}}
  .step{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px}
  .step::before{counter-increment:s;content:counter(s);font-family:var(--mono);font-weight:700;
    color:var(--accent);font-size:13px;display:block;margin-bottom:8px}
  .step h4{margin:0 0 4px;font-size:.98rem}
  .step p{margin:0;color:var(--muted);font-size:13.5px}

  /* submit */
  .submit{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:30px;
    display:grid;grid-template-columns:1.2fr 1fr;gap:28px}
  @media(max-width:720px){.submit{grid-template-columns:1fr}}
  pre{margin:0;background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:14px 16px;
    overflow-x:auto;font-family:var(--mono);font-size:12.5px;line-height:1.7;color:var(--text)}
  .submit li{color:var(--muted);font-size:14.5px;margin:6px 0}

  footer{border-top:1px solid var(--border);padding:34px 0 46px;color:var(--muted);font-size:13px}
  .foot-in{display:flex;flex-wrap:wrap;gap:16px;justify-content:space-between;align-items:center}
  .foot-in a{color:var(--muted);text-decoration:none;margin-right:16px}
  .foot-in a:hover{color:var(--text)}
  .hash{font-family:var(--mono);font-size:12px}
</style>
</head>
<body>
<nav><div class="wrap nav-in">
  <a class="brand" href="#top" style="text-decoration:none">
    <span class="mk">SF</span><span>SynthFin&nbsp;Trading&nbsp;Bench</span>
  </a>
  <div class="nav-links">
    <a href="#leaderboard">Leaderboard</a>
    <a href="#how" class="nav-hide">How it works</a>
    <a href="#submit">Submit</a>
    <a href="@@GITHUB@@" class="nav-cta">GitHub</a>
    <button class="theme-btn" id="themeBtn" aria-label="Toggle theme" title="Toggle theme">◐</button>
  </div>
</div></nav>

<header class="hero" id="top">
  <svg class="tape" preserveAspectRatio="none" viewBox="0 0 1200 400" aria-hidden="true">
    <defs>
      <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="var(--accent)" stop-opacity=".18"/>
        <stop offset="1" stop-color="var(--accent)" stop-opacity="0"/>
      </linearGradient>
    </defs>
    <path fill="url(#g)" stroke="none" d="M0,300 L60,290 110,305 160,250 210,268 260,210 320,232 370,180 430,205 480,150 540,172 600,120 660,158 720,96 780,140 840,80 900,124 960,70 1020,118 1080,66 1140,110 1200,74 L1200,400 L0,400 Z"/>
    <path fill="none" stroke="var(--accent)" stroke-width="2" stroke-opacity=".55" d="M0,300 L60,290 110,305 160,250 210,268 260,210 320,232 370,180 430,205 480,150 540,172 600,120 660,158 720,96 780,140 840,80 900,124 960,70 1020,118 1080,66 1140,110 1200,74"/>
  </svg>
  <div class="wrap hero-in">
    <p class="eyebrow">Contamination-free · fully synthetic markets</p>
    <h1>The trading benchmark models can't have memorized.</h1>
    <p class="lede">LLM agents manage a portfolio through curated macro regimes - AI-bubble deflations,
      credit crises, geopolitical shocks - seeing only point-in-time prices, filings, news and
      estimates. Every market is <b>machine-generated and unpublished</b>, so no model has ever seen
      it. Scored on realized, risk-adjusted <b>stock-selection skill</b>.</p>
    <div class="cta-row">
      <a class="btn btn-primary" href="#leaderboard">View the leaderboard</a>
      <a class="btn btn-ghost" href="@@GITHUB@@">Read the methodology →</a>
    </div>
    <div class="stats">
      <div class="stat"><div class="v">@@NSCEN@@</div><div class="k">macro scenarios</div></div>
      <div class="stat"><div class="v">@@NTICKERS@@</div><div class="k">tickers per scenario</div></div>
      <div class="stat"><div class="v accent">0%</div><div class="k">training contamination</div></div>
      <div class="stat"><div class="v">~2&nbsp;years</div><div class="k">simulated per scenario</div></div>
    </div>
  </div>
</header>

<main>
<section id="leaderboard">
  <div class="wrap">
    <div class="sec-head">
      <h2>Leaderboard</h2>
      <p class="sec-note">Ranked by Sharpe across scenarios. <b>Appraisal</b> is annualized
        alpha ÷ idiosyncratic vol - skill net of market beta.</p>
    </div>
    @@STATUS@@
    <div class="board-wrap">
      <div class="board-scroll">
        <table id="board">
          <thead><tr>
            <th data-k="rank">#</th>
            <th data-k="name">Agent</th>
            <th data-k="ret">Return</th>
            <th data-k="sharpe">Sharpe</th>
            <th data-k="alpha">Alpha</th>
            <th data-k="appraisal" data-active>Appraisal <span class="caret">▼</span></th>
            <th data-k="beta">Beta</th>
            <th data-k="mdd">Max DD</th>
            <th data-k="win">Win</th>
          </tr></thead>
          <tbody id="board-body">@@ROWS@@
          </tbody>
        </table>
      </div>
      <div class="board-foot">
        <span>corpus · @@PROFILE@@</span>
        <span>hash · @@HASH@@…</span>
        <span>rebalance · @@REBAL@@d</span>
        <span>cost · 5bps</span>
      </div>
    </div>
  </div>
</section>

@@SPOTLIGHT@@

<section id="why" style="background:var(--surface-2)">
  <div class="wrap">
    <div class="sec-head"><h2>Why it's different</h2>
      <p class="sec-note">Real-market trading benchmarks fight data leakage. This one removes it by construction.</p></div>
    <div class="pillars">
      <div class="card"><div class="ic">∅</div><h3>Provably uncontaminated</h3>
        <p>The data does not exist outside this benchmark. There is no corpus to have trained on - so a
          high score measures capability, never memorization.</p></div>
      <div class="card"><div class="ic">↻</div><h3>Refreshable</h3>
        <p>Suspect leakage after release? Generate a new corpus with new seeds and re-run. The task is
          fixed; the data is disposable - the property static benchmarks can't offer.</p></div>
      <div class="card"><div class="ic">α</div><h3>Skill, not luck</h3>
        <p>Returns regress onto the market to isolate alpha; the appraisal ratio measures stock-selection
          skill net of beta. Beating a rising tide doesn't count.</p></div>
    </div>
  </div>
</section>

<section id="how">
  <div class="wrap">
    <div class="sec-head"><h2>How it works</h2>
      <p class="sec-note">A sequential portfolio task with strict no-lookahead execution.</p></div>
    <div class="steps">
      <div class="step"><h4>Observe</h4><p>Each period the agent sees point-in-time prices, 10-Q
        financials, earnings, analyst estimates, ownership and news - all dated on or before the day.</p></div>
      <div class="step"><h4>Allocate</h4><p>It returns a target portfolio as weights across the universe.
        Malformed output is clipped to the constraints, never invalid.</p></div>
      <div class="step"><h4>Execute</h4><p>Orders fill at the next session's open with transaction costs.
        The one-bar delay removes same-bar lookahead.</p></div>
      <div class="step"><h4>Score</h4><p>Over ~2 years the book is marked daily; we report Sharpe,
        drawdown, and CAPM alpha / appraisal against the market.</p></div>
    </div>
  </div>
</section>

<section id="submit" style="background:var(--surface-2)">
  <div class="wrap">
    <div class="sec-head"><h2>Submit a model</h2>
      <p class="sec-note">Run the open harness on the frozen corpus, or open a PR with an agent spec and
        we'll run it.</p></div>
    <div class="submit">
      <div>
        <pre>pip install -e ".[all]"

# add your model to a config, then:
python -m bench.runner configs/default.yaml \
    --run my-model

# rebuild this leaderboard from results:
python scripts/build_site.py --run my-model</pre>
      </div>
      <div>
        <ol style="padding-left:18px;margin:0">
          <li>Every submission pins the model id + decoding params, the corpus content-hash, and the
            harness commit.</li>
          <li>We verify by re-scoring your trajectories with the public scorer.</li>
          <li>Numbers are published alongside the baselines and cross-scenario variance - a single
            averaged number without a baseline isn't meaningful.</li>
        </ol>
      </div>
    </div>
  </div>
</section>
</main>

<footer>
  <div class="wrap foot-in">
    <div>
      <a href="@@GITHUB@@">GitHub</a>
      <a href="@@GITHUB@@/blob/main/docs/METHODOLOGY.md">Methodology</a>
      <a href="@@GITHUB@@/blob/main/docs/CONTAMINATION.md">Contamination statement</a>
      <a href="@@GITHUB@@/blob/main/docs/DATA_CARD.md">Data card</a>
    </div>
    <div class="hash">Apache-2.0 · corpus @@PROFILE@@ · @@HASH@@…</div>
  </div>
</footer>

<script>
(function(){
  // theme toggle: stamp data-theme so it overrides prefers-color-scheme both ways
  var root=document.documentElement, btn=document.getElementById('themeBtn');
  var saved=null; try{saved=localStorage.getItem('sftb-theme');}catch(e){}
  if(saved) root.setAttribute('data-theme',saved);
  btn.addEventListener('click',function(){
    var cur=root.getAttribute('data-theme');
    if(!cur){cur=matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';}
    var next=cur==='dark'?'light':'dark';
    root.setAttribute('data-theme',next);
    try{localStorage.setItem('sftb-theme',next);}catch(e){}
  });

  // client-side column sort
  var body=document.getElementById('board-body');
  var rows=Array.prototype.slice.call(body.querySelectorAll('tr'));
  function val(tr,k){
    var idx={rank:0,name:1,ret:2,sharpe:3,alpha:4,appraisal:5,beta:6,mdd:7,win:8}[k];
    var td=tr.children[idx];
    if(k==='name') return td.innerText.trim().toLowerCase();
    var m=td.innerText.replace(/[%+]/g,'').replace(/[^0-9.\-]/g,'');
    return parseFloat(m);
  }
  var dir={};
  document.querySelectorAll('#board thead th').forEach(function(th){
    th.addEventListener('click',function(){
      var k=th.getAttribute('data-k');
      dir[k]=!dir[k];
      var asc=dir[k];
      document.querySelectorAll('#board thead th').forEach(function(x){
        x.removeAttribute('data-active');var c=x.querySelector('.caret');if(c)c.remove();});
      th.setAttribute('data-active','');
      var car=document.createElement('span');car.className='caret';car.textContent=asc?'▲':'▼';
      th.appendChild(document.createTextNode(' '));th.appendChild(car);
      rows.sort(function(a,b){
        var va=val(a,k),vb=val(b,k);
        if(typeof va==='string'){return asc?va.localeCompare(vb):vb.localeCompare(va);}
        if(isNaN(va))va=-Infinity; if(isNaN(vb))vb=-Infinity;
        return asc?va-vb:vb-va;
      });
      rows.forEach(function(r){body.appendChild(r);});
    });
  });
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
