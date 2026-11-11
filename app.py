"""Streamlit dashboard for the Responsible AI LLM evaluation framework.

Put this file next to runner.py and config.yaml, then run:
    streamlit run app.py
"""
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from llm_clients import build_client
from runner import EVALUATORS, rating

CONFIG_PATH = "config.yaml"
RESULTS = Path("results")

st.set_page_config(page_title="Responsible AI Scorecard", layout="wide")
st.title("Responsible AI Scorecard for LLMs")
st.caption("100% = best. Each pillar is scored by its own evaluator.")

cfg = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
weights = cfg.get("weights", {})


def run_all(model_cfgs, pillars, bar):
    """Same logic as runner.main(), but with a progress bar and no printing."""
    scorecard, skipped = {}, []
    total, done = len(model_cfgs) * len(pillars), 0
    RESULTS.mkdir(exist_ok=True)

    for model_cfg in model_cfgs:
        name = model_cfg.get("name", "?")
        try:
            client = build_client(model_cfg)
        except KeyError as e:
            skipped.append(f"{name}: missing environment variable {e}")
            done += len(pillars)
            bar.progress(done / total)
            continue
        except Exception as e:
            skipped.append(f"{name}: could not start ({e})")
            done += len(pillars)
            bar.progress(done / total)
            continue

        delay = model_cfg.get("delay_seconds", cfg.get("request_delay_seconds", 0))
        scorecard[client.name] = {}
        for pillar in pillars:
            bar.progress(done / total, text=f"{client.name}: evaluating {pillar}...")
            ev = EVALUATORS[pillar]()
            result = ev.run(client, delay)
            scorecard[client.name][pillar] = {
                "percent": round(result.score * 100, 1),
                "n_errors": ev.errors,
                "metrics": result.metrics,
            }
            (RESULTS / f"{client.name}_{pillar}.json").write_text(
                json.dumps(asdict(result), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            done += 1

        total_w = sum(weights.get(p, 1) for p in pillars)
        scorecard[client.name]["OVERALL"] = round(
            sum(scorecard[client.name][p]["percent"] * weights.get(p, 1)
                for p in pillars) / total_w, 1)

    if scorecard:
        (RESULTS / "scorecard.json").write_text(
            json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8")
    return scorecard, skipped


def show(scorecard):
    pillars = [k for k in next(iter(scorecard.values())) if k != "OVERALL"]

    rows = []
    for model, s in scorecard.items():
        row = {"Model": model}
        row.update({p: s[p]["percent"] for p in pillars})
        row["OVERALL"] = s["OVERALL"]
        row["Rating"] = rating(s["OVERALL"])
        rows.append(row)
    df = pd.DataFrame(rows)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Scorecard")
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False),
                           "scorecard.csv", "text/csv")
        st.caption("Bands: 90-100 Excellent | 75-89 Good | "
                   "50-74 Needs improvement | <50 Poor")
    with right:
        st.subheader("Pillar comparison")
        fig = go.Figure()
        for model, s in scorecard.items():
            vals = [s[p]["percent"] for p in pillars]
            fig.add_trace(go.Scatterpolar(r=vals + vals[:1],
                                          theta=pillars + pillars[:1],
                                          name=model, fill="toself", opacity=0.6))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Details per model")
    for model, s in scorecard.items():
        with st.expander(f"{model}  -  {s['OVERALL']}% ({rating(s['OVERALL'])})"):
            for p in pillars:
                st.markdown(f"**{p}: {s[p]['percent']}%**")
                if s[p].get("n_errors"):
                    st.warning(f"{s[p]['n_errors']} API call(s) failed; "
                               "this score may be unreliable.")
                st.json(s[p].get("metrics", {}), expanded=False)


with st.sidebar:
    mode = st.radio("Mode", ["Run new evaluation", "View saved results"])
    if mode == "Run new evaluation":
        names = [m["name"] for m in cfg["models"]]
        chosen_models = st.multiselect("Models", names, default=names[:1])
        chosen_pillars = st.multiselect("Pillars", list(EVALUATORS),
                                        default=cfg["pillars"])
        run = st.button("Run evaluation", type="primary",
                        disabled=not (chosen_models and chosen_pillars))

if mode == "Run new evaluation":
    if run:
        selected = [m for m in cfg["models"] if m["name"] in chosen_models]
        bar = st.progress(0.0, text="Starting...")
        t0 = time.time()
        scorecard, skipped = run_all(selected, chosen_pillars, bar)
        bar.empty()
        for msg in skipped:
            st.warning(f"Skipped {msg}")
        if scorecard:
            st.session_state["scorecard"] = scorecard
            st.success(f"Done in {time.time() - t0:.0f}s. Saved to results/.")
    if "scorecard" in st.session_state:
        show(st.session_state["scorecard"])
    else:
        st.info("Pick models and pillars in the sidebar, then click Run evaluation.")
else:
    path = RESULTS / "scorecard.json"
    if path.exists():
        show(json.loads(path.read_text(encoding="utf-8")))
    else:
        st.info("No saved results yet. Run an evaluation first.")
