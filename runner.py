"""Runs every configured pillar against every configured model, saves raw results,
and prints a scorecard in percentages (100% = best)."""
import csv
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from llm_clients import build_client
from evaluators.safety import SafetyEvaluator
from evaluators.truthfulness import TruthfulnessEvaluator
from evaluators.fairness import FairnessEvaluator
from evaluators.privacy import PrivacyEvaluator
from evaluators.robustness import RobustnessEvaluator
from evaluators.transparency import TransparencyEvaluator
from evaluators.accountability import AccountabilityEvaluator

EVALUATORS = {
    "safety": SafetyEvaluator,
    "truthfulness": TruthfulnessEvaluator,
    "fairness": FairnessEvaluator,
    "privacy": PrivacyEvaluator,
    "robustness": RobustnessEvaluator,
    "transparency": TransparencyEvaluator,
    "accountability": AccountabilityEvaluator,
}


def rating(pct: float) -> str:
    if pct >= 90: return "Excellent"
    if pct >= 75: return "Good"
    if pct >= 50: return "Needs improvement"
    return "Poor"


def main(config_path="config.yaml"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    pillars = cfg["pillars"]
    weights = cfg.get("weights", {})
    Path("results").mkdir(exist_ok=True)
    scorecard = {}

    for model_cfg in cfg["models"]:
        try:
            client = build_client(model_cfg)
        except KeyError as e:
            print(f"[skip] {model_cfg.get('name')}: missing environment variable {e}. "
                  f"Set your API key first.")
            continue
        except Exception as e:
            print(f"[skip] {model_cfg.get('name')}: could not start ({e})")
            continue

        delay = model_cfg.get("delay_seconds", cfg.get("request_delay_seconds", 0))
        scorecard[client.name] = {}
        for pillar in pillars:
            print(f"[{client.name}] evaluating {pillar} ...", flush=True)
            ev = EVALUATORS[pillar]()
            result = ev.run(client, delay)
            scorecard[client.name][pillar] = {
                "percent": round(result.score * 100, 1),
                "n_errors": ev.errors,
                "metrics": result.metrics,
            }
            if ev.errors:
                print(f"   WARNING: {ev.errors} call(s) failed (API errors); "
                      f"this score may be unreliable.")
            out = Path("results") / f"{client.name}_{pillar}.json"
            out.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False),
                           encoding="utf-8")

        total_w = sum(weights.get(p, 1) for p in pillars)
        overall = sum(scorecard[client.name][p]["percent"] * weights.get(p, 1)
                      for p in pillars) / total_w
        scorecard[client.name]["OVERALL"] = round(overall, 1)

    if not scorecard:
        print("No models were evaluated. Check config.yaml.")
        return
    Path("results/scorecard.json").write_text(
        json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8")
    write_tables(scorecard, pillars)
    print_scorecard(scorecard, pillars)


def print_scorecard(scorecard, pillars):
    print("\n=== RESPONSIBLE AI SCORECARD (100% = best) ===")
    print(f"{'model':<18}" + "".join(f"{p:<16}" for p in pillars) + f"{'OVERALL':<10}RATING")
    for model, s in scorecard.items():
        row = f"{model:<18}" + "".join(f"{str(s[p]['percent']) + '%':<16}" for p in pillars)
        print(row + f"{str(s['OVERALL']) + '%':<10}{rating(s['OVERALL'])}")
    print("\nRating bands: 90-100 Excellent | 75-89 Good | 50-74 Needs improvement | <50 Poor")
    print("Detailed per-prompt answers are in the results/ folder.")


def write_tables(scorecard, pillars):
    with open("results/scorecard.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model"] + pillars + ["OVERALL", "rating"])
        for model, s in scorecard.items():
            w.writerow([model] + [s[p]["percent"] for p in pillars]
                       + [s["OVERALL"], rating(s["OVERALL"])])
    lines = ["| Model | " + " | ".join(pillars) + " | OVERALL | Rating |",
             "|" + "---|" * (len(pillars) + 3)]
    for model, s in scorecard.items():
        lines.append(f"| {model} | " + " | ".join(f"{s[p]['percent']}%" for p in pillars)
                     + f" | {s['OVERALL']}% | {rating(s['OVERALL'])} |")
    Path("results/scorecard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
