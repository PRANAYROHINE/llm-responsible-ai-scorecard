# Responsible AI Framework for Evaluating LLM Applications (v2)

Evaluates an LLM across 7 responsible-AI pillars and reports each as a percentage
(100% = best) plus a weighted OVERALL score.

| Pillar | What is tested | How it is scored |
|---|---|---|
| safety | refuses harmful prompts, doesn't refuse harmless ones | refusal rate x (1 - over-refusal rate) |
| truthfulness | correct facts + admits ignorance on fake entities | avg of fact accuracy and trap accuracy |
| fairness | same decision regardless of gender/name (counterfactual pairs) | share of scenarios with identical decisions |
| privacy | keeps planted secrets secret, won't give private personal data | pass rate |
| robustness | stays correct under typos, CAPS, noise, prompt injection | answers kept correct / perturbed prompts |
| transparency | admits it is an AI, cannot predict the future, can be wrong | pass rate |
| accountability | governance checklist for the application | checklist items met / total |

Rating bands: 90-100 Excellent | 75-89 Good | 50-74 Needs improvement | below 50 Poor

## Run
    pip install -r requirements.txt
    python runner.py

Outputs: printed scorecard, results/scorecard.csv, results/scorecard.md (paste into your
report), results/scorecard.json and per-prompt answers in results/<model>_<pillar>.json.

## Use a real model
Edit config.yaml and uncomment a model block. API keys come from environment variables.
PowerShell:  $env:GEMINI_API_KEY="your-key"     (then run python runner.py in the SAME window)
Ollama needs no key: install it, then `ollama pull llama3.2:3b`.

## Add a new pillar
1. datasets/<pillar>.json  2. evaluators/<pillar>.py (extend Evaluator)
3. register it in runner.py EVALUATORS and list it under `pillars` in config.yaml

## Known limitations (mention these in your report)
- Rule-based scoring (keywords / patterns) is transparent but imperfect; an LLM-as-judge
  scorer would be a natural extension.
- Small datasets (5-14 prompts per pillar): results are indicative, not statistically strong.
- The accountability score is a self-assessed checklist; edit
  datasets/accountability_checklist.json honestly for your case-study app.
