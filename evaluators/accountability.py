"""ACCOUNTABILITY & GOVERNANCE. Cannot be tested with prompts, so it is a checklist
about the evaluated application. Edit datasets/accountability_checklist.json and set
"answer" to true/false honestly for the app you are evaluating.

  score = checklist items answered true / total items
This is an application-level score, so every model gets the same value.
"""
from .base import Evaluator, PillarResult
from .utils import load_json


class AccountabilityEvaluator(Evaluator):
    pillar = "accountability"

    def __init__(self, dataset_path="datasets/accountability_checklist.json"):
        super().__init__()
        self.items = load_json(dataset_path)

    def run(self, client, delay=0.0):
        yes = sum(1 for i in self.items if i["answer"])
        score = yes / max(len(self.items), 1)
        return PillarResult(self.pillar, round(score, 3),
                            {"items_met": yes, "items_total": len(self.items)}, self.items)
