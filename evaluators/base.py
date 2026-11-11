import time
from dataclasses import dataclass, field


@dataclass
class PillarResult:
    pillar: str
    score: float                 # 0.0 (worst) to 1.0 (best)
    metrics: dict = field(default_factory=dict)
    details: list = field(default_factory=list)   # per-prompt records


class Evaluator:
    pillar = "base"

    def __init__(self):
        self.errors = 0          # number of calls that still failed after retries

    def ask(self, client, prompt: str, delay: float = 0.0, retries: int = 4) -> str:
        """Call the model, retrying on API errors (e.g. free-tier rate limits)."""
        resp = ""
        for attempt in range(retries):
            resp = client.generate(prompt)
            if not resp.startswith("[ERROR]"):
                break
            time.sleep(8 * (attempt + 1))
        if resp.startswith("[ERROR]"):
            self.errors += 1
        time.sleep(delay)
        return resp

    def run(self, client, delay: float = 0.0) -> PillarResult:
        raise NotImplementedError
