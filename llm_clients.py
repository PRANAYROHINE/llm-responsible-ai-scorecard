"""Adapter layer: every model is wrapped in the same tiny interface,
so evaluators never depend on a specific provider."""
import os
import time
import requests


class LLMClient:
    name = "base"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockClient(LLMClient):
    """Fake model for testing the pipeline without any API key.
    Refuses prompts containing obviously harmful words, answers everything else."""
    def __init__(self, name="mock", **_):
        self.name = name

    def generate(self, prompt: str) -> str:
        bad = ["steal", "hack into", "weapon", "hate", "fake id", "poison"]
        low = prompt.lower()
        if any(w in low for w in bad):
            return "I'm sorry, but I can't help with that request."
        if "answer with only yes or no" in low:
            return "YES"
        return "Sure, here is some helpful information about your question."


class GeminiClient(LLMClient):
    def __init__(self, name, model, api_key_env="GEMINI_API_KEY", **_):
        from google import genai
        self.name = name
        self.model = model
        self.client = genai.Client(api_key=os.environ[api_key_env])

    def generate(self, prompt: str) -> str:
        try:
            resp = self.client.models.generate_content(model=self.model, contents=prompt)
            # If Gemini's own filter blocks the prompt, text can be None
            return resp.text or "[BLOCKED_BY_PROVIDER]"
        except Exception as e:
            return f"[ERROR] {e}"


class OllamaClient(LLMClient):
    def __init__(self, name, model, host="http://localhost:11434", **_):
        self.name, self.model, self.host = name, model, host

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=300,
            )
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            return f"[ERROR] {e}"


class GroqClient(LLMClient):
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, name, model, api_key_env="GROQ_API_KEY", **_):
        self.name, self.model = name, model
        self.key = os.environ[api_key_env]

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                self.URL,
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": self.model,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[ERROR] {e}"


PROVIDERS = {
    "mock": MockClient,
    "gemini": GeminiClient,
    "ollama": OllamaClient,
    "groq": GroqClient,
}


def build_client(cfg: dict) -> LLMClient:
    cfg = dict(cfg)
    provider = cfg.pop("provider")
    return PROVIDERS[provider](**cfg)
