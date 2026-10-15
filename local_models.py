"""Lightweight Ollama connectivity checks; never downloads a model implicitly."""

from dataclasses import dataclass

import httpx

from config import Settings


def normalized_model(name: str) -> str:
    return name if ":" in name else name + ":latest"


@dataclass(frozen=True)
class ModelStatus:
    online: bool
    models: tuple[str, ...] = ()
    error: str = ""

    def has(self, name: str) -> bool:
        return normalized_model(name) in {normalized_model(m) for m in self.models}


def inspect_ollama(settings: Settings) -> ModelStatus:
    try:
        # Ignore host HTTP proxies for private document traffic.
        with httpx.Client(timeout=3, trust_env=False) as client:
            response = client.get(f"{settings.ollama_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
        names = tuple(sorted(
            model["name"] for model in models
            if model.get("name") and not model.get("remote_host")
            and "cloud" not in model["name"].lower()
        ))
        return ModelStatus(online=True, models=names)
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        return ModelStatus(False, error="Cannot reach local Ollama. Start Ollama, then refresh the connection.")
