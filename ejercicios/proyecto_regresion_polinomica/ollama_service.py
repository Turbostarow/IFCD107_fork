"""Minimal client for the Ollama chat API used by the Flask endpoint."""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"

SYSTEM_PROMPT = """Eres un asistente docente especializado en regresión y aprendizaje automático.
Explica conceptos y resultados de forma clara. Distingue los datos sintéticos de los reales.
No afirmes haber ejecutado entrenamientos ni inventes métricas. Para recomendaciones,
considera el tamaño y tipo de datos, validación, sobreajuste y métricas pertinentes.
Responde en el idioma de la pregunta y no solicites ni reveles secretos o credenciales."""


class OllamaServiceError(RuntimeError):
    """Raised when the Ollama service cannot return a valid chat response."""


def ask_ollama(prompt: str) -> dict[str, str]:
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    timeout = max(1, int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120")))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise OllamaServiceError(f"Ollama respondió con HTTP {exc.code}.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise OllamaServiceError("Ollama no está disponible o agotó el tiempo de espera.") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OllamaServiceError("Ollama devolvió una respuesta no válida.") from exc

    message = result.get("message") if isinstance(result, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise OllamaServiceError("Ollama no devolvió contenido para la respuesta.")
    return {"model": str(result.get("model", model)), "answer": content.strip()}
