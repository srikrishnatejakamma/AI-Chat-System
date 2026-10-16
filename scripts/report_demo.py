"""Try document upload and report generation through the real API in an isolated workspace."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api import create_app
from config import Settings
from local_models import inspect_ollama


def checked(response):
    response.raise_for_status()
    return response.json()


def ask(client, question, document_id):
    chat = checked(client.post("/api/chats"))
    chat_id = chat["id"]
    chat = checked(client.post(f"/api/chats/{chat_id}/messages", json={
        "content": question, "document_ids": [document_id], "top_k": 3,
    }))
    deadline = time.monotonic() + 600
    next_notice = time.monotonic() + 15
    while chat["active"]:
        if time.monotonic() > deadline:
            checked(client.post(f"/api/chats/{chat_id}/stop"))
            raise RuntimeError("The local model did not finish within 10 minutes.")
        if time.monotonic() >= next_notice:
            print("Waiting for the local CPU model...", flush=True)
            next_notice = time.monotonic() + 15
        time.sleep(0.5)
        chat = checked(client.get(f"/api/chats/{chat_id}"))
    message = chat["messages"][-1]
    version = message["versions"][message["selected_version"]]
    if version["status"] != "complete":
        raise RuntimeError(f"Generation did not complete: {version}")
    return {"question": question, "answer": message["content"], "sources": message["sources"]}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "report-demo")
    args = parser.parse_args()
    settings = Settings.from_env()
    status = inspect_ollama(settings)
    if not status.online:
        parser.exit(1, "Ollama is offline. Start the local Ollama service before this experiment.\n")
    if not all(status.has(model) for model in (settings.chat_model, settings.embedding_model)):
        parser.exit(1, "Download the configured chat and embedding models before this experiment.\n")
    args.output.mkdir(parents=True, exist_ok=True)
    sample = ROOT / "samples" / "handbook.txt"
    # Ignore only cleanup failures caused by Chroma's native Windows mmap handles.
    # The directory is isolated from the user's persistent documents and chats.
    with tempfile.TemporaryDirectory(prefix="workspace-", dir=args.output, ignore_cleanup_errors=True) as directory:
        app = create_app(replace(settings, data_dir=Path(directory)))
        with TestClient(app) as client:
            uploaded = checked(client.post("/api/documents", files={
                "file": (sample.name, sample.read_bytes(), "text/plain"),
            }))
            print(f"Uploaded {sample.name}: {uploaded['chunks']} indexed passages.", flush=True)
            report = ask(client,
                "What is the annual learning budget, what can it cover, and what approval is needed? "
                "Write a concise Markdown report using exactly these headings in this order: "
                "## Summary, ## Findings, ## Next steps. Cite factual claims with excerpt numbers. "
                "Clearly mark proposed actions as recommendations. Use only explicitly stated policies; "
                "do not invent eligibility restrictions. Keep the report under 150 words.", uploaded["document_id"])
            print(report["answer"], flush=True)
            (args.output / "report.md").write_text(report["answer"] + "\n", encoding="utf-8")
            missing = ask(client, "What is the CEO's birthday?", uploaded["document_id"])
            print("Unsupported question: " + missing["answer"], flush=True)
            checks = {
                "expected_budget": bool(re.search(r"30,?000", report["answer"])),
                "source_citation": bool(re.search(r"\[\d+\]", report["answer"])) and bool(report["sources"]),
                "report_heading_order": [line.strip() for line in report["answer"].splitlines() if line.startswith("## ")]
                    == ["## Summary", "## Findings", "## Next steps"],
                "unsupported_question": missing["answer"].strip()
                    == "I cannot find this information in the uploaded document.",
            }
            result = {"model": settings.chat_model, "embedding_model": settings.embedding_model,
                      "training_performed": False, "checks": checks, "report": report, "unsupported": missing}
            (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(checks, indent=2), flush=True)
            if not all(checks.values()):
                parser.exit(1, f"Some checks failed; inspect {args.output / 'result.json'}.\n")
    print(f"Real-model RAG report demo passed. Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
