"""Opt-in real-model smoke check. Uses a temporary library; never alters user data."""

from dataclasses import replace
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from chat import stream_answer
from ingest import DocumentLibrary
from local_models import inspect_ollama


def main():
    settings = Settings.from_env()
    status = inspect_ollama(settings)
    if not status.online or not all(status.has(m) for m in (settings.chat_model, settings.embedding_model)):
        raise SystemExit("Start Ollama and download both configured models before running the live smoke check.")
    # Chroma keeps native mmap handles open on Windows until process exit.
    # Cleanup is best effort; all ingestion and answer assertions remain strict.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        library = DocumentLibrary(replace(settings, data_dir=Path(directory)))
        sample = Path(__file__).resolve().parents[1] / "samples" / "handbook.txt"
        result = library.ingest(sample.name, sample.read_bytes())
        docs = library.retrieve("What is the annual learning budget?")
        assert docs and any("30,000" in doc.page_content for doc in docs), "Expected evidence was not retrieved."
        print(f"Indexed {result.chunks} passages; retrieved {len(docs)} excerpts.", flush=True)
        answer = "".join(stream_answer(settings, "What is the annual learning budget?", docs, []))
        print(answer, flush=True)
        assert "30,000" in answer or "30000" in answer, "The model did not return the expected budget."
        assert any(f"[{i}]" in answer for i in range(1, len(docs) + 1)), "Expected source citation missing."
        missing = "".join(stream_answer(settings, "What is the CEO's birthday?", docs, []))
        print(f"Missing information check: {missing}", flush=True)
        assert "cannot find this information" in missing.lower(), "The model did not abstain for an unsupported question."
        library.delete(result.document_id)
        assert not library.list_documents(), "Document deletion failed."
        print("Live local-model smoke check passed.", flush=True)


if __name__ == "__main__":
    main()
