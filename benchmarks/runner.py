"""Live backend measurements and adapters for saved experiment outputs."""

import asyncio
from contextlib import aclosing
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import time

from benchmarks.scoring import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def saved_demo_records(paths, cases):
    """Grade historical outputs against their actual saved requests, without inventing timings."""
    by_id = {case["id"]: case for case in cases}
    records = []
    for path in paths:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict) or not all(key in payload for key in ("model", "report", "unsupported")):
            raise ValueError(f"{path}: expected a scripts/report_demo.py result.json file.")
        report_case = deepcopy(by_id["report"])
        report_case["question"] = payload["report"]["question"]
        word_limit = re.search(r"(?:under|fewer than)\s+(\d+)\s+words", report_case["question"], re.I)
        if word_limit:
            report_case["max_words"] = int(word_limit.group(1)) - 1
        else:
            report_case.pop("max_words", None)
        profile = f"saved-{path.parent.name}-{fingerprint(report_case)[:8]}"
        for key, case in (("report", report_case), ("unsupported", deepcopy(by_id["unsupported"]))):
            saved = payload[key]
            case["question"] = saved["question"]
            records.append({
                "record_id": fingerprint([str(path.resolve()), key])[:16],
                "model": payload["model"], "profile": profile, "repeat": 1,
                "case": case, "answer": saved["answer"], "sources": saved["sources"],
                "status": "complete", "error": None, "timing": {},
                "provenance": {"mode": "saved_demo", "file": str(path), "timing_available": False},
            })
    return records


def load_predictions(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("records"), list):
        raise ValueError("Expected a benchmark results.json file with schema_version 1 and a records list.")
    if not payload["records"]:
        raise ValueError("No prediction records found.")
    return payload


def resolve_documents(cases):
    paths = {}
    basenames = {}
    for case in cases:
        for name in case["documents"]:
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError(f"Benchmark document must be inside this checkout: {name}")
            if path.name in basenames and basenames[path.name] != path:
                raise ValueError("Benchmark source filenames must be unique to make source-scope checks unambiguous.")
            if not path.is_file():
                raise ValueError(f"Benchmark document not found: {name}")
            paths[name] = path
            basenames[path.name] = path
    return paths


def seconds(value):
    return value / 1_000_000_000 if type(value) is int and value >= 0 else None


async def measure_case(library, settings, case, document_ids, *, repeat, seed, timeout, max_tokens, context_size=None):
    # These are the same prompt, retrieval, and streaming model factory used by the app.
    from chat import NOT_FOUND, build_messages, chat_model, source_records

    record = {
        "record_id": fingerprint([settings.chat_model, case, repeat])[:16],
        "model": settings.chat_model, "profile": "live", "repeat": repeat,
        "case": deepcopy(case), "answer": "", "sources": [], "status": "error", "error": None,
        "timing": {}, "done_reason": None,
    }
    model = chat_model(settings)
    updates = {"seed": seed}
    if max_tokens is not None:
        updates["num_predict"] = max_tokens
    if context_size is not None:
        updates["num_ctx"] = context_size
    model = model.model_copy(update=updates)
    record["generation_options"] = {
        "seed": seed, "temperature": model.temperature, "num_ctx": model.num_ctx, "num_predict": model.num_predict,
    }
    metadata, usage = {}, {}
    started = time.perf_counter()
    generation_started = None
    try:
        docs = await asyncio.to_thread(
            library.retrieve, case["question"], document_ids=document_ids, k=3,
        )
        record["timing"]["retrieval_seconds"] = time.perf_counter() - started
        record["sources"] = source_records(docs)
        generation_started = time.perf_counter()
        if not docs:
            # The app returns its fixed abstention without invoking the model in this case.
            record["answer"] = NOT_FOUND
            record["model_invoked"] = False
        else:
            record["model_invoked"] = True
            async with asyncio.timeout(timeout):
                async with aclosing(model.astream(build_messages(case["question"], docs, []))) as stream:
                    async for chunk in stream:
                        if isinstance(chunk.content, str) and chunk.content:
                            record["timing"].setdefault("time_to_first_token_seconds", time.perf_counter() - started)
                            record["answer"] += chunk.content
                        metadata.update(chunk.response_metadata or {})
                        if chunk.usage_metadata:
                            usage.update(chunk.usage_metadata)
        record["status"] = "complete" if record["answer"].strip() else "error"
        if record["status"] == "error":
            record["error"] = "The model returned an empty answer."
    except TimeoutError:
        record["status"] = "timeout"
        record["error"] = f"Generation exceeded {timeout:g} seconds; the stream was cancelled."
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        finished = time.perf_counter()
        record["timing"]["total_seconds"] = finished - started
        if generation_started is not None:
            record["timing"]["generation_seconds"] = finished - generation_started
        record["done_reason"] = metadata.get("done_reason")
        for source, destination in (("load_duration", "load_seconds"), ("prompt_eval_duration", "prompt_eval_seconds"),
                                    ("eval_duration", "eval_duration_seconds")):
            if seconds(metadata.get(source)) is not None:
                record["timing"][destination] = seconds(metadata[source])
        for source, fallback, destination in (("eval_count", "output_tokens", "output_tokens"),
                                               ("prompt_eval_count", "input_tokens", "input_tokens")):
            value = metadata.get(source, usage.get(fallback))
            if type(value) is int and value >= 0:
                record["timing"][destination] = value
    return record


async def run_live(args, suite, progress=None, checkpoint=None):
    # Heavy dependencies and all network access stay behind the explicit --live mode.
    import httpx
    from config import Settings
    from ingest import DocumentLibrary
    from local_models import inspect_ollama, normalized_model
    from chat import SYSTEM_PROMPT

    settings = Settings.from_env()
    selected_cases = [case for case in suite["cases"] if not args.cases or case["id"] in args.cases]
    models = args.models or [settings.chat_model]
    if len({normalized_model(model) for model in models}) != len(models):
        raise ValueError("Choose distinct models; aliases such as phi3 and phi3:latest refer to the same model.")
    model_settings = [replace(settings, chat_model=model) for model in models]
    status = inspect_ollama(settings)
    if not status.online:
        raise ValueError("Ollama is offline. Start the local service before running --live.")
    missing = [model for model in models + [settings.embedding_model] if not status.has(model)]
    if missing:
        raise ValueError("Models are not installed: " + ", ".join(missing) + ". Downloads are never started automatically.")
    document_paths = resolve_documents(suite["cases"])
    digests, server_version = {}, None
    with httpx.Client(timeout=5, trust_env=False) as client:
        try:
            for model in client.get(settings.ollama_url + "/api/tags").json().get("models", []):
                if model.get("name") in status.models:
                    digests[normalized_model(model["name"])] = model.get("digest")
            server_version = client.get(settings.ollama_url + "/api/version").json().get("version")
        except (httpx.HTTPError, ValueError, TypeError):
            pass
    metadata = {
        "mode": "live_backend", "suite_fingerprint": fingerprint(suite),
        "system_prompt_fingerprint": fingerprint(SYSTEM_PROMPT),
        "pipeline_fingerprints": {name: fingerprint((ROOT / name).read_text(encoding="utf-8"))
                                  for name in ("chat.py", "ingest.py", "config.py")},
        "source_fingerprints": {name: sha256(path.read_bytes()).hexdigest()
                                for name, path in document_paths.items()},
        "models": models, "model_digests": digests, "ollama_version": server_version,
        "embedding_model": settings.embedding_model, "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap, "top_k": 3, "history": [],
        "generation_timeout_seconds": args.timeout, "seed_base": args.seed, "repetitions": args.repeat,
        "selected_case_ids": [case["id"] for case in selected_cases],
        "python_host": {"platform": platform.platform(), "python": platform.python_version(), "logical_cpus": os.cpu_count()},
        "packages": {name: importlib.metadata.version(name) for name in ("langchain-ollama", "langchain-chroma", "chromadb")},
        "timing_scope": "Backend retrieval and generation; excludes ingestion and UI/HTTP/persistence. No forced cache reset or warm-up.",
    }
    records = []
    # The workspace is newly created under the selected output directory. Never use the app's data/.
    with tempfile.TemporaryDirectory(prefix="workspace-", dir=args.output, ignore_cleanup_errors=True) as directory:
        library = DocumentLibrary(replace(settings, data_dir=Path(directory)))
        indexed = {}
        ingest_started = time.perf_counter()
        for name, path in document_paths.items():
            if progress:
                progress(f"Indexing benchmark fixture: {name}")
            result = await asyncio.to_thread(library.ingest, path.name, path.read_bytes())
            indexed[name] = result.document_id
        metadata["ingestion_seconds"] = time.perf_counter() - ingest_started
        for config in model_settings:
            for repetition in range(1, args.repeat + 1):
                for case in selected_cases:
                    if progress:
                        progress(f"Running {config.chat_model}: {case['id']} (repeat {repetition}/{args.repeat})")
                    async def heartbeat():
                        started = time.perf_counter()
                        while True:
                            await asyncio.sleep(15)
                            progress(f"  Still running {case['id']} ({time.perf_counter() - started:.0f}s elapsed)...")

                    notice = asyncio.create_task(heartbeat()) if progress else None
                    try:
                        record = await measure_case(
                            library, config, case, [indexed[name] for name in case["documents"]],
                            repeat=repetition, seed=args.seed + repetition - 1, timeout=args.timeout, max_tokens=args.max_tokens,
                            context_size=args.context_size,
                        )
                    finally:
                        if notice:
                            notice.cancel()
                            await asyncio.gather(notice, return_exceptions=True)
                    records.append(record)
                    if checkpoint:
                        checkpoint(records, metadata)
                    if progress:
                        progress(f"  {record['status']}: {record['timing']['total_seconds']:.2f}s, {len(record['answer'].split())} words")
    return records, metadata
