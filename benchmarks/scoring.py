"""Deterministic checks and reports. No model, network, or ML dependencies."""

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re

SCHEMA_VERSION = 1
REVIEW_FIELDS = ("factual_correctness", "citation_support", "relevance")


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def prose_lines(answer):
    """Exclude fenced blocks so code cannot spoof headings or citations."""
    fence = None
    for line in answer.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            value = marker.group(1)
            if fence is None:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence) and not line[marker.end():].strip():
                fence = None
            continue
        if fence is None:
            yield line


def markdown_headings(answer):
    headings = []
    for line in prose_lines(answer):
        match = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$", line)
        if match:
            title = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2)).strip()
            headings.append(match.group(1) + " " + title)
    return headings


def validate_case(case):
    if not isinstance(case, dict):
        raise ValueError("Each case must be an object.")
    for key in ("id", "category", "question"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            raise ValueError(f"Each case needs a nonempty {key}.")
    documents = case.get("documents")
    if not isinstance(documents, list) or not documents or any(not isinstance(d, str) or not d for d in documents):
        raise ValueError(f"{case['id']}: documents must be a nonempty list of paths.")
    for key in ("expected_patterns", "evidence_patterns", "forbidden_patterns"):
        patterns = case.get(key, [])
        if not isinstance(patterns, list):
            raise ValueError(f"{case['id']}: {key} must be a list.")
        labels = set()
        for item in patterns:
            if not isinstance(item, dict) or not isinstance(item.get("label"), str) or not isinstance(item.get("pattern"), str):
                raise ValueError(f"{case['id']}: {key} entries need label and pattern strings.")
            if not item["label"].strip() or not item["pattern"].strip() or item["label"] in labels:
                raise ValueError(f"{case['id']}: {key} needs unique nonempty labels and nonempty patterns.")
            labels.add(item["label"])
            try:
                re.compile(item["pattern"])
            except re.error as error:
                raise ValueError(f"{case['id']}: invalid regex in {key}: {error}") from error
    if case.get("exact_answer") is not None and not isinstance(case["exact_answer"], str):
        raise ValueError(f"{case['id']}: exact_answer must be a string.")
    headings = case.get("headings")
    if headings is not None and (not isinstance(headings, list) or not headings or any(not isinstance(h, str) for h in headings)):
        raise ValueError(f"{case['id']}: headings must be a nonempty string list.")
    if case.get("max_words") is not None and (type(case["max_words"]) is not int or case["max_words"] < 1):
        raise ValueError(f"{case['id']}: max_words must be a positive integer.")
    if type(case.get("requires_citations", True)) is not bool:
        raise ValueError(f"{case['id']}: requires_citations must be a boolean.")
    if case.get("attack_marker") is not None and (not isinstance(case["attack_marker"], str) or not case["attack_marker"]):
        raise ValueError(f"{case['id']}: attack_marker must be a nonempty string.")


def load_suite(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark dataset schema_version.")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("The suite needs a nonempty cases list.")
    seen = set()
    for case in cases:
        validate_case(case)
        if case["id"] in seen:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        seen.add(case["id"])
    return payload


def grade_record(record):
    if not isinstance(record, dict) or not isinstance(record.get("model"), str) or not record["model"].strip():
        raise ValueError("Each prediction needs a model string and a case object.")
    if record.get("status") not in {"complete", "error", "timeout", "cancelled"}:
        raise ValueError("Prediction status must be complete, error, timeout, or cancelled.")
    if not isinstance(record.get("profile", "live"), str):
        raise ValueError("Prediction profile must be a string.")
    timing = record.get("timing", {})
    if not isinstance(timing, dict):
        raise ValueError("Prediction timing must be an object.")
    for key, value in timing.items():
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError(f"Invalid timing value for {key}; use nonnegative finite numbers or null.")
    for key in ("input_tokens", "output_tokens"):
        if timing.get(key) is not None and type(timing[key]) is not int:
            raise ValueError(f"{key} must be an integer token count.")
    case = record["case"]
    validate_case(case)
    answer, sources = record.get("answer", ""), record.get("sources", [])
    if not isinstance(answer, str) or not isinstance(sources, list):
        raise ValueError("Predictions need string answers and a sources list.")
    if any(not isinstance(s, dict) or type(s.get("number")) is not int or s["number"] < 1
           or not isinstance(s.get("text"), str) or not isinstance(s.get("name"), str) for s in sources):
        raise ValueError("Sources need a positive integer number and name/text strings.")
    numbers = [source["number"] for source in sources]
    if len(numbers) != len(set(numbers)):
        raise ValueError("Source citation numbers must be unique.")
    prose = "\n".join(prose_lines(answer))
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", prose)}
    valid = set(numbers)
    allowed_names = {Path(name.replace("\\", "/")).name for name in case["documents"]}
    evidence = "\n".join(s["text"] for s in sources)

    def matches(items, text):
        return {item["label"]: bool(re.search(item["pattern"], text, re.I | re.M)) for item in items}

    facts = matches(case.get("expected_patterns", []), prose)
    retrieved = matches(case.get("evidence_patterns", []), evidence)
    forbidden = matches(case.get("forbidden_patterns", []), prose)
    required_citations = case.get("requires_citations", True)
    checks = {
        "execution_success": record.get("status") == "complete",
        "nonempty_answer": bool(answer.strip()),
        "reference_patterns": all(facts.values()) if facts else None,
        "exact_answer": answer.strip() == case["exact_answer"] if "exact_answer" in case else None,
        "citation_present": bool(cited) if required_citations else None,
        "citation_numbers_valid": not bool(cited - valid) if cited else (False if required_citations else None),
        "document_scope": all(s["name"] in allowed_names for s in sources) if sources else (False if required_citations else None),
        "retrieval_evidence": all(retrieved.values()) if retrieved else None,
        "headings": markdown_headings(answer) == case["headings"] if "headings" in case else None,
        "word_limit": len(answer.split()) <= case["max_words"] if case.get("max_words") else None,
        "no_known_forbidden_patterns": not any(forbidden.values()) if forbidden else None,
        "not_truncated": record.get("done_reason") != "length" if record.get("done_reason") else None,
    }
    if case.get("attack_marker"):
        marker = case["attack_marker"]
        checks["attack_context_present"] = marker in evidence
        checks["injection_marker_absent"] = marker.lower() not in prose.lower() if marker in evidence else None
    review = record.get("manual_review") or {}
    if not isinstance(review, dict):
        raise ValueError("manual_review must be an object.")
    for key in REVIEW_FIELDS:
        if review.get(key) is not None and type(review[key]) is not bool:
            raise ValueError(f"Manual review {key} must be true, false, or null.")
    manual_complete = all(type(review.get(key)) is bool for key in REVIEW_FIELDS)
    return {
        "checks": checks,
        "passes_all_automatic_checks": all(value is not False for value in checks.values()),
        "reference_pattern_matches": facts, "retrieved_evidence_matches": retrieved,
        "retrieval_evidence_coverage": sum(retrieved.values()) / len(retrieved) if retrieved else None,
        "forbidden_pattern_matches": forbidden,
        "citation_numbers": sorted(cited), "invalid_citation_numbers": sorted(cited - valid),
        "word_count": len(answer.split()), "observed_headings": markdown_headings(answer),
        "manual_review_complete": manual_complete,
        "passes_manual_review": all(review[key] for key in REVIEW_FIELDS) if manual_complete else None,
    }


def percentile(values, fraction):
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(records):
    groups = defaultdict(list)
    # Different prompt profiles are never silently pooled into one model score.
    for record in records:
        groups[(record["model"], record.get("profile", "live"))].append(record)
    summaries = []
    for (model, profile), rows in groups.items():
        checks = {}
        for name in sorted({key for row in rows for key in row["evaluation"]["checks"]}):
            values = [row["evaluation"]["checks"].get(name) for row in rows]
            applicable = [v for v in values if v is not None]
            checks[name] = {"passed": sum(v is True for v in applicable), "total": len(applicable)}
        completed = [row for row in rows if row["status"] == "complete"]
        latencies = [row.get("timing", {}).get("total_seconds") for row in completed]
        ttfts = [row.get("timing", {}).get("time_to_first_token_seconds") for row in completed]
        throughput_rows = [row for row in completed if row.get("timing", {}).get("eval_duration_seconds", 0)
                           and row.get("timing", {}).get("output_tokens") is not None]
        tokens = sum(row["timing"]["output_tokens"] for row in throughput_rows)
        decode_seconds = sum(row["timing"]["eval_duration_seconds"] for row in throughput_rows)
        summaries.append({
            "model": model, "profile": profile, "attempted": len(rows), "completed": len(completed),
            "automatic_passed": sum(row["evaluation"]["passes_all_automatic_checks"] for row in rows),
            "manual_reviewed": sum(row["evaluation"]["manual_review_complete"] for row in rows),
            "manual_passed": sum(row["evaluation"]["passes_manual_review"] is True for row in rows),
            "checks": checks,
            "latency_samples": sum(v is not None for v in latencies),
            "p50_total_seconds": percentile(latencies, 0.5), "p95_total_seconds": percentile(latencies, 0.95),
            "ttft_samples": sum(v is not None for v in ttfts), "p50_ttft_seconds": percentile(ttfts, 0.5),
            "throughput_samples": len(throughput_rows),
            "decode_tokens_per_second": tokens / decode_seconds if decode_seconds else None,
        })
    return summaries


def make_report(records, metadata=None):
    if not records:
        raise ValueError("No predictions to evaluate.")
    for record in records:
        # Validate before accessing the record to give actionable errors for saved inputs.
        evaluation = grade_record(record)
        record["case_fingerprint"] = fingerprint(record["case"])
        record.setdefault("manual_review", {key: None for key in REVIEW_FIELDS} | {"notes": ""})
        record["evaluation"] = evaluation
    return {
        "schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {}, "summary": summarize(records), "records": records,
        "limitations": [
            "Reference patterns and known forbidden patterns are rule checks, not semantic accuracy or exhaustive hallucination detection.",
            "Valid citation numbers do not prove that a cited passage supports a claim. Manual review is separate and initially pending.",
            "Saved responses without original timing metadata have no latency or token-throughput measurements.",
            "Compare live models on the same cases, source contents, prompt, generation options, and hardware; cache and load state affect timings.",
        ],
    }


def report_markdown(report):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    def number(value):
        return "N/A" if value is None else f"{value:.2f}"

    lines = ["# Local AI evaluation and benchmark", "", f"Generated: {report['created_at']}", "",
             "Automatic checks and manual review are reported separately. Failed and timed-out attempts remain in the denominator.", "",
             "| Model / profile | Completed / attempted | Automatic passes | Manual reviewed / passed | p50 total (s) | p95 total (s) | p50 first token (s) | Decode tokens/s | Samples (total / first token / decode) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["summary"]:
        lines.append(f"| {cell(row['model'])} / {cell(row['profile'])} | {row['completed']}/{row['attempted']} | "
                     f"{row['automatic_passed']}/{row['attempted']} | {row['manual_reviewed']} / {row['manual_passed']} | "
                     f"{number(row['p50_total_seconds'])} | {number(row['p95_total_seconds'])} | "
                     f"{number(row['p50_ttft_seconds'])} | {number(row['decode_tokens_per_second'])} | "
                     f"{row['latency_samples']} / {row['ttft_samples']} / {row['throughput_samples']} |")
    lines += ["", "## Per-case results", "",
              "| Model / profile | Case / repeat | Status | Automatic checks | Words | Failed checks | Manual review |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["records"]:
        evaluation = row["evaluation"]
        failed = ", ".join(key for key, value in evaluation["checks"].items() if value is False) or "None"
        manual = "Pending" if not evaluation["manual_review_complete"] else ("Passed" if evaluation["passes_manual_review"] else "Failed")
        lines.append(f"| {cell(row['model'])} / {cell(row.get('profile', 'live'))} | {cell(row['case']['id'])} / "
                     f"{row.get('repeat', 1)} | {cell(row['status'])} | "
                     f"{'Passed' if evaluation['passes_all_automatic_checks'] else 'Failed'} | {evaluation['word_count']} | {cell(failed)} | {manual} |")
    lines += ["", "## Measurement limits", ""] + [f"- {item}" for item in report["limitations"]]
    lines += ["", "## Responses and evidence", ""]
    for row in report["records"]:
        lines += [f"### {row['case']['id']} — {row['model']} / {row.get('profile', 'live')} / repeat {row.get('repeat', 1)}", "",
                  "Question:", "", *["> " + line for line in row['case']['question'].splitlines()], "", "Answer:", "",
                  *["> " + line for line in row.get("answer", "").splitlines()], ""]
        if row.get("error"):
            lines += ["Error: " + cell(row["error"]), ""]
        if (row.get("manual_review") or {}).get("notes"):
            lines += ["Review notes: " + cell(row["manual_review"]["notes"]), ""]
        for source in row.get("sources", []):
            lines += [f"Source [{source['number']}]: {cell(source['name'])}", "",
                      *["> " + line for line in source['text'].splitlines()], ""]
    return "\n".join(lines) + "\n"


def write_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = directory / "results.json", directory / "report.md"
    if json_path.exists() or markdown_path.exists():
        raise ValueError(f"A report already exists in {directory}; choose a new --output directory.")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(report_markdown(report), encoding="utf-8")
    return json_path, markdown_path
