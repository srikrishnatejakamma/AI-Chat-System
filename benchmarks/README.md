# AI evaluation and benchmarking

The suite evaluates the local RAG backend with fixed fixtures and saves every answer, retrieved excerpt, automatic check, and available timing measurement. It also scores saved answers without running a model. No GPU, training libraries, cloud judge, new Python dependencies, or model downloads are required.

## Quick start

Run these commands from the repository root. Validate the eight cases and their three source fixtures without calling Ollama:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --validate
```

Score the two previously saved real-model report trials without new inference:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --saved-demo artifacts/report-demo/result.json artifacts/report-demo/strict/result.json
```

Those files are local artifacts from the earlier experiment and are not included in a fresh checkout. This mode keeps the original questions, applies a word limit only if the saved request specified one, and keeps the two prompt profiles separate. Old timings were not recorded, so their speed measurements are **N/A**, rather than zero.

With local Ollama and the configured models running, try one short live case:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --live --case budget --max-tokens 64 --timeout 120
```

The token cap is an experiment override, recorded in the results. Reaching the cap or timing out fails the corresponding automatic checks. CPU inference may take several minutes; use a larger timeout for longer answers. The benchmark never starts containers or downloads missing models automatically.

Run the full suite using the app's generation defaults and three repetitions:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --live --repeat 3 --timeout 600
```

Compare two **already-installed** local models on the same cases with a common context size. Replace `report-phi3` with your installed model name:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --live --models phi3 report-phi3 --context-size 4096 --repeat 3 --timeout 600
```

Use `--case` repeatedly to select cases, `--suite` for another case file, and `--output` for a new results directory. For example, `--case budget --case unsupported`. Model aliases such as `phi3` and `phi3:latest` cannot be included as separate models in the same comparison.

## What is evaluated

| Case | Checks |
| --- | --- |
| `budget` | Expected budget pattern, retrieved evidence, citation presence and valid numbers |
| `approval` | Eligible purchase categories, manager approval, known invented eligibility wording |
| `remote-work` | Three-day allowance and Friday scheduling |
| `leave` | Annual allowance and carry-forward amount |
| `unsupported` | Exact missing-information answer for the CEO's birthday |
| `report` | Reference patterns, exactly three `##` headings, fewer than 150 words, known unsupported restrictions |
| `document-scope` | Contractor budget of INR 8,000 while the conflicting employee policy is also indexed |
| `prompt-injection` | Correct budget without repeating an attack marker; the attack must actually appear in retrieved context |

The live runner indexes the entire fixture corpus in a newly created temporary database, including when only one case is selected. Each question retrieves from that case's selected documents. It uses the app's `DocumentLibrary`, `build_messages`, and `chat_model` factory with empty conversation history and three retrieved passages. It does not touch existing documents or saved chats. It measures backend behavior; it does not benchmark React rendering, HTTP overhead, chat persistence, Stop/Retry controls, or the upload endpoint.

These are a small development suite, informed partly by failures already observed in the sample handbook trials. They are not an independent or comprehensive measure of general model capability. Keep additional unseen cases outside training and prompt development when evaluating a fine-tuned model.

## Quality scores and their limits

- **Reference-pattern checks** detect the configured expected strings or regular expressions. They do not establish that the answer is factually correct: negation, contradictions, or additional invented claims can still pass them.
- **Retrieval evidence coverage** is the fraction of expected evidence patterns found in retrieved passages. It is not relevance precision over all retrieved text or a general document-recall score.
- **Citation validity** checks that numbered references exist in the retrieved source list. It does not establish that those passages support the cited claims.
- **Report structure** checks exact heading level, text, and order, excluding fenced code blocks. Word count uses whitespace-separated words, including headings. “Under 150 words” means at most 149.
- **Known forbidden patterns** catch specific previously observed errors. They do not detect every unsupported claim, and wording quoted in a negation can still trigger a rule.
- **Automatic case passes** require successful execution and every applicable automatic check to pass. Errors, timeouts, missing evidence, missing attack exposure, and reported token truncation are not counted as successes. Non-applicable checks are `null`.

Manual review is separate and initially pending. To review an answer against its excerpts, edit its `manual_review` object in `results.json`:

```json
{
  "factual_correctness": false,
  "citation_support": false,
  "relevance": true,
  "notes": "The answer adds an eligibility restriction absent from the cited excerpt."
}
```

Then recalculate the report without new inference:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --rescore artifacts/benchmarks/YOUR_RUN/results.json --output artifacts/benchmarks/reviewed
```

Rescoring preserves the saved questions, evaluation criteria, answers, evidence, timings, and manual judgments. It recomputes automatic results and summaries. A complete manual review needs all three boolean judgments. Automatic passes remain separate from manual passes.

## Timing measurements

| Measurement | Definition |
| --- | --- |
| Total latency | Wall-clock retrieval plus generation for one case; excludes fixture ingestion |
| Time to first token | Time from starting retrieval to receiving the first nonempty answer chunk |
| Retrieval time | Document lookup and query embedding time |
| Model load and prompt time | Ollama's reported durations, when available |
| Decode throughput | Total generated tokens divided by total decoding duration across completed measured cases |
| p50 / p95 latency | Linearly interpolated percentiles of completed-case latency, with sample counts shown |

Token throughput uses Ollama's `eval_count` and `eval_duration`, rather than word or character estimates. See the [Ollama chat response documentation](https://docs.ollama.com/api/chat). Missing token metadata, first-token measurements, or saved-run timings remain unavailable. Failed attempts stay in the quality denominator; latency aggregates use completed attempts, and raw failed-attempt timings are retained in JSON.

The runner does not force a cold start, unload models, or add hidden warm-up requests. Cache state, model loading, concurrent app use, and shared Docker resources affect the measurements. Run comparisons while the app is idle, inspect model-load durations, and collect enough repetitions before interpreting percentiles. A single sample is a smoke measurement. Python host information describes the caller, which may differ from Ollama's Docker/WSL execution environment; GPU utilization and peak memory are not measured.

Seeds start at 42 by default and increment across repetitions; use `--seed` to choose another base. Results record the selected cases, model digests, Ollama and package versions, pipeline and source fingerprints, embedding/chunking settings, and requested generation options. For a fair comparison, use matching fixtures, prompts, criteria, token/context limits, and execution conditions. The default context differs between some app models; `--context-size` provides a shared override.

## Output and automation

Results default to a unique UTC timestamp directory under ignored `artifacts/benchmarks/`:

- `results.json`: raw predictions, evidence, metadata, metrics, and editable manual-review fields.
- `report.md`: comparison tables, failures, responses, excerpts, and measurement limits.
- `progress.json`: live-run checkpoint after each finished attempt, useful if a later case is interrupted. This has the same report schema and can also be rescored.

The runner refuses to overwrite an existing report. Generation timeouts cancel the model stream and preserve any partial answer as a failed record. Retrieval uses the existing library's network timeout; an already-running retrieval may finish before process shutdown. An interrupt during a case preserves the prior completed-attempt checkpoint.

Exit code 0 means grading/report generation completed, even if quality checks failed. Add `--require-pass` for a CI gate: automatic failures then return 1 after saving the report. Invalid configuration or unavailable models return 2; keyboard interruption returns 130. The gate checks automatic rules, not pending manual review.

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --live --case budget --case unsupported --require-pass
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_benchmarks.py -v
```

Custom suites follow `cases.json` and its `schema_version`. Source paths resolve relative to the repository root, remain inside the checkout, and must have distinct filenames. Keep confidential fixtures in an ignored local directory; results contain their retrieved excerpts. Custom regexes and cases should be reviewed before use.
