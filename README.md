# Local / Document Chat

A private Streamlit workspace for chatting with your PDF, DOCX, and TXT files. Ollama generates embeddings and answers locally; Chroma keeps the indexed text and vectors on your disk. No cloud API key is required.

## Start on this machine

The Python environment is installed in `.venv`. Ollama runs in local Rancher Desktop. Start Rancher Desktop, then run:

```powershell
.\scripts\start-local.ps1
```

Open **http://localhost:8501**. This uses `.env`, which selects `phi3` and `nomic-embed-text`. Ollama must be running and both models must finish downloading. The sidebar shows connection and model readiness. If the Ollama container is stopped, run `docker compose up -d ollama` (the Docker launcher below detects Rancher's executable if it is missing from PATH).

## Run both services with Docker

Start Rancher Desktop or Docker Desktop with Linux containers enabled. In PowerShell, from this folder:

```powershell
.\scripts\start-docker.ps1
```

The script builds the app, starts Ollama, and downloads `nomic-embed-text` and `phi3`. Phi-3 is the smaller option from the requirements and suits this machine's approximately 8 GB Docker memory allocation. Open **http://localhost:8501**. Downloads may take several minutes; chat works offline after installation and model downloads finish.

For the larger model in the original requirements, allocate sufficient Docker memory (12–16 GB is more comfortable), then run:

```powershell
.\scripts\start-docker.ps1 -ChatModel 'llama3.1:8b'
```

If PowerShell blocks local scripts, use these commands instead (also suitable for other shells):

```text
docker compose up --build -d
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.1:8b
```

Plain Compose defaults to `llama3.1:8b` unless overridden; this checkout's local `.env` selects `phi3` for the available Docker memory. You can select another installed chat model in the sidebar. Alternatively copy `.env.example` to `.env`, change `LLM_MODEL`, then recreate the app with `docker compose up -d`.

## Use it

1. Confirm Ollama is connected. Use **Refresh connection** after downloading models.
2. Upload one or more PDF, DOCX, or UTF-8 TXT files in the sidebar, then click **Index documents**.
3. Ask a question in **Conversation**. Use **Search in** to focus on one document or the whole library.
4. Expand the retrieved excerpts below an answer to inspect the source text and PDF page numbers. Citation numbers correspond to these excerpts.
5. Use **Document library** to remove an indexed document, **New chat** to clear the current conversation, or **Download conversation** to export it.

Try `samples/handbook.txt` and ask **“What is the annual learning budget?”** The answer should be **INR 30,000** with a source reference. Ask **“What is the CEO's birthday?”** to check the model's missing-information response.

The system instructs the local model to answer only from retrieved passages and say `I cannot find this information in the uploaded document.` when unsupported. Like any generative model, it can still make mistakes; excerpts show what was retrieved, rather than guaranteeing every claim is supported. Retrieval uses the current question; phrase follow-ups with the subject included for better results.

## Native Python + Ollama

Install Python 3.11 or 3.12 and [Ollama](https://ollama.com/download). Set `OLLAMA_NO_CLOUD=1` on the **Ollama server process** and restart it to disable remote inference. See [Ollama's local-only configuration](https://docs.ollama.com/faq#how-do-i-disable-ollama-cloud-features).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.1:8b
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Ollama must be running. On a machine without the desktop service, start `ollama serve` in another terminal. For a smaller model, run `ollama pull phi3` and set `$env:LLM_MODEL = 'phi3'` before launching Streamlit.

On macOS/Linux, create a venv with `python3 -m venv .venv`, install with `.venv/bin/python -m pip install -r requirements.txt`, and launch with `.venv/bin/python -m streamlit run app.py`.

You can also ingest without the UI:

```powershell
.\.venv\Scripts\python.exe ingest.py .\samples\handbook.txt
```

## Configuration and storage

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama endpoint; Compose uses `http://ollama:11434`. |
| `LLM_MODEL` | `llama3.1:8b` | Default chat model. |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model, downloaded separately. |
| `DATA_DIR` | `<project>/data` | Persistent storage; Compose uses `/app/data`. |

`.env` is read by Docker Compose and the native Python app. Explicit environment variables take precedence over `.env` values. Restart the app after changing configuration.

The app supports local loopback, `ollama`, and `host.docker.internal` endpoints. Cloud model names are rejected. Compose disables Ollama cloud features. Streamlit usage statistics, Chroma telemetry, and LangChain tracing are disabled. Initial dependency and model downloads still use the internet. Keep the service on your own machine; it has no authentication or user separation.

Native Chroma data lives in `data/chroma_db`. Docker stores documents in the `document_data` volume and models in `ollama_models`. Both survive `docker compose down`. **`docker compose down -v` permanently deletes both volumes.** Back up the document volume while the app is stopped if you need a consistent database copy.

Only extracted text, embeddings, and filename metadata are retained; uploaded originals are parsed in memory. Same-content uploads are skipped even if renamed. Changed contents with the same filename are separate entries. Files are limited to 25 MB each. Textless scanned PDFs need OCR first, encrypted PDFs need an unlocked copy, and DOCX headers/footers and embedded images are not indexed. Chat history is session-only and is cleared when removing a document; previously exported conversations remain on your disk.

Embedding model and chunking settings identify separate collections so incompatible vectors never mix. Changing the embedding model requires re-uploading documents to the new collection. The old collection remains on disk and reappears if you switch back.

## Checks

The offline suite uses real document parsers and persistent Chroma with deterministic test embeddings; no Ollama service or downloads are required. It covers ingestion, duplicate handling, rollback, deletion, scoped retrieval, parser errors, source mapping, and Streamlit UI behavior.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Or inside the built Docker image:

```text
docker compose run --rm --no-deps app python -m unittest discover -s tests -v
```

After downloading both models, exercise real embeddings, retrieval, generation, citation, missing-information handling, and deletion with:

```text
docker compose exec app python scripts/smoke_test.py
```

## Troubleshooting

- **Docker command missing:** `scripts/start-docker.ps1` detects the standard Rancher Desktop and Docker Desktop paths. For manual commands, add the appropriate `resources/.../bin` folder to your terminal PATH.
- **Ollama offline:** check `docker compose ps` and `docker compose logs ollama`. For native runs, start Ollama and verify the endpoint.
- **Rancher Desktop resets localhost connections:** a healthy container can still be unreachable through affected Windows loopback forwarding. See the [Rancher Desktop issue](https://github.com/rancher-sandbox/rancher-desktop/issues/7720). Update/restart Rancher Desktop or use native Ollama; keep the app's local-only binding.
- **Model download fails with DNS/TLS errors:** retry the `ollama pull` command after connectivity recovers. Initial model downloads must complete before chat can work; the app shows missing models explicitly.
- **Model unavailable:** run the pull commands shown in the sidebar, then refresh. The app never silently downloads a model.
- **Out of memory / slow answers:** use Phi-3, lower the retrieved-passage count, or allocate more memory to Docker. CPU generation can take a while; GPU acceleration is not enabled by this Compose file.
- **Port already in use:** stop the conflicting service or change the left side of the relevant Compose port mapping. Keep the `127.0.0.1:` binding for local access.
- **Need logs or a restart:** use `docker compose logs --tail 100 app` or `docker compose restart app`.

## Code map

`app.py` is the Streamlit UI, `ingest.py` handles parsing and Chroma storage, `chat.py` builds grounded prompts and streams local answers, `local_models.py` checks installed models, and `config.py` holds runtime settings. This follows the modular Streamlit/Ollama/Chroma architecture in `Requirment.txt` without unnecessary cloud or sentence-transformer dependencies.
#   A I - C h a t - S y s t e m  
 
