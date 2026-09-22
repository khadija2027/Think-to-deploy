# HR document assistant

An authentication-free FastAPI chat application with an Airflow document pipeline.
Employees open the chat directly; there are no LDAP services, accounts, JWTs, or
session cookies in the application. Airflow retains its own operator login.

## Run with Docker

1. Start Docker Desktop with Linux containers.
2. Put your HR documents in `T2D_Chatbot_for_HR/dataset/` (subfolders are supported).
   Supported formats: PDF, DOCX, DOC, XLSX, XLS, UTF-8 TXT and Markdown.
3. Use `.env.example` as a reference to update your existing `.env`. Do not replace
   an existing API key. By default `LLM_PROVIDER=ollama` keeps generation local.
   Run Ollama on the host with `ollama pull qwen3:4b` and `ollama serve`.
   To use OpenRouter first, set `LLM_PROVIDER=auto`, `OPENROUTER_API_KEY`, and a model
   available to your account. Existing API keys are ignored in Ollama-only mode.
   On Docker Desktop, the API reaches the host through `host.docker.internal`.
4. Build and start:

   ```sh
   docker compose up -d --build
   ```

5. Open the chat at http://localhost:8000. Until ingestion completes, the page
   reports that documents are unavailable and questions receive a useful 503 error.
6. Open Airflow at http://localhost:9080 (default operator account: `admin` / `admin`).
   Trigger `safran_robust_faiss_rag_pipeline`, or use:

   ```sh
   docker compose exec airflow-scheduler airflow dags trigger safran_robust_faiss_rag_pipeline
   ```

The DAG is unpaused by default and scheduled daily. Initial builds download Python,
OCR and CPU PyTorch dependencies. The first ingestion and first question also
download the embedding model into persistent caches. Model generation requires
either a working OpenRouter account/model or a reachable Ollama instance.

## Workflow

```text
dataset/ (read-only mount)
  -> discover and fingerprint the corpus
  -> extract text and tables (page-by-page OCR for scanned PDFs)
  -> best-effort personal-information masking
  -> chunks of at most 1,000 characters, overlapping by 200
  -> normalized embeddings, batched by 64
  -> immutable FAISS + chunks.json + manifest.json snapshot
  -> atomically publish current.json on the shared rag-index volume

Browser -> POST /api/ask {"question": "..."}
  -> load current snapshot and its declared embedding model
  -> encode question -> retrieve up to five passages
  -> context with source filenames -> Ollama (or OpenRouter with Ollama fallback in auto mode)
  -> answer + retrieved sources -> browser
```

The API uses the model recorded by the pipeline, avoiding embedding mismatches.
The default is `paraphrase-multilingual-MiniLM-L12-v2`. Changing `EMBEDDING_MODEL`
requires recreating the Airflow containers and running ingestion again. The API
reloads a new snapshot on the next question without a restart.

Unchanged corpora skip processing. Changed or deleted documents trigger a full
rebuild from the remaining files. Removing all documents publishes an empty
snapshot so deleted content cannot continue to appear in answers. A parsing or
indexing failure leaves the last successful index available; input files are never
moved or deleted. The corpus limit is 5,000 chunks and each source file is limited
to 50 MB. Exceeding a limit fails the run instead of silently omitting content.

Airflow passes only working-directory paths through XCom. Intermediate JSON files
and reports live on `pipeline-data`; vectors and matching chunks live on `rag-index`.
PostgreSQL holds Airflow metadata, not chat history. Old index snapshots and working
files are retained for debugging; this demo does not automatically prune them.

PII masking uses regular expressions and is not complete anonymization. Source
filenames remain visible in citations. Answers are generated from retrieved text;
the UI lists retrieved sources, which are not a claim that every answer is correct.
Conversations and quality metrics are not persisted or fabricated.

## Configuration and endpoints

See `.env.example` for provider, embedding-model and Airflow operator settings.
Docker binds the web interfaces to localhost. The API is intentionally open to
anyone who can reach its port.

| Endpoint | Purpose |
| --- | --- |
| `GET /` or `/chatbot` | Chat without login |
| `POST /api/ask` | JSON question; answer and source metadata |
| `GET /api/status` | Published index metadata |
| `GET /health` | Process liveness and index status |
| `GET /ready` | 200 when index artifacts exist and contain chunks, otherwise 503 |
| `GET /docs` | Interactive API documentation |

Readiness reports index availability; it does not make a paid LLM request or
download the embedding model. Provider errors are returned as 502, and unavailable
retrieval as 503. Actual loading checks vector dimensions and chunk alignment.

## Validation and troubleshooting

```sh
docker compose config --quiet
docker compose ps
docker compose logs --tail=100 airflow-init airflow-scheduler fastapi
docker compose exec airflow-scheduler airflow dags list-import-errors
```

The regression tests use real FAISS with deterministic embeddings, so tests do not
need an API key or model download. Use Python 3.11:

```sh
python -m venv .venv
# Windows
.venv/Scripts/python -m pip install -r requirements/test.txt
.venv/Scripts/python -m unittest discover -s tests -v
```

For an integration test inside Docker (PowerShell, from the repository root):

```powershell
docker compose run --rm --no-deps -v "${PWD}/tests:/tests:ro" airflow-scheduler python /tests/container_smoke.py
```

This test creates temporary synthetic TXT, DOCX, XLSX and scanned PDF documents,
runs the real Airflow DAG and embedding model, and checks retrieval and citations.
It uses isolated index files and a stubbed answer generator; it does not publish
sample policies into the chat's knowledge base or make paid LLM calls.

## Project layout and local commands

```text
docker/                     API and Airflow Dockerfiles
requirements/               Shared dependency pins and service-specific lists
T2D_Chatbot_for_HR/
  rag_core/                 Retrieval, generation, ingestion, web app and CLI
  frontend_chatbot/          HTML templates and static assets
  dags/                     Airflow pipeline definition
  dataset/                  Your source documents
  init_airflow.py            Airflow database and operator initialization
tests/                      Regression and container integration tests
docker-compose.yaml         Service configuration and persistent volumes
```

The API uses `rag_core.web:create_app` as its single application factory.
For local development, install the root `requirements.txt` with Python 3.11,
then run these commands from `T2D_Chatbot_for_HR/`:

```sh
python -m rag_core serve
python -m rag_core status
python -m rag_core ask "Quelle est la politique de congés ?"
```

Local commands read environment variables and default to the local `rag_index/`
directory; they do not automatically load `.env` or Docker's index volume.
To query the running Docker index, use
`docker compose exec fastapi python -m rag_core status` from the repository root.
The old standalone app and CLI wrappers were consolidated into this module.
Legacy index files and the old profile-index merger have been removed;
regenerate the index through Airflow.

Compose waits for PostgreSQL health and successful database initialization before
starting Airflow, following [Docker's dependency conditions](https://docs.docker.com/compose/how-tos/startup-order/).
The image installs the same Airflow version as its base image, following
[Airflow's image-extension guidance](https://airflow.apache.org/docs/docker-stack/build.html).
