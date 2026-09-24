# Ragas evaluation

The evaluator runs the actual chatbot HTTP endpoint with `test_dataset.json`.
It reconstructs the exact ordered prompt passages from the returned source/chunk
identifiers and immutable index version. References are never sent to the chatbot.
Groq is used only as the evaluation judge; the chatbot keeps its own model.
The judge receives questions, reference answers, chatbot answers and retrieved text.

If judge authentication is unavailable, collect locally first:

```powershell
docker compose -f ../docker-compose.yaml -f evaluation/compose.yaml run --no-deps rag-evaluation --collect-only
```

After collection finishes, correct `GROQ_API_KEY` in `evaluation/.env.groq`
and run the normal command again. It reuses the saved answers. Create a new
container with `docker compose run` after changing credentials; `docker start`
on an old container retains its old environment. Only one process can write a
given output directory at a time.

From `T2D_Chatbot_for_HR`, with the normal stack already running:

```powershell
docker compose -f ../docker-compose.yaml -f evaluation/compose.yaml build rag-evaluation
docker compose -f ../docker-compose.yaml -f evaluation/compose.yaml run --no-deps rag-evaluation --output /evaluation/results/groq
```

The Compose configuration reads `GROQ_API_KEY` and `GROQ_MODEL` from the ignored
`evaluation/.env.groq` file. The selected judge is `openai/gpt-oss-120b`, hosted
by Groq at `https://api.groq.com/openai/v1`. Never place keys in the dataset.
Provider account rate limits and pricing apply.
The runner spaces judge requests by 20 seconds, retries transient failures, and
uses separate requests for Ragas's multiple relevancy samples because Groq accepts
only one completion per request. A full evaluation can take several hours.
Collection-only resumes ignore judge settings because collection makes no judge calls.

To score the existing baseline collection while it continues running:

```powershell
docker compose -f ../docker-compose.yaml -f evaluation/compose.yaml run --no-deps rag-evaluation --answers-from /evaluation/results/baseline --output /evaluation/results/groq
```

This waits for each saved answer and does not regenerate it. Collection and scoring
must use separate output directories. Stop the scoring container if collection
is abandoned, since it waits for missing records.

For a small independent smoke test:

```powershell
docker compose -f ../docker-compose.yaml -f evaluation/compose.yaml run --no-deps rag-evaluation --limit 2 --output /evaluation/results/smoke
```

For the 90-question dataset, run nine batches of ten using the same judge and
the existing output directory. After quota is available, run:

```powershell
docker compose -f ../docker-compose.yaml -f evaluation/compose.yaml run --rm --no-deps rag-evaluation --answers-from /evaluation/results/baseline --output /evaluation/results/groq --next-batch
```

Each invocation runs only the first unfinished batch and then exits. Repeat after
checking quota; do not launch nine jobs in parallel. Use `--batch 1` through
`--batch 9` instead of `--next-batch` to select a specific batch. Existing successful
metrics (including zero scores) are reused. `scores.csv` and `summary.json` retain
the full-dataset report; `batches.json` lists membership and scored counts.
An exhausted rate limit stops scoring after SDK retries, preserving checkpoints.
Batching does not increase quota. Judge model, prompts and generation settings
are unchanged.

Groq outputs in `results/groq/` (original collection remains in `results/baseline/`):

- `metadata.json`: dataset hash, index version, judge and embedding model.
- `records/*.json`: references, actual responses, exact retrieved passages, latency,
  individual metric scores, failures and skipped metrics. Saved incrementally.
- `ragas_inputs.jsonl`: successful question/context/response/reference records.
- `scores.csv`: scores per question; blank values are not zero scores.
- `summary.json`: means, valid sample counts, failures and source-availability groups.

Rerunning resumes saved answers and successful metric scores. Failed metrics are
retried; generation failures remain recorded so they are visible in the baseline.
Use a new output directory for a new chatbot configuration or a changed dataset,
index or judge. Do not change the chatbot while a baseline is running.

Answerable questions use Ragas context precision, context recall, faithfulness,
response relevancy and factual correctness (F1). Response relevancy uses the
existing multilingual embedding model locally, without another embedding API.
Unanswerable questions use a separate Ragas AspectCritic for appropriate abstention;
they are excluded from answerable-question metric averages.

The dataset initially contained 30 questions referencing
`avantages_sociaux_safran_modele.pptx`, absent from the published index. These are
flagged and included, with separate grouped means. Check the current report for
the actual missing-source list. Do not interpret missing-source failures as solely
a generation problem. Reference answers still require human verification.

Judge scores are estimates, not proof of correctness. Review individual failures
and compare runs with the same judge and fixed dataset. A completed run may contain
generation or metric errors: always inspect the error counts and scored denominators.

Runner checks (no judge calls):

```powershell
docker compose -f ../docker-compose.yaml -f evaluation/compose.yaml run --rm --no-deps --entrypoint python rag-evaluation -m unittest test_runner
```

Ragas documentation: https://docs.ragas.io/en/v0.3.7/
