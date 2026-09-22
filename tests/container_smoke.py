"""Run inside the Airflow image; isolates generated fixtures from real documents.

Exercises the real DAG, parsers/OCR, embeddings, FAISS and retrieval. Generation is
stubbed to avoid provider costs and make assertions deterministic.
"""
import os
import tempfile
from pathlib import Path

import pendulum
from airflow.models import DagBag
from docx import Document
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

from rag_core.pipeline import discover, parse_document
from rag_core.service import RAGService


with tempfile.TemporaryDirectory(prefix="rag-smoke-", dir="/opt/airflow/pipeline") as directory:
    root = Path(directory)
    dataset = root / "dataset"
    dataset.mkdir()
    os.environ.update(DATASET_DIR=str(dataset), PIPELINE_WORK_DIR=str(root / "work"),
                      RAG_INDEX_DIR=str(root / "index"))
    (dataset / "leave.txt").write_text(
        "TEST DOCUMENT ONLY. Annual leave requests must be submitted through the HR portal. "
        "The employee's manager approves the request. Contact test@example.com.", encoding="utf-8")
    document = Document()
    document.add_paragraph("TEST DOCUMENT ONLY. Salary statements are available in the payroll portal.")
    document.save(dataset / "salary.docx")
    pd.DataFrame({"Topic": ["Training"], "Policy": ["Training requests require manager approval."]}).to_excel(
        dataset / "training.xlsx", index=False)
    # A PDF containing only a raster image must use the OCR fallback.
    image = Image.new("RGB", (1600, 400), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=36)
    draw.text((50, 80), "TEST DOCUMENT. Safety training is mandatory.", fill="black", font=font)
    image.save(dataset / "scanned.pdf", resolution=150)
    assert "training" in parse_document(dataset / "scanned.pdf").lower(), "OCR did not extract the scanned text"

    bag = DagBag(dag_folder="/opt/airflow/dags", include_examples=False)
    assert not bag.import_errors, bag.import_errors
    dag = bag.get_dag("safran_robust_faiss_rag_pipeline")
    result = dag.test(execution_date=pendulum.now("UTC"))
    assert str(result.state) == "success", result.state

    prompts = []
    service = RAGService(generator=lambda prompt: prompts.append(prompt) or "Use the HR portal.")
    answer = service.ask("How do I request annual leave?")
    assert answer["sources"][0]["document"] == "leave.txt", answer
    assert "[Source: leave.txt]" in prompts[0]
    assert "test@example.com" not in prompts[0]
    assert discover("unchanged-smoke") is None
    print("SMOKE PASS: DAG, TXT/DOCX/XLSX/scanned PDF, real embeddings, FAISS, citations, idempotency")
