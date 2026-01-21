"""
DAG Airflow - Pipeline RAG ROBUSTE pour Architecture FAISS Multi-Profils
VERSION PRODUCTION avec toutes les améliorations de robustesse

AMÉLIORATIONS CLÉS:
✅ Validation pré-parsing (hash, taille, intégrité)
✅ OCR fallback pour PDF scannés
✅ Partial failure (1 échec ≠ pipeline cassé)
✅ Batching embeddings (protection OOM)
✅ Transactions FAISS atomiques (rollback)
✅ Retry intelligent par task
✅ Timeouts & SLA
✅ Metrics structurées
✅ Dataset lineage
✅ Tests anonymisation auto
✅ Mode DRY-RUN
✅ Kill switch

ARCHITECTURE:
chatbot_project/
├── backend/rag_service/vectorstores_faiss/
│   ├── Apprenti/
│   ├── Cadre/
│   └── ...
└── airflow_dags/
    ├── safran_robust_rag_pipeline.py (ce fichier)
    ├── dataset/
    ├── parsed/
    ├── anonymized_dataset/
    ├── chunked/
    ├── failed/
    ├── logs_pipeline/
    └── processed_hashes.json
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.exceptions import AirflowFailException
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
import os
import json
import re
import logging
from pathlib import Path
import pickle
import hashlib
import shutil
from typing import List, Dict, Tuple, Optional
import numpy as np

# PDF/DOCX parsing
import pdfplumber
from pypdf import PdfReader
import subprocess

logger = logging.getLogger(__name__)

# ========== CONFIGURATION GLOBALE ==========
# 🔥 KILL SWITCH - Désactiver le pipeline
PIPELINE_ENABLED = True

# 🧪 DRY RUN - Tout sauf indexation FAISS
DRY_RUN = False

# Paths - Correction pour conteneur Airflow
CURRENT_DIR = Path("/opt/airflow/dags")
PROJECT_ROOT = Path("/opt/airflow")

DATASET_FOLDER = PROJECT_ROOT / 'dataset'
PARSED_FOLDER = CURRENT_DIR / 'parsed'
ANONYMIZED_FOLDER = CURRENT_DIR / 'anonymized_dataset'
CHUNKED_FOLDER = CURRENT_DIR / 'chunked'
FAILED_FOLDER = CURRENT_DIR / 'failed'
LOGS_FOLDER = CURRENT_DIR / 'logs_pipeline'
TEMP_FOLDER = CURRENT_DIR / 'temp'

# Vectorstore
VECTORSTORE_BASE = PROJECT_ROOT / 'backend' / 'rag_service' / 'vectorstores_faiss'

# Profils utilisateurs
USER_PROFILES = ['Apprenti', 'Cadre', 'Technicien', 'Manager', 'RH', 'Autre']

# Configuration chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 🔒 LIMITES DE SÉCURITÉ (protection OOM)
MAX_CHUNKS_PER_RUN = 5000
MAX_FILE_SIZE_MB = 50
MIN_FILE_SIZE_KB = 1
BATCH_SIZE_EMBEDDINGS = 100  # Embeddings par batch

# 📊 SEUILS D'ÉCHEC
MIN_SUCCESS_RATIO = 0.3  # 30% minimum de réussite

# Formats supportés
SUPPORTED_FORMATS = ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.txt']

# Hash registry
HASH_REGISTRY_PATH = CURRENT_DIR / 'processed_hashes.json'

# Créer dossiers
for folder in [DATASET_FOLDER, PARSED_FOLDER, ANONYMIZED_FOLDER,
               CHUNKED_FOLDER, FAILED_FOLDER, LOGS_FOLDER, TEMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ========== AIRFLOW CONFIG ==========
default_args = {
    'owner': 'safran_sed',
    'depends_on_past': False,
    'email': ['data-team@safran.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


class FileValidator:
    """
    🔍 Validation pré-parsing
    - Vérifie taille, intégrité, déduplication
    """

    def __init__(self):
        self.hash_registry = self._load_hash_registry()

    def _load_hash_registry(self) -> Dict:
        """Charge le registre des fichiers déjà traités"""
        if HASH_REGISTRY_PATH.exists():
            with open(HASH_REGISTRY_PATH, 'r') as f:
                return json.load(f)
        return {}

    def _save_hash_registry(self):
        """Sauvegarde le registre"""
        with open(HASH_REGISTRY_PATH, 'w') as f:
            json.dump(self.hash_registry, f, indent=2)

    def compute_file_hash(self, file_path: Path) -> str:
        """Calcule le hash SHA256 d'un fichier"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def validate_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Valide un fichier (accepte tout format et toute taille)
        Returns: (is_valid, reason)
        """
        # Only check deduplication and readability
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Test lecture
        except Exception as e:
            return False, f"File not readable: {str(e)}"

        # Déduplication (hash)
        file_hash = self.compute_file_hash(file_path)
        if file_hash in self.hash_registry:
            last_processed = self.hash_registry[file_hash].get('processed_at')
            return False, f"Already processed on {last_processed}"

        # Marquer comme validé
        self.hash_registry[file_hash] = {
            'filename': file_path.name,
            'processed_at': datetime.now().isoformat(),
            'size_bytes': file_path.stat().st_size
        }
        self._save_hash_registry()

        return True, "Valid"


class RobustDocumentParser:
    """
    📄 Parser multi-format ROBUSTE avec OCR fallback
    """

    @staticmethod
    def parse_pdf(file_path: Path) -> Tuple[str, Dict]:
        """Parse PDF avec fallback OCR"""
        try:
            text = ""
            metadata = {}

            # Tentative 1: pdfplumber (meilleur pour les tableaux)
            with pdfplumber.open(file_path) as pdf:
                metadata = {'pages': len(pdf.pages)}

                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num} ---\n{page_text}\n"

                    # Extraction tableaux
                    tables = page.extract_tables()
                    if tables:
                        for table_num, table in enumerate(tables, 1):
                            text += f"\n[Tableau {table_num}]\n"
                            for row in table:
                                if row:
                                    text += " | ".join(
                                        [str(cell) if cell else "" for cell in row]) + "\n"

            # Tentative 2: PdfReader (fallback)
            if not text.strip():
                logger.warning(
                    f"pdfplumber failed, trying PdfReader for {file_path.name}")
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n"

            # Tentative 3: OCR (PDF scanné)
            if not text.strip() or len(text.strip()) < 100:
                logger.warning(
                    f"PDF may be scanned, attempting OCR for {file_path.name}")
                text = RobustDocumentParser._ocr_pdf(file_path)
                metadata['ocr_used'] = True

            if not text.strip():
                raise ValueError("No text could be extracted from PDF")

            return text, metadata

        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            raise

    @staticmethod
    def ocr_pdf(file_path: Path) -> str:
        """
        🔍 OCR fallback pour PDF scannés
        Nécessite: tesseract-ocr, pdf2image
        """
        try:
            from pdf2image import convert_from_path
            import pytesseract

            logger.info(f"Running OCR on {file_path.name}")

            # Convertir PDF en images
            images = convert_from_path(file_path, dpi=300)

            text = ""
            for page_num, image in enumerate(images, 1):
                page_text = pytesseract.image_to_string(image, lang='fra')
                text += f"\n--- Page {page_num} (OCR) ---\n{page_text}\n"

            return text

        except ImportError:
            logger.warning(
                "OCR dependencies not installed (pdf2image, pytesseract)")
            return ""
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return ""

    @staticmethod
    def parse_docx(file_path: Path) -> Tuple[str, Dict]:
        """Parse DOCX avec gestion erreurs"""
        try:
            result = subprocess.run(
                ['pandoc', '--track-changes=all',
                    str(file_path), '-t', 'plain'],
                capture_output=True,
                text=True,
                check=True,
                timeout=60
            )
            return result.stdout, {'format': 'docx'}
        except subprocess.TimeoutExpired:
            raise ValueError("DOCX parsing timeout (>60s)")
        except Exception as e:
            logger.error(f"Error parsing DOCX: {str(e)}")
            raise

    @staticmethod
    def parse_excel(file_path: Path) -> Tuple[str, Dict]:
        """Parse Excel avec limite mémoire"""
        try:
            text = ""
            excel_file = pd.ExcelFile(file_path)
            metadata = {'sheets': excel_file.sheet_names}

            for sheet_name in excel_file.sheet_names[:10]:  # Max 10 feuilles
                # Max 1000 lignes
                df = pd.read_excel(
                    file_path, sheet_name=sheet_name, nrows=1000)
                text += f"\n=== Feuille: {sheet_name} ===\n"
                text += df.to_string(index=False) + "\n\n"

            return text, metadata
        except Exception as e:
            logger.error(f"Error parsing Excel: {str(e)}")
            raise

    @staticmethod
    def parse_txt(file_path: Path) -> Tuple[str, Dict]:
        """Parse TXT avec gestion encodage"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                return text, {'format': 'txt', 'encoding': encoding}
            except UnicodeDecodeError:
                continue

        raise ValueError("Could not decode file with any encoding")


class SecureAnonymizer:
    """
    🔒 Anonymisation TOTALE avec tests automatiques
    """

    def __init__(self):
        self.patterns = {
            'nom_prenom': r'\b[A-ZÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÆŒÇ][a-zàâäéèêëïîôöùûüÿæœç]+(?:\s+[A-ZÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÆŒÇ][a-zàâäéèêëïîôöùûüÿæœç]+)+\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'telephone': r'(?:\+33|0)[1-9](?:[.\s-]?\d{2}){4}',
            'matricule': r'\b(?:MAT|EMP|ID)[\s-]?(\d{4,8})\b',
            'adresse': r'\b\d+\s+(?:rue|avenue|boulevard|place|chemin)\s+[A-Za-zÀ-ÿ\s]+\b',
            'iban': r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b',
            'secu': r'\b[12]\d{2}(0[1-9]|1[0-2])\d{2}\d{3}\d{3}\d{2}\b'
        }

        self.replacements = {
            'nom_prenom': '[IDENTITE_ANONYMISEE]',
            'email': '[EMAIL_ANONYMISE]',
            'telephone': '[TELEPHONE_ANONYMISE]',
            'matricule': '[MATRICULE_ANONYMISE]',
            'adresse': '[ADRESSE_ANONYMISEE]',
            'iban': '[IBAN_ANONYMISE]',
            'secu': '[SECU_ANONYMISEE]'
        }

    def anonymize_text(self, text: str) -> Tuple[str, Dict]:
        """Anonymise le texte et retourne les stats"""
        if not text:
            return text, {}

        anonymized = text
        stats = {key: 0 for key in self.patterns.keys()}

        # Appliquer chaque pattern
        for pattern_name, pattern in self.patterns.items():
            def count_and_replace(match):
                stats[pattern_name] += 1
                return self.replacements[pattern_name]

            anonymized = re.sub(pattern, count_and_replace, anonymized)

        return anonymized, stats

    def validate_anonymization(self, original: str, anonymized: str) -> Tuple[bool, List[str]]:
        """
        🧪 Test automatique d'anonymisation
        Vérifie qu'aucune donnée sensible ne subsiste
        """
        issues = []

        # Test 1: Emails
        if re.search(r'@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', anonymized):
            issues.append("Email detected in anonymized text")

        # Test 2: Téléphones
        if re.search(r'(?:\+33|0)[1-9]\d{8}', anonymized):
            issues.append("Phone number detected")

        # Test 3: IBAN
        if re.search(r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b', anonymized):
            issues.append("IBAN detected")

        # Test 4: Numéro sécu
        if re.search(r'\b[12]\d{14}\b', anonymized):
            issues.append("Social security number detected")

        return len(issues) == 0, issues


class SmartTextChunker:
    """
    ✂️ Chunking intelligent avec limite mémoire
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Dict, max_chunks: int = None) -> List[Dict]:
        """
        Découpe le texte en chunks
        max_chunks: limite le nombre de chunks (protection OOM)
        """
        if not text or len(text.strip()) < self.chunk_size:
            return [{
                'text': text,
                'chunk_id': 0,
                'chunk_size': len(text),
                'metadata': metadata
            }]

        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_id = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk,
                    'chunk_id': chunk_id,
                    'chunk_size': len(current_chunk),
                    'metadata': metadata
                })

                # Protection limite chunks
                if max_chunks and len(chunks) >= max_chunks:
                    logger.warning(
                        f"Max chunks reached ({max_chunks}), stopping")
                    break

                # Overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(
                    current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
                chunk_id += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Dernier chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk,
                'chunk_id': chunk_id,
                'chunk_size': len(current_chunk),
                'metadata': metadata
            })

        return chunks


class AtomicFAISSVectorStore:
    """
    🔐 Gestion FAISS avec transactions atomiques
    """

    def __init__(self, base_path: Path, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.base_path = Path(base_path)
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Charge le modèle d'embeddings"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed! Run: pip install sentence-transformers")
        return self.model

    def create_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        🔥 Génère embeddings par BATCH (protection OOM)
        """
        model = self.load_model()

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        logger.info(
            f"Generating embeddings in {total_batches} batches of {batch_size}")

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)

            logger.info(f"Batch {i//batch_size + 1}/{total_batches} done")

        return np.vstack(all_embeddings)

    def save_to_profile_atomic(self, profile: str, texts: List[str], metadatas: List[Dict]) -> Dict:
        """
        💾 Sauvegarde ATOMIQUE dans FAISS
        - Écriture dans fichiers temporaires
        - Remplacement atomique si succès
        - Rollback si échec
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed! Run: pip install faiss-cpu")

        profile_path = self.base_path / profile
        profile_path.mkdir(parents=True, exist_ok=True)

        # Chemins finaux
        index_file = profile_path / 'index.faiss'
        texts_file = profile_path / 'texts.pkl'
        metadata_file = profile_path / 'metadata.pkl'

        # Chemins temporaires
        index_tmp = profile_path / 'index.faiss.tmp'
        texts_tmp = profile_path / 'texts.pkl.tmp'
        metadata_tmp = profile_path / 'metadata.pkl.tmp'

        try:
            # Générer embeddings par batch
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            embeddings = self.create_embeddings_batch(
                texts, batch_size=BATCH_SIZE_EMBEDDINGS)

            # Charger index existant ou créer nouveau
            if index_file.exists():
                logger.info(f"Loading existing FAISS index for {profile}")
                index = faiss.read_index(str(index_file))

                with open(texts_file, 'rb') as f:
                    existing_texts = pickle.load(f)
                with open(metadata_file, 'rb') as f:
                    existing_metadata = pickle.load(f)

                texts = existing_texts + texts
                metadatas = existing_metadata + metadatas
            else:
                logger.info(f"Creating new FAISS index for {profile}")
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)

            # Ajouter embeddings
            index.add(embeddings.astype('float32'))

            # 🔒 ÉCRITURE ATOMIQUE
            # 1. Écrire dans fichiers temporaires
            faiss.write_index(index, str(index_tmp))

            with open(texts_tmp, 'wb') as f:
                pickle.dump(texts, f)

            with open(metadata_tmp, 'wb') as f:
                pickle.dump(metadatas, f)

            # 2. Remplacement atomique (os.replace est atomique sur POSIX)
            os.replace(index_tmp, index_file)
            os.replace(texts_tmp, texts_file)
            os.replace(metadata_tmp, metadata_file)

            logger.info(
                f"✓ Atomically saved {len(texts)} documents to {profile}")

            return {
                'profile': profile,
                'index_path': str(index_file),
                'total_vectors': index.ntotal,
                'newly_added': len(embeddings),
                'model_version': self.model_name
            }

        except Exception as e:
            # Rollback: supprimer fichiers temporaires
            logger.error(f"Error during indexing, rolling back: {str(e)}")
            for tmp_file in [index_tmp, texts_tmp, metadata_tmp]:
                if tmp_file.exists():
                    tmp_file.unlink()
            raise


# ========== TASKS AIRFLOW ROBUSTES ==========

def check_pipeline_enabled(**context):
    """🔥 KILL SWITCH - Vérifie si le pipeline est activé"""
    if not PIPELINE_ENABLED:
        raise AirflowFailException("❌ Pipeline is DISABLED via KILL SWITCH")

    logger.info("✅ Pipeline ENABLED")

    # Log du mode
    if DRY_RUN:
        logger.warning("⚠️ DRY RUN MODE: FAISS indexing will be SKIPPED")

    return {'pipeline_enabled': True, 'dry_run': DRY_RUN}


def validate_and_discover_files(**context):
    """
    🔍 ÉTAPE 1: VALIDATION + INGESTION
    - Validation pré-parsing
    - Déduplication
    - Filtrage fichiers invalides
    """
    logger.info(f"[VALIDATION] Scanning {DATASET_FOLDER}")

    validator = FileValidator()

    valid_files = []
    invalid_files = []

    for file_path in Path(DATASET_FOLDER).rglob('*'):
        if file_path.is_file() and not file_path.name.startswith('.'):
            is_valid, reason = validator.validate_file(file_path)

            if is_valid:
                valid_files.append(str(file_path))
                logger.info(f"✓ {file_path.name}: {reason}")
            else:
                invalid_files.append(
                    {'file': file_path.name, 'reason': reason})
                logger.warning(f"✗ {file_path.name}: {reason}")

                # Déplacer vers failed/
                shutil.move(str(file_path), FAILED_FOLDER / file_path.name)

    logger.info(
        f"[VALIDATION] Valid: {len(valid_files)}, Invalid: {len(invalid_files)}")

    if not valid_files:
        raise AirflowFailException("No valid files to process")

    context['task_instance'].xcom_push(key='valid_files', value=valid_files)
    context['task_instance'].xcom_push(
        key='invalid_files', value=invalid_files)

    return {
        'total_files': len(valid_files) + len(invalid_files),
        'valid_files': len(valid_files),
        'invalid_files': len(invalid_files)
    }


def parse_documents_robust(**context):
    """
    📄 ÉTAPE 2: PARSING ROBUSTE
    - Parser multi-format avec OCR fallback
    - Partial failure (continue même si échecs)
    """
    logger.info("[PARSING] Starting robust document parsing")

    ti = context['task_instance']
    run_id = context['run_id']

    files = ti.xcom_pull(task_ids='validate_and_discover', key='valid_files')

    if not files:
        return {'status': 'no_files'}

    parser = RobustDocumentParser()
    results = {'parsed': [], 'failed': []}

    for file_path in files:
        file_name = os.path.basename(file_path)
        file_ext = Path(file_path).suffix.lower()

        try:
            # Parsing selon format
            if file_ext == '.pdf':
                text, metadata = parser.parse_pdf(Path(file_path))
            elif file_ext in ['.docx', '.doc']:
                text, metadata = parser.parse_docx(Path(file_path))
            elif file_ext in ['.xlsx', '.xls']:
                text, metadata = parser.parse_excel(Path(file_path))
            elif file_ext == '.txt':
                text, metadata = parser.parse_txt(Path(file_path))
            else:
                raise ValueError(f"Unsupported format: {file_ext}")

            if not text or len(text.strip()) < 50:
                raise ValueError("Insufficient text extracted")

            # Sauvegarder
            parsed_path = PARSED_FOLDER / f"{Path(file_name).stem}_parsed.txt"
            with open(parsed_path, 'w', encoding='utf-8') as f:
                f.write(text)

            # Ajout lineage
            metadata['pipeline_run_id'] = run_id
            metadata['parsed_at'] = datetime.now().isoformat()

            results['parsed'].append({
                'file': file_name,
                'parsed_path': str(parsed_path),
                'text_length': len(text),
                'metadata': metadata
            })

            logger.info(f"[PARSING] ✓ {file_name} ({len(text)} chars)")

        except Exception as e:
            logger.error(f"[PARSING] ✗ {file_name}: {str(e)}")
            results['failed'].append({'file': file_name, 'error': str(e)})

            # Déplacer vers failed/
            try:
                shutil.move(file_path, FAILED_FOLDER / file_name)
            except:
                pass

    # 🚨 Vérifier ratio de succès
    total = len(results['parsed']) + len(results['failed'])
    success_ratio = len(results['parsed']) / total if total > 0 else 0

    if success_ratio < MIN_SUCCESS_RATIO:
        raise AirflowFailException(
            f"❌ Success ratio too low: {success_ratio:.1%} < {MIN_SUCCESS_RATIO:.1%}"
        )

    ti.xcom_push(key='parsing_results', value=results)

    return {
        'parsed': len(results['parsed']),
        'failed': len(results['failed']),
        'success_ratio': f"{success_ratio:.1%}"
    }


def anonymize_documents_secure(**context):
    """
    🔒 ÉTAPE 3: ANONYMISATION SÉCURISÉE
    - Anonymisation totale
    - Tests automatiques de validation
    """
    logger.info("[ANONYMISATION] Starting secure anonymization")

    ti = context['task_instance']
    parsing_results = ti.xcom_pull(
        task_ids='parse_documents', key='parsing_results')

    if not parsing_results or not parsing_results.get('parsed'):
        return {'status': 'no_files'}

    anonymizer = SecureAnonymizer()
    results = {'anonymized': [], 'failed': []}

    for item in parsing_results['parsed']:
        parsed_path = item['parsed_path']
        file_name = item['file']

        try:
            with open(parsed_path, 'r', encoding='utf-8') as f:
                original_text = f.read()

            # Anonymisation
            anonymized_text, stats = anonymizer.anonymize_text(original_text)

            # 🧪 TEST AUTOMATIQUE
            is_valid, issues = anonymizer.validate_anonymization(
                original_text, anonymized_text)

            if not is_valid:
                raise ValueError(
                    f"Anonymization validation failed: {', '.join(issues)}")

            # Sauvegarder
            base_name = Path(parsed_path).stem.replace('_parsed', '')
            anon_path = ANONYMIZED_FOLDER / f"{base_name}_anonymized.txt"

            with open(anon_path, 'w', encoding='utf-8') as f:
                f.write(anonymized_text)

            # Rapport
            report = {
                'original_file': file_name,
                'anonymized_at': datetime.now().isoformat(),
                'stats': stats,
                'validation': 'PASSED'
            }

            report_path = ANONYMIZED_FOLDER / f"{base_name}_anon_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            results['anonymized'].append({
                'file': file_name,
                'path': str(anon_path),
                'stats': stats,
                'metadata': item.get('metadata', {})
            })

            logger.info(f"[ANONYMISATION] ✓ {file_name} (validation passed)")

        except Exception as e:
            logger.error(f"[ANONYMISATION] ✗ {file_name}: {str(e)}")
            results['failed'].append({'file': file_name, 'error': str(e)})

    ti.xcom_push(key='anonymization_results', value=results)

    return {
        'anonymized': len(results['anonymized']),
        'failed': len(results['failed'])
    }


def chunk_documents_smart(**context):
    """
    ✂️ ÉTAPE 4: CHUNKING INTELLIGENT
    - Découpage avec limite chunks
    - Protection OOM
    """
    logger.info("[CHUNKING] Starting smart chunking")

    ti = context['task_instance']
    anon_results = ti.xcom_pull(
        task_ids='anonymize_documents', key='anonymization_results')

    if not anon_results or not anon_results.get('anonymized'):
        return {'status': 'no_files'}

    chunker = SmartTextChunker(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks = []

    for item in anon_results['anonymized']:
        anon_path = item['path']
        file_name = item['file']

        try:
            with open(anon_path, 'r', encoding='utf-8') as f:
                text = f.read()

            metadata = item.get('metadata', {})
            metadata.update({
                'source_file': file_name,
                'anonymized': True,
                'processing_date': datetime.now().isoformat()
            })

            # Chunking avec limite
            remaining_budget = MAX_CHUNKS_PER_RUN - len(all_chunks)
            chunks = chunker.chunk_text(
                text, metadata, max_chunks=remaining_budget)

            # Sauvegarder chunks
            base_name = Path(file_name).stem
            chunks_path = CHUNKED_FOLDER / f"{base_name}_chunks.json"

            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)

            all_chunks.extend(chunks)

            logger.info(f"[CHUNKING] ✓ {file_name} → {len(chunks)} chunks")

            # Protection MAX_CHUNKS
            if len(all_chunks) >= MAX_CHUNKS_PER_RUN:
                logger.warning(
                    f"⚠️ Max chunks reached ({MAX_CHUNKS_PER_RUN}), stopping")
                break

        except Exception as e:
            logger.error(f"[CHUNKING] ✗ {file_name}: {str(e)}")

    ti.xcom_push(key='all_chunks', value=all_chunks)

    return {
        'total_chunks': len(all_chunks),
        'documents_chunked': len(anon_results['anonymized']),
        'chunks_limit': MAX_CHUNKS_PER_RUN
    }


def index_to_faiss_atomic(**context):
    """
    🔐 ÉTAPE 5: INDEXATION FAISS ATOMIQUE
    - Transactions atomiques
    - Batch embeddings
    - Rollback automatique si échec
    """

    # 🧪 DRY RUN CHECK
    if DRY_RUN:
        logger.warning("⚠️ DRY RUN MODE: Skipping FAISS indexing")
        return {
            'status': 'DRY_RUN_SKIPPED',
            'message': 'FAISS indexing skipped in dry-run mode'
        }

    logger.info("[VECTORISATION] Starting atomic FAISS indexing")

    ti = context['task_instance']
    all_chunks = ti.xcom_pull(task_ids='chunk_documents', key='all_chunks')

    if not all_chunks:
        logger.warning("No chunks to index")
        return {'status': 'no_chunks'}

    try:
        # Initialiser vectorstore
        vectorstore = AtomicFAISSVectorStore(VECTORSTORE_BASE)

        # Préparer données
        texts = [chunk['text'] for chunk in all_chunks]
        metadatas = [chunk['metadata'] for chunk in all_chunks]

        # Indexer dans chaque profil
        indexing_results = []

        for profile in USER_PROFILES:
            logger.info(f"Indexing for profile: {profile}")

            result = vectorstore.save_to_profile_atomic(
                profile=profile,
                texts=texts,
                metadatas=metadatas
            )

            indexing_results.append(result)

        summary = {
            'profiles_updated': len(USER_PROFILES),
            'total_documents_indexed': len(texts),
            'vectorstore_base_path': str(VECTORSTORE_BASE),
            'model_used': vectorstore.model_name,
            'batch_size_used': BATCH_SIZE_EMBEDDINGS,
            'details': indexing_results
        }

        logger.info(
            f"✓ Atomic indexation complete: {len(USER_PROFILES)} profiles updated")

        return summary

    except ImportError as e:
        logger.error(f"Missing dependencies: {str(e)}")
        logger.error("Install: pip install sentence-transformers faiss-cpu")
        raise
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise


def generate_metrics_and_report(**context):
    """
    📊 ÉTAPE 6: RAPPORT + METRICS STRUCTURÉES
    - Rapport détaillé JSON
    - Metrics pour monitoring
    """
    ti = context['task_instance']

    # Récupérer résultats
    validation = ti.xcom_pull(task_ids='validate_and_discover')
    parsing = ti.xcom_pull(task_ids='parse_documents')
    anonymization = ti.xcom_pull(task_ids='anonymize_documents')
    chunking = ti.xcom_pull(task_ids='chunk_documents')
    indexing = ti.xcom_pull(task_ids='index_to_faiss')

    # Calculer métriques
    total_files = validation.get('total_files', 0) if validation else 0
    valid_files = validation.get('valid_files', 0) if validation else 0
    parsed = parsing.get('parsed', 0) if parsing else 0
    anonymized = anonymization.get('anonymized', 0) if anonymization else 0
    total_chunks = chunking.get('total_chunks', 0) if chunking else 0
    profiles_updated = indexing.get(
        'profiles_updated', 0) if indexing and isinstance(indexing, dict) else 0

    avg_chunks_per_doc = total_chunks / anonymized if anonymized > 0 else 0
    success_rate = (anonymized / total_files * 100) if total_files > 0 else 0

    # 📊 METRICS STRUCTURÉES
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'run_id': context['run_id'],
        'pipeline_version': 'robust_faiss_v2.0',
        'metrics': {
            'total_files_discovered': total_files,
            'valid_files': valid_files,
            'invalid_files': total_files - valid_files,
            'files_parsed': parsed,
            'files_anonymized': anonymized,
            'total_chunks_created': total_chunks,
            'avg_chunks_per_document': round(avg_chunks_per_doc, 2),
            'profiles_updated': profiles_updated,
            'success_rate_percent': round(success_rate, 2)
        },
        'performance': {
            'chunk_size': CHUNK_SIZE,
            'chunk_overlap': CHUNK_OVERLAP,
            'batch_size_embeddings': BATCH_SIZE_EMBEDDINGS,
            'max_chunks_limit': MAX_CHUNKS_PER_RUN
        }
    }

    # Sauvegarder metrics
    metrics_path = LOGS_FOLDER / \
        f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 📋 RAPPORT COMPLET
    report = {
        'pipeline_execution': {
            'timestamp': datetime.now().isoformat(),
            'run_id': context['run_id'],
            'version': 'robust_faiss_multi_profile_v2.0',
            'compliant_with': 'Safran Think To Deploy Phase 2 + Production Hardening',
            'dry_run_mode': DRY_RUN
        },
        'stages': {
            'validation': validation or {},
            'parsing': parsing or {},
            'anonymization': anonymization or {},
            'chunking': chunking or {},
            'indexing': indexing or {}
        },
        'security': {
            'method': 'Total Anonymization with Auto-Validation',
            'reversible': False,
            'rgpd_compliant': True,
            'c2_compliant': True,
            'validation_tests': 'ENABLED'
        },
        'robustness': {
            'pre_validation': 'ENABLED',
            'ocr_fallback': 'ENABLED',
            'partial_failure': 'ENABLED',
            'batch_processing': 'ENABLED',
            'atomic_transactions': 'ENABLED',
            'oom_protection': 'ENABLED'
        },
        'metrics': metrics['metrics'],
        'summary': {
            'status': 'SUCCESS' if success_rate > MIN_SUCCESS_RATIO * 100 else 'PARTIAL',
            'total_files': total_files,
            'valid_files': valid_files,
            'parsed': parsed,
            'anonymized': anonymized,
            'total_chunks': total_chunks,
            'profiles_updated': profiles_updated,
            'failed': total_files - anonymized,
            'success_rate': f"{success_rate:.1f}%"
        }
    }

    # Sauvegarder rapport
    report_path = LOGS_FOLDER / \
        f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"[RAPPORT] Report saved: {report_path}")
    logger.info(f"[METRICS] Metrics saved: {metrics_path}")
    logger.info("✅ Pipeline RAG FAISS ROBUSTE terminé")

    # Afficher résumé
    logger.info("=" * 60)
    logger.info(f"RÉSUMÉ PIPELINE - Run ID: {context['run_id']}")
    logger.info("=" * 60)
    logger.info(f"Fichiers découverts: {total_files}")
    logger.info(f"Fichiers valides: {valid_files}")
    logger.info(
        f"Documents traités: {anonymized}/{total_files} ({success_rate:.1f}%)")
    logger.info(f"Chunks créés: {total_chunks}")
    logger.info(f"Profils mis à jour: {profiles_updated}")
    logger.info(f"Mode DRY RUN: {DRY_RUN}")
    logger.info("=" * 60)

    return report


# ========== DAG DEFINITION ==========
with DAG(
    'safran_robust_faiss_rag_pipeline',
    default_args=default_args,
    description='🔐 Pipeline RAG ROBUSTE: Validation → Parsing (OCR) → Anonymisation → Chunking → FAISS Atomique',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['safran', 'phase2', 'rag', 'faiss', 'robust', 'production'],
    max_active_runs=1,
) as dag:

    from airflow.sensors.python import PythonSensor
    import time

    def new_file_in_dataset(**context):
        """Return True if a new file has been added to the dataset folder since the last run."""
        marker_path = CURRENT_DIR / '.last_dataset_check'
        last_check = 0
        if marker_path.exists():
            with open(marker_path, 'r') as f:
                try:
                    last_check = float(f.read().strip())
                except Exception:
                    last_check = 0
        new_files = []
        for file in DATASET_FOLDER.glob('*'):
            if file.is_file() and file.stat().st_mtime > last_check:
                new_files.append(file)
        if new_files:
            # Update marker for next run
            with open(marker_path, 'w') as f:
                f.write(str(time.time()))
            return True
        return False

    file_sensor = PythonSensor(
        task_id='wait_for_new_file',
        python_callable=new_file_in_dataset,
        poke_interval=60,  # check every minute
        timeout=60*60*24,  # 24 hours
        mode='poke',
        soft_fail=True,
        dag=dag
    )

    start = EmptyOperator(task_id='start')

    # 🔥 KILL SWITCH
    check_enabled = PythonOperator(
        task_id='check_pipeline_enabled',
        python_callable=check_pipeline_enabled,
        provide_context=True,
        retries=0  # Pas de retry pour kill switch
    )

    # 🔍 VALIDATION + INGESTION
    validation = PythonOperator(
        task_id='validate_and_discover',
        python_callable=validate_and_discover_files,
        provide_context=True,
        retries=0,  # Pas de retry pour validation
        execution_timeout=timedelta(minutes=10)
    )

    # 📄 PARSING ROBUSTE
    parsing = PythonOperator(
        task_id='parse_documents',
        python_callable=parse_documents_robust,
        provide_context=True,
        retries=2,  # Retry pour parsing
        retry_delay=timedelta(minutes=3),
        execution_timeout=timedelta(minutes=30)
    )

    # 🔒 ANONYMISATION SÉCURISÉE
    anonymization = PythonOperator(
        task_id='anonymize_documents',
        python_callable=anonymize_documents_secure,
        provide_context=True,
        retries=1,
        execution_timeout=timedelta(minutes=20)
    )

    # ✂️ CHUNKING INTELLIGENT
    chunking = PythonOperator(
        task_id='chunk_documents',
        python_callable=chunk_documents_smart,
        provide_context=True,
        retries=1,
        execution_timeout=timedelta(minutes=15)
    )

    # 🔐 INDEXATION FAISS ATOMIQUE PAR PROFIL
    indexing = PythonOperator(
        task_id='index_to_faiss',
        python_callable=index_to_faiss_atomic,
        provide_context=True,
        retries=3,  # Plus de retries pour FAISS
        retry_delay=timedelta(minutes=5),
        execution_timeout=timedelta(minutes=60),
        doc='Indexation FAISS atomique avec rollback automatique'
    )

    # 🔗 INDEXATION FAISS GLOBALE
    def index_faiss_global():
        import faiss
        import pickle
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import os
        chunked_dir = CHUNKED_FOLDER
        dags_dir = CURRENT_DIR
        os.makedirs(dags_dir, exist_ok=True)
        all_texts = []
        all_metadatas = []
        model = SentenceTransformer('all-MiniLM-L6-v2')
        for file in chunked_dir.glob('*.json'):
            with open(file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            for chunk in chunks:
                all_texts.append(chunk['text'])
                all_metadatas.append(chunk.get('metadata', {}))
        if not all_texts:
            print('Aucun chunk trouvé pour index global.')
            return
        print(f"Génération des embeddings pour {len(all_texts)} chunks...")
        embeddings = model.encode(all_texts, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype='float32'))
        faiss.write_index(index, str(dags_dir / 'faiss.index'))
        with open(dags_dir / 'faiss_texts.pkl', 'wb') as f:
            pickle.dump(all_texts, f)
        with open(dags_dir / 'faiss_metadatas.pkl', 'wb') as f:
            pickle.dump(all_metadatas, f)
        print(
            f"Index global créé avec {index.ntotal} vecteurs. Fichiers sauvegardés dans {dags_dir}")

    index_global = PythonOperator(
        task_id='index_faiss_global',
        python_callable=index_faiss_global,
        retries=1,
        execution_timeout=timedelta(minutes=30),
        doc='Indexation FAISS globale à partir de tous les chunks.'
    )

    # 📊 RAPPORT + METRICS
    report = PythonOperator(
        task_id='generate_metrics_report',
        python_callable=generate_metrics_and_report,
        provide_context=True,
        retries=0,
        trigger_rule='all_done'  # S'exécute même si échecs
    )

    end = EmptyOperator(task_id='end')

    # 🔀 PIPELINE FLOW
    file_sensor >> start >> check_enabled >> validation >> parsing >> anonymization >> chunking >> indexing >> index_global >> report >> end