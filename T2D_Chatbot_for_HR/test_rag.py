import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import time

# ======================================
# LOAD ENV
# ======================================
current_dir = Path.cwd()
env_file = current_dir / ".env"

if env_file.exists():
    print(f"📝 .env trouvé dans: {env_file}")
    load_dotenv(env_file)
else:
    parent_env = current_dir.parent / ".env"
    if parent_env.exists():
        print(f"📝 .env trouvé dans: {parent_env}")
        load_dotenv(parent_env)
    else:
        print(f"⚠️ Aucun .env trouvé.")

# ======================================
# CONFIGURATION
# ======================================
SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'analyse de documents RH de Safran.

RÈGLES STRICTES :
1. Tu dois UNIQUEMENT utiliser les informations présentes dans le CONTEXTE fourni ci-dessous
2. Si l'information n'est PAS dans le contexte, tu dois répondre : \"Je ne trouve pas cette information dans les documents fournis.\"
3. Tu ne dois JAMAIS inventer, supposer ou utiliser tes connaissances générales
4. Tu dois TOUJOURS citer la source entre crochets [Source: nom_du_document]

STRUCTURE DE RÉPONSE :
1. Réponse directe (2-3 phrases maximum)
2. Détails complémentaires si pertinents
3. Source(s) utilisée(s)
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "dags/faiss.index")
CHUNKS_PATH = os.path.join(
    BASE_DIR, "chunked/Manuel_Regles_RH_Complet_chunks.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# OpenRouter configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "mistralai/mistral-tiny"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Ollama configuration (fallback)
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"

REQUEST_TIMEOUT = 120
MAX_RETRIES = 2
RETRY_DELAY = 5

print(f"[DEBUG] OPENROUTER_API_KEY loaded: {bool(OPENROUTER_API_KEY)}")
if OPENROUTER_API_KEY:
    print(f"[DEBUG] API Key starts with: {OPENROUTER_API_KEY[:20]}...")

# ======================================
# LOAD MODELS AT STARTUP
# ======================================
print("Loading FAISS index...")
try:
    index = faiss.read_index(FAISS_INDEX_PATH)
    print("✅ FAISS index loaded")
except Exception as e:
    print(f"❌ Error loading FAISS index: {e}")
    raise

print("Loading chunks...")
try:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"✅ Chunks loaded ({len(chunks)} chunks)")
except Exception as e:
    print(f"❌ Error loading chunks: {e}")
    raise

print("Loading embedding model...")
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Embedding model loaded")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    raise

print("✅ RAG system ready!")


# ======================================
# LLM FUNCTIONS
# ======================================
def call_openrouter(prompt, system_prompt=None, model=OPENROUTER_MODEL, retry_count=0):
    """Call OpenRouter LLM with retry logic"""

    if not OPENROUTER_API_KEY:
        print("⚠️ OpenRouter API key not found, falling back to Ollama")
        return call_ollama(prompt, system_prompt, OLLAMA_MODEL)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8001",
        "X-Title": "RAG Assistant Safran"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt if system_prompt else ""},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 500
    }

    try:
        print(
            f"🔗 Appel OpenRouter (tentative {retry_count + 1}/{MAX_RETRIES + 1})...")

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()
        data = response.json()

        if "choices" in data and data["choices"]:
            print("✅ Réponse reçue d'OpenRouter")
            return data["choices"][0]["message"]["content"]
        else:
            return f"[Erreur OpenRouter] Réponse malformée"

    except requests.exceptions.Timeout:
        print(f"⏱️ Timeout OpenRouter")

        if retry_count < MAX_RETRIES:
            print(f"⏳ Attente de {RETRY_DELAY}s avant nouvelle tentative...")
            time.sleep(RETRY_DELAY)
            return call_openrouter(prompt, system_prompt, model, retry_count + 1)
        else:
            print("💡 Basculement vers Ollama...")
            return call_ollama(prompt, system_prompt, OLLAMA_MODEL)

    except requests.exceptions.ConnectionError:
        print(f"🚫 Erreur de connexion OpenRouter")
        print("💡 Basculement vers Ollama...")
        return call_ollama(prompt, system_prompt, OLLAMA_MODEL)

    except requests.exceptions.HTTPError as e:
        print(f"❌ Erreur HTTP {response.status_code}")
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", "")
            print(f"   Message: {error_msg}")
        except:
            pass

        if response.status_code == 401:
            print("⚠️ Clé API invalide, basculement vers Ollama...")
            return call_ollama(prompt, system_prompt, OLLAMA_MODEL)
        elif response.status_code == 429:
            print("⚠️ Limite atteinte, basculement vers Ollama...")
            return call_ollama(prompt, system_prompt, OLLAMA_MODEL)
        else:
            print("💡 Basculement vers Ollama...")
            return call_ollama(prompt, system_prompt, OLLAMA_MODEL)

    except Exception as e:
        print(f"❌ Erreur OpenRouter: {e}")
        print("💡 Basculement vers Ollama...")
        return call_ollama(prompt, system_prompt, OLLAMA_MODEL)


def call_ollama(prompt, system_prompt=None, model=OLLAMA_MODEL):
    """Call Ollama LLM as fallback"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        print(f"🔗 Appel Ollama ({model})...")
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        result = response.json().get("response", "")
        if result:
            print("✅ Réponse reçue d'Ollama")
        return result

    except requests.exceptions.ConnectionError:
        return "[Erreur] Impossible de se connecter à Ollama. Assurez-vous qu'Ollama est en cours d'exécution sur http://localhost:11434"
    except Exception as e:
        return f"[Erreur Ollama] {e}"


# ======================================
# RAG FUNCTION
# ======================================
def get_model_answer(question):
    """Get answer from RAG model for a given question"""

    try:
        print(f"\n📝 Question: {question}")

        # Embed the question
        print("🔍 Recherche de contexte pertinent...")
        q_emb = embedder.encode([question])

        # Search FAISS index
        D, I = index.search(np.array(q_emb).astype(np.float32), TOP_K)

        # Handle empty results
        if I is None or len(I[0]) == 0 or I[0][0] == -1:
            return "[Aucun contexte trouvé pour cette question dans la base de documents.]"

        # Retrieve relevant chunks
        retrieved_chunks = [chunks[i]
                            for i in I[0] if i != -1 and i < len(chunks)]
        if not retrieved_chunks:
            return "[Aucun contexte trouvé pour cette question dans la base de documents.]"

        retrieved_texts = [
            chunk["text"] if isinstance(
                chunk, dict) and "text" in chunk else str(chunk)
            for chunk in retrieved_chunks
        ]

        # Truncate context
        MAX_CONTEXT_CHARS = 6000
        context = "\n".join(retrieved_texts)[:MAX_CONTEXT_CHARS]

        print(f"📌 Contexte trouvé: {len(context)} caractères")

        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        # Call LLM (OpenRouter with Ollama fallback)
        print("🤖 Appel du modèle LLM...")
        answer = call_openrouter(prompt, system_prompt=SYSTEM_PROMPT)

        return answer

    except Exception as e:
        print(f"❌ Erreur dans get_model_answer: {e}")
        return f"[Erreur RAG] {str(e)}"


# ======================================
# DASHBOARD FUNCTIONS
# ======================================
def get_metrics():
    """Return placeholder metrics"""
    return {
        'intent_accuracy': 92,
        'validation_rate': 88,
        'escalation_rate': 5,
        'avg_response_time': 2.3,
        'active_users': 15,
        'user_satisfaction': 4.5,
        'total_conversations': 240,
        'total_questions': 580,
        'correct_answers': 534,
        'escalated_questions': 29
    }


def get_conversation_stats():
    """Return placeholder conversation stats"""
    return {
        'by_profil': {'RH': 120, 'CDI': 280, 'CDD': 140, 'Stagiaire': 40},
        'by_hour': {str(h): 20 + (h * 2) for h in range(24)},
        'feedback_distribution': {'positive': 480, 'negative': 60, 'none': 40},
        'similarity_distribution': {'high': 400, 'medium': 150, 'low': 30},
        'avg_response_time_by_profil': {'RH': 1.8, 'CDI': 2.5, 'CDD': 2.1, 'Stagiaire': 3.0}
    }


# ======================================
# MAIN
# ======================================
if __name__ == "__main__":
    question = input("Enter your question: ")
    print("\nSending to RAG system...")
    answer = get_model_answer(question)
    print("\n--- Model Answer ---\n")
    print(answer)
# ============================================================================
