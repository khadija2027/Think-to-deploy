import sys
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import time
import requests


# OpenRouter DeepSeek configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "deepseek/deepseek-r1-0528:free"

# OpenRouter DeepSeek API call


def call_openrouter(prompt, system_prompt=None):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8001",
        "X-Title": "RAG Assistant Safran"
    }
    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt if system_prompt else ""},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        print("Appel OpenRouter DeepSeek API...")
        response = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(
            data), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"]
        return "[Erreur OpenRouter] Réponse malformée"
    except Exception as e:
        print(f"Erreur OpenRouter: {e}")
        return f"[Erreur OpenRouter] {e}"


sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

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

# ======================================
# CONFIGURATION
# ======================================
SYSTEM_PROMPT = """
Vous êtes un assistant IA expert, rigoureux et prudent. Vous NE DEVEZ JAMAIS inventer d'informations ni répondre sans preuve explicite dans le contexte fourni. Si la réponse n'est pas dans le contexte, dites simplement : "Je ne sais pas répondre à cette question avec les informations fournies." Ne faites AUCUNE supposition. Répondez uniquement sur la base du contexte fourni. Si la question porte sur une personne, un client ou un cas, ne donnez aucune information personnelle ou confidentielle. Si la question est hors sujet ou non professionnelle, refusez poliment. Répondez en français, de façon concise, professionnelle et factuelle. Si le contexte est vide, dites-le explicitement. N'inventez jamais de sources ou de citations. Ne donnez jamais la source à la fin de la réponse. Si la question est trop générale ou manque de détails, posez des questions à l'utilisateur pour obtenir des précisions AVANT de répondre. Ne donnez jamais d'avis médical, fiscal ou légal."
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "dags/faiss.index")
CHUNKS_PATH = os.path.join(
    BASE_DIR, "chunked/Manuel_Regles_RH_Complet_chunks.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# OpenRouter configuration
REQUEST_TIMEOUT = 3600
MAX_RETRIES = 1
RETRY_DELAY = 10

# ======================================
# LAZY LOADING
# ======================================
_index = None
_chunks = None
_embedder = None


def get_index():
    """Charge le FAISS index de manière lazy"""
    global _index
    if _index is None:
        print("Loading FAISS index...")
        try:
            _index = faiss.read_index(FAISS_INDEX_PATH)
            print("✅ FAISS index loaded")
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
            raise
    return _index


def get_chunks():
    """Charge les chunks de manière lazy"""
    global _chunks
    if _chunks is None:
        print("Loading chunks...")
        try:
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                _chunks = json.load(f)
            print(f"✅ Chunks loaded ({len(_chunks)} chunks)")
        except Exception as e:
            print(f"❌ Error loading chunks: {e}")
            raise
    return _chunks


def get_embedder():
    """Charge le modèle d'embedding de manière lazy"""
    global _embedder
    if _embedder is None:
        print("Loading embedding model...")
        try:
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
    return _embedder


# Gemini API call
def call_gemini(prompt, system_prompt=None):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {"parts": [
                {"text": (system_prompt + "\n" if system_prompt else "") + prompt}
            ]}
        ]
    }
    try:
        print("Appel Gemini API...")
        response = requests.post(
            GEMINI_URL, headers=headers, params={"key": GEMINI_API_KEY}, json=data, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        if "candidates" in result and result["candidates"]:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        return "[Erreur Gemini] Réponse malformée"
    except Exception as e:
        print(f"Erreur Gemini: {e}")
        return f"[Erreur Gemini] {e}"


# ======================================
# RAG FUNCTION
# ======================================
def get_model_answer(question):
    """Get answer from RAG model for a given question"""

    try:
        print(f"\nQuestion: {question}")

        # Load models lazily
        index = get_index()
        chunks = get_chunks()
        embedder = get_embedder()

        # Embed the question
        print("Recherche de contexte pertinent...")
        q_emb = embedder.encode([question])

        # Search FAISS index
        distances, indices = index.search(
            np.array(q_emb).astype(np.float32), TOP_K)

        print(f"Top {TOP_K} résultats trouvés")

        # Retrieve relevant chunks
        retrieved_texts = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(chunks):
                chunk = chunks[idx]
                text = chunk["text"] if isinstance(
                    chunk, dict) and "text" in chunk else str(chunk)
                retrieved_texts.append(text)
                print(f"   Chunk {idx}")

        if not retrieved_texts:
            return "Aucun contexte trouvé pour répondre à cette question."

        # Build context
        MAX_CONTEXT_CHARS = 6000
        context = "\n\n".join(retrieved_texts)[:MAX_CONTEXT_CHARS]

        print(f"Contexte: {len(context)} caractères")

        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        # Call OpenRouter DeepSeek LLM
        print("Appel du modèle DeepSeek (OpenRouter)...")
        answer = call_openrouter(prompt, system_prompt=SYSTEM_PROMPT)
        return answer

    except Exception as e:
        print(f"Erreur dans get_model_answer: {e}")
        import traceback
        traceback.print_exc()
        return f"[Erreur RAG] {str(e)}"


# ======================================
# DASHBOARD FUNCTIONS
# ======================================
def get_metrics():
    """Return dashboard metrics"""
    # Charger les vraies données
    BASE = os.path.dirname(os.path.abspath(__file__))
    FEEDBACK_FILE = os.path.join(BASE, "frontend_chatbot/feedbacks.json")
    HISTORY_FILE = os.path.join(
        BASE, "frontend_chatbot/conversation_history.json")

    # Feedbacks
    feedbacks = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            feedbacks = json.load(f)

    # Conversations
    conversations = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            conversations = json.load(f)

    # Métriques dynamiques
    total_conversations = len(conversations)
    total_questions = len(conversations)
    correct_answers = sum(1 for c in conversations if c.get(
        "answer") and not str(c.get("answer")).lower().startswith("[erreur"))
    escalated_questions = sum(
        1 for c in conversations if "escalade" in str(c.get("answer")).lower())
    avg_response_time = 2.3  # Placeholder, à calculer si temps dispo
    active_users = len(set(c.get("username") for c in conversations))

    # Satisfaction utilisateur (feedbacks)
    pos = sum(1 for f in feedbacks if f.get("type") == "positive")
    neg = sum(1 for f in feedbacks if f.get("type") == "negative")
    none = len(feedbacks) - pos - neg
    user_satisfaction = round(
        (pos / len(feedbacks) * 5) if feedbacks else 0, 2)

    # Précision (correct_answers / total_questions)
    intent_accuracy = round(
        (correct_answers / total_questions * 100) if total_questions else 0, 2)
    validation_rate = round((pos / total_questions * 100)
                            if total_questions else 0, 2)
    escalation_rate = round(
        (escalated_questions / total_questions * 100) if total_questions else 0, 2)

    return {
        'intent_accuracy': intent_accuracy,
        'validation_rate': validation_rate,
        'escalation_rate': escalation_rate,
        'avg_response_time': avg_response_time,
        'active_users': active_users,
        'user_satisfaction': user_satisfaction,
        'total_conversations': total_conversations,
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'escalated_questions': escalated_questions
    }


def get_conversation_stats():
    """Return conversation statistics"""
    BASE = os.path.dirname(os.path.abspath(__file__))
    FEEDBACK_FILE = os.path.join(BASE, "frontend_chatbot/feedbacks.json")
    HISTORY_FILE = os.path.join(
        BASE, "frontend_chatbot/conversation_history.json")

    feedbacks = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            feedbacks = json.load(f)

    conversations = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            conversations = json.load(f)

    # By profil
    by_profil = {}
    for c in conversations:
        profil = c.get("profil", "CDI")
        by_profil[profil] = by_profil.get(profil, 0) + 1

    # By hour
    by_hour = {}
    for c in conversations:
        ts = c.get("timestamp")
        if ts:
            hour = ts[11:13]
            by_hour[hour] = by_hour.get(hour, 0) + 1

    # Feedback distribution
    pos = sum(1 for f in feedbacks if f.get("type") == "positive")
    neg = sum(1 for f in feedbacks if f.get("type") == "negative")
    none = len(feedbacks) - pos - neg
    feedback_distribution = {"positive": pos, "negative": neg, "none": none}

    # Similarity distribution (placeholder)
    similarity_distribution = {"high": 0, "medium": 0, "low": 0}

    # Avg response time by profil (placeholder)
    avg_response_time_by_profil = {}
    for profil in by_profil:
        avg_response_time_by_profil[profil] = 2.5  # Placeholder

    return {
        'by_profil': by_profil,
        'by_hour': by_hour,
        'feedback_distribution': feedback_distribution,
        'similarity_distribution': similarity_distribution,
        'avg_response_time_by_profil': avg_response_time_by_profil
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