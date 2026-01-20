#!/usr/bin/env python3
"""
Script de diagnostic complet pour déboguer le problème d'API OpenRouter
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests

print("=" * 80)
print("🔍 DIAGNOSTIC COMPLET - RAG ASSISTANT")
print("=" * 80)

# ============================================================================
# 1. VÉRIFIER LA STRUCTURE DES FICHIERS
# ============================================================================
print("\n1️⃣ STRUCTURE DES FICHIERS")
print("-" * 80)

t2d_path = Path(r"C:\AirflowProjects\T2D")
print(f"📁 Répertoire T2D: {t2d_path}")
print(f"   Existe: {t2d_path.exists()}")

if t2d_path.exists():
    print("\n   Fichiers importants:")
    important_files = [
        ("test_rag.py", t2d_path / "test_rag.py"),
        (".env", t2d_path / ".env"),
        ("faiss.index", t2d_path / "dags" / "faiss.index"),
        ("chunks.json", t2d_path / "chunked" /
         "Manuel_Regles_RH_Complet_chunks.json"),
    ]

    for name, path in important_files:
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {name}: {path}")

# ============================================================================
# 2. VÉRIFIER LES VARIABLES D'ENVIRONNEMENT
# ============================================================================
print("\n2️⃣ VARIABLES D'ENVIRONNEMENT")
print("-" * 80)

print(f"📁 Répertoire courant: {os.getcwd()}")

# Essayez de charger depuis T2D
print(f"\n🔍 Tentative de chargement de .env depuis T2D...")
env_path = t2d_path / ".env"
if env_path.exists():
    print(f"   ✅ .env trouvé à: {env_path}")
    result = load_dotenv(env_path)
    print(f"   load_dotenv() retourné: {result}")
else:
    print(f"   ❌ .env NOT trouvé à: {env_path}")

# Vérifiez la clé API
api_key = os.getenv("OPENROUTER_API_KEY")
print(f"\n🔑 OPENROUTER_API_KEY:")
print(f"   Chargée: {bool(api_key)}")

if api_key:
    print(f"   Longueur: {len(api_key)}")
    print(f"   Premiers 30 caractères: {api_key[:30]}...")
    print(f"   Commence par 'sk-or-v1-': {api_key.startswith('sk-or-v1-')}")

    # Vérifiez les caractères problématiques
    issues = []
    if " " in api_key:
        issues.append("Contient des espaces")
    if "\n" in api_key:
        issues.append("Contient des sauts de ligne")
    if "\t" in api_key:
        issues.append("Contient des tabulations")
    if api_key != api_key.strip():
        issues.append("Contient des espaces de début/fin")

    if issues:
        print(f"   ⚠️ PROBLÈMES DÉTECTÉS:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print(f"   ✅ Format semble valide")
else:
    print(f"   ❌ CLÉ NON CHARGÉE!")

# ============================================================================
# 3. TESTER LA CONNEXION À OPENROUTER
# ============================================================================
print("\n3️⃣ TEST DE CONNEXION À OPENROUTER")
print("-" * 80)

if not api_key:
    print("❌ Impossible de tester sans clé API")
else:
    print(f"🔗 Envoi d'une requête de test à OpenRouter...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8001",
        "X-Title": "Diagnostic Test"
    }

    payload = {
        "model": "mistralai/mistral-tiny",
        "messages": [
            {"role": "user", "content": "Say 'Hello' in exactly one word"}
        ],
        "stream": False,
        "max_tokens": 50
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        print(f"\n   📊 Statut HTTP: {response.status_code}")

        if response.status_code == 200:
            print(f"   ✅ SUCCÈS!")
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            print(f"   💬 Réponse: {answer}")
        else:
            print(f"   ❌ ERREUR HTTP {response.status_code}")
            print(f"   📝 Réponse complète:")
            try:
                print(f"      {response.json()}")
            except:
                print(f"      {response.text}")

    except requests.exceptions.Timeout:
        print(f"   ❌ TIMEOUT - OpenRouter ne répond pas")
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ ERREUR DE CONNEXION: {e}")
    except Exception as e:
        print(f"   ❌ ERREUR: {type(e).__name__}: {e}")

# ============================================================================
# 4. TESTER L'IMPORT DE test_rag.py
# ============================================================================
print("\n4️⃣ TEST D'IMPORT DE test_rag.py")
print("-" * 80)

sys.path.insert(0, str(t2d_path))
os.chdir(str(t2d_path))

print(f"📁 Répertoire courant changé à: {os.getcwd()}")
print(f"📝 sys.path[0]: {sys.path[0]}")

try:
    print(f"\n🔍 Tentative d'import de test_rag...")
    from test_rag import get_model_answer, get_metrics, get_conversation_stats
    print(f"   ✅ Import réussi!")

    # Testez une fonction simple
    print(f"\n🧪 Test de get_metrics()...")
    metrics = get_metrics()
    print(f"   ✅ get_metrics() fonctionne")
    print(f"   Résultat: {list(metrics.keys())}")

except ImportError as e:
    print(f"   ❌ ERREUR D'IMPORT: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"   ❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 5. RÉSUMÉ
# ============================================================================
print("\n" + "=" * 80)
print("📋 RÉSUMÉ DES RÉSULTATS")
print("=" * 80)

print("\n✅ Actions à faire si des erreurs sont détectées:")
print("   1. Vérifiez que T2D existe et contient tous les fichiers")
print("   2. Vérifiez que .env existe et contient OPENROUTER_API_KEY")
print("   3. Vérifiez que la clé API commence par 'sk-or-v1-'")
print("   4. Vérifiez votre connexion internet")
print("   5. Vérifiez que OpenRouter est accessible")
print("\n" + "=" * 80)
