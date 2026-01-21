"""
Module de gestion de l'historique des conversations
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Chemin du fichier de stockage
HISTORY_FILE = Path(__file__).parent / "conversation_history.json"


def save_conversation(username: str, question: str, answer: str, profil: str = "CDI"):
    """
    Enregistrer une conversation dans l'historique

    Args:
        username: Nom d'utilisateur
        question: Question posée
        answer: Réponse fournie
        profil: Profil de l'utilisateur (CDI, CDD, Stagiaire, RH)
    """
    try:
        # Charger l'historique existant
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []

        # Créer une nouvelle entrée
        entry = {
            "username": username,
            "profil": profil,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }

        # Ajouter à l'historique
        history.append(entry)

        # Sauvegarder
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(f"Conversation enregistrée pour {username}")
        return True

    except Exception as e:
        print(f"Erreur lors de l'enregistrement de la conversation: {e}")
        return False


def get_conversation_history(username: str = None) -> List[Dict]:
    """
    Récupérer l'historique des conversations

    Args:
        username: Si spécifié, retourne uniquement les conversations de cet utilisateur

    Returns:
        Liste des conversations
    """
    try:
        if not HISTORY_FILE.exists():
            return []

        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)

        # Filtrer par utilisateur si demandé
        if username:
            history = [h for h in history if h.get("username") == username]

        # Trier par timestamp (le plus récent en premier)
        history = sorted(history, key=lambda x: x.get(
            "timestamp", ""), reverse=True)

        return history

    except Exception as e:
        print(f"Erreur lors de la lecture de l'historique: {e}")
        return []


def clear_conversation_history(username: str = None):
    """
    Effacer l'historique des conversations

    Args:
        username: Si spécifié, efface uniquement les conversations de cet utilisateur
    """
    try:
        if not HISTORY_FILE.exists():
            return True

        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)

        if username:
            # Supprimer uniquement les conversations de cet utilisateur
            history = [h for h in history if h.get("username") != username]
        else:
            # Effacer tout
            history = []

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(f"Historique effacé")
        return True

    except Exception as e:
        print(f"Erreur lors de l'effacement de l'historique: {e}")
        return False


def get_statistics_by_profil() -> Dict:
    """
    Obtenir les statistiques par profil d'utilisateur
    """
    try:
        history = get_conversation_history()

        stats = {
            "RH": 0,
            "CDI": 0,
            "CDD": 0,
            "Stagiaire": 0
        }

        for entry in history:
            profil = entry.get("profil", "CDI")
            if profil in stats:
                stats[profil] += 1

        return stats

    except Exception as e:
        print(f"Erreur lors du calcul des statistiques: {e}")
        return {"RH": 0, "CDI": 0, "CDD": 0, "Stagiaire": 0}