from fastapi import APIRouter, HTTPException, status
from typing import List
import logging

from app.services.ldap_auth import LDAPAuth
from app.utils.jwt import create_anonymous_token, verify_token
from app.models.schemas import Token, UserLogin

router = APIRouter(prefix="/auth", tags=["authentication"])
logger = logging.getLogger(__name__)

@router.post("/login", response_model=Token)
async def login(user_data: UserLogin):
    """
    Authentifie un utilisateur via LDAP
    
    - Vérifie les credentials via LDAP
    - Récupère les groupes de l'utilisateur
    - Génère un JWT avec ID anonymisé
    """
    auth_service = LDAPAuth()
    
    # Authentification LDAP
    user_info = auth_service.authenticate_user(
        username=user_data.matricule,
        password=user_data.password
    )
    
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants incorrects",
        )
    
    # Génération du token anonymisé
    access_token = create_anonymous_token(
        username=user_data.matricule,
        groups=user_info["groups"]
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        anonymous_id=user_info["uid"],
        groups=user_info["groups"]
    )

@router.post("/verify")
async def verify_token_endpoint(token_data: dict):
    """
    Vérifie si un token est valide
    """
    token = token_data.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Token requis")
    
    try:
        payload = verify_token(token)
        return {
            "valid": True,
            "user_id": payload.get("anonymous_id"),
            "expires_at": payload.get("exp"),
            "roles": payload.get("groups", [])
        }
    except HTTPException:
        # verify_token lève déjà une HTTPException
        return {"valid": False}
    except Exception as e:
        # Pour les autres erreurs
        logger.error(f"Erreur vérification token: {e}")
        return {"valid": False}