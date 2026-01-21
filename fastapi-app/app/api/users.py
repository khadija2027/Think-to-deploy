from fastapi import APIRouter, Depends, HTTPException, status, Form
from typing import List

from app.core.dependencies import get_current_user
from app.models.schemas import UserInfo, TokenData
from app.services.ldap_auth import LDAPAuth

router = APIRouter(prefix="/users", tags=["users"])
api_router = APIRouter(prefix="/api", tags=["api"])

@router.get("/me", response_model=UserInfo)
async def get_current_user_info(
    current_user: TokenData = Depends(get_current_user)
):
    """
    Récupère les informations de l'utilisateur courant
    """
    auth_service = LDAPAuth()
    user_info = auth_service.get_user_info(current_user.anonymous_id)
    
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )
    
    return UserInfo(**user_info)

@api_router.post("/ask")
async def ask_question(
    question: str = Form(...),
    current_user: TokenData = Depends(get_current_user)
):
    """
    Endpoint pour poser des questions au chatbot
    """
    # Pour l'instant, retourner une réponse simple
    # TODO: Intégrer la logique du chatbot depuis T2D_Chatbot_for_HR
    return {
        "answer": f"Bonjour {current_user.anonymous_id}! Votre question '{question}' a été reçue. Cette fonctionnalité sera bientôt disponible avec l'intégration complète du chatbot.",
        "question": question,
        "user": current_user.anonymous_id
    }