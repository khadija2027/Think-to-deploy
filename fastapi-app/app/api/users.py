from fastapi import APIRouter, Depends, HTTPException, status, Form
from typing import List
import logging

from app.core.dependencies import get_current_user
from app.models.schemas import UserInfo, TokenData
from app.services.ldap_auth import LDAPAuth

logger = logging.getLogger(__name__)

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