from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from app.core.dependencies import get_current_user, require_admin
from app.models.schemas import TokenData
from app.services.ldap_auth import LDAPAuth

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/dashboard")
async def admin_dashboard(
    current_user: TokenData = Depends(require_admin)
):
    """
    Dashboard administrateur
    """
    return {
        "message": "Bienvenue dans le dashboard admin",
        "user": current_user.anonymous_id,
        "roles": current_user.groups
    }

@router.get("/users")
async def list_all_users(
    current_user: TokenData = Depends(require_admin)
):
    """
    Liste tous les utilisateurs (admin seulement)
    """
    auth_service = LDAPAuth()
    users = auth_service.list_all_users()
    
    return {
        "count": len(users),
        "users": users
    }