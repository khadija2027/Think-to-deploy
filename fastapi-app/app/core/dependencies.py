from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.utils.jwt import verify_token
from app.models.schemas import TokenData

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    """
    Extrait et valide l'utilisateur depuis le token JWT
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalide ou expiré",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = verify_token(credentials.credentials)
        anonymous_id = payload.get("anonymous_id")
        groups = payload.get("groups", [])
        
        if anonymous_id is None:
            raise credentials_exception
            
        return TokenData(
            anonymous_id=anonymous_id,
            groups=groups
        )
    except Exception:
        raise credentials_exception

def require_admin(current_user: TokenData = Depends(get_current_user)):
    """
    Vérifie que l'utilisateur est administrateur
    """
    if "admins" not in [g.lower() for g in current_user.groups]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès refusé. Seuls les administrateurs peuvent accéder à cette ressource."
        )
    return current_user

def require_role(required_roles: list):
    """
    Factory pour vérifier plusieurs rôles
    """
    def role_checker(current_user: TokenData = Depends(get_current_user)):
        user_roles = [g.lower() for g in current_user.groups]
        required_lower = [r.lower() for r in required_roles]
        
        if not any(role in user_roles for role in required_lower):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permissions insuffisantes. Rôles requis: {required_roles}"
            )
        return current_user
    return role_checker