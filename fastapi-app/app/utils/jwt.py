from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import hashlib
from jose import JWTError, jwt
from fastapi import HTTPException
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

def create_anonymous_id(username: str) -> str:
    """
    Crée un ID anonymisé à partir du username
    """
    salt = settings.JWT_SECRET_KEY.encode()
    hash_input = f"{username}{salt.hex()}".encode()
    
    hash_obj = hashlib.sha256(hash_input)
    return hash_obj.hexdigest()[:16]

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Crée un JWT token avec expiration"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt

def create_anonymous_token(username: str, groups: List[str]) -> str:
    """
    Crée un token JWT anonymisé
    """
    anonymous_id = create_anonymous_id(username)
    
    token_data = {
        "sub": anonymous_id,
        "anonymous_id": anonymous_id,
        "groups": groups,
        "auth_type": "ldap_anonymous"
    }
    
    return create_access_token(token_data)

def verify_token(token: str) -> Dict[str, Any]:
    """Vérifie et décode un token JWT"""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Token invalide ou expiré")