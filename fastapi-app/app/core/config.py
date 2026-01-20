# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List, Union


class Settings(BaseSettings):
    """Configuration de l'application"""
    
    # Configuration générale
    PROJECT_NAME: str = "FastAPI LDAP Auth"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS
    BACKEND_CORS_ORIGINS: Union[str, List[str]] = ["*"]
    
    # Configuration LDAP
    LDAP_SERVER: str
    LDAP_PORT: int =  636
    LDAP_USE_TLS: bool = False
    LDAP_BASE_DN: str
    LDAP_USER_DN: str
    LDAP_GROUP_DN: str
    LDAP_BIND_DN: str
    LDAP_BIND_PASSWORD: str
    
    # Configuration JWT
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
    @property
    def ldap_server_uri(self) -> str:
        """Construit l'URI du serveur LDAP"""
        protocol = "ldaps" if self.LDAP_USE_TLS else "ldap"
        return f"{protocol}://{self.LDAP_SERVER}:{self.LDAP_PORT}"
    
    def get_cors_origins(self) -> List[str]:
        """Retourne les origines CORS sous forme de liste"""
        if isinstance(self.BACKEND_CORS_ORIGINS, str):
            if self.BACKEND_CORS_ORIGINS == "*":
                return ["*"]
            return [origin.strip() for origin in self.BACKEND_CORS_ORIGINS.split(",")]
        return self.BACKEND_CORS_ORIGINS


# Instance globale des settings
settings = Settings()