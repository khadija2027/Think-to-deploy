# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.auth import router as auth_router
from app.api.users import router as users_router, api_router
from app.api.admin import router as admin_router
from app.api.pages import router as pages_router

import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log vers la console
    ]
)

# Augmenter le niveau de log pour ldap3 si nécessaire
logging.getLogger('ldap3').setLevel(logging.WARNING)  # ou INFO pour plus de détails

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API d'authentification avec LDAP et JWT",
    version="1.0.0",
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monter les fichiers statiques
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Inclure les routeurs
app.include_router(pages_router)  # Routes des pages web (doit être avant les autres)
app.include_router(auth_router)   # Routes API
app.include_router(users_router)
app.include_router(api_router)
app.include_router(admin_router)

@app.get("/api")
async def api_root():
    """Route racine de l'API"""
    return {
        "message": "API d'authentification LDAP",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )