from fastapi import APIRouter, Request, Form, HTTPException, status, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import Optional

from app.services.ldap_auth import LDAPAuth
from app.utils.jwt import create_anonymous_token, verify_token

router = APIRouter(tags=["pages"])

# Configuration des templates
templates = Jinja2Templates(directory="/T2D_Chatbot_for_HR/frontend_chatbot/templates")

@router.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    """Page de login"""
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    """Traitement du formulaire de login"""
    auth_service = LDAPAuth()
    
    # Authentification LDAP
    user_info = auth_service.authenticate_user(
        username=username,
        password=password
    )
    
    if not user_info:
        # Retourner à la page de login avec erreur
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Identifiants incorrects"
            },
            status_code=401
        )
    
    # Génération du token
    access_token = create_anonymous_token(
        username=username,
        groups=user_info["groups"]
    )
    
    # Vérifier si l'utilisateur est admin
    is_admin = "admins" in [g.lower() for g in user_info["groups"]]
    
    # Créer une réponse de redirection vers l'application chatbot
    chatbot_url = f"http://localhost:8001/auth/jwt?token={access_token}"
    if is_admin:
        response = RedirectResponse(url=chatbot_url, status_code=302)
    else:
        response = RedirectResponse(url=chatbot_url, status_code=302)
    
    # Stocker le token dans un cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=1800,  # 30 minutes
        samesite="lax"
    )
    
    return response

@router.get("/logout")
async def logout():
    """Déconnexion"""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("access_token")
    return response