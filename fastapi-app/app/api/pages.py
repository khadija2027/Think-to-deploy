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
    
    # Créer une réponse de redirection
    if is_admin:
        response = RedirectResponse(url="/dashboard", status_code=302)
    else:
        response = RedirectResponse(url="/chatbot", status_code=302)
    
    # Stocker le token dans un cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=1800,  # 30 minutes
        samesite="lax"
    )
    
    return response

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Dashboard pour les admins"""
    # Récupérer le token du cookie
    token = request.cookies.get("access_token")
    
    if not token:
        return RedirectResponse(url="/", status_code=302)
    
    try:
        # Vérifier le token
        payload = verify_token(token)
        groups = payload.get("groups", [])
        
        # Vérifier que l'utilisateur est admin
        if "admins" not in [g.lower() for g in groups]:
            return RedirectResponse(url="/chatbot", status_code=302)
        
        # Récupérer des infos utilisateur
        anonymous_id = payload.get("anonymous_id")
        
        # Récupérer la liste des utilisateurs
        auth_service = LDAPAuth()
        users = auth_service.list_all_users()
        
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "user_id": anonymous_id,
                "groups": groups,
                "users": users,
                "user_count": len(users),
                "metrics": {
                    "active_users": len(users),
                    "total_conversations": 1247,
                    "user_satisfaction": 94.0,
                    "intent_accuracy": 87.5,
                    "validation_rate": 92.3,
                    "escalation_rate": 5.2,
                    "avg_response_time": 2.3
                },
                "stats": {
                    "feedback_distribution": {
                        "positive": 85,
                        "negative": 12,
                        "none": 3
                    }
                },
                "conversations": [
                    {
                        "question": "Comment demander des congés ?",
                        "reponse": "Pour demander des congés, vous devez utiliser le formulaire en ligne accessible depuis votre espace employé. Sélectionnez les dates souhaitées et choisissez le type de congé (annuel, maladie, etc.).",
                        "profil": anonymous_id,
                        "similarite": 0.95,
                        "timestamp": "2024-01-20 14:30:00",
                        "feedback": "positive"
                    },
                    {
                        "question": "Informations sur la mutuelle santé",
                        "reponse": "La mutuelle santé couvre les frais médicaux, dentaires et optiques selon les niveaux choisis. Les cotisations sont prélevées automatiquement sur votre salaire.",
                        "profil": "employe001",
                        "similarite": 0.89,
                        "timestamp": "2024-01-20 12:15:00",
                        "feedback": "positive"
                    }
                ]
            }
        )
    except Exception:
        return RedirectResponse(url="/", status_code=302)

@router.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    """Interface chatbot pour les utilisateurs"""
    # Récupérer le token du cookie
    token = request.cookies.get("access_token")
    
    if not token:
        return RedirectResponse(url="/", status_code=302)
    
    try:
        # Vérifier le token
        payload = verify_token(token)
        groups = payload.get("groups", [])
        anonymous_id = payload.get("anonymous_id")
        
        return templates.TemplateResponse(
            "chatbot.html",
            {
                "request": request,
                "user_id": anonymous_id,
                "groups": groups
            }
        )
    except Exception:
        return RedirectResponse(url="/", status_code=302)

@router.get("/logout")
async def logout():
    """Déconnexion"""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("access_token")
    return response