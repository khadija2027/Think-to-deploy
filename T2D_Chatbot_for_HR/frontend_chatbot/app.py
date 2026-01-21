from starlette.middleware.sessions import SessionMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi import FastAPI, Request, Depends, Form
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import jwt
from jwt import PyJWTError

# ======================================
# PATH CONFIGURATION
# ======================================
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent.resolve()

print(f"📁 APP_DIR: {APP_DIR}")
print(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")

# Add T2D to sys.path for imports
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"

print(f"📁 Static dir exists: {STATIC_DIR.exists()}")
print(f"📁 Templates dir exists: {TEMPLATES_DIR.exists()}")
if TEMPLATES_DIR.exists():
    print(f"📁 Templates files: {list(TEMPLATES_DIR.glob('*.html'))}")

# Load environment variables from .env file
load_dotenv(dotenv_path=APP_DIR / ".env")

# ======================================
# JWT CONFIGURATION (pour intégration avec app principale)
# ======================================
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "ma-cle-secrete-pour-developpement-changez-moi")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# ======================================
# IMPORT RAG MODEL
# ======================================
try:
    from test_rag import get_model_answer, get_metrics, get_conversation_stats
    print("✅ RAG model imported successfully")
except ImportError as e:
    print(f"❌ Failed to import RAG model: {e}")

    def get_model_answer(question):
        return f"[Demo] Answer to: {question}"

    def get_metrics():
        return {}

    def get_conversation_stats():
        return {}

# ======================================
# EMAIL NOTIFICATION UTILITY
# ======================================


def send_hr_notification(subject, body, to_emails=["hr@example.com"]):
    # Configure your SMTP server here
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.example.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USER = os.getenv("SMTP_USER", "your_email@example.com")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "your_password")
    FROM_EMAIL = SMTP_USER

    msg = MIMEMultipart()
    msg['From'] = FROM_EMAIL
    msg['To'] = ", ".join(to_emails)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(FROM_EMAIL, to_emails, msg.as_string())
        print(f"✅ Notification sent to HR: {to_emails}")
    except Exception as e:
        print(f"❌ Failed to send HR notification: {e}")


# ======================================
# IMPORT CONVERSATION HISTORY
# ======================================
try:
    from conversation_history import save_conversation, get_conversation_history
    print("✅ Conversation history module imported successfully")
except ImportError as e:
    print(f"⚠️ Conversation history module not found: {e}")

    def save_conversation(username, question, answer, profil="CDI"):
        pass

    def get_conversation_history(username=None):
        return []

# ======================================
# AUTHENTICATION
# ======================================


def authenticate(username: str, password: str) -> Optional[dict]:
    """Authenticate user with demo accounts"""
    demo_accounts = {
        "admin": {"password": "admin", "profil": "RH", "nom_complet": "Administrator"},
        "rh_user": {"password": "password123", "profil": "RH", "nom_complet": "Manager RH"},
        "user.cdi": {"password": "password123", "profil": "CDI", "nom_complet": "Employé CDI"},
        "user.cdd": {"password": "password123", "profil": "CDD", "nom_complet": "Employé CDD"},
        "user.stagiaire": {"password": "password123", "profil": "Stagiaire", "nom_complet": "Stagiaire"},
    }
    if username in demo_accounts and demo_accounts[username]["password"] == password:
        return {
            "username": username,
            "profil": demo_accounts[username]["profil"],
            "nom_complet": demo_accounts[username]["nom_complet"]
        }
    return None


def get_current_user(request: Request) -> dict:
    """Get current user from session"""
    user = request.session.get("user")
    if user:
        return user
    return {"username": "guest", "profil": "CDI", "nom_complet": "Guest User"}


def validate_jwt_token(token: str) -> dict:
    """Validate JWT token from main authentication app"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username = payload.get("anonymous_id")
        groups = payload.get("groups", [])
        
        # Déterminer le profil basé sur les groupes
        profil = "CDI"  # Par défaut
        if "admins" in [g.lower() for g in groups]:
            profil = "RH"
        
        return {
            "username": username,
            "profil": profil,
            "nom_complet": username,  # On utilise le username comme nom complet
            "authenticated_via": "jwt"
        }
    except PyJWTError:
        return None


# ======================================
# FASTAPI APP INITIALIZATION
# ======================================
app = FastAPI(title="Assistant RH Intelligent - RAG")

# Add session middleware
app.add_middleware(SessionMiddleware,
                   secret_key="your-secret-key-change-in-production")

# Create directories if they don't exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

print("✅ App initialized")

# ======================================
# ROUTES
# ======================================


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Assistant RH API is running"}


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login"""
    user = authenticate(username, password)
    if user:
        request.session["user"] = user
        print(f"✅ User logged in: {user['username']}")
        return RedirectResponse(url="/chatbot", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})


@app.get("/logout")
async def logout(request: Request):
    """Logout"""
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/auth/jwt")
async def auth_via_jwt(request: Request, token: str):
    """Authenticate via JWT token from main app"""
    user = validate_jwt_token(token)
    if user:
        request.session["user"] = user
        print(f"✅ User authenticated via JWT: {user['username']}")
        
        # Rediriger vers la page appropriée selon le profil
        if user.get("profil") == "RH":
            return RedirectResponse(url="/dashboard", status_code=303)
        else:
            return RedirectResponse(url="/chatbot", status_code=303)
    
    return RedirectResponse(url="/login", status_code=303)


@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request, user: dict = Depends(get_current_user)):
    """Chatbot page"""
    print(f"📄 Serving chatbot page for {user['username']}")
    if user.get("profil") == "RH":
        return RedirectResponse(url="/dashboard", status_code=303)

    return templates.TemplateResponse(
        "chatbot.html",
        {"request": request, "user": user, "profil": user.get("profil")}
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, user: dict = Depends(get_current_user)):
    """Dashboard page"""
    if user.get("profil") != "RH":
        return RedirectResponse(url="/chatbot", status_code=303)

    try:
        metrics = get_metrics() if callable(get_metrics) else {}
        stats = get_conversation_stats() if callable(get_conversation_stats) else {}
    except:
        metrics = {}
        stats = {}

    # Charger les feedbacks réels
    FEEDBACK_FILE = APP_DIR / "feedbacks.json"
    feedbacks = []
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
        except Exception as e:
            print(f"❌ Erreur chargement feedbacks: {e}")

    # === NOTIFY HR IF MODEL PERFORMANCE IS BAD ===
    # Critère 1 : 5 feedbacks négatifs consécutifs
    BAD_PERFORMANCE = False
    NEGATIVE_FEEDBACK_STREAK = 5
    if feedbacks:
        last_feedbacks = feedbacks[-NEGATIVE_FEEDBACK_STREAK:]
        if len(last_feedbacks) == NEGATIVE_FEEDBACK_STREAK and all(fb.get('type') == 'negative' for fb in last_feedbacks):
            BAD_PERFORMANCE = True
    # Critère 2 : (optionnel) temps de réponse élevé ou autre critère
    # if stats.get('avg_response_time') and stats['avg_response_time'] > 10:
    #     BAD_PERFORMANCE = True
    if BAD_PERFORMANCE:
        body = (
            "Bonjour équipe RH,\n\n"
            "Le chatbot RH vient de recevoir 5 feedbacks négatifs consécutifs de la part des utilisateurs.\n"
            "Cela peut indiquer un problème de pertinence ou de qualité des réponses générées.\n\n"
            "Merci de consulter le dashboard pour plus de détails et d'intervenir si nécessaire.\n\n"
            "Ceci est un message automatique envoyé le {date}.\n\n"
            "Cordialement,\nL'assistant RH automatique"
        ).format(date=datetime.now().strftime('%d/%m/%Y à %H:%M'))
        send_hr_notification(
            subject="Alerte : 5 feedbacks négatifs consécutifs - Chatbot RH",
            body=body,
            to_emails=["taggaaikrame@gmail.com"]
        )

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "user": user, "metrics": metrics,
            "stats": stats, "feedbacks": feedbacks}
    )


@app.post("/api/ask")
async def ask_question(request: Request, user: dict = Depends(get_current_user)):
    """API endpoint to query RAG model"""
    try:
        data = await request.json()
        question = data.get("question", "").strip()

        if not question:
            return JSONResponse({"error": "No question provided"}, status_code=400)

        answer = get_model_answer(question)

        # Si le modèle retourne une erreur (sous forme de string ou d'objet), notifier RH
        if isinstance(answer, dict) and answer.get("error"):
            body = (
                "Bonjour équipe RH,\n\n"
                "Le chatbot RH a rencontré une erreur lors de la génération d'une réponse :\n"
                f"{answer.get('error')}\n\n"
                "Merci de vérifier le système ou de consulter les logs pour plus d'informations.\n\n"
                "Ceci est un message automatique envoyé le {date}.\n\n"
                "Cordialement,\nL'assistant RH automatique"
            ).format(date=datetime.now().strftime('%d/%m/%Y à %H:%M'))
            send_hr_notification(
                subject="Alerte : Erreur du modèle RAG",
                body=body,
                to_emails=["taggaaikrame@gmail.com"]
            )
        elif isinstance(answer, str) and ("error" in answer.lower() or "erreur" in answer.lower()):
            body = (
                "Bonjour équipe RH,\n\n"
                "Le chatbot RH a rencontré une erreur lors de la génération d'une réponse :\n"
                f"{answer}\n\n"
                "Merci de vérifier le système ou de consulter les logs pour plus d'informations.\n\n"
                "Ceci est un message automatique envoyé le {date}.\n\n"
                "Cordialement,\nL'assistant RH automatique"
            ).format(date=datetime.now().strftime('%d/%m/%Y à %H:%M'))
            send_hr_notification(
                subject="Alerte : Erreur du modèle RAG",
                body=body,
                to_emails=["taggaaikrame@gmail.com"]
            )

        # Save conversation
        save_conversation(
            username=user.get("username", "unknown"),
            question=question,
            answer=answer,
            profil=user.get("profil", "CDI")
        )

        return JSONResponse({
            "answer": answer,
            "question": question,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ Error in /api/ask: {str(e)}")
        # Notifier RH en cas d'exception
        body = (
            "Bonjour équipe RH,\n\n"
            "Une exception a été levée lors de la génération d'une réponse par le chatbot RH :\n"
            f"{str(e)}\n\n"
            "Merci de vérifier le système ou de consulter les logs pour plus d'informations.\n\n"
            "Ceci est un message automatique envoyé le {date}.\n\n"
            "Cordialement,\nL'assistant RH automatique"
        ).format(date=datetime.now().strftime('%d/%m/%Y à %H:%M'))
        send_hr_notification(
            subject="Alerte : Exception lors de la génération de réponse",
            body=body,
            to_emails=["taggaaikrame@gmail.com"]
        )
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/feedback")
async def submit_feedback(request: Request, user: dict = Depends(get_current_user)):
    """Submit feedback"""
    try:
        data = await request.json()
        feedback_type = data.get("type")
        comment = data.get("comment", "").strip()

        FEEDBACK_FILE = APP_DIR / "feedbacks.json"
        entry = {
            "timestamp": datetime.now().isoformat(),
            "username": user.get("username", "unknown"),
            "profil": user.get("profil", "CDI"),
            "type": feedback_type,
            "comment": comment
        }

        # Load existing feedbacks
        if FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
        else:
            feedbacks = []

        # Add new feedback
        feedbacks.append(entry)

        # Save feedbacks
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(feedbacks, f, ensure_ascii=False, indent=2)

        print(f"✅ Feedback saved from {user.get('username')}")
        return JSONResponse({"message": "Merci pour votre feedback !"})

    except Exception as e:
        print(f"❌ Error in /api/feedback: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/history")
async def get_history(request: Request, user: dict = Depends(get_current_user)):
    """Get conversation history"""
    try:
        if user.get("profil") == "RH":
            # RH can see all conversations
            history = get_conversation_history()
        else:
            # Users can only see their own
            history = get_conversation_history(user.get("username"))

        return JSONResponse({"history": history})
    except Exception as e:
        print(f"❌ Error in /api/history: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rag_model": "loaded"
    }


# ======================================
# MAIN ENTRY POINT
# ======================================
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🚀 Starting RAG Assistant (FastAPI)")
    print("=" * 60)
    print(f"📁 App Dir: {APP_DIR}")
    print(f"📁 Templates: {TEMPLATES_DIR}")
    print(f"📁 Static: {STATIC_DIR}")
    print("=" * 60)
    print("🌐 Access URLs:")
    print("  http://127.0.0.1:8001/login")
    print("=" * 60)

    # Test manuel de l'envoi d'email RH
    print("Test d'envoi d'email RH...")
    from datetime import datetime
    body = (
        "Bonjour équipe RH,\n\n"
        "Ceci est un test automatique de l'envoi d'alerte par email depuis le chatbot RH.\n\n"
        "Merci de ne pas tenir compte de ce message.\n\n"
        "Ceci est un message automatique envoyé le {date}.\n\n"
        "Cordialement,\nL'assistant RH automatique"
    ).format(date=datetime.now().strftime('%d/%m/%Y à %H:%M'))
    send_hr_notification(
        subject="[TEST] Alerte automatique Chatbot RH",
        body=body,
        to_emails=["taggaaikrame@gmail.com"]
    )

    print("Testing HR notification email...")
    send_hr_notification(
        subject="[TEST] Chatbot Notification",
        body="This is a test email from your FastAPI chatbot notification system.",
        to_emails=[os.getenv("SMTP_USER")]
    )

    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)