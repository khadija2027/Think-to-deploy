from starlette.middleware.sessions import SessionMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, Response
from fastapi import FastAPI, Request, Depends, HTTPException, Form
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import traceback

# ======================================
# PATH CONFIGURATION
# ======================================
APP_DIR = Path(__file__).parent.resolve()  # frontend_chatbot folder
PROJECT_ROOT = APP_DIR.parent.resolve()    # T2D directory

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

# ======================================
# IMPORT RAG MODEL
# ======================================
try:
    from test_rag import get_model_answer, get_metrics, get_conversation_stats
    print("✅ RAG model imported successfully")
except ImportError as e:
    print(f"❌ Failed to import RAG model: {e}")
    # Create dummy functions for testing

    def get_model_answer(question):
        return f"[Demo] Answer to: {question}"

    def get_metrics():
        return {}

    def get_conversation_stats():
        return {}

# ======================================
# AUTHENTICATION
# ======================================


def authenticate(username: str, password: str) -> Optional[dict]:
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
    user = request.session.get("user")
    if user:
        return user
    return {"username": "guest", "profil": "CDI", "nom_complet": "Guest User"}


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
# FIXED ROUTES
# ======================================


@app.get("/")
async def root():
    """Simple root endpoint that always works"""
    return {"message": "Assistant RH API is running", "endpoints": [
        "/login - Login page",
        "/chatbot - Chatbot interface",
        "/dashboard - Dashboard (RH only)",
        "/health - Health check",
        "/api/ask - Ask questions (POST)"
    ]}


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page - DIRECT ACCESS"""
    print("📄 Serving login page")
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login"""
    user = authenticate(username, password)
    if user:
        request.session["user"] = user
        print(f"✅ User logged in: {user['username']}")
        # 303 for POST redirect
        return RedirectResponse(url="/chatbot", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})


@app.get("/logout")
async def logout(request: Request):
    """Logout"""
    request.session.clear()
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

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "user": user, "metrics": metrics, "stats": stats}
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

        return JSONResponse({
            "answer": answer,
            "question": question,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ Error in /api/ask: {str(e)}")
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
# DEBUG ENDPOINTS
# ======================================


@app.get("/debug/templates")
async def debug_templates():
    """Debug: List available templates"""
    templates_list = []
    if TEMPLATES_DIR.exists():
        for file in TEMPLATES_DIR.glob("*.html"):
            templates_list.append(file.name)
    return {"templates": templates_list, "template_dir": str(TEMPLATES_DIR)}


@app.get("/debug/session")
async def debug_session(request: Request):
    """Debug: Show session data"""
    return {"session": request.session, "cookies": request.cookies}

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
    print("  http://localhost:8001")
    print("  http://127.0.0.1:8001")
    print("  http://127.0.0.1:8001/login  <-- START HERE")
    print("=" * 60)

    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
