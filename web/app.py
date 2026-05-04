import os
import uuid
import json
from urllib.parse import quote
import base64
import re
import subprocess
import tempfile

import hashlib
import binascii
import httpx
from fastapi import FastAPI, Request, Form, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from sqlalchemy import create_engine, Column, String, Text, ForeignKey, JSON

def hash_password(password: str) -> str:
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt, 100000)
    pwdhash = binascii.hexlify(pwdhash)
    return (salt + pwdhash).decode('ascii')

def verify_password(stored_password: str, provided_password: str) -> bool:
    if len(stored_password) < 64:
        return stored_password == provided_password # legacy fallback
    salt = stored_password[:64].encode('ascii')
    stored_hash = stored_password[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha512', provided_password.encode('utf-8'), salt, 100000)
    pwdhash = binascii.hexlify(pwdhash).decode('ascii')
    return pwdhash == stored_hash

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Database setup
DATABASE_DIR = "./data"
UPLOADS_DIR = "./data/uploads"
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/db.sqlite"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

AI_API_KEY = os.environ.get("AI_API_KEY", "")
AI_API_URL = os.environ.get("AI_API_URL", "")
AI_CHAT_MODEL = os.environ.get("AI_CHAT_MODEL", "")
AI_CODE_MODEL = os.environ.get("AI_CODE_MODEL", "")

class UserDB(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True, index=True)
    password = Column(String)
    email = Column(String, default="")

class ExperimentDB(Base):
    __tablename__ = "experiments"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.username"))
    name = Column(String)
    description = Column(Text)
    levels = Column(JSON)  # Nested list: [[node1], [node2, node3]]
    is_public = Column(Text, default="false") # Store as string "true"/"false" for sqlite simplicity

class NodeNotesDB(Base):
    __tablename__ = "node_notes"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    exp_id = Column(String, ForeignKey("experiments.id"))
    node_id = Column(String)
    notes_html = Column(Text, default="")
    chat_history = Column(JSON, default=list)
    annotation_url = Column(String, default="")

import datetime
import html as _html

class NodeVersionDB(Base):
    __tablename__ = "node_versions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    exp_id = Column(String)
    node_id = Column(String)
    html_snapshot = Column(Text, default="")
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())
    message = Column(String, default="")

class ProjectDB(Base):
    __tablename__ = "projects"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    user_id = Column(String, ForeignKey("users.username"))
    name = Column(String, default="")
    status = Column(String, default="Активный")  # Активный | Приостановлен | Завершён | Отменён

class ReagentDB(Base):
    __tablename__ = "reagents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    user_id = Column(String, ForeignKey("users.username"))
    name = Column(String, default="")
    supplier = Column(String, default="")
    category = Column(String, default="Жидкий")
    cas = Column(String, default="")
    formula = Column(String, default="")
    concentration = Column(String, default="")
    storage = Column(String, default="")
    amount = Column(String, default="0")
    min_amount = Column(String, default="0")
    unit = Column(String, default="мл")
    expiry = Column(String, default="")
    project = Column(String, default="")
    location = Column(String, default="")

class ConsumableDB(Base):
    __tablename__ = "consumables"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    user_id = Column(String, ForeignKey("users.username"))
    name = Column(String, default="")
    supplier = Column(String, default="")
    category = Column(String, default="Пластик")
    amount = Column(String, default="0")
    min_amount = Column(String, default="0")
    unit = Column(String, default="шт")
    location = Column(String, default="")
    comment = Column(String, default="")

Base.metadata.create_all(bind=engine)

from sqlalchemy import text

# Auto-migrate email column if missing
try:
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN email VARCHAR DEFAULT ''"))
except Exception as e:
    print(f"Migration notice: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()
app.mount("/static/uploads", StaticFiles(directory="data/uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

NOTEBOOKS_BASE = os.environ.get("NOTEBOOKS_PATH", "/workspace/notebooks")

# ── Auth ──

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    return templates.TemplateResponse(request, "index.html", {"error": error})

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if not user:
        return RedirectResponse(url="/?error=UserNotFound", status_code=302)
    if not verify_password(user.password, password):
        return RedirectResponse(url="/?error=WrongPassword", status_code=302)
    return RedirectResponse(url=f"/dashboard?user={username}", status_code=302)

@app.post("/register")
async def register(username: str = Form(...), email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if user:
        return RedirectResponse(url="/?error=UserExists", status_code=302)
    
    new_user = UserDB(username=username, email=email, password=hash_password(password))
    db.add(new_user)
    db.commit()
    return RedirectResponse(url=f"/dashboard?user={username}", status_code=302)

# ── Dashboard ──

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: str, db: Session = Depends(get_db)):
    user_exps = db.query(ExperimentDB).filter(ExperimentDB.user_id == user).all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": user,
            "experiments": user_exps,
        },
    )

# ── Projects page ──

def parse_amount(val):
    try:
        return float(val.replace(",", ".")) if val else 0.0
    except (ValueError, AttributeError):
        return 0.0

@app.get("/projects", response_class=HTMLResponse)
async def projects_page(request: Request, user: str, db: Session = Depends(get_db)):
    projects = db.query(ProjectDB).filter(ProjectDB.user_id == user).all()
    return templates.TemplateResponse(
        request, "projects.html",
        {"user": user, "projects": projects},
    )

# ── Reagents page ──

@app.get("/reagents", response_class=HTMLResponse)
async def reagents_page(request: Request, user: str, db: Session = Depends(get_db)):
    reagents = db.query(ReagentDB).filter(ReagentDB.user_id == user).all()
    low_reagents = sum(1 for r in reagents if parse_amount(r.amount) <= parse_amount(r.min_amount))
    return templates.TemplateResponse(
        request, "reagents.html",
        {"user": user, "reagents": reagents, "low_reagents": low_reagents},
    )

# ── Consumables page ──

@app.get("/consumables", response_class=HTMLResponse)
async def consumables_page(request: Request, user: str, db: Session = Depends(get_db)):
    consumables = db.query(ConsumableDB).filter(ConsumableDB.user_id == user).all()
    low_consumables = sum(1 for c in consumables if parse_amount(c.amount) <= parse_amount(c.min_amount))
    return templates.TemplateResponse(
        request, "consumables.html",
        {"user": user, "consumables": consumables, "low_consumables": low_consumables},
    )

# ── Experiments ──

@app.post("/create_experiment")
async def create_experiment(
    user: str = Form(...), name: str = Form(...), description: str = Form(...),
    db: Session = Depends(get_db)
):
    exp_id = str(uuid.uuid4())[:8]
    user_dir = os.path.join(NOTEBOOKS_BASE, user)
    exp_dir = os.path.join(user_dir, f"{name}_{exp_id}")
    os.makedirs(exp_dir, exist_ok=True)

    notebook_content = {
        "cells": [{"cell_type": "markdown", "id": "intro", "metadata": {},
                    "source": [f"# Эксперимент: {name}\n", f"**Описание:** {description}\n"]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5
    }
    with open(os.path.join(exp_dir, "experiment.ipynb"), "w") as f:
        json.dump(notebook_content, f)

    new_exp = ExperimentDB(
        id=exp_id, user_id=user, name=name, description=description,
        levels=[[{"id": "root", "title": "Начало исследования", "content": "Инициализация среды."}]]
    )
    db.add(new_exp)
    db.commit()
    return RedirectResponse(url=f"/dashboard?user={user}", status_code=302)

@app.get("/experiment/{user}/{exp_id}", response_class=HTMLResponse)
async def experiment_detail(request: Request, user: str, exp_id: str, db: Session = Depends(get_db)):
    exp = db.query(ExperimentDB).filter(ExperimentDB.id == exp_id, ExperimentDB.user_id == user).first()
    if not exp:
        return RedirectResponse(url=f"/dashboard?user={user}")
    return templates.TemplateResponse(
        request, "experiment_detail.html", {"user": user, "exp": exp}
    )

@app.post("/experiment/{user}/{exp_id}/add_node")
async def add_node(
    user: str, exp_id: str,
    title: str = Form(...), content: str = Form(...), level_idx: int = Form(-1),
    db: Session = Depends(get_db)
):
    exp = db.query(ExperimentDB).filter(ExperimentDB.id == exp_id, ExperimentDB.user_id == user).first()
    if exp:
        new_node = {"id": str(uuid.uuid4())[:8], "title": title, "content": content}
        levels = list(exp.levels)
        if level_idx == -1:
            levels.append([new_node])
        elif 0 <= level_idx < len(levels):
            level = list(levels[level_idx])
            level.append(new_node)
            levels[level_idx] = level
        exp.levels = levels
        db.commit()
    return RedirectResponse(url=f"/experiment/{user}/{exp_id}", status_code=302)

# ── Node detail page ──

def _find_node(exp, node_id):
    """Find a node dict by id inside the experiment levels."""
    for level in exp.levels:
        for node in level:
            if node["id"] == node_id:
                return node
    return None

@app.get("/experiment/{user}/{exp_id}/node/{node_id}", response_class=HTMLResponse)
async def node_detail(request: Request, user: str, exp_id: str, node_id: str, db: Session = Depends(get_db)):
    exp = db.query(ExperimentDB).filter(ExperimentDB.id == exp_id, ExperimentDB.user_id == user).first()
    if not exp:
        return RedirectResponse(url=f"/dashboard?user={user}")

    node = _find_node(exp, node_id)
    if not node:
        return RedirectResponse(url=f"/experiment/{user}/{exp_id}")

    # Get or create notes
    notes = db.query(NodeNotesDB).filter(NodeNotesDB.exp_id == exp_id, NodeNotesDB.node_id == node_id).first()
    if not notes:
        notes = NodeNotesDB(id=str(uuid.uuid4())[:8], exp_id=exp_id, node_id=node_id, notes_html="", chat_history=[])
        db.add(notes)
        db.commit()
        db.refresh(notes)

    return templates.TemplateResponse(
        request, "node_detail.html",
        {"user": user, "exp": exp, "node": node, "notes": notes}
    )

# ── API: save notes ──

@app.post("/api/node/save")
async def save_node_notes(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    exp_id = body.get("exp_id")
    node_id = body.get("node_id")
    html = body.get("html", "")
    annotation_url = body.get("annotation_url", "")

    notes = db.query(NodeNotesDB).filter(NodeNotesDB.exp_id == exp_id, NodeNotesDB.node_id == node_id).first()
    if notes:
        notes.notes_html = html
        if annotation_url:
            notes.annotation_url = annotation_url
        db.commit()
        return JSONResponse({"ok": True})
    return JSONResponse({"ok": False}, status_code=404)

# ── API: export node to DOCX ──

def _html_to_text_for_docx(s: str) -> str:
    if not s:
        return ""
    # Keep basic line breaks before stripping tags
    s = re.sub(r"(?i)<br\\s*/?>", "\n", s)
    s = re.sub(r"(?i)</p\\s*>", "\n\n", s)
    s = re.sub(r"(?i)</div\\s*>", "\n", s)
    s = re.sub(r"(?i)</li\\s*>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = _html.unescape(s)
    # Normalize whitespace/newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

@app.get("/api/node/export_docx")
async def export_node_docx(exp_id: str, node_id: str, db: Session = Depends(get_db)):
    notes = db.query(NodeNotesDB).filter(NodeNotesDB.exp_id == exp_id, NodeNotesDB.node_id == node_id).first()
    text = _html_to_text_for_docx((notes.notes_html if notes else "") or "")

    try:
        from docx import Document  # type: ignore
    except Exception:
        return JSONResponse({"ok": False, "error": "DOCX export unavailable (python-docx not installed)."}, status_code=501)

    doc = Document()
    if text:
        for block in text.split("\n\n"):
            doc.add_paragraph(block)
    else:
        doc.add_paragraph("")

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    fname = f"node_{node_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers,
    )

# ── API: Versioning (Git-like branches) ──

@app.post("/api/node/version/create")
async def create_version(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    exp_id = body.get("exp_id")
    node_id = body.get("node_id")
    message = body.get("message", "Commit")
    
    notes = db.query(NodeNotesDB).filter(NodeNotesDB.exp_id == exp_id, NodeNotesDB.node_id == node_id).first()
    if notes:
        version = NodeVersionDB(
            exp_id=exp_id, node_id=node_id, html_snapshot=notes.notes_html, message=message
        )
        db.add(version)
        db.commit()
        return JSONResponse({"ok": True, "version_id": version.id})
    return JSONResponse({"ok": False})

@app.get("/api/node/version/list")
async def list_versions(exp_id: str, node_id: str, db: Session = Depends(get_db)):
    versions = db.query(NodeVersionDB).filter(NodeVersionDB.exp_id == exp_id, NodeVersionDB.node_id == node_id).order_by(NodeVersionDB.created_at.desc()).all()
    return JSONResponse({"versions": [{"id": v.id, "created_at": v.created_at, "message": v.message} for v in versions]})

@app.get("/api/node/version/{version_id}")
async def get_version_snapshot(version_id: str, db: Session = Depends(get_db)):
    v = db.query(NodeVersionDB).filter(NodeVersionDB.id == version_id).first()
    if not v:
        return JSONResponse({"ok": False}, status_code=404)
    return JSONResponse(
        {
            "id": v.id,
            "created_at": v.created_at,
            "message": v.message,
            "html_snapshot": v.html_snapshot or "",
        }
    )

@app.post("/api/node/version/rollback")
async def rollback_version(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    version_id = body.get("version_id")
    
    version = db.query(NodeVersionDB).filter(NodeVersionDB.id == version_id).first()
    if version:
        notes = db.query(NodeNotesDB).filter(NodeNotesDB.exp_id == version.exp_id, NodeNotesDB.node_id == version.node_id).first()
        if notes:
            notes.notes_html = version.html_snapshot
            db.commit()
            return JSONResponse({"ok": True})
    return JSONResponse({"ok": False})

# ── API: STT ──

@app.post("/api/stt")
async def run_stt(file: UploadFile = File(...)):
    """
    Server-side transcription (works in Firefox too).
    Uses faster-whisper if available. Expects audio in webm/wav/m4a/etc.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return JSONResponse(
            {"text": "Транскрибация недоступна на сервере (не установлен движок распознавания)."},
            status_code=501,
        )

    data = await file.read()
    if not data:
        return JSONResponse({"text": "Пустой аудиофайл."}, status_code=400)

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.webm")
        with open(in_path, "wb") as f:
            f.write(data)

        # Lazy global model cache
        global _whisper_model
        try:
            _whisper_model
        except NameError:
            _whisper_model = None
        if _whisper_model is None:
            _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

        # First try decoding original container (works if PyAV supports it)
        try:
            segments, _info = _whisper_model.transcribe(in_path, language="ru")
        except Exception:
            # Fallback: convert to wav via ffmpeg if available
            out_path = os.path.join(td, "out.wav")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", out_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                return JSONResponse(
                    {"text": "Не удалось преобразовать аудио. Установите ffmpeg или используйте поддерживаемый формат."},
                    status_code=500,
                )
            segments, _info = _whisper_model.transcribe(out_path, language="ru")
        text = " ".join([(s.text or "").strip() for s in segments]).strip()
        if not text:
            text = "Не удалось распознать речь. Попробуйте говорить ближе к микрофону."
        return {"text": text}

# ── API: upload image ──

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1] if "." in file.filename else "bin"
    fname = f"{uuid.uuid4().hex[:12]}.{ext}"
    path = os.path.join(UPLOADS_DIR, fname)
    
    # Stream for better stability
    with open(path, "wb") as f:
        while content := await file.read(1024 * 1024):  # 1MB chunks
            f.write(content)
            
    return JSONResponse({"url": f"/static/uploads/{fname}"})

# ── API: AI chat ──

@app.post("/api/chat")
async def chat_with_ai(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    exp_id = body.get("exp_id")
    node_id = body.get("node_id")
    user_message = body.get("message", "")

    html_context = body.get("html_context", "")

    notes = db.query(NodeNotesDB).filter(NodeNotesDB.exp_id == exp_id, NodeNotesDB.node_id == node_id).first()
    history = list(notes.chat_history) if notes and notes.chat_history else []

    system_prompt = (
        "Ты — ассистент-учёный для биоинформатики. Помогай с анализом данных, кодом и результатами. "
        "Пользователь может попросить тебя внести изменения в свой документ (добавить таблицу, исправить текст). "
        "В этом случае, вместо обычного ответа в чат, ты должен вернуть ПОЛНЫЙ новый HTML код документа, обернутый в теги [EDIT_DOCUMENT] и [/EDIT_DOCUMENT]. "
        "Если пользователь НЕ просит изменять документ, просто отвечай как обычно."
    )
    if html_context:
        system_prompt += f"\n\nТекущий HTML документа:\n{html_context}"

    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-20:]:  # Keep recent history
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_message})

    if not AI_API_KEY or not AI_API_URL or not AI_CHAT_MODEL:
        assistant_msg = "AI‑ассистент не настроен на сервере (нет переменных окружения AI_API_KEY/AI_API_URL/AI_CHAT_MODEL)."
    else:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    AI_API_URL,
                    headers={
                        "Authorization": f"Bearer {AI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": AI_CHAT_MODEL,
                        "messages": messages,
                        "max_tokens": 2048,
                    },
                )
                data = resp.json()
                assistant_msg = data["choices"][0]["message"]["content"]
        except Exception as e:
            assistant_msg = f"Ошибка AI API: {str(e)}"

    edit_html = None
    if "[EDIT_DOCUMENT]" in assistant_msg and "[/EDIT_DOCUMENT]" in assistant_msg:
        start_idx = assistant_msg.find("[EDIT_DOCUMENT]") + len("[EDIT_DOCUMENT]")
        end_idx = assistant_msg.find("[/EDIT_DOCUMENT]")
        edit_html = assistant_msg[start_idx:end_idx].strip()
        assistant_msg = "Я внес изменения в документ согласно вашему запросу."
    
    # Save to history
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_msg})
    if notes:
        notes.chat_history = history
        db.commit()

    return JSONResponse({"reply": assistant_msg, "edit_html": edit_html})

import sys
import io
from contextlib import redirect_stdout

# Per-experiment kernel states (shared variables like in Jupyter)
kernels = {}

@app.post("/api/kernel/execute")
async def execute_code(request: Request):
    body = await request.json()
    exp_id = body.get("exp_id")
    code = body.get("code")

    if exp_id not in kernels:
        kernels[exp_id] = {"__builtins__": __builtins__}

    f = io.StringIO()
    error = None
    try:
        with redirect_stdout(f):
            # Run code in the experiment's specific namespace
            exec(code, kernels[exp_id])
        output = f.getvalue()
    except Exception as e:
        output = str(e)
        error = True

    return JSONResponse({"output": output, "error": error})

@app.post("/api/kernel/autocomplete")
async def autocomplete_code(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    code_context = body.get("code", "")
    
    # Ask AI provider to complete code given context.
    prompt = f"Заверши следующий код на Python. Верни ТОЛЬКО сам код, без форматирования markdown и пояснений. Код должен начинаться с того места, где оборвался контекст:\n\n{code_context}"
    
    if not AI_API_KEY or not AI_API_URL or not AI_CODE_MODEL:
        completion = ""
    else:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    AI_API_URL,
                    headers={
                        "Authorization": f"Bearer {AI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": AI_CODE_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 150,
                        "temperature": 0.2,
                    },
                )
                data = resp.json()
                completion = data["choices"][0]["message"]["content"].strip()
                # Remove markdown backticks if any
                if completion.startswith("```python"):
                    completion = completion.replace("```python", "").replace("```", "").strip()
                elif completion.startswith("```"):
                    completion = completion.replace("```", "").strip()
        except Exception as e:
            completion = ""

    return JSONResponse({"completion": completion})

@app.post("/api/ocr")
async def run_ocr(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    image_b64 = body.get("image_b64", "")
    exp_id = body.get("exp_id")
    node_id = body.get("node_id")

    if not image_b64:
        return JSONResponse({"text": "Не удалось получить данные изображения."})

    # Prefer local OCR (works reliably without "vision" model limitations)
    try:
        from PIL import Image
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return JSONResponse({"text": "OCR недоступен: не установлены зависимости (Pillow/rapidocr-onnxruntime)."}, status_code=501)

    # data URL like: data:image/jpeg;base64,....
    m = re.match(r"^data:image/[^;]+;base64,(.*)$", image_b64)
    if not m:
        return JSONResponse({"text": "Неверный формат изображения."}, status_code=400)

    raw = base64.b64decode(m.group(1))
    with tempfile.TemporaryDirectory() as td:
        img_path = os.path.join(td, "img.png")
        with open(img_path, "wb") as f:
            f.write(raw)
        try:
            Image.open(img_path).verify()
        except Exception:
            return JSONResponse({"text": "Не удалось прочитать изображение."}, status_code=400)

        ocr = RapidOCR()
        result, _elapsed = ocr(img_path)
        if not result:
            return JSONResponse({"text": "Текст не найден."})

        # result: list of [box, text, score]
        lines = []
        for item in result:
            try:
                text = (item[1] or "").strip()
                if text:
                    lines.append(text)
            except Exception:
                continue
        text_out = "\n".join(lines).strip() or "Текст не найден."
        return JSONResponse({"text": text_out})

# ── API: Reagents ──

@app.get("/api/reagents")
async def get_reagents(user: str, db: Session = Depends(get_db)):
    items = db.query(ReagentDB).filter(ReagentDB.user_id == user).all()
    return JSONResponse({"items": [{"id":r.id,"name":r.name,"supplier":r.supplier,"category":r.category,"cas":r.cas,"formula":r.formula,"concentration":r.concentration,"storage":r.storage,"amount":r.amount,"min_amount":r.min_amount,"unit":r.unit,"expiry":r.expiry,"project":r.project,"location":r.location} for r in items]})

@app.post("/api/reagents")
async def create_reagent(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    r = ReagentDB(user_id=body.get("user"), name=body.get("name",""), supplier=body.get("supplier",""), category=body.get("category","Жидкий"), cas=body.get("cas",""), formula=body.get("formula",""), concentration=body.get("concentration",""), storage=body.get("storage",""), amount=body.get("amount","0"), min_amount=body.get("min_amount","0"), unit=body.get("unit","мл"), expiry=body.get("expiry",""), project=body.get("project",""), location=body.get("location",""))
    db.add(r)
    db.commit()
    return JSONResponse({"ok": True, "id": r.id})

@app.put("/api/reagents/{item_id}")
async def update_reagent(item_id: str, request: Request, db: Session = Depends(get_db)):
    r = db.query(ReagentDB).filter(ReagentDB.id == item_id).first()
    if not r:
        return JSONResponse({"ok": False}, status_code=404)
    body = await request.json()
    r.name = body.get("name", r.name)
    r.supplier = body.get("supplier", r.supplier)
    r.category = body.get("category", r.category)
    r.cas = body.get("cas", r.cas)
    r.formula = body.get("formula", r.formula)
    r.concentration = body.get("concentration", r.concentration)
    r.storage = body.get("storage", r.storage)
    r.amount = body.get("amount", r.amount)
    r.min_amount = body.get("min_amount", r.min_amount)
    r.unit = body.get("unit", r.unit)
    r.expiry = body.get("expiry", r.expiry)
    r.project = body.get("project", r.project)
    r.location = body.get("location", r.location)
    db.commit()
    return JSONResponse({"ok": True})

@app.delete("/api/reagents/{item_id}")
async def delete_reagent(item_id: str, db: Session = Depends(get_db)):
    r = db.query(ReagentDB).filter(ReagentDB.id == item_id).first()
    if r:
        db.delete(r)
        db.commit()
    return JSONResponse({"ok": True})

# ── API: Consumables ──

@app.get("/api/consumables")
async def get_consumables(user: str, db: Session = Depends(get_db)):
    items = db.query(ConsumableDB).filter(ConsumableDB.user_id == user).all()
    return JSONResponse({"items": [{"id":c.id,"name":c.name,"supplier":c.supplier,"category":c.category,"amount":c.amount,"min_amount":c.min_amount,"unit":c.unit,"location":c.location,"comment":c.comment} for c in items]})

@app.post("/api/consumables")
async def create_consumable(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    c = ConsumableDB(user_id=body.get("user"), name=body.get("name",""), supplier=body.get("supplier",""), category=body.get("category","Пластик"), amount=body.get("amount","0"), min_amount=body.get("min_amount","0"), unit=body.get("unit","шт"), location=body.get("location",""), comment=body.get("comment",""))
    db.add(c)
    db.commit()
    return JSONResponse({"ok": True, "id": c.id})

@app.put("/api/consumables/{item_id}")
async def update_consumable(item_id: str, request: Request, db: Session = Depends(get_db)):
    c = db.query(ConsumableDB).filter(ConsumableDB.id == item_id).first()
    if not c:
        return JSONResponse({"ok": False}, status_code=404)
    body = await request.json()
    c.name = body.get("name", c.name)
    c.supplier = body.get("supplier", c.supplier)
    c.category = body.get("category", c.category)
    c.amount = body.get("amount", c.amount)
    c.min_amount = body.get("min_amount", c.min_amount)
    c.unit = body.get("unit", c.unit)
    c.location = body.get("location", c.location)
    c.comment = body.get("comment", c.comment)
    db.commit()
    return JSONResponse({"ok": True})

@app.delete("/api/consumables/{item_id}")
async def delete_consumable(item_id: str, db: Session = Depends(get_db)):
    c = db.query(ConsumableDB).filter(ConsumableDB.id == item_id).first()
    if c:
        db.delete(c)
        db.commit()
    return JSONResponse({"ok": True})

# ── API: Projects ──

@app.get("/api/projects")
async def get_projects(user: str, db: Session = Depends(get_db)):
    items = db.query(ProjectDB).filter(ProjectDB.user_id == user).all()
    return JSONResponse({"items": [{"id": p.id, "name": p.name, "status": p.status} for p in items]})

@app.post("/api/projects")
async def create_project(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    p = ProjectDB(
        user_id=body.get("user"),
        name=body.get("name", ""),
        status=body.get("status", "Активный"),
    )
    db.add(p)
    db.commit()
    return JSONResponse({"ok": True, "id": p.id})

@app.delete("/api/projects/{item_id}")
async def delete_project(item_id: str, db: Session = Depends(get_db)):
    p = db.query(ProjectDB).filter(ProjectDB.id == item_id).first()
    if p:
        db.delete(p)
        db.commit()
    return JSONResponse({"ok": True})

@app.post("/experiment/{user}/{exp_id}/toggle_public")
async def toggle_public(user: str, exp_id: str, db: Session = Depends(get_db)):
    exp = db.query(ExperimentDB).filter(ExperimentDB.id == exp_id, ExperimentDB.user_id == user).first()
    if exp:
        exp.is_public = "true" if exp.is_public == "false" else "false"
        db.commit()
    return RedirectResponse(url=f"/experiment/{user}/{exp_id}", status_code=302)

@app.get("/share/{exp_id}", response_class=HTMLResponse)
async def public_experiment(request: Request, exp_id: str, db: Session = Depends(get_db)):
    exp = db.query(ExperimentDB).filter(ExperimentDB.id == exp_id, ExperimentDB.is_public == "true").first()
    if not exp:
        return HTMLResponse("Эксперимент не найден или не является публичным", status_code=404)
    
    return templates.TemplateResponse(
        request, "experiment_detail.html", {"user": exp.user_id, "exp": exp, "is_public_view": True}
    )

@app.get("/share/{exp_id}/node/{node_id}", response_class=HTMLResponse)
async def public_node_detail(request: Request, exp_id: str, node_id: str, db: Session = Depends(get_db)):
    exp = db.query(ExperimentDB).filter(ExperimentDB.id == exp_id, ExperimentDB.is_public == "true").first()
    if not exp:
        return HTMLResponse("Эксперимент не найден или не является публичным", status_code=404)
    
    node = _find_node(exp, node_id)
    if not node:
        return HTMLResponse("Узел не найден", status_code=404)
    
    notes = db.query(NodeNotesDB).filter(NodeNotesDB.exp_id == exp_id, NodeNotesDB.node_id == node_id).first()
    return templates.TemplateResponse(
        request, "node_detail.html",
        {"user": exp.user_id, "exp": exp, "node": node, "notes": notes, "is_public_view": True}
    )
