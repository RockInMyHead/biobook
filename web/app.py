import os
import uuid
import json
import time
import hmac
import secrets
import threading
from urllib.parse import quote, urlencode, urlparse, parse_qs, unquote_plus
import base64
import re
import subprocess
import tempfile

import hashlib
import binascii
import httpx
from fastapi import FastAPI, Request, Form, Depends, UploadFile, File, Query, HTTPException
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, Response, FileResponse
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from sqlalchemy import create_engine, Column, String, Text, ForeignKey, JSON, DateTime, Integer, func, distinct

def is_russian_email(email: str) -> bool:
    """Check if email domain is a Russian one."""
    domain = email.lower().split("@")[-1] if "@" in email else ""
    russian_domains = {".ru", ".su", ".рф", "xn--p1ai"}
    return any(domain.endswith(d) for d in russian_domains)


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
KERNEL_OUTPUT_ROOT = os.path.join(DATABASE_DIR, "kernel_outputs")
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(KERNEL_OUTPUT_ROOT, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/db.sqlite"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

AI_API_KEY = os.environ.get("AI_API_KEY", "")
AI_API_URL = os.environ.get("AI_API_URL", "")
AI_CHAT_MODEL = os.environ.get("AI_CHAT_MODEL", "")
AI_CODE_MODEL = os.environ.get("AI_CODE_MODEL", "")
AI_HTTP_PROXY = os.environ.get("AI_HTTP_PROXY", "").strip()
# Prefer dedicated OCR key. Do not fall back to DeepSeek/chat keys for OpenRouter vision.
_OCR_KEY_RAW = (os.environ.get("OCR_API_KEY", "") or "").strip()
_OCR_URL_RAW = (os.environ.get("OCR_API_URL", "") or "").strip()
OCR_API_URL = _OCR_URL_RAW or "https://openrouter.ai/api/v1/chat/completions"
if _OCR_KEY_RAW:
    OCR_API_KEY = _OCR_KEY_RAW
elif AI_API_KEY and "openrouter.ai" not in OCR_API_URL.lower():
    # Same OpenAI-compatible endpoint as chat (rare); otherwise leave empty → local OCR.
    OCR_API_KEY = AI_API_KEY.strip()
elif AI_API_KEY.startswith("sk-or-"):
    OCR_API_KEY = AI_API_KEY.strip()
else:
    OCR_API_KEY = ""
DEFAULT_OCR_VISION_MODEL = "qwen/qwen-2.5-vl-7b-instruct"
OCR_VISION_MODEL = (
    os.environ.get("OCR_VISION_MODEL", DEFAULT_OCR_VISION_MODEL).strip()
    or DEFAULT_OCR_VISION_MODEL
)
OCR_VISION_MODELS_EXTRA = os.environ.get("OCR_VISION_MODELS", "").strip()
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "").strip()  # comma-separated
TG_HTTP_PROXY = (
    os.environ.get("TG_HTTP_PROXY", "").strip()
    or os.environ.get("AI_HTTP_PROXY", "").strip()
    or os.environ.get("HTTPS_PROXY", "").strip()
    or os.environ.get("HTTP_PROXY", "").strip()
)
TG_CHATS_FILE = os.path.join(DATABASE_DIR, "tg_chats.json")
_TG_LOCK = threading.Lock()
_TG_AI_FAIL_LAST: dict[str, float] = {}
_TG_AI_FAIL_COOLDOWN_SEC = 300
_TG_POLLER_STARTED = False


def _ocr_vision_model_candidates() -> list[str]:
    """Primary from env, then alternates (Claude → Qwen VL → Llama Vision → OpenAI)."""
    if OCR_VISION_MODELS_EXTRA:
        fallbacks = tuple(s.strip() for s in OCR_VISION_MODELS_EXTRA.split(",") if s.strip())
    else:
        fallbacks = (
            "qwen/qwen-2.5-vl-7b-instruct",
            "qwen/qwen-2.5-vl-72b-instruct",
            "meta-llama/llama-3.2-11b-vision-instruct",
            "anthropic/claude-3-haiku",
            "openai/gpt-4o-mini",
        )
    out: list[str] = []
    for m in (OCR_VISION_MODEL, *fallbacks):
        if m and m not in out:
            out.append(m)
    return out


def _ocr_should_try_next_model(err: RuntimeError) -> bool:
    msg = str(err).lower()
    return any(
        token in msg
        for token in (
            "404",
            "403",
            "429",
            "no endpoints",
            "not found",
            "invalid model",
            "terms of service",
            "prohibited",
            "unsupported",
            "capacity",
            "overloaded",
        )
    )


def _ocr_prepare_image_data_url(image_data_url: str) -> str:
    """Downscale very large images before vision OCR."""
    m = re.match(r"^(data:image/[^;]+;base64,)(.*)$", image_data_url, re.DOTALL)
    if not m:
        return image_data_url
    prefix, b64 = m.group(1), m.group(2)
    try:
        from PIL import Image
        import io

        raw = base64.b64decode(b64)
        if len(raw) <= 900_000:
            return image_data_url
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        max_side = 2048
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return image_data_url


def ai_http_client(timeout: float = 60.0) -> httpx.AsyncClient:
    kwargs: dict = {"timeout": timeout}
    if AI_HTTP_PROXY:
        kwargs["proxy"] = AI_HTTP_PROXY
    return httpx.AsyncClient(**kwargs)


def format_ai_request_error(exc: Exception) -> str:
    if isinstance(exc, httpx.ConnectTimeout):
        return (
            "Не удалось подключиться к AI API (таймаут). С сервера VK Cloud часто недоступен "
            "api.deepseek.com — укажите AI_HTTP_PROXY в /etc/biobook.env или смените провайдера."
        )
    if isinstance(exc, httpx.ReadTimeout):
        return "AI API не ответил вовремя. Попробуйте короче запрос или позже."
    if isinstance(exc, httpx.ConnectError):
        return (
            "Сеть до AI API недоступна с сервера. Проверьте исходящий HTTPS или задайте AI_HTTP_PROXY."
        )
    msg = str(exc).strip()
    return f"Oшибка AI API: {msg or exc.__class__.__name__}"


async def _ocr_via_vision_model(image_data_url: str, model: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all visible text from this notebook or document photo (handwriting, print, tables). "
                            "Return plain text only; keep paragraphs and table row order. No markdown or commentary."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    headers = {
        "Authorization": f"Bearer {OCR_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://anticus.tech",
        "X-Title": "Antikus Bio-Notebook OCR",
    }
    async with ai_http_client(timeout=120) as client:
        resp = await client.post(OCR_API_URL, headers=headers, json=payload)
        try:
            data = resp.json()
        except Exception:
            data = {"error": (resp.text or "")[:400]}
        if resp.status_code >= 400:
            err = ""
            if isinstance(data, dict):
                err = (
                    (data.get("error") or {}).get("message")
                    if isinstance(data.get("error"), dict)
                    else data.get("error")
                ) or str(data.get("message", ""))
            raise RuntimeError(f"{model} ({resp.status_code}): {err or 'ошибка API'}")
        text = ""
        if isinstance(data, dict):
            text = (
                (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            ).strip()
        if not text:
            raise RuntimeError(f"{model}: пустой ответ")
        return text


async def ocr_via_vision_api(image_data_url: str) -> str:
    """Recognize text in an image via OpenRouter (OpenAI-compatible vision chat)."""
    if not OCR_API_KEY:
        raise RuntimeError("OCR_API_KEY не задан на сервере.")
    prepared = _ocr_prepare_image_data_url(image_data_url)
    last_err: Exception | None = None
    for model in _ocr_vision_model_candidates():
        try:
            return await _ocr_via_vision_model(prepared, model)
        except RuntimeError as e:
            last_err = e
            if _ocr_should_try_next_model(e):
                continue
            raise
    raise RuntimeError(
        f"OpenRouter OCR: ни одна vision-модель не приняла запрос. Последняя ошибка: {last_err}"
    )


BASE_PATH = os.environ.get("BASE_PATH", "/bio-notebook").rstrip("/")
# Set EXPERIMENT_GRAPH_ENABLED=1 in /etc/biobook.env to show «Граф связей» again.
EXPERIMENT_GRAPH_ENABLED = os.environ.get("EXPERIMENT_GRAPH_ENABLED", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
LAB_PLANNER_ENABLED = os.environ.get("LAB_PLANNER_ENABLED", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ADMIN_KEY = os.environ.get("ADMIN_KEY", "admin123")
LAB_ADMIN_PATH = (os.environ.get("LAB_ADMIN_PATH", "/lab-admin") or "/lab-admin").rstrip("/")
if not LAB_ADMIN_PATH.startswith("/"):
    LAB_ADMIN_PATH = "/" + LAB_ADMIN_PATH
SITE_ORIGIN = os.environ.get("SITE_ORIGIN", "https://anticus.tech").rstrip("/")
YANDEX_CLIENT_ID = os.environ.get("YANDEX_CLIENT_ID", "")
YANDEX_CLIENT_SECRET = os.environ.get("YANDEX_CLIENT_SECRET", "")
OAUTH_STATE_SECRET = os.environ.get("OAUTH_STATE_SECRET", ADMIN_KEY)
LEGAL_BASE_URL = os.environ.get("LEGAL_BASE_URL", SITE_ORIGIN).rstrip("/")
SESSION_COOKIE = "biobook_sid"
SESSION_DAYS = 30
SESSION_SECRET = os.environ.get("SESSION_SECRET") or OAUTH_STATE_SECRET
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() != "false"
VISITOR_COOKIE = "bio_vid"
VISITOR_COOKIE_MAX_AGE = 365 * 24 * 3600


def make_session_token(username: str) -> str:
    exp = int(time.time()) + SESSION_DAYS * 86400
    payload = f"{username}|{exp}"
    sig = hmac.new(SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    raw = f"{payload}|{sig}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def parse_session_token(token: str | None) -> str | None:
    if not token:
        return None
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        username, exp_s, sig = raw.rsplit("|", 2)
        payload = f"{username}|{exp_s}"
        expected = hmac.new(SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(sig, expected):
            return None
        if int(exp_s) < int(time.time()):
            return None
        if not re.match(r"^[\w.-]{1,32}$", username):
            return None
        return username
    except Exception:
        return None


def get_session_username(request: Request) -> str | None:
    return parse_session_token(request.cookies.get(SESSION_COOKIE))


def attach_session_cookie(response: RedirectResponse, username: str) -> RedirectResponse:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=make_session_token(username),
        max_age=SESSION_DAYS * 86400,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )
    return response


def clear_session_cookie(response: RedirectResponse) -> RedirectResponse:
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


def redirect_to_dashboard(username: str) -> RedirectResponse:
    resp = RedirectResponse(url=f"{BASE_PATH}/dashboard?user={username}", status_code=302)
    return attach_session_cookie(resp, username)


def path_is_public(path: str) -> bool:
    if path in ("/", "/favicon.ico"):
        return True
    if path.startswith("/static"):
        return True
    if path.startswith("/auth/"):
        return True
    if path.startswith("/share/"):
        return True
    if path.startswith("/admin") or path.startswith(LAB_ADMIN_PATH):
        return True
    if path.startswith("/invite"):
        return True
    if path in ("/login", "/register", "/logout"):
        return True
    if path == "/guides" or path.startswith("/guides/"):
        return True
    for slug in MARKETING_PAGES:
        if path in (f"/{slug}", f"/{slug}/"):
            return True
    if path == "/sitemap-guides.xml":
        return True
    if path == "/api/labos-lead":
        return True
    if path == "/api/tg/webhook":
        return True
    return False


def yandex_oauth_enabled() -> bool:
    return bool(YANDEX_CLIENT_ID and YANDEX_CLIENT_SECRET)


def yandex_redirect_uri() -> str:
    return f"{SITE_ORIGIN}{BASE_PATH}/auth/yandex/callback"


def sign_oauth_state(data: dict) -> str:
    payload = json.dumps(data, separators=(",", ":"), sort_keys=True)
    sig = hmac.new(OAUTH_STATE_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}|{sig}".encode()).decode()


def verify_oauth_state(state: str, max_age: int = 900) -> dict | None:
    try:
        decoded = base64.urlsafe_b64decode(state.encode()).decode()
        payload, sig = decoded.rsplit("|", 1)
        expected = hmac.new(OAUTH_STATE_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        data = json.loads(payload)
        if time.time() - float(data.get("ts", 0)) > max_age:
            return None
        return data
    except Exception:
        return None


def sanitize_username(raw: str, fallback: str) -> str:
    cleaned = re.sub(r"[^\w.-]", "", (raw or "").lower())[:32]
    return cleaned or fallback


def unique_username(db: Session, base: str) -> str:
    base = sanitize_username(base, "user")[:28] or "user"
    candidate = base
    n = 1
    while db.query(UserDB).filter(UserDB.username == candidate).first():
        suffix = str(n)
        candidate = f"{base[: 32 - len(suffix)]}{suffix}"
        n += 1
        if n > 9999:
            return f"user_{uuid.uuid4().hex[:8]}"
    return candidate


def username_from_email(email: str) -> str:
    local = (email or "").split("@", 1)[0]
    return sanitize_username(local, f"user_{uuid.uuid4().hex[:6]}")


def find_user_by_login(db: Session, login: str):
    """Resolve login by username or email (case-insensitive for email)."""
    login = (login or "").strip()
    if not login:
        return None
    user = db.query(UserDB).filter(UserDB.username == login).first()
    if user:
        return user
    email = login.lower()
    return (
        db.query(UserDB)
        .filter(func.lower(UserDB.email) == email)
        .first()
    )


def upload_public_url(fname: str) -> str:
    return f"{BASE_PATH}/static/uploads/{fname}"


def normalize_upload_urls(html: str) -> str:
    """Rewrite legacy upload paths so images load under BASE_PATH (e.g. /bio-notebook)."""
    if not html:
        return html
    prefix = BASE_PATH or ""
    html = re.sub(
        r'src="(?!https?://|data:|/)((?:[a-f0-9]{8,12})\.(?:png|jpg|jpeg|gif|webp))"',
        lambda m: f'src="{prefix}/static/uploads/{m.group(1)}"',
        html,
        flags=re.I,
    )
    if prefix:
        html = html.replace('src="/static/uploads/', f'src="{prefix}/static/uploads/')
    return html



class BasePathMiddleware:
    def __init__(self, app, base_path: str):
        self.app = app
        self.base_path = base_path

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not self.base_path:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path == self.base_path or path.startswith(self.base_path + "/"):
            scope = dict(scope)
            scope["root_path"] = self.base_path
            scope["path"] = path[len(self.base_path):] or "/"

        await self.app(scope, receive, send)


class UserDB(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True, index=True)
    password = Column(String)
    email = Column(String, default="")
    last_login = Column(String, default="")
    is_banned = Column(String, default="false")
    yandex_id = Column(String, default="")
    display_name = Column(String, default="")
    auth_provider = Column(String, default="password")
    agree_marketing = Column(String, default="false")
    plan = Column(String, default="free")  # free | student | corp
    plan_expires_at = Column(String, default="")  # ISO; paid/trial end → downgrade to free

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
    description = Column(Text, default="")
    end_date = Column(String, default="")

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

class ProjectTaskDB(Base):
    __tablename__ = "project_tasks"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    project_id = Column(String, ForeignKey("projects.id"))
    title = Column(String, default="")
    description = Column(Text, default="")
    assignee = Column(String, default="")
    status = Column(String, default="Открыто")
    priority = Column(String, default="Средний")
    start_date = Column(String, default="")
    end_date = Column(String, default="")
    created_by = Column(String, default="")
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())

class ProjectMemberDB(Base):
    __tablename__ = "project_members"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    project_id = Column(String, ForeignKey("projects.id"))
    username = Column(String, ForeignKey("users.username"))

class ProjectInviteDB(Base):
    __tablename__ = "project_invites"
    code = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:12])
    project_id = Column(String, ForeignKey("projects.id"))
    created_by = Column(String, default="")
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())

class FriendDB(Base):
    __tablename__ = "friends"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    user_id = Column(String, ForeignKey("users.username"))
    friend_id = Column(String, ForeignKey("users.username"))
    status = Column(String, default="pending")  # pending | accepted | rejected
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())

class EquipmentDB(Base):
    __tablename__ = "equipment"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    owner_id = Column(String, ForeignKey("users.username"))
    name = Column(String, default="")
    description = Column(Text, default="")
    status = Column(String, default="available")  # available | in_use | maintenance
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())

class EquipmentBookingDB(Base):
    __tablename__ = "equipment_bookings"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    equipment_id = Column(String, ForeignKey("equipment.id"))
    user_id = Column(String, ForeignKey("users.username"))
    start_date = Column(String, default="")
    end_date = Column(String, default="")
    status = Column(String, default="pending")  # pending | approved | declined | completed
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())

class SupportTicketDB(Base):
    __tablename__ = "support_tickets"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    user_id = Column(String, ForeignKey("users.username"))
    subject = Column(String, default="")
    message = Column(Text, default="")
    status = Column(String, default="open")  # open | in_progress | resolved | closed
    priority = Column(String, default="medium")  # low | medium | high | critical
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())
    updated_at = Column(String, default=lambda: datetime.datetime.now().isoformat())

class SupportReplyDB(Base):
    __tablename__ = "support_replies"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    ticket_id = Column(String, ForeignKey("support_tickets.id"))
    author = Column(String, default="")  # user_id or 'admin'
    message = Column(Text, default="")
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())

class AuditLogDB(Base):
    __tablename__ = "audit_logs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    user_id = Column(String, default="")
    action = Column(String, default="")  # create | update | delete | login
    entity_type = Column(String, default="")  # experiment | project | friend | equipment | user
    entity_id = Column(String, default="")
    details = Column(Text, default="")  # JSON
    ip_address = Column(String, default="")
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())

class OrganizationDB(Base):
    __tablename__ = "organizations"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    name = Column(String, default="")
    owner_id = Column(String, ForeignKey("users.username"))
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())

class OrganizationMemberDB(Base):
    __tablename__ = "organization_members"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    org_id = Column(String, ForeignKey("organizations.id"))
    user_id = Column(String, ForeignKey("users.username"))
    role = Column(String, default="member")  # admin | member | viewer

class ReferralLinkDB(Base):
    __tablename__ = "referral_links"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    user_id = Column(String, ForeignKey("users.username"))
    code = Column(String, unique=True, default="")
    clicks = Column(String, default="0")
    registrations = Column(String, default="0")
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())

# ── Friend Chats ──

class FriendChatDB(Base):
    __tablename__ = "friend_chats"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    user1 = Column(String, ForeignKey("users.username"))
    user2 = Column(String, ForeignKey("users.username"))
    last_message = Column(String, default="")
    last_sender = Column(String, default="")
    last_time = Column(String, default="")
    user1_last_read_at = Column(String, default="")
    user2_last_read_at = Column(String, default="")
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())


class FriendMessageDB(Base):
    __tablename__ = "friend_messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    chat_id = Column(String, ForeignKey("friend_chats.id"))
    sender = Column(String, ForeignKey("users.username"))
    content = Column(Text, default="")
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())


# ── Wiki & File Exchange ──

class WikiPageDB(Base):
    __tablename__ = "wiki_pages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    exp_id = Column(String, ForeignKey("experiments.id"))
    title = Column(String, default="")
    content = Column(Text, default="")
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())
    updated_at = Column(String, default=lambda: datetime.datetime.now().isoformat())
    created_by = Column(String, ForeignKey("users.username"))


class ExperimentFileDB(Base):
    __tablename__ = "experiment_files"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    exp_id = Column(String, ForeignKey("experiments.id"))
    filename = Column(String, default="")
    original_name = Column(String, default="")
    file_size = Column(String, default="0")
    mime_type = Column(String, default="")
    uploaded_by = Column(String, ForeignKey("users.username"))
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())


class ServiceVisitDB(Base):
    """Анонимные визиты на HTML-страницы (cookie bio_vid)."""
    __tablename__ = "service_visits"
    id = Column(Integer, primary_key=True, autoincrement=True)
    visitor_id = Column(String(64), index=True, nullable=False)
    path = Column(String(512), default="")
    referrer = Column(String(1024), default="")
    referrer_host = Column(String(255), default="")
    source = Column(String(255), default="")
    search_query = Column(String(512), default="")
    utm_source = Column(String(255), default="")
    utm_medium = Column(String(255), default="")
    utm_campaign = Column(String(255), default="")
    utm_term = Column(String(255), default="")
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())


class LabosLeadDB(Base):
    """Заявки LabOS с лендинга (модальная форма)."""
    __tablename__ = "labos_leads"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), default="")
    email = Column(String(320), default="")
    org = Column(String(300), default="")
    message = Column(Text, default="")
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())
    ip = Column(String(64), default="")


Base.metadata.create_all(bind=engine)

from sqlalchemy import text

# Auto-migrate columns if missing
for table, column, col_type in [
    ("users", "email", "VARCHAR DEFAULT ''"),
    ("projects", "description", "TEXT DEFAULT ''"),
    ("projects", "end_date", "VARCHAR DEFAULT ''"),
    ("friends", "id", "VARCHAR"),
    ("friends", "user_id", "VARCHAR"),
    ("friends", "friend_id", "VARCHAR"),
    ("friends", "status", "VARCHAR DEFAULT 'pending'"),
    ("friends", "created_at", "VARCHAR"),
    ("equipment", "id", "VARCHAR"),
    ("equipment", "owner_id", "VARCHAR"),
    ("equipment", "name", "VARCHAR DEFAULT ''"),
    ("equipment", "description", "TEXT DEFAULT ''"),
    ("equipment", "status", "VARCHAR DEFAULT 'available'"),
    ("equipment", "created_at", "VARCHAR"),
    ("equipment_bookings", "id", "VARCHAR"),
    ("equipment_bookings", "equipment_id", "VARCHAR"),
    ("equipment_bookings", "user_id", "VARCHAR"),
    ("equipment_bookings", "start_date", "VARCHAR DEFAULT ''"),
    ("equipment_bookings", "end_date", "VARCHAR DEFAULT ''"),
    ("equipment_bookings", "status", "VARCHAR DEFAULT 'pending'"),
    ("equipment_bookings", "created_at", "VARCHAR"),
    ("support_tickets", "id", "VARCHAR"),
    ("support_tickets", "user_id", "VARCHAR"),
    ("support_tickets", "subject", "VARCHAR DEFAULT ''"),
    ("support_tickets", "message", "TEXT DEFAULT ''"),
    ("support_tickets", "status", "VARCHAR DEFAULT 'open'"),
    ("support_tickets", "priority", "VARCHAR DEFAULT 'medium'"),
    ("support_tickets", "created_at", "VARCHAR"),
    ("support_tickets", "updated_at", "VARCHAR"),
    ("support_replies", "id", "VARCHAR"),
    ("support_replies", "ticket_id", "VARCHAR"),
    ("support_replies", "author", "VARCHAR DEFAULT ''"),
    ("support_replies", "message", "TEXT DEFAULT ''"),
    ("support_replies", "created_at", "VARCHAR"),
    ("audit_logs", "id", "VARCHAR"),
    ("audit_logs", "user_id", "VARCHAR DEFAULT ''"),
    ("audit_logs", "action", "VARCHAR DEFAULT ''"),
    ("audit_logs", "entity_type", "VARCHAR DEFAULT ''"),
    ("audit_logs", "entity_id", "VARCHAR DEFAULT ''"),
    ("audit_logs", "details", "TEXT DEFAULT ''"),
    ("audit_logs", "ip_address", "VARCHAR DEFAULT ''"),
    ("audit_logs", "created_at", "VARCHAR"),
    ("organizations", "id", "VARCHAR"),
    ("organizations", "name", "VARCHAR DEFAULT ''"),
    ("organizations", "owner_id", "VARCHAR"),
    ("organizations", "created_at", "VARCHAR"),
    ("friend_chats", "id", "VARCHAR"),
    ("friend_chats", "user1", "VARCHAR"),
    ("friend_chats", "user2", "VARCHAR"),
    ("friend_chats", "last_message", "VARCHAR DEFAULT ''"),
    ("friend_chats", "last_sender", "VARCHAR DEFAULT ''"),
    ("friend_chats", "last_time", "VARCHAR DEFAULT ''"),
    ("friend_chats", "user1_last_read_at", "VARCHAR DEFAULT ''"),
    ("friend_chats", "user2_last_read_at", "VARCHAR DEFAULT ''"),
    ("friend_chats", "created_at", "VARCHAR"),
    ("friend_messages", "id", "VARCHAR"),
    ("friend_messages", "chat_id", "VARCHAR"),
    ("friend_messages", "sender", "VARCHAR"),
    ("friend_messages", "content", "TEXT DEFAULT ''"),
    ("friend_messages", "created_at", "VARCHAR"),
    ("organization_members", "id", "VARCHAR"),
    ("organization_members", "org_id", "VARCHAR"),
    ("organization_members", "user_id", "VARCHAR"),
    ("organization_members", "role", "VARCHAR DEFAULT 'member'"),
    ("referral_links", "id", "VARCHAR"),
    ("referral_links", "user_id", "VARCHAR"),
    ("referral_links", "code", "VARCHAR DEFAULT ''"),
    ("referral_links", "clicks", "VARCHAR DEFAULT '0'"),
    ("referral_links", "registrations", "VARCHAR DEFAULT '0'"),
    ("referral_links", "created_at", "VARCHAR"),
    ("users", "last_login", "VARCHAR DEFAULT ''"),
    ("users", "is_banned", "VARCHAR DEFAULT 'false'"),
    ("users", "yandex_id", "VARCHAR DEFAULT ''"),
    ("users", "display_name", "VARCHAR DEFAULT ''"),
    ("users", "auth_provider", "VARCHAR DEFAULT 'password'"),
    ("users", "agree_marketing", "VARCHAR DEFAULT 'false'"),
    ("users", "plan", "VARCHAR DEFAULT 'free'"),
    ("users", "plan_expires_at", "VARCHAR DEFAULT ''"),
    ("service_visits", "referrer", "VARCHAR DEFAULT ''"),
    ("service_visits", "referrer_host", "VARCHAR DEFAULT ''"),
    ("service_visits", "source", "VARCHAR DEFAULT ''"),
    ("service_visits", "search_query", "VARCHAR DEFAULT ''"),
    ("service_visits", "utm_source", "VARCHAR DEFAULT ''"),
    ("service_visits", "utm_medium", "VARCHAR DEFAULT ''"),
    ("service_visits", "utm_campaign", "VARCHAR DEFAULT ''"),
    ("service_visits", "utm_term", "VARCHAR DEFAULT ''"),
]:
    try:
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
    except Exception as e:
        print(f"Migration notice ({table}.{column}): {e}")

# Backfill: everyone starts on Free plan
try:
    with engine.begin() as conn:
        conn.execute(text("UPDATE users SET plan = 'free' WHERE plan IS NULL OR plan = ''"))
except Exception as e:
    print(f"Plan backfill notice: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ── Audit Log Helper ──

def audit_log(db: Session, user_id: str, action: str, entity_type: str, entity_id: str, details: str = "", ip: str = ""):
    log = AuditLogDB(user_id=user_id, action=action, entity_type=entity_type, entity_id=entity_id, details=details, ip_address=ip)
    db.add(log)
    db.commit()


# ── Telegram alerts (registrations / errors / DeepSeek) ──

def _tg_escape(text: str) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _tg_load_chat_ids() -> list[str]:
    ids: list[str] = []
    if TG_CHAT_ID:
        for part in TG_CHAT_ID.split(","):
            s = part.strip()
            if s and s not in ids:
                ids.append(s)
    try:
        with open(TG_CHATS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("chat_ids") or []:
            s = str(item).strip()
            if s and s not in ids:
                ids.append(s)
    except Exception:
        pass
    return ids


def _tg_remember_chat(chat_id) -> bool:
    cid = str(chat_id).strip()
    if not cid:
        return False
    with _TG_LOCK:
        ids = _tg_load_chat_ids()
        if cid in ids:
            return False
        ids.append(cid)
        try:
            os.makedirs(DATABASE_DIR, exist_ok=True)
            with open(TG_CHATS_FILE, "w", encoding="utf-8") as f:
                json.dump({"chat_ids": ids}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return True


def _tg_api(method: str, payload: dict | None = None, *, timeout: float = 25.0) -> dict:
    if not TG_BOT_TOKEN:
        return {"ok": False, "description": "no token"}
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/{method}"
    kwargs: dict = {"timeout": timeout}
    if TG_HTTP_PROXY:
        kwargs["proxy"] = TG_HTTP_PROXY
    try:
        with httpx.Client(**kwargs) as client:
            if payload is None:
                resp = client.get(url)
            else:
                resp = client.post(url, json=payload)
            try:
                return resp.json()
            except Exception:
                return {"ok": False, "description": (resp.text or "")[:300]}
    except Exception as e:
        return {"ok": False, "description": str(e)}


def _tg_send_to(chat_id: str, text: str) -> bool:
    data = _tg_api(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text[:3900],
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=12.0,
    )
    return bool(data.get("ok"))


def tg_notify(text: str) -> None:
    """Fire-and-forget Telegram alert to all known admin chats."""
    if not TG_BOT_TOKEN:
        return

    def _deliver():
        chats = _tg_load_chat_ids()
        if not chats:
            return
        body = text.strip()
        if not body:
            return
        for chat_id in chats:
            try:
                _tg_send_to(chat_id, body)
            except Exception:
                pass

    threading.Thread(target=_deliver, daemon=True).start()


def tg_notify_ai_issue(where: str, detail: str) -> None:
    """Alert when DeepSeek / AI API fails; debounced per signature."""
    sig = f"{where}|{(detail or '')[:160]}"
    now = time.time()
    with _TG_LOCK:
        last = _TG_AI_FAIL_LAST.get(sig, 0.0)
        if now - last < _TG_AI_FAIL_COOLDOWN_SEC:
            return
        _TG_AI_FAIL_LAST[sig] = now
    tg_notify(
        "⚠️ <b>DeepSeek / AI недоступен</b>\n"
        f"Где: {_tg_escape(where)}\n"
        f"Детали: {_tg_escape((detail or '')[:500])}\n"
        f"Время: {_tg_escape(datetime.datetime.now().isoformat(timespec='seconds'))}"
    )


def _tg_handle_update(update: dict) -> None:
    msg = update.get("message") or update.get("edited_message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return
    is_new = _tg_remember_chat(chat_id)
    text = (msg.get("text") or "").strip()
    if text.startswith("/start") or is_new:
        _tg_send_to(
            str(chat_id),
            "✅ <b>Bio-Notebook</b>: алерты подключены.\n"
            "Сюда приходят регистрации, заявки LabOS, ошибки сервера и сбои DeepSeek/AI.",
        )
        if is_new:
            ai_ok = bool(AI_API_KEY and AI_API_URL and AI_CHAT_MODEL)
            tg_notify(
                "🔔 <b>Новый чат для алертов</b>\n"
                f"chat_id: <code>{_tg_escape(chat_id)}</code>\n"
                f"AI настроен: {'да' if ai_ok else 'нет (AI_API_* пусты)'}"
            )


def _tg_poller_loop() -> None:
    """Long-poll getUpdates so /start binds chat_id without HTTPS webhook."""
    offset = 0
    # Drop any leftover webhook so getUpdates works.
    _tg_api("deleteWebhook", {"drop_pending_updates": False}, timeout=10.0)
    while True:
        try:
            data = _tg_api(
                "getUpdates",
                {"timeout": 25, "offset": offset, "allowed_updates": ["message"]},
                timeout=35.0,
            )
            if not data.get("ok"):
                time.sleep(5)
                continue
            for upd in data.get("result") or []:
                try:
                    uid = int(upd.get("update_id", 0))
                    offset = max(offset, uid + 1)
                except Exception:
                    pass
                try:
                    _tg_handle_update(upd)
                except Exception:
                    pass
        except Exception:
            time.sleep(5)


def start_tg_poller() -> None:
    global _TG_POLLER_STARTED
    if not TG_BOT_TOKEN or _TG_POLLER_STARTED:
        return
    _TG_POLLER_STARTED = True
    threading.Thread(target=_tg_poller_loop, daemon=True, name="tg-poller").start()
    # One-shot boot notice if chats already known
    def _boot():
        time.sleep(2)
        chats = _tg_load_chat_ids()
        if not chats:
            return
        ai_ok = bool(AI_API_KEY and AI_API_URL and AI_CHAT_MODEL)
        tg_notify(
            "🟢 <b>Bio-Notebook</b> перезапущен\n"
            f"Алерты: OK · чатов: {len(chats)}\n"
            f"AI: {'настроен' if ai_ok else 'не настроен (AI_API_* пусты)'}"
        )
        if not ai_ok:
            tg_notify_ai_issue("startup", "AI_API_KEY / AI_API_URL / AI_CHAT_MODEL не заданы в окружении")

    threading.Thread(target=_boot, daemon=True).start()


app = FastAPI()
app.mount("/static/uploads", StaticFiles(directory="data/uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")
if BASE_PATH:
    app.mount(f"{BASE_PATH}/static/uploads", StaticFiles(directory="data/uploads"), name="uploads_prefixed")
    app.mount(f"{BASE_PATH}/static", StaticFiles(directory="static"), name="static_prefixed")
PLAN_LABELS = {
    "free": "Free",
    "student": "Pro",  # legacy id; product plan «Студент» = Pro
    "pro": "Pro",
    "corp": "LabOS",
    "labos": "LabOS",
}

WELCOME_TRIAL_DAYS = 30


def _parse_iso_dt(value: str) -> datetime.datetime | None:
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00").split("+")[0])
    except Exception:
        return None


def grant_welcome_trial(user: UserDB) -> None:
    """New accounts: 1 month of «Студент» without payment."""
    until = datetime.datetime.now() + datetime.timedelta(days=WELCOME_TRIAL_DAYS)
    user.plan = "student"
    user.plan_expires_at = until.isoformat(timespec="seconds")


def sync_user_plan_effective(user: UserDB, db: Session) -> str:
    """If trial/subscription end date passed, move user back to free."""
    plan = (getattr(user, "plan", None) or "free").strip() or "free"
    if plan not in PLAN_LABELS:
        plan = "free"
    expires = (getattr(user, "plan_expires_at", None) or "").strip()
    if plan != "free" and plan != "corp" and expires:
        end = _parse_iso_dt(expires)
        if end and datetime.datetime.now() >= end:
            user.plan = "free"
            db.commit()
            return "free"
    return plan


def _plan_context(request: Request):
    plan = getattr(request.state, "user_plan", None) or "free"
    return {
        "user_plan": plan,
        "plan_label": PLAN_LABELS.get(plan, "Бесплатный"),
        "graph_enabled": EXPERIMENT_GRAPH_ENABLED,
    }


templates = Jinja2Templates(directory="templates", context_processors=[_plan_context])
templates.env.globals["base_path"] = BASE_PATH
templates.env.globals["lab_admin_path"] = LAB_ADMIN_PATH
templates.env.globals["site_origin"] = SITE_ORIGIN
templates.env.filters["fix_uploads"] = normalize_upload_urls


def _format_ru_date(iso: str) -> str:
    if not iso:
        return ""
    months = (
        "",
        "января",
        "февраля",
        "марта",
        "апреля",
        "мая",
        "июня",
        "июля",
        "августа",
        "сентября",
        "октября",
        "ноября",
        "декабря",
    )
    try:
        y, m, d = iso.split("-")[:3]
        return f"{int(d)} {months[int(m)]} {y}"
    except Exception:
        return iso


templates.env.filters["ru_date"] = _format_ru_date
app.add_middleware(BasePathMiddleware, base_path=BASE_PATH)

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDE_CONTENT_DIR = os.path.join(WEB_DIR, "content", "guides")
MARKETING_CONTENT_DIR = os.path.join(WEB_DIR, "content", "pages")


def _request_inner_path(request: Request) -> str:
    path = request.url.path or "/"
    if BASE_PATH and path.startswith(BASE_PATH + "/"):
        return path[len(BASE_PATH) :] or "/"
    if BASE_PATH and path == BASE_PATH:
        return "/"
    return path


def _should_track_service_visit(inner_path: str, method: str) -> bool:
    if method.upper() != "GET":
        return False
    if inner_path.startswith("/api/"):
        return False
    if inner_path.startswith("/static"):
        return False
    if inner_path.startswith(LAB_ADMIN_PATH) or inner_path.startswith("/admin"):
        return False
    if inner_path in ("/favicon.ico", "/sitemap-guides.xml", "/robots.txt"):
        return False
    return True


def _own_visit_hosts() -> set[str]:
    hosts = {"bionotebook.ru", "www.bionotebook.ru", "anticus.tech", "www.anticus.tech", "176.109.103.209"}
    for raw in (SITE_ORIGIN, LEGAL_BASE_URL):
        try:
            host = (urlparse(raw or "").hostname or "").lower().strip(".")
            if host:
                hosts.add(host)
                if host.startswith("www."):
                    hosts.add(host[4:])
                else:
                    hosts.add("www." + host)
        except Exception:
            pass
    return hosts


def _first_query_value(params: dict, *keys: str) -> str:
    for key in keys:
        vals = params.get(key)
        if not vals:
            continue
        val = vals[0] if isinstance(vals, list) else vals
        text = unquote_plus(str(val or "")).strip()
        if text:
            return text[:512]
    return ""


def _search_query_from_referer(referer: str) -> str:
    if not referer:
        return ""
    try:
        parsed = urlparse(referer)
    except Exception:
        return ""
    host = (parsed.hostname or "").lower()
    params = parse_qs(parsed.query or "")
    if "google." in host or host.endswith("google.com"):
        return _first_query_value(params, "q", "query")
    if "yandex." in host or host in ("ya.ru", "www.ya.ru"):
        return _first_query_value(params, "text", "query", "q")
    if "bing." in host or "duckduckgo." in host or "search.yahoo." in host:
        return _first_query_value(params, "q", "p", "query")
    if "mail.ru" in host:
        return _first_query_value(params, "q", "query")
    return _first_query_value(params, "q", "text", "query", "p")


def _visit_source_label(referrer_host: str, utm_source: str = "") -> str:
    src = (utm_source or "").strip()
    if src:
        return src[:255]
    host = (referrer_host or "").lower().strip(".")
    if not host:
        return "Прямой заход"
    if host in _own_visit_hosts():
        return "Внутренний переход"
    if "google." in host or host.endswith("google.com"):
        return "Google"
    if "yandex." in host or host in ("ya.ru", "www.ya.ru"):
        return "Яндекс"
    if "bing." in host:
        return "Bing"
    if "duckduckgo." in host:
        return "DuckDuckGo"
    if "t.me" in host or "telegram." in host:
        return "Telegram"
    if host in ("vk.com", "m.vk.com", "away.vk.com"):
        return "VK"
    return host[:255]


def _parse_visit_attribution(referer: str, request_url: str) -> dict:
    ref = (referer or "").strip()
    req = (request_url or "").strip()
    ref_host = ""
    try:
        if ref:
            ref_host = (urlparse(ref).hostname or "").lower().strip(".")
    except Exception:
        ref_host = ""

    utm_source = utm_medium = utm_campaign = utm_term = ""
    landing_q = ""
    try:
        if req:
            q = parse_qs(urlparse(req).query or "")
            utm_source = _first_query_value(q, "utm_source")[:255]
            utm_medium = _first_query_value(q, "utm_medium")[:255]
            utm_campaign = _first_query_value(q, "utm_campaign")[:255]
            utm_term = _first_query_value(q, "utm_term")[:255]
            landing_q = _first_query_value(q, "q", "query", "text", "search")
    except Exception:
        pass

    search_query = utm_term or landing_q or _search_query_from_referer(ref)
    source = _visit_source_label(ref_host, utm_source)
    return {
        "referrer": ref[:1024],
        "referrer_host": ref_host[:255],
        "source": source,
        "search_query": search_query[:512],
        "utm_source": utm_source,
        "utm_medium": utm_medium,
        "utm_campaign": utm_campaign,
        "utm_term": utm_term,
    }


def _record_service_visit(
    visitor_id: str,
    inner_path: str,
    *,
    referer: str = "",
    request_url: str = "",
) -> None:
    if not visitor_id:
        return
    attr = _parse_visit_attribution(referer, request_url)
    db = SessionLocal()
    try:
        db.add(
            ServiceVisitDB(
                visitor_id=visitor_id[:64],
                path=(inner_path or "/")[:512],
                referrer=attr["referrer"],
                referrer_host=attr["referrer_host"],
                source=attr["source"],
                search_query=attr["search_query"],
                utm_source=attr["utm_source"],
                utm_medium=attr["utm_medium"],
                utm_campaign=attr["utm_campaign"],
                utm_term=attr["utm_term"],
                created_at=datetime.datetime.now().isoformat(),
            )
        )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _parse_service_visit_day(raw: str) -> datetime.date | None:
    if not raw:
        return None
    s = raw.strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        if len(s) >= 19 and s[10] == "T":
            return datetime.datetime.fromisoformat(s[:19]).date()
        return datetime.datetime.fromisoformat(s[:10]).date()
    except ValueError:
        return None


def _service_visit_daily_series(db: Session, days: int = 30) -> list[dict]:
    now = datetime.datetime.now()
    start_date = (now - datetime.timedelta(days=days - 1)).date()
    buckets: dict[str, dict] = {}
    for i in range(days):
        d = start_date + datetime.timedelta(days=i)
        key = d.isoformat()
        buckets[key] = {"date": key, "visits": 0, "visitors": set()}

    since = datetime.datetime.combine(start_date, datetime.time.min).isoformat()
    rows = (
        db.query(ServiceVisitDB.visitor_id, ServiceVisitDB.created_at)
        .filter(ServiceVisitDB.created_at >= since)
        .all()
    )
    for visitor_id, created_at in rows:
        day = _parse_service_visit_day(created_at or "")
        if not day:
            continue
        key = day.isoformat()
        if key not in buckets:
            continue
        buckets[key]["visits"] += 1
        buckets[key]["visitors"].add(visitor_id)

    return [
        {
            "date": key,
            "visits": buckets[key]["visits"],
            "visitors_unique": len(buckets[key]["visitors"]),
        }
        for key in sorted(buckets.keys())
    ]


def _service_visit_top(
    db: Session,
    column,
    *,
    days: int | None = None,
    limit: int = 20,
    exclude_empty: bool = True,
) -> list[dict]:
    q = db.query(
        column.label("label"),
        func.count(ServiceVisitDB.id).label("visits"),
        func.count(distinct(ServiceVisitDB.visitor_id)).label("visitors"),
    )
    if days is not None:
        since = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
        q = q.filter(ServiceVisitDB.created_at >= since)
    if exclude_empty:
        q = q.filter(column.isnot(None)).filter(column != "")
    rows = (
        q.group_by(column)
        .order_by(func.count(ServiceVisitDB.id).desc())
        .limit(limit)
        .all()
    )
    out = []
    for row in rows:
        label = (row.label or "").strip()
        if not label and exclude_empty:
            continue
        out.append(
            {
                "label": label or "—",
                "visits": int(row.visits or 0),
                "visitors": int(row.visitors or 0),
            }
        )
    return out


def _service_visit_analytics(db: Session) -> dict:
    total_views = db.query(func.count(ServiceVisitDB.id)).scalar() or 0
    unique_total = db.query(func.count(distinct(ServiceVisitDB.visitor_id))).scalar() or 0
    now = datetime.datetime.now()

    def unique_since(days: int) -> int:
        since = (now - datetime.timedelta(days=days)).isoformat()
        return (
            db.query(func.count(distinct(ServiceVisitDB.visitor_id)))
            .filter(ServiceVisitDB.created_at >= since)
            .scalar()
            or 0
        )

    def views_since(days: int) -> int:
        since = (now - datetime.timedelta(days=days)).isoformat()
        return db.query(func.count(ServiceVisitDB.id)).filter(ServiceVisitDB.created_at >= since).scalar() or 0

    return {
        "visitors_unique_total": int(unique_total),
        "visitors_unique_24h": unique_since(1),
        "visitors_unique_7d": unique_since(7),
        "visitors_unique_30d": unique_since(30),
        "visits_total": int(total_views),
        "visits_24h": views_since(1),
        "visits_7d": views_since(7),
        "visits_30d": views_since(30),
        "visits_daily": _service_visit_daily_series(db, 30),
        "visit_sources": _service_visit_top(db, ServiceVisitDB.source, limit=20),
        "visit_sources_30d": _service_visit_top(db, ServiceVisitDB.source, days=30, limit=20),
        "visit_queries": _service_visit_top(db, ServiceVisitDB.search_query, limit=30),
        "visit_queries_30d": _service_visit_top(db, ServiceVisitDB.search_query, days=30, limit=30),
        "visit_referrers": _service_visit_top(db, ServiceVisitDB.referrer_host, limit=20),
        "visit_campaigns": _service_visit_top(db, ServiceVisitDB.utm_campaign, limit=15),
    }


@app.middleware("http")
async def service_visit_middleware(request: Request, call_next):
    inner = _request_inner_path(request)
    visitor_id = (request.cookies.get(VISITOR_COOKIE) or "").strip()
    new_visitor = False
    if not visitor_id:
        visitor_id = uuid.uuid4().hex
        new_visitor = True
    referer = (request.headers.get("referer") or request.headers.get("referrer") or "").strip()
    request_url = str(request.url)
    response = await call_next(request)
    if new_visitor:
        response.set_cookie(
            VISITOR_COOKIE,
            visitor_id,
            max_age=VISITOR_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=COOKIE_SECURE,
        )
    if _should_track_service_visit(inner, request.method):
        _record_service_visit(
            visitor_id,
            inner,
            referer=referer,
            request_url=request_url,
        )
    return response


@app.middleware("http")
async def head_requests_middleware(request: Request, call_next):
    if request.method.upper() == "HEAD":
        request.scope["method"] = "GET"
        response = await call_next(request)
        headers = {k: v for k, v in response.headers.items()}
        return Response(status_code=response.status_code, headers=headers)
    return await call_next(request)


@app.middleware("http")
async def canonical_slash_redirect_middleware(request: Request, call_next):
    if request.method.upper() not in ("GET", "HEAD"):
        return await call_next(request)
    path = request.url.path or "/"
    if BASE_PATH and path.startswith(BASE_PATH):
        inner = path[len(BASE_PATH) :] or "/"
    else:
        inner = path
    if "//" in path or "//" in inner:
        collapsed = re.sub(r"/+", "/", path)
        target = collapsed
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return RedirectResponse(url=target, status_code=301)
    if inner == "/guides":
        target = f"{BASE_PATH}/guides/"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return RedirectResponse(url=target, status_code=301)
    m = re.match(r"^/guides/([^/]+)$", inner)
    if m:
        target = f"{BASE_PATH}/guides/{m.group(1)}/"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return RedirectResponse(url=target, status_code=301)
    for slug in MARKETING_PAGES:
        if inner == f"/{slug}":
            target = f"{BASE_PATH}/{slug}/"
            if request.url.query:
                target = f"{target}?{request.url.query}"
            return RedirectResponse(url=target, status_code=301)
    return await call_next(request)


SEO_INDEXABLE_PATHS = {"/"}


def path_seo_indexable(path: str) -> bool:
    if path in SEO_INDEXABLE_PATHS:
        return True
    if path == "/guides" or path.startswith("/guides/"):
        return True
    for slug in MARKETING_PAGES:
        if path in (f"/{slug}", f"/{slug}/"):
            return True
    return False

_NOT_FOUND_HTML = (
    '<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
    '<meta name="robots" content="noindex">'
    "<title>Страница не найдена · Bio-Notebook</title>"
    "<style>body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;"
    "background:#05070a;color:#e6edf3;font-family:system-ui,sans-serif;text-align:center;padding:1.5rem}"
    "a{color:#64CEFB}</style></head><body><main><h1>404 — страница не найдена</h1>"
    '<p>Такой страницы нет. <a href="{home}/">На главную Bio-Notebook</a></p></main></body></html>'
)


def _route_exists(path: str, method: str) -> bool:
    from starlette.routing import Match

    if method.upper() == "HEAD":
        method = "GET"
    scope = {"type": "http", "path": path, "method": method, "root_path": ""}
    for route in app.router.routes:
        try:
            match, _ = route.matches(scope)
        except Exception:
            continue
        if match != Match.NONE:
            return True
    return False


def _not_found_response(path: str):
    if path.startswith("/api/"):
        return JSONResponse({"error": "NotFound"}, status_code=404)
    return HTMLResponse(_NOT_FOUND_HTML.replace("{home}", BASE_PATH), status_code=404)


@app.middleware("http")
async def session_auth_middleware(request: Request, call_next):
    """
    Enforce signed session cookie. The old ?user= query param is NOT proof of identity
    (anyone could open a friend's cabinet by changing the URL).
    """
    request.state.auth_user = get_session_username(request)
    request.state.user_plan = "free"
    path = request.url.path or "/"

    # Normalize path if somehow still prefixed (middleware order can vary)
    if BASE_PATH and path.startswith(BASE_PATH + "/"):
        path = path[len(BASE_PATH):] or "/"
    elif BASE_PATH and path == BASE_PATH:
        path = "/"

    session_user = request.state.auth_user
    query_user = request.query_params.get("user")

    # Logged-in user opening ?user=someone_else → bounce back to own cabinet
    if session_user and query_user and query_user != session_user and not path_is_public(path):
        # Keep other query params, force user=session
        q = dict(request.query_params)
        q["user"] = session_user
        target = request.url.replace(query=urlencode(q))
        return RedirectResponse(str(target), status_code=302)

    # HTML pages expect ?user= — fill from session if missing
    if (
        session_user
        and request.method == "GET"
        and not query_user
        and not path_is_public(path)
        and not path.startswith("/api/")
        and not path.startswith("/experiment/")
        and not path.startswith("/share/")
    ):
        q = dict(request.query_params)
        q["user"] = session_user
        return RedirectResponse(str(request.url.replace(query=urlencode(q))), status_code=302)

    if not session_user and not _route_exists(path, request.method):
        resp = _not_found_response(path)
        resp.headers["X-Robots-Tag"] = "noindex, nofollow"
        return resp

    if path == "/pricing" and request.method in ("GET", "HEAD") and not session_user:
        return RedirectResponse(url=f"{SITE_ORIGIN}{BASE_PATH}/#ceny", status_code=301)

    # Protected pages require a session (no more anonymous ?user= access)
    if not path_is_public(path) and not session_user:
        if path.startswith("/api/"):
            return JSONResponse({"error": "LoginRequired"}, status_code=401)
        return RedirectResponse(url=f"{BASE_PATH}/login?error=LoginRequired", status_code=302)

    # Plan for templates: prefer session identity
    plan_user = session_user or query_user
    if plan_user:
        db = SessionLocal()
        try:
            u = db.query(UserDB).filter(UserDB.username == plan_user).first()
            if u:
                plan = sync_user_plan_effective(u, db)
                request.state.user_plan = plan
                request.state.plan_expires_at = (getattr(u, "plan_expires_at", None) or "").strip()
        except Exception:
            request.state.user_plan = "free"
        finally:
            db.close()

    response = await call_next(request)
    if (
        not path_seo_indexable(path)
        and (not path.startswith("/static") or path.startswith("/static/uploads/"))
    ):
        response.headers.setdefault("X-Robots-Tag", "noindex, nofollow")
    return response


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    path = request.url.path or "/"
    if BASE_PATH and path.startswith(BASE_PATH + "/"):
        path = path[len(BASE_PATH):] or "/"
    detail = getattr(exc, "detail", None)
    if path.startswith("/api/") or (detail and detail != "Not Found"):
        return JSONResponse({"detail": detail or "Not Found"}, status_code=404)
    return _not_found_response(path)


@app.exception_handler(StarletteHTTPException)
async def biobook_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return await not_found_handler(request, exc)
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def biobook_validation_exception_handler(request: Request, exc: RequestValidationError):
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(Exception)
async def biobook_unhandled_exception_handler(request: Request, exc: Exception):
    path = request.url.path or "/"
    tg_notify(
        "🔥 <b>Ошибка сервера</b>\n"
        f"Путь: <code>{_tg_escape(path)}</code>\n"
        f"Тип: {_tg_escape(type(exc).__name__)}\n"
        f"Текст: {_tg_escape(str(exc)[:700])}\n"
        f"Время: {_tg_escape(datetime.datetime.now().isoformat(timespec='seconds'))}"
    )
    if path.startswith("/api/") or (BASE_PATH and path.startswith(BASE_PATH + "/api/")):
        return JSONResponse({"detail": "Internal Server Error"}, status_code=500)
    return HTMLResponse(
        "<h1>500</h1><p>Внутренняя ошибка сервера. Мы уже получили сигнал.</p>",
        status_code=500,
    )


@app.on_event("startup")
async def biobook_startup_tg():
    start_tg_poller()


NOTEBOOKS_BASE = os.environ.get("NOTEBOOKS_PATH", "/workspace/notebooks")

# Visible landing FAQ and FAQPage JSON-LD must match word-for-word (see FAQ in index.html).
FAQ_LANDING: list[dict[str, str]] = [
    {
        "question": "Сколько стоят тарифы?",
        "answer": (
            "Free — 0 ₽. Pro — 319 ₽/мес (агент и все возможности). LabOS — индивидуально, оставьте заявку."
        ),
        "answer_html": (
            'Free — 0 ₽. Pro — 319 ₽/мес (агент и все возможности). LabOS — индивидуально, '
            '<a href="#ceny" style="color:#0e7490">оставьте заявку</a>.'
        ),
    },
    {
        "question": "Нужна ли карта на старте?",
        "answer": "Нет — тариф Free доступен без карты.",
    },
    {
        "question": "Чем Bio-Notebook отличается от Word?",
        "answer": (
            "Структура по эксперименту, OCR, версии и ИИ по вашим записям — сравнение."
        ),
        "answer_html": (
            "Структура по эксперименту, OCR, версии и ИИ по вашим записям — "
            '<a href="#sravnenie" style="color:#0e7490">сравнение</a>.'
        ),
    },
    {
        "question": "Что входит в OCR и ИИ-агент?",
        "answer": (
            "OCR распознаёт рукопись и фото приборов в узле эксперимента. "
            "ИИ-агент доступен на тарифе Pro: отчёты, проверка расчётов и презентации по вашим записям."
        ),
    },
    {
        "question": "Как экспортировать отчёт?",
        "answer": (
            "Соберите узлы эксперимента в журнале и выгрузите отчёт в DOCX или PDF из кабинета — "
            "перед сдачей сверьте оформление с методичкой кафедры."
        ),
    },
    {
        "question": "Подойдёт ли для кафедры?",
        "answer": (
            "Да: Free и Pro — публичные тарифы на лендинге; для кафедры или лаборатории — LabOS, "
            "условия обсуждаем по заявке."
        ),
        "answer_html": (
            "Да: Free и Pro — "
            '<a href="#ceny" style="color:#0e7490">публичные тарифы</a>; '
            "для кафедры или лаборатории — LabOS, условия обсуждаем по заявке."
        ),
    },
    {
        "question": "Кому лучше выбрать другое?",
        "answer": (
            "Промышленная лаборатория, GLP или корпоративный ELN — смотрите сравнение электронных "
            "журналов и условия вендоров, не рейтинг «лучший». Либо оставьте заявку на LabOS."
        ),
        "answer_html": (
            "Промышленная лаборатория, GLP или корпоративный ELN — смотрите "
            f'<a href="{BASE_PATH}/sravnenie-elektronnyh-laboratornyh-zhurnalov/" style="color:#0e7490">'
            "сравнение электронных журналов</a> и условия вендоров, не рейтинг «лучший». "
            'Либо оставьте заявку на <a href="#ceny" style="color:#0e7490">LabOS</a>.'
        ),
    },
    {
        "question": "НДС включён в цену?",
        "answer": "Нет. Указанные цены без НДС: НДС не облагается.",
    },
]

GUIDE_ARTICLES: dict[str, dict] = {
    "kak-oformit-otchet-laboratornoy": {
        "slug": "kak-oformit-otchet-laboratornoy",
        "h1": "Как оформить отчёт по лабораторной работе",
        "seo_title": "Отчёт по лабораторной работе: структура, пример и шаблон",
        "title": "Как оформить отчёт по лабораторной работе",
        "description": "Структура отчёта: титульный лист, ход работы, таблицы и выводы. Шаблон DOCX и выгрузка из журнала Bio-Notebook. Методичка кафедры важнее универсального образца.",
        "published": "2026-11-10",
        "updated": "2026-11-11",
        "author": "Команда Bio-Notebook",
        "body_html": "",
    },
    "kak-vesti-laboratornyy-zhurnal": {
        "slug": "kak-vesti-laboratornyy-zhurnal",
        "h1": "Как вести лабораторный журнал",
        "h1_em": "",
        "kicker": "// Практика записей",
        "seo_title": "Как вести лабораторный журнал: правила записей и пример",
        "title": "Как вести лабораторный журнал",
        "description": "Что писать в лабораторном журнале каждый раз, как исправлять записи, пример таблицы и перенос рукописи в электронный журнал Bio-Notebook.",
        "published": "2026-11-11",
        "updated": "2026-11-11",
        "author": "Команда Bio-Notebook",
        "hero_image": "guide-vesti-hero.jpg",
        "mid_image": "guide-vesti-paper-digital.jpg",
        "mid_image_caption": "Бумага удобна у стенда; электронный журнал — для поиска, OCR и экспорта отчёта.",
        "mid_image_after_h2": "bumaga-elektronika",
        "body_html": "",
    },
    "vyvod-laboratornoy-raboty": {
        "slug": "vyvod-laboratornoy-raboty",
        "h1": "Как написать вывод к лабораторной работе",
        "seo_title": "Вывод лабораторной работы: как написать и примеры",
        "title": "Как написать вывод к лабораторной работе",
        "description": "Структура вывода: цель, результат с единицами, погрешность. Шаблон фразы и учебные примеры для вуза; как проверить расчёты перед выводом в Bio-Notebook.",
        "published": "2026-11-11",
        "updated": "2026-11-11",
        "author": "Команда Bio-Notebook",
        "body_html": "",
    },
    "oformlenie-laboratornoy-raboty-trebovaniya-primer": {
        "slug": "oformlenie-laboratornoy-raboty-trebovaniya-primer",
        "h1": "Оформление лабораторной работы",
        "h1_em": "требования и пример",
        "kicker": "// Требования РФ",
        "seo_title": "Оформление лабораторной работы: требования ГОСТ и пример",
        "title": "Оформление лабораторной работы: требования и пример",
        "description": "Оформление лабораторной по практике вузов РФ: ГОСТ 7.32-2017 и ГОСТ 2.105-2019 как ориентиры, поля, шрифт, титульный лист, таблицы, рисунки, чек-лист и учебный пример.",
        "published": "2026-11-12",
        "updated": "2026-11-12",
        "author": "Команда Bio-Notebook",
        "hero_image": "guide-oformlenie-hero.jpg",
        "mid_image": "",
        "body_html": "",
    },
    "zhurnal-laboratornyh-rabot-chto-pisat-oformit": {
        "slug": "zhurnal-laboratornyh-rabot-chto-pisat-oformit",
        "h1": "Журнал лабораторных работ",
        "h1_em": "что писать и как оформить",
        "kicker": "// Рабочая тетрадь",
        "seo_title": "Журнал лабораторных работ: что писать и как оформить",
        "title": "Журнал лабораторных работ: что писать и как оформить",
        "description": "Что писать в журнале лабораторных работ и как оформить тетрадь по практике вузов РФ: шапка, таблицы, исправления, отличие от отчёта, учебный пример страницы.",
        "published": "2026-11-12",
        "updated": "2026-11-12",
        "author": "Команда Bio-Notebook",
        "hero_image": "guide-zhurnal-lab-rabot-hero.jpg",
        "body_html": "",
    },
    "oformlenie-laboratornogo-zhurnala-gost-kafedra": {
        "slug": "oformlenie-laboratornogo-zhurnala-gost-kafedra",
        "h1": "Оформление лабораторного журнала",
        "h1_em": "по ГОСТ и требованиям кафедры",
        "kicker": "// Журнал и кафедра",
        "seo_title": "Оформление лабораторного журнала по ГОСТ и требованиям кафедры",
        "title": "Оформление лабораторного журнала по ГОСТ и требованиям кафедры",
        "description": "Научпоп-гайд: зачем лабораторный журнал, миф про ГОСТ на тетрадь, требования кафедры, чем журнал отличается от отчёта, шаблон страницы и чек-лист допуска.",
        "published": "2026-11-12",
        "updated": "2026-11-12",
        "author": "Команда Bio-Notebook",
        "hero_image": "guide-zhurnal-gost-hero.jpg",
        "body_html": "",
    },
}

MARKETING_PAGES: dict[str, dict] = {
    "sravnenie-elektronnyh-laboratornyh-zhurnalov": {
        "slug": "sravnenie-elektronnyh-laboratornyh-zhurnalov",
        "h1": "Какой электронный лабораторный журнал выбрать",
        "seo_title": "Электронные лабораторные журналы: сравнение для вуза и лаборатории",
        "description": "Сравнение бумажного журнала, Word, Bio-Notebook и корпоративных ELN без «лучшего» рейтинга. Для студента, кафедры и лаборатории — честные плюсы и минусы.",
        "published": "2026-11-11",
        "updated": "2026-11-11",
        "body_file": "sravnenie-eln-body.html",
    },
}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_ico():
    path = os.path.join(WEB_DIR, "static", "favicon.ico")
    if not os.path.isfile(path):
        path = os.path.join(WEB_DIR, "static", "favicon-32.png")
    return FileResponse(path, media_type="image/x-icon" if path.endswith(".ico") else "image/png")


@app.get("/sitemap-guides.xml", response_class=Response)
async def sitemap_guides_xml():
    """Фрагмент URL статей для включения в общий sitemap хоста."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    base = f"{SITE_ORIGIN}{BASE_PATH}"
    static_urls = [
        (f"{base}/", "2026-11-12"),
        (f"{base}/guides/", "2026-11-12"),
    ]
    for loc, lastmod in static_urls:
        lines.append(f"  <url><loc>{loc}</loc><lastmod>{lastmod}</lastmod></url>")
    for art in GUIDE_ARTICLES.values():
        if not _load_guide_body_html(art["slug"]).strip():
            continue
        loc = f"{base}/guides/{art['slug']}/"
        lines.append(f"  <url><loc>{loc}</loc><lastmod>{art.get('updated', art['published'])}</lastmod></url>")
    for page in MARKETING_PAGES.values():
        body = _load_marketing_body_html(page)
        if not body.strip():
            continue
        loc = f"{base}/{page['slug']}/"
        lines.append(f"  <url><loc>{loc}</loc><lastmod>{page.get('updated', page['published'])}</lastmod></url>")
    lines.append("</urlset>")
    return Response(content="\n".join(lines), media_type="application/xml")


def _load_marketing_body_html(page: dict) -> str:
    fname = page.get("body_file") or f"{page['slug']}-body.html"
    path = os.path.join(MARKETING_CONTENT_DIR, fname)
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _marketing_page_resolved(slug: str) -> dict | None:
    base = MARKETING_PAGES.get(slug)
    if not base:
        return None
    page = dict(base)
    page["body_html"] = _load_marketing_body_html(page)
    return page


def _marketing_page_json_ld(page: dict) -> str:
    page_url = f"{SITE_ORIGIN}{BASE_PATH}/{page['slug']}/"
    org_id = f"{SITE_ORIGIN}/#organization"
    software_id = f"{SITE_ORIGIN}{BASE_PATH}/#software"
    headline = page.get("h1") or page["seo_title"]
    data = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "BreadcrumbList",
                "itemListElement": [
                    {"@type": "ListItem", "position": 1, "name": "Главная", "item": f"{SITE_ORIGIN}/"},
                    {"@type": "ListItem", "position": 2, "name": "Bio-Notebook", "item": f"{SITE_ORIGIN}{BASE_PATH}/"},
                    {"@type": "ListItem", "position": 3, "name": headline, "item": page_url},
                ],
            },
            {
                "@type": "Article",
                "@id": f"{page_url}#article",
                "headline": headline,
                "description": page["description"],
                "datePublished": page.get("published") or page["updated"],
                "dateModified": page["updated"],
                "author": {"@id": org_id},
                "publisher": {"@id": org_id},
                "about": {"@id": software_id},
                "image": f"{SITE_ORIGIN}{BASE_PATH}/static/og-image.png",
                "mainEntityOfPage": page_url,
                "url": page_url,
                "inLanguage": "ru",
            },
        ],
    }
    return json.dumps(data, ensure_ascii=False)


def _rewrite_guide_internal_links(text: str) -> str:
    """Map legacy /bio-notebook/... links to current BASE_PATH (empty on standalone)."""
    prefix = f"{BASE_PATH}/" if BASE_PATH else "/"
    return text.replace('href="/bio-notebook/', f'href="{prefix}').replace(
        'src="/bio-notebook/', f'src="{prefix}'
    )


def _inject_guide_mid_image(article: dict, body: str) -> str:
    mid = (article.get("mid_image") or "").strip()
    after = (article.get("mid_image_after_h2") or "").strip()
    if not mid or not after or f'id="{after}"' not in body:
        return body
    caption = (article.get("mid_image_caption") or "").strip()
    src = f"{BASE_PATH}/static/{mid}" if BASE_PATH else f"/static/{mid}"
    figure = (
        f'<figure class="guide-figure">'
        f'<img src="{src}" width="1280" height="720" alt="" loading="lazy" decoding="async">'
    )
    if caption:
        figure += f"<figcaption>{caption}</figcaption>"
    figure += "</figure>"
    marker = f'<h2 id="{after}"'
    idx = body.find(marker)
    if idx == -1:
        return body
    # Insert after the closing </h2> of that section heading
    close = body.find("</h2>", idx)
    if close == -1:
        return body
    insert_at = close + len("</h2>")
    return body[:insert_at] + figure + body[insert_at:]


def _load_guide_body_html(slug: str) -> str:
    path = os.path.join(GUIDE_CONTENT_DIR, f"{slug}-body.html")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.lstrip().startswith("<!--"):
        end = text.find("-->")
        if end != -1:
            text = text[end + 3 :].lstrip()
    return _rewrite_guide_internal_links(text)


def _guide_article_resolved(slug: str) -> dict | None:
    base = GUIDE_ARTICLES.get(slug)
    if not base:
        return None
    article = dict(base)
    body = _load_guide_body_html(slug) or article.get("body_html", "")
    article["body_html"] = _inject_guide_mid_image(article, body)
    article.setdefault("h1", article.get("title", ""))
    article.setdefault("seo_title", article.get("title", ""))
    return article


def _guide_article_json_ld(article: dict) -> str:
    page_url = f"{SITE_ORIGIN}{BASE_PATH}/guides/{article['slug']}/"
    headline = article.get("h1") or article["title"]
    org_id = f"{SITE_ORIGIN}/#organization"
    software_id = f"{SITE_ORIGIN}{BASE_PATH}/#software"
    data = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "BreadcrumbList",
                "@id": f"{page_url}#breadcrumb",
                "itemListElement": [
                    {"@type": "ListItem", "position": 1, "name": "Главная", "item": f"{SITE_ORIGIN}/"},
                    {"@type": "ListItem", "position": 2, "name": "Bio-Notebook", "item": f"{SITE_ORIGIN}{BASE_PATH}/"},
                    {"@type": "ListItem", "position": 3, "name": "Статьи", "item": f"{SITE_ORIGIN}{BASE_PATH}/guides/"},
                    {"@type": "ListItem", "position": 4, "name": headline, "item": page_url},
                ],
            },
            {
                "@type": "Article",
                "@id": f"{page_url}#article",
                "headline": headline,
                "description": article["description"],
                "datePublished": article["published"],
                "dateModified": article["updated"],
                "author": {"@id": org_id},
                "publisher": {"@id": org_id},
                "about": {"@id": software_id},
                "image": f"{SITE_ORIGIN}{BASE_PATH}/static/og-image.png",
                "mainEntityOfPage": page_url,
                "url": page_url,
                "breadcrumb": {"@id": f"{page_url}#breadcrumb"},
                "inLanguage": "ru",
            },
        ],
    }
    return json.dumps(data, ensure_ascii=False)


def _guides_hub_json_ld(articles: list[dict]) -> str:
    page_url = f"{SITE_ORIGIN}{BASE_PATH}/guides/"
    items = [
        {
            "@type": "ListItem",
            "position": i + 1,
            "url": f"{SITE_ORIGIN}{BASE_PATH}/guides/{a['slug']}/",
            "name": a.get("title") or a.get("h1"),
        }
        for i, a in enumerate(articles)
    ]
    data = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "BreadcrumbList",
                "@id": f"{page_url}#breadcrumb",
                "itemListElement": [
                    {"@type": "ListItem", "position": 1, "name": "Главная", "item": f"{SITE_ORIGIN}/"},
                    {"@type": "ListItem", "position": 2, "name": "Bio-Notebook", "item": f"{SITE_ORIGIN}{BASE_PATH}/"},
                    {"@type": "ListItem", "position": 3, "name": "Статьи", "item": page_url},
                ],
            },
            {
                "@type": "CollectionPage",
                "@id": f"{page_url}#webpage",
                "url": page_url,
                "name": "Статьи: отчёты, лабораторный журнал, протоколы",
                "isPartOf": {"@id": f"{SITE_ORIGIN}/#website"},
                "breadcrumb": {"@id": f"{page_url}#breadcrumb"},
                "mainEntity": {"@id": f"{page_url}#itemlist"},
            },
            {
                "@type": "ItemList",
                "@id": f"{page_url}#itemlist",
                "itemListElement": items,
            },
        ],
    }
    return json.dumps(data, ensure_ascii=False)


@app.get("/guides", response_class=HTMLResponse)
@app.get("/guides/", response_class=HTMLResponse)
async def guides_index(request: Request):
    articles = sorted(GUIDE_ARTICLES.values(), key=lambda a: a["published"], reverse=True)
    return templates.TemplateResponse(
        request,
        "guides_index.html",
        {
            "articles": articles,
            "guides_json_ld": _guides_hub_json_ld(articles),
        },
    )


@app.get("/guides/{slug}", response_class=HTMLResponse)
@app.get("/guides/{slug}/", response_class=HTMLResponse)
async def guide_detail(request: Request, slug: str):
    key = slug.strip("/")
    article = _guide_article_resolved(key)
    if not article or not (article.get("body_html") or "").strip():
        return _not_found_response(f"/guides/{slug}")
    return templates.TemplateResponse(
        request,
        "guide_article.html",
        {
            "article": article,
            "article_json_ld": _guide_article_json_ld(article),
        },
    )


@app.get("/sravnenie-elektronnyh-laboratornyh-zhurnalov", response_class=HTMLResponse)
@app.get("/sravnenie-elektronnyh-laboratornyh-zhurnalov/", response_class=HTMLResponse)
async def marketing_eln_compare(request: Request):
    page = _marketing_page_resolved("sravnenie-elektronnyh-laboratornyh-zhurnalov")
    if not page or not (page.get("body_html") or "").strip():
        return _not_found_response("/sravnenie-elektronnyh-laboratornyh-zhurnalov/")
    return templates.TemplateResponse(
        request,
        "marketing_page.html",
        {
            "page": page,
            "page_json_ld": _marketing_page_json_ld(page),
        },
    )

@app.post("/api/labos-lead")
async def api_labos_lead(request: Request, db: Session = Depends(get_db)):
    """Публичная заявка LabOS с лендинга."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Некорректный JSON"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"ok": False, "error": "Некорректные данные"}, status_code=400)

    name = str(payload.get("name") or "").strip()[:200]
    email = str(payload.get("email") or "").strip()[:320]
    org = str(payload.get("org") or "").strip()[:300]
    message = str(payload.get("message") or "").strip()[:4000]

    if len(name) < 2:
        return JSONResponse({"ok": False, "error": "Укажите имя"}, status_code=400)
    if "@" not in email or "." not in email.split("@")[-1]:
        return JSONResponse({"ok": False, "error": "Укажите корректный email"}, status_code=400)

    ip = (request.client.host if request.client else "") or ""
    fwd = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if fwd:
        ip = fwd[:64]

    lead = LabosLeadDB(name=name, email=email, org=org, message=message, ip=ip[:64])
    db.add(lead)
    db.commit()
    tg_notify(
        "🧪 <b>LabOS заявка</b>\n"
        f"Имя: {_tg_escape(name)}\n"
        f"Email: {_tg_escape(email)}\n"
        f"Орг: {_tg_escape(org) or '—'}\n"
        f"Сообщение: {_tg_escape(message) or '—'}\n"
        f"IP: {_tg_escape(ip)}\n"
        f"id: {lead.id}"
    )
    return JSONResponse({"ok": True, "id": lead.id})


@app.post("/api/tg/webhook")
async def api_tg_webhook(request: Request):
    """Optional Telegram webhook (HTTPS). Long-poller is primary on this host."""
    if not TG_BOT_TOKEN:
        return JSONResponse({"ok": False}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False}, status_code=400)
    if isinstance(body, dict):
        try:
            _tg_handle_update(body)
        except Exception:
            pass
    return JSONResponse({"ok": True})


# ── Auth ──

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    session_user = get_session_username(request)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "error": error,
            "session_user": session_user or "",
            "yandex_enabled": yandex_oauth_enabled(),
            "legal_base_url": LEGAL_BASE_URL,
            "welcome_trial_days": WELCOME_TRIAL_DAYS,
            "faq_landing": FAQ_LANDING,
        },
    )

@app.get("/login", response_class=HTMLResponse)
async def auth_page(request: Request, mode: str = "login", error: str = ""):
    # Already logged in → home
    session_user = get_session_username(request)
    if session_user and not error:
        return RedirectResponse(url=f"{BASE_PATH}/dashboard?user={session_user}", status_code=302)
    if mode not in ("login", "register"):
        mode = "login"
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "mode": mode,
            "error": error,
            "yandex_enabled": yandex_oauth_enabled(),
            "legal_base_url": LEGAL_BASE_URL,
            "welcome_trial_days": WELCOME_TRIAL_DAYS,
        },
    )

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), request: Request = None, db: Session = Depends(get_db)):
    user = find_user_by_login(db, username)
    if not user:
        return RedirectResponse(url=f"{BASE_PATH}/login?error=UserNotFound", status_code=302)
    if user.is_banned == "true":
        return RedirectResponse(url=f"{BASE_PATH}/login?error=UserBanned", status_code=302)
    if user.auth_provider == "yandex" and not user.password:
        return RedirectResponse(url=f"{BASE_PATH}/login?error=UseYandexLogin", status_code=302)
    if not verify_password(user.password, password):
        return RedirectResponse(url=f"{BASE_PATH}/login?error=WrongPassword", status_code=302)
    user.last_login = datetime.datetime.now().isoformat()
    db.commit()
    ip = request.client.host if request else ""
    audit_log(db, user.username, "login", "user", user.username, ip=ip)
    return redirect_to_dashboard(user.username)

@app.post("/register")
async def register(
    email: str = Form(...),
    password: str = Form(...),
    username: str = Form(None),
    agree_terms: str = Form(None),
    agree_offer: str = Form(None),
    agree_privacy: str = Form(None),
    agree_data: str = Form(None),
    agree_marketing: str = Form(None),
    request: Request = None,
    db: Session = Depends(get_db),
):
    terms_ok = bool(agree_terms) or all([agree_offer, agree_privacy, agree_data])
    if not terms_ok:
        return RedirectResponse(url=f"{BASE_PATH}/login?mode=register&error=ConsentRequired", status_code=302)

    email = (email or "").strip().lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return RedirectResponse(url=f"{BASE_PATH}/login?mode=register&error=InvalidEmail", status_code=302)
    if len(password or "") < 6:
        return RedirectResponse(url=f"{BASE_PATH}/login?mode=register&error=WeakPassword", status_code=302)

    if db.query(UserDB).filter(func.lower(UserDB.email) == email).first():
        return RedirectResponse(url=f"{BASE_PATH}/login?mode=register&error=EmailExists", status_code=302)

    raw_username = (username or "").strip()
    if raw_username:
        if not re.match(r"^[\w.-]{3,32}$", raw_username):
            return RedirectResponse(url=f"{BASE_PATH}/login?mode=register&error=InvalidUsername", status_code=302)
        if db.query(UserDB).filter(UserDB.username == raw_username).first():
            return RedirectResponse(url=f"{BASE_PATH}/login?mode=register&error=UserExists", status_code=302)
        final_username = raw_username
    else:
        final_username = unique_username(db, username_from_email(email))

    new_user = UserDB(
        username=final_username,
        email=email,
        password=hash_password(password),
        auth_provider="password",
        agree_marketing="true" if agree_marketing else "false",
        plan="free",
    )
    grant_welcome_trial(new_user)
    db.add(new_user)
    db.commit()

    code = str(uuid.uuid4())[:8]
    ref = ReferralLinkDB(user_id=final_username, code=code)
    db.add(ref)
    db.commit()

    ip = request.client.host if request else ""
    audit_log(
        db, final_username, "register", "user", final_username,
        details=f"welcome_trial_{WELCOME_TRIAL_DAYS}d", ip=ip,
    )
    tg_notify(
        "🆕 <b>Регистрация</b>\n"
        f"Логин: <code>{_tg_escape(final_username)}</code>\n"
        f"Email: {_tg_escape(email)}\n"
        f"Способ: пароль\n"
        f"Trial: {WELCOME_TRIAL_DAYS}д\n"
        f"IP: {_tg_escape(ip)}"
    )
    return redirect_to_dashboard(final_username)


@app.get("/logout")
async def logout():
    resp = RedirectResponse(url=f"{BASE_PATH}/", status_code=302)
    return clear_session_cookie(resp)


@app.get("/auth/yandex")
async def auth_yandex(
    mode: str = Query("login"),
    agree_marketing: bool = Query(False),
):
    if not yandex_oauth_enabled():
        return RedirectResponse(url=f"{BASE_PATH}/login?error=YandexNotConfigured", status_code=302)
    if mode not in ("login", "register"):
        mode = "login"
    state = sign_oauth_state(
        {
            "mode": mode,
            "marketing": agree_marketing,
            "ts": time.time(),
            "nonce": secrets.token_hex(8),
        }
    )
    params = urlencode(
        {
            "response_type": "code",
            "client_id": YANDEX_CLIENT_ID,
            "redirect_uri": yandex_redirect_uri(),
            "state": state,
        }
    )
    return RedirectResponse(f"https://oauth.yandex.ru/authorize?{params}", status_code=302)


@app.get("/auth/yandex/callback")
async def auth_yandex_callback(
    request: Request,
    code: str = Query(None),
    state: str = Query(None),
    error: str = Query(None),
    db: Session = Depends(get_db),
):
    if error or not code or not state:
        return RedirectResponse(url=f"{BASE_PATH}/login?error=YandexAuthFailed", status_code=302)

    state_data = verify_oauth_state(state)
    if not state_data:
        return RedirectResponse(url=f"{BASE_PATH}/login?error=YandexAuthFailed", status_code=302)

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            token_resp = await client.post(
                "https://oauth.yandex.ru/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": YANDEX_CLIENT_ID,
                    "client_secret": YANDEX_CLIENT_SECRET,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            token_resp.raise_for_status()
            access_token = token_resp.json().get("access_token")
            if not access_token:
                raise ValueError("missing access_token")

            user_resp = await client.get(
                "https://login.yandex.ru/info",
                headers={"Authorization": f"OAuth {access_token}"},
                params={"format": "json"},
            )
            user_resp.raise_for_status()
            yandex_user = user_resp.json()
    except Exception:
        return RedirectResponse(url=f"{BASE_PATH}/login?error=YandexAuthFailed", status_code=302)

    yandex_id = str(yandex_user.get("id", ""))
    email = (yandex_user.get("default_email") or "").strip().lower()
    login_name = yandex_user.get("login") or yandex_user.get("display_name") or ""
    display_name = (yandex_user.get("real_name") or yandex_user.get("display_name") or login_name).strip()

    if not yandex_id:
        return RedirectResponse(url=f"{BASE_PATH}/login?error=YandexAuthFailed", status_code=302)

    user = db.query(UserDB).filter(UserDB.yandex_id == yandex_id).first()
    if not user and email:
        user = db.query(UserDB).filter(UserDB.email == email).first()
        if user:
            user.yandex_id = yandex_id
            if display_name:
                user.display_name = display_name
            if user.auth_provider != "yandex":
                user.auth_provider = "yandex" if not user.password else user.auth_provider

    if not user:
        if state_data.get("mode") == "login":
            return RedirectResponse(url=f"{BASE_PATH}/login?error=YandexAccountNotFound", status_code=302)
        base_username = sanitize_username(login_name, f"ya_{yandex_id[:8]}")
        username = unique_username(db, base_username)
        user = UserDB(
            username=username,
            email=email,
            password="",
            yandex_id=yandex_id,
            display_name=display_name,
            auth_provider="yandex",
            agree_marketing="true" if state_data.get("marketing") else "false",
            plan="free",
        )
        grant_welcome_trial(user)
        db.add(user)
        db.commit()

        ref = ReferralLinkDB(user_id=username, code=str(uuid.uuid4())[:8])
        db.add(ref)
        db.commit()
        audit_log(
            db, username, "register", "user", username,
            details=f"yandex_oauth;welcome_trial_{WELCOME_TRIAL_DAYS}d",
            ip=request.client.host if request.client else "",
        )
        tg_notify(
            "🆕 <b>Регистрация</b>\n"
            f"Логин: <code>{_tg_escape(username)}</code>\n"
            f"Email: {_tg_escape(email) or '—'}\n"
            f"Имя: {_tg_escape(display_name) or '—'}\n"
            f"Способ: Яндекс\n"
            f"Trial: {WELCOME_TRIAL_DAYS}д\n"
            f"IP: {_tg_escape(request.client.host if request.client else '')}"
        )

    if user.is_banned == "true":
        return RedirectResponse(url=f"{BASE_PATH}/login?error=UserBanned", status_code=302)

    user.last_login = datetime.datetime.now().isoformat()
    db.commit()
    audit_log(db, user.username, "login", "user", user.username, details="yandex_oauth", ip=request.client.host if request.client else "")
    return redirect_to_dashboard(user.username)

# ── Dashboard ──

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: str, db: Session = Depends(get_db)):
    user_exps = db.query(ExperimentDB).filter(ExperimentDB.user_id == user).all()
    # Other users' public experiments
    other_public = db.query(ExperimentDB).filter(
        ExperimentDB.user_id != user,
        ExperimentDB.is_public == "true",
    ).order_by(ExperimentDB.id.desc()).limit(12).all()

    # Friend list for determining friend status
    friend_rels = db.query(FriendDB).filter(
        (FriendDB.user_id == user) | (FriendDB.friend_id == user),
    ).all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": user,
            "experiments": user_exps,
            "other_public": other_public,
            "friend_rels": friend_rels,
            "active_page": "dashboard",
            "lab_planner_enabled": LAB_PLANNER_ENABLED,
        },
    )

# ── Projects page ──

def parse_amount(val):
    try:
        return float(val.replace(",", ".")) if val else 0.0
    except (ValueError, AttributeError):
        return 0.0

@app.get("/projects", response_class=HTMLResponse)
async def projects_page(request: Request, user: str, project: str = None, db: Session = Depends(get_db)):
    projects = db.query(ProjectDB).filter(ProjectDB.user_id == user).all()
    all_users = db.query(UserDB).all()
    current_project = None
    project_tasks = []
    project_members = []
    gantt_tasks: list[dict] = []
    if project:
        current_project = db.query(ProjectDB).filter(ProjectDB.id == project).first()
        if current_project:
            project_tasks = db.query(ProjectTaskDB).filter(ProjectTaskDB.project_id == project).all()
            members = db.query(ProjectMemberDB).filter(ProjectMemberDB.project_id == project).all()
            project_members = [m.username for m in members]
            for t in project_tasks:
                start = (t.start_date or "").strip()
                end = (t.end_date or "").strip()
                if start and end:
                    gantt_tasks.append(
                        {
                            "id": t.id,
                            "title": t.title or "",
                            "assignee": t.assignee or "",
                            "status": t.status or "",
                            "start_date": start,
                            "end_date": end,
                            "priority": t.priority or "Средний",
                        }
                    )
    return templates.TemplateResponse(
        request, "projects.html",
        {
            "user": user,
            "projects": projects,
            "current_project": current_project,
            "project_tasks": project_tasks,
            "project_members": project_members,
            "all_users": all_users,
            "gantt_tasks": gantt_tasks,
            "active_page": "projects",
        },
    )

# ── Reagents page ──

@app.get("/reagents", response_class=HTMLResponse)
async def reagents_page(request: Request, user: str, db: Session = Depends(get_db)):
    reagents = db.query(ReagentDB).filter(ReagentDB.user_id == user).all()
    low_reagents = sum(1 for r in reagents if parse_amount(r.amount) <= parse_amount(r.min_amount))
    return templates.TemplateResponse(
        request, "reagents.html",
        {"user": user, "reagents": reagents, "low_reagents": low_reagents, "active_page": "reagents"},
    )

# ── Consumables page ──

@app.get("/consumables", response_class=HTMLResponse)
async def consumables_page(request: Request, user: str, db: Session = Depends(get_db)):
    consumables = db.query(ConsumableDB).filter(ConsumableDB.user_id == user).all()
    low_consumables = sum(1 for c in consumables if parse_amount(c.amount) <= parse_amount(c.min_amount))
    return templates.TemplateResponse(
        request, "consumables.html",
        {"user": user, "consumables": consumables, "low_consumables": low_consumables, "active_page": "consumables"},
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
    audit_log(db, user, "create", "experiment", exp_id, json.dumps({"name": name}))
    return RedirectResponse(url=f"{BASE_PATH}/dashboard?user={user}", status_code=302)

@app.get("/experiment/{user}/{exp_id}", response_class=HTMLResponse)
async def experiment_detail(request: Request, user: str, exp_id: str, viewing_user: str = None, db: Session = Depends(get_db)):
    exp = db.query(ExperimentDB).filter(ExperimentDB.id == exp_id, ExperimentDB.user_id == user).first()
    if not exp:
        return RedirectResponse(url=f"{BASE_PATH}/dashboard?user={user}")

    # Check friendship if viewing as another user
    is_owner = not viewing_user or viewing_user == user
    fr_status = "none"
    if not is_owner:
        fr = db.query(FriendDB).filter(
            ((FriendDB.user_id == viewing_user) & (FriendDB.friend_id == user)) |
            ((FriendDB.user_id == user) & (FriendDB.friend_id == viewing_user)),
        ).first()
        if fr:
            fr_status = fr.status

    return templates.TemplateResponse(
        request, "experiment_detail.html",
        {
            "user": user,
            "exp": exp,
            "viewing_user": viewing_user or user,
            "is_owner": is_owner,
            "fr_status": fr_status,
            "active_page": "dashboard",
        }
    )

# ── Global experiment relationship graph API ──

import math
import re
from collections import Counter

_GRAPH_STOPWORDS = {
    # RU function / generic lab words
    "этот", "эта", "эти", "этой", "этого", "этих", "того", "также", "после", "перед",
    "когда", "если", "или", "для", "при", "что", "как", "все", "всё", "его", "её",
    "они", "мы", "вы", "был", "была", "было", "были", "есть", "будет", "будут",
    "можно", "нужно", "надо", "очень", "более", "менее", "только", "через", "между",
    "потом", "сейчас", "здесь", "туда", "сюда", "куда", "откуда", "почему", "потому",
    "начало", "исследования", "исследование", "исследований", "эксперимент",
    "эксперимента", "эксперименты", "описание", "система", "системы", "данные",
    "данных", "результат", "результаты", "результатов", "узел", "узлы", "узлов",
    "тест", "теста", "тесты", "тестов", "версия", "новый", "новая", "новое",
    "новые", "проба", "пробы", "работа", "работы", "проект", "проекта", "лаборатория",
    "лаборатории", "метод", "метода", "методы", "анализ", "анализа", "образец",
    "образцы", "таблица", "таблицы", "файл", "файлы", "запись", "записи", "заметка",
    "заметки", "инициализация", "среды", "среды", "контент", "текст", "название",
    "пользователь", "пользователя", "доступ", "статус", "этап", "этапы",
    # EN
    "this", "that", "with", "from", "have", "been", "were", "will", "would",
    "could", "should", "about", "into", "over", "after", "before", "between",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "what", "which", "while", "than", "them", "they", "their",
    "test", "tests", "data", "node", "nodes", "experiment", "experiments",
    "description", "start", "init", "root", "demo", "version", "system",
    "result", "results", "sample", "samples", "method", "methods", "analysis",
    "project", "projects", "lab", "laboratory", "note", "notes", "file", "files",
    "table", "content", "text", "title", "user", "users", "status", "stage",
}


def _fix_mojibake(text: str) -> str:
    """Repair UTF-8 text that was wrongly decoded as latin-1/cp1252."""
    if not text or not isinstance(text, str):
        return text or ""
    # Typical mojibake markers for Cyrillic mis-decoded as Latin-1
    if ("Ð" in text or "Ñ" in text) and any(ord(c) > 127 for c in text):
        for enc in ("latin-1", "cp1252"):
            try:
                fixed = text.encode(enc).decode("utf-8")
                if fixed and ("Ð" not in fixed or fixed.count("Ð") < text.count("Ð")):
                    return fixed
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
    return text


def _tokenize_ident(text: str) -> list:
    """Split CamelCase / snake_case / glued identifiers into tokens."""
    if not text:
        return []
    text = _fix_mojibake(text)
    text = re.sub(r"([a-zа-яё])([A-ZА-ЯЁ])", r"\1 \2", text)
    text = re.sub(r"([A-Za-zА-Яа-яё])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([A-Za-zА-Яа-яё])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ")
    return re.findall(r"[а-яёa-z][а-яёa-z0-9]{2,}", text.lower())


def _extract_graph_keywords(text: str, *, allow_short: bool = False) -> set:
    if not text:
        return set()
    text = _fix_mojibake(text)
    text = re.sub(r"<[^>]+>", " ", text)
    tokens = _tokenize_ident(text)
    out = set()
    for w in tokens:
        if w in _GRAPH_STOPWORDS or w.isdigit():
            continue
        if len(w) < (3 if allow_short else 4):
            continue
        out.add(w)
    return out



# Human-readable concept families for richer link explanations
_CONCEPT_ALIASES = {
    "diag": "diagnostics",
    "diagnostic": "diagnostics",
    "diagnostics": "diagnostics",
    "diagnostictest": "diagnostics",
    "diagnosticexperiment": "diagnostics",
    "диагност": "diagnostics",
    "диагностика": "diagnostics",
    "диагностики": "diagnostics",
    "диагностический": "diagnostics",
    "тестовый": "testing",
    "testing": "testing",
    "checking": "testing",
    "healthcheck": "health_check",
    "health": "health_check",
    "работоспособности": "health_check",
    "проверка": "health_check",
    "фиксация": "mouse_fixation",
    "фиксации": "mouse_fixation",
    "фиксатора": "mouse_fixation",
    "fixation": "mouse_fixation",
    "restrainer": "mouse_fixation",
    "мышей": "mouse_fixation",
    "mouse": "mouse_fixation",
    "антикусь": "anticus",
    "anticus": "anticus",
    "стресса": "stress",
    "stress": "stress",
    "демо": "demo",
    "demo": "demo",
}

_CONCEPT_LABELS = {
    "diagnostics": "Диагностика",
    "testing": "Тестирование",
    "health_check": "Проверка системы",
    "mouse_fixation": "Фиксация животных",
    "anticus": "Антикусь",
    "stress": "Стресс / поведение",
    "demo": "Демо-эксперимент",
}

_WEAK_STUBS = {"diag", "test", "demo", "lab", "new", "tmp", "data", "node", "root"}


def _concept_of(token: str) -> str:
    t = (token or "").lower().strip()
    if not t:
        return ""
    if t in _CONCEPT_ALIASES:
        return _CONCEPT_ALIASES[t]
    for alias, concept in _CONCEPT_ALIASES.items():
        if len(alias) >= 4 and (t.startswith(alias) or alias.startswith(t)):
            return concept
    return t


def _concept_label(concept: str) -> str:
    if concept in _CONCEPT_LABELS:
        return _CONCEPT_LABELS[concept]
    return concept.replace("_", " ").capitalize()


def _to_concepts(tokens) -> set:
    return {_concept_of(t) for t in tokens if _concept_of(t)}


def _strength_label(strength: float) -> str:
    if strength >= 0.7:
        return "сильная"
    if strength >= 0.4:
        return "средняя"
    return "слабая"


@app.get("/api/experiments/graph")
async def experiments_graph(
    user: str,
    scope: str = "all",
    db: Session = Depends(get_db),
):
    """
    Production knowledge graph.
    scope:
      - mine: only current user's experiments
      - network: user + accepted friends (all their exps) + public others
      - all: everything in DB (admin-style overview)
    """
    if not EXPERIMENT_GRAPH_ENABLED:
        return JSONResponse({"nodes": [], "links": [], "meta": {"disabled": True}})

    scope = (scope or "all").lower()
    if scope not in ("mine", "network", "all"):
        scope = "all"

    friend_ids = set()
    if scope in ("network", "all"):
        fr = db.query(FriendDB).filter(
            (FriendDB.user_id == user) | (FriendDB.friend_id == user),
            FriendDB.status == "accepted",
        ).all()
        for f in fr:
            friend_ids.add(f.user_id if f.user_id != user else f.friend_id)

    if scope == "mine":
        exps = db.query(ExperimentDB).filter(ExperimentDB.user_id == user).all()
    elif scope == "all":
        exps = db.query(ExperimentDB).all()
    else:
        own = db.query(ExperimentDB).filter(ExperimentDB.user_id == user).all()
        friend_exps = (
            db.query(ExperimentDB).filter(ExperimentDB.user_id.in_(friend_ids)).all()
            if friend_ids else []
        )
        public_other = db.query(ExperimentDB).filter(
            ExperimentDB.user_id != user,
            ExperimentDB.is_public == "true",
        ).all()
        by_id = {e.id: e for e in own}
        for e in friend_exps:
            by_id[e.id] = e
        for e in public_other:
            by_id.setdefault(e.id, e)
        exps = list(by_id.values())

    exp_data = []
    for exp in exps:
        name = _fix_mojibake(exp.name or "")
        desc = _fix_mojibake(exp.description or "")
        keywords = _extract_graph_keywords(name, allow_short=True) | _extract_graph_keywords(desc)
        name_tokens = _extract_graph_keywords(name, allow_short=True)
        if exp.levels:
            for level in exp.levels:
                for node in level:
                    keywords |= _extract_graph_keywords(node.get("title", "") or "", allow_short=True)
                    content = node.get("content", "") or ""
                    if content:
                        keywords |= _extract_graph_keywords(content[:500])
        concepts = _to_concepts(keywords)
        name_concepts = _to_concepts(name_tokens)
        exp_data.append({
            "id": exp.id,
            "name": name,
            "user_id": exp.user_id,
            "description": desc,
            "is_public": exp.is_public == "true",
            "node_count": sum(len(level) for level in (exp.levels or [])),
            "keywords": keywords,
            "name_tokens": name_tokens,
            "concepts": concepts,
            "name_concepts": name_concepts,
        })

    df = Counter()
    for ed in exp_data:
        for c in ed["concepts"]:
            df[c] += 1
    n_docs = max(len(exp_data), 1)

    def concept_idf(c: str) -> float:
        return math.log(1.0 + n_docs / max(df[c], 1))

    rare_enough = {c for c, cnt in df.items() if cnt / n_docs <= 0.6}
    for ed in exp_data:
        ed["concepts"] = ed["concepts"] & rare_enough
        ed["name_concepts"] = ed["name_concepts"] & rare_enough

    candidates = []
    for i in range(len(exp_data)):
        for j in range(i + 1, len(exp_data)):
            a, b = exp_data[i], exp_data[j]
            shared_concepts = a["concepts"] & b["concepts"]
            name_shared = a["name_concepts"] & b["name_concepts"]
            shared_all = shared_concepts | name_shared
            if not shared_all:
                continue

            meaningful = {
                c for c in shared_all
                if c not in _WEAK_STUBS and (c in _CONCEPT_LABELS or len(c) >= 5 or c in name_shared)
            }
            if not meaningful and not name_shared:
                continue
            if not meaningful:
                meaningful = set(name_shared)

            union = a["concepts"] | b["concepts"]
            jaccard = len(shared_concepts) / max(len(union), 1) if union else 0.0
            idf_sum = sum(concept_idf(c) for c in meaningful)
            score = jaccard * (1.0 + idf_sum / 5.0)
            score = max(score, 0.05 * idf_sum)

            same_author = a["user_id"] == b["user_id"]
            if name_shared:
                score += 0.08 * sum(concept_idf(c) for c in name_shared)
            if same_author and meaningful:
                score *= 1.15

            desc_a = _extract_graph_keywords(a["description"])
            desc_b = _extract_graph_keywords(b["description"])
            desc_shared = _to_concepts(desc_a) & _to_concepts(desc_b) & rare_enough
            if desc_shared:
                score += 0.03 * sum(concept_idf(c) for c in desc_shared)
                meaningful |= desc_shared

            if score < 0.06:
                continue

            topics = sorted(meaningful, key=lambda c: -concept_idf(c))[:6]
            topic_labels = [_concept_label(c) for c in topics]

            reasons = []
            if name_shared:
                labels = ", ".join(_concept_label(c) for c in sorted(name_shared, key=lambda c: -concept_idf(c))[:3])
                reasons.append(f"Совпадение в названии: {labels}")
            topic_only = meaningful - name_shared
            if topic_only:
                labels = ", ".join(_concept_label(c) for c in sorted(topic_only, key=lambda c: -concept_idf(c))[:3])
                reasons.append(f"Общие темы: {labels}")
            if same_author:
                reasons.append(f"Один автор: {a['user_id']}")
            if desc_shared:
                labels = ", ".join(_concept_label(c) for c in sorted(desc_shared, key=lambda c: -concept_idf(c))[:2])
                reasons.append(f"Похожие описания: {labels}")

            if name_shared and same_author:
                summary = f"Оба про «{topic_labels[0]}» · один автор"
            elif name_shared:
                summary = f"Схожие названия вокруг темы «{topic_labels[0]}»"
            elif same_author:
                summary = f"Общая тема «{topic_labels[0]}» у одного автора"
            else:
                summary = f"Связаны темой «{topic_labels[0]}»"

            candidates.append({
                "source": a["id"],
                "target": b["id"],
                "weight": round(score, 4),
                "shared_count": len(meaningful),
                "common": topic_labels,
                "topics": topic_labels,
                "reasons": reasons[:4],
                "summary": summary,
                "same_author": same_author,
                "name_overlap": bool(name_shared),
            })

    candidates.sort(key=lambda x: -x["weight"])
    max_edges = max(10, int(len(exp_data) * 3))
    links = candidates[:max_edges]
    if len(links) > 10:
        floor = links[0]["weight"] * 0.28
        strong = [l for l in links if l["weight"] >= floor]
        links = strong if len(strong) >= 3 else links

    degree = Counter()
    for l in links:
        degree[l["source"]] += 1
        degree[l["target"]] += 1

    nodes = []
    for ed in exp_data:
        top_concepts = sorted(ed["concepts"] | ed["name_concepts"], key=lambda c: -concept_idf(c))[:8]
        top_kw = [_concept_label(c) for c in top_concepts]
        desc = ed["description"]
        nodes.append({
            "id": ed["id"],
            "label": ed["name"] or ed["id"],
            "user_id": ed["user_id"],
            "username": ed["user_id"],
            "description": (desc[:160] + "…") if len(desc) > 160 else desc,
            "node_count": ed["node_count"],
            "is_own": ed["user_id"] == user,
            "is_friend": ed["user_id"] in friend_ids,
            "is_public": ed["is_public"],
            "degree": degree[ed["id"]],
            "keywords": top_kw,
        })

    max_weight = max((l["weight"] for l in links), default=1.0) or 1.0
    for l in links:
        strength = round(l["weight"] / max_weight, 3)
        l["strength"] = strength
        l["strength_label"] = _strength_label(strength)

    return JSONResponse({
        "nodes": nodes,
        "links": links,
        "meta": {
            "scope": scope,
            "node_count": len(nodes),
            "link_count": len(links),
            "max_weight": max_weight,
            "candidates_scored": len(candidates),
        },
    })


@app.get("/experiment_graph", response_class=HTMLResponse)
async def experiment_graph_page(request: Request, user: str, db: Session = Depends(get_db)):
    if not EXPERIMENT_GRAPH_ENABLED:
        return RedirectResponse(
            url=f"{BASE_PATH}/dashboard?user={user}",
            status_code=302,
        )
    exps = db.query(ExperimentDB).filter(ExperimentDB.user_id == user).all()
    return templates.TemplateResponse(
        request, "experiment_graph.html",
        {"user": user, "experiments": exps, "active_page": "graph"}
    )

@app.get("/pricing", response_class=HTMLResponse)
async def pricing_page(request: Request, user: str = "", db: Session = Depends(get_db)):
    plan = "free"
    if user:
        u = db.query(UserDB).filter(UserDB.username == user).first()
        if u:
            plan = sync_user_plan_effective(u, db)
    plans = [
        {
            "id": "free",
            "name": "Free",
            "price": "0 ₽",
            "period": "навсегда",
            "desc": "Журнал без ИИ-агента — старт без карты",
            "features": [
                "Эксперименты и узлы",
                "OCR и голосовой ввод",
                "Отчёты DOCX / PDF",
                "Без ИИ-агента",
            ],
            "cta": "Текущий тариф" if plan == "free" else "Перейти",
            "highlight": False,
        },
        {
            "id": "student",
            "name": "Pro",
            "price": "319 ₽",
            "period": "в месяц",
            "desc": "Агент и все возможности Bio-Notebook",
            "features": [
                "Всё из Free",
                "ИИ-агент по вашим записям",
                "Презентации из отчёта",
                "Друзья, чаты, 20 ГБ",
            ],
            "cta": "Текущий тариф" if plan in ("student", "pro") else "Выбрать 319 ₽",
            "highlight": True,
        },
        {
            "id": "corp",
            "name": "LabOS",
            "price": "обсудим",
            "period": "для кафедр и лабораторий",
            "desc": "Кастомные условия, роли и внедрение",
            "features": [
                "Всё из Pro",
                "Роли и командная работа",
                "Договор, счёт, данные в РФ",
                "Персональный менеджер",
            ],
            "cta": "Текущий тариф" if plan in ("corp", "labos") else "Давайте обсудим!",
            "highlight": False,
        },
    ]
    return templates.TemplateResponse(
        request, "pricing.html",
        {
            "user": user,
            "current_plan": plan,
            "plan_label": PLAN_LABELS.get(plan, "Бесплатный"),
            "plans": plans,
            "active_page": "pricing",
        },
    )

# ── Friends ──

@app.get("/friends", response_class=HTMLResponse)
async def friends_page(request: Request, user: str, db: Session = Depends(get_db)):
    session_user = get_session_username(request)
    if session_user:
        user = session_user
    # Friends (accepted)
    friends = db.query(FriendDB).filter(
        (FriendDB.user_id == user) | (FriendDB.friend_id == user),
        FriendDB.status == "accepted",
    ).all()

    friend_usernames = set()
    for f in friends:
        friend_usernames.add(f.user_id if f.user_id != user else f.friend_id)

    friend_details = db.query(UserDB).filter(UserDB.username.in_(friend_usernames)).all() if friend_usernames else []

    # Pending requests TO me
    pending_in = db.query(FriendDB).filter(
        FriendDB.friend_id == user,
        FriendDB.status == "pending",
    ).all()

    # Pending requests FROM me
    pending_out = db.query(FriendDB).filter(
        FriendDB.user_id == user,
        FriendDB.status == "pending",
    ).all()

    pending_senders = db.query(UserDB).filter(
        UserDB.username.in_([p.user_id for p in pending_in])
    ).all() if pending_in else []

    return templates.TemplateResponse(
        request, "friends.html",
        {
            "user": user,
            "friends": friend_details,
            "pending_in": pending_senders,
            "pending_out": [p.friend_id for p in pending_out],
            "active_page": "friends",
        }
    )

@app.get("/profile/{username}", response_class=HTMLResponse)
async def user_profile(request: Request, username: str, user: str, db: Session = Depends(get_db)):
    """Public profile of a researcher. `user` = current viewer."""
    profile = db.query(UserDB).filter(UserDB.username == username).first()
    if not profile:
        return HTMLResponse("Пользователь не найден", status_code=404)

    is_own = user == username
    fr_status = "none"
    if not is_own:
        fr = db.query(FriendDB).filter(
            ((FriendDB.user_id == user) & (FriendDB.friend_id == username)) |
            ((FriendDB.user_id == username) & (FriendDB.friend_id == user)),
        ).first()
        if fr:
            if fr.status == "accepted":
                fr_status = "accepted"
            elif fr.status == "pending":
                fr_status = "pending_out" if fr.user_id == user else "pending_in"

    public_exps = db.query(ExperimentDB).filter(
        ExperimentDB.user_id == username,
        ExperimentDB.is_public == "true",
    ).all()
    total_exps = db.query(ExperimentDB).filter(ExperimentDB.user_id == username).count()
    friends_count = db.query(FriendDB).filter(
        (FriendDB.user_id == username) | (FriendDB.friend_id == username),
        FriendDB.status == "accepted",
    ).count()

    show_email = is_own or fr_status == "accepted"

    return templates.TemplateResponse(
        request, "profile.html",
        {
            "user": user,
            "profile": profile,
            "is_own": is_own,
            "fr_status": fr_status,
            "public_exps": public_exps,
            "total_exps": total_exps,
            "friends_count": friends_count,
            "show_email": show_email,
            "active_page": "friends",
        },
    )


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, user: str, db: Session = Depends(get_db)):
    """Full-page chat with friends."""
    return templates.TemplateResponse(
        request, "chat.html",
        {
            "user": user,
            "start_chat": "",
            "active_page": "chat",
        },
    )


@app.get("/chat/{username}", response_class=HTMLResponse)
async def chat_with_user(request: Request, user: str, username: str, db: Session = Depends(get_db)):
    """Open chat with a specific user."""
    return templates.TemplateResponse(
        request, "chat.html",
        {
            "user": user,
            "start_chat": username,
            "active_page": "chat",
        },
    )


@app.post("/api/friends/request")
async def send_friend_request(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    user_id = body.get("user_id")
    friend_id = body.get("friend_id")
    session_user = get_session_username(request)
    if not session_user or session_user != user_id:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not user_id or not friend_id or user_id == friend_id:
        return JSONResponse({"error": "Invalid request"}, status_code=400)
    existing = db.query(FriendDB).filter(
        ((FriendDB.user_id == user_id) & (FriendDB.friend_id == friend_id)) |
        ((FriendDB.user_id == friend_id) & (FriendDB.friend_id == user_id))
    ).first()
    if existing:
        return JSONResponse({"error": "Запрос уже отправлен или пользователь уже в друзьях"}, status_code=400)
    friend = FriendDB(user_id=user_id, friend_id=friend_id, status="pending")
    db.add(friend)
    db.commit()
    return JSONResponse({"ok": True, "message": "Запрос в друзья отправлен"})

@app.post("/api/friends/accept")
async def accept_friend_request(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    session_user = get_session_username(request)
    if not session_user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    friend_id = (body.get("friend_id") or "").strip()
    if not friend_id:
        return JSONResponse({"error": "Invalid request"}, status_code=400)
    user_id = session_user
    req = db.query(FriendDB).filter(
        FriendDB.user_id == friend_id,
        FriendDB.friend_id == user_id,
        FriendDB.status == "pending",
    ).first()
    if not req:
        return JSONResponse({"error": "Запрос не найден"}, status_code=404)
    req.status = "accepted"
    db.commit()
    return JSONResponse({"ok": True, "message": "Запрос принят"})

@app.post("/api/friends/reject")
async def reject_friend_request(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    session_user = get_session_username(request)
    if not session_user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    friend_id = (body.get("friend_id") or "").strip()
    if not friend_id:
        return JSONResponse({"error": "Invalid request"}, status_code=400)
    user_id = session_user
    req = db.query(FriendDB).filter(
        FriendDB.user_id == friend_id,
        FriendDB.friend_id == user_id,
        FriendDB.status == "pending",
    ).first()
    if not req:
        return JSONResponse({"error": "Запрос не найден"}, status_code=404)
    req.status = "rejected"
    db.commit()
    return JSONResponse({"ok": True, "message": "Запрос отклонён"})

@app.post("/api/friends/remove")
async def remove_friend(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    user_id = body.get("user_id")
    friend_id = body.get("friend_id")
    session_user = get_session_username(request)
    if not session_user or session_user != user_id:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    req = db.query(FriendDB).filter(
        ((FriendDB.user_id == user_id) & (FriendDB.friend_id == friend_id)) |
        ((FriendDB.user_id == friend_id) & (FriendDB.friend_id == user_id)),
        FriendDB.status == "accepted",
    ).first()
    if not req:
        return JSONResponse({"error": "Дружба не найдена"}, status_code=404)
    db.delete(req)
    db.commit()
    return JSONResponse({"ok": True, "message": "Пользователь удалён из друзей"})

@app.get("/api/friends/check")
async def check_friendship(user: str, other: str, db: Session = Depends(get_db)):
    req = db.query(FriendDB).filter(
        ((FriendDB.user_id == user) & (FriendDB.friend_id == other)) |
        ((FriendDB.user_id == other) & (FriendDB.friend_id == user)),
    ).first()
    if not req:
        return JSONResponse({"status": "none"})
    return JSONResponse({"status": req.status})


# ── Friend Chats API ──

def _chat_read_attr(chat: FriendChatDB, username: str, *, peer: bool = False) -> str:
    """Column name for this user's (or peer's) last-read watermark on a chat."""
    if chat.user1 == username:
        return "user2_last_read_at" if peer else "user1_last_read_at"
    return "user1_last_read_at" if peer else "user2_last_read_at"


def _chat_last_read_at(chat: FriendChatDB, username: str) -> str:
    return getattr(chat, _chat_read_attr(chat, username), "") or ""


def _chat_peer_last_read_at(chat: FriendChatDB, username: str) -> str:
    return getattr(chat, _chat_read_attr(chat, username, peer=True), "") or ""


def _count_unread_in_chat(db, chat: FriendChatDB, username: str) -> int:
    last_read = _chat_last_read_at(chat, username)
    q = db.query(FriendMessageDB).filter(
        FriendMessageDB.chat_id == chat.id,
        FriendMessageDB.sender != username,
    )
    if last_read:
        q = q.filter(FriendMessageDB.created_at > last_read)
    return q.count()


@app.get("/api/friends/chats")
async def list_friend_chats(request: Request, db: Session = Depends(get_db)):
    user = get_session_username(request)
    if not user:
        return JSONResponse({"error": "LoginRequired"}, status_code=401)
    chats = db.query(FriendChatDB).filter(
        (FriendChatDB.user1 == user) | (FriendChatDB.user2 == user)
    ).order_by(FriendChatDB.last_time.desc()).all()
    result = []
    for c in chats:
        other = c.user2 if c.user1 == user else c.user1
        unread = _count_unread_in_chat(db, c, user)
        result.append({
            "id": c.id,
            "other": other,
            "last_message": c.last_message or "",
            "last_sender": c.last_sender or "",
            "last_time": c.last_time or c.created_at or "",
            "unread_count": unread,
            "has_unread": unread > 0,
        })
    return JSONResponse(result)


@app.post("/api/friends/chats/create")
async def create_friend_chat(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    user = get_session_username(request)
    other = body.get("other", "")
    if not user or not other:
        return JSONResponse({"error": "LoginRequired"}, status_code=401)
    if user == other:
        return JSONResponse({"error": "Cannot chat with yourself"}, status_code=400)
    # Check existing
    existing = db.query(FriendChatDB).filter(
        ((FriendChatDB.user1 == user) & (FriendChatDB.user2 == other)) |
        ((FriendChatDB.user1 == other) & (FriendChatDB.user2 == user)),
    ).first()
    if existing:
        return JSONResponse({"chat_id": existing.id})
    chat = FriendChatDB(user1=user, user2=other)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return JSONResponse({"chat_id": chat.id})


@app.get("/api/friends/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: str, request: Request, db: Session = Depends(get_db)):
    user = get_session_username(request)
    if not user:
        return JSONResponse({"error": "LoginRequired"}, status_code=401)
    chat = db.query(FriendChatDB).filter(FriendChatDB.id == chat_id).first()
    if not chat or (chat.user1 != user and chat.user2 != user):
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    msgs = db.query(FriendMessageDB).filter(
        FriendMessageDB.chat_id == chat_id
    ).order_by(FriendMessageDB.created_at.asc()).all()
    return JSONResponse({
        "messages": [
            {
                "id": m.id,
                "sender": m.sender,
                "content": m.content,
                "created_at": m.created_at,
            }
            for m in msgs
        ],
        "my_last_read_at": _chat_last_read_at(chat, user),
        "peer_last_read_at": _chat_peer_last_read_at(chat, user),
    })


@app.get("/api/friends/chats/{chat_id}/read-state")
async def chat_read_state(chat_id: str, request: Request, db: Session = Depends(get_db)):
    user = get_session_username(request)
    if not user:
        return JSONResponse({"error": "LoginRequired"}, status_code=401)
    chat = db.query(FriendChatDB).filter(FriendChatDB.id == chat_id).first()
    if not chat or (chat.user1 != user and chat.user2 != user):
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    return JSONResponse({
        "my_last_read_at": _chat_last_read_at(chat, user),
        "peer_last_read_at": _chat_peer_last_read_at(chat, user),
    })


@app.post("/api/friends/chats/{chat_id}/read")
async def mark_chat_read(chat_id: str, request: Request, db: Session = Depends(get_db)):
    user = get_session_username(request)
    if not user:
        return JSONResponse({"error": "LoginRequired"}, status_code=401)
    chat = db.query(FriendChatDB).filter(FriendChatDB.id == chat_id).first()
    if not chat or (chat.user1 != user and chat.user2 != user):
        return JSONResponse({"error": "Chat not found"}, status_code=404)

    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    up_to = (body.get("up_to") or "").strip()
    if up_to:
        read_at = up_to
    else:
        latest = (
            db.query(FriendMessageDB)
            .filter(FriendMessageDB.chat_id == chat_id)
            .order_by(FriendMessageDB.created_at.desc())
            .first()
        )
        read_at = (latest.created_at if latest else datetime.datetime.now().isoformat())

    attr = _chat_read_attr(chat, user)
    current = getattr(chat, attr, "") or ""
    if not current or read_at > current:
        setattr(chat, attr, read_at)
        db.commit()

    return JSONResponse({
        "ok": True,
        "my_last_read_at": getattr(chat, attr, "") or read_at,
        "peer_last_read_at": _chat_peer_last_read_at(chat, user),
    })


@app.post("/api/friends/chats/{chat_id}/send")
async def send_chat_message(chat_id: str, request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    user = get_session_username(request)
    if not user:
        return JSONResponse({"error": "LoginRequired"}, status_code=401)
    chat = db.query(FriendChatDB).filter(FriendChatDB.id == chat_id).first()
    if not chat or (chat.user1 != user and chat.user2 != user):
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    content = (body.get("content", "") or "").strip()
    if not content:
        return JSONResponse({"error": "Empty message"}, status_code=400)
    msg = FriendMessageDB(chat_id=chat_id, sender=user, content=content)
    db.add(msg)
    chat.last_message = content[:100]
    chat.last_sender = user
    chat.last_time = datetime.datetime.now().isoformat()
    db.commit()
    db.refresh(msg)
    return JSONResponse({
        "id": msg.id,
        "sender": msg.sender,
        "content": msg.content,
        "created_at": msg.created_at,
    })


@app.get("/api/friends/chats/{chat_id}/unread")
async def get_chat_unread(chat_id: str, request: Request, db: Session = Depends(get_db)):
    user = get_session_username(request)
    if not user:
        return JSONResponse({"error": "LoginRequired"}, status_code=401)
    chat = db.query(FriendChatDB).filter(FriendChatDB.id == chat_id).first()
    if not chat or (chat.user1 != user and chat.user2 != user):
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    timestamp = request.query_params.get("after", "0")
    msgs = db.query(FriendMessageDB).filter(
        FriendMessageDB.chat_id == chat_id,
        FriendMessageDB.sender != user,
        FriendMessageDB.created_at > timestamp,
    ).order_by(FriendMessageDB.created_at.asc()).all()
    return JSONResponse([
        {
            "id": m.id,
            "sender": m.sender,
            "content": m.content,
            "created_at": m.created_at,
        }
        for m in msgs
    ])


def _chat_message_preview(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return "Новое сообщение"
    if text.startswith("🎤||"):
        return "Голосовое сообщение"
    if text.startswith("📎||"):
        parts = text.split("||")
        if len(parts) > 2 and parts[2].strip():
            return parts[2].strip()
        return "Файл"
    if len(text) > 120:
        return text[:117] + "…"
    return text


@app.get("/api/friends/chats/unread-summary")
async def friends_chats_unread_summary(request: Request, db: Session = Depends(get_db)):
    """Unread incoming chat messages since client watermark (localStorage)."""
    user = get_session_username(request)
    if not user:
        return JSONResponse({"count": 0, "latest": None})

    after = (request.query_params.get("after") or "").strip()
    chats = db.query(FriendChatDB).filter(
        (FriendChatDB.user1 == user) | (FriendChatDB.user2 == user)
    ).all()
    chat_ids = [c.id for c in chats]
    if not chat_ids:
        return JSONResponse({"count": 0, "latest": None})

    chat_by_id = {c.id: c for c in chats}
    q = db.query(FriendMessageDB).filter(
        FriendMessageDB.chat_id.in_(chat_ids),
        FriendMessageDB.sender != user,
    )
    if after and after != "0":
        q = q.filter(FriendMessageDB.created_at > after)

    count = q.count()
    latest = q.order_by(FriendMessageDB.created_at.desc()).first()
    latest_payload = None
    if latest:
        chat = chat_by_id.get(latest.chat_id)
        friend = None
        if chat:
            friend = chat.user2 if chat.user1 == user else chat.user1
        latest_payload = {
            "sender": latest.sender,
            "friend": friend or latest.sender,
            "preview": _chat_message_preview(latest.content),
            "chat_id": latest.chat_id,
            "created_at": latest.created_at,
        }
    return JSONResponse({"count": count, "latest": latest_payload})


# ── Equipment ──

@app.get("/equipment", response_class=HTMLResponse)
async def equipment_page(request: Request, user: str, db: Session = Depends(get_db)):
    friends = db.query(FriendDB).filter(
        (FriendDB.user_id == user) | (FriendDB.friend_id == user),
        FriendDB.status == "accepted",
    ).all()
    friend_usernames = set()
    for f in friends:
        friend_usernames.add(f.user_id if f.user_id != user else f.friend_id)
    friend_usernames.add(user)  # include own
    
    all_equipment = db.query(EquipmentDB).filter(
        EquipmentDB.owner_id.in_(friend_usernames)
    ).order_by(EquipmentDB.created_at.desc()).all()
    
    # Get bookings for all equipment
    equip_ids = [e.id for e in all_equipment]
    bookings = db.query(EquipmentBookingDB).filter(
        EquipmentBookingDB.equipment_id.in_(equip_ids)
    ).order_by(EquipmentBookingDB.created_at.desc()).all() if equip_ids else []
    
    return templates.TemplateResponse(
        request, "equipment.html",
        {
            "user": user,
            "equipment_list": all_equipment,
            "bookings": bookings,
            "friends": list(friend_usernames - {user}),
            "active_page": "equipment",
        }
    )

@app.post("/api/equipment/add")
async def add_equipment(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    equip = EquipmentDB(
        owner_id=body.get("owner_id"),
        name=body.get("name", ""),
        description=body.get("description", ""),
        status="available",
    )
    db.add(equip)
    db.commit()
    return JSONResponse({"ok": True, "id": equip.id})

@app.post("/api/equipment/delete")
async def delete_equipment(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    equip = db.query(EquipmentDB).filter(
        EquipmentDB.id == body.get("id"),
        EquipmentDB.owner_id == body.get("user_id"),
    ).first()
    if not equip:
        return JSONResponse({"error": "Оборудование не найдено"}, status_code=404)
    db.delete(equip)
    db.commit()
    return JSONResponse({"ok": True})

@app.post("/api/equipment/book")
async def book_equipment(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    equip = db.query(EquipmentDB).filter(EquipmentDB.id == body.get("equipment_id")).first()
    if not equip:
        return JSONResponse({"error": "Оборудование не найдено"}, status_code=404)
    
    # Check for overlapping approved bookings
    existing = db.query(EquipmentBookingDB).filter(
        EquipmentBookingDB.equipment_id == body.get("equipment_id"),
        EquipmentBookingDB.status.in_(["pending", "approved"]),
        EquipmentBookingDB.start_date <= body.get("end_date", ""),
        EquipmentBookingDB.end_date >= body.get("start_date", ""),
    ).first()
    
    booking = EquipmentBookingDB(
        equipment_id=body.get("equipment_id"),
        user_id=body.get("user_id"),
        start_date=body.get("start_date", ""),
        end_date=body.get("end_date", ""),
        status="pending",
    )
    db.add(booking)
    db.commit()
    return JSONResponse({"ok": True, "id": booking.id, "warning": "Обнаружено пересечение бронирований" if existing else None})

@app.post("/api/equipment/booking/approve")
async def approve_booking(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    booking = db.query(EquipmentBookingDB).filter(EquipmentBookingDB.id == body.get("id")).first()
    if not booking:
        return JSONResponse({"error": "Бронирование не найдено"}, status_code=404)
    booking.status = "approved"
    db.commit()
    return JSONResponse({"ok": True})

@app.post("/api/equipment/booking/decline")
async def decline_booking(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    booking = db.query(EquipmentBookingDB).filter(EquipmentBookingDB.id == body.get("id")).first()
    if not booking:
        return JSONResponse({"error": "Бронирование не найдено"}, status_code=404)
    booking.status = "declined"
    db.commit()
    return JSONResponse({"ok": True})

@app.post("/api/equipment/booking/complete")
async def complete_booking(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    booking = db.query(EquipmentBookingDB).filter(EquipmentBookingDB.id == body.get("id")).first()
    if not booking:
        return JSONResponse({"error": "Бронирование не найдено"}, status_code=404)
    booking.status = "completed"
    db.commit()
    return JSONResponse({"ok": True})

@app.post("/api/equipment/booking/cancel")
async def cancel_booking(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    booking = db.query(EquipmentBookingDB).filter(
        EquipmentBookingDB.id == body.get("id"),
        EquipmentBookingDB.user_id == body.get("user_id"),
    ).first()
    if not booking:
        return JSONResponse({"error": "Бронирование не найдено"}, status_code=404)
    booking.status = "declined"
    db.commit()
    return JSONResponse({"ok": True})

@app.get("/api/equipment/bookings")
async def get_equipment_bookings(id: str, db: Session = Depends(get_db)):
    bookings = db.query(EquipmentBookingDB).filter(
        EquipmentBookingDB.equipment_id == id
    ).order_by(EquipmentBookingDB.created_at.asc()).all()
    return JSONResponse({
        "bookings": [
            {
                "id": b.id,
                "equipment_id": b.equipment_id,
                "user_id": b.user_id,
                "start_date": b.start_date,
                "end_date": b.end_date,
                "status": b.status,
            }
            for b in bookings
        ]
    })

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
    return RedirectResponse(url=f"{BASE_PATH}/experiment/{user}/{exp_id}", status_code=302)

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
        return RedirectResponse(url=f"{BASE_PATH}/dashboard?user={user}")

    node = _find_node(exp, node_id)
    if not node:
        return RedirectResponse(url=f"{BASE_PATH}/experiment/{user}/{exp_id}")

    # Get or create notes
    notes = db.query(NodeNotesDB).filter(NodeNotesDB.exp_id == exp_id, NodeNotesDB.node_id == node_id).first()
    if not notes:
        notes = NodeNotesDB(id=str(uuid.uuid4())[:8], exp_id=exp_id, node_id=node_id, notes_html="", chat_history=[])
        db.add(notes)
        db.commit()
        db.refresh(notes)

    resp = templates.TemplateResponse(
        request, "node_detail.html",
        {"user": user, "exp": exp, "node": node, "notes": notes, "is_public_view": False}
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp

# ── API: save notes ──

@app.post("/api/node/save")
async def save_node_notes(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    exp_id = body.get("exp_id")
    node_id = body.get("node_id")
    html = normalize_upload_urls(body.get("html", ""))
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

_whisper_model = None


def _ensure_whisper_runtime_env() -> None:
    cache_root = (os.environ.get("HF_HOME") or "/opt/bio-notebook/.cache/huggingface").strip()
    os.makedirs(cache_root, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("XDG_CACHE_HOME", os.path.dirname(cache_root.rstrip("/")))
    proxy = (os.environ.get("AI_HTTP_PROXY") or os.environ.get("HTTPS_PROXY") or "").strip()
    if proxy:
        os.environ.setdefault("HTTPS_PROXY", proxy)
        os.environ.setdefault("HTTP_PROXY", proxy)


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    _ensure_whisper_runtime_env()
    from faster_whisper import WhisperModel

    model_name = (os.environ.get("WHISPER_MODEL") or "base").strip() or "base"
    _whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _whisper_model


@app.post("/api/stt")
async def run_stt(file: UploadFile = File(...)):
    """
    Server-side transcription (works in Firefox too).
    Uses faster-whisper if available. Expects audio in webm/wav/m4a/etc.
    """
    try:
        model = _get_whisper_model()
    except ModuleNotFoundError:
        return JSONResponse(
            {"text": "Транскрибация недоступна на сервере (не установлен faster-whisper)."},
            status_code=501,
        )
    except Exception as e:
        print(f"STT model init error: {e}")
        return JSONResponse(
            {"text": "Не удалось загрузить модель распознавания. Попробуйте позже или обратитесь в поддержку."},
            status_code=503,
        )

    data = await file.read()
    if not data:
        return JSONResponse({"text": "Пустой аудиофайл."}, status_code=400)

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.webm")
        with open(in_path, "wb") as f:
            f.write(data)

        # First try decoding original container (works if PyAV supports it)
        try:
            segments, _info = model.transcribe(in_path, language="ru")
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
            segments, _info = model.transcribe(out_path, language="ru")
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
            
    return JSONResponse({"url": upload_public_url(fname)})

# ── API: AI chat ──

def _strip_code_fences(code: str) -> str:
    code = (code or "").strip()
    m = re.match(r"^```(?:python|py|bash|r)?\s*\n(.*?)```$", code, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return code


def _extract_tag_block(reply: str, open_tag: str, close_tag: str) -> tuple[str | None, str]:
    """Return (inner content, reply with block removed) or (None, reply)."""
    pattern = re.compile(
        re.escape(open_tag) + r"\s*(.*?)\s*" + re.escape(close_tag),
        re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(reply or "")
    if not m:
        return None, reply
    inner = m.group(1).strip()
    new_reply = ((reply or "")[: m.start()] + (reply or "")[m.end() :]).strip()
    return inner, new_reply


def _extract_python_fences(text: str) -> list[str]:
    blocks: list[str] = []
    for m in re.finditer(r"```(?:python|py)\s*\n(.*?)```", text or "", re.DOTALL | re.IGNORECASE):
        body = (m.group(1) or "").strip()
        if body:
            blocks.append(body)
    return blocks


def _user_wants_notebook_change(user_message: str) -> bool:
    u = (user_message or "").lower()
    if not u:
        return False
    keys = (
        "таблиц", "блокнот", "ячей", "notebook", "insert_code", "csv",
        "статистик", "добав", "встав", "код", "анализ", "p-value", "p value",
        "сводн", "to_csv", "run",
    )
    return any(k in u for k in keys)


def _parse_assistant_notebook_actions(msg: str, user_message: str = "") -> tuple[str, dict]:
    """Extract notebook edit actions from the model reply."""
    reply = msg or ""
    actions: dict = {}

    edit_inner, reply = _extract_tag_block(reply, "[EDIT_DOCUMENT]", "[/EDIT_DOCUMENT]")
    if edit_inner is not None:
        actions["edit_html"] = edit_inner

    append_inner, reply = _extract_tag_block(reply, "[APPEND_DOCUMENT]", "[/APPEND_DOCUMENT]")
    if append_inner is not None:
        actions["append_html"] = append_inner

    cell_match = re.search(
        r"\[EDIT_CODE_CELL:(\d+)\]\s*(.*?)\s*\[/EDIT_CODE_CELL\]",
        reply,
        re.DOTALL | re.IGNORECASE,
    )
    if cell_match:
        actions["edit_code_cell"] = {
            "index": int(cell_match.group(1)),
            "code": _strip_code_fences(cell_match.group(2)),
        }
        reply = re.sub(
            r"\[EDIT_CODE_CELL:\d+\].*?\[/EDIT_CODE_CELL\]",
            "",
            reply,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    insert_inner, reply = _extract_tag_block(reply, "[INSERT_CODE]", "[/INSERT_CODE]")
    if insert_inner is not None:
        actions["insert_code"] = _strip_code_fences(insert_inner)

    if not actions and _user_wants_notebook_change(user_message):
        blocks = _extract_python_fences(reply)
        if blocks:
            actions["insert_code"] = max(blocks, key=len)
            reply = re.sub(r"```(?:python|py)\s*\n.*?```", "", reply, count=1, flags=re.DOTALL | re.IGNORECASE).strip()

    if actions and not reply.strip():
        if actions.get("insert_code"):
            reply = "Добавил код в блокнот. Нажмите Run в ячейке, чтобы увидеть таблицы и CSV."
        elif actions.get("edit_code_cell"):
            reply = f"Обновил ячейку {actions['edit_code_cell']['index']}. Нажмите Run, чтобы выполнить."
        elif actions.get("append_html"):
            reply = "Добавил таблицу в документ блокнота."
        elif actions.get("edit_html"):
            reply = "Обновил документ в блокноте."

    return reply.strip() or "Готово.", actions


def _format_notebook_context(notebook_context) -> str:
    if not notebook_context:
        return "Ячеек кода пока нет."
    lines = []
    for cell in notebook_context:
        if not isinstance(cell, dict):
            continue
        idx = cell.get("index")
        code = (cell.get("code") or "").strip()
        if idx is None:
            continue
        preview = code if len(code) <= 2000 else code[:2000] + "\n# ... (обрезано)"
        lines.append(f"Ячейка {idx}:\n{preview or '(пусто)'}")
    return "\n\n".join(lines) if lines else "Ячеек кода пока нет."


@app.post("/api/chat")
async def chat_with_ai(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    exp_id = body.get("exp_id")
    node_id = body.get("node_id")
    user_message = body.get("message", "")
    regenerate = bool(body.get("regenerate"))
    html_context = body.get("html_context", "")
    notebook_context = body.get("notebook_context") or []

    notes = db.query(NodeNotesDB).filter(NodeNotesDB.exp_id == exp_id, NodeNotesDB.node_id == node_id).first()
    if not notes:
        notes = NodeNotesDB(
            id=str(uuid.uuid4())[:8],
            exp_id=exp_id,
            node_id=node_id,
            notes_html="",
            chat_history=[],
        )
        db.add(notes)
        db.commit()
        db.refresh(notes)

    history = list(notes.chat_history) if notes.chat_history else []

    # Regenerate: truncate history at the matching user turn, then re-ask
    if regenerate and history:
        cut = None
        for i in range(len(history) - 1, -1, -1):
            h = history[i]
            if h.get("role") == "user" and h.get("content") == user_message:
                cut = i
                break
        if cut is not None:
            history = history[:cut]

    system_prompt = (
        "Ты — ассистент-учёный для биоинформатики и работаешь с лабораторным блокнотом пользователя. "
        "Блокнот содержит текстовые заметки и Python-ячейки кода.\n\n"
        "ВАЖНО — куда писать ответ:\n"
        "1) Если пользователь просит написать/добавить/вставить код, скрипт, пайплайн, анализ — "
        "НЕ выводи длинный код в чат. Оберни только Python-код в [INSERT_CODE]...[/INSERT_CODE].\n"
        "2) Если нужно изменить существующую ячейку N (нумерация с 1) — "
        "используй [EDIT_CODE_CELL:N]...[/EDIT_CODE_CELL].\n"
        "3) Если нужно изменить весь текст заметки — ПОЛНЫЙ HTML в [EDIT_DOCUMENT]...[/EDIT_DOCUMENT].\n"
        "4) Если нужно ДОБАВИТЬ HTML-таблицу/блок в конец заметки (не затирая код-ячейки) — "
        "[APPEND_DOCUMENT]...[/APPEND_DOCUMENT] с фрагментом HTML (<table>…</table> и т.п.).\n"
        "5) В чате пиши кратко (1–2 предложения): что сделал. НЕ пиши «добавил в блокнот», если не использовал теги из п.1–4.\n"
        "6) Внутри тегов INSERT_CODE и EDIT_CODE_CELL не используй markdown-ограждения ```.\n"
        "7) Запросы про таблицы, статистику (n, mean, SD, SEM, median, min/max), тесты (p-value), CSV — "
        "это код: INSERT_CODE или EDIT_CODE_CELL. Код должен: посчитать таблицы, "
        "сохранить summary_stats.csv и test_results.csv (или аналогичные имена), "
        "вывести обе через print(df.to_string(index=False)).\n"
        "8) Если пользователь просит «в блокнот», «в ячейку», «добавь таблицу» — обязательно INSERT_CODE, EDIT_CODE_CELL или APPEND_DOCUMENT."
    )
    system_prompt += f"\n\nТекущие ячейки кода:\n{_format_notebook_context(notebook_context)}"
    if html_context:
        trimmed_html = html_context if len(html_context) <= 12000 else html_context[:12000] + "\n<!-- ... -->"
        system_prompt += f"\n\nТекущий HTML документа (для текстовых правок):\n{trimmed_html}"

    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-20:]:  # Keep recent history
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_message})

    if not AI_API_KEY or not AI_API_URL or not AI_CHAT_MODEL:
        assistant_msg = "AI‑ассистент не настроен на сервере (нет переменных окружения AI_API_KEY/AI_API_URL/AI_CHAT_MODEL)."
        tg_notify_ai_issue("chat", assistant_msg)
    else:
        try:
            async with ai_http_client(timeout=90) as client:
                resp = await client.post(
                    AI_API_URL,
                    headers={
                        "Authorization": f"Bearer {AI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": AI_CHAT_MODEL,
                        "messages": messages,
                    },
                )
                try:
                    data = resp.json()
                except Exception:
                    data = {"error": (resp.text or "")[:300]}

                if resp.status_code >= 400:
                    err = ""
                    if isinstance(data, dict):
                        err = data.get("error", {}).get("message") or data.get("error") or str(data.get("message", ""))
                    assistant_msg = f"Ошибка AI API ({resp.status_code}): {err or 'неизвестная ошибка'}"
                    tg_notify_ai_issue("chat", assistant_msg)
                else:
                    assistant_msg = ""
                    if isinstance(data, dict):
                        assistant_msg = (
                            (data.get("choices") or [{}])[0].get("message", {}).get("content")
                            or ""
                        )
                    if not assistant_msg:
                        assistant_msg = "AI вернул пустой ответ."
                        tg_notify_ai_issue("chat", assistant_msg)
        except Exception as e:
            assistant_msg = format_ai_request_error(e)
            tg_notify_ai_issue("chat", assistant_msg)

    assistant_msg, notebook_actions = _parse_assistant_notebook_actions(assistant_msg, user_message)

    # Save to history
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_msg})
    notes.chat_history = history
    db.commit()

    return JSONResponse({
        "reply": assistant_msg,
        "edit_html": notebook_actions.get("edit_html"),
        "append_html": notebook_actions.get("append_html"),
        "insert_code": notebook_actions.get("insert_code"),
        "edit_code_cell": notebook_actions.get("edit_code_cell"),
    })


@app.post("/api/chat/clear")
async def clear_chat_history(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    exp_id = body.get("exp_id")
    node_id = body.get("node_id")
    if not exp_id or not node_id:
        return JSONResponse({"ok": False, "error": "missing ids"}, status_code=400)

    notes = db.query(NodeNotesDB).filter(NodeNotesDB.exp_id == exp_id, NodeNotesDB.node_id == node_id).first()
    if notes:
        notes.chat_history = []
        db.commit()
    return JSONResponse({"ok": True})

import sys
import io
from contextlib import redirect_stdout

# Per-experiment kernel states (shared variables like in Jupyter)
kernels = {}
_kernel_exec_locks: dict[str, threading.Lock] = {}
_KERNEL_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


def _kernel_output_dir(exp_id: str) -> str:
    safe = re.sub(r"[^\w.\-]", "", (exp_id or "default"))[:80] or "default"
    path = os.path.join(KERNEL_OUTPUT_ROOT, safe)
    os.makedirs(path, exist_ok=True)
    return path


def _snapshot_kernel_artifacts(dir_path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        names = os.listdir(dir_path)
    except OSError:
        return out
    for name in names:
        fp = os.path.join(dir_path, name)
        if not os.path.isfile(fp):
            continue
        if os.path.splitext(name)[1].lower() not in _KERNEL_IMAGE_EXTS:
            continue
        try:
            out[name] = os.path.getmtime(fp)
        except OSError:
            pass
    return out


def _kernel_artifacts_changed(dir_path: str, before: dict[str, float]) -> list[str]:
    after = _snapshot_kernel_artifacts(dir_path)
    changed: list[str] = []
    for name in set(before) | set(after):
        if after.get(name, -1) != before.get(name, -2) and name in after:
            changed.append(name)
    return sorted(changed)


COMPUTE_UNITS = {
    "rtx4090": {"id": "rtx4090", "name": "RTX 4090", "vram": "24 GB"},
    "tesla_a100": {"id": "tesla_a100", "name": "Tesla A100", "vram": "80 GB"},
    "rtx3090": {"id": "rtx3090", "name": "RTX 3090", "vram": "24 GB"},
}

@app.post("/api/kernel/execute")
async def execute_code(request: Request):
    body = await request.json()
    exp_id = body.get("exp_id") or "default"
    code = body.get("code")
    compute_unit_id = (body.get("compute_unit") or "rtx4090").strip()
    compute_unit = COMPUTE_UNITS.get(compute_unit_id, COMPUTE_UNITS["rtx4090"])

    if exp_id not in kernels:
        kernels[exp_id] = {"__builtins__": __builtins__}

    # Expose selected accelerator metadata to the notebook namespace
    kernels[exp_id]["COMPUTE_UNIT"] = compute_unit["name"]
    kernels[exp_id]["COMPUTE_VRAM"] = compute_unit["vram"]
    kernels[exp_id]["COMPUTE_UNIT_ID"] = compute_unit["id"]

    out_dir = _kernel_output_dir(exp_id)
    kernels[exp_id]["KERNEL_OUTPUT_DIR"] = out_dir
    lock = _kernel_exec_locks.setdefault(exp_id, threading.Lock())
    before_artifacts = _snapshot_kernel_artifacts(out_dir)

    f = io.StringIO()
    error = None
    images: list[dict[str, str]] = []
    try:
        os.environ["MPLBACKEND"] = "Agg"
        with lock:
            prev_cwd = os.getcwd()
            try:
                os.chdir(out_dir)
                with redirect_stdout(f):
                    exec(code, kernels[exp_id])
            finally:
                os.chdir(prev_cwd)
        output = f.getvalue()
        for name in _kernel_artifacts_changed(out_dir, before_artifacts):
            images.append(
                {
                    "name": name,
                    "url": f"{BASE_PATH}/api/kernel/artifact/{quote(exp_id, safe='')}/{quote(name, safe='')}",
                }
            )
    except Exception as e:
        output = str(e)
        error = True

    return JSONResponse({
        "output": output,
        "error": error,
        "images": images,
        "output_dir": out_dir,
        "compute_unit": compute_unit,
    })


@app.get("/api/kernel/artifact/{exp_id}/{filename}")
async def kernel_artifact(exp_id: str, filename: str):
    safe_name = os.path.basename(filename or "")
    if not safe_name or safe_name != filename or ".." in filename:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    if os.path.splitext(safe_name)[1].lower() not in _KERNEL_IMAGE_EXTS:
        return JSONResponse({"error": "Unsupported type"}, status_code=400)
    path = os.path.join(_kernel_output_dir(exp_id), safe_name)
    if not os.path.isfile(path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    media = "application/octet-stream"
    low = safe_name.lower()
    if low.endswith(".png"):
        media = "image/png"
    elif low.endswith((".jpg", ".jpeg")):
        media = "image/jpeg"
    elif low.endswith(".gif"):
        media = "image/gif"
    elif low.endswith(".webp"):
        media = "image/webp"
    elif low.endswith(".svg"):
        media = "image/svg+xml"
    return FileResponse(path, media_type=media, filename=safe_name)

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
            async with ai_http_client(timeout=30) as client:
                resp = await client.post(
                    AI_API_URL,
                    headers={
                        "Authorization": f"Bearer {AI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": AI_CODE_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                if resp.status_code >= 400 or not isinstance(data, dict):
                    completion = ""
                else:
                    completion = (
                        (data.get("choices") or [{}])[0].get("message", {}).get("content")
                        or ""
                    ).strip()
                # Remove markdown backticks if any
                if completion.startswith("```python"):
                    completion = completion.replace("```python", "").replace("```", "").strip()
                elif completion.startswith("```"):
                    completion = completion.replace("```", "").strip()
        except Exception as e:
            completion = ""

    return JSONResponse({"completion": completion})


# ── API: Wiki pages for experiment ──

EXPERIMENT_FILES_DIR = os.environ.get(
    "EXPERIMENT_FILES_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "experiment_files"),
)
os.makedirs(EXPERIMENT_FILES_DIR, exist_ok=True)


@app.get("/experiment/{user}/{exp_id}/wiki")
async def wiki_page(request: Request, user: str, exp_id: str, db: Session = Depends(get_db)):
    exp = db.query(ExperimentDB).filter(ExperimentDB.id == exp_id).first()
    if not exp:
        return HTMLResponse("Experiment not found", status_code=404)
    pages = db.query(WikiPageDB).filter(WikiPageDB.exp_id == exp_id).order_by(WikiPageDB.updated_at.desc()).all()
    session_user = get_session_username(request)
    is_owner = session_user == user if session_user else False
    return templates.TemplateResponse(
        request,
        "wiki.html",
        {
            "user": user,
            "exp": exp,
            "pages": pages,
            "viewing_user": session_user or user,
            "is_owner": is_owner,
            "active_page": "dashboard",
        },
    )


@app.get("/experiment/{user}/{exp_id}/wiki/{page_id}")
async def wiki_view_page(request: Request, user: str, exp_id: str, page_id: str, db: Session = Depends(get_db)):
    exp = db.query(ExperimentDB).filter(ExperimentDB.id == exp_id).first()
    page = db.query(WikiPageDB).filter(WikiPageDB.id == page_id, WikiPageDB.exp_id == exp_id).first()
    if not exp or not page:
        return HTMLResponse("Page not found", status_code=404)
    session_user = get_session_username(request)
    is_owner = session_user == user if session_user else False
    return templates.TemplateResponse(
        request,
        "wiki_edit.html",
        {
            "user": user,
            "exp": exp,
            "page": page,
            "viewing_user": session_user or user,
            "is_owner": is_owner,
            "active_page": "dashboard",
        },
    )


@app.post("/api/wiki/save")
async def wiki_save(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    exp_id = body.get("exp_id")
    page_id = body.get("page_id")
    title = body.get("title", "").strip()
    content = body.get("content", "")
    username = get_session_username(request) or body.get("username", "")
    if not exp_id or not title:
        return JSONResponse({"ok": False, "error": "missing fields"}, status_code=400)
    if page_id:
        page = db.query(WikiPageDB).filter(WikiPageDB.id == page_id, WikiPageDB.exp_id == exp_id).first()
        if page:
            page.title = title
            page.content = content
            page.updated_at = datetime.datetime.now().isoformat()
            db.commit()
            return JSONResponse({"ok": True, "page_id": page.id})
    # Create new page
    page = WikiPageDB(
        exp_id=exp_id,
        title=title,
        content=content,
        created_by=username,
    )
    db.add(page)
    db.commit()
    return JSONResponse({"ok": True, "page_id": page.id})


@app.post("/api/wiki/delete")
async def wiki_delete(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    page_id = body.get("page_id")
    exp_id = body.get("exp_id")
    page = db.query(WikiPageDB).filter(WikiPageDB.id == page_id, WikiPageDB.exp_id == exp_id).first()
    if page:
        db.delete(page)
        db.commit()
    return JSONResponse({"ok": True})


@app.get("/api/experiment/{exp_id}/wiki/pages")
async def list_wiki_pages(exp_id: str, db: Session = Depends(get_db)):
    pages = (
        db.query(WikiPageDB)
        .filter(WikiPageDB.exp_id == exp_id)
        .order_by(WikiPageDB.updated_at.desc())
        .all()
    )
    return JSONResponse({
        "pages": [
            {
                "id": p.id,
                "title": p.title,
                "content": p.content or "",
                "created_at": p.created_at,
                "updated_at": p.updated_at,
                "created_by": p.created_by or "",
            }
            for p in pages
        ]
    })


# ── API: File exchange for experiment ──


@app.get("/api/experiment/{exp_id}/files")
async def list_experiment_files(exp_id: str, db: Session = Depends(get_db)):
    files = (
        db.query(ExperimentFileDB)
        .filter(ExperimentFileDB.exp_id == exp_id)
        .order_by(ExperimentFileDB.created_at.desc())
        .all()
    )
    return JSONResponse([
        {
            "id": f.id,
            "original_name": f.original_name,
            "file_size": int(f.file_size or 0),
            "mime_type": f.mime_type,
            "uploaded_by": f.uploaded_by,
            "created_at": f.created_at,
        }
        for f in files
    ])


@app.post("/api/experiment/{exp_id}/files/upload")
async def upload_experiment_file(
    request: Request,
    exp_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    ext = file.filename.split(".")[-1] if "." in file.filename else "bin"
    fname = f"{uuid.uuid4().hex[:12]}.{ext}"
    exp_dir = os.path.join(EXPERIMENT_FILES_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, fname)

    total = 0
    with open(path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
            total += len(chunk)

    username = get_session_username(request) or "unknown"
    ef = ExperimentFileDB(
        exp_id=exp_id,
        filename=fname,
        original_name=file.filename or fname,
        file_size=str(total),
        mime_type=file.content_type or "application/octet-stream",
        uploaded_by=username,
    )
    db.add(ef)
    db.commit()
    return JSONResponse({"ok": True, "file_id": ef.id})


@app.get("/api/experiment/{exp_id}/files/{file_id}/download")
async def download_experiment_file(exp_id: str, file_id: str, db: Session = Depends(get_db)):
    ef = db.query(ExperimentFileDB).filter(ExperimentFileDB.id == file_id, ExperimentFileDB.exp_id == exp_id).first()
    if not ef:
        return JSONResponse({"error": "File not found"}, status_code=404)
    path = os.path.join(EXPERIMENT_FILES_DIR, exp_id, ef.filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "File not on disk"}, status_code=404)
    from fastapi.responses import FileResponse
    return FileResponse(path, filename=ef.original_name, media_type=ef.mime_type)


@app.post("/api/experiment/{exp_id}/files/delete")
async def delete_experiment_file(exp_id: str, request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    file_id = body.get("file_id")
    exp_id = body.get("exp_id") or exp_id
    ef = db.query(ExperimentFileDB).filter(ExperimentFileDB.id == file_id, ExperimentFileDB.exp_id == exp_id).first()
    if ef:
        path = os.path.join(EXPERIMENT_FILES_DIR, exp_id, ef.filename)
        if os.path.exists(path):
            os.remove(path)
        db.delete(ef)
        db.commit()
    return JSONResponse({"ok": True})


def _ocr_rapid_from_data_url(image_b64: str) -> str:
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR

    m = re.match(r"^data:image/[^;]+;base64,(.*)$", image_b64, re.DOTALL)
    if not m:
        raise ValueError("Неверный формат изображения.")
    raw = base64.b64decode(m.group(1))
    with tempfile.TemporaryDirectory() as td:
        img_path = os.path.join(td, "img.png")
        with open(img_path, "wb") as f:
            f.write(raw)
        Image.open(img_path).verify()
        ocr = RapidOCR()
        result, _elapsed = ocr(img_path)
        if not result:
            return "Текст не найден."
        lines = []
        for item in result:
            try:
                text = (item[1] or "").strip()
                if text:
                    lines.append(text)
            except Exception:
                continue
        return "\n".join(lines).strip() or "Текст не найден."


@app.post("/api/ocr")
async def run_ocr(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    image_b64 = body.get("image_b64", "")

    if not image_b64:
        return JSONResponse({"text": "Не удалось получить данные изображения."})

    if not re.match(r"^data:image/[^;]+;base64,", image_b64):
        return JSONResponse({"text": "Неверный формат изображения."}, status_code=400)

    vision_err = ""
    if OCR_API_KEY:
        try:
            text_out = await ocr_via_vision_api(image_b64)
            return JSONResponse({"text": text_out or "Текст не найден.", "source": "vision"})
        except Exception as e:
            vision_err = str(e).strip() or format_ai_request_error(e)

    try:
        text_out = _ocr_rapid_from_data_url(image_b64)
        prefix = ""
        if vision_err:
            prefix = f"(OpenRouter недоступен — использован локальный OCR)\n\n"
        return JSONResponse({"text": prefix + text_out, "source": "local"})
    except ImportError:
        msg = "OCR недоступен: установите rapidocr-onnxruntime на сервере."
        if vision_err:
            msg = f"OpenRouter: {vision_err}. {msg}"
        return JSONResponse({"text": msg}, status_code=502)
    except Exception as e:
        msg = str(e).strip() or "Локальный OCR не смог распознать изображение."
        if vision_err:
            msg = f"OpenRouter: {vision_err}. Локальный OCR: {msg}"
        return JSONResponse({"text": msg}, status_code=502)

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
    return JSONResponse({"items": [{"id": p.id, "name": p.name, "status": p.status, "description": p.description or "", "end_date": p.end_date or ""} for p in items]})

@app.post("/api/projects")
async def create_project(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    username = (body.get("user") or "").strip()
    name = (body.get("name") or "").strip()
    if not username:
        return JSONResponse({"ok": False, "error": "Нужен пользователь"}, status_code=400)
    if not name:
        return JSONResponse({"ok": False, "error": "Введите название проекта"}, status_code=400)
    project_id = str(uuid.uuid4())[:8]
    p = ProjectDB(
        id=project_id,
        user_id=username,
        name=name,
        status=body.get("status") or "Активный",
        description=(body.get("description") or "").strip(),
        end_date=(body.get("end_date") or "").strip(),
    )
    db.add(p)
    db.flush()
    # Creator becomes first member
    member = ProjectMemberDB(project_id=project_id, username=username)
    db.add(member)
    db.commit()
    return JSONResponse({"ok": True, "id": project_id})

@app.put("/api/projects/{item_id}")
async def update_project(item_id: str, request: Request, db: Session = Depends(get_db)):
    p = db.query(ProjectDB).filter(ProjectDB.id == item_id).first()
    if not p:
        return JSONResponse({"ok": False}, status_code=404)
    body = await request.json()
    p.name = body.get("name", p.name)
    p.status = body.get("status", p.status)
    p.description = body.get("description", p.description)
    p.end_date = body.get("end_date", p.end_date)
    db.commit()
    return JSONResponse({"ok": True})

@app.delete("/api/projects/{item_id}")
async def delete_project(item_id: str, db: Session = Depends(get_db)):
    p = db.query(ProjectDB).filter(ProjectDB.id == item_id).first()
    if p:
        db.query(ProjectTaskDB).filter(ProjectTaskDB.project_id == item_id).delete()
        db.query(ProjectMemberDB).filter(ProjectMemberDB.project_id == item_id).delete()
        db.query(ProjectInviteDB).filter(ProjectInviteDB.project_id == item_id).delete()
        db.delete(p)
        db.commit()
    return JSONResponse({"ok": True})

@app.get("/api/projects/{item_id}")
async def get_project_detail(item_id: str, db: Session = Depends(get_db)):
    p = db.query(ProjectDB).filter(ProjectDB.id == item_id).first()
    if not p:
        return JSONResponse({"ok": False}, status_code=404)
    tasks = db.query(ProjectTaskDB).filter(ProjectTaskDB.project_id == item_id).all()
    members = db.query(ProjectMemberDB).filter(ProjectMemberDB.project_id == item_id).all()
    invites = db.query(ProjectInviteDB).filter(ProjectInviteDB.project_id == item_id).all()
    member_names = [m.username for m in members]
    return JSONResponse({
        "ok": True,
        "project": {"id": p.id, "name": p.name, "status": p.status, "description": p.description or "", "end_date": p.end_date or "", "user_id": p.user_id},
        "tasks": [{"id": t.id, "title": t.title, "description": t.description, "assignee": t.assignee, "status": t.status, "priority": t.priority, "start_date": t.start_date, "end_date": t.end_date, "created_by": t.created_by} for t in tasks],
        "members": member_names,
        "invites": [{"code": i.code, "created_by": i.created_by, "created_at": i.created_at} for i in invites],
    })

# ── API: Project Tasks ──

@app.post("/api/projects/{project_id}/tasks")
async def create_task(project_id: str, request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    t = ProjectTaskDB(
        project_id=project_id,
        title=body.get("title", ""),
        description=body.get("description", ""),
        assignee=body.get("assignee", ""),
        status=body.get("status", "Открыто"),
        priority=body.get("priority", "Средний"),
        start_date=body.get("start_date", ""),
        end_date=body.get("end_date", ""),
        created_by=body.get("user", ""),
    )
    db.add(t)
    db.commit()
    return JSONResponse({"ok": True, "id": t.id})

@app.put("/api/projects/{project_id}/tasks/{task_id}")
async def update_task(project_id: str, task_id: str, request: Request, db: Session = Depends(get_db)):
    t = db.query(ProjectTaskDB).filter(ProjectTaskDB.id == task_id, ProjectTaskDB.project_id == project_id).first()
    if not t:
        return JSONResponse({"ok": False}, status_code=404)
    body = await request.json()
    t.title = body.get("title", t.title)
    t.description = body.get("description", t.description)
    t.assignee = body.get("assignee", t.assignee)
    t.status = body.get("status", t.status)
    t.priority = body.get("priority", t.priority)
    t.start_date = body.get("start_date", t.start_date)
    t.end_date = body.get("end_date", t.end_date)
    db.commit()
    return JSONResponse({"ok": True})

@app.delete("/api/projects/{project_id}/tasks/{task_id}")
async def delete_task(project_id: str, task_id: str, db: Session = Depends(get_db)):
    t = db.query(ProjectTaskDB).filter(ProjectTaskDB.id == task_id, ProjectTaskDB.project_id == project_id).first()
    if t:
        db.delete(t)
        db.commit()
    return JSONResponse({"ok": True})

# ── API: Project Members ──

@app.post("/api/projects/{project_id}/members")
async def add_member(project_id: str, request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    username = body.get("username", "")
    existing = db.query(ProjectMemberDB).filter(ProjectMemberDB.project_id == project_id, ProjectMemberDB.username == username).first()
    if not existing:
        m = ProjectMemberDB(project_id=project_id, username=username)
        db.add(m)
        db.commit()
    return JSONResponse({"ok": True})

@app.delete("/api/projects/{project_id}/members/{username}")
async def remove_member(project_id: str, username: str, db: Session = Depends(get_db)):
    m = db.query(ProjectMemberDB).filter(ProjectMemberDB.project_id == project_id, ProjectMemberDB.username == username).first()
    if m:
        db.delete(m)
        db.commit()
    return JSONResponse({"ok": True})

# ── API: Project Invites ──

@app.post("/api/projects/{project_id}/invite")
async def create_invite(project_id: str, request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    inv = ProjectInviteDB(project_id=project_id, created_by=body.get("user", ""))
    db.add(inv)
    db.commit()
    return JSONResponse({"ok": True, "code": inv.code})

@app.get("/api/invite/{code}")
async def get_invite(code: str, db: Session = Depends(get_db)):
    inv = db.query(ProjectInviteDB).filter(ProjectInviteDB.code == code).first()
    if not inv:
        return JSONResponse({"ok": False, "error": "Invite not found"}, status_code=404)
    project = db.query(ProjectDB).filter(ProjectDB.id == inv.project_id).first()
    return JSONResponse({"ok": True, "project_id": inv.project_id, "project_name": project.name if project else "Unknown", "created_by": inv.created_by})

@app.get("/invite/{code}", response_class=HTMLResponse)
async def invite_page(request: Request, code: str, db: Session = Depends(get_db)):
    inv = db.query(ProjectInviteDB).filter(ProjectInviteDB.code == code).first()
    if not inv:
        return HTMLResponse("Ссылка-приглашение недействительна или истекла.", status_code=404)
    project = db.query(ProjectDB).filter(ProjectDB.id == inv.project_id).first()
    project_name = project.name if project else "Проект"
    return templates.TemplateResponse(request, "invite.html", {"code": code, "project_name": project_name, "created_by": inv.created_by})

@app.post("/api/invite/{code}/accept")
async def accept_invite(code: str, request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    username = body.get("username", "")
    inv = db.query(ProjectInviteDB).filter(ProjectInviteDB.code == code).first()
    if not inv:
        return JSONResponse({"ok": False, "error": "Invite not found"}, status_code=404)
    existing = db.query(ProjectMemberDB).filter(ProjectMemberDB.project_id == inv.project_id, ProjectMemberDB.username == username).first()
    if not existing:
        m = ProjectMemberDB(project_id=inv.project_id, username=username)
        db.add(m)
        db.commit()
    return JSONResponse({"ok": True, "project_id": inv.project_id})

@app.get("/api/users/search")
async def search_users(q: str, db: Session = Depends(get_db)):
    users = db.query(UserDB).filter(UserDB.username.ilike(f"%{q}%")).limit(10).all()
    return JSONResponse({"users": [{"username": u.username, "email": u.email or ""} for u in users]})

@app.get("/api/project/{project_id}/gantt")
@app.get("/api/projects/{project_id}/gantt")
async def get_gantt_data(project_id: str, db: Session = Depends(get_db)):
    """Return tasks with date info for Gantt chart rendering."""
    p = db.query(ProjectDB).filter(ProjectDB.id == project_id).first()
    if not p:
        return JSONResponse({"ok": False}, status_code=404)
    tasks = db.query(ProjectTaskDB).filter(ProjectTaskDB.project_id == project_id).all()
    gantt_tasks = []
    for t in tasks:
        if t.start_date and t.end_date:
            gantt_tasks.append({
                "id": t.id,
                "title": t.title,
                "assignee": t.assignee,
                "status": t.status,
                "start_date": t.start_date,
                "end_date": t.end_date,
                "priority": t.priority,
            })
    return JSONResponse({
        "ok": True,
        "project": {"name": p.name, "start_date": "", "end_date": p.end_date or ""},
        "tasks": gantt_tasks,
    })

@app.post("/experiment/{user}/{exp_id}/toggle_public")
async def toggle_public(user: str, exp_id: str, db: Session = Depends(get_db)):
    exp = db.query(ExperimentDB).filter(ExperimentDB.id == exp_id, ExperimentDB.user_id == user).first()
    if exp:
        exp.is_public = "true" if exp.is_public == "false" else "false"
        db.commit()
    return RedirectResponse(url=f"{BASE_PATH}/experiment/{user}/{exp_id}", status_code=302)

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

# ── Admin Panel ──

ADMIN_TZ_OFFSET_HOURS = int(os.environ.get("ADMIN_TZ_OFFSET_HOURS", "3") or 3)
_ADMIN_TZ = datetime.timezone(datetime.timedelta(hours=ADMIN_TZ_OFFSET_HOURS))
_RU_MONTHS_SHORT = ("янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")


def _admin_local_dt(value: str) -> datetime.datetime | None:
    """Audit timestamps are naive server-local (UTC on prod) → admin timezone."""
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(_ADMIN_TZ)


def _admin_fmt_dt(dt: datetime.datetime | None) -> str:
    if not dt:
        return "—"
    return f"{dt.day} {_RU_MONTHS_SHORT[dt.month - 1]} {dt:%H:%M}"


def _audit_log_level(action: str) -> str:
    a = (action or "").lower()
    if a in ("delete", "ban"):
        return "error"
    if a in ("reject", "failed"):
        return "warning"
    return "info"


def _audit_humanize(row) -> tuple[str, str, str]:
    """Returns (kind, title, subtitle) for admin UI."""
    action = (row.action or "").strip().lower()
    entity = (row.entity_type or "").strip().lower()
    raw = (row.details or "").strip()
    data = {}
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                data = parsed
        except Exception:
            data = {}

    if action == "login":
        via = "через Яндекс" if "yandex" in raw else "по паролю"
        return "login", "Вошёл в систему", via
    if action == "register":
        parts = []
        if "yandex" in raw:
            parts.append("через Яндекс")
        m = re.search(r"welcome_trial_(\d+)d", raw)
        if m:
            parts.append(f"пробный «Студент» на {m.group(1)} дн.")
        return "register", "Зарегистрировался", " · ".join(parts)
    if action == "create" and entity == "experiment":
        name = data.get("name") or data.get("title") or ""
        sub = f"«{name}»" if name else ""
        if data.get("source") == "planner":
            sub = (sub + " · через ИИ-планировщик").strip(" ·")
        return "experiment", "Создал эксперимент", sub
    if action == "update" and entity == "user":
        target = (row.entity_id or "").strip()
        changed = [k for k in data.keys() if k not in ("key", "username")]
        sub = f"{target}: {', '.join(changed)}" if changed else target
        return "admin", "Админ изменил пользователя", sub
    if action == "delete":
        return "delete", f"Удаление ({entity or 'объект'})", (row.entity_id or "")[:24]
    title = {"create": "Создание", "update": "Изменение"}.get(action, action or "Событие")
    if entity:
        title = f"{title}: {entity}"
    sub = data.get("name") or (raw if raw and len(raw) <= 80 and not raw.startswith("{") else "")
    return "other", title, sub


def _audit_row_dict(row) -> dict:
    created = (row.created_at or "").strip()
    local = _admin_local_dt(created)
    kind, title, subtitle = _audit_humanize(row)
    action = row.action or ""
    entity = row.entity_type or ""
    return {
        "id": row.id,
        "timestamp": _admin_fmt_dt(local),
        "created_at": created,
        "ts_iso": local.isoformat() if local else "",
        "day": local.strftime("%Y-%m-%d") if local else "",
        "user": row.user_id or "—",
        "action": action,
        "entity_type": entity,
        "entity_id": row.entity_id or "",
        "event": f"{action}:{entity}" if action and entity else action or entity or "event",
        "kind": kind,
        "title": title,
        "subtitle": subtitle,
        "detail": f"{title} — {subtitle}" if subtitle else title,
        "level": _audit_log_level(action),
        "ip": row.ip_address or "",
    }


def _legacy_admin_activity_logs(db: Session, limit: int) -> list[dict]:
    """Fallback when audit_logs is empty (no schema changes)."""
    logs: list[dict] = []
    for e in db.query(ExperimentDB).order_by(ExperimentDB.id.desc()).limit(limit).all():
        logs.append({
            "timestamp": "—", "created_at": "", "ts_iso": "", "day": "",
            "event": "create:experiment", "user": e.user_id, "kind": "experiment",
            "title": "Создал эксперимент", "subtitle": f"«{e.name}»",
            "detail": f"Создал эксперимент — «{e.name}»", "level": "info",
            "action": "create", "entity_type": "experiment", "entity_id": e.id, "ip": "",
        })
    return logs[:limit]


def _fetch_admin_activity(db: Session, limit: int = 100) -> list[dict]:
    rows = (
        db.query(AuditLogDB)
        .order_by(AuditLogDB.created_at.desc())
        .limit(limit)
        .all()
    )
    if rows:
        return [_audit_row_dict(r) for r in rows]
    return _legacy_admin_activity_logs(db, limit)


def _admin_last_activity_by_user(db: Session, scan_limit: int = 400) -> dict[str, dict]:
    out: dict[str, dict] = {}
    rows = (
        db.query(AuditLogDB)
        .order_by(AuditLogDB.created_at.desc())
        .limit(scan_limit)
        .all()
    )
    for row in rows:
        uid = (row.user_id or "").strip()
        if uid and uid not in out:
            out[uid] = _audit_row_dict(row)
    return out


def _admin_activity_insights(db: Session, users: list, days: int = 30) -> dict:
    """Read-only aggregates over audit_logs for admin charts."""
    from collections import Counter, defaultdict

    now_local = datetime.datetime.now(_ADMIN_TZ)
    today = now_local.date()
    day_keys = [(today - datetime.timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    first_day = day_keys[0]

    per_day = {d: Counter() for d in day_keys}
    active_per_day = defaultdict(set)
    events_per_user = Counter()
    user_days = defaultdict(set)
    kinds_total = Counter()
    registered_at: dict[str, str] = {}
    last_login_at: dict[str, str] = {}

    for row in db.query(AuditLogDB).all():
        uid = (row.user_id or "").strip()
        local = _admin_local_dt(row.created_at or "")
        kind, _t, _s = _audit_humanize(row)
        action = (row.action or "").lower()
        if uid and local:
            iso = local.isoformat()
            if action == "register" and (uid not in registered_at or iso < registered_at[uid]):
                registered_at[uid] = iso
            if action == "login" and iso > last_login_at.get(uid, ""):
                last_login_at[uid] = iso
        if not local:
            continue
        day = local.date().isoformat()
        if day < first_day or day not in per_day:
            continue
        per_day[day][kind] += 1
        kinds_total[kind] += 1
        if uid and uid != "admin":
            active_per_day[day].add(uid)
            events_per_user[uid] += 1
            user_days[uid].add(day)

    def _active_since(n: int) -> int:
        keys = set(day_keys[-n:])
        return len({u for d in keys for u in active_per_day.get(d, set())})

    new_7d = sum(
        1 for iso in registered_at.values()
        if iso[:10] >= (today - datetime.timedelta(days=6)).isoformat()
    )
    plans = Counter((getattr(u, "plan", None) or "free").strip() or "free" for u in users)
    usernames = {u.username for u in users}
    top = [
        {"username": u, "events": c}
        for u, c in events_per_user.most_common(8)
        if u in usernames
    ]
    spark_days = day_keys[-14:]
    sparks = {
        u: [1 if d in user_days.get(u, set()) else 0 for d in spark_days]
        for u in usernames
    }

    def _label(d: str) -> str:
        y, m, dd = d.split("-")
        return f"{int(dd)} {_RU_MONTHS_SHORT[int(m) - 1]}"

    return {
        "labels": [_label(d) for d in day_keys],
        "series": {
            "login": [per_day[d]["login"] for d in day_keys],
            "register": [per_day[d]["register"] for d in day_keys],
            "experiment": [per_day[d]["experiment"] for d in day_keys],
            "other": [per_day[d]["other"] + per_day[d]["admin"] + per_day[d]["delete"] for d in day_keys],
            "active_users": [len(active_per_day.get(d, set())) for d in day_keys],
        },
        "kinds": dict(kinds_total),
        "plans": {
            "labels": [PLAN_LABELS.get(k, k) for k in plans.keys()],
            "keys": list(plans.keys()),
            "values": list(plans.values()),
        },
        "top_users": top,
        "kpi": {
            "active_today": _active_since(1),
            "active_7d": _active_since(7),
            "active_30d": _active_since(days),
            "new_7d": new_7d,
            "events_30d": sum(kinds_total.values()),
        },
        "registered_at": registered_at,
        "last_login_at": last_login_at,
        "events_per_user": dict(events_per_user),
        "sparks": sparks,
    }


def admin_required(request: Request):
    """Check admin authorization."""
    auth = request.headers.get("X-Admin-Key", "")
    if auth == ADMIN_KEY:
        return True
    # Also check query param or form field
    return False

@app.get(LAB_ADMIN_PATH, response_class=HTMLResponse)
async def admin_login_page(request: Request, error: str = None):
    return templates.TemplateResponse(request, "admin_login.html", {"error": error})

@app.get("/admin", include_in_schema=False)
async def admin_legacy_entry_redirect():
    return RedirectResponse(url=f"{BASE_PATH}{LAB_ADMIN_PATH}", status_code=301)

@app.post("/admin")
@app.post(LAB_ADMIN_PATH)
async def admin_login(admin_key: str = Form(...)):
    if admin_key == ADMIN_KEY:
        return RedirectResponse(
            url=f"{BASE_PATH}{LAB_ADMIN_PATH}/dashboard?key={admin_key}", status_code=302
        )
    return RedirectResponse(
        url=f"{BASE_PATH}{LAB_ADMIN_PATH}?error=InvalidKey", status_code=302
    )

@app.get("/admin/dashboard", include_in_schema=False)
async def admin_legacy_dashboard_redirect(key: str = ""):
    q = f"?key={key}" if key else ""
    return RedirectResponse(url=f"{BASE_PATH}{LAB_ADMIN_PATH}/dashboard{q}", status_code=301)

@app.get(f"{LAB_ADMIN_PATH}/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, key: str, db: Session = Depends(get_db)):
    if key != ADMIN_KEY:
        return RedirectResponse(
            url=f"{BASE_PATH}{LAB_ADMIN_PATH}?error=InvalidKey", status_code=302
        )

    # Gather stats
    users = db.query(UserDB).all()
    experiments = db.query(ExperimentDB).all()
    projects = db.query(ProjectDB).all()
    reagents = db.query(ReagentDB).all()
    consumables = db.query(ConsumableDB).all()

    public_exps = [e for e in experiments if e.is_public == "true"]
    private_exps = [e for e in experiments if e.is_public != "true"]

    last_activity_map = _admin_last_activity_by_user(db)
    insights = _admin_activity_insights(db, users)
    user_list = []
    for u in users:
        user_exps = db.query(ExperimentDB).filter(ExperimentDB.user_id == u.username).all()
        user_exp_count = len(user_exps)
        user_project_count = db.query(ProjectDB).filter(ProjectDB.user_id == u.username).count()
        user_reagent_count = db.query(ReagentDB).filter(ReagentDB.user_id == u.username).count()
        user_consumable_count = db.query(ConsumableDB).filter(ConsumableDB.user_id == u.username).count()
        user_friend_count = db.query(FriendDB).filter(
            (FriendDB.user_id == u.username) | (FriendDB.friend_id == u.username),
            FriendDB.status == "accepted",
        ).count()
        user_equipment_count = db.query(EquipmentDB).filter(EquipmentDB.owner_id == u.username).count()
        total_nodes = sum(sum(len(level) for level in (e.levels or [])) for e in user_exps)
        plan = (u.plan or "free").strip() or "free"
        act = last_activity_map.get(u.username) or {}
        login_iso = insights["last_login_at"].get(u.username, "")
        if not login_iso and u.last_login:
            dt = _admin_local_dt(u.last_login)
            login_iso = dt.isoformat() if dt else ""
        reg_iso = insights["registered_at"].get(u.username, "")
        user_list.append({
            "username": u.username,
            "email": u.email or "",
            "display_name": u.display_name or "",
            "auth_provider": u.auth_provider or "password",
            "plan": plan,
            "plan_label": PLAN_LABELS.get(plan, "Бесплатный"),
            "plan_expires_at": _admin_fmt_dt(_admin_local_dt(u.plan_expires_at or "")).rsplit(" ", 1)[0] if u.plan_expires_at else "",
            "is_banned": u.is_banned or "false",
            "last_login": login_iso,
            "last_login_text": _admin_fmt_dt(_admin_local_dt(login_iso)) if login_iso else "",
            "registered_text": _admin_fmt_dt(_admin_local_dt(reg_iso)).rsplit(" ", 1)[0] if reg_iso else "",
            "exp_count": user_exp_count,
            "project_count": user_project_count,
            "reagent_count": user_reagent_count,
            "consumable_count": user_consumable_count,
            "friend_count": user_friend_count,
            "equipment_count": user_equipment_count,
            "total_nodes": total_nodes,
            "events_30d": insights["events_per_user"].get(u.username, 0),
            "spark": insights["sparks"].get(u.username, []),
            "last_activity_at": act.get("ts_iso") or "",
            "last_activity_text": act.get("timestamp") or "",
            "last_activity_title": act.get("title") or "",
            "last_activity_subtitle": act.get("subtitle") or "",
            "last_activity_kind": act.get("kind") or "",
        })
    user_list.sort(
        key=lambda row: row.get("last_activity_at") or row.get("last_login") or "",
        reverse=True,
    )
    recent_activity = _fetch_admin_activity(db, limit=80)

    # Build experiment list with node counts
    exp_list = []
    for e in experiments:
        node_count = sum(len(level) for level in (e.levels or []))
        exp_list.append({
            "id": e.id,
            "name": e.name,
            "user_id": e.user_id,
            "is_public": e.is_public or "false",
            "node_count": node_count,
        })

    tickets = db.query(SupportTicketDB).all()
    open_tickets = [t for t in tickets if t.status == "open"]

    # Daily registrations (last 7 days) — from audit "register", not last_login
    from collections import Counter
    reg_days = Counter()
    for iso in (insights.get("registered_at") or {}).values():
        day = (iso or "")[:10]
        if day:
            reg_days[day] += 1

    today_local = datetime.datetime.now(_ADMIN_TZ).date()
    daily_regs = []
    for i in range(6, -1, -1):
        day = (today_local - datetime.timedelta(days=i)).isoformat()
        daily_regs.append({"date": day, "count": int(reg_days.get(day, 0))})

    stats = {
        "total_users": len(users),
        "total_experiments": len(experiments),
        "public_experiments": len(public_exps),
        "private_experiments": len(private_exps),
        "total_projects": len(projects),
        "total_reagents": len(reagents),
        "total_consumables": len(consumables),
        "open_tickets": len(open_tickets),
        "avg_experiments_per_user": round(len(experiments) / max(len(users), 1), 1),
        "recent_experiments": sorted(experiments, key=lambda e: str(getattr(e, 'id', '')), reverse=True)[:5],
        "daily_registrations": daily_regs,
        "uptime": "—",
    }

    return templates.TemplateResponse(
        request, "admin.html",
        {
            "admin_key": key,
            "users": user_list,
            "all_experiments": exp_list,
            "stats": stats,
            "recent_activity": recent_activity,
            "insights": {k: v for k, v in insights.items() if k not in ("registered_at", "last_login_at", "events_per_user", "sparks")},
        }
    )

@app.post("/admin/delete_user")
@app.post(f"{LAB_ADMIN_PATH}/delete_user")
async def admin_delete_user(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    provided_key = body.get("admin_key", "")
    username = body.get("username", "")

    if provided_key != ADMIN_KEY:
        return JSONResponse({"ok": False, "error": "Invalid admin key"}, status_code=403)
    if not username:
        return JSONResponse({"ok": False, "error": "Username required"}, status_code=400)

    user = db.query(UserDB).filter(UserDB.username == username).first()
    if not user:
        return JSONResponse({"ok": False, "error": "User not found"}, status_code=404)

    # Delete all user data
    db.query(ExperimentDB).filter(ExperimentDB.user_id == username).delete()
    db.query(ProjectDB).filter(ProjectDB.user_id == username).delete()
    db.query(ReagentDB).filter(ReagentDB.user_id == username).delete()
    db.query(ConsumableDB).filter(ConsumableDB.user_id == username).delete()
    db.delete(user)
    db.commit()

    return JSONResponse({"ok": True})

@app.post("/admin/delete_experiment")
@app.post(f"{LAB_ADMIN_PATH}/delete_experiment")
async def admin_delete_experiment(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    provided_key = body.get("admin_key", "")
    exp_id = body.get("exp_id", "")

    if provided_key != ADMIN_KEY:
        return JSONResponse({"ok": False, "error": "Invalid admin key"}, status_code=403)
    if not exp_id:
        return JSONResponse({"ok": False, "error": "Experiment ID required"}, status_code=400)

    # Delete experiment and related data
    db.query(NodeNotesDB).filter(NodeNotesDB.exp_id == exp_id).delete()
    db.query(NodeVersionDB).filter(NodeVersionDB.exp_id == exp_id).delete()
    exp = db.query(ExperimentDB).filter(ExperimentDB.id == exp_id).first()
    if exp:
        db.delete(exp)
        db.commit()

    return JSONResponse({"ok": True})

# ── Admin API: User Dashboard ──

@app.get("/admin/api/user_dashboard")
@app.get(f"{LAB_ADMIN_PATH}/api/user_dashboard")
async def admin_user_dashboard(key: str, user: str = None, db: Session = Depends(get_db)):
    if key != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    
    if user:
        users_q = db.query(UserDB).filter(UserDB.username == user).all()
    else:
        users_q = db.query(UserDB).all()
    
    result = []
    for u in users_q:
        exps = db.query(ExperimentDB).filter(ExperimentDB.user_id == u.username).all()
        projects = db.query(ProjectDB).filter(ProjectDB.user_id == u.username).all()
        reagents = db.query(ReagentDB).filter(ReagentDB.user_id == u.username).all()
        consumables = db.query(ConsumableDB).filter(ConsumableDB.user_id == u.username).all()
        friends = db.query(FriendDB).filter(
            (FriendDB.user_id == u.username) | (FriendDB.friend_id == u.username),
            FriendDB.status == "accepted",
        ).count()
        equip = db.query(EquipmentDB).filter(EquipmentDB.owner_id == u.username).count()
        
        total_nodes = sum(sum(len(level) for level in (e.levels or [])) for e in exps)
        
        result.append({
            "username": u.username,
            "email": u.email or "",
            "experiments": len(exps),
            "projects": len(projects),
            "reagents": len(reagents),
            "consumables": len(consumables),
            "friends": friends,
            "equipment": equip,
            "total_nodes": total_nodes,
            "has_experiments": len(exps) > 0,
        })
    
    return JSONResponse({"users": result})

# ── Admin API: Corporate ──

@app.get("/admin/api/corporate")
@app.get(f"{LAB_ADMIN_PATH}/api/corporate")
async def admin_corporate(key: str, db: Session = Depends(get_db)):
    if key != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    
    all_users = db.query(UserDB).all()
    projects = db.query(ProjectDB).all()
    project_members = db.query(ProjectMemberDB).all()
    
    return JSONResponse({
        "total_users": len(all_users),
        "total_projects": len(projects),
        "project_members": len(project_members),
        "total_experiments": db.query(ExperimentDB).count(),
        "users_with_projects": len(set(p.user_id for p in projects)),
        "avg_members_per_project": round(len(project_members) / max(len(projects), 1), 1),
        "corp_users_list": [
            {"username": u.username, "email": u.email or "", "projects": db.query(ProjectDB).filter(ProjectDB.user_id == u.username).count()}
            for u in all_users[:20]
        ],
    })

# ── Admin API: Referrals ──

@app.get("/admin/api/referrals")
@app.get(f"{LAB_ADMIN_PATH}/api/referrals")
async def admin_referrals(key: str, db: Session = Depends(get_db)):
    if key != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    
    friends_count = db.query(FriendDB).filter(FriendDB.status == "accepted").count()
    pending_count = db.query(FriendDB).filter(FriendDB.status == "pending").count()
    total_users = db.query(UserDB).count()
    invites = db.query(ProjectInviteDB).count()
    
    from collections import Counter
    all_friends = db.query(FriendDB).filter(FriendDB.status == "accepted").all()
    user_friend_count = Counter()
    for f in all_friends:
        user_friend_count[f.user_id] += 1
        user_friend_count[f.friend_id] += 1
    
    top_users = [{"username": u, "count": c} for u, c in user_friend_count.most_common(10)]
    
    return JSONResponse({
        "total_friendships": friends_count,
        "pending_requests": pending_count,
        "total_users": total_users,
        "total_invites": invites,
        "referral_rate": round(friends_count / max(total_users, 1) * 100, 1),
        "top_users": top_users,
    })

# ── Admin API: Support ──

@app.get("/admin/api/support")
@app.get(f"{LAB_ADMIN_PATH}/api/support")
async def admin_support(key: str, db: Session = Depends(get_db)):
    if key != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    
    total_users = db.query(UserDB).count()
    experiments = db.query(ExperimentDB).count()
    
    return JSONResponse({
        "open_tickets": 0,
        "resolved_tickets": 0,
        "total_queries": 0,
        "avg_response_time": "—",
        "satisfaction_rate": "—",
        "total_users": total_users,
        "total_experiments": experiments,
        "needs_setup": True,
        "message": "Раздел требует создания системы тикетов (таблица support_tickets)",
    })

# ── Admin API: Logs ──

@app.get("/admin/api/logs")
@app.get(f"{LAB_ADMIN_PATH}/api/logs")
async def admin_logs(key: str, limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    if key != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    cap = max(1, min(int(limit or 100), 500))
    off = max(0, int(offset or 0))
    all_logs = _fetch_admin_activity(db, limit=cap + off + 200)
    page = all_logs[off : off + cap]

    log_levels = {"info": 0, "warning": 0, "error": 0}
    for entry in all_logs:
        lvl = entry.get("level") or "info"
        log_levels[lvl] = log_levels.get(lvl, 0) + 1

    return JSONResponse(
        {
            "logs": page,
            "total": len(all_logs),
            "levels": log_levels,
        }
    )

# ── Admin API: Analytics ──

@app.get("/admin/api/analytics")
@app.get(f"{LAB_ADMIN_PATH}/api/analytics")
async def admin_analytics(key: str, db: Session = Depends(get_db)):
    if key != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    
    users = db.query(UserDB).all()
    experiments = db.query(ExperimentDB).all()
    projects = db.query(ProjectDB).all()
    reagents = db.query(ReagentDB).all()
    consumables = db.query(ConsumableDB).all()
    friends = db.query(FriendDB).all()
    equipment = db.query(EquipmentDB).all()
    bookings = db.query(EquipmentBookingDB).all()
    
    total_nodes = sum(sum(len(level) for level in (e.levels or [])) for e in experiments)
    
    top_users = sorted(
        [{"username": u.username, "experiments": db.query(ExperimentDB).filter(ExperimentDB.user_id == u.username).count(),
          "nodes": sum(sum(len(level) for level in (e.levels or [])) for e in db.query(ExperimentDB).filter(ExperimentDB.user_id == u.username).all())}
         for u in users],
        key=lambda x: x["experiments"], reverse=True
    )[:20]
    visit_stats = _service_visit_analytics(db)

    return JSONResponse({
        "total_users": len(users),
        "total_experiments": len(experiments),
        "total_projects": len(projects),
        "total_reagents": len(reagents),
        "total_consumables": len(consumables),
        "total_nodes": total_nodes,
        "total_friendships": len(friends),
        "total_equipment": len(equipment),
        "total_bookings": len(bookings),
        "public_experiments": sum(1 for e in experiments if e.is_public == "true"),
        "private_experiments": sum(1 for e in experiments if e.is_public != "true"),
        "avg_experiments_per_user": round(len(experiments) / max(len(users), 1), 2),
        "avg_nodes_per_experiment": round(total_nodes / max(len(experiments), 1), 1),
        "user_breakdown": top_users,
        **visit_stats,
    })

# ── Admin API: Support Tickets ──

@app.get("/admin/api/tickets")
@app.get(f"{LAB_ADMIN_PATH}/api/tickets")
async def admin_get_tickets(key: str, status: str = None, priority: str = None, db: Session = Depends(get_db)):
    if key != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    q = db.query(SupportTicketDB)
    if status and status != "all":
        q = q.filter(SupportTicketDB.status == status)
    if priority and priority != "all":
        q = q.filter(SupportTicketDB.priority == priority)
    tickets = q.order_by(SupportTicketDB.created_at.desc()).all()
    
    result = []
    for t in tickets:
        replies = db.query(SupportReplyDB).filter(SupportReplyDB.ticket_id == t.id).order_by(SupportReplyDB.created_at.asc()).all()
        result.append({
            "id": t.id,
            "user_id": t.user_id,
            "subject": t.subject,
            "message": t.message,
            "status": t.status,
            "priority": t.priority,
            "created_at": t.created_at,
            "updated_at": t.updated_at,
            "reply_count": len(replies),
            "replies": [{"id": r.id, "author": r.author, "message": r.message, "created_at": r.created_at} for r in replies],
        })
    
    counts = {"open": 0, "in_progress": 0, "resolved": 0, "closed": 0}
    for t in tickets:
        counts[t.status] = counts.get(t.status, 0) + 1
    
    return JSONResponse({"tickets": result, "counts": counts})

@app.post("/admin/api/tickets/reply")
@app.post(f"{LAB_ADMIN_PATH}/api/tickets/reply")
async def admin_reply_ticket(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    if body.get("key") != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    reply = SupportReplyDB(ticket_id=body.get("ticket_id"), author="admin", message=body.get("message", ""))
    db.add(reply)
    ticket = db.query(SupportTicketDB).filter(SupportTicketDB.id == body.get("ticket_id")).first()
    if ticket and ticket.status == "open":
        ticket.status = "in_progress"
        ticket.updated_at = datetime.datetime.now().isoformat()
    db.commit()
    return JSONResponse({"ok": True})

@app.post("/admin/api/tickets/status")
@app.post(f"{LAB_ADMIN_PATH}/api/tickets/status")
async def admin_update_ticket_status(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    if body.get("key") != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    ticket = db.query(SupportTicketDB).filter(SupportTicketDB.id == body.get("ticket_id")).first()
    if not ticket:
        return JSONResponse({"error": "Ticket not found"}, status_code=404)
    ticket.status = body.get("status", ticket.status)
    ticket.updated_at = datetime.datetime.now().isoformat()
    db.commit()
    return JSONResponse({"ok": True})

# ── Admin API: User Management ──

@app.post("/admin/api/user/edit")
@app.post(f"{LAB_ADMIN_PATH}/api/user/edit")
async def admin_edit_user(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    if body.get("key") != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    user = db.query(UserDB).filter(UserDB.username == body.get("username")).first()
    if not user:
        return JSONResponse({"error": "User not found"}, status_code=404)
    if "email" in body:
        user.email = body["email"]
    if "is_banned" in body:
        user.is_banned = body["is_banned"]
    db.commit()
    audit_log(db, "admin", "update", "user", body.get("username", ""), json.dumps(body))
    return JSONResponse({"ok": True})

@app.post("/admin/api/user/bulk")
@app.post(f"{LAB_ADMIN_PATH}/api/user/bulk")
async def admin_bulk_user_action(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    if body.get("key") != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    action = body.get("action", "")
    usernames = body.get("usernames", [])
    if action == "delete":
        for u in usernames:
            db.query(ExperimentDB).filter(ExperimentDB.user_id == u).delete()
            db.query(ProjectDB).filter(ProjectDB.user_id == u).delete()
            db.query(ReagentDB).filter(ReagentDB.user_id == u).delete()
            db.query(ConsumableDB).filter(ConsumableDB.user_id == u).delete()
            db.query(FriendDB).filter((FriendDB.user_id == u) | (FriendDB.friend_id == u)).delete()
            db.query(UserDB).filter(UserDB.username == u).delete()
        db.commit()
    elif action == "ban":
        for u in usernames:
            usr = db.query(UserDB).filter(UserDB.username == u).first()
            if usr: usr.is_banned = "true"
        db.commit()
    elif action == "unban":
        for u in usernames:
            usr = db.query(UserDB).filter(UserDB.username == u).first()
            if usr: usr.is_banned = "false"
        db.commit()
    return JSONResponse({"ok": True})

# ── Admin API: Organizations ──

@app.get("/admin/api/organizations")
@app.get(f"{LAB_ADMIN_PATH}/api/organizations")
async def admin_get_organizations(key: str, db: Session = Depends(get_db)):
    if key != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    orgs = db.query(OrganizationDB).all()
    result = []
    for o in orgs:
        members = db.query(OrganizationMemberDB).filter(OrganizationMemberDB.org_id == o.id).all()
        result.append({
            "id": o.id,
            "name": o.name,
            "owner_id": o.owner_id,
            "created_at": o.created_at,
            "member_count": len(members),
            "members": [{"user_id": m.user_id, "role": m.role} for m in members],
        })
    return JSONResponse({"organizations": result})

@app.post("/admin/api/organizations/create")
@app.post(f"{LAB_ADMIN_PATH}/api/organizations/create")
async def admin_create_organization(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    if body.get("key") != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    org = OrganizationDB(name=body.get("name"), owner_id=body.get("owner_id"))
    db.add(org)
    db.commit()
    # Add owner as admin member
    member = OrganizationMemberDB(org_id=org.id, user_id=body.get("owner_id"), role="admin")
    db.add(member)
    db.commit()
    return JSONResponse({"ok": True, "id": org.id})

# ── Admin API: Referrals Management ──

@app.get("/admin/api/referral_codes")
@app.get(f"{LAB_ADMIN_PATH}/api/referral_codes")
async def admin_get_referral_codes(key: str, db: Session = Depends(get_db)):
    if key != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    links = db.query(ReferralLinkDB).all()
    result = []
    for l in links:
        user = db.query(UserDB).filter(UserDB.username == l.user_id).first()
        result.append({
            "code": l.code,
            "user_id": l.user_id,
            "email": user.email if user else "",
            "clicks": int(l.clicks or 0),
            "registrations": int(l.registrations or 0),
            "created_at": l.created_at,
        })
    return JSONResponse({"codes": result})

# ── Admin API: CSV Export ──

@app.get("/admin/api/export/users")
@app.get(f"{LAB_ADMIN_PATH}/api/export/users")
async def admin_export_users_csv(key: str, db: Session = Depends(get_db)):
    if key != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    users = db.query(UserDB).all()
    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Username", "Email", "Last Login", "Is Banned"])
    for u in users:
        writer.writerow([u.username, u.email, u.last_login, u.is_banned])
    return JSONResponse({"csv": output.getvalue()})

# ── Admin API: Login as user ──

@app.get("/admin/api/login_as")
@app.get(f"{LAB_ADMIN_PATH}/api/login_as")
async def admin_login_as(key: str, username: str):
    if key != ADMIN_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    return RedirectResponse(url=f"{BASE_PATH}/dashboard?user={username}")

# ── AI Planner: чат на дашборде (план эксперимента → создание) ──

_PLANNER_STATE: dict = {}

def _planner_wants_new_plan(msg: str) -> bool:
    low = msg.lower()
    return any(k in low for k in ("новый план", "другой эксперимент", "другое исследование", "перепланируй", "новый эксперимент"))

def _extract_plan_json(text: str):
    """Вытащить {"title":..., "steps":[...]} из ответа модели."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if "```" in t[3:] else t
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(t[start:end + 1])
    except Exception:
        return None
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    steps = [str(s).strip() for s in steps if str(s).strip()][:12]
    if not steps:
        return None
    return {"title": str(data.get("title") or "Эксперимент").strip()[:120] or "Эксперимент", "steps": steps}

def _heuristic_plan(message: str) -> dict:
    """Fallback plan when LLM is slow/unavailable."""
    raw = (message or "").strip()
    title = raw[:120] or "Новый эксперимент"
    low = raw.lower()
    if "пцр" in low or "pcr" in low:
        title = title if len(title) > 8 else "ПЦР: оптимизация температуры отжига"
        steps = [
            "Подготовить master mix, праймеры, матрицу ДНК и контроли",
            "Составить градиент температур отжига (например 55–65 °C)",
            "Запрограммировать цикл ПЦР и провести амплификацию",
            "Оценить специфичность продукта на агарозном геле",
            "Выбрать оптимальную температуру отжига и зафиксировать протокол",
        ]
    elif "e. coli" in low or "кинетик" in low or "рост" in low:
        title = title if len(title) > 8 else "Кинетика роста E. coli"
        steps = [
            "Подготовить среды (LB/M9) и стартовую культуру",
            "Засеять пробы и задать температуру инкубации",
            "Снимать ОD600 по расписанию (например каждые 10–15 мин)",
            "Построить кривую роста и рассчитать μmax",
            "Сформулировать выводы и сохранить сырые данные",
        ]
    elif "мик" in low or "антибиот" in low:
        title = title if len(title) > 8 else "Определение МИК антибиотика"
        steps = [
            "Подготовить культуру тест-штамма и серийные разведения антибиотика",
            "Засеять микропланшет или диски по протоколу",
            "Инкубировать при заданной температуре и времени",
            "Оценить минимальную inhibitory concentration (МИК)",
            "Зафиксировать результат и условия эксперимента",
        ]
    else:
        steps = [
            "Сформулировать цель, гипотезу и критерии успеха",
            "Подготовить реактивы, расходники и оборудование",
            f"Выполнить основной этап: {title[:70]}",
            "Зафиксировать промежуточные измерения и наблюдения",
            "Обработать данные и записать выводы в журнал",
        ]
    return {"title": title, "steps": steps}


def _planner_plan_response(state: dict, plan: dict, *, note: str = "") -> JSONResponse:
    state["plan"] = plan
    steps = plan["steps"]
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    prefix = (note + "\n\n") if note else ""
    text = (
        f"{prefix}План эксперимента «{plan['title']}»:\n\n{numbered}\n\n"
        "Нажмите «Создать эксперимент», чтобы записать этот план в журнал — "
        "появятся шаги, к которым можно добавлять данные и заметки."
    )
    return JSONResponse({
        "ok": True,
        "type": "plan",
        "title": plan["title"],
        "steps": steps,
        "text": text,
    })


async def _llm_planner_chat(messages) -> tuple[str | None, str | None]:
    """OpenAI-compatible chat. Returns (text, error_message)."""
    if not (AI_API_KEY and AI_API_URL and AI_CHAT_MODEL):
        err = "AI не настроен на сервере (AI_API_KEY / AI_API_URL / AI_CHAT_MODEL)."
        tg_notify_ai_issue("planner", err)
        return None, err
    try:
        async with ai_http_client(timeout=35.0) as client:
            resp = await client.post(
                AI_API_URL,
                headers={"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"},
                json={"model": AI_CHAT_MODEL, "messages": messages, "temperature": 0.4},
            )
            try:
                data = resp.json()
            except Exception:
                data = {"error": (resp.text or "")[:300]}
            if resp.status_code >= 400:
                err = ""
                if isinstance(data, dict):
                    err = data.get("error", {}).get("message") if isinstance(data.get("error"), dict) else data.get("error")
                    err = err or str(data.get("message", ""))
                msg = f"Ошибка AI API ({resp.status_code}): {err or 'неизвестная ошибка'}"
                tg_notify_ai_issue("planner", msg)
                return None, msg
            msg = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            msg = msg.strip()
            if not msg:
                empty = "AI вернул пустой ответ."
                tg_notify_ai_issue("planner", empty)
                return None, empty
            return msg, None
    except Exception as e:
        msg = format_ai_request_error(e)
        tg_notify_ai_issue("planner", msg)
        return None, msg

@app.post("/api/planner/chat")
async def planner_chat(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    session_user = get_session_username(request)
    username = (body.get("user") or "").strip() or (session_user or "")
    message = (body.get("message") or "").strip()
    if not username:
        return JSONResponse({"ok": False, "error": "Войдите в аккаунт"}, status_code=401)
    if session_user and username != session_user:
        return JSONResponse({"ok": False, "error": "Неверный пользователь сессии"}, status_code=403)
    if body.get("reset"):
        _PLANNER_STATE[username] = {"history": [], "plan": None}
        return JSONResponse({"ok": True, "type": "chat",
                             "text": "Новый диалог готов. Опишите эксперимент — я предложу план шагов и помогу создать его в журнале."})

    if not message:
        return JSONResponse({"ok": False, "error": "Пустое сообщение"})

    state = _PLANNER_STATE.setdefault(username, {"history": [], "plan": None})
    history = state["history"]
    history.append({"role": "user", "content": message})

    want_plan = state["plan"] is None or _planner_wants_new_plan(message)
    if want_plan:
        state["plan"] = None

    if want_plan:
        system = (
            "Ты — ИИ-планировщик лабораторного журнала Bio-Notebook. "
            "Пользователь описывает эксперимент — составь понятное название и пошаговый протокол (5–9 коротких шагов). "
            "Ответь ТОЛЬКО валидным JSON без пояснений и разметки в формате: "
            "{\"title\": \"Название эксперимента\", \"steps\": [\"Шаг 1\", \"Шаг 2\", ...]}"
        )
    else:
        plan = state["plan"]
        system = (
            "Ты — научный ассистент в лабораторном журнале Bio-Notebook. "
            f"У пользователя эксперимент «{plan['title']}» с планом: {'; '.join(plan['steps'])}. "
            "Отвечай кратко, по делу, в контексте этого эксперимента. Помогай уточнить протокол, "
            "методику, расчёты или интерпретацию. Если просят новый план — предложи спланировать новый эксперимент."
        )

    msgs = [{"role": "system", "content": system}] + history[-14:]
    ai_text, ai_err = await _llm_planner_chat(msgs)

    if want_plan:
        plan = _extract_plan_json(ai_text) if ai_text else None
        if plan:
            history.append({"role": "assistant", "content": ai_text or json.dumps(plan, ensure_ascii=False)})
            return _planner_plan_response(state, plan, note="Отличная задача!")
        fallback = _heuristic_plan(message)
        note = "Составил базовый план по вашему описанию."
        if ai_err:
            note = f"ИИ сейчас недоступен ({ai_err}). {note}"
        history.append({"role": "assistant", "content": json.dumps(fallback, ensure_ascii=False)})
        return _planner_plan_response(state, fallback, note=note)

    if ai_text:
        history.append({"role": "assistant", "content": ai_text})
        return JSONResponse({"ok": True, "type": "chat", "text": ai_text})

    text = ai_err or "ИИ-ассистент временно недоступен. Попробуйте ещё раз через минуту."
    history.append({"role": "assistant", "content": text})
    return JSONResponse({"ok": True, "type": "chat", "text": text})

@app.post("/api/planner/create")
async def planner_create(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    username = (body.get("user") or "").strip() or (get_session_username(request) or "")
    title = (body.get("title") or "Эксперимент").strip()[:120]
    steps = body.get("steps") or []
    steps = [str(s).strip() for s in steps if str(s).strip()][:12]
    if not username:
        return JSONResponse({"ok": False, "error": "auth"}, status_code=401)
    if not title or not steps:
        return JSONResponse({"ok": False, "error": "Нет названия или шагов плана"})

    exp_id = str(uuid.uuid4())[:8]
    levels = [[{"id": "root", "title": "Начало исследования", "content": "Инициализация среды."}]]
    for i, step in enumerate(steps, start=1):
        levels.append([{"id": str(uuid.uuid4())[:8], "title": f"Шаг {i}. {step[:90]}", "content": step}])

    # Папка-ноутбук — по возможности (на сервере с NOTEBOOKS_PATH), иначе пропускаем
    try:
        user_dir = os.path.join(NOTEBOOKS_BASE, username)
        exp_dir = os.path.join(user_dir, f"{title}_{exp_id}")
        os.makedirs(exp_dir, exist_ok=True)
        notebook_content = {
            "cells": [{"cell_type": "markdown", "id": "intro", "metadata": {},
                       "source": [f"# Эксперимент: {title}\n", "## План (создан ИИ-ассистентом)\n"] +
                                 [f"- {s}\n" for s in steps]}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5,
        }
        with open(os.path.join(exp_dir, "experiment.ipynb"), "w", encoding="utf-8") as f:
            json.dump(notebook_content, f, ensure_ascii=False)
    except Exception:
        pass

    new_exp = ExperimentDB(id=exp_id, user_id=username, name=title,
                           description="План эксперимента создан ИИ-ассистентом: " + "; ".join(steps),
                           levels=levels)
    db.add(new_exp)
    db.commit()
    audit_log(db, username, "create", "experiment", exp_id, json.dumps({"name": title, "source": "planner"}))
    _PLANNER_STATE.pop(username, None)
    return JSONResponse({"ok": True, "exp_id": exp_id})
