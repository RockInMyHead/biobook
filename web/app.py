import os
import uuid
import json
import time
import hmac
import secrets
import threading
import io
from urllib.parse import quote, urlencode, urlparse, parse_qs, unquote_plus
import base64
import re
import subprocess
import tempfile
import mimetypes
import shutil
import sysconfig
import logging

import hashlib
import binascii
import httpx
import nh3
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
_TG_LOG = logging.getLogger("uvicorn.error.telegram")
_TG_JOURNAL_SIZE = 500


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
        "X-Title": "Antikus Lab-Journal OCR",
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
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")
LAB_ADMIN_PATH = (os.environ.get("LAB_ADMIN_PATH", "/lab-admin") or "/lab-admin").rstrip("/")
if not LAB_ADMIN_PATH.startswith("/"):
    LAB_ADMIN_PATH = "/" + LAB_ADMIN_PATH
SITE_ORIGIN = os.environ.get("SITE_ORIGIN", "https://labjournal.tech").rstrip("/")
YANDEX_CLIENT_ID = os.environ.get("YANDEX_CLIENT_ID", "")
YANDEX_CLIENT_SECRET = os.environ.get("YANDEX_CLIENT_SECRET", "")
SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_urlsafe(48)
OAUTH_STATE_SECRET = os.environ.get("OAUTH_STATE_SECRET") or SESSION_SECRET
LEGAL_BASE_URL = os.environ.get("LEGAL_BASE_URL", SITE_ORIGIN).rstrip("/")
# Legacy hosts that should 301 to SITE_ORIGIN (same app / same VM).
LEGACY_HOSTS = {
    h.strip().lower()
    for h in os.environ.get(
        "LEGACY_HOSTS",
        "bionotebook.ru,www.bionotebook.ru,www.labjournal.tech",
    ).split(",")
    if h.strip()
}
SESSION_COOKIE = "biobook_sid"
ADMIN_COOKIE = "biobook_admin"
ADMIN_SESSION_SECONDS = 8 * 60 * 60
SESSION_DAYS = 30
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() != "false"
VISITOR_COOKIE = "bio_vid"
VISITOR_COOKIE_MAX_AGE = 365 * 24 * 3600


def get_session_username(request: Request) -> str | None:
    return getattr(request.state, "auth_user", None)


def _session_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def attach_session_cookie(response: RedirectResponse, username: str) -> RedirectResponse:
    token = secrets.token_urlsafe(32)
    now = int(time.time())
    with SessionLocal() as db:
        db.query(LoginSessionDB).filter(LoginSessionDB.expires_at <= now).delete()
        db.add(LoginSessionDB(token_hash=_session_digest(token), user_id=username, expires_at=now + SESSION_DAYS * 86400))
        db.commit()
    response.set_cookie(
        key=SESSION_COOKIE, value=token, max_age=SESSION_DAYS * 86400,
        httponly=True, secure=COOKIE_SECURE, samesite="lax", path="/",
    )
    return response


def clear_session_cookie(response: RedirectResponse) -> RedirectResponse:
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


def redirect_to_dashboard(username: str) -> RedirectResponse:
    resp = RedirectResponse(url=f"{BASE_PATH}/dashboard?user={username}", status_code=302)
    return attach_session_cookie(resp, username)


def path_is_public(path: str) -> bool:
    if path in (
        "/",
        "/index.html",
        "/favicon.ico",
        "/yandex_bd93356772fa18fd.html",
        "/llms.txt",
    ):
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
    if path in ("/login", "/register", "/logout", "/login/", "/register/", "/logout/"):
        return True
    if path == "/guides" or path.startswith("/guides/"):
        return True
    for slug in MARKETING_PAGES:
        if path in (f"/{slug}", f"/{slug}/"):
            return True
    for slug in LEGAL_PAGES:
        if path in (f"/{slug}", f"/{slug}/", f"/legal/{slug}", f"/legal/{slug}/"):
            return True
    if path in ("/rekvizity", "/rekvizity/"):
        return True
    if path == "/legal" or path.startswith("/legal/"):
        return True
    if path in ("/sitemap-guides.xml", "/sitemap.xml", "/robots.txt", "/llms.txt"):
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


AVATAR_MAX_BYTES = 5 * 1024 * 1024
AVATAR_ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "gif"}


def avatar_src(url: str | None) -> str:
    """Public URL for stored avatar (local upload or absolute https)."""
    raw = (url or "").strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://") or raw.startswith("data:"):
        return raw
    if raw.startswith("/static/uploads/"):
        return f"{BASE_PATH}{raw}" if BASE_PATH else raw
    if BASE_PATH and raw.startswith(f"{BASE_PATH}/static/uploads/"):
        return raw
    if "/" not in raw and "." in raw:
        return upload_public_url(raw)
    return raw


def _avatar_local_path(stored: str) -> str | None:
    """Return filesystem path if avatar is a local upload we own."""
    raw = (stored or "").strip()
    if not raw or raw.startswith("http://") or raw.startswith("https://"):
        return None
    if BASE_PATH and raw.startswith(BASE_PATH + "/"):
        raw = raw[len(BASE_PATH):]
    if raw.startswith("/static/uploads/"):
        fname = raw.rsplit("/", 1)[-1]
    elif "/" not in raw and "\\" not in raw:
        fname = raw
    else:
        return None
    if not fname.startswith("avatar_") or ".." in fname:
        return None
    path = os.path.abspath(os.path.join(UPLOADS_DIR, fname))
    root = os.path.abspath(UPLOADS_DIR)
    if not path.startswith(root + os.sep):
        return None
    return path


def _delete_local_avatar(stored: str) -> None:
    path = _avatar_local_path(stored)
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def yandex_avatar_url(default_avatar_id: str | None) -> str:
    aid = (default_avatar_id or "").strip()
    if not aid or aid in ("0/0-0", "0"):
        return ""
    return f"https://avatars.yandex.net/get-yapic/{aid}/islands-200"


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
    avatar_url = Column(String, default="")  # /static/uploads/... or https:// (e.g. Yandex)
    auth_provider = Column(String, default="password")
    agree_marketing = Column(String, default="false")
    plan = Column(String, default="free")  # free | student | corp
    plan_expires_at = Column(String, default="")  # ISO; paid/trial end → downgrade to free

class LoginSessionDB(Base):
    __tablename__ = "login_sessions"
    token_hash = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.username"), nullable=False, index=True)
    expires_at = Column(Integer, nullable=False, index=True)


class MediaUploadDB(Base):
    __tablename__ = "media_uploads"
    filename = Column(String, primary_key=True)
    owner_id = Column(String, index=True, nullable=False)
    exp_id = Column(String, index=True)
    chat_id = Column(String, index=True)
    size = Column(Integer, nullable=False)
    media_type = Column(String, nullable=False)
    original_name = Column(String, nullable=False)


class DataCleanupDB(Base):
    __tablename__ = "data_cleanup"
    path = Column(String, primary_key=True)
    is_directory = Column(Integer, nullable=False, default=0)


class DataMigrationDB(Base):
    __tablename__ = "data_migrations"
    name = Column(String, primary_key=True)


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
    revision = Column(Integer, nullable=False, default=0, server_default="0")

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
    author = Column(String, default="")

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

class TelegramJournalDB(Base):
    __tablename__ = "telegram_journal"
    id = Column(Integer, primary_key=True)
    created_at = Column(String, nullable=False)
    level = Column(String, nullable=False)
    event = Column(String, nullable=False)
    details = Column(JSON, nullable=False)

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
    user_agent = Column(String(512), default="")
    is_bot = Column(String(8), default="0", index=True)  # "1" | "0"
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


class ExportTemplateDB(Base):
    """Пользовательский шаблон экспорта (DOCX/PDF/PPTX) — визуальный бланк."""
    __tablename__ = "export_templates"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:10])
    user_id = Column(String, ForeignKey("users.username"), index=True)
    name = Column(String, default="Шаблон кафедры")
    kind = Column(String, default="docx")  # docx | pdf | pptx
    html = Column(Text, default="")
    created_at = Column(String, default=lambda: datetime.datetime.now().isoformat())
    updated_at = Column(String, default=lambda: datetime.datetime.now().isoformat())


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
    ("users", "avatar_url", "VARCHAR DEFAULT ''"),
    ("users", "auth_provider", "VARCHAR DEFAULT 'password'"),
    ("users", "agree_marketing", "VARCHAR DEFAULT 'false'"),
    ("users", "plan", "VARCHAR DEFAULT 'free'"),
    ("users", "plan_expires_at", "VARCHAR DEFAULT ''"),
    ("node_notes", "revision", "INTEGER NOT NULL DEFAULT 0"),
    ("node_versions", "author", "VARCHAR DEFAULT ''"),
    ("service_visits", "referrer", "VARCHAR DEFAULT ''"),
    ("service_visits", "referrer_host", "VARCHAR DEFAULT ''"),
    ("service_visits", "source", "VARCHAR DEFAULT ''"),
    ("service_visits", "search_query", "VARCHAR DEFAULT ''"),
    ("service_visits", "utm_source", "VARCHAR DEFAULT ''"),
    ("service_visits", "utm_medium", "VARCHAR DEFAULT ''"),
    ("service_visits", "utm_campaign", "VARCHAR DEFAULT ''"),
    ("service_visits", "utm_term", "VARCHAR DEFAULT ''"),
    ("service_visits", "user_agent", "VARCHAR DEFAULT ''"),
    ("service_visits", "is_bot", "VARCHAR DEFAULT '0'"),
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

def _tg_journal(level: str, event: str, **details) -> None:
    # Callers pass operational metadata only, never messages, URLs or exception text.
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    _TG_LOG.log(getattr(logging, level), "telegram event=%s %s", event, json.dumps(details, ensure_ascii=False))
    try:
        with SessionLocal.begin() as db:
            entry = TelegramJournalDB(created_at=stamp, level=level, event=event, details=details)
            db.add(entry)
            db.flush()
            db.query(TelegramJournalDB).filter(TelegramJournalDB.id <= entry.id - _TG_JOURNAL_SIZE).delete(
                synchronize_session=False
            )
    except Exception as exc:
        _TG_LOG.error("telegram event=journal_write_failed exception=%s", type(exc).__name__)


def _tg_escape(text: str) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _tg_configured_chat_ids() -> list[str]:
    return list(dict.fromkeys(part.strip() for part in TG_CHAT_ID.split(",") if part.strip()))


def _tg_load_chat_ids() -> list[str]:
    ids = _tg_configured_chat_ids()
    try:
        with open(TG_CHATS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("chat_ids") or []:
            s = str(item).strip()
            if s and s not in ids:
                ids.append(s)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        _tg_journal("ERROR", "subscriptions_read_failed", exception=type(exc).__name__)
    return ids


def _tg_remember_chat(chat_id) -> bool | None:
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
        except OSError as exc:
            _tg_journal("ERROR", "subscription_save_failed", exception=type(exc).__name__)
            return None
        return True


def _tg_failure_reason(code: int, description: str) -> str:
    description = description.lower()
    for fragment, explanation in (
        ("terminated by other getupdates", "Другой процесс уже опрашивает этого бота"),
        ("while webhook is active", "Активный webhook блокирует getUpdates"),
        ("bot was blocked", "Получатель заблокировал бота"),
        ("bot was kicked", "Бот удалён из чата"),
        ("chat not found", "Telegram не нашёл чат"),
        ("can't parse entities", "Telegram отклонил разметку сообщения"),
        ("message is too long", "Сообщение превышает лимит Telegram"),
    ):
        if fragment in description:
            return explanation
    return {
        401: "Telegram не принял токен бота",
        403: "Telegram запретил операцию",
        409: "Конфликт получения обновлений",
        429: "Превышен лимит запросов Telegram",
    }.get(code, "Telegram API отклонил запрос")


def _tg_api(method: str, payload: dict | None = None, *, timeout: float = 25.0) -> tuple[dict, dict]:
    started = time.monotonic()
    details = {"method": method, "connection": "прокси" if TG_HTTP_PROXY else "напрямую", "timeout_s": timeout}
    if not TG_BOT_TOKEN:
        return {"ok": False}, {**details, "reason": "Токен бота не задан"}
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
        details["http_status"] = resp.status_code
        try:
            data = resp.json()
        except ValueError:
            data = None
        if not isinstance(data, dict) or type(data.get("ok")) is not bool:
            details["reason"] = "Ответ не соответствует формату Telegram API"
        elif not resp.is_success or not data["ok"]:
            code = data.get("error_code")
            if type(code) is int:
                details["telegram_code"] = code
            details["reason"] = _tg_failure_reason(details.get("telegram_code", resp.status_code), str(data.get("description") or ""))
            params = data.get("parameters")
            retry_after = params.get("retry_after") if isinstance(params, dict) else None
            if type(retry_after) is int and 0 < retry_after <= 86_400:
                details["retry_after_s"] = retry_after
        elif method == "getUpdates" and not isinstance(data.get("result"), list):
            details["reason"] = "Ответ getUpdates не содержит список обновлений"
        else:
            return data, {**details, "elapsed_ms": round((time.monotonic() - started) * 1000)}
    except Exception as exc:
        details["exception"] = type(exc).__name__
        details["reason"] = "Истекло время ожидания Telegram" if isinstance(exc, httpx.TimeoutException) else "Запрос к Telegram не выполнен"
        cause = exc.__cause__
        while cause is not None:
            details["cause"] = type(cause).__name__
            if isinstance(cause, OSError) and cause.errno is not None:
                details["errno"] = cause.errno
                break
            cause = cause.__cause__
    return {"ok": False}, {**details, "elapsed_ms": round((time.monotonic() - started) * 1000)}


def _tg_send_to(chat_id: str, text: str, *, purpose: str) -> bool:
    data, details = _tg_api(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text[:3900],
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=12.0,
    )
    ok = data["ok"]
    if not ok or purpose != "journal":
        _tg_journal("INFO" if ok else "ERROR", "send_ok" if ok else "send_failed", purpose=purpose, **details)
    return ok


def tg_notify(text: str, *, purpose: str, problem: dict | None = None) -> None:
    """Fire-and-forget Telegram alert to all known admin chats."""
    if not TG_BOT_TOKEN:
        return

    def _deliver():
        if problem is not None:
            _tg_journal("ERROR", "alert_triggered", purpose=purpose, **problem)
        chats = _tg_load_chat_ids()
        if not chats:
            _tg_journal("WARNING", "no_recipients", purpose=purpose)
            return
        body = text.strip()
        if not body:
            return
        for chat_id in chats:
            _tg_send_to(chat_id, body, purpose=purpose)

    try:
        threading.Thread(target=_deliver, daemon=True, name="tg-delivery").start()
    except RuntimeError as exc:
        _tg_journal("ERROR", "delivery_start_failed", purpose=purpose, exception=type(exc).__name__)


def tg_notify_ai_issue(where: str, detail: str, *, exception: Exception | None = None) -> None:
    """Alert when DeepSeek / AI API fails; debounced per signature."""
    sig = f"{where}|{(detail or '')[:160]}"
    now = time.time()
    with _TG_LOCK:
        last = _TG_AI_FAIL_LAST.get(sig, 0.0)
        if now - last < _TG_AI_FAIL_COOLDOWN_SEC:
            return
        _TG_AI_FAIL_LAST[sig] = now
    status = re.search(r"\(([45]\d\d)\):", detail or "")
    problem = {"component": where}
    if exception is not None:
        problem["exception"] = type(exception).__name__
    elif status:
        problem["http_status"] = int(status.group(1))
    elif not (AI_API_KEY and AI_API_URL and AI_CHAT_MODEL) and where != "ai-agent-vision":
        problem["reason"] = "Настройки AI не заданы"
    elif detail == "AI вернул пустой ответ.":
        problem["reason"] = "AI вернул пустой ответ"
    tg_notify(
        "⚠️ <b>DeepSeek / AI недоступен</b>\n"
        f"Где: {_tg_escape(where)}\n"
        f"Детали: {_tg_escape((detail or '')[:500])}\n"
        f"Время: {_tg_escape(datetime.datetime.now().isoformat(timespec='seconds'))}",
        purpose="DeepSeek / AI",
        problem=problem,
    )


_TG_EVENT_LABELS = {
    "poller_started": "Запущен опрос Telegram",
    "webhook_removed": "Webhook отключён для опроса",
    "webhook_remove_failed": "Не удалось отключить webhook",
    "poll_failed": "Не удалось получить обновления",
    "poll_recovered": "Получение обновлений восстановлено",
    "update_failed": "Не удалось обработать обновление",
    "send_ok": "Telegram принял сообщение",
    "send_failed": "Не удалось отправить сообщение",
    "delivery_start_failed": "Не удалось запустить отправку",
    "subscriptions_read_failed": "Не удалось прочитать список получателей",
    "subscription_save_failed": "Не удалось сохранить получателя",
    "no_recipients": "Нет чатов для отправки",
    "alert_triggered": "Сбой Lab-Journal, для которого создано уведомление",
}

_TG_DETAIL_LABELS = {
    "method": "метод", "purpose": "уведомление", "component": "компонент", "route": "маршрут",
    "connection": "подключение", "http_status": "HTTP", "telegram_code": "код Telegram",
    "exception": "исключение", "cause": "причина", "errno": "системный код",
    "elapsed_ms": "длительность, мс", "timeout_s": "таймаут, с", "retry_after_s": "лимит Telegram, с",
    "retry_in_s": "следующий опрос через, с", "failures": "сбоев подряд", "update_id": "обновление",
    "source": "источник", "configured_chats": "разрешённых чатов",
}


def _tg_send_journal(chat_id: str, command: list[str]) -> None:
    errors_only = command[1:2] == ["errors"]
    count = command[2:3] if errors_only else command[1:2]
    limit = max(1, min(20, int(count[0][:8]))) if count and re.fullmatch(r"[0-9]+", count[0]) else 10
    try:
        with SessionLocal() as db:
            query = db.query(TelegramJournalDB)
            if errors_only:
                query = query.filter(TelegramJournalDB.level.in_(("ERROR", "WARNING")))
            rows = query.order_by(TelegramJournalDB.id.desc()).limit(limit).all()
    except Exception as exc:
        _TG_LOG.error("telegram event=journal_read_failed exception=%s", type(exc).__name__)
        _tg_send_to(chat_id, "Журнал недоступен. Проверьте журнал службы Biolab на сервере.", purpose="journal")
        return
    if not rows:
        _tg_send_to(chat_id, "В журнале нет ошибок и предупреждений." if errors_only else "В журнале пока нет записей.", purpose="journal")
        return

    title = "Ошибки и предупреждения" if errors_only else "Журнал Telegram-бота"
    page = f"<b>{title}</b> · время МСК"
    for row in rows:
        when = datetime.datetime.fromisoformat(row.created_at).astimezone(
            datetime.timezone(datetime.timedelta(hours=3))
        ).strftime("%d.%m %H:%M:%S")
        label = _TG_EVENT_LABELS.get(row.event, row.event)
        details = " · ".join(f"{name} {row.details[key]}" for key, name in _TG_DETAIL_LABELS.items() if key in row.details)
        entry = f"<b>#{row.id} · {when} · {row.level}</b>\n{_tg_escape(label)}"
        if row.details.get("reason"):
            entry += f"\n{_tg_escape(row.details['reason'])}"
        if details:
            entry += f"\n{_tg_escape(details)}"
        if len(page) + len(entry) + 2 > 3900:
            if not _tg_send_to(chat_id, page, purpose="journal"):
                return
            page = f"<b>{title} · продолжение</b>"
        page += f"\n\n{entry}"
    _tg_send_to(chat_id, page, purpose="journal")


def _tg_handle_update(update: dict, *, source: str) -> None:
    msg = update.get("message") or update.get("edited_message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return
    chat_id = str(chat_id)
    text = (msg.get("text") or "").strip()
    command = text.lower().split()
    name = command[0].split("@", 1)[0] if command else ""
    if name == "/logs":
        if source == "poller":
            if chat_id in _tg_configured_chat_ids():
                _tg_send_journal(chat_id, command)
            else:
                _tg_send_to(chat_id, "Журнал доступен только чатам, заданным администратором Biolab.", purpose="journal")
        return

    is_new = _tg_remember_chat(chat_id)
    if is_new is None:
        _tg_send_to(chat_id, "Не удалось сохранить подписку. Попробуйте /start ещё раз.", purpose="подписка")
        return
    if name == "/start" or is_new:
        journal_help = "\n/logs — журнал бота. /logs errors — ошибки и предупреждения." if chat_id in _tg_configured_chat_ids() else ""
        _tg_send_to(
            chat_id,
            "✅ <b>Lab-Journal</b>: алерты подключены.\n"
            f"Сюда приходят регистрации, заявки LabOS, ошибки сервера и сбои DeepSeek/AI.{journal_help}",
            purpose="подписка",
        )
        if is_new:
            ai_ok = bool(AI_API_KEY and AI_API_URL and AI_CHAT_MODEL)
            tg_notify(
                "🔔 <b>Новый чат для алертов</b>\n"
                f"chat_id: <code>{_tg_escape(chat_id)}</code>\n"
                f"AI настроен: {'да' if ai_ok else 'нет (AI_API_* пусты)'}",
                purpose="новый чат",
            )


def _tg_process_update(update, *, source: str) -> None:
    try:
        _tg_handle_update(update, source=source)
    except Exception as exc:
        if source == "webhook":
            _TG_LOG.warning("telegram event=webhook_update_failed exception=%s", type(exc).__name__)
            return
        details = {"source": source, "exception": type(exc).__name__}
        update_id = update.get("update_id") if isinstance(update, dict) else None
        if type(update_id) is int:
            details["update_id"] = update_id
        _tg_journal("ERROR", "update_failed", **details)


def _tg_poller_loop() -> None:
    """Long-poll getUpdates so /start binds chat_id without HTTPS webhook."""
    offset = 0
    webhook, details = _tg_api("deleteWebhook", {"drop_pending_updates": False}, timeout=10.0)
    _tg_journal("INFO" if webhook["ok"] else "ERROR", "webhook_removed" if webhook["ok"] else "webhook_remove_failed", **details)
    failures, last_signature, last_report = 0, None, 0.0
    while True:
        data, details = _tg_api(
            "getUpdates",
            {"timeout": 25, "offset": offset, "allowed_updates": ["message"]},
            timeout=35.0,
        )
        if not data["ok"]:
            failures += 1
            signature = {key: value for key, value in details.items() if key != "elapsed_ms"}
            wait = max(5, details.get("retry_after_s", 0))
            now = time.monotonic()
            if signature != last_signature or now - last_report >= 60:
                _tg_journal("ERROR", "poll_failed", failures=failures, retry_in_s=wait, **details)
                last_signature, last_report = signature, now
            time.sleep(wait)
            continue
        if failures:
            _tg_journal("INFO", "poll_recovered", failures=failures, **details)
            failures, last_signature = 0, None
        for update in data["result"]:
            update_id = update.get("update_id") if isinstance(update, dict) else None
            if type(update_id) is int:
                offset = max(offset, update_id + 1)
            _tg_process_update(update, source="poller")


def start_tg_poller() -> None:
    global _TG_POLLER_STARTED
    if not TG_BOT_TOKEN:
        _TG_LOG.info("telegram event=poller_disabled reason=missing_token")
        return
    if _TG_POLLER_STARTED:
        return
    _tg_journal("INFO", "poller_started", connection="прокси" if TG_HTTP_PROXY else "напрямую", configured_chats=len(_tg_configured_chat_ids()))
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
            "🟢 <b>Lab-Journal</b> перезапущен\n"
            f"Чатов для уведомлений: {len(chats)}\n"
            f"AI: {'настроен' if ai_ok else 'не настроен (AI_API_* пусты)'}",
            purpose="перезапуск",
        )
        if not ai_ok:
            tg_notify_ai_issue("startup", "AI_API_KEY / AI_API_URL / AI_CHAT_MODEL не заданы в окружении")

    threading.Thread(target=_boot, daemon=True).start()


app = FastAPI()
@app.get("/static/uploads/{filename}", include_in_schema=False)
def serve_upload(filename: str, request: Request):
    with SessionLocal() as db:
        return _serve_protected_upload(db, request, filename)


app.mount("/static", StaticFiles(directory="static"), name="static")
PLAN_LABELS = {
    "free": "Free",
    "student": "Pro",  # legacy id; product plan «Студент» = Pro
    "pro": "Pro",
    "corp": "LabOS",
    "labos": "LabOS",
}

WELCOME_TRIAL_DAYS = 30
STORAGE_LIMIT_FREE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
STORAGE_LIMIT_PRO_BYTES = 20 * 1024 * 1024 * 1024  # 20 GB


def _parse_iso_dt(value: str) -> datetime.datetime | None:
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00").split("+")[0])
    except Exception:
        return None


def grant_welcome_trial(user: UserDB) -> None:
    """New signups get Pro for WELCOME_TRIAL_DAYS (default 30), then sync drops to Free."""
    days = max(1, int(WELCOME_TRIAL_DAYS or 30))
    end = datetime.datetime.now() + datetime.timedelta(days=days)
    user.plan = "pro"
    user.plan_expires_at = end.replace(microsecond=0).isoformat()


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
            user.plan_expires_at = ""
            db.commit()
            return "free"
    return plan


def plan_allows_ai(plan: str) -> bool:
    return (plan or "free").strip() in ("student", "pro", "corp", "labos")


def storage_limit_bytes(plan: str) -> int:
    return STORAGE_LIMIT_PRO_BYTES if plan_allows_ai(plan) else STORAGE_LIMIT_FREE_BYTES


def _dir_size_bytes(path: str) -> int:
    total = 0
    if not path or not os.path.isdir(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _user_storage_bytes(username: str, db: Session) -> int:
    total = _dir_size_bytes(os.path.join(NOTEBOOKS_BASE, username))
    total += db.query(func.coalesce(func.sum(MediaUploadDB.size), 0)).filter_by(owner_id=username).scalar()
    # Attachment sizes are stored as strings in older databases.
    rows = db.query(ExperimentFileDB.file_size).join(ExperimentDB, ExperimentDB.id == ExperimentFileDB.exp_id)
    for (size,) in rows.filter(ExperimentDB.user_id == username):
        try:
            total += max(0, int(size))
        except (TypeError, ValueError):
            continue
    return total


def format_bytes_ru(n: int) -> str:
    n = max(0, int(n or 0))
    for unit, div in (("ГБ", 1024 ** 3), ("МБ", 1024 ** 2), ("КБ", 1024)):
        if n >= div:
            val = n / div
            return f"{val:.1f} {unit}" if val < 10 else f"{val:.0f} {unit}"
    return f"{n} Б"


def lab_report_template_levels() -> list:
    """Student-friendly lab report stages (not «root/node»)."""
    return [
        [{"id": "goal", "title": "1. Цель работы", "content": "Сформулируйте цель лабораторной работы."}],
        [{"id": "materials", "title": "2. Материалы и оборудование", "content": "Перечислите реактивы, приборы и ПО."}],
        [{"id": "procedure", "title": "3. Ход работы", "content": "Опишите шаги эксперимента по порядку."}],
        [{"id": "results", "title": "4. Результаты", "content": "Запишите наблюдения, таблицы и фото."}],
        [{"id": "conclusion", "title": "5. Выводы", "content": "Сформулируйте выводы и сравните с целью."}],
    ]


def blank_lab_levels() -> list:
    return [[{"id": "start", "title": "Запись", "content": "Начните запись лабораторной работы."}]]


def _plan_context(request: Request):
    plan = getattr(request.state, "user_plan", None) or "free"
    expires = (getattr(request.state, "plan_expires_at", None) or "").strip()
    label = PLAN_LABELS.get(plan, "Free")
    if plan != "free" and expires:
        end = _parse_iso_dt(expires)
        if end and end > datetime.datetime.now():
            label = f"{label} · до {end.strftime('%d.%m.%Y')}"
    return {
        "user_plan": plan,
        "plan_label": label,
        "plan_expires_at": expires,
        "graph_enabled": EXPERIMENT_GRAPH_ENABLED,
        "user_avatar_url": avatar_src(getattr(request.state, "user_avatar_url", "") or ""),
    }


templates = Jinja2Templates(directory="templates", context_processors=[_plan_context])
templates.env.globals["base_path"] = BASE_PATH
templates.env.globals["lab_admin_path"] = LAB_ADMIN_PATH
templates.env.globals["site_origin"] = SITE_ORIGIN
templates.env.globals["legal_base_url"] = LEGAL_BASE_URL
templates.env.filters["fix_uploads"] = lambda value: sanitize_note_html(value or "")
templates.env.filters["parse_amount"] = lambda value: parse_amount(value)
templates.env.filters["avatar_src"] = avatar_src


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
LEGAL_CONTENT_DIR = os.path.join(WEB_DIR, "content", "legal")

LEGAL_PAGES = {
    "public-offer": {
        "slug": "public-offer",
        "h1": "Публичная оферта",
        "seo_title": "Публичная оферта",
        "description": "Публичная оферта на использование сервиса Lab-Journal (labjournal.tech).",
        "updated": "18 ноября 2026 г.",
        "file": "public-offer.html",
    },
    "privacy": {
        "slug": "privacy",
        "h1": "Политика конфиденциальности",
        "seo_title": "Политика конфиденциальности",
        "description": "Политика обработки персональных данных пользователей Lab-Journal.",
        "updated": "14 ноября 2026 г.",
        "file": "privacy.html",
    },
    "consent-personal-data": {
        "slug": "consent-personal-data",
        "h1": "Согласие на обработку персональных данных",
        "seo_title": "Согласие на обработку персональных данных",
        "description": "Согласие на обработку персональных данных при регистрации в Lab-Journal.",
        "updated": "14 ноября 2026 г.",
        "file": "consent-personal-data.html",
    },
    "consent-mailing": {
        "slug": "consent-mailing",
        "h1": "Согласие на рассылку",
        "seo_title": "Согласие на рекламно-информационную рассылку",
        "description": "Согласие на получение рекламно-информационной рассылки Lab-Journal.",
        "updated": "14 ноября 2026 г.",
        "file": "consent-mailing.html",
    },
    "requisites": {
        "slug": "requisites",
        "h1": "Реквизиты ООО «Антикусь»",
        "seo_title": "Реквизиты ООО «Антикусь»",
        "description": "Реквизиты ООО «Антикусь»: ОГРН, ИНН, КПП и контакты оператора Lab-Journal.",
        "updated": "20 ноября 2026 г.",
        "file": "requisites.html",
    },
}


def _load_legal_body_html(slug: str) -> str:
    meta = LEGAL_PAGES.get(slug) or {}
    fname = meta.get("file") or f"{slug}.html"
    path = os.path.join(LEGAL_CONTENT_DIR, fname)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _legal_page_resolved(slug: str):
    meta = LEGAL_PAGES.get(slug)
    if not meta:
        return None
    page = dict(meta)
    body = _load_legal_body_html(slug).strip()
    if not body:
        return None
    page["body_html"] = body
    return page


def _request_inner_path(request: Request) -> str:
    path = request.url.path or "/"
    if BASE_PATH and path.startswith(BASE_PATH + "/"):
        return path[len(BASE_PATH) :] or "/"
    if BASE_PATH and path == BASE_PATH:
        return "/"
    return path


_BOT_UA_NEEDLES = (
    "bot", "crawl", "spider", "slurp", "fetcher", "scanner",
    "facebookexternalhit", "preview", "lighthouse", "pagespeed",
    "pingdom", "uptimerobot", "python-requests", "python-urllib",
    "curl/", "wget/", "httpclient", "scrapy", "httpx", "aiohttp",
    "semrush", "ahrefs", "mj12", "dotbot", "petalbot", "bytespider",
    "yandexbot", "googlebot", "bingbot", "duckduckbot", "baiduspider",
    "applebot", "chatgpt", "gptbot", "claudebot", "ccbot", "anthropic",
    "headless", "phantomjs", "selenium", "puppeteer", "playwright",
    "monitoring", "statuscake", "siteaudit", "linkchecker",
    "go-http-client", "java/", "libwww", "okhttp",
)


def _is_bot_user_agent(ua: str) -> bool:
    u = (ua or "").strip().lower()
    if not u or len(u) < 12:
        return True
    return any(n in u for n in _BOT_UA_NEEDLES)


def _should_track_service_visit(inner_path: str, method: str) -> bool:
    if method.upper() != "GET":
        return False
    if inner_path.startswith("/api/"):
        return False
    if inner_path.startswith("/static"):
        return False
    if inner_path.startswith(LAB_ADMIN_PATH) or inner_path.startswith("/admin"):
        return False
    if inner_path in (
        "/favicon.ico",
        "/sitemap-guides.xml",
        "/sitemap.xml",
        "/robots.txt",
        "/llms.txt",
    ):
        return False
    # tech / legal noise often hit by auditors — still track, but analytics can filter
    return True


def _own_visit_hosts() -> set[str]:
    hosts = {
        "labjournal.tech",
        "www.labjournal.tech",
        "bionotebook.ru",
        "www.bionotebook.ru",
        "anticus.tech",
        "www.anticus.tech",
        "176.109.103.209",
    }
    hosts.update(LEGACY_HOSTS)
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
    user_agent: str = "",
    is_bot: bool = False,
) -> None:
    if not visitor_id:
        return
    # Do not persist obvious bots — keeps uniques honest
    if is_bot:
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
                user_agent=(user_agent or "")[:512],
                is_bot="0",
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


_NOISE_PATH_PREFIXES = (
    "/privacy",
    "/public-offer",
    "/consent-",
    "/requisites",
    "/rekvizity",
    "/legal/",
    "/terms",
    "/cookie",
)
_PRODUCT_PATH_PREFIXES = (
    "/",
    "/login",
    "/register",
    "/cabinet",
    "/ai",
    "/guides",
    "/pricing",
    "/about",
    "/experiments",
    "/projects",
    "/inventory",
    "/equipment",
    "/friends",
    "/support",
    "/planner",
    "/lab",
)


def _is_noise_path(path: str) -> bool:
    p = (path or "/").strip() or "/"
    return any(p == pref or p.startswith(pref) for pref in _NOISE_PATH_PREFIXES if pref != "/")


def _is_product_path(path: str) -> bool:
    p = (path or "/").strip() or "/"
    if _is_noise_path(p):
        return False
    if p == "/":
        return True
    return any(p == pref or p.startswith(pref + "/") or p.startswith(pref + "?") for pref in _PRODUCT_PATH_PREFIXES if pref != "/")


def _classify_visitor_quality(hits: list[dict]) -> str:
    """bot | noise | bounce | human | engaged — for legacy + live rows."""
    if not hits:
        return "noise"
    for h in hits:
        ua = (h.get("user_agent") or "").strip()
        if h.get("is_bot") == "1" or (ua and _is_bot_user_agent(ua)):
            return "bot"
    paths = [(h.get("path") or "/") for h in hits]
    n = len(hits)
    if all(_is_noise_path(p) for p in paths):
        return "noise"
    if n >= 2:
        return "engaged"
    # single hit
    path = paths[0]
    source = (hits[0].get("source") or "").strip().lower()
    referrer = (hits[0].get("referrer") or "").strip()
    if _is_noise_path(path):
        return "noise"
    # lone homepage / empty direct with no product signal → bounce (often scanners)
    if path == "/" and not referrer and source in ("", "прямой заход", "direct", "прямой"):
        return "bounce"
    if _is_product_path(path) or referrer or source not in ("", "прямой заход", "direct", "прямой"):
        return "human"
    return "bounce"


def _quality_passes(q: str, mode: str) -> bool:
    """mode: raw | human | engaged"""
    if mode == "raw":
        return True
    if mode == "engaged":
        return q == "engaged"
    # human = people-ish: human + engaged (exclude bot/noise/bounce)
    return q in ("human", "engaged")


def _load_visit_rows(db: Session, since: str | None = None) -> list:
    q = db.query(
        ServiceVisitDB.visitor_id,
        ServiceVisitDB.path,
        ServiceVisitDB.source,
        ServiceVisitDB.referrer,
        ServiceVisitDB.referrer_host,
        ServiceVisitDB.search_query,
        ServiceVisitDB.utm_campaign,
        ServiceVisitDB.user_agent,
        ServiceVisitDB.is_bot,
        ServiceVisitDB.created_at,
    )
    if since:
        q = q.filter(ServiceVisitDB.created_at >= since)
    return q.all()


def _group_visits_by_visitor(rows) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for r in rows:
        vid = r.visitor_id or ""
        by.setdefault(vid, []).append(
            {
                "path": r.path or "/",
                "source": r.source or "",
                "referrer": r.referrer or "",
                "referrer_host": r.referrer_host or "",
                "search_query": r.search_query or "",
                "utm_campaign": r.utm_campaign or "",
                "user_agent": getattr(r, "user_agent", None) or "",
                "is_bot": getattr(r, "is_bot", None) or "0",
                "created_at": r.created_at or "",
            }
        )
    return by


def _service_visit_top_filtered(
    by_visitor: dict[str, list[dict]],
    field: str,
    *,
    mode: str = "human",
    limit: int = 20,
    exclude_empty: bool = True,
) -> list[dict]:
    tallies: dict[str, dict] = {}
    for vid, hits in by_visitor.items():
        q = _classify_visitor_quality(hits)
        if not _quality_passes(q, mode):
            continue
        # first-touch attribution
        label = (hits[0].get(field) or "").strip()
        if exclude_empty and not label:
            continue
        if not label:
            label = "—"
        slot = tallies.setdefault(label, {"visits": 0, "visitors": set()})
        slot["visits"] += len(hits)
        slot["visitors"].add(vid)
    ranked = sorted(tallies.items(), key=lambda kv: (-kv[1]["visits"], kv[0]))[:limit]
    return [
        {"label": label, "visits": int(data["visits"]), "visitors": len(data["visitors"])}
        for label, data in ranked
    ]


def _service_visit_daily_series_filtered(
    by_visitor: dict[str, list[dict]],
    *,
    days: int = 30,
    mode: str = "human",
) -> list[dict]:
    now = datetime.datetime.now()
    start_date = (now - datetime.timedelta(days=days - 1)).date()
    buckets: dict[str, dict] = {}
    for i in range(days):
        d = start_date + datetime.timedelta(days=i)
        key = d.isoformat()
        buckets[key] = {"date": key, "visits": 0, "visitors": set()}

    for vid, hits in by_visitor.items():
        q = _classify_visitor_quality(hits)
        if not _quality_passes(q, mode):
            continue
        for h in hits:
            day = _parse_service_visit_day(h.get("created_at") or "")
            if not day:
                continue
            key = day.isoformat()
            if key not in buckets:
                continue
            buckets[key]["visits"] += 1
            buckets[key]["visitors"].add(vid)

    return [
        {
            "date": key,
            "visits": buckets[key]["visits"],
            "visitors_unique": len(buckets[key]["visitors"]),
        }
        for key in sorted(buckets.keys())
    ]


def _count_period(by_visitor: dict[str, list[dict]], mode: str, since_iso: str | None) -> tuple[int, int]:
    """Returns (unique_visitors, pageviews) for quality mode and optional since filter."""
    uniques = 0
    views = 0
    for vid, hits in by_visitor.items():
        q = _classify_visitor_quality(hits)
        if not _quality_passes(q, mode):
            continue
        if since_iso:
            filtered = [h for h in hits if (h.get("created_at") or "") >= since_iso]
            if not filtered:
                continue
            views += len(filtered)
        else:
            views += len(hits)
        uniques += 1
    return uniques, views


def _service_visit_analytics(db: Session) -> dict:
    now = datetime.datetime.now()
    all_rows = _load_visit_rows(db)
    by_all = _group_visits_by_visitor(all_rows)

    quality_counts = {"bot": 0, "noise": 0, "bounce": 0, "human": 0, "engaged": 0}
    for hits in by_all.values():
        quality_counts[_classify_visitor_quality(hits)] += 1

    since_30 = (now - datetime.timedelta(days=30)).isoformat()
    by_30 = _group_visits_by_visitor([r for r in all_rows if (r.created_at or "") >= since_30])

    def period_bundle(mode: str) -> dict:
        u_tot, v_tot = _count_period(by_all, mode, None)
        out = {
            "visitors_unique_total": u_tot,
            "visits_total": v_tot,
        }
        for days, key_u, key_v in (
            (1, "visitors_unique_24h", "visits_24h"),
            (7, "visitors_unique_7d", "visits_7d"),
            (30, "visitors_unique_30d", "visits_30d"),
        ):
            since = (now - datetime.timedelta(days=days)).isoformat()
            by_window = _group_visits_by_visitor([r for r in all_rows if (r.created_at or "") >= since])
            u, v = _count_period(by_window, mode, None)
            out[key_u] = u
            out[key_v] = v
        return out

    def mode_payload(mode: str, base: dict) -> dict:
        return {
            **base,
            "visits_daily": _service_visit_daily_series_filtered(by_all, days=30, mode=mode),
            "visit_sources": _service_visit_top_filtered(by_all, "source", mode=mode, limit=20),
            "visit_sources_30d": _service_visit_top_filtered(by_30, "source", mode=mode, limit=20),
            "visit_queries": _service_visit_top_filtered(by_all, "search_query", mode=mode, limit=30),
            "visit_queries_30d": _service_visit_top_filtered(by_30, "search_query", mode=mode, limit=30),
            "visit_referrers": _service_visit_top_filtered(by_all, "referrer_host", mode=mode, limit=20),
            "visit_campaigns": _service_visit_top_filtered(by_all, "utm_campaign", mode=mode, limit=15),
        }

    human = period_bundle("human")
    engaged = period_bundle("engaged")
    raw_true = period_bundle("raw")

    regs_total = db.query(func.count(UserDB.username)).scalar() or 0

    def regs_since(days: int) -> int:
        since = (now - datetime.timedelta(days=days)).isoformat()
        try:
            return (
                db.query(func.count(AuditLogDB.id))
                .filter(AuditLogDB.action == "register")
                .filter(AuditLogDB.created_at >= since)
                .scalar()
                or 0
            )
        except Exception:
            return 0

    human_u = max(int(human["visitors_unique_total"] or 0), 1)
    conversion_pct = round(100.0 * float(regs_total) / float(human_u), 2)
    human_full = mode_payload("human", human)

    return {
        **human_full,
        "visit_filter": "human",
        "visit_modes": {
            "human": human_full,
            "engaged": mode_payload("engaged", engaged),
            "raw": mode_payload("raw", raw_true),
        },
        "visit_quality": quality_counts,
        "visit_raw": raw_true,
        "registrations_total": int(regs_total),
        "registrations_24h": int(regs_since(1)),
        "registrations_7d": int(regs_since(7)),
        "registrations_30d": int(regs_since(30)),
        "conversion_pct": conversion_pct,
        "visit_filter_note": (
            "Люди = без ботов, юр.шума и одиночных прямых заходов на /. "
            "Вовлечённые = 2+ просмотра. Сырые = всё из БД (вкл. сканеры)."
        ),
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
    user_agent = (request.headers.get("user-agent") or "").strip()
    is_bot = _is_bot_user_agent(user_agent)
    was_head = bool(getattr(request.state, "biobook_was_head", False) or request.scope.get("biobook_was_head"))
    response = await call_next(request)
    if new_visitor and not is_bot:
        response.set_cookie(
            VISITOR_COOKIE,
            visitor_id,
            max_age=VISITOR_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=COOKIE_SECURE,
        )
    if (not was_head) and (not is_bot) and _should_track_service_visit(inner, request.method):
        _record_service_visit(
            visitor_id,
            inner,
            referer=referer,
            request_url=request_url,
            user_agent=user_agent,
            is_bot=False,
        )
    return response


@app.middleware("http")
async def head_requests_middleware(request: Request, call_next):
    if request.method.upper() == "HEAD":
        request.state.biobook_was_head = True
        request.scope["biobook_was_head"] = True
        request.scope["method"] = "GET"
        response = await call_next(request)
        headers = {k: v for k, v in response.headers.items()}
        return Response(status_code=response.status_code, headers=headers)
    return await call_next(request)


@app.middleware("http")
async def canonical_host_redirect_middleware(request: Request, call_next):
    """301 legacy / www hosts → SITE_ORIGIN (labjournal.tech)."""
    if request.method.upper() not in ("GET", "HEAD"):
        return await call_next(request)
    host = (request.headers.get("host") or "").split(":")[0].lower().strip(".")
    canonical_host = (urlparse(SITE_ORIGIN).hostname or "").lower().strip(".")
    if host and canonical_host and host != canonical_host and host in LEGACY_HOSTS:
        target = f"{SITE_ORIGIN}{request.url.path}"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return RedirectResponse(url=target, status_code=301)
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
    for slug in LEGAL_PAGES:
        if inner == f"/{slug}/":
            target = f"{BASE_PATH}/{slug}"
            if request.url.query:
                target = f"{target}?{request.url.query}"
            return RedirectResponse(url=target, status_code=301)
    # Auth pages: trailing slash → canonical without slash
    if inner in ("/login/", "/register/", "/logout/"):
        target = f"{BASE_PATH}{inner.rstrip('/')}"
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
    for slug in LEGAL_PAGES:
        if path in (f"/{slug}", f"/{slug}/", f"/legal/{slug}", f"/legal/{slug}/"):
            return True
    return False

_NOT_FOUND_HTML = (
    '<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
    '<meta name="robots" content="noindex">'
    "<title>Страница не найдена · Lab-Journal</title>"
    "<style>body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;"
    "background:#05070a;color:#e6edf3;font-family:system-ui,sans-serif;text-align:center;padding:1.5rem}"
    "a{color:#64CEFB}</style></head><body><main><h1>404 — страница не найдена</h1>"
    '<p>Такой страницы нет. <a href="{home}/">На главную Lab-Journal</a></p></main></body></html>'
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
    request.state.auth_user = None
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with SessionLocal() as db:
            user = (
                db.query(UserDB).join(LoginSessionDB, LoginSessionDB.user_id == UserDB.username)
                .filter(LoginSessionDB.token_hash == _session_digest(token),
                        LoginSessionDB.expires_at > int(time.time()),
                        UserDB.is_banned != "true").first()
            )
            if user:
                request.state.auth_user = user.username

    request.state.user_plan = "free"
    request.state.user_avatar_url = ""
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

    # Public paths (incl. legal trailing-slash variants) must reach slash-redirect /
    # route handlers even when Starlette has no exact match for the raw path.
    if not session_user and not path_is_public(path) and not _route_exists(path, request.method):
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
                request.state.user_avatar_url = (getattr(u, "avatar_url", None) or "").strip()
        except Exception:
            request.state.user_plan = "free"
        finally:
            db.close()

    response = await call_next(request)
    if path in ("/robots.txt", "/sitemap.xml", "/sitemap-guides.xml", "/llms.txt"):
        return response
    if (
        not path_seo_indexable(path)
        and (not path.startswith("/static") or path.startswith("/static/uploads/"))
    ):
        response.headers.setdefault("X-Robots-Tag", "noindex, nofollow")
    return response


@app.middleware("http")
async def legacy_bio_notebook_path_redirect_middleware(request: Request, call_next):
    """When site runs at apex (BASE_PATH empty), 301 /bio-notebook/* → /*.

    Registered after session_auth so it runs outermost: session_auth otherwise
    404s unknown paths before this redirect can fire.
    """
    if request.method.upper() not in ("GET", "HEAD"):
        return await call_next(request)
    if BASE_PATH:
        return await call_next(request)
    path = request.url.path or "/"
    if path == "/bio-notebook" or path.startswith("/bio-notebook/"):
        rest = path[len("/bio-notebook") :] or "/"
        if not rest.startswith("/"):
            rest = "/" + rest
        target = rest
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return RedirectResponse(url=target, status_code=301)
    return await call_next(request)


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
        f"Время: {_tg_escape(datetime.datetime.now().isoformat(timespec='seconds'))}",
        purpose="ошибка сервера",
        problem={"route": getattr(request.scope.get("route"), "path", None) or "не определён", "exception": type(exc).__name__},
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
        "question": "Чем Lab-Journal отличается от Word?",
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
            "OCR распознаёт рукопись и фото приборов на этапе эксперимента. "
            "ИИ-агент доступен на тарифе Pro: отчёты, проверка расчётов и презентации по вашим записям."
        ),
    },
    {
        "question": "Как экспортировать отчёт?",
        "answer": (
            "Соберите этапы эксперимента в журнале и выгрузите отчёт в DOCX или PDF из кабинета — "
            "перед сдачей сверьте оформление с методичкой кафедры."
        ),
    },
    {
        "question": "Подойдёт ли для кафедры?",
        "answer": (
            "Да: Free и Pro — публичные тарифы; для кафедры или лаборатории — LabOS, "
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
        "description": "Структура отчёта: титульный лист, ход работы, таблицы и выводы. Шаблон DOCX и выгрузка из журнала Lab-Journal. Методичка кафедры важнее универсального образца.",
        "published": "2026-11-10",
        "updated": "2026-11-18",
        "author": "Команда Lab-Journal",
        "hero_image": "guide-otchet-hero.jpg",
        "body_html": "",
    },
    "kak-vesti-laboratornyy-zhurnal": {
        "slug": "kak-vesti-laboratornyy-zhurnal",
        "h1": "Как вести лабораторный журнал",
        "h1_em": "",
        "kicker": "// Практика записей",
        "seo_title": "Как вести лабораторный журнал: правила записей и пример",
        "title": "Как вести лабораторный журнал",
        "description": "Что писать в лабораторном журнале каждый раз, как исправлять записи, пример таблицы и перенос рукописи в электронный журнал Lab-Journal.",
        "published": "2026-11-11",
        "updated": "2026-11-13",
        "author": "Команда Lab-Journal",
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
        "description": "Структура вывода: цель, результат с единицами, погрешность. Шаблон фразы и учебные примеры для вуза; как проверить расчёты перед выводом в Lab-Journal.",
        "published": "2026-11-11",
        "updated": "2026-11-11",
        "author": "Команда Lab-Journal",
        "hero_image": "guide-otchet-hero.jpg",
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
        "updated": "2026-11-18",
        "author": "Команда Lab-Journal",
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
        "author": "Команда Lab-Journal",
        "hero_image": "guide-zhurnal-lab-rabot-hero.jpg",
        "body_html": "",
    },
    "oformlenie-laboratornogo-zhurnala-gost-kafedra": {
        "slug": "oformlenie-laboratornogo-zhurnala-gost-kafedra",
        "h1": "Оформление лабораторного журнала",
        "h1_em": "по ГОСТ и требованиям кафедры",
        "h1_em_plain": True,
        "kicker": "// Журнал и кафедра",
        "seo_title": "Оформление лабораторного журнала по ГОСТ и требованиям кафедры",
        "title": "Оформление лабораторного журнала по ГОСТ и требованиям кафедры",
        "description": "Научпоп-гайд: зачем лабораторный журнал, миф про ГОСТ на тетрадь, требования кафедры, чем журнал отличается от отчёта, шаблон страницы и чек-лист допуска.",
        "published": "2026-11-12",
        "updated": "2026-11-12",
        "author": "Команда Lab-Journal",
        "hero_image": "guide-zhurnal-gost-hero.jpg",
        "body_html": "",
    },
}

MARKETING_PAGES: dict[str, dict] = {
    "sravnenie-elektronnyh-laboratornyh-zhurnalov": {
        "slug": "sravnenie-elektronnyh-laboratornyh-zhurnalov",
        "h1": "Какой электронный лабораторный журнал выбрать",
        "seo_title": "Электронные лабораторные журналы: сравнение для вуза и лаборатории",
        "description": "Сравнение бумажного журнала, Word, Lab-Journal и корпоративных ELN без «лучшего» рейтинга. Для студента, кафедры и лаборатории — честные плюсы и минусы.",
        "published": "2026-11-11",
        "updated": "2026-11-13",
        "body_file": "sravnenie-eln-body.html",
    },
}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_ico():
    path = os.path.join(WEB_DIR, "static", "favicon.ico")
    if not os.path.isfile(path):
        path = os.path.join(WEB_DIR, "static", "favicon-32.png")
    return FileResponse(path, media_type="image/x-icon" if path.endswith(".ico") else "image/png")


def _build_sitemap_xml() -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    base = f"{SITE_ORIGIN}{BASE_PATH}"
    guide_dates = [
        (art.get("updated") or art.get("published") or "")
        for art in GUIDE_ARTICLES.values()
        if _load_guide_body_html(art["slug"]).strip()
    ]
    guides_hub_lastmod = max(guide_dates) if guide_dates else "2026-11-13"
    for loc, lastmod in (
        (f"{base}/", "2026-11-19"),
        (f"{base}/guides/", guides_hub_lastmod),
    ):
        lines.append(f"  <url><loc>{loc}</loc><lastmod>{lastmod}</lastmod></url>")
    for art in GUIDE_ARTICLES.values():
        if not _load_guide_body_html(art["slug"]).strip():
            continue
        loc = f"{base}/guides/{art['slug']}/"
        lines.append(
            f"  <url><loc>{loc}</loc><lastmod>{art.get('updated', art['published'])}</lastmod></url>"
        )
    for page in MARKETING_PAGES.values():
        if not _load_marketing_body_html(page).strip():
            continue
        loc = f"{base}/{page['slug']}/"
        lines.append(
            f"  <url><loc>{loc}</loc><lastmod>{page.get('updated', page['published'])}</lastmod></url>"
        )
    for slug in LEGAL_PAGES:
        if slug == "public-offer":
            legal_mod = "2026-11-18"
        elif slug == "requisites":
            legal_mod = "2026-11-20"
        else:
            legal_mod = "2026-11-14"
        lines.append(f"  <url><loc>{LEGAL_BASE_URL}/{slug}</loc><lastmod>{legal_mod}</lastmod></url>")
    lines.append("</urlset>")
    return "\n".join(lines)


@app.get("/robots.txt", response_class=Response)
async def robots_txt():
    sitemap_url = f"{SITE_ORIGIN}{BASE_PATH}/sitemap.xml" if BASE_PATH else f"{SITE_ORIGIN}/sitemap.xml"
    llms_url = f"{SITE_ORIGIN}{BASE_PATH}/llms.txt" if BASE_PATH else f"{SITE_ORIGIN}/llms.txt"
    body = (
        "User-agent: *\n"
        "Disallow: /api/\n"
        "Disallow: /auth/\n"
        "Disallow: /admin\n"
        f"Disallow: {LAB_ADMIN_PATH}\n"
        "Disallow: /login\n"
        "Disallow: /register\n"
        "Disallow: /dashboard\n"
        "Disallow: /ai-chat\n"
        "Disallow: /account\n"
        "Disallow: /experiment/\n"
        "Disallow: /friends\n"
        "Disallow: /chat\n"
        "Disallow: /share/\n"
        "Disallow: /invite\n"
        "Disallow: /equipment\n"
        "Disallow: /projects\n"
        "Disallow: /materials\n"
        "Disallow: /consumables\n"
        "Disallow: /static/uploads/\n"
        "Clean-param: utm_source&utm_medium&utm_campaign&utm_content&utm_term&yclid\n"
        "\n"
        f"Sitemap: {sitemap_url}\n"
        f"# LLM context: {llms_url}\n"
    )
    return Response(content=body, media_type="text/plain; charset=utf-8")


@app.get("/llms.txt", response_class=Response)
async def llms_txt():
    path = os.path.join(WEB_DIR, "llms.txt")
    try:
        with open(path, encoding="utf-8") as f:
            body = f.read()
    except OSError:
        body = (
            "# Lab-Journal\n\n"
            "> Электронный лабораторный журнал. Канон: https://labjournal.tech\n"
        )
    return Response(content=body, media_type="text/plain; charset=utf-8")


@app.get("/sitemap.xml", response_class=Response)
@app.get("/sitemap-guides.xml", response_class=Response)
async def sitemap_xml():
    return Response(content=_build_sitemap_xml(), media_type="application/xml")


def _load_marketing_body_html(page: dict) -> str:
    fname = page.get("body_file") or f"{page['slug']}-body.html"
    path = os.path.join(MARKETING_CONTENT_DIR, fname)
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return _rewrite_guide_internal_links(f.read().strip())


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
                    {"@type": "ListItem", "position": 2, "name": "Lab-Journal", "item": f"{SITE_ORIGIN}{BASE_PATH}/"},
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
                    {"@type": "ListItem", "position": 2, "name": "Lab-Journal", "item": f"{SITE_ORIGIN}{BASE_PATH}/"},
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


def _guides_hub_lastmod(articles: list[dict] | None = None) -> str:
    arts = articles if articles is not None else list(GUIDE_ARTICLES.values())
    dates = [(a.get("updated") or a.get("published") or "") for a in arts]
    dates = [d for d in dates if d]
    return max(dates) if dates else "2026-11-13"


def _guides_hub_json_ld(articles: list[dict]) -> str:
    page_url = f"{SITE_ORIGIN}{BASE_PATH}/guides/"
    hub_mod = _guides_hub_lastmod(articles)
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
                    {"@type": "ListItem", "position": 2, "name": "Lab-Journal", "item": f"{SITE_ORIGIN}{BASE_PATH}/"},
                    {"@type": "ListItem", "position": 3, "name": "Статьи", "item": page_url},
                ],
            },
            {
                "@type": "CollectionPage",
                "@id": f"{page_url}#webpage",
                "url": page_url,
                "name": "Статьи: отчёты, лабораторный журнал, протоколы",
                "dateModified": hub_mod,
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


@app.get("/legal/{slug}")
@app.get("/legal/{slug}/")
async def legal_alias_redirect(slug: str):
    key = (slug or "").strip("/")
    if key not in LEGAL_PAGES:
        return _not_found_response(f"/legal/{slug}")
    target = f"{BASE_PATH}/{key}" if BASE_PATH else f"/{key}"
    return RedirectResponse(url=target, status_code=301)


@app.get("/public-offer", response_class=HTMLResponse)
@app.get("/public-offer/", response_class=HTMLResponse)
@app.get("/privacy", response_class=HTMLResponse)
@app.get("/privacy/", response_class=HTMLResponse)
@app.get("/consent-personal-data", response_class=HTMLResponse)
@app.get("/consent-personal-data/", response_class=HTMLResponse)
@app.get("/consent-mailing", response_class=HTMLResponse)
@app.get("/consent-mailing/", response_class=HTMLResponse)
@app.get("/requisites", response_class=HTMLResponse)
@app.get("/requisites/", response_class=HTMLResponse)
@app.get("/rekvizity", response_class=HTMLResponse)
@app.get("/rekvizity/", response_class=HTMLResponse)
async def legal_document_page(request: Request):
    slug = (_request_inner_path(request) or "/").strip("/")
    if slug == "rekvizity":
        target = f"{BASE_PATH}/requisites" if BASE_PATH else "/requisites"
        return RedirectResponse(url=target, status_code=301)
    page = _legal_page_resolved(slug)
    if not page:
        return _not_found_response(f"/{slug}")
    return templates.TemplateResponse(
        request,
        "legal_page.html",
        {
            "page": page,
            "legal_base_url": LEGAL_BASE_URL,
        },
    )


@app.get("/guides", response_class=HTMLResponse)
@app.get("/guides/", response_class=HTMLResponse)
async def guides_index(request: Request):
    articles = sorted(GUIDE_ARTICLES.values(), key=lambda a: a["published"], reverse=True)
    hub_updated = _guides_hub_lastmod(articles)
    return templates.TemplateResponse(
        request,
        "guides_index.html",
        {
            "articles": articles,
            "guides_json_ld": _guides_hub_json_ld(articles),
            "hub_updated": hub_updated,
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
    consent = payload.get("consent_personal_data") in (True, "true", "1", 1, "on", "yes")

    if not consent:
        return JSONResponse(
            {"ok": False, "error": "Нужно согласие на обработку персональных данных"},
            status_code=400,
        )
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
        f"id: {lead.id}",
        purpose="заявка LabOS",
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
        _tg_process_update(body, source="webhook")
    return JSONResponse({"ok": True})


# ── Auth ──

@app.get("/index.html", include_in_schema=False)
async def index_html_redirect():
    """Close A8 tail: old anticus links to …/index.html → apex home."""
    return RedirectResponse(url=f"{BASE_PATH}/", status_code=301)


@app.get("/yandex_bd93356772fa18fd.html", include_in_schema=False)
async def yandex_webmaster_verification():
    path = os.path.join(WEB_DIR, "yandex_bd93356772fa18fd.html")
    return FileResponse(path, media_type="text/html; charset=utf-8")


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
        plan_expires_at="",
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
        details=f"plan_pro;welcome_trial_{WELCOME_TRIAL_DAYS}d",
        ip=ip,
    )
    tg_notify(
        "🆕 <b>Регистрация</b>\n"
        f"Логин: <code>{_tg_escape(final_username)}</code>\n"
        f"Email: {_tg_escape(email)}\n"
        f"Способ: пароль\n"
        f"План: Pro (пробный {WELCOME_TRIAL_DAYS} дн.)\n"
        f"IP: {_tg_escape(ip)}",
        purpose="регистрация по паролю",
    )
    resp = RedirectResponse(
        url=f"{BASE_PATH}/dashboard?user={final_username}&bn_signup=1&bn_method=email&bn_plan=pro",
        status_code=302,
    )
    return attach_session_cookie(resp, final_username)


@app.get("/logout")
async def logout(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        db.query(LoginSessionDB).filter(LoginSessionDB.token_hash == _session_digest(token)).delete()
        db.commit()
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

    ya_avatar = yandex_avatar_url(yandex_user.get("default_avatar_id"))

    user = db.query(UserDB).filter(UserDB.yandex_id == yandex_id).first()
    if not user and email:
        user = db.query(UserDB).filter(UserDB.email == email).first()
        if user:
            user.yandex_id = yandex_id
            if display_name:
                user.display_name = display_name
            if user.auth_provider != "yandex":
                user.auth_provider = "yandex" if not user.password else user.auth_provider
            if ya_avatar and not (user.avatar_url or "").strip():
                user.avatar_url = ya_avatar

    is_new_yandex_user = False
    if not user:
        if state_data.get("mode") == "login":
            return RedirectResponse(url=f"{BASE_PATH}/login?error=YandexAccountNotFound", status_code=302)
        is_new_yandex_user = True
        base_username = sanitize_username(login_name, f"ya_{yandex_id[:8]}")
        username = unique_username(db, base_username)
        user = UserDB(
            username=username,
            email=email,
            password="",
            yandex_id=yandex_id,
            display_name=display_name,
            avatar_url=ya_avatar,
            auth_provider="yandex",
            agree_marketing="true" if state_data.get("marketing") else "false",
            plan="free",
            plan_expires_at="",
        )
        grant_welcome_trial(user)
        db.add(user)
        db.commit()

        ref = ReferralLinkDB(user_id=username, code=str(uuid.uuid4())[:8])
        db.add(ref)
        db.commit()
        audit_log(
            db, username, "register", "user", username,
            details=f"yandex_oauth;plan_pro;welcome_trial_{WELCOME_TRIAL_DAYS}d",
            ip=request.client.host if request.client else "",
        )
        tg_notify(
            "🆕 <b>Регистрация</b>\n"
            f"Логин: <code>{_tg_escape(username)}</code>\n"
            f"Email: {_tg_escape(email) or '—'}\n"
            f"Имя: {_tg_escape(display_name) or '—'}\n"
            f"Способ: Яндекс\n"
            f"План: Pro (пробный {WELCOME_TRIAL_DAYS} дн.)\n"
            f"IP: {_tg_escape(request.client.host if request.client else '')}",
            purpose="регистрация через Яндекс",
        )

    if user.is_banned == "true":
        return RedirectResponse(url=f"{BASE_PATH}/login?error=UserBanned", status_code=302)

    user.last_login = datetime.datetime.now().isoformat()
    db.commit()
    audit_log(db, user.username, "login", "user", user.username, details="yandex_oauth", ip=request.client.host if request.client else "")
    if is_new_yandex_user:
        resp = RedirectResponse(
            url=f"{BASE_PATH}/dashboard?user={user.username}&bn_signup=1&bn_method=yandex_id&bn_plan=pro",
            status_code=302,
        )
        return attach_session_cookie(resp, user.username)
    return redirect_to_dashboard(user.username)

# ── Dashboard ──

@app.get("/ai-chat", response_class=HTMLResponse)
async def ai_chat_page(request: Request, user: str = "", db: Session = Depends(get_db)):
    session_user = get_session_username(request)
    username = session_user or (user or "").strip()
    if not username:
        return RedirectResponse(url=f"{BASE_PATH}/login?error=LoginRequired", status_code=302)
    if session_user and user and user != session_user:
        return RedirectResponse(url=f"{BASE_PATH}/ai-chat?user={session_user}", status_code=302)
    u = db.query(UserDB).filter(UserDB.username == username).first()
    plan = sync_user_plan_effective(u, db) if u else "free"
    return templates.TemplateResponse(
        request,
        "ai_chat.html",
        {
            "user": username,
            "active_page": "ai_chat",
            "user_plan": plan,
            "ai_enabled": plan_allows_ai(plan),
            "lab_planner_enabled": LAB_PLANNER_ENABLED,
        },
    )


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

    u = db.query(UserDB).filter(UserDB.username == user).first()
    plan = sync_user_plan_effective(u, db) if u else "free"
    show_onboarding = len(user_exps) == 0

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
            "show_onboarding": show_onboarding,
            "user_plan": plan,
            "ai_enabled": plan_allows_ai(plan),
        },
    )

# ── Projects page ──

def parse_amount(val):
    try:
        return float(val.replace(",", ".")) if val else 0.0
    except (ValueError, AttributeError):
        return 0.0

def _active_username(request: Request) -> str:
    username = get_session_username(request)
    if not username:
        raise HTTPException(401, "LoginRequired")
    return username


def _owned_experiment(db: Session, request: Request, exp_id: str) -> ExperimentDB:
    experiment = db.query(ExperimentDB).filter(
        ExperimentDB.id == exp_id, ExperimentDB.user_id == _active_username(request)
    ).first()
    if not experiment:
        raise HTTPException(404, "Эксперимент не найден.")
    return experiment


def _projects_for_user(db: Session, username: str):
    joined = db.query(ProjectMemberDB.project_id).filter(ProjectMemberDB.username == username)
    return db.query(ProjectDB).filter((ProjectDB.user_id == username) | ProjectDB.id.in_(joined))


def _require_project(db: Session, request: Request, project_id: str, *, owner_only: bool = False) -> ProjectDB:
    username = _active_username(request)
    projects = db.query(ProjectDB).filter(ProjectDB.user_id == username) if owner_only else _projects_for_user(db, username)
    project = projects.filter(ProjectDB.id == project_id).first()
    if not project:
        raise HTTPException(404, "Проект не найден.")
    return project


@app.get("/projects", response_class=HTMLResponse)
async def projects_page(request: Request, user: str, project: str = None, db: Session = Depends(get_db)):
    projects = _projects_for_user(db, user).all()
    all_users = []
    current_project = None
    project_tasks = []
    project_members = []
    gantt_tasks: list[dict] = []
    if project:
        current_project = _projects_for_user(db, user).filter(ProjectDB.id == project).first()
        if current_project:
            if current_project.user_id == user:
                all_users = db.query(UserDB.username).order_by(UserDB.username).all()
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
async def reagents_page(request: Request, user: str, tab: str = "", db: Session = Depends(get_db)):
    reagents = db.query(ReagentDB).filter(ReagentDB.user_id == user).all()
    consumables = db.query(ConsumableDB).filter(ConsumableDB.user_id == user).all()
    low_reagents = sum(1 for r in reagents if parse_amount(r.amount) <= parse_amount(r.min_amount))
    low_consumables = sum(1 for c in consumables if parse_amount(c.amount) <= parse_amount(c.min_amount))
    materials_tab = "consumables" if (tab or "").lower() == "consumables" else "reagents"
    return templates.TemplateResponse(
        request,
        "materials.html",
        {
            "user": user,
            "reagents": reagents,
            "consumables": consumables,
            "low_reagents": low_reagents,
            "low_consumables": low_consumables,
            "materials_tab": materials_tab,
            "active_page": "reagents",
        },
    )

# ── Consumables page (merged into /reagents tabs) ──

@app.get("/consumables", response_class=HTMLResponse)
async def consumables_page(request: Request, user: str):
    return RedirectResponse(
        url=f"{BASE_PATH}/reagents?user={user}&tab=consumables",
        status_code=302,
    )

# ── Experiments ──

@app.post("/create_experiment")
async def create_experiment(
    request: Request, user: str = Form(...), name: str = Form(...), description: str = Form(...),
    template: str = Form("lab"),
    db: Session = Depends(get_db)
):
    user = _active_username(request)
    exp_id = str(uuid.uuid4())[:8]
    user_dir = os.path.join(NOTEBOOKS_BASE, user)
    exp_dir = os.path.join(user_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    notebook_content = {
        "cells": [{"cell_type": "markdown", "id": "intro", "metadata": {},
                    "source": [f"# Лабораторная: {name}\n", f"**Описание:** {description}\n"]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5
    }
    with open(os.path.join(exp_dir, "experiment.ipynb"), "w") as f:
        json.dump(notebook_content, f)

    tpl = (template or "lab").strip().lower()
    levels = lab_report_template_levels() if tpl != "blank" else blank_lab_levels()
    new_exp = ExperimentDB(
        id=exp_id, user_id=user, name=name, description=description,
        levels=levels,
    )
    db.add(new_exp)
    # Seed starter note text for first stage
    first = levels[0][0] if levels and levels[0] else None
    if first:
        db.add(NodeNotesDB(
            id=str(uuid.uuid4())[:8],
            exp_id=exp_id,
            node_id=first["id"],
            notes_html=(
                f"<h2>{_html.escape(first.get('title') or '')}</h2>"
                f"<p>{_html.escape(first.get('content') or '')}</p>"
                "<p><em>Подсказка: добавьте фото кнопкой «Фото», распознайте текст кнопкой «OCR», "
                "экспортируйте отчёт в DOCX/PDF.</em></p>"
            ),
            chat_history=[],
        ))
    db.commit()
    audit_log(db, user, "create", "experiment", exp_id, json.dumps({"name": name, "template": tpl}))
    if first:
        return RedirectResponse(
            url=f"{BASE_PATH}/experiment/{user}/{exp_id}/node/{first['id']}",
            status_code=302,
        )
    return RedirectResponse(url=f"{BASE_PATH}/experiment/{user}/{exp_id}", status_code=302)

@app.get("/experiment/{user}/{exp_id}", response_class=HTMLResponse)
async def experiment_detail(request: Request, user: str, exp_id: str, viewing_user: str = None, db: Session = Depends(get_db)):
    exp = db.query(ExperimentDB).filter(ExperimentDB.id == exp_id, ExperimentDB.user_id == user).first()
    if not exp:
        return RedirectResponse(url=f"{BASE_PATH}/dashboard?user={user}")

    actor = _active_username(request)
    if actor != exp.user_id:
        if exp.is_public == "true":
            return RedirectResponse(url=f"{BASE_PATH}/share/{exp.id}", status_code=302)
        raise HTTPException(404, "Эксперимент не найден.")
    return templates.TemplateResponse(
        request, "experiment_detail.html",
        {
            "user": user,
            "exp": exp,
            "viewing_user": actor,
            "is_owner": True,
            "fr_status": "none",
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
                "Эксперименты и этапы",
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
            "desc": "Агент и все возможности Lab-Journal",
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


def _accepted_friend_usernames(db: Session, user: str) -> list[str]:
    friends = db.query(FriendDB).filter(
        (FriendDB.user_id == user) | (FriendDB.friend_id == user),
        FriendDB.status == "accepted",
    ).all()
    names = []
    for f in friends:
        names.append(f.friend_id if f.user_id == user else f.user_id)
    return sorted(set(names))


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, user: str, db: Session = Depends(get_db)):
    """Full-page chat with friends."""
    return templates.TemplateResponse(
        request, "chat.html",
        {
            "user": user,
            "start_chat": "",
            "active_page": "chat",
            "friend_usernames": _accepted_friend_usernames(db, user),
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
            "friend_usernames": _accepted_friend_usernames(db, user),
        },
    )


@app.get("/account", response_class=HTMLResponse)
async def account_settings(request: Request, db: Session = Depends(get_db)):
    session_user = get_session_username(request)
    if not session_user:
        return RedirectResponse(url=f"{BASE_PATH}/login?error=LoginRequired", status_code=302)
    u = db.query(UserDB).filter(UserDB.username == session_user).first()
    if not u:
        return RedirectResponse(url=f"{BASE_PATH}/login?error=UserNotFound", status_code=302)
    plan = sync_user_plan_effective(u, db)
    used = _user_storage_bytes(session_user, db)
    limit = storage_limit_bytes(plan)
    return templates.TemplateResponse(
        request,
        "account.html",
        {
            "user": session_user,
            "profile": u,
            "active_page": "account",
            "user_plan": plan,
            "plan_label": PLAN_LABELS.get(plan, "Free"),
            "plan_expires_at": (u.plan_expires_at or "").strip(),
            "storage_used": used,
            "storage_limit": limit,
            "storage_used_label": format_bytes_ru(used),
            "storage_limit_label": format_bytes_ru(limit),
            "storage_pct": min(100, int(round(100 * used / limit))) if limit else 0,
            "can_change_password": bool((u.password or "").strip()) and (u.auth_provider or "password") != "yandex",
            "msg": request.query_params.get("msg", ""),
            "error": request.query_params.get("error", ""),
        },
    )


@app.post("/account/password")
async def account_change_password(
    request: Request,
    current_password: str = Form(""),
    new_password: str = Form(""),
    new_password2: str = Form(""),
    db: Session = Depends(get_db),
):
    session_user = get_session_username(request)
    if not session_user:
        return RedirectResponse(url=f"{BASE_PATH}/login?error=LoginRequired", status_code=302)
    u = db.query(UserDB).filter(UserDB.username == session_user).first()
    if not u or not (u.password or "").strip():
        return RedirectResponse(url=f"{BASE_PATH}/account?error=NoPassword", status_code=302)
    if len(new_password or "") < 6:
        return RedirectResponse(url=f"{BASE_PATH}/account?error=WeakPassword", status_code=302)
    if new_password != new_password2:
        return RedirectResponse(url=f"{BASE_PATH}/account?error=PasswordMismatch", status_code=302)
    if not verify_password(u.password, current_password or ""):
        return RedirectResponse(url=f"{BASE_PATH}/account?error=WrongPassword", status_code=302)
    u.password = hash_password(new_password)
    db.query(LoginSessionDB).filter(LoginSessionDB.user_id == session_user).delete()
    db.commit()
    audit_log(db, session_user, "password_change", "user", session_user, details="")
    response = RedirectResponse(url=f"{BASE_PATH}/account?msg=PasswordChanged", status_code=302)
    return attach_session_cookie(response, session_user)


@app.post("/api/profile/avatar")
async def upload_profile_avatar(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    session_user = get_session_username(request)
    if not session_user:
        return JSONResponse({"ok": False, "error": "LoginRequired"}, status_code=401)
    u = db.query(UserDB).filter(UserDB.username == session_user).first()
    if not u:
        return JSONResponse({"ok": False, "error": "UserNotFound"}, status_code=404)

    raw_name = (file.filename or "photo.jpg").strip()
    ext = raw_name.rsplit(".", 1)[-1].lower() if "." in raw_name else ""
    ctype = (file.content_type or "").lower()
    if ext not in AVATAR_ALLOWED_EXT and ctype not in (
        "image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif",
    ):
        return JSONResponse(
            {"ok": False, "error": "Нужен файл JPG, PNG, WebP или GIF"},
            status_code=400,
        )
    if ext not in AVATAR_ALLOWED_EXT:
        ext = {"image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png",
               "image/webp": "webp", "image/gif": "gif"}.get(ctype, "jpg")
    if ext == "jpeg":
        ext = "jpg"

    data = await file.read(AVATAR_MAX_BYTES + 1)
    if not data:
        return JSONResponse({"ok": False, "error": "Пустой файл"}, status_code=400)
    if len(data) > AVATAR_MAX_BYTES:
        return JSONResponse({"ok": False, "error": "Файл больше 5 МБ"}, status_code=400)

    # light magic-byte check
    head = data[:16]
    ok_magic = (
        head.startswith(b"\xff\xd8\xff")
        or head.startswith(b"\x89PNG\r\n\x1a\n")
        or head.startswith(b"GIF87a")
        or head.startswith(b"GIF89a")
        or (len(head) >= 12 and head[0:4] == b"RIFF" and head[8:12] == b"WEBP")
    )
    if not ok_magic:
        return JSONResponse({"ok": False, "error": "Файл не похож на изображение"}, status_code=400)

    safe_user = re.sub(r"[^a-zA-Z0-9_-]", "_", session_user)[:40] or "user"
    fname = f"avatar_{safe_user}_{uuid.uuid4().hex[:10]}.{ext}"
    path = os.path.join(UPLOADS_DIR, fname)
    with open(path, "wb") as f:
        f.write(data)

    old = (u.avatar_url or "").strip()
    u.avatar_url = f"/static/uploads/{fname}"
    db.commit()
    _delete_local_avatar(old)
    audit_log(db, session_user, "avatar_upload", "user", session_user, details=fname)
    return JSONResponse({"ok": True, "url": avatar_src(u.avatar_url)})


@app.delete("/api/profile/avatar")
async def delete_profile_avatar(request: Request, db: Session = Depends(get_db)):
    session_user = get_session_username(request)
    if not session_user:
        return JSONResponse({"ok": False, "error": "LoginRequired"}, status_code=401)
    u = db.query(UserDB).filter(UserDB.username == session_user).first()
    if not u:
        return JSONResponse({"ok": False, "error": "UserNotFound"}, status_code=404)
    old = (u.avatar_url or "").strip()
    if not old:
        return JSONResponse({"ok": True, "url": ""})
    u.avatar_url = ""
    db.commit()
    _delete_local_avatar(old)
    audit_log(db, session_user, "avatar_delete", "user", session_user, details="")
    return JSONResponse({"ok": True, "url": ""})


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


def _can_chat(db: Session, user: str, other: str) -> bool:
    return db.query(FriendDB.id).filter(
        _friend_pair_filter(user, other), FriendDB.status == "accepted"
    ).first() is not None


def _friend_pair_filter(user: str, other: str):
    return ((FriendDB.user_id == user) & (FriendDB.friend_id == other)) | (
        (FriendDB.user_id == other) & (FriendDB.friend_id == user)
    )


@app.post("/api/friends/chats/create")
async def create_friend_chat(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    user = get_session_username(request)
    other = body.get("other", "")
    if not user or not other:
        return JSONResponse({"error": "LoginRequired"}, status_code=401)
    if user == other:
        return JSONResponse({"error": "Cannot chat with yourself"}, status_code=400)
    if not _can_chat(db, user, other):
        return JSONResponse({"error": "Чат доступен только друзьям."}, status_code=403)
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
    other = chat.user2 if chat.user1 == user else chat.user1
    if not _can_chat(db, user, other):
        return JSONResponse({"error": "Отправка доступна только друзьям."}, status_code=403)
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
        owner_id=_active_username(request),
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
        EquipmentDB.owner_id == _active_username(request),
    ).first()
    if not equip:
        return JSONResponse({"error": "Оборудование не найдено"}, status_code=404)
    db.query(EquipmentBookingDB).filter_by(equipment_id=equip.id).delete()
    db.delete(equip)
    db.commit()
    return JSONResponse({"ok": True})

def _date_range(start: str, end: str, *, required: bool = False) -> tuple[str, str]:
    def normalized(value: str) -> str:
        if not value:
            return ""
        try:
            if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError
            return datetime.date.fromisoformat(value).isoformat()
        except ValueError:
            raise HTTPException(422, "Укажите корректную дату.") from None

    start, end = normalized(start), normalized(end)
    if required and (not start or not end):
        raise HTTPException(422, "Укажите даты начала и окончания.")
    if start and end and start > end:
        raise HTTPException(422, "Дата окончания не может быть раньше начала.")
    return start, end


def _equipment_for_user(db: Session, request: Request, equipment_id: str) -> EquipmentDB:
    equipment = db.get(EquipmentDB, equipment_id)
    user = _active_username(request)
    if not equipment or (equipment.owner_id != user and not _can_chat(db, user, equipment.owner_id)):
        raise HTTPException(404, "Оборудование не найдено.")
    return equipment


@app.post("/api/equipment/book")
async def book_equipment(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    equipment = _equipment_for_user(db, request, body.get("equipment_id"))
    start, end = _date_range(body.get("start_date"), body.get("end_date"), required=True)
    overlap = db.query(EquipmentBookingDB).filter(
        EquipmentBookingDB.equipment_id == equipment.id,
        EquipmentBookingDB.status.in_(["pending", "approved"]),
        EquipmentBookingDB.start_date <= end,
        EquipmentBookingDB.end_date >= start,
    ).first()
    booking = EquipmentBookingDB(equipment_id=equipment.id, user_id=_active_username(request),
                                 start_date=start, end_date=end, status="pending")
    db.add(booking)
    db.commit()
    return JSONResponse({"ok": True, "id": booking.id,
                         "warning": "Обнаружено пересечение бронирований" if overlap else None})


def _owner_booking(db: Session, request: Request, booking_id: str) -> EquipmentBookingDB:
    booking = db.query(EquipmentBookingDB).join(EquipmentDB, EquipmentDB.id == EquipmentBookingDB.equipment_id).filter(
        EquipmentBookingDB.id == booking_id, EquipmentDB.owner_id == _active_username(request)
    ).first()
    if not booking:
        raise HTTPException(404, "Бронирование не найдено.")
    return booking


@app.post("/api/equipment/booking/approve")
async def approve_booking(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    # Lock writes before checking overlap so simultaneous approvals serialize in SQLite.
    db.connection().exec_driver_sql("BEGIN IMMEDIATE")
    booking = _owner_booking(db, request, body.get("id"))
    if booking.status == "approved":
        db.commit()
        return JSONResponse({"ok": True})
    if booking.status != "pending":
        raise HTTPException(409, "Заявка больше не ожидает подтверждения.")
    start, end = _date_range(booking.start_date, booking.end_date, required=True)
    conflict = db.query(EquipmentBookingDB.id).filter(
        EquipmentBookingDB.equipment_id == booking.equipment_id,
        EquipmentBookingDB.status == "approved", EquipmentBookingDB.id != booking.id,
        EquipmentBookingDB.start_date <= end, EquipmentBookingDB.end_date >= start,
    ).first()
    if conflict:
        return JSONResponse({"error": "На это время уже подтверждена другая бронь."}, status_code=409)
    booking.status = "approved"
    db.commit()
    return JSONResponse({"ok": True})


@app.post("/api/equipment/booking/decline")
async def decline_booking(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    booking = _owner_booking(db, request, body.get("id"))
    if booking.status not in ('pending', 'approved'):
        raise HTTPException(409, "Текущий статус заявки не допускает это действие.")
    booking.status = "declined"
    db.commit()
    return JSONResponse({"ok": True})

@app.post("/api/equipment/booking/complete")
async def complete_booking(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    booking = _owner_booking(db, request, body.get("id"))
    if booking.status not in ('approved',):
        raise HTTPException(409, "Текущий статус заявки не допускает это действие.")
    booking.status = "completed"
    db.commit()
    return JSONResponse({"ok": True})

@app.post("/api/equipment/booking/cancel")
async def cancel_booking(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    booking = db.query(EquipmentBookingDB).filter(
        EquipmentBookingDB.id == body.get("id"),
        EquipmentBookingDB.user_id == _active_username(request),
        EquipmentBookingDB.status.in_(("pending", "approved")),
    ).first()
    if not booking:
        return JSONResponse({"error": "Бронирование не найдено"}, status_code=404)
    booking.status = "declined"
    db.commit()
    return JSONResponse({"ok": True})

@app.get("/api/equipment/bookings")
async def get_equipment_bookings(id: str, request: Request, db: Session = Depends(get_db)):
    _equipment_for_user(db, request, id)
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
    request: Request, user: str, exp_id: str,
    title: str = Form(...), content: str = Form(...), level_idx: int = Form(-1),
    db: Session = Depends(get_db)
):
    _owned_experiment(db, request, exp_id)
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
    _owned_experiment(db, request, exp_id)
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
        {"user": user, "exp": exp, "node": node, "notes": notes, "is_public_view": False,
         "pptx_export_enabled": plan_allows_ai(getattr(request.state, "user_plan", "free"))}
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp

# ── API: save notes ──

def _version_author_label(request: Request, db: Session) -> str:
    username = get_session_username(request) or ""
    if not username:
        return ""
    user = db.query(UserDB).filter(UserDB.username == username).first()
    display = ((user.display_name if user else "") or "").strip()
    return display or username


def _maybe_create_node_version(
    db: Session,
    *,
    exp_id: str,
    node_id: str,
    html: str,
    author: str = "",
    message: str = "Автосохранение",
    force: bool = False,
) -> NodeVersionDB | None:
    """Persist a snapshot when content changed (autosave is already debounced in UI)."""
    latest = (
        db.query(NodeVersionDB)
        .filter(NodeVersionDB.exp_id == exp_id, NodeVersionDB.node_id == node_id)
        .order_by(NodeVersionDB.created_at.desc())
        .first()
    )
    if latest and (latest.html_snapshot or "") == (html or "") and not force:
        return None
    if latest and (latest.html_snapshot or "") == (html or "") and force:
        return None
    version = NodeVersionDB(
        exp_id=exp_id,
        node_id=node_id,
        html_snapshot=html or "",
        message=message,
        author=author or "",
    )
    db.add(version)
    return version


def _sanitize_note_attribute(tag: str, attribute: str, value: str) -> str | None:
    if attribute in {"id", "data-cell-id"}:
        return value if re.fullmatch(r"cell_[a-zA-Z0-9_-]{1,40}", value) else None
    if tag == "a" and attribute == "href" and value.lower().lstrip().startswith("data:"):
        return None
    if tag == "img" and attribute == "src":
        if value.lower().startswith("data:"):
            return value if re.match(r"^data:image/(png|jpeg|gif|webp);base64,", value, re.I) else None
        parsed = urlparse(value)
        if parsed.scheme:
            return value if parsed.scheme.lower() in {"https", "http"} else None
        return value if not value.startswith(("//", "\\")) else None
    return value


def sanitize_note_html(value: str) -> str:
    return nh3.clean(
        normalize_upload_urls(value),
        tags={"a", "b", "blockquote", "br", "button", "code", "div", "em", "h1", "h2", "h3",
              "h4", "h5", "h6", "hr", "i", "img", "li", "ol", "p", "pre", "s", "section",
              "small", "span", "strike", "strong", "sub", "sup", "table", "tbody", "td",
              "textarea", "tfoot", "th", "thead", "tr", "u", "ul"},
        attributes={"*": {"class", "style", "title", "contenteditable"},
                    "a": {"href"}, "img": {"src", "alt", "width", "height"},
                    "div": {"id"}, "button": {"type", "data-cell-id"},
                    "i": {"data-lucide"}, "textarea": {"placeholder", "spellcheck", "rows", "cols"},
                    "td": {"colspan", "rowspan"}, "th": {"colspan", "rowspan", "scope"}},
        clean_content_tags={"script", "style", "iframe", "object", "embed", "svg", "math"},
        url_schemes={"http", "https", "mailto", "data"},
        filter_style_properties={"color", "background-color", "font-size", "font-family",
                                 "font-weight", "text-align", "text-decoration", "width",
                                 "height", "max-width", "transform", "white-space"},
        attribute_filter=_sanitize_note_attribute,
    )


def _store_note(db: Session, request: Request, exp_id: str, node_id: str, html: str,
                revision: int, *, annotation_url: str | None = None, message: str = "Автосохранение"):
    if type(revision) is not int or revision < 0:
        raise HTTPException(409, "Обновите страницу перед сохранением заметки.")
    html = sanitize_note_html(html)
    fields = {NodeNotesDB.notes_html: html, NodeNotesDB.revision: revision + 1}
    if annotation_url is not None:
        fields[NodeNotesDB.annotation_url] = _sanitize_note_attribute("img", "src", annotation_url) or ""
    query = db.query(NodeNotesDB).filter_by(exp_id=exp_id, node_id=node_id)
    if query.filter(NodeNotesDB.revision == revision).update(fields, synchronize_session=False) != 1:
        db.rollback()
        raise HTTPException(409, "Заметка изменилась. Скопируйте свою правку и обновите страницу.")
    version = _maybe_create_node_version(db, exp_id=exp_id, node_id=node_id, html=html,
                                         author=_version_author_label(request, db), message=message)
    db.commit()
    return {"ok": True, "revision": revision + 1, "version_id": version.id if version else None}


@app.post("/api/node/save")
async def save_node_notes(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    exp_id, node_id, html = body.get("exp_id"), body.get("node_id"), body.get("html")
    _owned_experiment(db, request, exp_id)
    if not isinstance(html, str):
        raise HTTPException(422, "Текст заметки должен быть строкой.")
    annotation = body.get("annotation_url")
    return _store_note(db, request, exp_id, node_id, html, body.get("revision"),
                       annotation_url=annotation if isinstance(annotation, str) else None)


# ── Export helpers (DOCX / PDF / PPTX) ──

def _export_meta(db: Session, exp_id: str, node_id: str, version_id: str | None = None) -> dict:
    exp = db.query(ExperimentDB).filter(ExperimentDB.id == exp_id).first()
    node = _find_node(exp, node_id) if exp else None
    title = ((node or {}).get("title") if isinstance(node, dict) else None) or f"Узел {node_id}"
    author = ""
    if exp and exp.user_id:
        user = db.query(UserDB).filter(UserDB.username == exp.user_id).first()
        author = ((user.display_name if user else "") or "").strip() or exp.user_id
    notes = db.query(NodeNotesDB).filter(NodeNotesDB.exp_id == exp_id, NodeNotesDB.node_id == node_id).first()
    if not node:
        raise HTTPException(404, "Этап не найден.")
    html = notes.notes_html if notes else ""
    if version_id:
        version = db.query(NodeVersionDB).filter_by(id=version_id, exp_id=exp_id, node_id=node_id).first()
        if not version:
            raise HTTPException(404, "Версия не найдена.")
        html = version.html_snapshot
    return {
        "title": title,
        "author": author,
        "experiment": (exp.name if exp else "") or "",
        "html": sanitize_note_html(html or ""),
        "media": {name for (name,) in db.query(MediaUploadDB.filename).filter_by(exp_id=exp_id)},
        "date": datetime.datetime.now().strftime("%d.%m.%Y %H:%M"),
    }


def _resolve_export_image_path(src: str, allowed_media: set[str]):
    if not src:
        return None
    src = _html.unescape(src.strip())
    if re.match(r"^data:image/(png|jpeg|gif|webp);base64,", src, re.I):
        try:
            raw = base64.b64decode(src.split(",", 1)[1], validate=True)
            from PIL import Image as PILImage
            with PILImage.open(io.BytesIO(raw)) as image:
                if image.format not in _NOTE_IMAGE_TYPES or image.width * image.height > 25_000_000:
                    return None
                image.verify()
            return io.BytesIO(raw)
        except (ValueError, OSError):
            return None
    matches = _note_media_names(urlparse(src).path)
    if len(matches) != 1:
        return None
    name = next(iter(matches))
    return _media_path(name) if name in allowed_media else None


def _parse_html_export_blocks(html: str, allowed_media: set[str] | None = None) -> list[dict]:
    """Turn notebook HTML into structured blocks for exporters."""
    from html.parser import HTMLParser
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    tables = []
    for table in soup.find_all("table"):
        if table.find_parent("table"):
            continue
        rows = []
        for row in table.find_all("tr"):
            if row.find_parent("table") is not table:
                continue
            cells = row.find_all(("th", "td"), recursive=False)
            if cells:
                rows.append([cell.get_text(" ", strip=True) for cell in cells])
        token = soup.new_tag("bb-export-table")
        token["data-index"] = str(len(tables))
        tables.append(rows)
        table.replace_with(token)

    class _Parser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.blocks: list[dict] = []
            self._buf: list[str] = []
            self._skip = 0
            self._heading: int | None = None
            self._in_li = False
            self._in_pre = False
            self._list_items: list[str] = []

        def _flush_text(self, force_para: bool = False):
            text = "".join(self._buf)
            self._buf = []
            if self._heading is not None:
                cleaned = re.sub(r"\s+", " ", text).strip()
                if cleaned:
                    self.blocks.append({"type": f"h{self._heading}", "text": cleaned})
                self._heading = None
                return
            if self._in_li:
                return
            if self._in_pre:
                return
            cleaned = re.sub(r"[ \t]+\n", "\n", text)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            if cleaned or force_para:
                if cleaned:
                    self.blocks.append({"type": "p", "text": cleaned})

        def handle_starttag(self, tag, attrs):
            tag = tag.lower()
            attrs_d = {k.lower(): (v or "") for k, v in attrs}
            if tag == "bb-export-table" and not self._skip:
                index = attrs_d.get("data-index", "")
                if index.isdecimal() and int(index) < len(tables):
                    self._flush_text()
                    rows = tables[int(index)]
                    if rows:
                        width = max(map(len, rows))
                        self.blocks.append({"type": "table", "rows": [row + [""] * (width - len(row)) for row in rows]})
                return
            if tag in ("script", "style", "button", "svg"):
                self._skip += 1
                return
            if self._skip:
                return
            if tag in ("h1", "h2", "h3", "h4"):
                self._flush_text()
                self._heading = int(tag[1])
                return
            if tag == "img":
                self._flush_text()
                src = attrs_d.get("src", "")
                alt = attrs_d.get("alt", "")
                path = _resolve_export_image_path(src, allowed_media or set())
                if path:
                    self.blocks.append({"type": "img", "path": path, "alt": alt})
                elif alt:
                    self.blocks.append({"type": "p", "text": f"[Изображение: {alt}]"})
                return
            if tag == "br":
                self._buf.append("\n")
                return
            if tag in ("p", "div", "section", "article", "tr"):
                self._flush_text()
                return
            if tag in ("ul", "ol"):
                self._flush_text()
                self._list_items = []
                return
            if tag == "li":
                self._flush_text()
                self._in_li = True
                self._buf = []
                return
            if tag in ("pre", "code") and not self._in_pre:
                # only treat standalone pre/code-cell bodies as code blocks
                if tag == "pre" or attrs_d.get("class", "").find("code") >= 0:
                    self._flush_text()
                    self._in_pre = True
                    self._buf = []
                return
            if tag == "hr":
                self._flush_text()
                self.blocks.append({"type": "p", "text": "— — —"})

        def handle_endtag(self, tag):
            tag = tag.lower()
            if tag in ("script", "style", "button", "svg"):
                self._skip = max(0, self._skip - 1)
                return
            if self._skip:
                return
            if tag in ("h1", "h2", "h3", "h4"):
                self._flush_text()
                return
            if tag in ("p", "div", "section", "article", "tr"):
                self._flush_text(force_para=False)
                return
            if tag == "li":
                text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
                self._buf = []
                self._in_li = False
                if text:
                    self._list_items.append(text)
                return
            if tag in ("ul", "ol"):
                for item in self._list_items:
                    self.blocks.append({"type": "li", "text": item})
                self._list_items = []
                return
            if tag in ("pre", "code") and self._in_pre:
                code = "".join(self._buf).strip("\n")
                self._buf = []
                self._in_pre = False
                if code:
                    self.blocks.append({"type": "code", "text": code})
                return

        def handle_data(self, data):
            if self._skip:
                return
            self._buf.append(data)

        def close(self):
            self._flush_text()
            super().close()

    parser = _Parser()
    try:
        parser.feed(str(soup))
        parser.close()
    except Exception:
        text = re.sub(r"(?i)<br\s*/?>", "\n", html or "")
        text = re.sub(r"(?i)</p\s*>", "\n\n", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = _html.unescape(text).strip()
        return [{"type": "p", "text": text}] if text else []
    return parser.blocks


def _build_docx_bytes(meta: dict) -> bytes:
    from docx import Document  # type: ignore
    from docx.shared import Cm, Pt, RGBColor  # type: ignore
    from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore

    doc = Document()
    title = doc.add_heading(meta["title"], level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    meta_line = " · ".join(
        part for part in (
            f"Автор: {meta['author']}" if meta.get("author") else "",
            f"Дата: {meta.get('date') or ''}",
            f"Эксперимент: {meta['experiment']}" if meta.get("experiment") else "",
        ) if part
    )
    if meta_line:
        p = doc.add_paragraph(meta_line)
        for run in p.runs:
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    blocks = _parse_html_export_blocks(meta.get("html") or "", meta.get("media"))
    if not blocks:
        doc.add_paragraph("(пусто)")
    for block in blocks:
        kind = block.get("type")
        if kind in ("h1", "h2", "h3", "h4"):
            level = int(kind[1])
            doc.add_heading(block.get("text") or "", level=min(level, 3))
        elif kind == "li":
            doc.add_paragraph(block.get("text") or "", style="List Bullet")
        elif kind == "code":
            p = doc.add_paragraph()
            run = p.add_run(block.get("text") or "")
            run.font.name = "Courier New"
            run.font.size = Pt(9)
        elif kind == "img":
            path = block.get("path")
            try:
                doc.add_picture(path, width=Cm(14))
                alt = (block.get("alt") or "").strip()
                if alt:
                    cap = doc.add_paragraph(alt)
                    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in cap.runs:
                        run.font.size = Pt(9)
                        run.font.italic = True
            except Exception:
                doc.add_paragraph(f"[Не удалось вставить изображение: {(os.path.basename(path) if isinstance(path, str) else 'встроенное фото')}]")
        elif kind == "table":
            rows = block["rows"]
            table = doc.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Table Grid"
            for tr, values in zip(table.rows, rows):
                for cell, value in zip(tr.cells, values):
                    cell.text = value
        else:
            text = block.get("text") or ""
            for part in text.split("\n\n"):
                part = part.strip()
                if part:
                    doc.add_paragraph(part)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _build_pdf_bytes(meta: dict) -> bytes:
    from reportlab.lib.pagesizes import A4  # type: ignore
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore
    from reportlab.lib.units import cm  # type: ignore
    from reportlab.platypus import (  # type: ignore
        SimpleDocTemplate, Paragraph, Spacer, Image, Preformatted, ListFlowable, ListItem, Table, TableStyle,
    )
    from reportlab.pdfbase import pdfmetrics  # type: ignore
    from reportlab.pdfbase.ttfonts import TTFont  # type: ignore
    from reportlab.lib.enums import TA_LEFT  # type: ignore
    from reportlab.lib import colors

    win_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    configured_font = os.environ.get("PDF_EXPORT_FONT", "")
    candidates = (
        (configured_font, os.environ.get("PDF_EXPORT_BOLD_FONT", "")),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        (os.path.join(win_fonts, "arial.ttf"), os.path.join(win_fonts, "arialbd.ttf")),
        ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    )
    for regular, bold in candidates:
        if regular and os.path.isfile(regular):
            pdfmetrics.registerFont(TTFont("ExportSans", regular))
            font_name, font_bold = "ExportSans", "ExportSans"
            if bold and os.path.isfile(bold):
                pdfmetrics.registerFont(TTFont("ExportSans-Bold", bold))
                font_bold = "ExportSans-Bold"
            break
    else:
        raise RuntimeError("Не установлен TrueType-шрифт для экспорта PDF.")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ExportTitle", fontName=font_bold, fontSize=18, leading=22, spaceAfter=8, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="ExportMeta", fontName=font_name, fontSize=9, leading=12, textColor="#555555", spaceAfter=14))
    styles.add(ParagraphStyle(name="ExportH1", fontName=font_bold, fontSize=14, leading=18, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="ExportH2", fontName=font_bold, fontSize=12, leading=16, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="ExportBody", fontName=font_name, fontSize=10, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="ExportCode", fontName="Courier", fontSize=8, leading=10, spaceAfter=8))

    def esc(s: str) -> str:
        return _html.escape(s or "").replace("\n", "<br/>")

    story = [Paragraph(esc(meta["title"]), styles["ExportTitle"])]
    meta_line = " · ".join(
        part for part in (
            f"Автор: {meta['author']}" if meta.get("author") else "",
            f"Дата: {meta.get('date') or ''}",
            f"Эксперимент: {meta['experiment']}" if meta.get("experiment") else "",
        ) if part
    )
    if meta_line:
        story.append(Paragraph(esc(meta_line), styles["ExportMeta"]))

    blocks = _parse_html_export_blocks(meta.get("html") or "", meta.get("media"))
    if not blocks:
        story.append(Paragraph("(пусто)", styles["ExportBody"]))
    for block in blocks:
        kind = block.get("type")
        if kind == "h1":
            story.append(Paragraph(esc(block.get("text")), styles["ExportH1"]))
        elif kind in ("h2", "h3", "h4"):
            story.append(Paragraph(esc(block.get("text")), styles["ExportH2"]))
        elif kind == "li":
            story.append(Paragraph("• " + esc(block.get("text")), styles["ExportBody"]))
        elif kind == "code":
            story.append(Preformatted(block.get("text") or "", styles["ExportCode"]))
        elif kind == "img":
            path = block.get("path")
            try:
                from reportlab.lib.utils import ImageReader  # type: ignore
                ir = ImageReader(path)
                iw, ih = float(ir.getSize()[0] or 1), float(ir.getSize()[1] or 1)
                # Fit inside page frame (A4 − margins). Tall phone photos
                # previously blew past frame height → "Flowable ... too large".
                max_w = float(doc.width)
                max_h = float(doc.height) * 0.92
                scale = min(max_w / iw, max_h / ih, 1.0)
                img = Image(path, width=iw * scale, height=ih * scale)
                img.hAlign = "LEFT"
                story.append(img)
                story.append(Spacer(1, 6))
            except Exception:
                story.append(Paragraph(esc(f"[Изображение: {(os.path.basename(path) if isinstance(path, str) else 'встроенное фото')}]"), styles["ExportBody"]))
        elif kind == "table":
            rows = [[Paragraph(esc(value), styles["ExportBody"]) for value in row] for row in block["rows"]]
            table = Table(rows, colWidths=[doc.width / len(rows[0])] * len(rows[0]), splitInRow=1)
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.extend((table, Spacer(1, 6)))
        else:
            text = (block.get("text") or "").strip()
            if text:
                story.append(Paragraph(esc(text).replace("\n", "<br/>"), styles["ExportBody"]))

    doc.build(story)
    return buf.getvalue()


def _build_pptx_bytes(meta: dict) -> bytes:
    from pptx import Presentation  # type: ignore
    from pptx.util import Inches, Pt  # type: ignore

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Title slide
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = meta.get("title") or "Lab-Journal"
    subtitle = " · ".join(
        part for part in (
            meta.get("author") or "",
            meta.get("date") or "",
            meta.get("experiment") or "",
        ) if part
    )
    if slide.placeholders and len(slide.placeholders) > 1:
        slide.placeholders[1].text = subtitle or "Экспорт из Lab-Journal"

    blocks = _parse_html_export_blocks(meta.get("html") or "", meta.get("media"))
    # Group into slides by headings; pack paragraphs
    slides_spec: list[dict] = []
    current = {"title": "Содержание", "items": []}
    for block in blocks:
        kind = block.get("type")
        if kind in ("h1", "h2"):
            if current["items"] or (current["title"] != "Содержание" and not slides_spec):
                slides_spec.append(current)
            current = {"title": block.get("text") or "Раздел", "items": []}
        elif kind in ("img", "table"):
            current["items"].append(block)
        else:
            text = (block.get("text") or "").strip()
            if text:
                prefix = "• " if kind == "li" else ""
                current["items"].append({"type": "text", "text": prefix + text})
    if current["items"] or not slides_spec:
        slides_spec.append(current)

    for spec in slides_spec:
        def new_slide():
            page = prs.slides.add_slide(prs.slide_layouts[6])
            title = page.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.8))
            title.text_frame.text = spec["title"]
            title.text_frame.paragraphs[0].font.size = Pt(28)
            title.text_frame.paragraphs[0].font.bold = True
            return page

        slide = new_slide()
        y = 1.2
        for item in spec["items"]:
            if y > 6.5:
                slide, y = new_slide(), 1.2
            if item.get("type") == "table":
                rows = item["rows"]
                for start in range(0, len(rows), 10):
                    chunk = rows[start:start + 10]
                    if y + len(chunk) * 0.4 > 6.8:
                        slide, y = new_slide(), 1.2
                    shape = slide.shapes.add_table(len(chunk), len(chunk[0]), Inches(0.6), Inches(y), Inches(12), Inches(len(chunk) * 0.4))
                    for i, row in enumerate(chunk):
                        for j, value in enumerate(row):
                            cell = shape.table.cell(i, j)
                            cell.text = value
                            cell.text_frame.paragraphs[0].font.size = Pt(13)
                    y += len(chunk) * 0.4 + 0.2
            elif item.get("type") == "img":
                if y + 3.2 > 6.8:
                    slide, y = new_slide(), 1.2
                path = item.get("path")
                try:
                    slide.shapes.add_picture(path, Inches(0.8), Inches(y), width=Inches(8))
                    y += 3.2
                except Exception:
                    box = slide.shapes.add_textbox(Inches(0.6), Inches(y), Inches(12), Inches(0.4))
                    box.text_frame.text = f"[Изображение: {(os.path.basename(path) if isinstance(path, str) else 'встроенное фото')}]"
                    y += 0.5
            else:
                box = slide.shapes.add_textbox(Inches(0.6), Inches(y), Inches(12), Inches(0.9))
                p = box.text_frame.paragraphs[0]
                p.text = item.get("text") or ""
                p.font.size = Pt(16)
                y += 0.55

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _export_filename(prefix: str, node_id: str, ext: str) -> str:
    safe = re.sub(r"[^\w.-]+", "_", str(node_id))[:40]
    return f"{prefix}_{safe}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"


@app.get("/api/node/export_docx")
async def export_node_docx(exp_id: str, node_id: str, request: Request, db: Session = Depends(get_db), version_id: str | None = None):
    _owned_experiment(db, request, exp_id)
    meta = _export_meta(db, exp_id, node_id, version_id)
    try:
        data = _build_docx_bytes(meta)
    except ImportError:
        return JSONResponse({"ok": False, "error": "DOCX export unavailable (python-docx not installed)."}, status_code=501)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"DOCX export failed: {e}"}, status_code=500)

    buf = io.BytesIO(data)
    fname = _export_filename("node", node_id, "docx")
    headers = {"Content-Disposition": f'attachment; filename="{fname}"', "Cache-Control": "private, no-store"}
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers,
    )


@app.get("/api/node/export_pdf")
async def export_node_pdf(exp_id: str, node_id: str, request: Request, db: Session = Depends(get_db), version_id: str | None = None):
    _owned_experiment(db, request, exp_id)
    meta = _export_meta(db, exp_id, node_id, version_id)
    try:
        data = _build_pdf_bytes(meta)
    except ImportError:
        return JSONResponse({"ok": False, "error": "PDF export unavailable (reportlab not installed)."}, status_code=501)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"PDF export failed: {e}"}, status_code=500)

    buf = io.BytesIO(data)
    fname = _export_filename("node", node_id, "pdf")
    headers = {"Content-Disposition": f'attachment; filename="{fname}"', "Cache-Control": "private, no-store"}
    return StreamingResponse(buf, media_type="application/pdf", headers=headers)


@app.get("/api/node/export_pptx")
async def export_node_pptx(request: Request, exp_id: str, node_id: str, db: Session = Depends(get_db), version_id: str | None = None):
    session_user = get_session_username(request)
    u = db.query(UserDB).filter(UserDB.username == session_user).first() if session_user else None
    plan = sync_user_plan_effective(u, db) if u else "free"
    if not plan_allows_ai(plan):
        return JSONResponse(
            {
                "ok": False,
                "error": "PPTX — в тарифе Pro. На Free доступны DOCX и PDF.",
                "upgrade": True,
            },
            status_code=403,
        )
    _owned_experiment(db, request, exp_id)
    meta = _export_meta(db, exp_id, node_id, version_id)
    try:
        data = _build_pptx_bytes(meta)
    except ImportError:
        return JSONResponse({"ok": False, "error": "PPTX export unavailable (python-pptx not installed)."}, status_code=501)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"PPTX export failed: {e}"}, status_code=500)

    buf = io.BytesIO(data)
    fname = _export_filename("node", node_id, "pptx")
    headers = {"Content-Disposition": f'attachment; filename="{fname}"', "Cache-Control": "private, no-store"}
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers=headers,
    )


def _tpl_field(key: str, label: str) -> str:
    return (
        f'<span class="tpl-field" data-field="{key}" contenteditable="false">{label}</span>'
    )


_DEFAULT_EXPORT_TEMPLATE_HTML = f"""
<p style="text-align:center;margin:0 0 0.35em;font-size:13pt;line-height:1.35;">Министерство науки и высшего образования РФ</p>
<p style="text-align:center;margin:0 0 0.35em;font-size:13pt;line-height:1.35;"><strong>{_tpl_field("university", "Вуз")}</strong></p>
<p style="text-align:center;margin:0 0 1.6em;font-size:12pt;color:#444;">Факультет / кафедра: {_tpl_field("faculty", "Кафедра")}</p>
<p style="text-align:center;margin:2.2em 0 0.4em;font-size:16pt;"><strong>ОТЧЁТ</strong></p>
<p style="text-align:center;margin:0 0 2em;font-size:14pt;">по лабораторной работе</p>
<p style="text-align:center;margin:0 0 0.5em;font-size:13pt;">«{_tpl_field("title", "Тема работы")}»</p>
<p style="margin:2.5em 0 0.4em;">Дисциплина: {_tpl_field("discipline", "Дисциплина")}</p>
<p style="margin:0 0 0.4em;">Студент: {_tpl_field("student", "Студент")} &nbsp;&nbsp; Группа: {_tpl_field("group", "Группа")}</p>
<p style="margin:0 0 2.5em;">Преподаватель: {_tpl_field("teacher", "Преподаватель")}</p>
<p style="text-align:center;margin-top:4em;">{_tpl_field("city", "Город")}, {_tpl_field("year", "Год")}</p>
<hr style="margin:2.5em 0;border:none;border-top:1px solid #ccc;">
<h2>1. Цель работы</h2>
<p>{_tpl_field("goal", "Цель")}</p>
<h2>2. Оборудование и материалы</h2>
<p>{_tpl_field("materials", "Материалы")}</p>
<h2>3. Ход работы</h2>
<p>{_tpl_field("procedure", "Ход работы")}</p>
<h2>4. Результаты</h2>
<p>{_tpl_field("results", "Результаты")}</p>
<h2>5. Выводы</h2>
<p>{_tpl_field("conclusion", "Выводы")}</p>
""".strip()


def _require_session_user(request: Request, user: str = "") -> str | None:
    session_user = get_session_username(request)
    username = session_user or (user or "").strip()
    return username or None


@app.get("/export-template", response_class=HTMLResponse)
async def export_template_page(
    request: Request,
    user: str = "",
    tpl_id: str = "",
    exp_id: str = "",
    node_id: str = "",
    db: Session = Depends(get_db),
):
    username = _require_session_user(request, user)
    if not username:
        return RedirectResponse(url=f"{BASE_PATH}/login?error=LoginRequired", status_code=302)
    if get_session_username(request) and user and user != get_session_username(request):
        q = f"user={get_session_username(request)}"
        if tpl_id:
            q += f"&tpl_id={tpl_id}"
        if exp_id:
            q += f"&exp_id={exp_id}"
        if node_id:
            q += f"&node_id={node_id}"
        return RedirectResponse(url=f"{BASE_PATH}/export-template?{q}", status_code=302)

    templates_list = (
        db.query(ExportTemplateDB)
        .filter(ExportTemplateDB.user_id == username)
        .order_by(ExportTemplateDB.updated_at.desc())
        .all()
    )
    current = None
    if tpl_id:
        current = (
            db.query(ExportTemplateDB)
            .filter(ExportTemplateDB.id == tpl_id, ExportTemplateDB.user_id == username)
            .first()
        )
    if not current and templates_list:
        current = templates_list[0]
    if not current:
        now = datetime.datetime.now().isoformat()
        current = ExportTemplateDB(
            id=str(uuid.uuid4())[:10],
            user_id=username,
            name="Шаблон кафедры",
            kind="docx",
            html=_DEFAULT_EXPORT_TEMPLATE_HTML,
            created_at=now,
            updated_at=now,
        )
        db.add(current)
        db.commit()
        db.refresh(current)
        templates_list = [current]

    back_url = ""
    if exp_id and node_id:
        back_url = f"{BASE_PATH}/experiment/{username}/{exp_id}/node/{node_id}"

    return templates.TemplateResponse(
        request,
        "export_template.html",
        {
            "user": username,
            "active_page": "export_template",
            "tpl": current,
            "templates_list": templates_list,
            "exp_id": exp_id,
            "node_id": node_id,
            "back_url": back_url,
        },
    )


@app.get("/api/export-template/list")
async def api_export_template_list(request: Request, db: Session = Depends(get_db)):
    username = get_session_username(request)
    if not username:
        return JSONResponse({"ok": False, "error": "login"}, status_code=401)
    rows = (
        db.query(ExportTemplateDB)
        .filter(ExportTemplateDB.user_id == username)
        .order_by(ExportTemplateDB.updated_at.desc())
        .all()
    )
    return {
        "ok": True,
        "items": [
            {
                "id": r.id,
                "name": r.name,
                "kind": r.kind,
                "updated_at": r.updated_at,
            }
            for r in rows
        ],
    }


@app.post("/api/export-template/save")
async def api_export_template_save(request: Request, db: Session = Depends(get_db)):
    username = get_session_username(request)
    if not username:
        return JSONResponse({"ok": False, "error": "login"}, status_code=401)
    body = await request.json()
    tpl_id = (body.get("id") or "").strip()
    name = (body.get("name") or "Шаблон кафедры").strip()[:120]
    kind = (body.get("kind") or "docx").strip().lower()
    if kind not in ("docx", "pdf", "pptx"):
        kind = "docx"
    html = body.get("html") or ""
    now = datetime.datetime.now().isoformat()

    row = None
    if tpl_id:
        row = (
            db.query(ExportTemplateDB)
            .filter(ExportTemplateDB.id == tpl_id, ExportTemplateDB.user_id == username)
            .first()
        )
    if row:
        row.name = name
        row.kind = kind
        row.html = html
        row.updated_at = now
    else:
        row = ExportTemplateDB(
            id=str(uuid.uuid4())[:10],
            user_id=username,
            name=name,
            kind=kind,
            html=html or _DEFAULT_EXPORT_TEMPLATE_HTML,
            created_at=now,
            updated_at=now,
        )
        db.add(row)
    db.commit()
    db.refresh(row)
    return {"ok": True, "id": row.id, "updated_at": row.updated_at}


@app.post("/api/export-template/create")
async def api_export_template_create(request: Request, db: Session = Depends(get_db)):
    username = get_session_username(request)
    if not username:
        return JSONResponse({"ok": False, "error": "login"}, status_code=401)
    body = await request.json()
    name = (body.get("name") or "Новый шаблон").strip()[:120]
    kind = (body.get("kind") or "docx").strip().lower()
    if kind not in ("docx", "pdf", "pptx"):
        kind = "docx"
    now = datetime.datetime.now().isoformat()
    row = ExportTemplateDB(
        id=str(uuid.uuid4())[:10],
        user_id=username,
        name=name,
        kind=kind,
        html=_DEFAULT_EXPORT_TEMPLATE_HTML,
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"ok": True, "id": row.id}


@app.post("/api/export-template/delete")
async def api_export_template_delete(request: Request, db: Session = Depends(get_db)):
    username = get_session_username(request)
    if not username:
        return JSONResponse({"ok": False, "error": "login"}, status_code=401)
    body = await request.json()
    tpl_id = (body.get("id") or "").strip()
    row = (
        db.query(ExportTemplateDB)
        .filter(ExportTemplateDB.id == tpl_id, ExportTemplateDB.user_id == username)
        .first()
    )
    if not row:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    db.delete(row)
    db.commit()
    return {"ok": True}

# ── API: Versioning (Git-like branches) ──

@app.post("/api/node/version/create")
async def create_version(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    exp_id = body.get("exp_id")
    _owned_experiment(db, request, exp_id)
    node_id = body.get("node_id")
    message = body.get("message", "Снимок")
    
    notes = db.query(NodeNotesDB).filter(NodeNotesDB.exp_id == exp_id, NodeNotesDB.node_id == node_id).first()
    if notes:
        version = _maybe_create_node_version(
            db,
            exp_id=exp_id,
            node_id=node_id,
            html=notes.notes_html or "",
            author=_version_author_label(request, db),
            message=message or "Снимок",
            force=True,
        )
        if not version:
            # identical content — still acknowledge latest
            latest = (
                db.query(NodeVersionDB)
                .filter(NodeVersionDB.exp_id == exp_id, NodeVersionDB.node_id == node_id)
                .order_by(NodeVersionDB.created_at.desc())
                .first()
            )
            return JSONResponse({"ok": True, "version_id": latest.id if latest else None, "unchanged": True})
        db.commit()
        return JSONResponse({"ok": True, "version_id": version.id})
    return JSONResponse({"ok": False})

@app.get("/api/node/version/list")
async def list_versions(exp_id: str, node_id: str, request: Request, db: Session = Depends(get_db)):
    _owned_experiment(db, request, exp_id)
    versions = db.query(NodeVersionDB).filter(NodeVersionDB.exp_id == exp_id, NodeVersionDB.node_id == node_id).order_by(NodeVersionDB.created_at.desc()).all()
    return JSONResponse({
        "versions": [
            {
                "id": v.id,
                "created_at": v.created_at,
                "message": v.message,
                "author": getattr(v, "author", "") or "",
            }
            for v in versions
        ]
    })

@app.get("/api/node/version/{version_id}")
async def get_version_snapshot(version_id: str, request: Request, db: Session = Depends(get_db)):
    v = db.query(NodeVersionDB).filter(NodeVersionDB.id == version_id).first()
    if not v:
        return JSONResponse({"ok": False}, status_code=404)
    _owned_experiment(db, request, v.exp_id)
    return JSONResponse(
        {
            "id": v.id,
            "created_at": v.created_at,
            "message": v.message,
            "html_snapshot": sanitize_note_html(v.html_snapshot or ""),
        }
    )

@app.post("/api/node/version/rollback")
async def rollback_version(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    version_id = body.get("version_id")
    
    version = db.query(NodeVersionDB).filter(NodeVersionDB.id == version_id).first()
    if not version:
        raise HTTPException(404, "Версия не найдена.")
    _owned_experiment(db, request, version.exp_id)
    return _store_note(db, request, version.exp_id, version.node_id, version.html_snapshot or "",
                       body.get("revision"), message="Восстановление версии")


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

# ── Uploaded media ──

_MEDIA_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,127}")
_MEDIA_LINK = re.compile(r"/static/uploads/([A-Za-z0-9._-]+)")
_NOTE_IMAGE_TYPES = {"PNG": ("png", "image/png"), "JPEG": ("jpg", "image/jpeg"),
                     "GIF": ("gif", "image/gif"), "WEBP": ("webp", "image/webp")}
_INLINE_TYPES = {media for _, media in _NOTE_IMAGE_TYPES.values()} | {"audio/webm", "audio/ogg", "audio/mp4"}
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _media_path(filename: str) -> str | None:
    if not _MEDIA_NAME.fullmatch(filename) or ".." in filename:
        return None
    path = os.path.join(UPLOADS_DIR, filename)
    return path if os.path.isfile(path) and not os.path.islink(path) else None


def _note_media_names(text: str) -> set[str]:
    return set(_MEDIA_LINK.findall(text or ""))


def _serve_protected_upload(db: Session, request: Request, filename: str):
    path = _media_path(filename)
    if not path:
        raise HTTPException(404, "Файл не найден.")
    media = db.get(MediaUploadDB, filename)
    headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"}
    if not media:
        # Profile pictures are intentionally public. No other unindexed upload is published.
        aliases = (filename, "/static/uploads/" + filename, f"{BASE_PATH}/static/uploads/{filename}")
        if not db.query(UserDB.username).filter(UserDB.avatar_url.in_(aliases)).first():
            raise HTTPException(404, "Файл не найден.")
        kind = mimetypes.guess_type(path)[0]
        if kind not in {m for _, m in _NOTE_IMAGE_TYPES.values()}:
            raise HTTPException(404, "Файл не найден.")
        return FileResponse(path, media_type=kind, headers=headers)

    user = get_session_username(request)
    allowed = bool(user and media.owner_id == user)
    if not allowed and media.chat_id and user:
        chat = db.get(FriendChatDB, media.chat_id)
        allowed = bool(chat and user in (chat.user1, chat.user2))
    if not allowed and media.exp_id:
        experiment = db.get(ExperimentDB, media.exp_id)
        if experiment and experiment.is_public == "true":
            notes = db.query(NodeNotesDB.notes_html, NodeNotesDB.annotation_url).filter_by(exp_id=media.exp_id)
            allowed = any(filename in _note_media_names((html or "") + " " + (annotation or "")) for html, annotation in notes)
    if not allowed:
        raise HTTPException(404, "Файл не найден.")
    inline = media.media_type in _INLINE_TYPES
    return FileResponse(path, media_type=media.media_type if inline else "application/octet-stream",
                        filename=None if inline else media.original_name, headers=headers)


def _image_type(path: str) -> tuple[str, str] | None:
    from PIL import Image as PILImage, UnidentifiedImageError
    try:
        with PILImage.open(path) as img:
            kind = _NOTE_IMAGE_TYPES.get(img.format)
            if kind and img.width * img.height <= 25_000_000:
                img.verify()
                return kind
    except (OSError, ValueError, UnidentifiedImageError, PILImage.DecompressionBombError):
        pass
    return None


async def _stage_upload(file: UploadFile) -> tuple[str, int]:
    size = 0
    fd, path = tempfile.mkstemp(prefix=".incoming-", dir=UPLOADS_DIR)
    try:
        with os.fdopen(fd, "wb") as dest:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > _MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Файл превышает 50 МБ.")
                dest.write(chunk)
        if not size:
            raise HTTPException(422, "Файл пуст.")
        return path, size
    except BaseException:
        os.unlink(path)
        raise


def _reserve_quota(db: Session, username: str, size: int):
    # SQLite write lock keeps the quota check and subsequent reservation atomic across workers.
    db.connection().exec_driver_sql("BEGIN IMMEDIATE")
    user = db.get(UserDB, username)
    if not user or user.is_banned == "true":
        raise HTTPException(401, "LoginRequired")
    plan = user.plan or "free"
    expiry = _parse_iso_dt(user.plan_expires_at or "")
    if expiry and expiry <= datetime.datetime.now():
        plan = "free"
    if _user_storage_bytes(username, db) + size > storage_limit_bytes(plan):
        raise HTTPException(413, "Недостаточно места в хранилище.")


@app.post("/api/upload")
async def upload_file(request: Request, file: UploadFile = File(...), exp_id: str = Form(""),
                      chat_id: str = Form(""), kind: str = Form(""), db: Session = Depends(get_db)):
    username = _active_username(request)
    if bool(exp_id) == bool(chat_id):
        raise HTTPException(422, "Укажите эксперимент или чат для файла.")
    if exp_id:
        _owned_experiment(db, request, exp_id)
    else:
        chat = db.get(FriendChatDB, chat_id)
        other = chat.user2 if chat and chat.user1 == username else chat.user1 if chat and chat.user2 == username else None
        if not other or not _can_chat(db, username, other):
            raise HTTPException(404, "Чат не найден.")
    staged, size = await _stage_upload(file)
    path = None
    try:
        if exp_id:
            image_type = _image_type(staged)
            if image_type is None:
                raise HTTPException(415, "Загрузите изображение JPEG, PNG, GIF или WebP.")
            ext, media = image_type
        elif kind == "voice":
            with open(staged, "rb") as stream:
                header = stream.read(16)
            if header.startswith(bytes.fromhex("1A45DFA3")):
                ext, media = "webm", "audio/webm"
            elif header.startswith(b"OggS"):
                ext, media = "ogg", "audio/ogg"
            elif header[4:8] == b"ftyp":
                ext, media = "m4a", "audio/mp4"
            else:
                raise HTTPException(415, "Этот формат аудио не поддерживается.")
        else:
            ext, media = "bin", "application/octet-stream"
        _reserve_quota(db, username, size)
        filename = f"{uuid.uuid4().hex}.{ext}"
        path = os.path.join(UPLOADS_DIR, filename)
        db.add(MediaUploadDB(filename=filename, owner_id=username, exp_id=exp_id or None,
                             chat_id=chat_id or None, size=size, media_type=media,
                             original_name=os.path.basename(file.filename or filename)))
        os.replace(staged, path)
        db.commit()
        return JSONResponse({"url": upload_public_url(filename)})
    except BaseException:
        if path and os.path.isfile(path):
            os.unlink(path)
        raise
    finally:
        if os.path.isfile(staged):
            os.unlink(staged)


def _index_legacy_media(db: Session):
    marker = "scoped-media-v1"
    db.connection().exec_driver_sql("BEGIN IMMEDIATE")
    if db.get(DataMigrationDB, marker):
        db.rollback()
        return
    refs = {}

    def remember(filename, owner, exp=None, chat=None):
        if owner:
            refs.setdefault(filename, set()).add((owner, exp, chat))

    for model, column in ((NodeNotesDB, NodeNotesDB.notes_html), (NodeVersionDB, NodeVersionDB.html_snapshot),
                          (NodeNotesDB, NodeNotesDB.annotation_url)):
        rows = db.query(model.exp_id, ExperimentDB.user_id, column).join(ExperimentDB, ExperimentDB.id == model.exp_id)
        for exp_id, owner, content in rows:
            for filename in _note_media_names(normalize_upload_urls(content or "")):
                remember(filename, owner, exp=exp_id)
    rows = db.query(FriendMessageDB.chat_id, FriendMessageDB.sender, FriendMessageDB.content)
    for chat_id, owner, content in rows:
        if (content or "").startswith(("🎤||", "📎||")):
            for filename in _note_media_names(content.split("||")[1]):
                remember(filename, owner, chat=chat_id)
    existing = {name for (name,) in db.query(MediaUploadDB.filename)}
    for filename, scopes in refs.items():
        path = _media_path(filename)
        if not path or filename in existing or len(scopes) != 1:
            continue
        owner, exp_id, chat_id = next(iter(scopes))
        detected = _image_type(path) if exp_id else None
        kind = detected[1] if detected else mimetypes.guess_type(path)[0]
        if chat_id:
            kind = {"video/webm": "audio/webm", "audio/ogg": "audio/ogg", "audio/mp4": "audio/mp4"}.get(kind)
        if not kind or kind not in _INLINE_TYPES:
            if exp_id:
                continue
            kind = "application/octet-stream"
        db.add(MediaUploadDB(filename=filename, owner_id=owner, exp_id=exp_id, chat_id=chat_id,
                             size=os.path.getsize(path), media_type=kind, original_name=filename))
    db.add(DataMigrationDB(name=marker))
    db.commit()


@app.on_event("startup")
def migrate_private_uploads():
    with SessionLocal() as db:
        _index_legacy_media(db)
        _drain_file_cleanup(db)


# ── API: AI chat ──

def _strip_code_fences(code: str) -> str:
    code = (code or "").strip()
    m = re.match(r"^```(?:python|py|bash|r)?\s*\n(.*?)```$", code, re.DOTALL | re.IGNORECASE)
    if m:
        code = m.group(1).strip()
    # Models sometimes leak HTML into code (e.g. </br>, <br/>).
    code = re.sub(r"(?i)</?\s*br\s*/?\s*>", "\n", code)
    code = re.sub(r"(?i)</?\s*(?:p|div|span|html|body)\s*>", "", code)
    code = _html.unescape(code)
    return code.strip()


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
    """True only when the user explicitly asks to put code/HTML into the notebook."""
    u = (user_message or "").lower()
    if not u:
        return False
    explicit = (
        "insert_code", "[insert_code]", "edit_code_cell", "append_document",
        "вставь код", "вставить код", "добавь код", "добавить код",
        "добавь ячейк", "вставь ячейк", "в блокнот код", "ячейку кода",
        "добавь таблиц", "вставь таблиц", "edit_document",
    )
    return any(k in u for k in explicit)


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

    # Drop notebook mutations unless the user explicitly asked for them.
    if actions and not _user_wants_notebook_change(user_message):
        actions = {}

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
    _owned_experiment(db, request, exp_id)
    node_id = body.get("node_id")
    user_message = body.get("message", "")
    regenerate = bool(body.get("regenerate"))
    html_context = body.get("html_context", "")
    notebook_context = body.get("notebook_context") or []

    session_user = get_session_username(request)
    if not session_user:
        return JSONResponse({"ok": False, "error": "LoginRequired", "reply": "Войдите в аккаунт."}, status_code=401)
    u = db.query(UserDB).filter(UserDB.username == session_user).first()
    plan = sync_user_plan_effective(u, db) if u else "free"
    if not plan_allows_ai(plan):
        pricing = f"{BASE_PATH}/pricing" if BASE_PATH else "/pricing"
        pricing_hash = f"{BASE_PATH}/#ceny" if BASE_PATH else "/#ceny"
        return JSONResponse({
            "ok": False,
            "error": "PlanRequired",
            "reply": (
                "ИИ-ассистент доступен на тарифе Pro (319 ₽/мес). Сейчас у вас Free — без ИИ.\n\n"
                f"[Подключить Pro — 319 ₽/мес]({pricing_hash}) · [Все тарифы]({pricing})"
            ),
            "pricing_url": pricing_hash,
        })

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

    notebook_empty = not any(
        isinstance(c, dict) and (c.get("code") or "").strip()
        for c in (notebook_context or [])
    )
    system_prompt = (
        "Ты — ассистент-учёный для биологии и лабораторного журнала. "
        "Блокнот содержит текстовые заметки и опционально Python-ячейки.\n\n"
        "ГЛАВНОЕ ПРАВИЛО ПРО КОД:\n"
        "По умолчанию отвечай ТОЛЬКО в чате: кратко, по делу, без вставки в блокнот. "
        "НЕ используй [INSERT_CODE], [EDIT_CODE_CELL], [EDIT_DOCUMENT], [APPEND_DOCUMENT], "
        "пока пользователь ЯВНО не попросит вставить/добавить/изменить код или текст в блокноте "
        "(формулировки вроде «вставь код», «добавь ячейку», «добавь таблицу в блокнот»).\n"
        "Вопросы, объяснения, советы, проверка текста, «что дальше», приветствия — только текст в чате, без тегов.\n\n"
        "Если пользователь ЯВНО просит изменить блокнот:\n"
        "1) Новый Python-код — [INSERT_CODE]...[/INSERT_CODE] (без markdown ``` и без HTML-тегов).\n"
        "2) Правка ячейки N — [EDIT_CODE_CELL:N]...[/EDIT_CODE_CELL].\n"
        "3) Замена всей заметки — [EDIT_DOCUMENT]...[/EDIT_DOCUMENT].\n"
        "4) Добавить HTML в конец заметки — [APPEND_DOCUMENT]...[/APPEND_DOCUMENT].\n"
        "5) В чате после тегов - 1 предложение, что сделал.\n"
        "6) Экспорт PDF/DOCX/PPTX - кнопками в панели; не пиши скрипты экспорта без явной просьбы."
    )
    if notebook_empty:
        system_prompt += (
            "\nБлокнот сейчас без ячеек кода. Не вставляй стартовый шаблон сам: "
            "предложи текстом, что можно сделать, и жди явной просьбы вставить код."
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
            tg_notify_ai_issue("chat", assistant_msg, exception=e)

    assistant_msg, notebook_actions = _parse_assistant_notebook_actions(assistant_msg, user_message)

    # Save to history
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_msg})
    notes.chat_history = history
    db.commit()

    return JSONResponse({
        "reply": assistant_msg,
        "edit_html": sanitize_note_html(notebook_actions.get("edit_html") or "") or None,
        "append_html": sanitize_note_html(notebook_actions.get("append_html") or "") or None,
        "insert_code": notebook_actions.get("insert_code"),
        "edit_code_cell": notebook_actions.get("edit_code_cell"),
    })


@app.post("/api/chat/clear")
async def clear_chat_history(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    exp_id = body.get("exp_id")
    _owned_experiment(db, request, exp_id)
    node_id = body.get("node_id")
    if not exp_id or not node_id:
        return JSONResponse({"ok": False, "error": "missing ids"}, status_code=400)

    notes = db.query(NodeNotesDB).filter(NodeNotesDB.exp_id == exp_id, NodeNotesDB.node_id == node_id).first()
    if notes:
        notes.chat_history = []
        db.commit()
    return JSONResponse({"ok": True})

import sys
import asyncio
# Per-user kernel locks (execution moved to isolated subprocess — see kernel_worker.py)
_kernel_exec_locks: dict[str, asyncio.Lock] = {}
_KERNEL_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
_KERNEL_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_worker.py")
_KERNEL_TIMEOUT_SEC = 20
_KERNEL_SECRET_ENV_KEYS = (
    "AI_API_KEY",
    "AI_API_URL",
    "AI_CHAT_MODEL",
    "AI_CODE_MODEL",
    "AI_HTTP_PROXY",
    "OCR_API_KEY",
    "OCR_API_URL",
    "OCR_VISION_MODEL",
    "OCR_VISION_MODELS",
    "TG_BOT_TOKEN",
    "TG_CHAT_ID",
    "TG_HTTP_PROXY",
    "YANDEX_CLIENT_ID",
    "YANDEX_CLIENT_SECRET",
    "OAUTH_STATE_SECRET",
    "SESSION_SECRET",
    "ADMIN_KEY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _kernel_user_hash(username: str) -> str:
    return hashlib.sha256((username or "").encode("utf-8")).hexdigest()[:16]


def _kernel_output_dir(username: str, exp_id: str) -> str:
    """Per-user private output dir (mode 0700). Same host UID still needs mount ns for /tmp."""
    safe = re.sub(r"[^\w.\-]", "", (exp_id or "default"))[:80] or "default"
    path = os.path.join(KERNEL_OUTPUT_ROOT, _kernel_user_hash(username), safe)
    os.makedirs(path, mode=0o700, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def _kernel_scrubbed_env(out_dir: str) -> dict[str, str]:
    """Minimal env for user code — no API secrets / tokens / proxies."""
    path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    env = {
        "PATH": path,
        "HOME": out_dir,
        "TMPDIR": "/tmp",
        "TMP": "/tmp",
        "TEMP": "/tmp",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "MPLBACKEND": "Agg",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    venv_python = sys.executable
    env["VIRTUAL_ENV"] = os.path.dirname(os.path.dirname(venv_python))
    # Defense in depth: never forward known secret keys even if env merge changes later
    for k in _KERNEL_SECRET_ENV_KEYS:
        env.pop(k, None)
    return env


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


def _kernel_sandbox_command(out_dir: str) -> list[str] | None:
    sandbox = shutil.which("bwrap") if sys.platform.startswith("linux") else None
    if not sandbox:
        return None
    cmd = [sandbox, "--die-with-parent", "--new-session", "--unshare-all", "--cap-drop", "ALL",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    runtime_paths = {sys.executable, os.path.realpath(sys.executable)}
    runtime_paths.update(path for name in ("stdlib", "platstdlib", "purelib", "platlib")
                         if (path := sysconfig.get_path(name)))
    library_dir = sysconfig.get_config_var("LIBDIR")
    python_library = sysconfig.get_config_var("LDLIBRARY")
    if library_dir and python_library:
        library = os.path.join(library_dir, python_library)
        runtime_paths.update((library, os.path.realpath(library)))
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if multiarch:
        runtime_paths.update(f"{base}/{multiarch}" for base in ("/usr/lib", "/lib"))
    runtime_paths.update(("/lib64", "/usr/lib64", "/etc/ld.so.cache", "/etc/fonts", "/usr/share/fonts"))
    # Mount runtime files individually. The source tree, databases and host network stay outside.
    for path in sorted(runtime_paths, key=lambda name: (name.count("/"), name)):
        if path and os.path.exists(path):
            cmd.extend(("--ro-bind", path, path))
    return [*cmd, "--ro-bind", _KERNEL_WORKER, "/runner.py",
            "--bind", os.path.abspath(out_dir), "/work", "--chdir", "/work",
            "--", sys.executable, "/runner.py"]


def _run_kernel_subprocess(code: str, out_dir: str, meta: dict) -> tuple[str, bool]:
    cmd = _kernel_sandbox_command(out_dir)
    if cmd is None:
        return "Безопасная среда Python недоступна. Запуск отключён.", True
    payload = json.dumps(
        {"code": code, "cwd": "/work", "ns_path": "/work/.kernel_ns.pkl",
         "meta": {**meta, "KERNEL_OUTPUT_DIR": "/work"}},
        ensure_ascii=False,
    ).encode("utf-8")
    env = _kernel_scrubbed_env("/work")
    try:
        proc = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            timeout=_KERNEL_TIMEOUT_SEC,
            env=env,
            cwd=out_dir,
        )
    except subprocess.TimeoutExpired:
        return f"Превышено время выполнения ({_KERNEL_TIMEOUT_SEC} с).", True
    except OSError:
        logging.getLogger(__name__).exception("Could not start the Python sandbox")
        return "Безопасная среда Python недоступна. Запуск отключён.", True

    try:
        data = json.loads(proc.stdout)
        if isinstance(data, dict) and isinstance(data.get("error"), bool):
            return str(data.get("output") or ""), data["error"] or proc.returncode != 0
    except (ValueError, UnicodeError):
        pass
    logging.getLogger(__name__).warning(
        "Python sandbox exited with code %s: %s",
        proc.returncode, proc.stderr[:500].decode("utf-8", "replace"),
    )
    return "Безопасная среда Python не ответила. Выполнение остановлено.", True


COMPUTE_UNITS = {
    "rtx4090": {"id": "rtx4090", "name": "RTX 4090", "vram": "24 GB"},
    "tesla_a100": {"id": "tesla_a100", "name": "Tesla A100", "vram": "80 GB"},
    "rtx3090": {"id": "rtx3090", "name": "RTX 3090", "vram": "24 GB"},
}

@app.post("/api/kernel/execute")
async def execute_code(request: Request, db: Session = Depends(get_db)):
    session_user = get_session_username(request)
    if not session_user:
        return JSONResponse({"output": "LoginRequired", "error": True, "images": []}, status_code=401)

    body = await request.json()
    exp_id = (body.get("exp_id") or "").strip() or "default"
    code = body.get("code") or ""
    if not isinstance(code, str):
        return JSONResponse({"output": "Invalid code", "error": True, "images": []}, status_code=400)
    if len(code) > 200_000:
        return JSONResponse({"output": "Code too large", "error": True, "images": []}, status_code=400)

    # Only the experiment owner can run code against its workspace
    if exp_id != "default":
        exp = (
            db.query(ExperimentDB)
            .filter(ExperimentDB.id == exp_id, ExperimentDB.user_id == session_user)
            .first()
        )
        if not exp:
            return JSONResponse({"output": "Experiment not found", "error": True, "images": []}, status_code=404)

    compute_unit_id = (body.get("compute_unit") or "rtx4090").strip()
    compute_unit = COMPUTE_UNITS.get(compute_unit_id, COMPUTE_UNITS["rtx4090"])

    out_dir = _kernel_output_dir(session_user, exp_id)
    lock_key = f"{session_user}:{exp_id}"
    lock = _kernel_exec_locks.setdefault(lock_key, asyncio.Lock())

    meta = {
        "COMPUTE_UNIT": compute_unit["name"],
        "COMPUTE_VRAM": compute_unit["vram"],
        "COMPUTE_UNIT_ID": compute_unit["id"],
        "KERNEL_OUTPUT_DIR": out_dir,
    }
    images: list[dict[str, str]] = []
    async with lock:
        before_artifacts = _snapshot_kernel_artifacts(out_dir)
        output, error = await asyncio.to_thread(_run_kernel_subprocess, code, out_dir, meta)
        changed_artifacts = _kernel_artifacts_changed(out_dir, before_artifacts)

    for name in changed_artifacts:
        images.append(
            {
                "name": name,
                "url": f"{BASE_PATH}/api/kernel/artifact/{quote(exp_id, safe='')}/{quote(name, safe='')}",
            }
        )

    return JSONResponse({
        "output": output,
        "error": error or None,
        "images": images,
        "output_dir": out_dir,
        "compute_unit": compute_unit,
    })


@app.get("/api/kernel/artifact/{exp_id}/{filename}")
async def kernel_artifact(exp_id: str, filename: str, request: Request, db: Session = Depends(get_db)):
    session_user = get_session_username(request)
    if not session_user:
        return JSONResponse({"error": "LoginRequired"}, status_code=401)
    safe_name = os.path.basename(filename or "")
    if not safe_name or safe_name != filename or ".." in filename:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    if os.path.splitext(safe_name)[1].lower() not in _KERNEL_IMAGE_EXTS:
        return JSONResponse({"error": "Unsupported type"}, status_code=400)
    if exp_id != "default":
        exp = (
            db.query(ExperimentDB)
            .filter(ExperimentDB.id == exp_id, ExperimentDB.user_id == session_user)
            .first()
        )
        if not exp:
            return JSONResponse({"error": "Not found"}, status_code=404)
    path = os.path.join(_kernel_output_dir(session_user, exp_id), safe_name)
    if not os.path.isfile(path) or os.path.islink(path):
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
    return FileResponse(path, media_type=media, filename=safe_name, headers={"X-Content-Type-Options": "nosniff"})

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
    _owned_experiment(db, request, exp_id)
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
    _owned_experiment(db, request, exp_id)
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
    _owned_experiment(db, request, exp_id)
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
    _owned_experiment(db, request, exp_id)
    page = db.query(WikiPageDB).filter(WikiPageDB.id == page_id, WikiPageDB.exp_id == exp_id).first()
    if page:
        db.delete(page)
        db.commit()
    return JSONResponse({"ok": True})


@app.get("/api/experiment/{exp_id}/wiki/pages")
async def list_wiki_pages(exp_id: str, request: Request, db: Session = Depends(get_db)):
    _owned_experiment(db, request, exp_id)
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
async def list_experiment_files(exp_id: str, request: Request, db: Session = Depends(get_db)):
    _owned_experiment(db, request, exp_id)
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
    _owned_experiment(db, request, exp_id)
    username = _active_username(request)
    staged, size = await _stage_upload(file)
    path = None
    try:
        _reserve_quota(db, username, size)
        _owned_experiment(db, request, exp_id)
        original = os.path.basename(file.filename or "file")
        ext = os.path.splitext(original)[1].lower()
        fname = uuid.uuid4().hex + (ext if re.fullmatch(r"\.[a-z0-9]{1,12}", ext) else ".bin")
        directory = os.path.join(EXPERIMENT_FILES_DIR, exp_id)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, fname)
        record = ExperimentFileDB(exp_id=exp_id, filename=fname, original_name=original, file_size=str(size),
                                  mime_type=file.content_type or "application/octet-stream", uploaded_by=username)
        db.add(record)
        os.replace(staged, path)
        db.commit()
        return JSONResponse({"ok": True, "file_id": record.id})
    except BaseException:
        if path and os.path.isfile(path):
            os.unlink(path)
        raise
    finally:
        if os.path.isfile(staged):
            os.unlink(staged)


@app.get("/api/experiment/{exp_id}/files/{file_id}/download")
async def download_experiment_file(exp_id: str, file_id: str, request: Request, db: Session = Depends(get_db)):
    _owned_experiment(db, request, exp_id)
    ef = db.query(ExperimentFileDB).filter(ExperimentFileDB.id == file_id, ExperimentFileDB.exp_id == exp_id).first()
    if not ef:
        return JSONResponse({"error": "File not found"}, status_code=404)
    path = os.path.join(EXPERIMENT_FILES_DIR, exp_id, ef.filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "File not on disk"}, status_code=404)
    from fastapi.responses import FileResponse
    return FileResponse(path, filename=ef.original_name, media_type="application/octet-stream",
                        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"})


@app.post("/api/experiment/{exp_id}/files/delete")
async def delete_experiment_file(exp_id: str, request: Request, db: Session = Depends(get_db)):
    _owned_experiment(db, request, exp_id)
    body = await request.json()
    file_id = body.get("file_id")
    ef = db.query(ExperimentFileDB).filter(ExperimentFileDB.id == file_id, ExperimentFileDB.exp_id == exp_id).first()
    if ef:
        path = os.path.join(EXPERIMENT_FILES_DIR, exp_id, ef.filename)
        _queue_file_cleanup(db, path)
        db.delete(ef)
        db.commit()
        _drain_file_cleanup(db)
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
async def get_reagents(user: str, request: Request, db: Session = Depends(get_db)):
    user = _active_username(request)
    items = db.query(ReagentDB).filter(ReagentDB.user_id == user).all()
    return JSONResponse({"items": [{"id":r.id,"name":r.name,"supplier":r.supplier,"category":r.category,"cas":r.cas,"formula":r.formula,"concentration":r.concentration,"storage":r.storage,"amount":r.amount,"min_amount":r.min_amount,"unit":r.unit,"expiry":r.expiry,"project":r.project,"location":r.location} for r in items]})

@app.post("/api/reagents")
async def create_reagent(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    r = ReagentDB(user_id=_active_username(request), name=body.get("name",""), supplier=body.get("supplier",""), category=body.get("category","Жидкий"), cas=body.get("cas",""), formula=body.get("formula",""), concentration=body.get("concentration",""), storage=body.get("storage",""), amount=body.get("amount","0"), min_amount=body.get("min_amount","0"), unit=body.get("unit","мл"), expiry=body.get("expiry",""), project=body.get("project",""), location=body.get("location",""))
    db.add(r)
    db.commit()
    return JSONResponse({"ok": True, "id": r.id})

@app.put("/api/reagents/{item_id}")
async def update_reagent(item_id: str, request: Request, db: Session = Depends(get_db)):
    r = db.query(ReagentDB).filter(ReagentDB.id == item_id, ReagentDB.user_id == _active_username(request)).first()
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
async def delete_reagent(item_id: str, request: Request, db: Session = Depends(get_db)):
    r = db.query(ReagentDB).filter(ReagentDB.id == item_id, ReagentDB.user_id == _active_username(request)).first()
    if r:
        db.delete(r)
        db.commit()
    return JSONResponse({"ok": True})

# ── API: Consumables ──

@app.get("/api/consumables")
async def get_consumables(user: str, request: Request, db: Session = Depends(get_db)):
    user = _active_username(request)
    items = db.query(ConsumableDB).filter(ConsumableDB.user_id == user).all()
    return JSONResponse({"items": [{"id":c.id,"name":c.name,"supplier":c.supplier,"category":c.category,"amount":c.amount,"min_amount":c.min_amount,"unit":c.unit,"location":c.location,"comment":c.comment} for c in items]})

@app.post("/api/consumables")
async def create_consumable(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    c = ConsumableDB(user_id=_active_username(request), name=body.get("name",""), supplier=body.get("supplier",""), category=body.get("category","Пластик"), amount=body.get("amount","0"), min_amount=body.get("min_amount","0"), unit=body.get("unit","шт"), location=body.get("location",""), comment=body.get("comment",""))
    db.add(c)
    db.commit()
    return JSONResponse({"ok": True, "id": c.id})

@app.put("/api/consumables/{item_id}")
async def update_consumable(item_id: str, request: Request, db: Session = Depends(get_db)):
    c = db.query(ConsumableDB).filter(ConsumableDB.id == item_id, ConsumableDB.user_id == _active_username(request)).first()
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
async def delete_consumable(item_id: str, request: Request, db: Session = Depends(get_db)):
    c = db.query(ConsumableDB).filter(ConsumableDB.id == item_id, ConsumableDB.user_id == _active_username(request)).first()
    if c:
        db.delete(c)
        db.commit()
    return JSONResponse({"ok": True})

# ── API: Projects ──

@app.get("/api/projects")
async def get_projects(user: str, request: Request, db: Session = Depends(get_db)):
    user = _active_username(request)
    items = _projects_for_user(db, user).all()
    return JSONResponse({"items": [{"id": p.id, "name": p.name, "status": p.status, "description": p.description or "", "end_date": p.end_date or ""} for p in items]})

@app.post("/api/projects")
async def create_project(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    username = _active_username(request)
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
    _require_project(db, request, item_id, owner_only=True)
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
async def delete_project(item_id: str, request: Request, db: Session = Depends(get_db)):
    project = _require_project(db, request, item_id, owner_only=True)
    _delete_project_rows(db, project)
    db.commit()
    return JSONResponse({"ok": True})

@app.get("/api/projects/{item_id}")
async def get_project_detail(item_id: str, request: Request, db: Session = Depends(get_db)):
    p = _require_project(db, request, item_id)
    tasks = db.query(ProjectTaskDB).filter(ProjectTaskDB.project_id == item_id).all()
    members = db.query(ProjectMemberDB).filter(ProjectMemberDB.project_id == item_id).all()
    invites = []
    if p.user_id == _active_username(request):
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

_TASK_STATUSES = {"Открыто", "В работе", "На проверке", "Завершено"}


def _task_values(body: dict, existing: ProjectTaskDB | None = None) -> dict:
    defaults = {"title": "", "description": "", "assignee": "", "status": "Открыто",
                "priority": "Средний", "start_date": "", "end_date": ""}
    values = {key: body.get(key, getattr(existing, key, fallback)) for key, fallback in defaults.items()}
    if not isinstance(values["title"], str) or not values["title"].strip():
        raise HTTPException(422, "Введите название задачи.")
    if not isinstance(values["status"], str) or values["status"] not in _TASK_STATUSES:
        raise HTTPException(422, "Выберите допустимый статус задачи.")
    values["title"] = values["title"].strip()
    values["start_date"], values["end_date"] = _date_range(values["start_date"], values["end_date"])
    return values


@app.post("/api/projects/{project_id}/tasks")
async def create_task(project_id: str, request: Request, db: Session = Depends(get_db)):
    _require_project(db, request, project_id)
    body = await request.json()
    t = ProjectTaskDB(project_id=project_id, created_by=_active_username(request), **_task_values(body))
    db.add(t)
    db.commit()
    return JSONResponse({"ok": True, "id": t.id})

@app.put("/api/projects/{project_id}/tasks/{task_id}")
async def update_task(project_id: str, task_id: str, request: Request, db: Session = Depends(get_db)):
    _require_project(db, request, project_id)
    t = db.query(ProjectTaskDB).filter(ProjectTaskDB.id == task_id, ProjectTaskDB.project_id == project_id).first()
    if not t:
        return JSONResponse({"ok": False}, status_code=404)
    body = await request.json()
    for key, value in _task_values(body, t).items():
        setattr(t, key, value)
    db.commit()
    return JSONResponse({"ok": True})

@app.delete("/api/projects/{project_id}/tasks/{task_id}")
async def delete_task(project_id: str, task_id: str, request: Request, db: Session = Depends(get_db)):
    _require_project(db, request, project_id)
    t = db.query(ProjectTaskDB).filter(ProjectTaskDB.id == task_id, ProjectTaskDB.project_id == project_id).first()
    if t:
        db.delete(t)
        db.commit()
    return JSONResponse({"ok": True})

# ── API: Project Members ──

@app.post("/api/projects/{project_id}/members")
async def add_member(project_id: str, request: Request, db: Session = Depends(get_db)):
    _require_project(db, request, project_id, owner_only=True)
    body = await request.json()
    username = body.get("username", "")
    existing = db.query(ProjectMemberDB).filter(ProjectMemberDB.project_id == project_id, ProjectMemberDB.username == username).first()
    if not existing:
        m = ProjectMemberDB(project_id=project_id, username=username)
        db.add(m)
        db.commit()
    return JSONResponse({"ok": True})

@app.delete("/api/projects/{project_id}/members/{username}")
async def remove_member(project_id: str, username: str, request: Request, db: Session = Depends(get_db)):
    _require_project(db, request, project_id, owner_only=True)
    m = db.query(ProjectMemberDB).filter(ProjectMemberDB.project_id == project_id, ProjectMemberDB.username == username).first()
    if m:
        db.delete(m)
        db.commit()
    return JSONResponse({"ok": True})

# ── API: Project Invites ──

@app.post("/api/projects/{project_id}/invite")
async def create_invite(project_id: str, request: Request, db: Session = Depends(get_db)):
    project = _require_project(db, request, project_id, owner_only=True)
    inv = ProjectInviteDB(project_id=project_id, created_by=project.user_id)
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
    username = _active_username(request)
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
    return JSONResponse({"users": [{"username": u.username} for u in users]})

@app.get("/api/project/{project_id}/gantt")
@app.get("/api/projects/{project_id}/gantt")
async def get_gantt_data(project_id: str, request: Request, db: Session = Depends(get_db)):
    _require_project(db, request, project_id)
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
async def toggle_public(user: str, exp_id: str, request: Request, db: Session = Depends(get_db)):
    _owned_experiment(db, request, exp_id)
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
            parts.append(f"пробный Pro на {m.group(1)} дн.")
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


def _admin_signature(expires: str) -> str:
    return hmac.new(SESSION_SECRET.encode(), f"admin:{ADMIN_KEY}:{expires}".encode(), hashlib.sha256).hexdigest()


def admin_required(request: Request) -> bool:
    token = request.cookies.get(ADMIN_COOKIE, "")
    expires, _, signature = token.partition(".")
    return bool(ADMIN_KEY and expires.isascii() and expires.isdecimal() and len(expires) <= 12
                and int(expires) > time.time() and signature.isascii()
                and hmac.compare_digest(signature, _admin_signature(expires)))


@app.get(LAB_ADMIN_PATH, response_class=HTMLResponse)
async def admin_login_page(request: Request, error: str = None):
    return templates.TemplateResponse(request, "admin_login.html", {"error": error})

@app.get("/admin", include_in_schema=False)
async def admin_legacy_entry_redirect():
    return RedirectResponse(url=f"{BASE_PATH}{LAB_ADMIN_PATH}", status_code=301)

@app.post("/admin")
@app.post(LAB_ADMIN_PATH)
async def admin_login(admin_key: str = Form(...)):
    if ADMIN_KEY and hmac.compare_digest(admin_key.encode(), ADMIN_KEY.encode()):
        expiry = str(int(time.time()) + ADMIN_SESSION_SECONDS)
        response = RedirectResponse(url=f"{BASE_PATH}{LAB_ADMIN_PATH}/dashboard", status_code=302)
        response.set_cookie(ADMIN_COOKIE, f"{expiry}.{_admin_signature(expiry)}",
                            max_age=ADMIN_SESSION_SECONDS, path=f"{BASE_PATH}{LAB_ADMIN_PATH}",
                            secure=COOKIE_SECURE, httponly=True, samesite="strict")
        return response
    return RedirectResponse(
        url=f"{BASE_PATH}{LAB_ADMIN_PATH}?error=InvalidKey", status_code=302
    )

@app.get("/admin/dashboard", include_in_schema=False)
async def admin_legacy_dashboard_redirect():
    return RedirectResponse(url=f"{BASE_PATH}{LAB_ADMIN_PATH}/dashboard", status_code=302)

@app.get(f"{LAB_ADMIN_PATH}/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    if not admin_required(request):
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
            "users": user_list,
            "all_experiments": exp_list,
            "stats": stats,
            "recent_activity": recent_activity,
            "insights": {k: v for k, v in insights.items() if k not in ("registered_at", "last_login_at", "events_per_user", "sparks")},
        }
    )

def _queue_file_cleanup(db: Session, path: str, *, directory: bool = False):
    normalized = os.path.abspath(path)
    roots = (UPLOADS_DIR, EXPERIMENT_FILES_DIR, NOTEBOOKS_BASE, KERNEL_OUTPUT_ROOT)
    if any(normalized.startswith(os.path.abspath(root) + os.sep) for root in roots):
        if not db.get(DataCleanupDB, normalized):
            db.add(DataCleanupDB(path=normalized, is_directory=int(directory)))


def _drain_file_cleanup(db: Session):
    roots = [os.path.realpath(root) + os.sep for root in
             (UPLOADS_DIR, EXPERIMENT_FILES_DIR, NOTEBOOKS_BASE, KERNEL_OUTPUT_ROOT)]
    for row in db.query(DataCleanupDB).all():
        parent_path = os.path.realpath(os.path.dirname(row.path)) + os.sep
        if not any(parent_path.startswith(root) for root in roots):
            continue
        try:
            if row.is_directory and not os.path.islink(row.path):
                shutil.rmtree(row.path)
            else:
                os.unlink(row.path)
        except FileNotFoundError:
            pass
        except OSError:
            # Keep the durable entry for a retry on the next cleanup/startup.
            continue
        db.delete(row)
    db.commit()


def _delete_project_rows(db: Session, project: ProjectDB):
    for model in (ProjectTaskDB, ProjectMemberDB, ProjectInviteDB):
        db.query(model).filter_by(project_id=project.id).delete()
    db.delete(project)


def _delete_experiment_rows(db: Session, exp: ExperimentDB):
    for upload in db.query(MediaUploadDB).filter_by(exp_id=exp.id):
        _queue_file_cleanup(db, os.path.join(UPLOADS_DIR, upload.filename))
    for model in (MediaUploadDB, NodeNotesDB, NodeVersionDB, ExperimentFileDB, WikiPageDB):
        db.query(model).filter_by(exp_id=exp.id).delete()
    _queue_file_cleanup(db, os.path.join(EXPERIMENT_FILES_DIR, exp.id), directory=True)
    _queue_file_cleanup(db, os.path.join(KERNEL_OUTPUT_ROOT, _kernel_user_hash(exp.user_id), exp.id), directory=True)
    notebook_root = os.path.join(NOTEBOOKS_BASE, exp.user_id)
    if os.path.isdir(notebook_root):
        for part in os.listdir(notebook_root):
            if part == exp.id or part.endswith("_" + exp.id):
                _queue_file_cleanup(db, os.path.join(notebook_root, part), directory=True)
    db.delete(exp)


def _delete_user_rows(db: Session, user: UserDB):
    username = user.username
    for exp in db.query(ExperimentDB).filter_by(user_id=username).all():
        _delete_experiment_rows(db, exp)
    for project in db.query(ProjectDB).filter_by(user_id=username).all():
        _delete_project_rows(db, project)
    for chat in db.query(FriendChatDB).filter((FriendChatDB.user1 == username) | (FriendChatDB.user2 == username)).all():
        for upload in db.query(MediaUploadDB).filter_by(chat_id=chat.id):
            _queue_file_cleanup(db, os.path.join(UPLOADS_DIR, upload.filename))
        db.query(MediaUploadDB).filter_by(chat_id=chat.id).delete()
        db.query(FriendMessageDB).filter_by(chat_id=chat.id).delete()
        db.delete(chat)
    for ticket in db.query(SupportTicketDB).filter_by(user_id=username).all():
        db.query(SupportReplyDB).filter_by(ticket_id=ticket.id).delete()
        db.delete(ticket)
    for equipment in db.query(EquipmentDB).filter_by(owner_id=username).all():
        db.query(EquipmentBookingDB).filter_by(equipment_id=equipment.id).delete()
        db.delete(equipment)
    for organization in db.query(OrganizationDB).filter_by(owner_id=username).all():
        db.query(OrganizationMemberDB).filter_by(org_id=organization.id).delete()
        db.delete(organization)
    for model in (ReagentDB, ConsumableDB, LoginSessionDB, ExportTemplateDB, ReferralLinkDB,
                  EquipmentBookingDB, OrganizationMemberDB):
        db.query(model).filter_by(user_id=username).delete()
    db.query(ProjectMemberDB).filter_by(username=username).delete()
    db.query(ProjectInviteDB).filter_by(created_by=username).delete()
    db.query(FriendDB).filter((FriendDB.user_id == username) | (FriendDB.friend_id == username)).delete()
    avatar = _avatar_local_path(user.avatar_url or "")
    if avatar:
        _queue_file_cleanup(db, avatar)
    for upload in db.query(MediaUploadDB).filter_by(owner_id=username):
        _queue_file_cleanup(db, os.path.join(UPLOADS_DIR, upload.filename))
    db.query(MediaUploadDB).filter_by(owner_id=username).delete()
    db.delete(user)


@app.post("/admin/delete_user")
@app.post(f"{LAB_ADMIN_PATH}/delete_user")
async def admin_delete_user(request: Request, db: Session = Depends(get_db)):
    if not admin_required(request):
        raise HTTPException(403, "Unauthorized")
    body = await request.json()
    user = db.get(UserDB, body.get("username"))
    if not user:
        raise HTTPException(404, "Пользователь не найден.")
    _delete_user_rows(db, user)
    db.commit()
    _drain_file_cleanup(db)
    return {"ok": True}


@app.post("/admin/delete_experiment")
@app.post(f"{LAB_ADMIN_PATH}/delete_experiment")
async def admin_delete_experiment(request: Request, db: Session = Depends(get_db)):
    if not admin_required(request):
        raise HTTPException(403, "Unauthorized")
    body = await request.json()
    exp = db.get(ExperimentDB, body.get("exp_id"))
    if not exp:
        raise HTTPException(404, "Эксперимент не найден.")
    _delete_experiment_rows(db, exp)
    db.commit()
    _drain_file_cleanup(db)
    return {"ok": True}


@app.get("/admin/api/user_dashboard")
@app.get(f"{LAB_ADMIN_PATH}/api/user_dashboard")
async def admin_user_dashboard(request: Request, user: str = None, db: Session = Depends(get_db)):
    if not admin_required(request):
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
async def admin_corporate(request: Request, db: Session = Depends(get_db)):
    if not admin_required(request):
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
async def admin_referrals(request: Request, db: Session = Depends(get_db)):
    if not admin_required(request):
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
async def admin_support(request: Request, db: Session = Depends(get_db)):
    if not admin_required(request):
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
async def admin_logs(request: Request, limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    if not admin_required(request):
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
async def admin_analytics(request: Request, db: Session = Depends(get_db)):
    if not admin_required(request):
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
async def admin_get_tickets(request: Request, status: str = None, priority: str = None, db: Session = Depends(get_db)):
    if not admin_required(request):
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
    if not admin_required(request):
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
    if not admin_required(request):
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
    if not admin_required(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    user = db.query(UserDB).filter(UserDB.username == body.get("username")).first()
    if not user:
        return JSONResponse({"error": "User not found"}, status_code=404)
    if "email" in body:
        user.email = body["email"]
    if "is_banned" in body:
        user.is_banned = "true" if body["is_banned"] in ("true", True) else "false"
        if user.is_banned == "true":
            db.query(LoginSessionDB).filter(LoginSessionDB.user_id == user.username).delete()
    db.commit()
    audit_log(db, "admin", "update", "user", body.get("username", ""), json.dumps(body))
    return JSONResponse({"ok": True})

@app.post("/admin/api/user/bulk")
@app.post(f"{LAB_ADMIN_PATH}/api/user/bulk")
async def admin_bulk_user_action(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    if not admin_required(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    action = body.get("action", "")
    usernames = body.get("usernames", [])
    if action == "delete":
        for username in set(usernames):
            user = db.get(UserDB, username)
            if user:
                _delete_user_rows(db, user)
        db.commit()
        _drain_file_cleanup(db)
    elif action == "ban":
        for u in usernames:
            usr = db.query(UserDB).filter(UserDB.username == u).first()
            if usr:
                usr.is_banned = "true"
                db.query(LoginSessionDB).filter(LoginSessionDB.user_id == u).delete()
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
async def admin_get_organizations(request: Request, db: Session = Depends(get_db)):
    if not admin_required(request):
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
    if not admin_required(request):
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
async def admin_get_referral_codes(request: Request, db: Session = Depends(get_db)):
    if not admin_required(request):
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
async def admin_export_users_csv(request: Request, db: Session = Depends(get_db)):
    if not admin_required(request):
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
        tg_notify_ai_issue("planner", msg, exception=e)
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
    u = db.query(UserDB).filter(UserDB.username == username).first()
    plan = sync_user_plan_effective(u, db) if u else "free"
    if not plan_allows_ai(plan):
        return JSONResponse({
            "ok": False,
            "error": "ИИ-планировщик доступен на тарифе Pro. Сейчас у вас Free.",
        }, status_code=403)
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
            "Ты — ИИ-планировщик лабораторного журнала Lab-Journal. "
            "Пользователь описывает эксперимент — составь понятное название и пошаговый протокол (5–9 коротких шагов). "
            "Ответь ТОЛЬКО валидным JSON без пояснений и разметки в формате: "
            "{\"title\": \"Название эксперимента\", \"steps\": [\"Шаг 1\", \"Шаг 2\", ...]}"
        )
    else:
        plan = state["plan"]
        system = (
            "Ты — научный ассистент в лабораторном журнале Lab-Journal. "
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

# ── AI Agent: обычный чат (не планировщик) ──

_AI_AGENT_STATE: dict = {}

_AI_AGENT_SYSTEM = (
    "Ты — ИИ-агент Lab-Journal: научный ассистент (физика, биология, химия, лабораторная практика).\n"
    "Отвечай по-русски. Стиль — как у ChatGPT: ясно, структурировано, без воды и канцелярита.\n\n"
    "ОФОРМЛЕНИЕ ТЕКСТА (Markdown, рендерится в чате):\n"
    "- Заголовок темы: одна строка `# ...`\n"
    "- Секции: `## 1. ...`, `## 2. ...` (не пиши # как обычный текст в середине абзаца)\n"
    "- Списки: `- пункт` или `1. пункт` — короткие строки, не простыни\n"
    "- Ключевые термины: **жирным**; формулы — KaTeX, не ASCII-костыли\n"
    "- Между секциями — пустая строка. Не склеивай всё в один абзац.\n"
    "- Не злоупотребляй заголовками: 2–5 секций достаточно. Без «Введение/Заключение», если не просили.\n"
    "- Сначала суть (2–4 предложения), потом детали списком или секциями.\n\n"
    "ФОРМУЛЫ — ТОЛЬКО KaTeX:\n"
    "- Inline: $E_k=\\tfrac12 m v^2$, блок: $$D=\\frac{k_B T}{6\\pi\\eta r}$$\n"
    "- ЗАПРЕЩЕНО ASCII-костыли без $: 10^-9, k_BT, √, x^2.\n"
    "- Не оборачивай формулы в ```.\n\n"
    "КОГДА ДЕЛАТЬ [[HTML]] (редко!):\n"
    "- ТОЛЬКО если пользователь просит калькулятор / симуляцию / интерактив / слайдеры / живой график,\n"
    "  или без интерактива ответ заметно хуже.\n"
    "- Обычные вопросы, объяснения, формулы, вывод — ОТВЕЧАЙ Markdown+KaTeX БЕЗ [[HTML]].\n"
    "- Не вставляй HTML «на всякий случай» и не дублируй текст ответа целиком в HTML.\n\n"
    "ЕСЛИ ДЕЛАЕШЬ [[HTML]]:\n"
    "[[HTML:Краткий заголовок]]\n"
    "<div class=\"card\">...</div>\n"
    "<script>/* интерактив */</script>\n"
    "[[/HTML]]\n"
    "В песочнице уже есть KaTeX, auto-render и helper labMath(el, tex, display?).\n"
    "Живые формулы — ТОЛЬКО так (одной строкой, без переносов внутри tex):\n"
    "  labMath(out, 's = v_0 t = ' + s.toFixed(3) + '\\\\,\\\\mathrm{м}');\n"
    "ЗАПРЕЩЕНО:\n"
    "- клеить число внутрь $...$ через span/textContent (ломает KaTeX);\n"
    "- писать `$...= ' + n + '\\\\mathrm{м}$` с переносом строки;\n"
    "- подключать свой CDN KaTeX.\n"
    "Правила качества:\n"
    "- Компактный UI, radius 12–16px, акцент #2563eb.\n"
    "- Один артефакт = одна задача. update()/recalc()/calculate() на input и при загрузке.\n"
    "- Без fetch/XHR/localStorage/cookie/window.open.\n\n"
    "Простые чертежи без JS: [[SVG]]...svg...[[/SVG]]\n\n"
    "Готовый виджет электростатики (если не нужен кастомный HTML):\n"
    "[[WIDGET:efield]]{\"mode\":\"point\",\"q\":2e-9,\"r\":0.1}[[/WIDGET]]\n"
    "mode: point|dipole|plane|sphere|cylinder. Дефолты: q=2e-9; d=1e-3; σ=5e-9; R=0.05 Q=1e-8; λ=3e-9; r=0.1.\n\n"
    "Не навязывай создание эксперимента. Только если ЯВНО просят план в журнал, в конце:\n"
    "[[PLAN]]{\"title\":\"...\",\"steps\":[\"шаг1\",\"шаг2\"]}[[/PLAN]]"
)


def _strip_plan_marker(text: str) -> tuple[str, dict | None]:
    """Extract optional [[PLAN]]...[[/PLAN]] from agent reply."""
    if not text:
        return "", None
    start = text.find("[[PLAN]]")
    end = text.find("[[/PLAN]]")
    if start == -1 or end == -1 or end <= start:
        return text.strip(), None
    raw = text[start + 8:end].strip()
    clean = (text[:start] + text[end + 9:]).strip()
    plan = _extract_plan_json(raw)
    return clean or text.strip(), plan


def _normalize_chat_images(raw) -> list[str]:
    """Validate data-URL images from client (max 4)."""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw[:4]:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s.startswith("data:image/") or ";base64," not in s:
            continue
        # rough size guard (~6MB base64)
        if len(s) > 8_000_000:
            continue
        out.append(_ocr_prepare_image_data_url(s))
    return out


async def _llm_vision_agent_chat(messages) -> tuple[str | None, str | None]:
    """Vision chat via OpenRouter (same key/models as OCR)."""
    if not OCR_API_KEY:
        return None, "Vision API не настроен (OCR_API_KEY). Нельзя разобрать фото."
    headers = {
        "Authorization": f"Bearer {OCR_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_ORIGIN or "https://labjournal.tech",
        "X-Title": "Lab-Journal AI Chat",
    }
    last_err: str | None = None
    for model in _ocr_vision_model_candidates():
        try:
            async with ai_http_client(timeout=90.0) as client:
                resp = await client.post(
                    OCR_API_URL,
                    headers=headers,
                    json={"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 4096},
                )
                try:
                    data = resp.json()
                except Exception:
                    data = {"error": (resp.text or "")[:300]}
                if resp.status_code >= 400:
                    err = ""
                    if isinstance(data, dict):
                        err = (
                            (data.get("error") or {}).get("message")
                            if isinstance(data.get("error"), dict)
                            else data.get("error")
                        ) or str(data.get("message", ""))
                    msg = f"{model} ({resp.status_code}): {err or 'ошибка API'}"
                    last_err = msg
                    if _ocr_should_try_next_model(RuntimeError(msg)):
                        continue
                    tg_notify_ai_issue("ai-agent-vision", msg)
                    return None, msg
                text = ""
                if isinstance(data, dict):
                    text = (
                        (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                    ).strip()
                if not text:
                    last_err = f"{model}: пустой ответ"
                    continue
                return text, None
        except Exception as e:
            last_err = format_ai_request_error(e)
            continue
    err = last_err or "Vision-модели недоступны."
    tg_notify_ai_issue("ai-agent-vision", err)
    return None, err


@app.post("/api/ai-agent/chat")
async def ai_agent_chat(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    session_user = get_session_username(request)
    username = (body.get("user") or "").strip() or (session_user or "")
    message = (body.get("message") or "").strip()
    images = _normalize_chat_images(body.get("images"))
    if not username:
        return JSONResponse({"ok": False, "error": "Войдите в аккаунт"}, status_code=401)
    if session_user and username != session_user:
        return JSONResponse({"ok": False, "error": "Неверный пользователь сессии"}, status_code=403)
    u = db.query(UserDB).filter(UserDB.username == username).first()
    plan = sync_user_plan_effective(u, db) if u else "free"
    if not plan_allows_ai(plan):
        return JSONResponse({
            "ok": False,
            "error": "ИИ-агент доступен на тарифе Pro. Сейчас у вас Free.",
        }, status_code=403)

    state = _AI_AGENT_STATE.setdefault(username, {"history": []})
    if body.get("reset"):
        state["history"] = []
        return JSONResponse({
            "ok": True,
            "type": "chat",
            "text": "Новый диалог. Спросите что угодно по лабораторной работе, методике или расчётам.",
        })
    if not message and not images:
        return JSONResponse({"ok": False, "error": "Пустое сообщение"})

    history = state["history"]
    user_text_for_history = message
    if images:
        prefix = f"[прикреплено фото: {len(images)}]"
        user_text_for_history = f"{prefix}\n{message}".strip() if message else prefix

    if images:
        # Multimodal turn via vision model; keep prior turns as text only
        user_parts: list[dict] = []
        prompt = message or "Посмотри на изображение(я) и ответь по-русски: что видно, что важно, что делать дальше."
        user_parts.append({"type": "text", "text": prompt})
        for url in images:
            user_parts.append({"type": "image_url", "image_url": {"url": url}})
        msgs = [{"role": "system", "content": _AI_AGENT_SYSTEM + "\nПользователь приложил фото — опирайся на изображение."}]
        for h in history[-12:]:
            content = h.get("content")
            if isinstance(content, str):
                msgs.append({"role": h["role"], "content": content})
        msgs.append({"role": "user", "content": user_parts})
        history.append({"role": "user", "content": user_text_for_history})
        ai_text, ai_err = await _llm_vision_agent_chat(msgs)
    else:
        history.append({"role": "user", "content": message})
        msgs = [{"role": "system", "content": _AI_AGENT_SYSTEM}] + history[-20:]
        ai_text, ai_err = await _llm_planner_chat(msgs)

    if not ai_text:
        text = ai_err or "ИИ-агент временно недоступен. Попробуйте ещё раз через минуту."
        history.append({"role": "assistant", "content": text})
        return JSONResponse({"ok": True, "type": "chat", "text": text})

    clean, plan_obj = _strip_plan_marker(ai_text)
    history.append({"role": "assistant", "content": clean})
    if len(history) > 40:
        state["history"] = history[-40:]

    if plan_obj:
        return JSONResponse({
            "ok": True,
            "type": "plan",
            "title": plan_obj["title"],
            "steps": plan_obj["steps"],
            "text": clean,
        })
    return JSONResponse({"ok": True, "type": "chat", "text": clean})


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
    levels = [[{"id": "start", "title": "План работы", "content": "Краткое описание эксперимента."}]]
    for i, step in enumerate(steps, start=1):
        levels.append([{"id": str(uuid.uuid4())[:8], "title": f"Шаг {i}. {step[:90]}", "content": step}])

    # Папка-ноутбук — по возможности (на сервере с NOTEBOOKS_PATH), иначе пропускаем
    try:
        user_dir = os.path.join(NOTEBOOKS_BASE, username)
        exp_dir = os.path.join(user_dir, exp_id)
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
