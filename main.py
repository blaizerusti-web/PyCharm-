# ================================
# mega.py — Super Alex (Jarvis persona; OpenAI key-rotation or Ollama;
# Telegram bot; OCR; optional voice IO; health server; watchdog; logging;
# humane/unrestricted modes; persistence; growth hooks)
# ================================

# ----------- stdlib -----------
import os, sys, json, re, time, threading, logging, hashlib, sqlite3, base64, uuid, random
from pathlib import Path
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, Any, List, Tuple

# -------- runtime memory (in-proc key/value) --------
MEM_RUNTIME: Dict[str, Any] = {}  # used for round-robin indexes, caches, toggles

def mget(key: str, default=None):
    return MEM_RUNTIME.get(key, default)

def mset(key: str, value: Any):
    MEM_RUNTIME[key] = value
    return value

# ----------- soft/optional imports -----------
import importlib

def _soft_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

# Optional packages (don’t crash if missing)
PIL = _soft_import("PIL")
Image = getattr(PIL, "Image", None) if PIL else None
pytesseract = _soft_import("pytesseract")
pydub = _soft_import("pydub")     # for simple TTS playback (optional)
sr = _soft_import("speech_recognition")  # optional voice input
requests = _soft_import("requests") or __import__("urllib.request")  # fallback to stdlib if needed

# Warn (but never crash) on optional deps
if sr is None:
    print("⚠️  SpeechRecognition not available — Jarvis voice input disabled.")
if pytesseract is None or Image is None:
    print("⚠️  OCR (pytesseract/Pillow) not available — /ocr disabled.")
try:
    # pydub warns about ffmpeg; that's fine
    if pydub:
        from pydub import AudioSegment, playback  # type: ignore
except Exception:
    pydub = None

# ----------- config helpers -----------
def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _split_csv(val: Optional[str]) -> List[str]:
    if not val:
        return []
    return [x.strip() for x in val.split(",") if x.strip()]

ROOT = Path(os.getenv("APP_ROOT", "."))

CFG: Dict[str, Any] = {
    # Backends
    "BACKEND": os.getenv("BACKEND", "openai"),  # "openai" | "ollama"
    "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "OPENAI_API_KEYS": _split_csv(os.getenv("OPENAI_API_KEYS") or os.getenv("OPENAI_API_KEY")),
    "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://localhost:11434"),
    "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct"),

    # Telegram
    "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN", ""),
    "ADMIN_IDS": [int(x) for x in _split_csv(os.getenv("ADMIN_IDS"))] if os.getenv("ADMIN_IDS") else [],

    # Modes & safety toggles
    "JARVIS_MODE": _env_bool("JARVIS_MODE", False),         # persona voice (witty assistant)
    "HUMANE_TONE": _env_bool("HUMANE_TONE", True),          # gentle tone & helpfulness
    "UNRESTRICTED_HINT": _env_bool("UNRESTRICTED_HINT", False),  # loosens style (still safe)

    # Features
    "ENABLE_OCR": _env_bool("ENABLE_OCR", True),
    "ENABLE_SPEECH": _env_bool("ENABLE_SPEECH", False),     # only works if sr installed
    "ENABLE_TTS": _env_bool("ENABLE_TTS", False),           # requires pydub+ffmpeg or OS TTS
    "ENABLE_WEB": _env_bool("ENABLE_WEB", False),           # placeholder hook
    "PERSIST_SQLITE": _env_bool("PERSIST_SQLITE", True),

    # Infra
    "PORT": _env_int("PORT", 8080),
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    "KEEPALIVE_SECS": _env_int("KEEPALIVE_SECS", 45),
    "HEALTH_SECRET": os.getenv("HEALTH_SECRET", hashlib.sha256(os.getenv("TELEGRAM_TOKEN","").encode()).hexdigest()[:10]),
}

# ----------- logging -----------
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, CFG["LOG_LEVEL"].upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("alex")

# ----------- SQLite persistence (growth hooks) -----------
DB = ROOT / "alex.sqlite3"

def db_init():
    if not CFG["PERSIST_SQLITE"]:
        return
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        role TEXT,
        text TEXT,
        meta TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT,
        detail TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()

def db_log_message(user_id: str, role: str, text: str, meta: Optional[Dict[str, Any]] = None):
    if not CFG["PERSIST_SQLITE"]:
        return
    try:
        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute("INSERT INTO messages(user_id, role, text, meta) VALUES (?,?,?,?)",
                    (user_id, role, text, json.dumps(meta or {})))
        conn.commit()
    except Exception as e:
        log.warning(f"db_log_message failed: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

def db_log_event(kind: str, detail: str):
    if not CFG["PERSIST_SQLITE"]:
        return
    try:
        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute("INSERT INTO events(kind, detail) VALUES (?,?)", (kind, detail))
        conn.commit()
    except Exception as e:
        log.warning(f"db_log_event failed: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

db_init()

# ----------- health server & keepalive -----------
class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            qs = parse_qs(urlparse(self.path).query)
            secret = qs.get("s", [""])[0]
            status = {
                "ok": True,
                "ts": datetime.utcnow().isoformat() + "Z",
                "backend": CFG["BACKEND"],
                "openai_model": CFG["OPENAI_MODEL"],
                "ollama_model": CFG["OLLAMA_MODEL"],
                "jarvis": CFG["JARVIS_MODE"],
                "humane": CFG["HUMANE_TONE"],
                "unrestricted_hint": CFG["UNRESTRICTED_HINT"],
                "sr": bool(sr),
                "ocr": bool(pytesseract and Image),
            }
            if secret != CFG["HEALTH_SECRET"]:
                self.send_response(403); self.end_headers(); self.wfile.write(b"forbidden")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode("utf-8"))
        except Exception as e:
            try:
                self.send_response(500); self.end_headers(); self.wfile.write(str(e).encode())
            except Exception:
                pass

def _start_health_server():
    port = CFG["PORT"]
    def _serve():
        try:
            httpd = HTTPServer(("0.0.0.0", port), _Health)
            log.info(f"Health server on :{port} (secret ?s={CFG['HEALTH_SECRET']})")
            httpd.serve_forever()
        except Exception as e:
            log.error(f"Health server error: {e}")
    th = threading.Thread(target=_serve, daemon=True); th.start()

def _keepalive():
    # lightweight heartbeat (updates an in-memory timestamp and logs)
    mset("last_tick", time.time())
    db_log_event("tick", datetime.utcnow().isoformat() + "Z")
    threading.Timer(CFG["KEEPALIVE_SECS"], _keepalive).start()

# ----------- prompts & personas -----------
BASE_SYSTEM = """You are Alex — a helpful, fast, developer-friendly assistant living inside a Telegram bot.
Be clear, concise, and practical. If code is requested, return runnable snippets and explain briefly.
Default to a friendly, humane style. Never expose secrets or tokens. Stay within platform limits.
"""

JARVIS_SPICE = """Adopt a confident “Jarvis” vibe: witty, succinct, solution-oriented.
Use bullet points when helpful, and provide next steps. Keep it classy, not snarky.
"""

UNRESTRICTED_SPICE = """Relax stylistic guardrails slightly. Be more direct and exploratory, but still respectful and safe.
Avoid illegal, dangerous, or disallowed content. Decline such requests firmly and propose safe alternatives.
"""

def build_system_prompt() -> str:
    parts = [BASE_SYSTEM]
    if CFG["HUMANE_TONE"]:
        parts.append("Keep a warm, encouraging tone when appropriate.\n")
    if CFG["JARVIS_MODE"]:
        parts.append(JARVIS_SPICE)
    if CFG["UNRESTRICTED_HINT"]:
        parts.append(UNRESTRICTED_SPICE)
    return "\n".join(parts).strip()

# ----------- key rotation (OpenAI) -----------
def _next_openai_key() -> Optional[str]:
    keys = CFG["OPENAI_API_KEYS"]
    if not keys:
        return None
    idx = mget("openai_rr", -1) + 1
    if idx >= len(keys):
        idx = 0
    mset("openai_rr", idx)
    return keys[idx]

# ----------- providers -----------
def _http_post(url: str, headers: Dict[str,str], payload: Dict[str,Any]) -> Tuple[int, Dict[str,Any]]:
    """Simple POST wrapper using requests if available; fallback to urllib."""
    try:
        if hasattr(requests, "post"):
            rsp = requests.post(url, headers=headers, json=payload, timeout=60)  # type: ignore
            return rsp.status_code, (rsp.json() if rsp.content else {})
        else:
            import urllib.request
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as f:   # nosec
                data = f.read()
                return 200, (json.loads(data.decode("utf-8")) if data else {})
    except Exception as e:
        return 0, {"error": str(e)}

def llm_chat(messages: List[Dict[str,str]]) -> str:
    """
    Provider-agnostic chat. messages: [{"role":"system/user/assistant","content": "..."}]
    Returns assistant text (best effort).
    """
    backend = CFG["BACKEND"].lower()

    if backend == "ollama":
        url = f"{CFG['OLLAMA_URL'].rstrip('/')}/api/chat"
        payload = {"model": CFG["OLLAMA_MODEL"], "messages": messages, "stream": False}
        code, data = _http_post(url, {"Content-Type":"application/json"}, payload)
        if code == 200 and isinstance(data, dict):
            try:
                # Typical Ollama chat format: { "message": {"role":"assistant","content":"..."} }
                if "message" in data and "content" in data["message"]:
                    return data["message"]["content"]
                # Some versions: {"choices":[{"message":{"content":...}}]}
                if "choices" in data and data["choices"]:
                    return data["choices"][0].get("message",{}).get("content","")
            except Exception:
                pass
        raise RuntimeError(f"Ollama error ({code}): {data}")

    # default: openai
    key = _next_openai_key()
    if not key:
        raise RuntimeError("No OPENAI_API_KEYS provided")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    payload = {"model": CFG["OPENAI_MODEL"], "messages": messages, "temperature": 0.5}
    code, data = _http_post("https://api.openai.com/v1/chat/completions", headers, payload)
    if code == 200 and isinstance(data, dict):
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            pass
    raise RuntimeError(f"OpenAI error ({code}): {data}")

def alex_reply(user_id: str, text: str, context: Optional[List[Dict[str,str]]] = None) -> str:
    """
    Build the full message list with system persona + short rolling context and query llm_chat().
    """
    system = build_system_prompt()
    msgs: List[Dict[str,str]] = [{"role":"system","content":system}]
    if context:
        # carry limited rolling context (last few turns)
        for m in context[-8:]:
            if m.get("role") in {"user","assistant"} and "content" in m:
                msgs.append({"role":m["role"], "content":m["content"]})
    msgs.append({"role":"user","content":text})
    db_log_message(user_id, "user", text, {"backend": CFG["BACKEND"]})
    out = llm_chat(msgs)
    db_log_message(user_id, "assistant", out, {"backend": CFG["BACKEND"]})
    return out

# ----------- OCR helper -----------
def run_ocr(image_bytes: bytes) -> str:
    if not (pytesseract and Image and CFG["ENABLE_OCR"]):
        return "OCR not available on this deployment."
    try:
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes))  # type: ignore
        txt = pytesseract.image_to_string(img)  # type: ignore
        return txt.strip() or "(no text found)"
    except Exception as e:
        return f"OCR failed: {e}"

# ----------- Voice helpers (safe stubs) -----------
def jarvis_listen(timeout: int = 7) -> str:
    if not (CFG["ENABLE_SPEECH"] and sr):
        return ""
    try:
        r = sr.Recognizer()  # type: ignore
        with sr.Microphone() as source:  # type: ignore
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=timeout)  # type: ignore
        return r.recognize_google(audio)  # type: ignore
    except Exception as e:
        log.warning(f"listen failed: {e}")
        return ""

def jarvis_say(text: str):
    # keep simple to avoid binary deps; we log instead
    if CFG["ENABLE_TTS"] and pydub:
        # You can plug in your TTS wav/bytes here and play with pydub.playback
        log.info(f"(TTS) {text[:200]}")
    else:
        log.info(f"(say) {text[:200]}")
# ----------- Telegram bot wiring -----------

# We support python-telegram-bot v20+
try:
    from telegram import Update, MessageEntity
    from telegram.constants import ChatAction, ParseMode
    from telegram.ext import (
        Application, ApplicationBuilder, CommandHandler, MessageHandler,
        ContextTypes, filters, CallbackContext
    )
except Exception as e:
    Update = None  # type: ignore
    log.warning("telegram libraries not available. Telegram features disabled.")

# ---- small utilities ----
def is_admin(user_id: int) -> bool:
    return (user_id in CFG["ADMIN_IDS"]) if CFG["ADMIN_IDS"] else True  # default open if no list

def ctx_key(chat_id: int) -> str:
    return f"ctx:{chat_id}"

def get_ctx(chat_id: int) -> List[Dict[str, str]]:
    return mget(ctx_key(chat_id), [])

def push_ctx(chat_id: int, role: str, content: str, max_len: int = 16):
    ctx = get_ctx(chat_id)
    ctx.append({"role": role, "content": content})
    if len(ctx) > max_len:
        ctx = ctx[-max_len:]
    mset(ctx_key(chat_id), ctx)

async def send_typing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.effective_chat.send_action(ChatAction.TYPING)  # type: ignore
    except Exception:
        pass

def md_escape(s: str) -> str:
    # basic sanitation for MarkdownV2 / HTML; we’re using HTML by default
    return s

# ---- command handlers ----
HELP_TEXT = (
    "🤖 <b>Alex / Jarvis</b>\n"
    "• Simply send a message to chat.\n"
    "• Send an image to run OCR (if enabled).\n"
    "• Voice input is optional and may be disabled on this server.\n\n"
    "<b>Commands</b>\n"
    "/start – Status & hello\n"
    "/help – This help\n"
    "/reset – Clear context\n"
    "/mode jarvis|plain – Toggle persona\n"
    "/humane on|off – Gentle tone\n"
    "/unrestrict on|off – Looser style (still safe)\n"
    "/backend openai|ollama – Select provider\n"
    "/model <name> – Set model for current backend\n"
    "/ocr on|off – Enable/disable OCR\n"
    "/tts on|off – Enable/disable simple TTS logs\n"
    "/stats – Basic stats\n"
)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_typing(update, context)
    info = (
        f"<b>Backend:</b> {CFG['BACKEND']}<br>"
        f"<b>Model:</b> {CFG['OPENAI_MODEL'] if CFG['BACKEND']=='openai' else CFG['OLLAMA_MODEL']}<br>"
        f"<b>Jarvis:</b> {CFG['JARVIS_MODE']} | <b>Humane:</b> {CFG['HUMANE_TONE']} | "
        f"<b>Unrestrict:</b> {CFG['UNRESTRICTED_HINT']}<br>"
        f"<b>OCR:</b> {bool(CFG['ENABLE_OCR'] and pytesseract and Image)} | "
        f"<b>Speech:</b> {bool(CFG['ENABLE_SPEECH'] and sr)} | "
        f"<b>TTS:</b> {CFG['ENABLE_TTS']}"
    )
    await update.effective_message.reply_html("👋 Hey! I’m <b>Alex</b> (Jarvis mode capable).\n\n" + info + "\n\n" + HELP_TEXT)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_html(HELP_TEXT)

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mset(ctx_key(update.effective_chat.id), [])  # type: ignore
    await update.effective_message.reply_text("Context cleared.")

async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):  # type: ignore
        return await update.effective_message.reply_text("Only admins can change mode.")
    arg = (context.args[0].lower() if context.args else "")
    if arg in {"jarvis", "on"}:
        CFG["JARVIS_MODE"] = True
    elif arg in {"plain", "off"}:
        CFG["JARVIS_MODE"] = False
    else:
        return await update.effective_message.reply_text("Usage: /mode jarvis|plain")
    await update.effective_message.reply_text(f"Jarvis mode = {CFG['JARVIS_MODE']}")
    log.info(f"Jarvis mode -> {CFG['JARVIS_MODE']}")

async def cmd_humane(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):  # type: ignore
        return await update.effective_message.reply_text("Only admins can change tone.")
    arg = (context.args[0].lower() if context.args else "")
    if arg in {"on","true"}:
        CFG["HUMANE_TONE"] = True
    elif arg in {"off","false"}:
        CFG["HUMANE_TONE"] = False
    else:
        return await update.effective_message.reply_text("Usage: /humane on|off")
    await update.effective_message.reply_text(f"Humane tone = {CFG['HUMANE_TONE']}")

async def cmd_unrestrict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):  # type: ignore
        return await update.effective_message.reply_text("Only admins can change style.")
    arg = (context.args[0].lower() if context.args else "")
    if arg in {"on","true"}:
        CFG["UNRESTRICTED_HINT"] = True
    elif arg in {"off","false"}:
        CFG["UNRESTRICTED_HINT"] = False
    else:
        return await update.effective_message.reply_text("Usage: /unrestrict on|off")
    await update.effective_message.reply_text(f"Unrestricted style hint = {CFG['UNRESTRICTED_HINT']}")

async def cmd_backend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):  # type: ignore
        return await update.effective_message.reply_text("Only admins can change backend.")
    arg = (context.args[0].lower() if context.args else "")
    if arg not in {"openai","ollama"}:
        return await update.effective_message.reply_text("Usage: /backend openai|ollama")
    CFG["BACKEND"] = arg
    await update.effective_message.reply_text(f"Backend set to {CFG['BACKEND']}")

async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):  # type: ignore
        return await update.effective_message.reply_text("Only admins can change model.")
    if not context.args:
        return await update.effective_message.reply_text("Usage: /model <model_name>")
    name = " ".join(context.args)
    if CFG["BACKEND"] == "openai":
        CFG["OPENAI_MODEL"] = name
    else:
        CFG["OLLAMA_MODEL"] = name
    await update.effective_message.reply_text(f"Model set to {name} for {CFG['BACKEND']}")

async def cmd_ocr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):  # type: ignore
        return await update.effective_message.reply_text("Only admins can toggle OCR.")
    arg = (context.args[0].lower() if context.args else "")
    if arg in {"on","true"}:
        CFG["ENABLE_OCR"] = True
    elif arg in {"off","false"}:
        CFG["ENABLE_OCR"] = False
    else:
        return await update.effective_message.reply_text("Usage: /ocr on|off")
    await update.effective_message.reply_text(f"OCR = {CFG['ENABLE_OCR']}")

async def cmd_tts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):  # type: ignore
        return await update.effective_message.reply_text("Only admins can toggle TTS.")
    arg = (context.args[0].lower() if context.args else "")
    if arg in {"on","true"}:
        CFG["ENABLE_TTS"] = True
    elif arg in {"off","false"}:
        CFG["ENABLE_TTS"] = False
    else:
        return await update.effective_message.reply_text("Usage: /tts on|off")
    await update.effective_message.reply_text(f"TTS = {CFG['ENABLE_TTS']}")

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # lightweight stats
    messages = 0
    try:
        if CFG["PERSIST_SQLITE"]:
            conn = sqlite3.connect(DB); cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM messages"); messages = cur.fetchone()[0]
            conn.close()
    except Exception:
        pass
    await update.effective_message.reply_html(
        f"<b>Stats</b>\nMessages stored: <b>{messages}</b>\nBackend: <b>{CFG['BACKEND']}</b>")

# ---- message routing ----
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id  # type: ignore
    text = update.effective_message.text or ""  # type: ignore
    if not text.strip():
        return
    await send_typing(update, context)
    push_ctx(chat_id, "user", text)
    try:
        answer = alex_reply(str(chat_id), text, get_ctx(chat_id))
    except Exception as e:
        log.error(f"LLM error: {e}")
        answer = f"Oops, provider error: {e}"
    push_ctx(chat_id, "assistant", answer)
    await update.effective_message.reply_html(md_escape(answer))

async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not CFG["ENABLE_OCR"]:
        return await update.effective_message.reply_text("OCR disabled on this server.")
    if not (pytesseract and Image):
        return await update.effective_message.reply_text("OCR library not available here.")
    try:
        await send_typing(update, context)
        photo = update.message.photo[-1]  # type: ignore
        file = await photo.get_file()
        b = await file.download_as_bytearray()
        ocr_text = run_ocr(bytes(b))
        await update.effective_message.reply_text(f"OCR:\n{ocr_text[:4000]}")
        if ocr_text.strip():
            chat_id = update.effective_chat.id  # type: ignore
            push_ctx(chat_id, "user", f"(image OCR) {ocr_text[:2000]}")
            answer = alex_reply(str(chat_id), f"Here is text I extracted: {ocr_text[:2000]}", get_ctx(chat_id))
            push_ctx(chat_id, "assistant", answer)
            await update.effective_message.reply_html(md_escape(answer))
    except Exception as e:
        log.error(f"OCR fail: {e}")
        await update.effective_message.reply_text(f"OCR failed: {e}")

async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # We don't transcribe server-side unless ASR is installed and enabled.
    if not (CFG["ENABLE_SPEECH"] and sr):
        return await update.effective_message.reply_text("Voice input not available on this server.")
    await update.effective_message.reply_text("Voice recognition is not configured for Telegram uploads here yet.")

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.exception("Telegram error", exc_info=context.error)

# ---- builder ----
def build_app() -> Optional[Application]:
    if Update is None or not CFG["TELEGRAM_TOKEN"]:
        log.warning("Telegram not configured; skipping bot init.")
        return None
    app = ApplicationBuilder().token(CFG["TELEGRAM_TOKEN"]).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("mode", cmd_mode))
    app.add_handler(CommandHandler("humane", cmd_humane))
    app.add_handler(CommandHandler("unrestrict", cmd_unrestrict))
    app.add_handler(CommandHandler("backend", cmd_backend))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("ocr", cmd_ocr))
    app.add_handler(CommandHandler("tts", cmd_tts))
    app.add_handler(CommandHandler("stats", cmd_stats))

    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)

    return app
    # ========== Alex Mega.py (Part 3/4) ==========

# =====================
# OCR & IMAGE PROCESSING
# =====================
def extract_text_from_image(image_path: str) -> str:
    """Extract text from images using pytesseract OCR."""
    if not OCR_AVAILABLE:
        logger.warning("OCR not available. Please install tesseract.")
        return ""
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        logger.info(f"OCR extracted text: {text[:100]}...")
        return text
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle image uploads in Telegram and return OCR text."""
    try:
        file = await update.message.photo[-1].get_file()
        filepath = f"downloads/{file.file_id}.jpg"
        os.makedirs("downloads", exist_ok=True)
        await file.download_to_drive(filepath)
        text = extract_text_from_image(filepath)
        await update.message.reply_text(f"📝 OCR Result:\n\n{text if text else 'No text found.'}")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error processing image: {e}")

# =====================
# JARVIS / HUMANE TONE
# =====================
def format_response(text: str, jarvis: bool = JARVIS_MODE, humane: bool = HUMANE_TONE) -> str:
    """Format AI responses based on Jarvis or Humane tone flags."""
    if jarvis:
        return f"🤖 Jarvis: {text}"
    elif humane:
        return f"{text} 🙂"
    else:
        return text

# =====================
# TELEGRAM BOT HANDLERS
# =====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 Alex is online and ready!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Echo back messages with AI response style."""
    user_text = update.message.text
    ai_reply = format_response(f"You said: {user_text}")
    await update.message.reply_text(ai_reply)

async def toggle_jarvis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global JARVIS_MODE
    JARVIS_MODE = not JARVIS_MODE
    await update.message.reply_text(f"Jarvis mode is now {'ON' if JARVIS_MODE else 'OFF'}.")

async def toggle_humane(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global HUMANE_TONE
    HUMANE_TONE = not HUMANE_TONE
    await update.message.reply_text(f"Humane tone is now {'ON' if HUMANE_TONE else 'OFF'}.")

# =====================
# BACKEND CHAT FUNCTION
# =====================
async def generate_response(prompt: str) -> str:
    """Generate AI response using OpenAI backend with fallback."""
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            },
            timeout=30
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return format_response(text)
    except Exception as e:
        logger.error(f"OpenAI request failed: {e}")
        return format_response("⚠️ Sorry, I couldn't process that request.")
        # ========== Alex Mega.py (Part 4/4) ==========

# =====================
# TELEGRAM BOT SETUP
# =====================
def run_telegram_bot():
    if not TELEGRAM_TOKEN:
        logger.warning("No Telegram token set. Skipping Telegram bot startup.")
        return
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("jarvis", toggle_jarvis))
    application.add_handler(CommandHandler("humane", toggle_humane))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    logger.info("Starting Telegram bot...")
    application.run_polling()

# =====================
# FLASK / KEEP-ALIVE SERVER
# =====================
class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h1>✅ Alex is alive</h1>")

def run_keep_alive_server(port=8080):
    server = HTTPServer(("", port), SimpleHandler)
    logger.info(f"Keep-alive server running on port {port}")
    server.serve_forever()

# =====================
# MAIN ENTRYPOINT
# =====================
def main():
    logger.info("🚀 Starting Alex Mega.py...")
    threads = []

    # Start keep-alive HTTP server
    t_server = threading.Thread(target=run_keep_alive_server, daemon=True)
    threads.append(t_server)
    t_server.start()

    # Start Telegram bot
    t_bot = threading.Thread(target=run_telegram_bot, daemon=True)
    threads.append(t_bot)
    t_bot.start()

    # Optional: Background monitoring loop
    while True:
        logger.info("✅ Alex heartbeat...")
        time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 Alex shutting down...")