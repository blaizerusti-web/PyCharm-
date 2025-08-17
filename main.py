# ---------- main.py | Alex (All-in-One: Natural Chat + Live Web + Memory + Self-Learning + Voice + Images + SQLite) ----------
# Single-file, Railway-ready. Persistent storage, health server, Telegram bot (PTB v20+), OpenAI (chat+TTS+images),
# SerpAPI enrichment, self-learning background worker, and simple config (startup visit counter).

import os, sys, time, csv, json, threading, logging, tempfile, requests, re, asyncio, signal, sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from openai import OpenAI
from aiohttp import web

# ---------------- Logging ----------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("alex")

# ---------------- Keys / Client ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY    = os.getenv("SERPAPI_KEY")  # optional for web search/news

if not TELEGRAM_TOKEN:
    sys.exit("❌ TELEGRAM_TOKEN not set in environment!")
if not OPENAI_KEY:
    sys.exit("❌ OPENAI_API_KEY not set in environment!")

client = OpenAI(api_key=OPENAI_KEY)

# ---------------- Data directory (Railway-friendly) ----------------
def select_data_dir() -> str:
    candidates = [
        os.getenv("RAILWAY_VOLUME_MOUNT_PATH"),  # preferred if present
        "/mnt/data",
        "/data",
        os.getcwd(),  # fallback
    ]
    for path in candidates:
        if not path:
            continue
        try:
            os.makedirs(path, exist_ok=True)
            testfile = os.path.join(path, ".rw_test")
            with open(testfile, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(testfile)
            return path
        except Exception as e:
            log.warning(f"Data dir candidate failed ({path}): {e}")
            continue
    return os.getcwd()

DATA_DIR = select_data_dir()
log.info(f"📁 Data directory: {DATA_DIR}")

# ---------------- Config (simple persistent JSON) ----------------
CONFIG_FILE = os.path.join(DATA_DIR, "alex_config.json")
def load_config() -> Dict[str, Any]:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"Config load issue: {e}")
    return {}

def save_config(cfg: Dict[str, Any]) -> None:
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.error(f"Config save issue: {e}")

config = load_config()
config["visits"] = int(config.get("visits", 0)) + 1
save_config(config)
log.info(f"✅ Alex has been started {config['visits']} times (persistent).")

# ---------------- Files (persisted) ----------------
LOG_FILE   = os.path.join(DATA_DIR, "ai_conversations.csv")   # conversation transcript
STATE_FILE = os.path.join(DATA_DIR, "alex_state.json")        # persona + long-term memory
DB_FILE    = os.path.join(DATA_DIR, "alex.db")                # SQLite mirror

# init CSV
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp_utc", "username", "user_id", "query", "reply"])

# ---------------- SQLite setup ----------------
def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

_db = db_connect()
_db_lock = threading.Lock()

def db_init():
    with _db_lock, _db:
        _db.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT,
            username TEXT,
            user_id INTEGER,
            query TEXT,
            reply TEXT
        )""")
        _db.execute("""
        CREATE TABLE IF NOT EXISTS kv (
            key TEXT PRIMARY KEY,
            value TEXT
        )""")
        _db.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            created_utc TEXT
        )""")
        _db.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            created_utc TEXT
        )""")

db_init()

def db_put_kv(key: str, value: str):
    with _db_lock, _db:
        _db.execute(
            "INSERT INTO kv(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value)
        )

def db_get_kv(key: str, default: Optional[str] = None) -> Optional[str]:
    with _db_lock:
        cur = _db.execute("SELECT value FROM kv WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else default

def db_add_convo(ts: str, username: str, user_id: int, query: str, reply: str):
    with _db_lock, _db:
        _db.execute(
            "INSERT INTO conversations(timestamp_utc, username, user_id, query, reply) VALUES(?,?,?,?,?)",
            (ts, username, user_id, query, reply)
        )

def db_add_fact(text: str):
    with _db_lock, _db:
        _db.execute("INSERT INTO facts(text, created_utc) VALUES(?, ?)",
                    (text, datetime.now(timezone.utc).isoformat()))

def db_add_note(text: str):
    with _db_lock, _db:
        _db.execute("INSERT INTO notes(text, created_utc) VALUES(?, ?)",
                    (text, datetime.now(timezone.utc).isoformat()))

# ---------------- State (JSON + KV mirror) ----------------
DEFAULT_PERSONA = (
    "You are Alex: friendly, concise, curious, and helpful. "
    "You answer naturally (like a smart friend), cite facts when relevant, "
    "admit uncertainty, and prefer actionable steps. You avoid fluff."
)
DEFAULT_STATE: Dict[str, Any] = {
    "persona": DEFAULT_PERSONA,
    "facts": [],          # enduring facts about user/preferences/workflows
    "notes": [],          # distilled takeaways from chats
    "last_news": [],      # cached recent news items
    "last_update_iso": None,
    "last_seen_row": 1,   # first data row in CSV is index 1
    "net_ok": False,
    "last_net_check": None,
    "last_net_ms": None
}

_state_lock = threading.Lock()

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        save_state(DEFAULT_STATE)
        db_put_kv("persona", DEFAULT_PERSONA)
        db_put_kv("last_seen_row", "1")
        return DEFAULT_STATE.copy()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in DEFAULT_STATE.items():
            data.setdefault(k, v)
        db_put_kv("persona", data.get("persona", DEFAULT_PERSONA))
        db_put_kv("last_seen_row", str(data.get("last_seen_row", 1)))
        return data
    except Exception as e:
        log.error(f"Failed to read state file: {e}")
        persona = db_get_kv("persona", DEFAULT_PERSONA)
        last_seen_row = int(db_get_kv("last_seen_row", "1"))
        data = DEFAULT_STATE.copy()
        data["persona"] = persona
        data["last_seen_row"] = last_seen_row
        return data

def save_state(data: Dict[str, Any]) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        db_put_kv("persona", data.get("persona", DEFAULT_PERSONA))
        db_put_kv("last_seen_row", str(data.get("last_seen_row", 1)))
    except Exception as e:
        log.error(f"Failed to write state file: {e}")

# ---------------- Uptime ----------------
start_time = time.time()
def get_uptime():
    s = int(time.time() - start_time)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s"

# ---------------- Conversation logging ----------------
def log_conversation(username: str, user_id: int, query: str, reply: str):
    ts = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, username, user_id, query, reply])
    db_add_convo(ts, username, user_id, query, reply)

# ---------------- Live connectivity ping (no prompt needed) ----------------
def net_ping() -> Dict[str, Optional[Any]]:
    ok, ms = False, None
    started = time.time()
    try:
        r = requests.get("https://www.google.com/generate_204", timeout=6)
        ok = (200 <= r.status_code < 400)
        ms = int((time.time() - started) * 1000)
    except Exception:
        ok = False
        ms = None

    with _state_lock:
        st = load_state()
        st["net_ok"] = bool(ok)
        st["last_net_check"] = datetime.now(timezone.utc).isoformat()
        st["last_net_ms"] = ms
        save_state(st)
    return {"ok": ok, "ms": ms}

# ---------------- Web Search / News (SerpAPI) ----------------
def search_google(query: str) -> str:
    if not SERPAPI_KEY:
        return "⚠️ Web search isn't enabled yet (missing SERPAPI_KEY)."
    try:
        res = requests.get(
            "https://serpapi.com/search.json",
            params={"q": query, "api_key": SERPAPI_KEY},
            timeout=15,
        )
        data = res.json()
        items = (data.get("organic_results") or [])[:5]
        if not items:
            return "No strong results found."
        lines = [f"- {it.get('title','(untitled)')} — {it.get('link','')}" for it in items]
        return "🔎 Top results:\n" + "\n".join(lines)
   
