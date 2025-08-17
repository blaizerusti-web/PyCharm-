# ---------- Alex (All-in-One: Natural Chat + Live Web + Memory + Self-Learning + Voice + Images + SQLite) ----------
import os, sys, time, csv, json, threading, logging, tempfile, requests, re, asyncio, signal, sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from openai import OpenAI
from aiohttp import web

# ---------------- Logging ----------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
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
    # Prefer a mounted volume if provided by Railway
    candidates = [
        os.getenv("RAILWAY_VOLUME_MOUNT_PATH"),
        "/mnt/data",
        "/data",
        os.getcwd(),  # fallback to local
    ]
    for path in candidates:
        if not path:
            continue
        try:
            os.makedirs(path, exist_ok=True)
            testfile = os.path.join(path, ".rw_test")
            with open(testfile, "w") as f:
                f.write("ok")
            os.remove(testfile)
            return path
        except Exception:
            continue
    return os.getcwd()

DATA_DIR = select_data_dir()
log.info(f"📁 Data directory: {DATA_DIR}")

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
        _db.execute("INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

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
        _db.execute("INSERT INTO facts(text, created_utc) VALUES(?, ?)", (text, datetime.now(timezone.utc).isoformat()))

def db_add_note(text: str):
    with _db_lock, _db:
        _db.execute("INSERT INTO notes(text, created_utc) VALUES(?, ?)", (text, datetime.now(timezone.utc).isoformat()))

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
    # Try JSON
    if not os.path.exists(STATE_FILE):
        save_state(DEFAULT_STATE)
        # Also mirror to kv
        db_put_kv("persona", DEFAULT_PERSONA)
        db_put_kv("last_seen_row", "1")
        return DEFAULT_STATE.copy()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in DEFAULT_STATE.items():
            data.setdefault(k, v)
        # Mirror persona/last_seen_row into kv (best effort)
        db_put_kv("persona", data.get("persona", DEFAULT_PERSONA))
        db_put_kv("last_seen_row", str(data.get("last_seen_row", 1)))
        return data
    except Exception as e:
        log.error(f"Failed to read state file: {e}")
        # Fallback to kv if available
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
        # Mirror to kv
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
    except Exception as e:
        return f"❌ Search failed: {e}"

def fetch_news(topic: str = "technology") -> List[Dict[str, str]]:
    if not SERPAPI_KEY:
        return []
    try:
        res = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google_news", "q": topic, "api_key": SERPAPI_KEY},
            timeout=15,
        )
        data = res.json()
        stories = []
        for item in (data.get("news_results") or [])[:5]:
            stories.append({"title": item.get("title", ""), "link": item.get("link", "")})
        return stories
    except Exception as e:
        log.error(f"News fetch error: {e}")
        return []

# --------------- Heuristic: should we auto-enrich with web? (no prompt needed) ---------------
WEB_TRIGGERS = re.compile(
    r"\b(today|now|current|latest|breaking|price|stock|score|weather|news|update|live|"
    r"this week|this month|tonight|forecast|release|launched|announced|earnings|who won|"
    r"when is|schedule|deadline|trending|reddit|twitter|x\.com)\b",
    re.IGNORECASE,
)

def should_web_enrich(text: str) -> bool:
    if WEB_TRIGGERS.search(text):
        return True
    if "http://" in text or "https://" in text:
        return True
    if "?" in text and re.search(r"[A-Z][a-z]{2,}\s", text):
        return True
    return False

def auto_web_enrich(text: str) -> Optional[str]:
    if not SERPAPI_KEY:
        return None
    try:
        res = requests.get(
            "https://serpapi.com/search.json",
            params={"q": text, "api_key": SERPAPI_KEY, "num": 5},
            timeout=15,
        )
        j = res.json()
        items = (j.get("organic_results") or [])[:3]
        if not items:
            return None
        bullets = [f"- {it.get('title','(untitled)')} — {it.get('link','')}" for it in items]
        return "🌐 Web (auto):\n" + "\n".join(bullets)
    except Exception as e:
        log.warning(f"Auto-enrich failed: {e}")
        return None

# ---------------- Prompt construction ----------------
def build_system_prompt() -> str:
    with _state_lock:
        st = load_state()
        persona = st.get("persona", DEFAULT_PERSONA)
        facts = st.get("facts", [])[:12]
        notes = st.get("notes", [])[-12:]

    blocks = [
        persona,
        "Use the following long-term memory when relevant:",
        *[f"- {f}" for f in facts],
        "Recent distilled notes:",
        *[f"- {n}" for n in notes],
        "Style: warm, direct, practical; avoid filler; prefer short paragraphs and lists; ask clarifying only when critical."
    ]
    return "\n".join(blocks)

def gpt_reply(user_text: str) -> str:
    system_prompt = build_system_prompt()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.6,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"OpenAI error: {e}")
        return "I hit a snag talking to my brain. Try again in a moment."

# ---------------- Voice (on demand with “alex say …”) ----------------
async def tts_send(update: Update, text: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            speech = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=text[:4096],
            )
            tmp.write(speech.content)
            path = tmp.name
        with open(path, "rb") as f:
            await update.message.reply_voice(f)
        os.remove(path)
    except Exception as e:
        log.error(f"TTS error: {e}")
        await update.message.reply_text("Couldn't generate audio; sent text instead.")

# ---------------- Commands ----------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hey Blaize 👋 Alex is online — learning and evolving 24/7.")

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"✅ Alive. Uptime {get_uptime()}")

async def cmd_net(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with _state_lock:
        st = load_state()
    ok = "✅" if st.get("net_ok") else "❌"
    ms = st.get("last_net_ms")
    ts = st.get("last_net_check")
    await update.message.reply_text(f"{ok} Net: {ms if ms is not None else '-'} ms | last check: {ts or '-'}")

async def cmd_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("What should we think about? 🙂  Example: `/ai best laptop under 1k`")
        return
    q = " ".join(context.args)
    prefix = auto_web_enrich(q) if should_web_enrich(q) else None
    reply = gpt_reply(q)
    final = f"{prefix}\n\n{reply}" if prefix else reply
    await update.message.reply_text(final)
    u = update.message.from_user
    log_conversation(u.username or "Unknown", u.id, q, final)

async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with _state_lock:
        st = load_state()
    persona = st.get("persona", DEFAULT_PERSONA)
    facts = st.get("facts", [])
    notes = st.get("notes", [])[-10:]
    msg = (
        f"🧠 **Persona**:\n{persona}\n\n"
        f"📌 **Facts** ({len(facts)}):\n" + ("\n".join([f"- {f}" for f in facts[:12]]) or "—") + "\n\n"
        f"🗒️ **Recent Notes**:\n" + ("\n".join([f"- {n}" for n in notes]) or "—")
    )
    await update.message.reply_text(msg)

async def cmd_learn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Teach me something to remember, e.g. `/learn I prefer short bullet answers.`")
        return
    fact = " ".join(context.args).strip()
    with _state_lock:
        st = load_state()
        st.setdefault("facts", []).insert(0, fact)
        st["facts"] = st["facts"][:60]
        save_state(st)
    db_add_fact(fact)
    await update.message.reply_text("Saved to long-term memory ✅")

async def cmd_resetmemory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with _state_lock:
        st = load_state()
        st["facts"], st["notes"] = [], []
        st["persona"] = DEFAULT_PERSONA
        save_state(st)
    await update.message.reply_text("Memory and persona reset ✅")

async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = " ".join(context.args).strip() if context.args else "technology"
    items = fetch_news(topic)
    if not items:
        await update.message.reply_text("News lookup needs `SERPAPI_KEY` or found nothing.")
        return
    lines = [f"- {it['title']} — {it['link']}" for it in items]
    await update.message.reply_text("📰 Latest:\n" + "\n".join(lines))
    with _state_lock:
        st = load_state()
        st["last_news"] = items
        save_state(st)

# ✅ Image generation
async def cmd_imagine(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("🎨 Describe what you want me to imagine. Example:\n`/imagine a futuristic city skyline`")
        return
    prompt = " ".join(context.args).strip()
    await update.message.reply_text(f"✨ Creating image: {prompt}")
    try:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024"
        )
        url = result.data[0].url
        await update.message.reply_photo(photo=url, caption=f"🖼️ {prompt}")
    except Exception as e:
        log.error(f"Image generation error: {e}")
        await update.message.reply_text("❌ Couldn't generate the image.")

# Extra: quick stats
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with _db_lock:
        conv_count = _db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        facts_count = _db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        notes_count = _db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    await update.message.reply_text(f"📊 Stats — conversations: {conv_count}, facts: {facts_count}, notes: {notes_count}")

# ---------------- Natural free chat / routing ----------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    u = update.message.from_user
    username = u.username or "Unknown"

    # quick pings
    if text.lower() == "you there?":
        await update.message.reply_text(f"Always here 👊 (uptime {get_uptime()})")
        return

    # search (manual)
    if text.lower().startswith("search "):
        q = text[7:].strip()
        await update.message.reply_text(search_google(q))
        return

    # voice only on explicit trigger
    if text.lower().startswith("alex say "):
        phrase = text[9:].strip()
        if not phrase:
            await update.message.reply_text("What should I say?")
            return
        await update.message.reply_text(f"🎙️ Okay: {phrase}")
        await tts_send(update, phrase)
        return

    # auto web enrichment when useful (no prompt required)
    prefix = auto_web_enrich(text) if should_web_enrich(text) else None

    # normal, natural AI reply (with memory/persona)
    reply = gpt_reply(text)
    final = f"{prefix}\n\n{reply}" if prefix else reply
    await update.message.reply_text(final)
    log_conversation(username, u.id, text, final)

# ---------------- Error handler ----------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.error("Exception while handling update:", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("Something glitched, but I’m back.")

# ---------------- Self-Learning Utilities ----------------
def read_new_rows_since(idx_start: int) -> List[Dict[str, str]]:
    rows = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            rdr = list(csv.DictReader(f))
        for i, row in enumerate(rdr, start=1):
            if i >= idx_start:
                rows.append(row)
    except Exception as e:
        log.error(f"Read CSV error: {e}")
    return rows

def summarize_and_update_persona(notes_text: str, current_persona: str) -> Dict[str, Any]:
    try:
        prompt = (
            "You are maintaining a long-lived AI assistant called Alex.\n"
            "Given the recent conversation snippets below, first produce 3-6 concise bullet 'Notes' "
            "about user preferences, recurring topics, or helpful procedures (actionable, durable). "
            "Then propose up to two subtle improvements to Alex's persona (voice/tone/skills) "
            "that will make him more helpful for this user. Keep persona changes small and compatible.\n\n"
            f"Current persona:\n{current_persona}\n\n"
            f"Recent conversation snippets:\n{notes_text}\n"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()

        notes, new_persona = [], current_persona
        for line in text.splitlines():
            s = line.strip(" •-").strip()
            if not s or s.lower().startswith(("persona", "notes")):
                continue
            notes.append(s)

        if "\n\n" in text:
            chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
            if len(chunks) >= 2:
                new_persona = chunks[-1][:1200]

        return {"notes_texts": notes[:6], "persona": new_persona or current_persona}
    except Exception as e:
        log.error(f"Self-learning summary error: {e}")
        return {"notes_texts": [], "persona": current_persona}

# ---------------- Background workers (every 1 minute) ----------------
def self_learning_worker(interval_seconds: int = 60):
    while True:
        try:
            net_ping()

            with _state_lock:
                st = load_state()
                start_idx = int(st.get("last_seen_row", 1))

            rows = read_new_rows_since(start_idx)
            if rows:
                snippets = []
                for r in rows[-40:]:
                    snippets.append(f"User: {r['query']}\nAlex: {r['reply']}")
                corpus = "\n\n".join(snippets)[-8000:]

                with _state_lock:
                    persona_before = st.get("persona", DEFAULT_PERSONA)
                upd = summarize_and_update_persona(corpus, persona_before)

                with _state_lock:
                    st = load_state()
                    # prepend new notes then cap
                    new_notes = upd["notes_texts"]
                    for n in new_notes:
                        db_add_note(n)
                    st["notes"] = (new_notes + st.get("notes", []))[:80]
                    st["persona"] = upd["persona"][:1600]
                    st["last_seen_row"] = start_idx + len(rows)
                    st["last_update_iso"] = datetime.now(timezone.utc).isoformat()
                    save_state(st)
                log.info("🧠 Self-learning pass complete.")

            if SERPAPI_KEY:
                items = fetch_news("technology")
                if items:
                    with _state_lock:
                        st = load_state()
                        st["last_news"] = items
                        save_state(st)

        except Exception as e:
            log.error(f"Self-learning loop error: {e}")

        time.sleep(max(10, int(interval_seconds)))  # ~60s cadence

# ---------------- Tiny HTTP health server (Railway) ----------------
async def handle_health(request):
    return web.Response(text="ok")

async def run_health_server():
    app = web.Application()
    app.add_routes([web.get("/health", handle_health), web.get("/", handle_health)])
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", "8080"))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log.info(f"🌐 Health server running on :{port}")

# ---------------- Bot runner ----------------
def build_bot_app() -> Application:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("ping",   cmd_ping))
    app.add_handler(CommandHandler("net",    cmd_net))
    app.add_handler(CommandHandler("ai",     cmd_ai))
    app.add_handler(CommandHandler("imagine", cmd_imagine))   # images
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("learn",  cmd_learn))
    app.add_handler(CommandHandler("resetmemory", cmd_resetmemory))
    app.add_handler(CommandHandler("news",   cmd_news))
    app.add_handler(CommandHandler("stats",  cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    return app

def run_bot_polling():
    app = build_bot_app()
    log.info("🚀 Alex is running (Telegram polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

# ---------------- Main (with self-heal + background learning) ----------------
def main():
    # background learner
    t = threading.Thread(target=self_learning_worker, kwargs={"interval_seconds": 60}, daemon=True)
    t.start()

    # health server in asyncio loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def start_all():
        asyncio.create_task(run_health_server())
        # run Telegram polling (blocking) in a thread to coexist with aiohttp
        loop.run_in_executor(None, run_bot_polling)

    def handle_sigterm(signum, frame):
        log.info("🛑 Received shutdown signal — exiting.")
        os._exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    loop.run_until_complete(start_all())
    loop.run_forever()

if __name__ == "__main__":
    main()
