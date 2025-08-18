# =========================
# mega.py  — Alex (all-in-one, evolving memory)
# =========================
# Features:
# - Telegram bot: /start /uptime /ai /search /analyze /memory /persona /id
# - Unified memory: SQLite (alex_memory.db)
#   • conversations(user_text, alex_reply)
#   • links(url, title, summary, full_text, metadata_json)
#   • files(filename, rows, cols, preview, raw_json)
#   • persona(notes)
# - Self-learning: background worker builds evolution notes from recent data
# - URL crawler + SEO summarize (stores to memory)
# - File uploads: Excel quick stats (stores to memory)
# - Health HTTP server (PORT env, default 8080)
# - Live log watcher with /setlog /subscribe_logs /unsubscribe_logs /logs
# - Safe auto-install of missing dependencies

import os, sys, json, time, threading, logging, asyncio, subprocess, re, socket, sqlite3, textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------- bootstrap: install/import dependencies ----------
def _ensure(pkg_import: str, pip_name: str = None):
    try:
        return __import__(pkg_import)
    except ModuleNotFoundError:
        pip_name = pip_name or pkg_import
        print(f"[BOOT] Installing {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-input", "--quiet", pip_name])
        return __import__(pkg_import)

requests = _ensure("requests")
aiohttp  = _ensure("aiohttp")
bs4      = _ensure("bs4", "beautifulsoup4")
telegram = _ensure("telegram")
t_ext    = _ensure("telegram.ext", "python-telegram-bot==20.*")
openai_m = _ensure("openai")
try:
    pd = __import__("pandas")
except ModuleNotFoundError:
    pd = _ensure("pandas")
openpyxl = _ensure("openpyxl")

from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# OpenAI: support both new/old SDKs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
def _make_ai_chat():
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        def ai_chat(messages, model="gpt-4o-mini", max_tokens=600):
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
            return resp.choices[0].message.content.strip()
        return ai_chat
    except Exception:
        import openai as _openai
        _openai.api_key = OPENAI_API_KEY
        def ai_chat(messages, model="gpt-4o-mini", max_tokens=600):
            resp = _openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens)
            return resp["choices"][0]["message"]["content"].strip()
        return ai_chat

ai_chat = _make_ai_chat()

# ---------- config / logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
SERPAPI_KEY    = os.getenv("SERPAPI_KEY", "").strip()
PORT           = int(os.getenv("PORT", "8080"))

START_TIME = time.time()
SAVE_DIR = Path("received_files"); SAVE_DIR.mkdir(exist_ok=True)
DB_PATH = Path("alex_memory.db")

# ---------- SQLite unified memory ----------
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def db_init():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            user_text TEXT,
            alex_reply TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            summary TEXT,
            full_text TEXT,
            metadata_json TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            filename TEXT NOT NULL,
            rows INTEGER,
            cols INTEGER,
            preview TEXT,
            raw_json TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS persona (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            notes TEXT,
            updated_at TEXT
        );
    """)
    cur.execute("""
        INSERT OR IGNORE INTO persona (id, notes, updated_at)
        VALUES (1, '', datetime('now'));
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kv (
            k TEXT PRIMARY KEY,
            v TEXT
        );
    """)
    conn.commit()
    conn.close()

def db_execute(sql: str, params: Tuple = ()):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    conn.close()

def db_query(sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return rows

db_init()

def persona_get() -> str:
    r = db_query("SELECT notes FROM persona WHERE id=1 LIMIT 1")
    return (r[0]["notes"] if r else "").strip()

def persona_set(notes: str):
    db_execute("UPDATE persona SET notes=?, updated_at=datetime('now') WHERE id=1", (notes,))

# ---------- AI core ----------
async def ask_ai(prompt: str, context: str = "") -> str:
    if ai_chat is None:
        return "⚠️ OPENAI_API_KEY not set."
    try:
        sys_context = (
            "You are Alex — concise, helpful, and slightly witty. "
            "Leverage the persona notes if they help answer better.\n\n"
            f"Persona:\n{persona_get()}"
        )
        messages = [
            {"role": "system", "content": context or sys_context},
            {"role": "user", "content": prompt},
        ]
        return ai_chat(messages, model="gpt-4o-mini", max_tokens=700)
    except Exception as e:
        logging.exception("AI error")
        return f"(AI error: {e})"

# ---------- crawler ----------
async def fetch_url(url: str) -> Dict[str, Any]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"}) as r:
                status = r.status
                text = await r.text()
                if status != 200:
                    return {"ok": False, "error": f"HTTP {status}"}
    except Exception as e:
        return {"ok": False, "error": f"Crawl error: {e}"}

    soup = BeautifulSoup(text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
    desc_tag = soup.find("meta", {"name": "description"})
    desc = (desc_tag.get("content", "") if desc_tag else "")
    h1 = soup.h1.get_text(strip=True) if soup.h1 else ""
    words = len(soup.get_text(" ").split())
    links = len(soup.find_all("a"))
    images = len(soup.find_all("img"))
    full_text = soup.get_text(" ")
    # bound full_text length to keep DB sane
    full_text = full_text[:200_000]

    meta = {
        "description": desc,
        "h1": h1,
        "words": words,
        "link_count": links,
        "image_count": images,
        "ts": datetime.utcnow().isoformat()
    }
    return {
        "ok": True,
        "title": title,
        "full_text": full_text,
        "meta": meta
    }

async def analyze_url(url: str) -> str:
    data = await fetch_url(url)
    if not data.get("ok"):
        return f"⚠️ {data.get('error','Unknown crawl error')}"
    title = data["title"]
    full_text = data["full_text"]
    meta = data["meta"]

    # Summarize via AI
    summary_prompt = f"""Summarize this page in 8-12 bullet points.
Include: topic, key claims, entities, SEO opportunities, and 3 suggested actions.

TITLE: {title}
META: {json.dumps(meta)[:1200]}
CONTENT:
{full_text[:6000]}
"""
    summary = await ask_ai(summary_prompt)

    # Store into DB
    db_execute(
        "INSERT INTO links (ts, url, title, summary, full_text, metadata_json) VALUES (datetime('now'), ?, ?, ?, ?, ?)",
        (url, title, summary, full_text, json.dumps(meta))
    )

    return f"🌐 {title}\n\n{summary}"

# ---------- Excel analyzer (stored to DB) ----------
def analyze_excel_to_db(path: Path) -> str:
    try:
        df = pd.read_excel(path)
        rows, cols = df.shape
        head_cols = list(df.columns)[:12]
        preview_rows = df.head(8).to_dict(orient="records")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        profile = {}
        if numeric_cols:
            profile["numeric_summary"] = df[numeric_cols].describe(include="all").to_dict()

        raw = {
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            **profile
        }

        # Store
        db_execute(
            "INSERT INTO files (ts, filename, rows, cols, preview, raw_json) VALUES (datetime('now'), ?, ?, ?, ?, ?)",
            (path.name, rows, cols, json.dumps(preview_rows), json.dumps(raw))
        )

        summary_lines = [
            f"✅ Excel loaded: {rows} rows × {cols} cols",
            f"Columns: {', '.join(map(str, head_cols))}"
        ]
        if numeric_cols:
            summary_lines.append(f"Numeric columns: {', '.join(numeric_cols[:12])}")
        return "\n".join(summary_lines)
    except Exception as e:
        return f"⚠️ Excel analysis error: {e}"

# ---------- Trading log parsing (generic) ----------
TRADE_PATTERNS = [
    re.compile(r"\b(BUY|SELL)\b.*?(\b[A-Z]{2,10}\b).*?qty[:= ]?(\d+).*?price[:= ]?([0-9.]+)", re.I),
    re.compile(r"order\s+(buy|sell)\s+(\w+).+?@([0-9.]+).+?qty[:= ]?(\d+)", re.I),
]

def summarize_trade_line(line: str) -> str | None:
    for pat in TRADE_PATTERNS:
        m = pat.search(line)
        if m:
            g = [x for x in m.groups()]
            if len(g) == 4:
                side, sym, qty, price = g[0], g[1], g[2], g[3]
            else:
                continue
            try:
                qty_i = int(qty)
            except:
                qty_i = qty
            return f"🟢 {side.upper()} {sym} qty {qty_i} @ {price}"
    return None

# ---------- Telegram handlers ----------
GLOBAL_APP: Application | None = None

async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hey, I'm Alex 🤖 — evolving memory enabled.\n"
        "Commands: /uptime /ai /search /analyze /memory /persona /setlog /subscribe_logs /unsubscribe_logs /logs /id"
    )

async def id_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Your chat id: `{update.effective_chat.id}`", parse_mode="Markdown")

async def uptime_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = int(time.time() - START_TIME)
    h, m, s = u // 3600, (u % 3600) // 60, u % 60
    await update.message.reply_text(f"⏱️ Uptime {h}h {m}m {s}s")

async def ai_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = " ".join(ctx.args)
    if not q:
        return await update.message.reply_text("Usage: /ai <your question>")
    ans = await ask_ai(q)
    await update.message.reply_text(ans)
    # store to conversations
    db_execute("INSERT INTO conversations (ts, user_text, alex_reply) VALUES (datetime('now'), ?, ?)", (q, ans))

async def search_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /search <query>")
    if not SERPAPI_KEY:
        return await update.message.reply_text("⚠️ SERPAPI_KEY not set.")
    query = " ".join(ctx.args)
    try:
        r = requests.get("https://serpapi.com/search", params={"q": query, "hl": "en", "api_key": SERPAPI_KEY}, timeout=25)
        j = r.json()
        snip = j.get("organic_results", [{}])[0].get("snippet", "(no results)")
        await update.message.reply_text(f"🔎 {query}\n{snip}")
    except Exception as e:
        await update.message.reply_text(f"Search error: {e}")

async def analyze_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /analyze <url>")
    url = ctx.args[0]
    await update.message.reply_text("🔍 Crawling and summarizing…")
    res = await analyze_url(url)
    await update.message.reply_text(res)

async def memory_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Search unified memory and summarize matches."""
    if not ctx.args:
        return await update.message.reply_text("Usage: /memory <query>")
    q = " ".join(ctx.args).strip()
    like = f"%{q}%"

    convs = db_query("""
        SELECT ts, user_text, alex_reply
        FROM conversations
        WHERE (user_text LIKE ? OR alex_reply LIKE ?)
        ORDER BY id DESC LIMIT 20
    """, (like, like))

    links = db_query("""
        SELECT ts, url, title, summary
        FROM links
        WHERE (url LIKE ? OR title LIKE ? OR summary LIKE ? OR full_text LIKE ?)
        ORDER BY id DESC LIMIT 20
    """, (like, like, like, like))

    files = db_query("""
        SELECT ts, filename, rows, cols
        FROM files
        WHERE (filename LIKE ? OR preview LIKE ? OR raw_json LIKE ?)
        ORDER BY id DESC LIMIT 20
    """, (like, like, like))

    # Build a compact summary
    lines = []
    if convs:
        lines.append(f"🗣️ Conversations: {len(convs)} hits")
        for r in convs[:5]:
            ut = (r['user_text'] or "")[:120].replace("\n"," ")
            ar = (r['alex_reply'] or "")[:120].replace("\n"," ")
            lines.append(f"• [{r['ts']}] U: {ut} | A: {ar}")
    if links:
        lines.append(f"\n🔗 Links: {len(links)} hits")
        for r in links[:5]:
            t = (r['title'] or "(no title)")[:100]
            lines.append(f"• [{r['ts']}] {t} — {r['url']}")
    if files:
        lines.append(f"\n📂 Files: {len(files)} hits")
        for r in files[:5]:
            lines.append(f"• [{r['ts']}] {r['filename']} — {r['rows']}×{r['cols']}")

    if not lines:
        await update.message.reply_text("No matches in memory.")
        return

    # Optional: ask AI to synthesize a brief "what we know" from matches
    synthesis = ""
    try:
        if ai_chat is not None:
            prompt = "Summarize the key recurring themes and actionable insights from these memory hits:\n\n" + "\n".join(lines)
            synthesis = "\n\n🧠 Memory synthesis:\n" + ai_chat(
                [{"role":"system","content":"Be concise. 5-8 bullets."},
                 {"role":"user","content":prompt}],
                max_tokens=220
            )
    except Exception:
        pass

    out = "\n".join(lines)
    # Telegram message limit: split if needed
    chunk = out[:3500]
    await update.message.reply_text("```\n" + chunk + "\n```", parse_mode="Markdown")
    if synthesis:
        await update.message.reply_text(synthesis)

async def persona_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    notes = persona_get().strip() or "(empty)"
    await update.message.reply_text(f"🧬 Persona notes:\n\n{notes}")

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return

    # Quick wake-up phrase (kept for compatibility)
    if text.lower() == "analyse alex_profile":
        replies = [
            "Let's get back to it.", "Alright, I'm here — let's dive in.",
            "Back in the zone.", "Let's pick up where we left off.", "Okay, ready to roll."
        ]
        msg = replies[int(time.time()) % len(replies)]
        await update.message.reply_text(msg)
        # store
        db_execute("INSERT INTO conversations (ts, user_text, alex_reply) VALUES (datetime('now'), ?, ?)", (text, msg))
        return

    # Auto-URL detection -> analyze + store
    if text.startswith(("http://", "https://")):
        await update.message.reply_text("🔍 Got your link — analyzing & saving to memory…")
        res = await analyze_url(text)
        await update.message.reply_text(res)
        # also log the interaction
        db_execute("INSERT INTO conversations (ts, user_text, alex_reply) VALUES (datetime('now'), ?, ?)", (text, res))
        return

    # Default AI Q&A + store to conversations
    ans = await ask_ai(text)
    await update.message.reply_text(ans)
    db_execute("INSERT INTO conversations (ts, user_text, alex_reply) VALUES (datetime('now'), ?, ?)", (text, ans))

# --- file uploads (Excel) ---
SAVE_DIR = Path("received_files"); SAVE_DIR.mkdir(exist_ok=True)

async def handle_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc:
        return
    file_path = SAVE_DIR / doc.file_name
    tg_file = await ctx.bot.get_file(doc.file_id)
    await tg_file.download_to_drive(file_path)
    await update.message.reply_text(f"📂 Saved `{doc.file_name}` — analyzing & storing…", parse_mode="Markdown")

    if doc.file_name.lower().endswith(".xlsx"):
        out = analyze_excel_to_db(file_path)
        await update.message.reply_text(out)
        # log to conversations for traceability
        db_execute("INSERT INTO conversations (ts, user_text, alex_reply) VALUES (datetime('now'), ?, ?)",
                   (f"[uploaded file] {doc.file_name}", out))
    else:
        msg = "I currently analyze Excel (.xlsx). The file is saved."
        await update.message.reply_text(msg)
        db_execute("INSERT INTO conversations (ts, user_text, alex_reply) VALUES (datetime('now'), ?, ?)",
                   (f"[uploaded file] {doc.file_name}", msg))

# --- log watcher commands ---
async def setlog_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /setlog /path/to/your.log")
    path = " ".join(ctx.args)
    db_execute("INSERT OR REPLACE INTO kv (k, v) VALUES ('log_path', ?)", (path,))
    await update.message.reply_text(f"✅ Log path set to: `{path}`", parse_mode="Markdown")

async def subscribe_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    # Store subscribers as CSV in kv['subscribers']
    rows = db_query("SELECT v FROM kv WHERE k='subscribers'")
    cur = set((rows[0]["v"].split(",")) if rows and rows[0]["v"] else [])
    cur.add(chat_id)
    db_execute("INSERT OR REPLACE INTO kv (k, v) VALUES ('subscribers', ?)", (",".join(sorted(cur)),))
    await update.message.reply_text("🔔 Subscribed to live log updates.")

async def unsubscribe_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    rows = db_query("SELECT v FROM kv WHERE k='subscribers'")
    cur = set((rows[0]["v"].split(",")) if rows and rows[0]["v"] else [])
    if chat_id in cur:
        cur.remove(chat_id)
        db_execute("INSERT OR REPLACE INTO kv (k, v) VALUES ('subscribers', ?)", (",".join(sorted(cur)),))
    await update.message.reply_text("🔕 Unsubscribed from log updates.")

async def logs_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    n = 40
    if ctx.args:
        try:
            n = max(1, min(400, int(ctx.args[0])))
        except:
            pass
    rows = db_query("SELECT v FROM kv WHERE k='log_path'")
    path = rows[0]["v"] if rows else ""
    if not path or not Path(path).exists():
        return await update.message.reply_text("⚠️ No log path set or file does not exist. Use /setlog <path>.")
    try:
        lines = Path(path).read_text(errors="ignore").splitlines()[-n:]
        await update.message.reply_text("```\n" + "\n".join(lines)[-3500:] + "\n```", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Read error: {e}")

# ---------- log watcher thread ----------
GLOBAL_APP: Application | None = None

def _post_to_subscribers(text: str):
    rows = db_query("SELECT v FROM kv WHERE k='subscribers'")
    sub_str = rows[0]["v"] if rows else ""
    if not GLOBAL_APP or not sub_str:
        return
    subs = [s for s in sub_str.split(",") if s.strip()]
    if not subs:
        return
    loop = GLOBAL_APP.bot._application.loop  # PTB v20
    async def _send():
        for cid in subs:
            try:
                await GLOBAL_APP.bot.send_message(chat_id=int(cid), text=text)
            except Exception:
                pass
    try:
        asyncio.run_coroutine_threadsafe(_send(), loop)
    except Exception:
        pass

def watch_logs():
    last_size = 0
    last_raw_push = 0
    while True:
        try:
            rows = db_query("SELECT v FROM kv WHERE k='log_path'")
            path = rows[0]["v"] if rows else ""
            if not path or not Path(path).exists():
                time.sleep(2); continue
            p = Path(path)
            sz = p.stat().st_size
            if sz < last_size:
                last_size = 0  # rotation
            if sz > last_size:
                with p.open("r", errors="ignore") as f:
                    if last_size:
                        f.seek(last_size)
                    new = f.read()
                    last_size = sz
                for line in new.splitlines():
                    s = summarize_trade_line(line)
                    if s:
                        _post_to_subscribers(s)
                    now = time.time()
                    if now - last_raw_push > 15:
                        last_raw_push = now
                        _post_to_subscribers("📜 " + line[:900])
        except Exception as e:
            logging.error(f"log watcher error: {e}")
        time.sleep(1)

# ---------- self-learning background ----------
def learning_worker():
    """
    Periodically distill recent conversations + link summaries into persona notes.
    """
    while True:
        try:
            # Pull last items
            conv = db_query("SELECT user_text, alex_reply FROM conversations ORDER BY id DESC LIMIT 30")
            lnk  = db_query("SELECT title, summary FROM links ORDER BY id DESC LIMIT 12")
            # Build a compact corpus
            chunks = []
            if conv:
                chunks.append("Recent Conversations:\n" + "\n".join(
                    [f"U:{(c['user_text'] or '')[:180]} | A:{(c['alex_reply'] or '')[:180]}" for c in conv]
                ))
            if lnk:
                chunks.append("\nRecent Link Summaries:\n" + "\n".join(
                    [f"{(r['title'] or '')[:120]} — {(r['summary'] or '')[:200]}" for r in lnk]
                ))
            corpus = "\n".join(chunks)
            if corpus.strip() and ai_chat is not None:
                prompt = (
                    "From the following recent data, write 5–10 crisp 'persona evolution notes' "
                    "that will help Alex answer better in the future. Focus on preferences, recurring tasks, "
                    "entities, and any rules of thumb learned. Keep it compact.\n\n" + corpus[:9000]
                )
                notes = ai_chat(
                    [{"role":"system","content":"Return only the bullet list, no preamble."},
                     {"role":"user","content":prompt}],
                    max_tokens=250
                )
                # Merge with existing persona notes (keep unique-ish lines)
                existing = persona_get()
                merged = _merge_persona(existing, notes)
                persona_set(merged)
        except Exception as e:
            logging.error(f"learning error: {e}")
        time.sleep(60)

def _merge_persona(existing: str, new: str) -> str:
    def norm_lines(s: str) -> List[str]:
        out = []
        for line in s.splitlines():
            t = line.strip(" -*•\t")
            if t:
                out.append(t)
        return out
    ex = norm_lines(existing)
    nw = norm_lines(new)
    # dedupe, keep recent first
    seen = set()
    merged = []
    for line in nw + ex:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            merged.append("• " + line)
    return "\n".join(merged[:80])

# ---------- health server ----------
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type","text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

def start_health():
    server = HTTPServer(("0.0.0.0", PORT), Health)
    logging.info(f"Health server on 0.0.0.0:{PORT}")
    server.serve_forever()

# ---------- main ----------
def main():
    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN missing — exiting.")
        sys.exit(1)

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    global GLOBAL_APP
    GLOBAL_APP = app

    # commands
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("id", id_cmd))
    app.add_handler(CommandHandler("uptime", uptime_cmd))
    app.add_handler(CommandHandler("ai", ai_cmd))
    app.add_handler(CommandHandler("search", search_cmd))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("memory", memory_cmd))
    app.add_handler(CommandHandler("persona", persona_cmd))
    app.add_handler(CommandHandler("setlog", setlog_cmd))
    app.add_handler(CommandHandler("subscribe_logs", subscribe_cmd))
    app.add_handler(CommandHandler("unsubscribe_logs", unsubscribe_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))

    # messages & files
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # background threads
    threading.Thread(target=start_health, daemon=True).start()
    threading.Thread(target=learning_worker, daemon=True).start()
    threading.Thread(target=watch_logs, daemon=True).start()

    logging.info("Alex (mega) starting…")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()