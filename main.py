# =========================
# mega_alex.py  —  Alex (All-in-One, Railway/Koyeb-ready)
# =========================
# Features
# - Telegram bot: /start /uptime /ai /search /analyze /id
# - Paste a URL → auto-crawl + SEO summary
# - File uploads: Excel (.xlsx) → stats + summary (pandas/openpyxl)
# - “Infinite memory” via SQLite (convos, urls, files, notes)
# - Recall & export: /memory /recall <query> /export_memory
# - Self-learning persona notes
# - Live log watcher: /setlog /subscribe_logs /unsubscribe_logs /logs
# - Health HTTP server (PORT env, default 8080)
# - Safe auto-install of missing deps

import os, sys, json, time, threading, logging, asyncio, subprocess, re, sqlite3, io, zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------- bootstrap: install/import dependencies ----------
def _ensure(pkg_import: str, pip_name: str | None = None):
    try:
        return __import__(pkg_import)
    except ModuleNotFoundError:
        pip_name = pip_name or pkg_import
        print(f"[BOOT] Installing {pip_name} …", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_name])
        return __import__(pkg_import)

requests = _ensure("requests")
aiohttp  = _ensure("aiohttp")
bs4      = _ensure("bs4", "beautifulsoup4")
telegram = _ensure("telegram")
_ensure("telegram.ext", "python-telegram-bot==20.*")
_ensure("openpyxl")
try:
    pd = __import__("pandas")
except ModuleNotFoundError:
    pd = _ensure("pandas")

from bs4 import BeautifulSoup
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# OpenAI (support new or legacy SDK)
try:
    from openai import OpenAI
    _oa_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def ai_chat(messages, model="gpt-4o-mini", max_tokens=700):
        r = _oa_client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        return r.choices[0].message.content.strip()
except Exception:
    import openai as _openai_legacy
    _openai_legacy.api_key = os.getenv("OPENAI_API_KEY")
    def ai_chat(messages, model="gpt-4o-mini", max_tokens=700):
        r = _openai_legacy.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens)
        return r["choices"][0]["message"]["content"].strip()

# ---------- config / logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SERPAPI_KEY    = os.getenv("SERPAPI_KEY", "").strip()
PORT           = int(os.getenv("PORT", "8080"))

if not TELEGRAM_TOKEN:
    logging.warning("TELEGRAM_TOKEN not set — the bot will exit at startup.")

START_TIME = time.time()
SAVE_DIR = Path("received_files"); SAVE_DIR.mkdir(exist_ok=True)

# ---------- SQLite “infinite memory” ----------
DB = Path("alex.db")

def _db():
    conn = sqlite3.connect(DB, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = _db()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT, role TEXT, content TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS urls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT, url TEXT, title TEXT, summary TEXT, raw_stats TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT, filename TEXT, path TEXT, summary TEXT, meta TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE, value TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS subscribers (
        chat_id TEXT PRIMARY KEY
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS config (
        key TEXT PRIMARY KEY, value TEXT
    );""")
    conn.commit(); conn.close()

def db_exec(sql: str, params: Tuple | List | None = None):
    def _run():
        conn = _db()
        cur = conn.cursor()
        cur.execute(sql, params or [])
        conn.commit()
        conn.close()
    return asyncio.to_thread(_run)

def db_query(sql: str, params: Tuple | List | None = None):
    def _run():
        conn = _db()
        cur = conn.cursor()
        cur.execute(sql, params or [])
        rows = cur.fetchall()
        conn.close()
        return rows
    return asyncio.to_thread(_run)

async def mem_add_message(chat_id: int, role: str, content: str):
    await db_exec("INSERT INTO messages(chat_id,role,content) VALUES (?,?,?)",
                  (str(chat_id), role, content))

async def mem_add_url(chat_id: int, url: str, title: str, summary: str, raw_stats: str):
    await db_exec("INSERT INTO urls(chat_id,url,title,summary,raw_stats) VALUES (?,?,?,?,?)",
                  (str(chat_id), url, title, summary, raw_stats))

async def mem_add_file(chat_id: int, filename: str, path: str, summary: str, meta: str):
    await db_exec("INSERT INTO files(chat_id,filename,path,summary,meta) VALUES (?,?,?,?,?)",
                  (str(chat_id), filename, path, summary, meta))

async def mem_set_note(key: str, value: str):
    await db_exec("INSERT INTO notes(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, value))

# ---------- AI core ----------
async def ask_ai(prompt: str, context: str = "") -> str:
    if not OPENAI_API_KEY:
        return "⚠️ OPENAI_API_KEY not set."
    try:
        messages = [
            {"role": "system", "content": context or "You are Alex — concise, helpful, proactive, and a little witty."},
            {"role": "user",   "content": prompt}
        ]
        return ai_chat(messages, model="gpt-4o-mini", max_tokens=700)
    except Exception as e:
        logging.exception("AI error")
        return f"(AI error: {e})"

# ---------- crawler ----------
async def fetch_url(url: str) -> Dict[str, Any] | Dict[str, str]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"}) as r:
                if r.status != 200:
                    return {"error": f"HTTP {r.status}"}
                text = await r.text()
        soup = BeautifulSoup(text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
        desc_tag = soup.find("meta", {"name": "description"})
        desc = (desc_tag.get("content", "") if desc_tag else "")
        h1 = soup.h1.get_text(strip=True) if soup.h1 else ""
        words = len(soup.get_text(" ").split())
        links = len(soup.find_all("a"))
        images = len(soup.find_all("img"))
        snippet = soup.get_text(" ")[:2000]
        return {
            "title": title, "desc": desc, "h1": h1,
            "words": words, "links": links, "images": images,
            "snippet": snippet
        }
    except Exception as e:
        return {"error": f"Crawl error: {e}"}

async def analyze_url(url: str) -> Tuple[str, str]:
    data = await fetch_url(url)
    if "error" in data: 
        return data["error"], ""
    raw_stats = f"Words:{data['words']} Links:{data['links']} Images:{data['images']}"
    prompt = (
        f"Summarize the page in 6 bullets, then list SEO opportunities, key entities, and actions.\n\n"
        f"Title: {data['title']}\nDesc: {data['desc']}\nH1: {data['h1']}\n{raw_stats}\n\n"
        f"Snippet:\n{data['snippet']}"
    )
    summary = await ask_ai(prompt)
    return summary, raw_stats

# ---------- Excel analyzer ----------
def _analyze_excel_sync(file_path: Path) -> Tuple[str, Dict[str, Any]]:
    try:
        df = pd.read_excel(file_path)  # requires openpyxl (bootstrapped)
        head_cols = ", ".join([str(c) for c in list(df.columns)[:12]])
        info = f"✅ Excel loaded: {df.shape[0]} rows × {df.shape[1]} cols\nColumns: {head_cols}"
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        meta = {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "columns": [str(c) for c in df.columns]}
        if numeric_cols:
            stats = df[numeric_cols].describe().to_string()
            info += f"\n\nNumeric summary:\n{stats[:1800]}"
            meta["numeric_summary"] = stats[:5000]
        else:
            meta["numeric_summary"] = ""
        return info, meta
    except Exception as e:
        return f"⚠️ Excel analysis error: {e}", {}

async def analyze_excel(file_path: Path) -> Tuple[str, Dict[str, Any]]:
    return await asyncio.to_thread(_analyze_excel_sync, file_path)

# ---------- trading log parsing (example/generic) ----------
TRADE_PATTERNS = [
    re.compile(r"\b(BUY|SELL)\b.*?([A-Z]{1,10}).*?qty[:= ]?(\d+).*?price[:= ]?([0-9.]+)", re.I),
    re.compile(r"order\s+(buy|sell)\s+(\w+).+?@([0-9.]+).+?qty[:= ]?(\d+)", re.I),
]

def summarize_trade_line(line: str) -> str | None:
    for pat in TRADE_PATTERNS:
        m = pat.search(line)
        if m:
            g = m.groups()
            if len(g) == 4:
                side, sym, qty, price = g[0], g[1], g[2], g[3]
                try: qty = int(qty)
                except: pass
                return f"🟢 {side.upper()} {sym} qty {qty} @ {price}"
    return None

# ---------- Telegram handlers ----------
GLOBAL_APP: Application | None = None

async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hey, I'm Alex 🤖\n"
        "Commands:\n"
        "/uptime /ai /search /analyze /id\n"
        "/setlog /subscribe_logs /unsubscribe_logs /logs\n"
        "/memory /recall <q> /export_memory\n"
        "Send a URL to analyze, or upload an Excel file."
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
    await mem_add_message(update.effective_chat.id, "user", "/ai " + q)
    await mem_add_message(update.effective_chat.id, "assistant", ans)

async def search_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /search <query>")
    if not SERPAPI_KEY:
        return await update.message.reply_text("⚠️ SERPAPI_KEY not set.")
    query = " ".join(ctx.args)
    try:
        r = requests.get("https://serpapi.com/search",
                         params={"q": query, "hl": "en", "api_key": SERPAPI_KEY}, timeout=25)
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
    summary, raw_stats = await analyze_url(url)
    await update.message.reply_text(summary[:4000])
    await mem_add_url(update.effective_chat.id, url, "auto", summary, raw_stats)

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    chat_id = update.effective_chat.id

    if text.lower() == "analyse alex_profile":
        replies = ["Let's get back to it.","Alright, I'm here — let's dive in.","Back in the zone.","Picking up where we left off.","Ready to roll."]
        msg = replies[int(time.time()) % len(replies)]
        await update.message.reply_text(msg)
        await mem_add_message(chat_id, "system", "wake")
        return

    if text.startswith("http://") or text.startswith("https://"):
        await update.message.reply_text("🔍 Got your link — analyzing…")
        summary, raw_stats = await analyze_url(text)
        await update.message.reply_text(summary[:4000])
        await mem_add_url(chat_id, text, "auto", summary, raw_stats)
    else:
        ans = await ask_ai(text)
        await update.message.reply_text(ans)
        await mem_add_message(chat_id, "user", text)
        await mem_add_message(chat_id, "assistant", ans)

# --- file uploads (Excel) ---
async def handle_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc: return
    file_path = SAVE_DIR / doc.file_name
    tg_file = await ctx.bot.get_file(doc.file_id)
    await tg_file.download_to_drive(file_path)
    await update.message.reply_text(f"📂 Saved `{doc.file_name}` — analyzing…", parse_mode="Markdown")

    if doc.file_name.lower().endswith(".xlsx"):
        out, meta = await analyze_excel(file_path)
        await update.message.reply_text(out[:4000])
        await mem_add_file(update.effective_chat.id, doc.file_name, str(file_path), out, json.dumps(meta))
    else:
        msg = "I currently analyze Excel (.xlsx). The file is saved."
        await update.message.reply_text(msg)
        await mem_add_file(update.effective_chat.id, doc.file_name, str(file_path), msg, "{}")

# --- memory/recall/export ---
async def memory_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    stats_msgs = await db_query("SELECT COUNT(*) FROM messages WHERE chat_id=?", (str(update.effective_chat.id),))
    stats_urls = await db_query("SELECT COUNT(*) FROM urls WHERE chat_id=?", (str(update.effective_chat.id),))
    stats_files = await db_query("SELECT COUNT(*) FROM files WHERE chat_id=?", (str(update.effective_chat.id),))
    persona = await db_query("SELECT value FROM notes WHERE key='persona_notes'")
    pm = persona[0][0] if persona else "(none yet)"
    await update.message.reply_text(
        f"🧠 Memory stats\n"
        f"- Messages: {stats_msgs[0][0]}\n"
        f"- URLs: {stats_urls[0][0]}\n"
        f"- Files: {stats_files[0][0]}\n\n"
        f"Persona notes:\n{pm[:1500]}"
    )

async def recall_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = " ".join(ctx.args) if ctx.args else ""
    if not q:
        return await update.message.reply_text("Usage: /recall <query>")
    # naive LIKE search across memory
    cid = str(update.effective_chat.id)
    rows1 = await db_query("SELECT 'message', content, ts FROM messages WHERE chat_id=? AND content LIKE ? ORDER BY ts DESC LIMIT 5", (cid, f"%{q}%"))
    rows2 = await db_query("SELECT 'url', url || ' — ' || substr(summary,1,120), ts FROM urls WHERE chat_id=? AND (url LIKE ? OR summary LIKE ?) ORDER BY ts DESC LIMIT 5", (cid, f"%{q}%", f"%{q}%"))
    rows3 = await db_query("SELECT 'file', filename || ' — ' || substr(summary,1,120), ts FROM files WHERE chat_id=? AND (filename LIKE ? OR summary LIKE ?) ORDER BY ts DESC LIMIT 5", (cid, f"%{q}%", f"%{q}%"))
    lines = []
    for tag, txt, ts in rows1 + rows2 + rows3:
        lines.append(f"[{tag}] {ts}  {txt}")
    if not lines:
        return await update.message.reply_text("No matches.")
    await update.message.reply_text("🔎 Recall results:\n" + "\n".join(lines)[:3800])

async def export_memory_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = str(update.effective_chat.id)
    messages = await db_query("SELECT role, content, ts FROM messages WHERE chat_id=? ORDER BY ts", (cid,))
    urls = await db_query("SELECT url, title, summary, raw_stats, ts FROM urls WHERE chat_id=? ORDER BY ts", (cid,))
    files = await db_query("SELECT filename, path, summary, meta, ts FROM files WHERE chat_id=? ORDER BY ts", (cid,))
    data = {"messages": messages, "urls": urls, "files": files}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("memory.json", json.dumps(data, indent=2))
    buf.seek(0)
    await update.message.reply_document(InputFile(buf, filename="alex_memory.zip"), caption="📦 Your memory export")

# --- log watcher commands + thread ---
async def setlog_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /setlog /path/to/your.log")
    path = " ".join(ctx.args)
    await db_exec("INSERT INTO config(key,value) VALUES('log_path',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (path,))
    await update.message.reply_text(f"✅ Log path set: `{path}`", parse_mode="Markdown")

async def subscribe_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await db_exec("INSERT OR IGNORE INTO subscribers(chat_id) VALUES (?)", (str(update.effective_chat.id),))
    await update.message.reply_text("🔔 Subscribed to live log updates.")

async def unsubscribe_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await db_exec("DELETE FROM subscribers WHERE chat_id=?", (str(update.effective_chat.id),))
    await update.message.reply_text("🔕 Unsubscribed from log updates.")

async def logs_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    n = 40
    if ctx.args:
        try: n = max(1, min(400, int(ctx.args[0])))
        except: pass
    cfg = (await db_query("SELECT value FROM config WHERE key='log_path'")) or []
    if not cfg: return await update.message.reply_text("⚠️ No log path set. Use /setlog <path>.")
    path = cfg[0][0]
    p = Path(path)
    if not p.exists(): return await update.message.reply_text("⚠️ Log file not found.")
    try:
        lines = p.read_text(errors="ignore").splitlines()[-n:]
        await update.message.reply_text("```\n" + "\n".join(lines)[-3500:] + "\n```", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Read error: {e}")

def _post_to_subscribers(app: Application, text: str):
    async def _send():
        rows = await db_query("SELECT chat_id FROM subscribers")
        for (cid,) in rows:
            try:
                await app.bot.send_message(chat_id=int(cid), text=text)
            except Exception:
                pass
    asyncio.run_coroutine_threadsafe(_send(), app.bot._application.loop)

def watch_logs(app: Application):
    last_size = 0
    last_raw_push = 0
    while True:
        try:
            cfg = sqlite3.connect(DB).cursor().execute("SELECT value FROM config WHERE key='log_path'").fetchone()
            if not cfg:
                time.sleep(2); continue
            path = cfg[0]
            p = Path(path)
            if not p.exists():
                time.sleep(2); continue
            sz = p.stat().st_size
            if sz < last_size: last_size = 0
            if sz > last_size:
                with p.open("r", errors="ignore") as f:
                    if last_size: f.seek(last_size)
                    new = f.read()
                    last_size = sz
                for line in new.splitlines():
                    s = summarize_trade_line(line)
                    if s: _post_to_subscribers(app, s)
                    now = time.time()
                    if now - last_raw_push > 15:
                        last_raw_push = now
                        _post_to_subscribers(app, "📜 " + line[:900])
        except Exception as e:
            logging.error(f"log watcher error: {e}")
        time.sleep(1)

# ---------- self-learning persona worker ----------
def learning_worker():
    while True:
        try:
            conn = _db(); cur = conn.cursor()
            cur.execute("SELECT content FROM messages WHERE role='user' ORDER BY id DESC LIMIT 20")
            recent = " ".join([r[0] for r in cur.fetchall()][::-1])
            conn.close()
            if recent:
                note = ai_chat(
                    [
                        {"role":"system","content":"From these recent user lines, write 1–2 short bullet persona notes (preferences, tone, recurring goals)."},
                        {"role":"user","content":recent}
                    ],
                    model="gpt-4o-mini", max_tokens=160
                )
                sqlite3.connect(DB).cursor().execute(
                    "INSERT INTO notes(key,value) VALUES('persona_notes',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (note,)
                ).connection.commit()
        except Exception as e:
            logging.error(f"learning error: {e}")
        time.sleep(60)

# ---------- health server ----------
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type","text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

def start_health():
    httpd = HTTPServer(("0.0.0.0", PORT), Health)
    logging.info(f"Health server on 0.0.0.0:{PORT}")
    httpd.serve_forever()

# ---------- main ----------
def main():
    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN missing — exiting.")
        sys.exit(1)

    init_db()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # commands
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("id", id_cmd))
    app.add_handler(CommandHandler("uptime", uptime_cmd))
    app.add_handler(CommandHandler("ai", ai_cmd))
    app.add_handler(CommandHandler("search", search_cmd))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("memory", memory_cmd))
    app.add_handler(CommandHandler("recall", recall_cmd))
    app.add_handler(CommandHandler("export_memory", export_memory_cmd))
    app.add_handler(CommandHandler("setlog", setlog_cmd))
    app.add_handler(CommandHandler("subscribe_logs", subscribe_cmd))
    app.add_handler(CommandHandler("unsubscribe_logs", unsubscribe_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))

    # messages
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # background services
    threading.Thread(target=start_health, daemon=True).start()
    threading.Thread(target=learning_worker, daemon=True).start()
    threading.Thread(target=watch_logs, args=(app,), daemon=True).start()

    logging.info("Alex mega bot starting…")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()