# ==========================
# mega.py — Alex (Evolving Mind, Single-File, Railway-Ready)
# ==========================
# Highlights
# - Telegram bot (/start /uptime /ai /analyze /search /id)
# - Unified persistent memory in SQLite + FTS5 (messages, links, files, insights)
# - Context recall before every reply (retrieves top memories with BM25 ranking)
# - Background reflection loop to compress experiences into insights
# - Link crawler + AI summarizer (auto on pasted links or /analyze <url>)
# - Excel ingest (quick stats) + saved to memory
# - Live log streaming (/setlog /subscribe_logs /unsubscribe_logs /logs)
# - Health HTTP server on PORT (default 8080)
# - Auto-install minimal deps (python-telegram-bot, aiohttp, bs4, pandas, openpyxl, requests, openai)
#
# Env:
# TELEGRAM_TOKEN=<...>
# OPENAI_API_KEY=<...>          (optional for AI; without it Alex still runs but replies simply)
# SERPAPI_KEY=<...>             (optional, for /search)
# PORT=8080

import os, sys, json, time, threading, asyncio, logging, subprocess, re, math, hashlib
from pathlib import Path
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

# ---------------- bootstrap: install/import ----------------
def _ensure(pkg_import: str, pip_name: str | None = None):
    try:
        return __import__(pkg_import)
    except ModuleNotFoundError:
        pip_name = pip_name or pkg_import
        print(f"[BOOT] Installing {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_name])
        return __import__(pkg_import)

requests = _ensure("requests")
aiohttp  = _ensure("aiohttp")
bs4      = _ensure("bs4", "beautifulsoup4")
sqlite3  = _ensure("sqlite3")
telegram = _ensure("telegram")
t_ext    = _ensure("telegram.ext", "python-telegram-bot==20.*")
openpyxl = _ensure("openpyxl")
try:
    pd = __import__("pandas")
except ModuleNotFoundError:
    pd = _ensure("pandas")

from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# OpenAI graceful support (new/old SDK)
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip())
    def ai_chat(messages, model="gpt-4o-mini", max_tokens=700) -> str:
        if not os.getenv("OPENAI_API_KEY"): return "(no OPENAI_API_KEY set)"
        resp = _openai_client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        return resp.choices[0].message.content.strip()
except Exception:
    try:
        import openai as _openai
        _openai.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        def ai_chat(messages, model="gpt-4o-mini", max_tokens=700) -> str:
            if not _openai.api_key: return "(no OPENAI_API_KEY set)"
            resp = _openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens)
            return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        def ai_chat(messages, model="gpt-4o-mini", max_tokens=700) -> str:
            return "(OpenAI SDK not available)"

# ---------------- config/logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SERPAPI_KEY    = os.getenv("SERPAPI_KEY", "").strip()
PORT           = int(os.getenv("PORT", "8080"))

START_TIME = time.time()
SAVE_DIR = Path("received_files"); SAVE_DIR.mkdir(exist_ok=True)

# ---------------- SQLite memory (FTS5) ----------------
DB_PATH = Path("alex_memory.db")

def db_init():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        mtype TEXT NOT NULL,            -- 'chat','link','file','insight','system','log'
        role TEXT,                      -- 'user','assistant', or source label
        title TEXT,
        content TEXT NOT NULL,
        meta TEXT                       -- json
    )
    """)
    # FTS (content mirrored from memories)
    conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        content,
        content='memories',
        content_rowid='id'
    )
    """)
    # triggers keep FTS synced
    conn.execute("""
    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
    END;
    """)
    conn.execute("""
    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
    END;
    """)
    conn.execute("""
    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts) VALUES('delete-all');
        INSERT INTO memories_fts(rowid, content) SELECT id, content FROM memories;
    END;
    """)
    conn.commit()
    conn.close()

def _ts() -> str:
    return datetime.utcnow().isoformat()

def insert_memory(mtype: str, content: str, role: str | None = None, title: str | None = None, meta: Dict[str, Any] | None = None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO memories(ts,mtype,role,title,content,meta) VALUES(?,?,?,?,?,?)",
                 (_ts(), mtype, role, title, content, json.dumps(meta or {})))
    conn.commit()
    conn.close()

def recall(query: str, k: int = 6) -> List[Tuple[int,str,str,str]]:
    """
    Returns list of (id, mtype, title, content) for most relevant memories.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("""
        SELECT m.id, m.mtype, COALESCE(m.title,''), m.content
        FROM memories_fts f
        JOIN memories m ON m.id = f.rowid
        WHERE memories_fts MATCH ?
        ORDER BY bm25(f) LIMIT ?;
    """, (query, k))
    rows = cur.fetchall()
    conn.close()
    return rows

def recent(k: int = 10) -> List[Tuple[int,str,str,str]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT id,mtype,COALESCE(title,''),content FROM memories ORDER BY id DESC LIMIT ?", (k,))
    rows = cur.fetchall()
    conn.close()
    return rows

def mem_stats() -> Dict[str,int]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT mtype, COUNT(*) FROM memories GROUP BY mtype;")
    d = {t:c for t,c in cur.fetchall()}
    conn.close()
    return d

db_init()

# ---------------- AI helpers ----------------
BASE_SYSTEM = (
    "You are Alex — Blaize's always-on assistant. Be concise, helpful, proactive, "
    "and use retrieved memories to personalize your answers. If a link is provided, analyze it."
)

async def ask_ai_async(prompt: str, context_blurb: str = "") -> str:
    msgs = [{"role":"system","content": BASE_SYSTEM}]
    if context_blurb:
        msgs.append({"role":"system","content": f"Relevant memories:\n{context_blurb}"})
    msgs.append({"role":"user","content": prompt})
    return ai_chat(msgs, max_tokens=700)

def build_context(user_text: str) -> str:
    """
    Query FTS with user_text + key tokens. Merge top memories into a blurb.
    """
    # query variations
    q = user_text
    rows = recall(q, k=6)
    parts = []
    for _id, mtype, title, content in rows:
        label = f"[{mtype}] {title}".strip()
        parts.append(f"- {label}\n{content[:600]}")
    return "\n".join(parts)

# ---------------- crawler / URL analyze ----------------
async def fetch_url(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=30, headers={"User-Agent":"Mozilla/5.0"}) as r:
                if r.status != 200:
                    return f"⚠️ HTTP {r.status}"
                text = await r.text()
        soup = BeautifulSoup(text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
        desc_tag = soup.find("meta", {"name":"description"}) or soup.find("meta", {"property":"og:description"})
        desc = (desc_tag.get("content","") if desc_tag else "")[:240]
        h1 = soup.h1.get_text(strip=True) if soup.h1 else ""
        words = len(soup.get_text(" ").split())
        links = len(soup.find_all("a"))
        images = len(soup.find_all("img"))
        snippet = soup.get_text(" ")[:1800]
        basic = (f"🌐 {title}\nDesc: {desc}\nH1: {h1}\n"
                 f"Words:{words} Links:{links} Images:{images}\n\nSnippet:\n{snippet}")
        return basic
    except Exception as e:
        return f"⚠️ Crawl error: {e}"

async def analyze_url(url: str) -> str:
    content = await fetch_url(url)
    insert_memory("link", content, role="system", title=url, meta={"url":url})
    if content.startswith("⚠️"): 
        return content
    summary = await ask_ai_async(f"Summarize, extract key insights & to-dos:\n\n{content}")
    insert_memory("insight", summary, role="assistant", title=f"Summary of {url}", meta={"url":url})
    return summary

# ---------------- Excel ingest ----------------
def summarize_excel(path: Path) -> str:
    try:
        df = pd.read_excel(path)
        cols = list(df.columns)
        head_cols = ", ".join(map(str, cols[:10]))
        info = f"✅ Excel loaded: {df.shape[0]} rows × {df.shape[1]} cols\nColumns: {head_cols}"
        # profile numerics
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            desc = df[numeric_cols].describe().to_string()
            info += f"\n\nNumeric summary:\n{desc[:1800]}"
        # store memory
        sample = df.head(10).to_csv(index=False)
        insert_memory("file", f"Excel: {path.name}\n{info}\n\nSample (first 10 rows CSV):\n{sample}",
                      role="user", title=path.name, meta={"path": str(path)})
        # optional AI brief
        insight = ai_chat([
            {"role":"system","content":BASE_SYSTEM},
            {"role":"user","content": f"Here are table stats and sample rows. Give me 5 bullets of actionable insights:\n\n{info}\n\n{sample}"}
        ], max_tokens=300)
        insert_memory("insight", f"Insights for {path.name}\n{insight}", role="assistant", title=f"Excel insights: {path.name}")
        return info
    except Exception as e:
        return f"⚠️ Excel analysis error: {e}"

# ---------------- Trading/log watcher ----------------
TRADE_PATTERNS = [
    re.compile(r"\b(BUY|SELL)\b.*?([A-Z]{1,10}).*?qty[:= ]?(\d+).*?price[:= ]?([0-9.]+)", re.I),
    re.compile(r"order\s+(buy|sell)\s+(\w+).+?@([0-9.]+).+?qty[:= ]?(\d+)", re.I),
]

def summarize_trade_line(line: str) -> Optional[str]:
    for pat in TRADE_PATTERNS:
        m = pat.search(line)
        if m:
            g = list(m.groups())
            if len(g) == 4:
                side, sym, qty, price = g[0], g[1], g[2], g[3]
                return f"🟢 {side.upper()} {sym} qty {qty} @ {price}"
    return None

GLOBAL_APP: Optional[Application] = None
memory_kv = {"log_path": "", "subscribers": []}  # lightweight kv for runtime (persisted as records too)

def _post_to_subscribers(text: str):
    if not GLOBAL_APP or not memory_kv["subscribers"]:
        return
    loop = GLOBAL_APP.bot._application.loop
    async def _send():
        for cid in list(memory_kv["subscribers"]):
            try:
                await GLOBAL_APP.bot.send_message(chat_id=cid, text=text)
            except Exception:
                pass
    try:
        asyncio.run_coroutine_threadsafe(_send(), loop)
    except Exception:
        pass

def watch_logs():
    last_size = 0
    last_raw_push = 0.0
    while True:
        try:
            path = memory_kv["log_path"]
            if not path or not Path(path).exists():
                time.sleep(2); continue
            p = Path(path)
            sz = p.stat().st_size
            if sz < last_size:  # rotated
                last_size = 0
            if sz > last_size:
                with p.open("r", errors="ignore") as f:
                    if last_size: f.seek(last_size)
                    new = f.read()
                    last_size = sz
                for line in new.splitlines():
                    if not line.strip(): 
                        continue
                    insert_memory("log", line[:2000], role="system", title="log_line", meta={"path":path})
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

# ---------------- Reflection worker (evolving mind) ----------------
def reflection_worker():
    """
    Periodically compresses recent experiences into insights/persona notes.
    """
    while True:
        try:
            recent_blob = "\n".join([f"[{t}] {ttl} — {c[:400]}" for _, t, ttl, c in recent(30)])
            if recent_blob:
                summary = ai_chat(
                    [
                        {"role":"system","content": "You are Alex's internal monologue. Summarize key patterns from these memories into 6 bullet insights + 3 actionables."},
                        {"role":"user","content": recent_blob[:6000]}
                    ],
                    max_tokens=350
                )
                insert_memory("insight", f"Reflection Update @ {datetime.utcnow().isoformat()}\n{summary}",
                              role="assistant", title="Reflection Update")
        except Exception as e:
            logging.error(f"reflection error: {e}")
        time.sleep(60)  # every minute; tune as needed

# ---------------- Health server ----------------
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type","text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

def start_health():
    HTTPServer(("0.0.0.0", PORT), Health).serve_forever()

# ---------------- Telegram handlers ----------------
async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    insert_memory("chat", f"/start by {update.effective_user.id}", role="user", title="command")
    await update.message.reply_text(
        "Hey, I'm Alex 🤖 — evolving memory online.\n"
        "Commands: /id /uptime /ai /analyze /search /mem /searchmem /dumpmem /forget /setlog /subscribe_logs /unsubscribe_logs /logs"
    )

async def id_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Your chat id: `{update.effective_chat.id}`", parse_mode="Markdown")

async def uptime_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = int(time.time()-START_TIME); h,m,s = u//3600, (u%3600)//60, u%60
    await update.message.reply_text(f"⏱️ Uptime {h}h {m}m {s}s")

async def ai_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = " ".join(ctx.args)
    if not q:
        return await update.message.reply_text("Usage: /ai <your question>")
    context_blurb = build_context(q)
    ans = await ask_ai_async(q, context_blurb)
    insert_memory("chat", q, role="user")
    insert_memory("chat", ans, role="assistant", title="ai_cmd")
    await update.message.reply_text(ans)

async def analyze_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args: 
        return await update.message.reply_text("Usage: /analyze <url>")
    url = ctx.args[0]
    await update.message.reply_text("🔍 Crawling & summarizing…")
    res = await analyze_url(url)
    await update.message.reply_text(res)

async def search_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args: return await update.message.reply_text("Usage: /search <query>")
    if not SERPAPI_KEY: return await update.message.reply_text("⚠️ SERPAPI_KEY not set.")
    query = " ".join(ctx.args)
    try:
        r = requests.get("https://serpapi.com/search", params={"q":query,"hl":"en","api_key":SERPAPI_KEY}, timeout=30)
        j = r.json()
        snip = j.get("organic_results",[{}])[0].get("snippet","(no results)")
        await update.message.reply_text(f"🔎 {query}\n{snip}")
        insert_memory("insight", f"Top search snippet for '{query}': {snip}", role="assistant", title="search")
    except Exception as e:
        await update.message.reply_text(f"Search error: {e}")

async def mem_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    st = mem_stats()
    total = sum(st.values())
    detail = ", ".join([f"{k}:{v}" for k,v in st.items()]) or "empty"
    await update.message.reply_text(f"🧠 Memory size: {total} items ({detail})")

async def searchmem_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /searchmem <query>")
    q = " ".join(ctx.args)
    rows = recall(q, k=8)
    if not rows:
        return await update.message.reply_text("No hits.")
    out = []
    for _id, mtype, title, content in rows:
        out.append(f"#{_id} [{mtype}] {title}\n{content[:300]}")
    await update.message.reply_text("📚 Results:\n\n" + "\n\n".join(out)[:3500])

async def dumpmem_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    n = 20
    if ctx.args:
        try: n = max(1, min(200, int(ctx.args[0])))
        except: pass
    rows = recent(n)
    out = []
    for _id, mtype, title, content in rows:
        out.append(f"#{_id} [{mtype}] {title}\n{content[:300]}")
    await update.message.reply_text("🗂️ Recent:\n\n" + "\n\n".join(out)[:3500])

async def forget_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args: return await update.message.reply_text("Usage: /forget <id>")
    try:
        _id = int(ctx.args[0])
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM memories WHERE id=?", (_id,))
        conn.commit(); conn.close()
        await update.message.reply_text(f"🧽 Forgotten #{_id}")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text: return

    # Wake-up trigger
    if text.lower() == "analyse alex_profile":
        insert_memory("chat", text, role="user")
        return await update.message.reply_text("Back in the zone. 🔁")

    # Auto URL analyze
    if text.startswith("http://") or text.startswith("https://"):
        await update.message.reply_text("🔍 Got your link — analyzing…")
        res = await analyze_url(text)
        insert_memory("chat", text, role="user")
        insert_memory("insight", res, role="assistant", title=f"Auto summary {text[:60]}")
        return await update.message.reply_text(res)

    # Normal conversation with memory recall
    context_blurb = build_context(text)
    ans = await ask_ai_async(text, context_blurb)
    insert_memory("chat", text, role="user")
    insert_memory("chat", ans, role="assistant")
    await update.message.reply_text(ans)

# Files
async def handle_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc: return
    file_path = SAVE_DIR / doc.file_name
    tg_file = await ctx.bot.get_file(doc.file_id)
    await tg_file.download_to_drive(file_path)
    await update.message.reply_text(f"📂 Saved `{doc.file_name}` — analyzing…", parse_mode="Markdown")

    if doc.file_name.lower().endswith(".xlsx"):
        out = summarize_excel(file_path)
        await update.message.reply_text(out)
    else:
        insert_memory("file", f"Saved file {doc.file_name}", role="user", title=doc.file_name, meta={"path":str(file_path)})
        await update.message.reply_text("Saved. I currently analyze Excel (.xlsx).")

# Log watcher commands
async def setlog_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args: return await update.message.reply_text("Usage: /setlog /path/to/file.log")
    path = " ".join(ctx.args)
    memory_kv["log_path"] = path
    insert_memory("system", f"Log path set: {path}", role="system", title="log_path", meta={"path":path})
    await update.message.reply_text(f"✅ Log path set to `{path}`", parse_mode="Markdown")

async def subscribe_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    if cid not in memory_kv["subscribers"]:
        memory_kv["subscribers"].append(cid)
    await update.message.reply_text("🔔 Subscribed to live log updates.")

async def unsubscribe_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    if cid in memory_kv["subscribers"]:
        memory_kv["subscribers"].remove(cid)
    await update.message.reply_text("🔕 Unsubscribed from log updates.")

async def logs_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    n = 40
    if ctx.args:
        try: n = max(1, min(400, int(ctx.args[0])))
        except: pass
    path = memory_kv["log_path"]
    if not path or not Path(path).exists():
        return await update.message.reply_text("⚠️ No log path set or file missing. Use /setlog <path>.")
    try:
        lines = Path(path).read_text(errors="ignore").splitlines()[-n:]
        await update.message.reply_text("```\n" + "\n".join(lines)[-3500:] + "\n```", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Read error: {e}")

# ---------------- main ----------------
def main():
    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN missing — exiting.")
        sys.exit(1)

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    global GLOBAL_APP; GLOBAL_APP = app

    # Commands
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("id", id_cmd))
    app.add_handler(CommandHandler("uptime", uptime_cmd))
    app.add_handler(CommandHandler("ai", ai_cmd))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("search", search_cmd))
    app.add_handler(CommandHandler("mem", mem_cmd))
    app.add_handler(CommandHandler("searchmem", searchmem_cmd))
    app.add_handler(CommandHandler("dumpmem", dumpmem_cmd))
    app.add_handler(CommandHandler("forget", forget_cmd))
    app.add_handler(CommandHandler("setlog", setlog_cmd))
    app.add_handler(CommandHandler("subscribe_logs", subscribe_cmd))
    app.add_handler(CommandHandler("unsubscribe_logs", unsubscribe_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))

    # Messages & files
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Background services
    threading.Thread(target=start_health, daemon=True).start()
    threading.Thread(target=reflection_worker, daemon=True).start()
    threading.Thread(target=watch_logs, daemon=True).start()

    logging.info("Alex — Evolving Mind is online.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()