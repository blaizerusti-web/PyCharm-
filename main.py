# =========================
# mega.py  — Alex (all-in-one, humane memory, Railway-ready)
# =========================
# Features
# - Telegram bot (python-telegram-bot v20)
# - Humane Memory “mind”: SQLite (summarize → tag → sentiment → merge/evolve)
# - Auto-install deps on boot
# - URL crawler + SEO summarize + memory ingest
# - Excel analyzer (.xlsx) + memory ingest
# - Live log watcher (/setlog, /subscribe_logs, /logs) + trade-line summarizer
# - Health server for Railway (PORT; default 8080)

import os, sys, json, time, re, logging, threading, hashlib, sqlite3, asyncio, subprocess
from pathlib import Path
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------- bootstrap: install/import ----------
def _ensure(import_name: str, pip_name: str | None = None):
    try:
        return __import__(import_name)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                               pip_name or import_name])
        return __import__(import_name)

requests = _ensure("requests")
aiohttp  = _ensure("aiohttp")
bs4      = _ensure("bs4", "beautifulsoup4")
t_ext    = _ensure("telegram.ext", "python-telegram-bot==20.*")
telegram = _ensure("telegram")
openai_m = _ensure("openai")
pandas_m = _ensure("pandas")
_ensure("openpyxl")  # reader for xlsx

from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ---------- config / logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SERPAPI_KEY    = os.getenv("SERPAPI_KEY", "").strip()
PORT           = int(os.getenv("PORT", "8080"))
HUMANE_TONE    = os.getenv("HUMANE_TONE", "1") == "1"  # capture emotion/tone

if not TELEGRAM_TOKEN:
    logging.warning("TELEGRAM_TOKEN not set.")

START_TIME = time.time()
SAVE_DIR = Path("received_files"); SAVE_DIR.mkdir(exist_ok=True)

# ---------- OpenAI helper (new + old SDK compatible) ----------
def _make_ai():
    try:
        from openai import OpenAI
        cli = OpenAI(api_key=OPENAI_API_KEY)
        def chat(messages, model="gpt-4o-mini", max_tokens=700):
            r = cli.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
            return r.choices[0].message.content.strip()
        return chat
    except Exception:
        import openai as _openai
        _openai.api_key = OPENAI_API_KEY
        def chat(messages, model="gpt-4o-mini", max_tokens=700):
            r = _openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens)
            return r["choices"][0]["message"]["content"].strip()
        return chat

AI_CHAT = _make_ai()

async def ask_ai(prompt: str, context: str = "") -> str:
    if not OPENAI_API_KEY:
        return "⚠️ OPENAI_API_KEY not set."
    return AI_CHAT([
        {"role":"system","content": context or "You are Alex — concise, helpful, and a little witty."},
        {"role":"user","content": prompt}
    ])

# ---------- SQLite humane memory ----------
DB_PATH = "alex_memory.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  source TEXT NOT NULL,     -- 'chat' | 'link' | 'file' | 'log'
  topic_key TEXT NOT NULL,  -- hashed topic key for merging
  title TEXT,               -- short headline
  content TEXT NOT NULL,    -- summarized essence
  tags TEXT,                -- comma-separated tags
  sentiment TEXT,           -- e.g., positive/neutral/negative
  raw_ref TEXT              -- optional reference (url/filename)
);
""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_topic ON notes(topic_key);")
conn.commit()

def _topic_key(text: str) -> str:
    """Hash of a coarse 'topic' so repeated info merges/evolves."""
    seed = text.lower()
    seed = re.sub(r"https?://\S+", "", seed)
    seed = re.sub(r"[^a-z0-9 ]+", " ", seed)
    seed = " ".join(seed.split()[:12])
    return hashlib.sha1(seed.encode()).hexdigest()

def _merge_contents(old: str, new: str) -> str:
    """Use AI to fold new info into old summary, keep concise."""
    try:
        merged = AI_CHAT([
            {"role":"system","content":"Merge the NEW info into the EXISTING summary. Keep ≤120 words, bullets allowed."},
            {"role":"user","content":f"EXISTING:\n{old}\n\nNEW:\n{new}"}
        ], max_tokens=220)
        return merged
    except Exception:
        return (old + "\n" + new)[:1200]

def _humane_summarize(text: str, capture_tone: bool = True) -> dict:
    """
    Return dict: {title, summary, tags, sentiment}
    """
    sysmsg = "You compress inputs into human-friendly memory: 3–6 bullets or 2–4 short sentences; include only essentials."
    if capture_tone and HUMANE_TONE:
        sysmsg += " Detect the writer's tone/emotion (positive/neutral/negative + 1 word)."

    prompt = f"""Digest this into humane memory.

TEXT:
{text}

Return JSON with keys: title, summary, tags (3-6, comma-separated), sentiment."""
    try:
        out = AI_CHAT([
            {"role":"system","content":sysmsg},
            {"role":"user","content":prompt}
        ], max_tokens=320)
        # be robust if it's not perfect JSON
        try:
            data = json.loads(out)
        except Exception:
            # quick repair: grab summary heuristically
            data = {"title":"", "summary":out.strip(), "tags":"", "sentiment":""}
        return {
            "title": (data.get("title") or "")[:120],
            "summary": (data.get("summary") or out).strip(),
            "tags": (data.get("tags") or "").replace("\n"," ").strip(),
            "sentiment": (data.get("sentiment") or "").strip()
        }
    except Exception as e:
        return {"title":"", "summary": text[:900], "tags":"", "sentiment":""}

def remember(source: str, text: str, raw_ref: str = "") -> int:
    """
    Summarize + tag + (optionally) tone; merge if same topic; store to SQLite.
    Returns note id.
    """
    digest = _humane_summarize(text, capture_tone=True)
    topic = _topic_key(digest["summary"] or text)

    # check existing by topic_key
    cur.execute("SELECT id, content FROM notes WHERE topic_key=? ORDER BY id DESC LIMIT 1", (topic,))
    row = cur.fetchone()
    if row:
        nid, existing = row
        merged = _merge_contents(existing, digest["summary"])
        cur.execute("""
            UPDATE notes
               SET ts=?, source=?, title=?, content=?, tags=?, sentiment=?, raw_ref=?
             WHERE id=?""",
            (datetime.utcnow().isoformat(), source, digest["title"], merged,
             digest["tags"], digest["sentiment"], raw_ref, nid))
        conn.commit()
        return nid
    else:
        cur.execute("""
            INSERT INTO notes (ts, source, topic_key, title, content, tags, sentiment, raw_ref)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), source, topic, digest["title"], digest["summary"],
              digest["tags"], digest["sentiment"], raw_ref))
        conn.commit()
        return cur.lastrowid

def recent_notes(n: int = 10) -> list[dict]:
    cur.execute("SELECT id, ts, source, title, content, tags, sentiment, raw_ref FROM notes ORDER BY id DESC LIMIT ?", (n,))
    rows = cur.fetchall()
    return [
        {"id":r[0], "ts":r[1], "source":r[2], "title":r[3] or "", "content":r[4],
         "tags":r[5] or "", "sentiment":r[6] or "", "ref":r[7] or ""}
        for r in rows
    ]

def export_all() -> list[dict]:
    cur.execute("SELECT id, ts, source, title, content, tags, sentiment, raw_ref FROM notes ORDER BY id ASC")
    rows = cur.fetchall()
    return [
        {"id":r[0], "ts":r[1], "source":r[2], "title":r[3] or "", "content":r[4],
         "tags":r[5] or "", "sentiment":r[6] or "", "ref":r[7] or ""}
        for r in rows
    ]

def forget_note(nid: int) -> bool:
    cur.execute("DELETE FROM notes WHERE id=?", (nid,))
    conn.commit()
    return cur.rowcount > 0

# ---------- URL crawler ----------
async def fetch_url(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=25, headers={"User-Agent":"Mozilla/5.0"}) as r:
                if r.status != 200:
                    return f"⚠️ HTTP {r.status}"
                text = await r.text()
        soup = BeautifulSoup(text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
        desc_tag = soup.find("meta", {"name":"description"})
        desc = (desc_tag.get("content","") if desc_tag else "")
        h1 = soup.h1.get_text(strip=True) if soup.h1 else ""
        words = len(soup.get_text(" ").split())
        links = len(soup.find_all("a"))
        images = len(soup.find_all("img"))
        snippet = soup.get_text(" ")[:1800]
        return (f"🌐 {title}\nDesc: {desc[:200]}\nH1: {h1}\n"
                f"Words:{words} Links:{links} Images:{images}\n\n"
                f"Snippet:\n{snippet}")
    except Exception as e:
        return f"⚠️ Crawl error: {e}"

async def analyze_url(url: str) -> str:
    content = await fetch_url(url)
    if content.startswith("⚠️"): return content
    summary = await ask_ai(
        "Summarize page in short bullets, key entities, actions, and SEO opportunities:\n\n"+content
    )
    # store into memory mind
    remember("link", summary, raw_ref=url)
    return summary

# ---------- Excel analyzer ----------
import pandas as pd
def analyze_excel(path: Path) -> str:
    try:
        df = pd.read_excel(path)
        head_cols = ", ".join(map(str, list(df.columns)[:12]))
        info = f"✅ Excel loaded: {df.shape[0]} rows × {df.shape[1]} cols\nColumns: {head_cols}"

        # small profiling
        num = df.select_dtypes(include="number")
        if not num.empty:
            info += "\n\nNumeric summary:\n" + num.describe().to_string()[:1800]

        # memory digest
        remember("file", f"Excel {path.name}: shape {df.shape}; columns {list(df.columns)!r}", raw_ref=str(path))
        return info
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
            g = list(m.groups())
            if len(g) == 4:
                side, sym, qty, price = g
                try: qty_i = int(qty)
                except: qty_i = qty
                return f"🟢 {side.upper()} {sym} qty {qty_i} @ {price}"
    return None

# ---------- Telegram handlers ----------
GLOBAL_APP: Application | None = None

async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hey, I'm Alex 🤖\n"
        "Try: /ai /analyze <url> /remember <text> /mem /exportmem\n"
        "Files: send .xlsx to analyze + learn.\n"
        "Logs: /setlog /subscribe_logs /logs\n"
    )

async def id_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Your chat id: `{update.effective_chat.id}`", parse_mode="Markdown")

async def uptime_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = int(time.time()-START_TIME); h,m,s = u//3600,(u%3600)//60,u%60
    await update.message.reply_text(f"⏱️ Uptime {h}h {m}m {s}s")

async def ai_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = " ".join(ctx.args)
    if not q: return await update.message.reply_text("Usage: /ai <your question>")
    ans = await ask_ai(q)
    # store lite memory of Q/A gist
    remember("chat", f"Q: {q}\nA gist: {ans[:300]}")
    await update.message.reply_text(ans)

async def search_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args: return await update.message.reply_text("Usage: /search <query>")
    if not SERPAPI_KEY: return await update.message.reply_text("⚠️ SERPAPI_KEY not set.")
    query = " ".join(ctx.args)
    try:
        r = requests.get("https://serpapi.com/search", params={"q":query,"hl":"en","api_key":SERPAPI_KEY}, timeout=25)
        j = r.json()
        snip = j.get("organic_results", [{}])[0].get("snippet", "(no results)")
        await update.message.reply_text(f"🔎 {query}\n{snip}")
        remember("chat", f"SERP for '{query}': {snip}")
    except Exception as e:
        await update.message.reply_text(f"Search error: {e}")

async def analyze_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args: return await update.message.reply_text("Usage: /analyze <url>")
    url = ctx.args[0]
    await update.message.reply_text("🔍 Crawling and summarizing…")
    res = await analyze_url(url)
    await update.message.reply_text(res)

async def remember_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = " ".join(ctx.args).strip()
    if not text:
        return await update.message.reply_text("Usage: /remember <text to add to memory>")
    nid = remember("chat", text)
    await update.message.reply_text(f"🧠 Noted (id {nid}). I’ll keep the essence, not the fluff.")

async def mem_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    n = 8
    if ctx.args:
        try: n = max(1, min(40, int(ctx.args[0])))
        except: pass
    items = recent_notes(n)
    if not items:
        return await update.message.reply_text("🧠 Memory is empty (for now).")
    lines = []
    for it in items:
        head = f"#{it['id']} [{it['source']}] {it['title'] or '(no title)'}"
        lines.append(head.strip())
        lines.append("  " + it["content"].replace("\n", "\n  ")[:500])
        if it["tags"]: lines.append(f"  tags: {it['tags']}")
        if it["ref"]:  lines.append(f"  ref: {it['ref']}")
        lines.append("")
    text = "\n".join(lines)
    await update.message.reply_text(text[:3900])

async def exportmem_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    data = export_all()
    path = Path("memory_export.json"); path.write_text(json.dumps(data, indent=2))
    await update.message.reply_document(document=str(path), filename="alex_memory.json")

async def forget_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args: return await update.message.reply_text("Usage: /forget <id>")
    try:
        nid = int(ctx.args[0])
        ok = forget_note(nid)
        await update.message.reply_text("🗑️ Forgotten." if ok else "Couldn’t find that id.")
    except:
        await update.message.reply_text("Enter a numeric id.")

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()

    if text.lower() == "analyse alex_profile":
        return await update.message.reply_text("Back in the zone. What’s next?")

    if text.startswith("http://") or text.startswith("https://"):
        await update.message.reply_text("🔍 Got your link — analyzing…")
        res = await analyze_url(text)
        return await update.message.reply_text(res)

    # default AI chat
    ans = await ask_ai(text)
    remember("chat", f"User said: {text}\nResponse gist: {ans[:280]}")
    await update.message.reply_text(ans)

# --- file uploads (Excel) ---
async def handle_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc: return
    file_path = SAVE_DIR / doc.file_name
    tg_file = await ctx.bot.get_file(doc.file_id)
    await tg_file.download_to_drive(file_path)
    await update.message.reply_text(f"📂 Saved `{doc.file_name}` — analyzing…", parse_mode="Markdown")

    if doc.file_name.lower().endswith(".xlsx"):
        out = analyze_excel(file_path)
        await update.message.reply_text(out)
    else:
        remember("file", f"Received file {doc.file_name}", raw_ref=str(file_path))
        await update.message.reply_text("Saved. I currently analyze Excel (.xlsx).")

# --- log watcher commands ---
MEM = {"log_path":"", "subscribers": []}  # lightweight runtime prefs

async def setlog_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /setlog /path/to/your.log")
    path = " ".join(ctx.args)
    MEM["log_path"] = path
    await update.message.reply_text(f"✅ Log path set to: `{path}`", parse_mode="Markdown")

async def subscribe_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    if cid not in MEM["subscribers"]:
        MEM["subscribers"].append(cid)
    await update.message.reply_text("🔔 Subscribed to live log updates.")

async def unsubscribe_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    if cid in MEM["subscribers"]:
        MEM["subscribers"].remove(cid)
    await update.message.reply_text("🔕 Unsubscribed.")

async def logs_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    n = 40
    if ctx.args:
        try: n = max(1, min(400, int(ctx.args[0])))
        except: pass
    path = MEM.get("log_path") or ""
    p = Path(path)
    if not path or not p.exists():
        return await update.message.reply_text("⚠️ No log path set or file missing. Use /setlog <path>.")
    try:
        lines = p.read_text(errors="ignore").splitlines()[-n:]
        await update.message.reply_text("```\n" + "\n".join(lines)[-3500:] + "\n```", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Read error: {e}")

# ---------- log watcher thread ----------
GLOBAL_APP: Application | None = None

def _post_to_subscribers(text: str):
    if not GLOBAL_APP or not MEM.get("subscribers"): return
    loop = GLOBAL_APP.bot._application.loop
    async def _send():
        for cid in list(MEM["subscribers"]):
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
    last_raw_push = 0
    while True:
        try:
            path = MEM.get("log_path") or ""
            if not path or not Path(path).exists():
                time.sleep(2); continue
            p = Path(path)
            sz = p.stat().st_size
            if sz < last_size: last_size = 0  # rotation
            if sz > last_size:
                with p.open("r", errors="ignore") as f:
                    if last_size: f.seek(last_size)
                    new = f.read()
                    last_size = sz
                for line in new.splitlines():
                    s = summarize_trade_line(line)
                    if s:
                        _post_to_subscribers(s)
                        remember("log", s, raw_ref=path)
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
    Periodically condense the last N notes into updated persona hints.
    (We simply keep folding knowledge into a compact mental model.)
    """
    while True:
        try:
            cur.execute("SELECT content FROM notes ORDER BY id DESC LIMIT 20")
            rec = [r[0] for r in cur.fetchall()]
            if rec:
                persona = AI_CHAT([
                    {"role":"system","content":"Create concise persona guidance from these memory snippets. 4–6 short bullets."},
                    {"role":"user","content":"\n\n".join(rec)}
                ], max_tokens=180)
                # Store/refresh a special 'persona' row (topic_key='__persona__')
                tk = "__persona__"
                cur.execute("SELECT id, content FROM notes WHERE topic_key=? ORDER BY id DESC LIMIT 1", (tk,))
                row = cur.fetchone()
                if row:
                    cur.execute("UPDATE notes SET ts=?, source=?, title=?, content=?, tags=?, sentiment=? WHERE id=?",
                                (datetime.utcnow().isoformat(),"system","Persona",persona,"persona,profile","",row[0]))
                else:
                    cur.execute("INSERT INTO notes (ts, source, topic_key, title, content, tags, sentiment, raw_ref) VALUES (?,?,?,?,?,?,?,?)",
                                (datetime.utcnow().isoformat(),"system",tk,"Persona",persona,"persona,profile","",""))
                conn.commit()
        except Exception as e:
            logging.error(f"learning error: {e}")
        time.sleep(60)

# ---------- health server ----------
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type","text/plain"); self.end_headers()
        self.wfile.write(b"ok")

def start_health():
    HTTPServer(("0.0.0.0", PORT), Health).serve_forever()

# ---------- main ----------
def main():
    global GLOBAL_APP
    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN missing — exiting."); sys.exit(1)

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    GLOBAL_APP = app

    # commands
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("id", id_cmd))
    app.add_handler(CommandHandler("uptime", uptime_cmd))
    app.add_handler(CommandHandler("ai", ai_cmd))
    app.add_handler(CommandHandler("search", search_cmd))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("remember", remember_cmd))
    app.add_handler(CommandHandler("mem", mem_cmd))
    app.add_handler(CommandHandler("exportmem", exportmem_cmd))
    app.add_handler(CommandHandler("forget", forget_cmd))
    app.add_handler(CommandHandler("setlog", setlog_cmd))
    app.add_handler(CommandHandler("subscribe_logs", subscribe_cmd))
    app.add_handler(CommandHandler("unsubscribe_logs", unsubscribe_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))

    # messages
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # background threads
    threading.Thread(target=start_health, daemon=True).start()
    threading.Thread(target=learning_worker, daemon=True).start()
    threading.Thread(target=watch_logs, daemon=True).start()

    logging.info("Alex mega bot starting…")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()