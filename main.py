import subprocess, sys

# Auto-install missing dependencies
required = ["openpyxl", "pandas"]
for package in required:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])# =========================
# main.py  (Alex — all-in-one, Railway-ready)
# =========================
# Features:
# - Telegram bot: /start /uptime /ai /search /analyze /id
# - File uploads: Excel (.xlsx) quick stats
# - URL crawler + SEO summarize
# - Memory (memory.json) + lightweight self-learning
# - Healthcheck HTTP server (PORT env, default 8080)
# - Live log watcher with /setlog /subscribe_logs /unsubscribe_logs /logs
# - Safe auto-install of missing dependencies (pandas, aiohttp, bs4, python-telegram-bot, openai)

import os, sys, json, time, threading, logging, asyncio, subprocess, re, socket
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------- bootstrap: install/import dependencies ----------
def _ensure(pkg_import: str, pip_name: str = None):
    """
    Try to import; if missing, pip install, then import.
    Returns imported module object.
    """
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

from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# OpenAI: support both old/new SDKs gracefully
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def ai_chat(messages, model="gpt-4o-mini", max_tokens=600):
        resp = _openai_client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        return resp.choices[0].message.content.strip()
except Exception:
    import openai as _openai
    _openai.api_key = os.getenv("OPENAI_API_KEY")
    def ai_chat(messages, model="gpt-4o-mini", max_tokens=600):
        resp = _openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens)
        return resp["choices"][0]["message"]["content"].strip()

# ---------- config / logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SERPAPI_KEY    = os.getenv("SERPAPI_KEY", "").strip()
PORT           = int(os.getenv("PORT", "8080"))

if not TELEGRAM_TOKEN:
    logging.warning("TELEGRAM_TOKEN is not set. Bot cannot start without it.")

START_TIME = time.time()
SAVE_DIR = Path("received_files"); SAVE_DIR.mkdir(exist_ok=True)

# ---------- memory ----------
memory_file = Path("memory.json")
if memory_file.exists():
    try:
        memory = json.loads(memory_file.read_text())
    except Exception:
        memory = {}
else:
    memory = {}

# sensible defaults
memory.setdefault("persona", {"notes": ""})
memory.setdefault("history", [])
memory.setdefault("log_path", "")          # path to the log file to watch
memory.setdefault("subscribers", [])       # list of chat_ids subscribed to log stream

def save_memory():
    try:
        memory_file.write_text(json.dumps(memory, indent=2))
    except Exception as e:
        logging.error(f"save_memory error: {e}")

# ---------- AI core ----------
async def ask_ai(prompt: str, context: str = "") -> str:
    if not OPENAI_API_KEY:
        return "⚠️ OPENAI_API_KEY not set."
    try:
        msg = [
            {"role": "system", "content": context or "You are Alex — concise, helpful, and a little witty."},
            {"role": "user",   "content": prompt}
        ]
        return ai_chat(msg, model="gpt-4o-mini", max_tokens=600)
    except Exception as e:
        logging.exception("AI error")
        return f"(AI error: {e})"

# ---------- crawler ----------
async def fetch_url(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"}) as r:
                if r.status != 200:
                    return f"⚠️ HTTP {r.status}"
                text = await r.text()
        soup = BeautifulSoup(text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
        desc_tag = soup.find("meta", {"name": "description"})
        desc = (desc_tag.get("content", "") if desc_tag else "")[:200]
        h1 = soup.h1.get_text(strip=True) if soup.h1 else ""
        words = len(soup.get_text(" ").split())
        links = len(soup.find_all("a"))
        images = len(soup.find_all("img"))
        snippet = soup.get_text(" ")[:1800]
        return (f"🌐 {title}\nDesc: {desc}\nH1: {h1}\n"
                f"Words:{words} Links:{links} Images:{images}\n\n"
                f"Snippet:\n{snippet}")
    except Exception as e:
        return f"⚠️ Crawl error: {e}"

async def analyze_url(url: str) -> str:
    content = await fetch_url(url)
    if content.startswith("⚠️"): return content
    return await ask_ai(f"Summarize the page and extract SEO opportunities, key entities and actions:\n\n{content}")

# ---------- Excel analyzer ----------
def analyze_excel(path: Path) -> str:
    try:
        df = pd.read_excel(path)
        head_cols = ", ".join(list(df.columns)[:10])
        info = f"✅ Excel loaded: {df.shape[0]} rows × {df.shape[1]} cols\nColumns: {head_cols}"
        # small profiling
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            desc = df[numeric_cols].describe().to_string()
            info += f"\n\nNumeric summary:\n{desc[:1800]}"
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
            g = [x for x in m.groups()]
            # normalize order of captures (varies by pattern)
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
        "Hey, I'm Alex 🤖\n"
        "Commands: /uptime /ai /search /analyze /setlog /subscribe_logs /unsubscribe_logs /logs /id"
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

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()

    if text.lower() == "analyse alex_profile":
        replies = [
            "Let's get back to it.", "Alright, I'm here — let's dive in.",
            "Back in the zone.", "Let's pick up where we left off.", "Okay, ready to roll."
        ]
        return await update.message.reply_text(replies[int(time.time()) % len(replies)])

    if text.startswith("http://") or text.startswith("https://"):
        await update.message.reply_text("🔍 Got your link — analyzing…")
        res = await analyze_url(text)
        return await update.message.reply_text(res)

    # default AI
    ans = await ask_ai(text)
    await update.message.reply_text(ans)

    # memory breadcrumb
    memory["history"].append(text)
    memory["history"] = memory["history"][-200:]  # trim
    save_memory()

# --- file uploads (Excel) ---
async def handle_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc:
        return
    file_path = SAVE_DIR / doc.file_name
    tg_file = await ctx.bot.get_file(doc.file_id)
    await tg_file.download_to_drive(file_path)
    await update.message.reply_text(f"📂 Saved `{doc.file_name}` — analyzing…", parse_mode="Markdown")

    if doc.file_name.lower().endswith(".xlsx"):
        out = analyze_excel(file_path)
        await update.message.reply_text(out)
    else:
        await update.message.reply_text("I can currently analyze Excel (.xlsx). The file is saved.")

# --- log watcher commands ---
async def setlog_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /setlog /path/to/your.log")
    path = " ".join(ctx.args)
    memory["log_path"] = path
    save_memory()
    await update.message.reply_text(f"✅ Log path set to: `{path}`", parse_mode="Markdown")

async def subscribe_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in memory["subscribers"]:
        memory["subscribers"].append(chat_id)
        save_memory()
    await update.message.reply_text("🔔 Subscribed to live log updates.")

async def unsubscribe_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in memory["subscribers"]:
        memory["subscribers"].remove(chat_id)
        save_memory()
    await update.message.reply_text("🔕 Unsubscribed from log updates.")

async def logs_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    n = 40
    if ctx.args:
        try:
            n = max(1, min(400, int(ctx.args[0])))
        except:
            pass
    path = memory.get("log_path") or ""
    if not path or not Path(path).exists():
        return await update.message.reply_text("⚠️ No log path set or file does not exist. Use /setlog <path>.")
    try:
        lines = Path(path).read_text(errors="ignore").splitlines()[-n:]
        await update.message.reply_text("```\n" + "\n".join(lines)[-3500:] + "\n```", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Read error: {e}")

# ---------- log watcher thread ----------
def _post_to_subscribers(text: str):
    if not GLOBAL_APP or not memory.get("subscribers"):
        return
    loop = GLOBAL_APP.bot._application.loop  # PTB v20 stores loop here
    async def _send():
        for cid in list(memory["subscribers"]):
            try:
                await GLOBAL_APP.bot.send_message(chat_id=cid, text=text)
            except Exception:
                pass
    try:
        asyncio.run_coroutine_threadsafe(_send(), loop)
    except Exception:
        pass

def watch_logs():
    """
    Tail the configured log file and push updates.
    Emits parsed trade summaries and raw lines (throttled) for activity.
    """
    last_size = 0
    last_raw_push = 0
    while True:
        try:
            path = memory.get("log_path") or ""
            if not path or not Path(path).exists():
                time.sleep(2)
                continue
            p = Path(path)
            sz = p.stat().st_size
            if sz < last_size:
                last_size = 0  # file rotated
            if sz > last_size:
                with p.open("r", errors="ignore") as f:
                    if last_size:
                        f.seek(last_size)
                    new = f.read()
                    last_size = sz
                for line in new.splitlines():
                    # summarize trades if matched
                    s = summarize_trade_line(line)
                    if s:
                        _post_to_subscribers(s)
                    # also occasionally push raw lines for heartbeat
                    now = time.time()
                    if now - last_raw_push > 15:
                        last_raw_push = now
                        _post_to_subscribers("📜 " + line[:900])
        except Exception as e:
            logging.error(f"log watcher error: {e}")
        time.sleep(1)

# ---------- self-learning background ----------
def learning_worker():
    while True:
        try:
            if memory.get("history"):
                snippet = " ".join(memory["history"][-10:])
                note = ai_chat(
                    [
                        {"role":"system","content":"From the recent conversation lines, write 1–2 concise persona notes."},
                        {"role":"user","content":snippet}
                    ],
                    model="gpt-4o-mini",
                    max_tokens=120
                )
                memory["persona"]["notes"] = note
                save_memory()
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
    server = HTTPServer(("0.0.0.0", PORT), Health)
    logging.info(f"Health server on 0.0.0.0:{PORT}")
    server.serve_forever()

# ---------- main ----------
def main():
    global GLOBAL_APP
    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN missing — exiting.")
        sys.exit(1)

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    GLOBAL_APP = app

    # commands
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("id", id_cmd))
    app.add_handler(CommandHandler("uptime", uptime_cmd))
    app.add_handler(CommandHandler("ai", ai_cmd))
    app.add_handler(CommandHandler("search", search_cmd))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
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

    logging.info("Alex bot starting…")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()