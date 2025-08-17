# ---------- Alex (all-in-one, env vars, OpenAI integrated) ----------
import os, sys, json, time, threading, subprocess, socket
from pathlib import Path
from typing import List, Dict, Any

# Auto-install (Replit self-heal)
def install_requirements():
    try:
        import pkg_resources
        reqs = []
        if Path("requirements.txt").exists():
            with open("requirements.txt") as f:
                reqs = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        if not reqs:
            return
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = [p for p in reqs if p.split("==")[0].lower() not in installed]
        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except Exception as e:
        print(f"[Installer] skipped: {e}")

install_requirements()

# Imports
from flask import Flask
import requests
from bs4 import BeautifulSoup
try:
    from duckduckgo_search import DDGS
    DUCK_OK = True
except:
    DUCK_OK = False

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)

# Google Sheets
SHEETS_OK = True
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
except:
    SHEETS_OK = False

# OpenAI
import openai

# ---------- Config ----------
BOT_NAME = os.getenv("BOT_NAME", "Alex")
USER_NAME = os.getenv("USER_NAME", "Blaize")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
PUBLIC_URL = os.getenv("PUBLIC_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Auto-detect Replit public URL if not set
if not PUBLIC_URL:
    try:
        host = socket.gethostname()
        PUBLIC_URL = f"https://{host}.id.repl.co"
        print(f"[Auto-URL] Using {PUBLIC_URL}")
    except Exception as e:
        print(f"[Auto-URL] Could not detect: {e}")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN")
openai.api_key = OPENAI_API_KEY

# ---------- Flask ----------
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Alex is running!"

@app.route("/health")
def health():
    return "ok"

def run_flask():
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, threaded=True)

# ---------- Memory ----------
MEM_FILE = Path("memory.json")
MAX_HISTORY = 10

def load_mem():
    if MEM_FILE.exists():
        try:
            return json.loads(MEM_FILE.read_text(encoding="utf-8"))
        except:
            pass
    return {}

def save_mem(mem):
    try:
        MEM_FILE.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")
    except:
        pass

MEM = load_mem()

def remember(chat_id, role, text):
    key = str(chat_id)
    MEM.setdefault(key, [])
    MEM[key].append({"role": role, "text": text[-2000:]})
    MEM[key] = MEM[key][-MAX_HISTORY:]
    save_mem(MEM)

def last_context(chat_id):
    items = MEM.get(str(chat_id), [])
    return "\n".join(f"{m['role']}: {m['text']}" for m in items) if items else ""

# ---------- Sheets ----------
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def build_sheets():
    if not (SHEETS_OK and GOOGLE_CREDENTIALS_JSON and GOOGLE_SHEET_ID):
        return None, None
    try:
        info = json.loads(GOOGLE_CREDENTIALS_JSON)
        creds = service_account.Credentials.from_service_account_info(info, scopes=SHEETS_SCOPES)
        svc = build("sheets", "v4", credentials=creds).spreadsheets()
        return svc, GOOGLE_SHEET_ID
    except Exception as e:
        print("[Sheets] init error:", e)
        return None, None

SHEETS, SHEET_ID = build_sheets()

def sheets_append(values, range_a1="Sheet1!A:C"):
    if not (SHEETS and SHEET_ID):
        return False
    try:
        body = {"values": values}
        SHEETS.values().append(
            spreadsheetId=SHEET_ID,
            range=range_a1,
            valueInputOption="USER_ENTERED",
            body=body,
        ).execute()
        return True
    except Exception as e:
        print("[Sheets] append error:", e)
        return False

# ---------- Web Search ----------
def looks_like_web(q):
    ql = q.lower()
    triggers = ("search", "look up", "find", "google", "news", "latest", "price", "wiki", "http", "https")
    return any(t in ql for t in triggers)

def fetch_url_text(url, timeout=10, max_chars=1400):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "aside", "nav"]):
            tag.decompose()
        text = " ".join(soup.get_text("\n").split())
        return text[:max_chars]
    except:
        return ""

def web_search_summary(query, k=5):
    if not DUCK_OK:
        return "Web search not available."
    results = []
    with DDGS() as ddgs:
        for hit in ddgs.text(query, max_results=k, region="us-en", safesearch="moderate"):
            href = hit.get("href")
            if not href:
                continue
            results.append({
                "title": hit.get("title") or "Result",
                "href": href,
                "snippet": hit.get("body", ""),
            })
    for item in results[:2]:
        body = fetch_url_text(item["href"])
        if body:
            item["snippet"] += " — " + body[:600]
    lines = [f"🔎 **Search:** {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. *{r['title']}*\n{r['snippet']}\n{r['href']}\n")
    return "\n".join(lines[:1 + 2*3])

# ---------- OpenAI Reply ----------
def ai_reply(user_text, ctx=""):
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are {BOT_NAME}, a helpful assistant for {USER_NAME}."},
                {"role": "user", "content": ctx + "\n" + user_text}
            ]
        )
        return resp.choices[0].message['content']
    except Exception as e:
        return f"⚠️ AI error: {e}"

# ---------- Telegram Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    remember(update.effective_chat.id, "user", "/start")
    intro = f"Hey! I’m {BOT_NAME} — ready to chat, search, and log to your sheet."
    remember(update.effective_chat.id, "bot", intro)
    sheets_append([[time.strftime("%Y-%m-%d %H:%M:%S"), str(update.effective_chat.id), "Started chat"]])
    await update.message.reply_text(intro)

async def web_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text("Usage: /web <your query>")
        return
    remember(update.effective_chat.id, "user", f"/web {query}")
    result = web_search_summary(query)
    remember(update.effective_chat.id, "bot", result[:1000])
    await update.message.reply_text(result, parse_mode=ParseMode.MARKDOWN)

async def message_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()
    if not text:
        await update.message.reply_text("Send me some text 🙂")
        return
    remember(chat_id, "user", text)

    if looks_like_web(text):
        result = web_search_summary(text)
        remember(chat_id, "bot", result[:1000])
        await update.message.reply_text(result, parse_mode=ParseMode.MARKDOWN)
        return

    ctx = last_context(chat_id)
    reply = ai_reply(text, ctx)
    remember(chat_id, "bot", reply)
    await update.message.reply_text(reply)

# ---------- Jobs ----------
async def daily_checkin(context: ContextTypes.DEFAULT_TYPE):
    if OWNER_ID:
        try:
            await context.bot.send_message(chat_id=OWNER_ID, text="Daily check-in 👋 Need anything?")
        except:
            pass

async def keepalive_ping(context: ContextTypes.DEFAULT_TYPE):
    if not PUBLIC_URL:
        return
    try:
        requests.get(PUBLIC_URL.rstrip("/") + "/health", timeout=5)
    except:
        pass

# ---------- Run ----------
def run_bot():
    app_tg = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app_tg.add_handler(CommandHandler("start", start))
    app_tg.add_handler(CommandHandler("web", web_cmd))
    app_tg.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_router))
    jq = app_tg.job_queue
    jq.run_repeating(daily_checkin, interval=86400, first=30)
    jq.run_repeating(keepalive_ping, interval=300, first=60)
    print("🚀 Alex is live.")
    app_tg.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    run_bot()
