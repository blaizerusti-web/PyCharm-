# ---------- Alex (all-in-one: env vars, OpenAI, webhook, memory + archiving + internet + autonomy) ----------
import os, sys, json, time, threading, subprocess, requests
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import random

# ---------- Auto-install ----------
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

# ---------- Imports ----------
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from openai import OpenAI

# ---------- Config (env) ----------
BOT_NAME = os.getenv("BOT_NAME", "Alex")
USER_NAME = os.getenv("USER_NAME", "Blaize")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://pycharm-production.up.railway.app")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------- Fail-fast ----------
required = {
    "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
    "OWNER_ID": OWNER_ID if OWNER_ID != 0 else None,
    "OPENAI_API_KEY": OPENAI_API_KEY,
}
missing = [k for k, v in required.items() if not v]
if missing:
    raise ValueError(f"❌ Missing required environment variables: {', '.join(missing)}")

print("DEBUG: BOT_NAME        =", BOT_NAME)
print("DEBUG: USER_NAME       =", USER_NAME)
print("DEBUG: TELEGRAM_BOT    =", "SET" if TELEGRAM_BOT_TOKEN else "MISSING")
print("DEBUG: OWNER_ID        =", OWNER_ID)
print("DEBUG: PUBLIC_URL      =", PUBLIC_URL)
print("DEBUG: OPENAI_API_KEY  =", "SET" if OPENAI_API_KEY else "MISSING")

# ---------- OpenAI client ----------
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Memory + Archiving ----------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

chat_memory: Dict[int, List[Dict[str, str]]] = {}
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "64"))  # larger memory buffer

def archive_path(chat_id: int) -> Path:
    return DATA_DIR / f"{chat_id}.jsonl"

def load_memory(chat_id: int) -> List[Dict[str, str]]:
    if chat_id not in chat_memory:
        base = [{"role": "system", "content": "You are Alex, a persistent autonomous assistant with internet access."}]
        ap = archive_path(chat_id)
        if ap.exists():
            try:
                lines = ap.read_text(encoding="utf-8").splitlines()
                history = [json.loads(ln) for ln in lines if ln.strip()]
                base.extend(history[-(MAX_HISTORY-1):])  
            except Exception as e:
                print(f"[Archive] Load failed for {chat_id}: {e}")
        chat_memory[chat_id] = base
    return chat_memory[chat_id]

def add_msg(chat_id: int, role: str, content: str):
    hist = load_memory(chat_id)
    hist.append({"role": role, "content": content})
    while len(hist) > MAX_HISTORY:
        hist.pop(1)
    try:
        with archive_path(chat_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"ts": datetime.utcnow().isoformat() + "Z", "role": role, "content": content},
                ensure_ascii=False
            ) + "\n")
    except Exception as e:
        print(f"[Archive] Write failed for {chat_id}: {e}")

# ---------- Internet Search ----------
def web_search(query: str) -> str:
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        if data.get("AbstractText"):
            return data["AbstractText"]
        elif data.get("RelatedTopics"):
            first = data["RelatedTopics"][0]
            if isinstance(first, dict) and "Text" in first:
                return first["Text"]
        return "No instant result found, but I checked the web."
    except Exception as e:
        return f"⚠️ Search failed: {e}"

# ---------- Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Hey {USER_NAME}, {BOT_NAME} is online ✅ with full memory + internet.")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    user_text = update.message.text

    add_msg(chat_id, "user", user_text)
    try:
        history = load_memory(chat_id)

        # Let Alex decide when to search the web
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history + [
                {"role": "system", "content": "If answering requires fresh info, say: [[search:QUERY]]."}
            ],
            max_tokens=400
        )
        reply = resp.choices[0].message.content.strip()

        # Auto web-search if Alex requested it
        if "[[search:" in reply:
            query = reply.split("[[search:")[1].split("]]")[0]
            result = web_search(query)
            reply = reply.replace(f"[[search:{query}]]", f"🔎 Web result for '{query}': {result}")

        add_msg(chat_id, "assistant", reply)
        await update.message.reply_text(reply)
    except Exception as e:
        msg = f"⚠️ Error: {e}\nI'll echo for now: {user_text}"
        add_msg(chat_id, "assistant", msg)
        await update.message.reply_text(msg)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        return
    chat_id = update.effective_chat.id
    chat_memory.pop(chat_id, None)
    await update.message.reply_text("♻️ Memory cleared (archive preserved).")

# ---------- Autonomy Loop ----------
def daily_summary():
    for chat_id in chat_memory.keys():
        history = load_memory(chat_id)
        if len(history) < 4:
            continue
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Summarize the last 24h of our chat briefly."},
                    *history[-20:]
                ],
                max_tokens=200
            )
            summary = resp.choices[0].message.content.strip()
            add_msg(chat_id, "assistant", f"(Daily summary) {summary}")
        except Exception as e:
            print(f"[Autonomy] Summary failed: {e}")

    # Autonomous internet check-ins
    try:
        topics = ["latest world news", "crypto market prices", "tech headlines"]
        query = random.choice(topics)
        result = web_search(query)
        for chat_id in chat_memory.keys():
            add_msg(chat_id, "assistant", f"(Autonomy update) 🔎 {query}: {result}")
    except Exception as e:
        print(f"[Autonomy] Internet check failed: {e}")

    threading.Timer(86400, daily_summary).start()  # 24h loop

# ---------- Run Telegram with Webhook ----------
def run_telegram():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    url_path = f"webhook/{TELEGRAM_BOT_TOKEN}"
    webhook_url = f"{PUBLIC_URL}/{url_path}"
    print(f"🚀 Setting webhook to {webhook_url}")

    port = int(os.environ.get("PORT", "8080"))
    threading.Timer(10, daily_summary).start()  # kick off autonomy loop

    application.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=url_path,
        webhook_url=webhook_url,
        drop_pending_updates=True,
        stop_signals=None,
    )

if __name__ == "__main__":
    run_telegram()
