# ---------- Alex (all-in-one: env vars, OpenAI, webhook, memory + archiving) ----------
import os, sys, json, time, threading, subprocess
from pathlib import Path
from typing import List, Dict
from datetime import datetime

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

# OpenAI (new client)
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

# In-RAM rolling memory (per chat)
chat_memory: Dict[int, List[Dict[str, str]]] = {}
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "32"))

def archive_path(chat_id: int) -> Path:
    return DATA_DIR / f"{chat_id}.jsonl"

def load_memory(chat_id: int) -> List[Dict[str, str]]:
    if chat_id not in chat_memory:
        chat_memory[chat_id] = [{"role": "system", "content": "You are Alex, a helpful assistant."}]
        # Warm start from last ~MAX_HISTORY-1 archived messages (optional)
        ap = archive_path(chat_id)
        if ap.exists():
            try:
                lines = ap.read_text(encoding="utf-8").splitlines()
                # take only the last MAX_HISTORY-1 user/assistant messages
                tail = []
                for ln in lines[-(MAX_HISTORY-1):]:
                    try:
                        tail.append(json.loads(ln))
                    except:
                        pass
                chat_memory[chat_id].extend(tail)
            except Exception as e:
                print(f"[Archive] Load failed for {chat_id}: {e}")
    return chat_memory[chat_id]

def add_msg(chat_id: int, role: str, content: str):
    hist = load_memory(chat_id)
    hist.append({"role": role, "content": content})
    # Trim to MAX_HISTORY (keep system at index 0)
    while len(hist) > MAX_HISTORY:
        hist.pop(1)
    # Append to archive (JSON Lines)
    try:
        with archive_path(chat_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"ts": datetime.utcnow().isoformat() + "Z", "role": role, "content": content},
                ensure_ascii=False
            ) + "\n")
    except Exception as e:
        print(f"[Archive] Write failed for {chat_id}: {e}")

# ---------- Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Hey {USER_NAME}, {BOT_NAME} is online ✅")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id  # ✅ PTB v20-safe
    user_text = update.message.text

    add_msg(chat_id, "user", user_text)
    try:
        history = load_memory(chat_id)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            max_tokens=300
        )
        reply = resp.choices[0].message.content.strip()
        add_msg(chat_id, "assistant", reply)
        await update.message.reply_text(reply)
    except Exception as e:
        # Fallback to simple echo if OpenAI fails
        msg = f"⚠️ OpenAI error: {e}\nI'll echo for now: {user_text}"
        add_msg(chat_id, "assistant", msg)
        await update.message.reply_text(msg)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Owner-only: clear RAM memory (archive remains)."""
    if update.effective_user.id != OWNER_ID:
        return
    chat_id = update.effective_chat.id
    chat_memory.pop(chat_id, None)
    await update.message.reply_text("♻️ Memory cleared for this chat (archive preserved).")

# ---------- Run Telegram with Webhook ----------
def run_telegram():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # IMPORTANT: url_path must EXACTLY match what you put in webhook_url
    url_path = f"webhook/{TELEGRAM_BOT_TOKEN}"
    webhook_url = f"{PUBLIC_URL}/{url_path}"
    print(f"🚀 Setting webhook to {webhook_url}")

    # Railway usually provides PORT=8080; always bind to $PORT
    port = int(os.environ.get("PORT", "8080"))

    application.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=url_path,      # ✅ matches the URL below
        webhook_url=webhook_url,
        drop_pending_updates=True,
        stop_signals=None,      # be nice on Railway
    )

if __name__ == "__main__":
    run_telegram()
