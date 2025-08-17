# ---------- Alex (All-in-One: Telegram + AI + Search + Logging + Uptime + Self-Heal) ----------
import os, time, logging, csv, requests, sys
from datetime import datetime
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from openai import OpenAI

# --- Logging setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# --- Load Environment Keys ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Exit early if any required key is missing
if not TELEGRAM_TOKEN:
    sys.exit("❌ TELEGRAM_TOKEN not set in environment!")
if not OPENAI_KEY:
    sys.exit("❌ OPENAI_API_KEY not set in environment!")

client = OpenAI(api_key=OPENAI_KEY)

# --- Uptime ---
start_time = time.time()
def get_uptime():
    uptime = int(time.time() - start_time)
    hours, remainder = divmod(uptime, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"

# --- CSV Logging ---
LOG_FILE = "ai_conversations.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "username", "user_id", "query", "reply"])

def log_conversation(username, user_id, query, reply):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), username, user_id, query, reply])

# --- Google Search via SerpAPI ---
def search_google(query):
    if not SERPAPI_KEY:
        return "⚠️ SERPAPI_KEY not set in environment!"
    url = "https://serpapi.com/search.json"
    params = {"q": query, "api_key": SERPAPI_KEY}
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        if "organic_results" in data:
            top = data["organic_results"][:3]
            results = "\n".join([f"{r['title']} - {r.get('link','')}" for r in top])
            return f"🔎 Top results for '{query}':\n{results}"
        else:
            return "⚠️ No results found."
    except Exception as e:
        return f"❌ Search failed: {e}"

# --- Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Alex is online and ready ✅\n\n⚡ Commands:\n- you there?\n- search <query>\n- /ai <query>")

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uptime = get_uptime()
    await update.message.reply_text(f"✅ I'm here! Uptime: {uptime}")

async def ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Please provide a query after /ai")
        return
    
    query = " ".join(context.args)
    username = update.message.from_user.username or "Unknown"
    user_id = update.message.from_user.id

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}]
        )
        reply = completion.choices[0].message.content.strip()
        await update.message.reply_text(reply)
        log_conversation(username, user_id, query, reply)
    except Exception as e:
        logging.error(f"AI error: {e}")
        await update.message.reply_text("⚠️ AI request failed.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    if text.lower() == "you there?":
        uptime = get_uptime()
        await update.message.reply_text(f"✅ Yes Blaize, I'm here.\n⏱ Uptime: {uptime}")

    elif text.lower().startswith("search "):
        query = text[7:].strip()
        results = search_google(query)
        await update.message.reply_text(results)

    else:
        await update.message.reply_text(
            "⚡ Commands:\n"
            "- you there?\n"
            "- search <query>\n"
            "- /ai <query>"
        )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logging.error(msg="Exception while handling update:", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("⚠️ Oops, something went wrong!")

# --- Main Bot Function ---
def run_bot():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("ai", ai))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logging.info("🚀 Alex is running...")
    app.run_polling()

# --- Self-Heal Loop ---
def main():
    while True:
        try:
            run_bot()
        except Exception as e:
            logging.error(f"💥 Bot crashed with error: {e}")
            logging.info("♻️ Restarting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    main()
