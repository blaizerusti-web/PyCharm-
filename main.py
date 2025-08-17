# ---------- Alex (Telegram bot with uptime + AI logging) ----------
import os, time, logging, csv
from datetime import datetime
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from openai import OpenAI

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Bot token & OpenAI key
TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE"))

# Track uptime
start_time = time.time()

def get_uptime():
    uptime = int(time.time() - start_time)
    hours, remainder = divmod(uptime, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"

# File to store logs
LOG_FILE = "ai_conversations.csv"

# Ensure CSV file has headers
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "username", "user_id", "query", "reply"])

def log_conversation(username, user_id, query, reply):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), username, user_id, query, reply])

# ---- Commands ----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Alex is awake and ready ✅")

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
        # Call GPT model
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}]
        )
        reply = completion.choices[0].message.content.strip()

        # Send reply to user
        await update.message.reply_text(reply)

        # Log conversation
        log_conversation(username, user_id, query, reply)

    except Exception as e:
        logging.error(f"AI error: {e}")
        await update.message.reply_text("⚠️ Sorry, AI request failed.")

# Error handler
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logging.error(msg="Exception while handling update:", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("⚠️ Oops, something went wrong!")

# ---- Main ----
def main():
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ai", ai))

    # Custom wake command
    app.add_handler(MessageHandler(filters.Regex(r'(?i)^you there\?$'), ping))

    # Error handling
    app.add_error_handler(error_handler)

    # Run bot
    logging.info("🤖 Alex is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
