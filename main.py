# ---------- Alex (all-in-one with keep-alive + uptime ping) ----------
import os, asyncio, logging, threading, time, requests
from datetime import timedelta
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from duckduckgo_search import DDGS
import feedparser
import openai

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("Alex")

# --- Flask server (for Railway keep-alive) ---
app = Flask(__name__)

@app.route("/")
def home():
    return "Alex is alive 🚀"

# --- Config from env vars ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAILWAY_URL = os.getenv("RAILWAY_URL")  # e.g. https://your-app.up.railway.app
openai.api_key = OPENAI_API_KEY

# --- Uptime tracker ---
START_TIME = time.time()

def get_uptime():
    uptime_seconds = int(time.time() - START_TIME)
    return str(timedelta(seconds=uptime_seconds))

# --- Telegram bot setup ---
application = Application.builder().token(TELEGRAM_TOKEN).build()

# --- Commands ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hey, I’m Alex — alive 24/7 hybrid mode ready!")

async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Usage: /search <your query>")
        return
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
        if results:
            reply = "\n\n".join([f"🔎 {r['title']}\n{r['href']}" for r in results])
        else:
            reply = "No results found."
    except Exception as e:
        logger.error(f"Search error: {e}")
        reply = "⚠️ Error fetching results."
    await update.message.reply_text(reply)

async def rss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        feed = feedparser.parse("https://news.google.com/rss")
        articles = "\n\n".join([f"📰 {e.title}\n{e.link}" for e in feed.entries[:5]])
        await update.message.reply_text(articles)
    except Exception as e:
        logger.error(f"RSS error: {e}")
        await update.message.reply_text("⚠️ Error fetching RSS feed.")

async def ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Usage: /ai <your prompt>")
        return
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}]
        )
        await update.message.reply_text(response["choices"][0]["message"]["content"])
    except Exception as e:
        logger.error(f"AI error: {e}")
        await update.message.reply_text("⚠️ AI request failed.")

# --- Uptime ping: "you there?" ---
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uptime_str = get_uptime()
    await update.message.reply_text(
        f"Yep, I’m here and alive ✅\n⏱️ Uptime: {uptime_str}"
    )

# --- Handlers ---
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("search", search))
application.add_handler(CommandHandler("rss", rss))
application.add_handler(CommandHandler("ai", ai))

# Custom "you there?" trigger
application.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"^(?i)you there\?$"), status))

# Default: any other text → AI
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ai))

# --- Keep-alive (self-ping) ---
def keep_alive():
    if not RAILWAY_URL:
        return
    while True:
        try:
            requests.get(RAILWAY_URL)
            logger.info("Keep-alive ping sent ✅")
        except Exception as e:
            logger.warning(f"Keep-alive failed: {e}")
        time.sleep(300)  # every 5 min

# --- Hybrid runner ---
async def main():
    if RAILWAY_URL:  # 🚀 On Railway → webhook
        threading.Thread(target=keep_alive, daemon=True).start()
        logger.info("Running in WEBHOOK mode...")
        await application.run_webhook(
            listen="0.0.0.0",
            port=int(os.environ.get("PORT", 8443)),
            url_path=TELEGRAM_TOKEN,
            webhook_url=f"{RAILWAY_URL}/{TELEGRAM_TOKEN}"
        )
    else:  # 🖥️ Local → polling
        logger.info("Running in POLLING mode...")
        await application.run_polling()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")
