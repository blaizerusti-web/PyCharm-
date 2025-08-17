# ---------- Alex (clean, autonomous, always-online, self-healing) ----------
import os, json, threading, asyncio, time, logging
import requests
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

# --- Flask server for Railway keep-alive ---
app = Flask(__name__)

@app.route("/")
def home():
    return "Alex is alive, self-healing, and running 24/7!"

# --- Config from environment variables ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# --- Telegram bot setup ---
application = Application.builder().token(TELEGRAM_TOKEN).build()

# --- Commands ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hey, I’m Alex — alive 24/7, logging everything, and self-healing!")

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
        feed_url = "https://news.google.com/rss"
        feed = feedparser.parse(feed_url)
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

# --- Handlers ---
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("search", search))
application.add_handler(CommandHandler("rss", rss))
application.add_handler(CommandHandler("ai", ai))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ai))

# --- Bot runner with auto-restart (self-healing loop) ---
def run_bot():
    while True:
        try:
            logger.info("Starting Alex bot (polling mode)...")
            asyncio.run(application.run_polling())
        except Exception as e:
            logger.error(f"Bot crashed with error: {e}. Restarting in 5s...")
            time.sleep(5)  # wait before restart to avoid crash loop

# --- Background thread for bot ---
threading.Thread(target=run_bot, daemon=True).start()

# --- Keep Flask alive (needed by Railway) ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
