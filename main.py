# ---------- Alex (webhook, Railway-optimized, logging, always-online) ----------
import os, json, logging
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from duckduckgo_search import DDGS
import feedparser
import openai

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("Alex")

# --- Flask server ---
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Alex is alive and running on Railway (webhook mode)!"

# --- Config from environment variables ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAILWAY_URL = os.getenv("RAILWAY_URL")  # e.g. https://your-app.up.railway.app
openai.api_key = OPENAI_API_KEY

# --- Telegram app ---
application = Application.builder().token(TELEGRAM_TOKEN).build()

# --- Commands ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hey, I’m Alex — alive 24/7 on Railway, webhook mode!")

async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Usage: /search <your query>")
        return
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
        reply = "\n\n".join([f"🔎 {r['title']}\n{r['href']}" for r in results]) if results else "No results found."
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
    query = update.message.text
    if not query:
        await update.message.reply_text("⚠️ Please type something after /ai or just send a message.")
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

# --- Telegram webhook route ---
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), application.bot)
    application.update_queue.put_nowait(update)
    return "ok"

# --- Start webhook on Railway ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("🚀 Starting Alex in webhook mode on Railway...")
    application.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=TELEGRAM_TOKEN,
        webhook_url=f"{RAILWAY_URL}/{TELEGRAM_TOKEN}"
    )
    app.run(host="0.0.0.0", port=port)
