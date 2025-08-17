# ---------- Alex (clean, autonomous, always-online) ----------
import os, time, json, threading
import requests
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from duckduckgo_search import DDGS
import feedparser
import openai

# --- Flask server for Railway keep-alive ---
app = Flask(__name__)

@app.route("/")
def home():
    return "Alex is alive and running 24/7!"

# --- Config from environment variables ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")   # e.g. https://your-app.up.railway.app/webhook
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# --- Telegram bot setup ---
application = Application.builder().token(TELEGRAM_TOKEN).build()

# --- Commands ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hey, I’m Alex — running 24/7 and ready to go!")

async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Usage: /search <your query>")
        return
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=3)]
    if results:
        reply = "\n\n".join([f"🔎 {r['title']}\n{r['href']}" for r in results])
    else:
        reply = "No results found."
    await update.message.reply_text(reply)

async def rss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    feed_url = "https://news.google.com/rss"
    feed = feedparser.parse(feed_url)
    articles = "\n\n".join([f"📰 {e.title}\n{e.link}" for e in feed.entries[:5]])
    await update.message.reply_text(articles)

async def ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Usage: /ai <your prompt>")
        return
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}]
    )
    await update.message.reply_text(response["choices"][0]["message"]["content"])

# --- Handlers ---
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("search", search))
application.add_handler(CommandHandler("rss", rss))
application.add_handler(CommandHandler("ai", ai))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ai))

# --- Run bot in background thread ---
def run_bot():
    application.run_webhook(
        listen="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        url_path="webhook",
        webhook_url=WEBHOOK_URL
    )

threading.Thread(target=run_bot, daemon=True).start()

# --- Keep Flask alive (needed by Railway) ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
