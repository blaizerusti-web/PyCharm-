# ---------- Alex (all-in-one, env vars, OpenAI integrated, webhook ready) ----------
import os, sys, json, time, threading, subprocess, socket, asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import time as dtime

# ---------- Auto-install (Railway/Replit self-heal) ----------
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
from flask import Flask
import requests
from bs4 import BeautifulSoup

# DuckDuckGo optional
try:
    from duckduckgo_search import DDGS
    DUCK_OK = True
except:
    DUCK_OK = False

# Telegram
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

# Google Sheets optional
SHEETS_OK = True
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
except:
    SHEETS_OK = False

# OpenAI
import openai

# ---------- Config (from environment variables) ----------
BOT_NAME = os.getenv("BOT_NAME", "Alex")
USER_NAME = os.getenv("USER_NAME", "Blaize")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://pycharm-production.up.railway.app")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------- Fail-fast checks ----------
required_vars = {
    "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
    "OWNER_ID": OWNER_ID if OWNER_ID != 0 else None,
    "OPENAI_API_KEY": OPENAI_API_KEY,
}
missing = [key for key, val in required_vars.items() if not val]
if missing:
    raise ValueError(f"❌ Missing required environment variables: {', '.join(missing)}")

# Debug
print("DEBUG: BOT_NAME =", BOT_NAME)
print("DEBUG: USER_NAME =", USER_NAME)
print("DEBUG: TELEGRAM_BOT_TOKEN =", "SET" if TELEGRAM_BOT_TOKEN else "MISSING")
print("DEBUG: OWNER_ID =", OWNER_ID)
print("DEBUG: GOOGLE_SHEET_ID =", GOOGLE_SHEET_ID)
print("DEBUG: PUBLIC_URL =", PUBLIC_URL)
print("DEBUG: OPENAI_API_KEY =", "SET" if OPENAI_API_KEY else "MISSING")

# OpenAI key
openai.api_key = OPENAI_API_KEY

# ---------- Flask ----------
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Alex is running!"

@app.route("/health")
def health():
    return "ok"

# ---------- Telegram Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Hey {USER_NAME}, {BOT_NAME} is online ✅")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":"You are Alex, a helpful assistant."},
                      {"role":"user","content":user_text}]
        )
        reply = response.choices[0].message.content
        await update.message.reply_text(reply)
    except Exception as e:
        await update.message.reply_text(f"❌ OpenAI error: {e}")

# ---------- Run Telegram with Webhook ----------
def run_telegram():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Webhook URL
    webhook_url = f"{PUBLIC_URL}/webhook/{TELEGRAM_BOT_TOKEN}"
    print(f"🚀 Setting webhook to {webhook_url}")

    application.run_webhook(
        listen="0.0.0.0",
        port=int(os.environ.get("PORT", 3000)),
        url_path=f"{TELEGRAM_BOT_TOKEN}",
        webhook_url=webhook_url
    )

if __name__ == "__main__":
    run_telegram()
