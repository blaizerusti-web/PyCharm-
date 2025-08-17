# ---------- Alex (all-in-one, env vars, OpenAI integrated) ----------
import os, sys, json, time, threading, subprocess, socket, asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import time as dtime   # ✅ example import, safe to leave in

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
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
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
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")  # if using service account JSON
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
PUBLIC_URL = os.getenv("PUBLIC_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------- Debug Env Vars ----------
print("DEBUG: BOT_NAME =", BOT_NAME)
print("DEBUG: USER_NAME =", USER_NAME)
print("DEBUG: TELEGRAM_BOT_TOKEN =", "SET" if TELEGRAM_BOT_TOKEN else "MISSING")
print("DEBUG: OWNER_ID =", OWNER_ID)
print("DEBUG: GOOGLE_SHEET_ID =", GOOGLE_SHEET_ID)
print("DEBUG: PUBLIC_URL =", PUBLIC_URL)
print("DEBUG: OPENAI_API_KEY =", "SET" if OPENAI_API_KEY else "MISSING")

# Auto-detect public URL if not set
if not PUBLIC_URL:
    try:
        host = socket.gethostname()
        PUBLIC_URL = f"https://{host}.id.repl.co"
        print(f"[Auto-URL] Using {PUBLIC_URL}")
    except Exception as e:
        print(f"[Auto-URL] Could not detect: {e}")

# ✅ Fail fast if token missing
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("❌ Missing TELEGRAM_BOT_TOKEN — set this in Railway/GitHub env vars")

# ✅ Set OpenAI key safely
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    print("⚠️ Warning: No OPENAI_API_KEY set, AI replies won’t work")

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

# ---------- Telegram Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Hey {USER_NAME}, {BOT_NAME} is online ✅")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not OPENAI_API_KEY:
        await update.message.reply_text("⚠️ No OpenAI API key set, can’t generate AI replies")
        return
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

# ---------- Run Telegram + Flask ----------
def run_telegram():
    app_builder = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app_builder.add_handler(CommandHandler("start", start))
    app_builder.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    app_builder.run_polling()

if __name__ == "__main__":
    # Run Flask in background thread
    threading.Thread(target=run_flask, daemon=True).start()
    # Run Telegram bot
    run_telegram()
