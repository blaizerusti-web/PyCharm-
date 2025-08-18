# ---------- Alex (all-in-one, Railway-ready) ----------
import os, sys, json, time, threading, socket, logging, requests, asyncio, aiohttp, subprocess
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Env
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
PORT = int(os.getenv("PORT", "8080"))

client = OpenAI(api_key=OPENAI_API_KEY)
START_TIME = time.time()

# ---------- Memory ----------
memory_file = Path("memory.json")
if memory_file.exists():
    try:
        memory = json.loads(memory_file.read_text())
    except:
        memory = {"persona": {}, "history": []}
else:
    memory = {"persona": {}, "history": []}

def save_memory():
    memory_file.write_text(json.dumps(memory, indent=2))

# ---------- AI core ----------
async def ask_ai(prompt: str, context: str = "") -> str:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content": context or "You are Alex, helpful, witty, always on."},
                      {"role":"user","content": prompt}],
            max_tokens=500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"AI error: {e}")
        return f"(AI error: {e})"

# ---------- Web crawler ----------
async def fetch_url(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}) as r:
                if r.status != 200:
                    return f"⚠️ Failed ({r.status})"
                text = await r.text()
                soup = BeautifulSoup(text, "html.parser")
                title = soup.title.string.strip() if soup.title else "No title"
                desc = (soup.find("meta", {"name":"description"}) or {}).get("content","")
                h1 = soup.h1.get_text(strip=True) if soup.h1 else ""
                words = len(soup.get_text().split())
                links = len(soup.find_all("a"))
                images = len(soup.find_all("img"))
                snippet = soup.get_text()[:1500]
                return (f"🌐 {title}\n"
                        f"Desc: {desc[:150]}\nH1: {h1}\n"
                        f"Words:{words} Links:{links} Images:{images}\n\n"
                        f"Snippet:\n{snippet}")
    except Exception as e:
        return f"⚠️ Crawl error: {e}"

async def analyze_url(url: str) -> str:
    content = await fetch_url(url)
    if content.startswith("⚠️"): return content
    summary = await ask_ai(f"Summarize and extract SEO notes:\n\n{content}")
    return summary

# ---------- Commands ----------
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hey, I'm Alex 🤖 Always on.")

async def uptime(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = int(time.time()-START_TIME)
    h,m,s = u//3600, (u%3600)//60, u%60
    await update.message.reply_text(f"⏱️ Uptime {h}h {m}m {s}s")

async def ai_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = " ".join(ctx.args)
    if not q: return await update.message.reply_text("Usage: /ai your question")
    ans = await ask_ai(q)
    await update.message.reply_text(ans)

async def analyze_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args: return await update.message.reply_text("Usage: /analyze <url>")
    url = ctx.args[0]
    await update.message.reply_text("🔍 Crawling...")
    result = await analyze_url(url)
    await update.message.reply_text(result)

# ---------- Auto link + search ----------
async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if not text: return
    if text.lower().startswith("search "):
        query = text[7:]
        if not SERPAPI_KEY:
            return await update.message.reply_text("No SERPAPI_KEY set.")
        try:
            r = requests.get("https://serpapi.com/search", params={"q":query,"hl":"en","api_key":SERPAPI_KEY})
            j = r.json()
            snippet = j.get("organic_results",[{}])[0].get("snippet","(no results)")
            return await update.message.reply_text(f"🔎 {query}\n{snippet}")
        except Exception as e:
            return await update.message.reply_text(f"Error: {e}")
    if text.startswith("http://") or text.startswith("https://"):
        await update.message.reply_text("🔍 Got your link — analyzing...")
        result = await analyze_url(text)
        await update.message.reply_text(result)
    else:
        ans = await ask_ai(text)
        await update.message.reply_text(ans)

# ---------- Self-learning worker ----------
def learning_worker():
    while True:
        try:
            if memory["history"]:
                snippet = " ".join(memory["history"][-5:])
                persona_update = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":"Update persona from last lines"},
                              {"role":"user","content": snippet}]
                ).choices[0].message.content
                memory["persona"]["notes"] = persona_update
                save_memory()
        except Exception as e:
            logging.error(f"Learning error {e}")
        time.sleep(60)

# ---------- Health server ----------
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(b"ok")
def run_health():
    HTTPServer(("0.0.0.0", PORT), H).serve_forever()

# ---------- Main ----------
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("uptime", uptime))
    app.add_handler(CommandHandler("ai", ai_cmd))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    threading.Thread(target=learning_worker, daemon=True).start()
    threading.Thread(target=run_health, daemon=True).start()

    logging.info("Bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()