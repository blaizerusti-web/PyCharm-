# ---------- Alex (all-in-one: env vars, OpenAI, webhook, memory + archiving + full internet autonomy) ----------
import os, sys, json, time, threading, subprocess, requests, asyncio, traceback
from pathlib import Path
from typing import List, Dict, Any, Set
from datetime import datetime, timedelta
import random

# =======================================
# Auto-install (best-effort, no crash)
# =======================================
def install_requirements():
    try:
        import pkg_resources
        want = {
            "python-telegram-bot": ">=20.0.0",
            "openai": ">=1.0.0",
            "beautifulsoup4": ">=4.12.0",
            "lxml": ">=4.9.0",
            "feedparser": ">=6.0.0",
        }
        if Path("requirements.txt").exists():
            with open("requirements.txt") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln and not ln.startswith("#"):
                        name = ln.split("==")[0].split(">=")[0].lower()
                        want[name] = ln.split(" ", 1)[0].split(";", 1)[0]
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = []
        for pkg, spec in want.items():
            base = pkg.lower()
            if base not in installed:
                missing.append(spec if any(op in spec for op in ["==", ">="]) else pkg)
        if missing:
            print("[Installer] Installing:", missing)
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except Exception as e:
        print(f"[Installer] skipped: {e}")

install_requirements()

# =======================================
# Imports
# =======================================
from bs4 import BeautifulSoup
import feedparser

from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# OpenAI (new client)
from openai import OpenAI

# =======================================
# Config (env)
# =======================================
BOT_NAME = os.getenv("BOT_NAME", "Alex")
USER_NAME = os.getenv("USER_NAME", "Blaize")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://pycharm-production.up.railway.app")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Autonomy & alerts
CRYPTO_ALERT_PCT = float(os.getenv("CRYPTO_ALERT_PCT", "4"))   # % move triggers alert
HOURLY_TOPICS = os.getenv("HOURLY_TOPICS", "latest world news, crypto market prices, tech headlines, AI research updates, financial markets today, weather in Melbourne").split(",")
WEEKLY_DAY = int(os.getenv("WEEKLY_DAY", "0"))  # 0=Monday ... 6=Sunday
TIMEZONE_OFFSET_MIN = int(os.getenv("TZ_OFFSET_MIN", "0"))  # optional, for timestamp labeling

required = {
    "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
    "OWNER_ID": OWNER_ID if OWNER_ID != 0 else None,
    "OPENAI_API_KEY": OPENAI_API_KEY,
}
missing = [k for k, v in required.items() if not v]
if missing:
    raise ValueError(f"❌ Missing required env vars: {', '.join(missing)}")

print("DEBUG: BOT_NAME        =", BOT_NAME)
print("DEBUG: USER_NAME       =", USER_NAME)
print("DEBUG: TELEGRAM_BOT    =", "SET" if TELEGRAM_BOT_TOKEN else "MISSING")
print("DEBUG: OWNER_ID        =", OWNER_ID)
print("DEBUG: PUBLIC_URL      =", PUBLIC_URL)
print("DEBUG: OPENAI_API_KEY  =", "SET" if OPENAI_API_KEY else "MISSING")
print("DEBUG: CRYPTO_ALERT_PCT=", CRYPTO_ALERT_PCT)

# =======================================
# OpenAI client
# =======================================
client = OpenAI(api_key=OPENAI_API_KEY)

# =======================================
# Data dirs & persistence
# =======================================
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
STATE_DIR = Path("state"); STATE_DIR.mkdir(exist_ok=True)
WATCH_FILE = STATE_DIR / "watch.json"
CHATS_FILE = STATE_DIR / "chats.json"

def now_utc_iso():
    return datetime.utcnow().isoformat() + "Z"

def load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except: pass
    return default

def save_json(path: Path, data: Any):
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[Persist] Save failed {path}: {e}")

# Known chats (for broadcast autonomy reports)
known_chats: Set[int] = set(load_json(CHATS_FILE, []))

def remember_chat(chat_id: int):
    if chat_id not in known_chats:
        known_chats.add(chat_id)
        save_json(CHATS_FILE, list(known_chats))

# Watchlist structure: topics + crypto tickers
watch = load_json(WATCH_FILE, {
    "topics": ["AI regulation", "Bitcoin", "OpenAI", "Ethereum"],
    "crypto": ["bitcoin", "ethereum", "solana"]
})
def save_watch():
    save_json(WATCH_FILE, watch)

# =======================================
# Memory + Archiving
# =======================================
chat_memory: Dict[int, List[Dict[str, str]]] = {}
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "64"))

def archive_path(chat_id: int) -> Path:
    return DATA_DIR / f"{chat_id}.jsonl"

def load_memory(chat_id: int) -> List[Dict[str, str]]:
    if chat_id not in chat_memory:
        base = [{"role": "system", "content": "You are Alex, an autonomous assistant with internet access. Be concise and helpful."}]
        ap = archive_path(chat_id)
        if ap.exists():
            try:
                lines = ap.read_text(encoding="utf-8").splitlines()
                history = []
                for ln in lines[-(MAX_HISTORY-1):]:
                    try:
                        rec = json.loads(ln)
                        # backward-compat for older archives
                        msg = {"role": rec.get("role","assistant"), "content": rec.get("content","")}
                        history.append(msg)
                    except: pass
                base.extend(history)
            except Exception as e:
                print(f"[Archive] Load failed for {chat_id}: {e}")
        chat_memory[chat_id] = base
    return chat_memory[chat_id]

def add_msg(chat_id: int, role: str, content: str):
    hist = load_memory(chat_id)
    hist.append({"role": role, "content": content})
    while len(hist) > MAX_HISTORY:
        hist.pop(1)
    try:
        with archive_path(chat_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"ts": now_utc_iso(), "role": role, "content": content},
                ensure_ascii=False
            ) + "\n")
    except Exception as e:
        print(f"[Archive] Write failed for {chat_id}: {e}")

# =======================================
# Internet Tools
# =======================================
def ddg_instant_answer(query: str) -> str:
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        r = requests.get(url, params=params, timeout=12)
        data = r.json()
        if data.get("AbstractText"):
            return data["AbstractText"]
        if data.get("RelatedTopics"):
            for t in data["RelatedTopics"]:
                if isinstance(t, dict) and t.get("Text"):
                    return t["Text"]
        return "No instant result found."
    except Exception as e:
        return f"⚠️ Search failed: {e}"

def fetch_url(url: str, limit=2000) -> str:
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "AlexBot/1.0"})
        text = r.text
        if "html" in r.headers.get("content-type", "").lower():
            soup = BeautifulSoup(text, "lxml")
            for tag in soup(["script","style","noscript"]): tag.decompose()
            text = soup.get_text(separator="\n")
        text = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])
        return text[:limit] if len(text) > limit else text
    except Exception as e:
        return f"⚠️ Fetch failed: {e}"

def fetch_rss(url: str, limit=3) -> str:
    try:
        fp = feedparser.parse(url)
        items = []
        for e in fp.entries[:limit]:
            items.append(f"- {e.get('title','(no title)')} — {e.get('link','')}")
        return "\n".join(items) if items else "No items."
    except Exception as e:
        return f"⚠️ RSS failed: {e}"

def crypto_simple_prices(ids: List[str]) -> Dict[str, float]:
    """Coingecko simple price (no API key). ids like ['bitcoin','ethereum']"""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": ",".join(ids), "vs_currencies": "usd"}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        out = {}
        for k, v in data.items():
            out[k] = float(v.get("usd", 0.0))
        return out
    except Exception as e:
        print(f"[Crypto] price error: {e}")
        return {}

# Track last crypto prices to detect moves
_last_prices = {}  # {"bitcoin": 62000.0 ...}

def check_crypto_alerts() -> List[str]:
    global _last_prices
    ids = list(set(watch.get("crypto", [])))
    if not ids: return []
    current = crypto_simple_prices(ids)
    alerts = []
    for cid, price in current.items():
        old = _last_prices.get(cid)
        _last_prices[cid] = price
        if old and price and old > 0:
            pct = ((price - old) / old) * 100.0
            if abs(pct) >= CRYPTO_ALERT_PCT:
                alerts.append(f"🔔 {cid.title()} moved {pct:.2f}% — now ${price:,.2f} (was ${old:,.2f})")
    return alerts

# =======================================
# Telegram Handlers
# =======================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    remember_chat(update.effective_chat.id)
    await update.message.reply_text(
        f"Hey {USER_NAME}, {BOT_NAME} is online ✅\n"
        f"- Memory & archive active\n- Webhook mode\n- Internet tools ready\n"
        f"Try: /watch, /status, /export, or just chat.\n"
        f"Autonomy is running hourly/daily/weekly."
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    remember_chat(update.effective_chat.id)
    msg = (
        f"🧠 {BOT_NAME} status\n"
        f"- Known chats: {len(known_chats)}\n"
        f"- Watch topics: {', '.join(watch.get('topics', []))}\n"
        f"- Watch crypto: {', '.join(watch.get('crypto', []))}\n"
        f"- Crypto alert threshold: {CRYPTO_ALERT_PCT}%\n"
        f"- Max history per chat: {MAX_HISTORY}\n"
        f"- Public URL: {PUBLIC_URL}\n"
    )
    await update.message.reply_text(msg)

async def export_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    remember_chat(update.effective_chat.id)
    chat_id = update.effective_chat.id
    path = archive_path(chat_id)
    if not path.exists():
        await update.message.reply_text("No archive yet for this chat.")
        return
    try:
        await update.message.reply_document(document=InputFile(str(path)), filename=f"{chat_id}.jsonl")
    except Exception as e:
        await update.message.reply_text(f"Export failed: {e}")

async def watch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ /watch add topic|crypto <value>   or   /watch remove topic|crypto <value>   or   /watch list """
    remember_chat(update.effective_chat.id)
    args = context.args
    if not args:
        await update.message.reply_text("Usage:\n/watch add topic <term>\n/watch add crypto <id>\n/watch remove topic|crypto <value>\n/watch list")
        return
    action = args[0].lower()
    if action == "list":
        await update.message.reply_text(
            "👀 Watchlist\nTopics:\n- " + "\n- ".join(watch.get("topics", [])) +
            "\nCrypto:\n- " + "\n- ".join(watch.get("crypto", []))
        )
        return
    if len(args) < 3:
        await update.message.reply_text("Provide type and value. e.g. /watch add topic Nvidia")
        return
    wtype = args[1].lower()
    value = " ".join(args[2:]).strip()
    if action == "add":
        if wtype == "topic":
            watch.setdefault("topics", [])
            if value not in watch["topics"]:
                watch["topics"].append(value)
        elif wtype == "crypto":
            watch.setdefault("crypto", [])
            if value not in watch["crypto"]:
                watch["crypto"].append(value)
        save_watch()
        await update.message.reply_text("✅ Added.")
    elif action == "remove":
        try:
            if wtype == "topic":
                watch.setdefault("topics", [])
                watch["topics"] = [t for t in watch["topics"] if t.lower() != value.lower()]
            elif wtype == "crypto":
                watch.setdefault("crypto", [])
                watch["crypto"] = [c for c in watch["crypto"] if c.lower() != value.lower()]
            save_watch()
            await update.message.reply_text("✅ Removed.")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
    else:
        await update.message.reply_text("Unknown action. Use add/remove/list.")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Owner-only: clear RAM memory (archive remains)."""
    if update.effective_user.id != OWNER_ID:
        return
    chat_id = update.effective_chat.id
    chat_memory.pop(chat_id, None)
    await update.message.reply_text("♻️ Memory cleared for this chat (archive preserved).")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    remember_chat(update.effective_chat.id)
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    user_text = update.message.text

    add_msg(chat_id, "user", user_text)
    try:
        history = load_memory(chat_id)
        # Give Alex tool-use affordances
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history + [
                {"role":"system","content":(
                    "You can request tools by emitting tags:\n"
                    "[[search:QUERY]] for DuckDuckGo instant answer\n"
                    "[[fetch:URL]] to fetch raw page text\n"
                    "[[rss:URL]] to read RSS headlines\n"
                    "[[crypto:ID1,ID2]] for Coingecko prices (ids like bitcoin,ethereum)\n"
                    "Only emit tags if needed; otherwise answer directly."
                )}
            ],
            max_tokens=500
        )
        reply = resp.choices[0].message.content.strip()

        # Auto-execute tool tags (can be multiple)
        # search
        while "[[search:" in reply:
            q = reply.split("[[search:")[1].split("]]")[0]
            result = ddg_instant_answer(q)
            reply = reply.replace(f"[[search:{q}]]", f"🔎 {q} → {result}")
        # fetch
        while "[[fetch:" in reply:
            url = reply.split("[[fetch:")[1].split("]]")[0]
            content = fetch_url(url)
            snippet = (content[:500] + "…") if len(content) > 500 else content
            reply = reply.replace(f"[[fetch:{url}]]", f"📄 {url}\n{snippet}")
        # rss
        while "[[rss:" in reply:
            url = reply.split("[[rss:")[1].split("]]")[0]
            items = fetch_rss(url, limit=5)
            reply = reply.replace(f"[[rss:{url}]]", f"📰 {url}\n{items}")
        # crypto
        while "[[crypto:" in reply:
            ids = reply.split("[[crypto:")[1].split("]]")[0]
            ids_list = [i.strip().lower() for i in ids.split(",") if i.strip()]
            prices = crypto_simple_prices(ids_list)
            if prices:
                line = " | ".join([f"{k.title()}: ${v:,.2f}" for k,v in prices.items()])
            else:
                line = "No prices."
            reply = reply.replace(f"[[crypto:{ids}]]", f"₿ {line}")

        add_msg(chat_id, "assistant", reply)
        await update.message.reply_text(reply)
    except Exception as e:
        err = f"⚠️ Error: {e}"
        add_msg(chat_id, "assistant", err)
        await update.message.reply_text(err)

# =======================================
# Autonomy Engine (hourly / daily / weekly)
# =======================================
def tz_now():
    return datetime.utcnow() + timedelta(minutes=TIMEZONE_OFFSET_MIN)

def broadcast(application: Application, text: str):
    for cid in list(known_chats):
        try:
            asyncio.run_coroutine_threadsafe(
                application.bot.send_message(chat_id=cid, text=text),
                application.bot.loop
            )
        except Exception as e:
            print(f"[Broadcast] {cid} failed: {e}")

def hourly_job(application: Application):
    try:
        # 1) Random topical scan
        topic = random.choice([t.strip() for t in HOURLY_TOPICS if t.strip()])
        result = ddg_instant_answer(topic)
        text = f"(Hourly) 🔎 {topic}\n{result}"
        broadcast(application, text)

        # 2) Crypto alerts (if movement)
        alerts = check_crypto_alerts()
        if alerts:
            broadcast(application, "\n".join(["(Hourly) " + a for a in alerts]))

        # 3) Watch topics ping (first 3)
        if watch.get("topics"):
            for t in random.sample(watch["topics"], k=min(2, len(watch["topics"]))):
                r = ddg_instant_answer(t)
                broadcast(application, f"(Watch) 👀 {t}\n{r}")
    except Exception as e:
        print("[Autonomy:hourly]", e, traceback.format_exc())
    finally:
        threading.Timer(3600, hourly_job, args=(application,)).start()

def daily_job(application: Application):
    try:
        for cid in list(known_chats):
            hist = load_memory(cid)
            if len(hist) < 4: 
                continue
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":"Summarize the last day for the user with actionable highlights."}] + hist[-30:],
                    max_tokens=220
                )
                summary = resp.choices[0].message.content.strip()
                add_msg(cid, "assistant", f"(Daily summary) {summary}")
                asyncio.run_coroutine_threadsafe(
                    application.bot.send_message(chat_id=cid, text=f"(Daily summary)\n{summary}"),
                    application.bot.loop
                )
            except Exception as e:
                print(f"[Autonomy:daily] {cid} {e}")
    finally:
        # schedule next in ~24h
        threading.Timer(86400, daily_job, args=(application,)).start()

def weekly_job(application: Application):
    try:
        today = tz_now().weekday()
        if today == WEEKLY_DAY:
            for cid in list(known_chats):
                hist = load_memory(cid)
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"system","content":"Create a concise weekly report: key events, trends, pending tasks, and proposed next actions."}] + hist[-60:],
                        max_tokens=350
                    )
                    report = resp.choices[0].message.content.strip()
                    add_msg(cid, "assistant", f"(Weekly report) {report}")
                    asyncio.run_coroutine_threadsafe(
                        application.bot.send_message(chat_id=cid, text=f"(Weekly report)\n{report}"),
                        application.bot.loop
                    )
                except Exception as e:
                    print(f"[Autonomy:weekly] {cid} {e}")
    finally:
        # check again in 24h
        threading.Timer(86400, weekly_job, args=(application,)).start()

# =======================================
# Run Telegram with Webhook
# =======================================
def run_telegram():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("export", export_chat))
    application.add_handler(CommandHandler("watch", watch_cmd))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    url_path = f"webhook/{TELEGRAM_BOT_TOKEN}"
    webhook_url = f"{PUBLIC_URL}/{url_path}"
    print(f"🚀 Setting webhook to {webhook_url}")

    port = int(os.environ.get("PORT", "8080"))

    # Kick off autonomy loops shortly after boot
    threading.Timer(10, hourly_job, args=(application,)).start()
    threading.Timer(20, daily_job, args=(application,)).start()
    threading.Timer(30, weekly_job, args=(application,)).start()

    application.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=url_path,
        webhook_url=webhook_url,
        drop_pending_updates=True,
        stop_signals=None,
    )

if __name__ == "__main__":
    run_telegram()
