# ---------- Alex (All-in-One: Natural Chat + Live Web + Memory + Self-Learning + Voice) ----------
import os, sys, time, csv, json, threading, logging, tempfile, requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from openai import OpenAI

# ---------------- Logging ----------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("alex")

# ---------------- Keys / Client ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_KEY    = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY   = os.getenv("SERPAPI_KEY")  # optional but enables web search/news

if not TELEGRAM_TOKEN:
    sys.exit("❌ TELEGRAM_TOKEN not set in environment!")
if not OPENAI_KEY:
    sys.exit("❌ OPENAI_API_KEY not set in environment!")

client = OpenAI(api_key=OPENAI_KEY)

# ---------------- Files (persisted) ----------------
LOG_FILE     = "ai_conversations.csv"     # conversation transcript
STATE_FILE   = "alex_state.json"          # persona + long-term memory

# init CSV
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp_utc", "username", "user_id", "query", "reply"])

# init state
DEFAULT_PERSONA = (
    "You are Alex: friendly, concise, curious, and helpful. "
    "You answer naturally (like a smart friend), cite facts when relevant, "
    "admit uncertainty, and prefer actionable steps. You avoid fluff."
)
DEFAULT_STATE: Dict[str, Any] = {
    "persona": DEFAULT_PERSONA,
    "facts": [],          # enduring facts about user/preferences/workflows
    "notes": [],          # distilled takeaways from chats
    "last_news": [],      # cached recent news items
    "last_update_iso": None,
    "last_seen_row": 1,   # first data row in CSV is index 1
}

_state_lock = threading.Lock()

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        save_state(DEFAULT_STATE)
        return DEFAULT_STATE.copy()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in DEFAULT_STATE.items():  # fill any missing keys after updates
            data.setdefault(k, v)
        return data
    except Exception as e:
        log.error(f"Failed to read state file: {e}")
        return DEFAULT_STATE.copy()

def save_state(data: Dict[str, Any]) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Failed to write state file: {e}")

# ---------------- Uptime ----------------
start_time = time.time()
def get_uptime():
    s = int(time.time() - start_time)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s"

# ---------------- Conversation logging ----------------
def log_conversation(username: str, user_id: int, query: str, reply: str):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), username, user_id, query, reply])

# ---------------- Web Search / News (SerpAPI) ----------------
def search_google(query: str) -> str:
    if not SERPAPI_KEY:
        return "⚠️ Web search isn't enabled yet (missing SERPAPI_KEY)."
    try:
        res = requests.get(
            "https://serpapi.com/search.json",
            params={"q": query, "api_key": SERPAPI_KEY},
            timeout=15,
        )
        data = res.json()
        items = (data.get("organic_results") or [])[:5]
        if not items:
            return "No strong results found."
        lines = [f"- {it.get('title','(untitled)')} — {it.get('link','')}" for it in items]
        return "🔎 Top results:\n" + "\n".join(lines)
    except Exception as e:
        return f"❌ Search failed: {e}"

def fetch_news(topic: str = "technology") -> List[Dict[str, str]]:
    if not SERPAPI_KEY:
        return []
    try:
        res = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google_news", "q": topic, "api_key": SERPAPI_KEY},
            timeout=15,
        )
        data = res.json()
        stories = []
        for item in (data.get("news_results") or [])[:5]:
            stories.append({"title": item.get("title", ""), "link": item.get("link", "")})
        return stories
    except Exception as e:
        log.error(f"News fetch error: {e}")
        return []

# ---------------- Prompt construction ----------------
def build_system_prompt() -> str:
    with _state_lock:
        st = load_state()
        persona = st.get("persona", DEFAULT_PERSONA)
        facts = st.get("facts", [])[:12]
        notes = st.get("notes", [])[-12:]

    blocks = [
        persona,
        "Use the following long-term memory when relevant:",
        *[f"- {f}" for f in facts],
        "Recent distilled notes:",
        *[f"- {n}" for n in notes],
        "Style: warm, direct, practical; avoid filler; prefer short paragraphs and lists; ask clarifying only when critical."
    ]
    return "\n".join(blocks)

def gpt_reply(user_text: str) -> str:
    system_prompt = build_system_prompt()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.6,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"OpenAI error: {e}")
        return "I hit a snag talking to my brain. Try again in a moment."

# ---------------- Voice (on demand with “alex say …”) ----------------
async def tts_send(update: Update, text: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            speech = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=text[:4096],
            )
            tmp.write(speech.content)
            path = tmp.name
        with open(path, "rb") as f:
            await update.message.reply_voice(f)
        os.remove(path)
    except Exception as e:
        log.error(f"TTS error: {e}")
        await update.message.reply_text("Couldn't generate audio; sent text instead.")

# ---------------- Commands ----------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hey Blaize 👋 Alex is online — learning and evolving 24/7.")

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"✅ Alive. Uptime {get_uptime()}")

async def cmd_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("What should we think about? 🙂  Example: `/ai best laptop under 1k`")
        return
    q = " ".join(context.args)
    reply = gpt_reply(q)
    await update.message.reply_text(reply)
    u = update.message.from_user
    log_conversation(u.username or "Unknown", u.id, q, reply)

async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with _state_lock:
        st = load_state()
    persona = st.get("persona", DEFAULT_PERSONA)
    facts = st.get("facts", [])
    notes = st.get("notes", [])[-10:]
    msg = (
        f"🧠 **Persona**:\n{persona}\n\n"
        f"📌 **Facts** ({len(facts)}):\n" + ("\n".join([f"- {f}" for f in facts[:12]]) or "—") + "\n\n"
        f"🗒️ **Recent Notes**:\n" + ("\n".join([f"- {n}" for n in notes]) or "—")
    )
    await update.message.reply_text(msg)

async def cmd_learn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Teach me something to remember, e.g. `/learn I prefer short bullet answers.`")
        return
    fact = " ".join(context.args).strip()
    with _state_lock:
        st = load_state()
        st.setdefault("facts", []).insert(0, fact)
        st["facts"] = st["facts"][:60]
        save_state(st)
    await update.message.reply_text("Saved to long-term memory ✅")

async def cmd_resetmemory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with _state_lock:
        st = load_state()
        st["facts"], st["notes"] = [], []
        st["persona"] = DEFAULT_PERSONA
        save_state(st)
    await update.message.reply_text("Memory and persona reset ✅")

async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = " ".join(context.args).strip() if context.args else "technology"
    items = fetch_news(topic)
    if not items:
        await update.message.reply_text("News lookup needs `SERPAPI_KEY` or found nothing.")
        return
    lines = [f"- {it['title']} — {it['link']}" for it in items]
    await update.message.reply_text("📰 Latest:\n" + "\n".join(lines))
    with _state_lock:
        st = load_state()
        st["last_news"] = items
        save_state(st)

# ---------------- Natural free chat / routing ----------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    u = update.message.from_user
    username = u.username or "Unknown"

    # quick pings
    if text.lower() == "you there?":
        await update.message.reply_text(f"Always here 👊 (uptime {get_uptime()})")
        return

    # search
    if text.lower().startswith("search "):
        q = text[7:].strip()
        await update.message.reply_text(search_google(q))
        return

    # voice only on explicit trigger
    if text.lower().startswith("alex say "):
        phrase = text[9:].strip()
        if not phrase:
            await update.message.reply_text("What should I say?")
            return
        await update.message.reply_text(f"🎙️ Okay: {phrase}")
        await tts_send(update, phrase)
        return

    # normal, natural AI reply (with memory/persona)
    reply = gpt_reply(text)
    await update.message.reply_text(reply)
    log_conversation(username, u.id, text, reply)

# ---------------- Error handler ----------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.error("Exception while handling update:", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("Something glitched, but I’m back.")

# ---------------- Self-Learning Loop ----------------
def read_new_rows_since(idx_start: int) -> List[Dict[str, str]]:
    rows = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            rdr = list(csv.DictReader(f))
        for i, row in enumerate(rdr, start=1):
            if i >= idx_start:
                rows.append(row)
    except Exception as e:
        log.error(f"Read CSV error: {e}")
    return rows

def summarize_and_update_persona(notes_text: str, current_persona: str) -> Dict[str, str]:
    """
    Ask the model to: (1) summarize takeaways, (2) propose small persona refinements.
    Returns dict with 'notes' (list text) and 'persona' (new string or original).
    """
    try:
        prompt = (
            "You are maintaining a long-lived AI assistant called Alex.\n"
            "Given the recent conversation snippets below, first produce 3-6 concise bullet 'Notes' "
            "about user preferences, recurring topics, or helpful procedures (actionable, durable). "
            "Then propose up to two subtle improvements to Alex's persona (voice/tone/skills) "
            "that will make him more helpful for this user. Keep persona changes small and compatible.\n\n"
            f"Current persona:\n{current_persona}\n\n"
            f"Recent conversation snippets:\n{notes_text}\n"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()

        # very lightweight parsing: expect sections "Notes:" and "Persona update:"
        notes, new_persona = [], current_persona
        # split lines; treat leading dash as bullet
        for line in text.splitlines():
            s = line.strip(" •-").strip()
            if not s:
                continue
            if s.lower().startswith("persona"):
                # the next non-empty lines become persona suggestions
                continue
            if s.lower().startswith("notes"):
                continue
            # If the model returned a paragraph, accept as note.
            notes.append(s)

        # If the model also suggested an updated persona as a single paragraph, keep last paragraph
        if "\n\n" in text:
            chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
            if len(chunks) >= 2:
                new_persona = chunks[-1][:1200]

        return {"notes_texts": notes[:6], "persona": new_persona or current_persona}
    except Exception as e:
        log.error(f"Self-learning summary error: {e}")
        return {"notes_texts": [], "persona": current_persona}

def self_learning_worker(interval_minutes: int = 30):
    """
    Runs forever in a background thread:
    - reads new log rows
    - summarizes key takeaways
    - updates notes/persona
    - fetches and caches news
    """
    while True:
        try:
            with _state_lock:
                st = load_state()
                start_idx = int(st.get("last_seen_row", 1))
            rows = read_new_rows_since(start_idx)

            if rows:
                # Build small corpus
                snippets = []
                for r in rows[-40:]:
                    snippets.append(f"User: {r['query']}\nAlex: {r['reply']}")
                corpus = "\n\n".join(snippets)[-8000:]

                with _state_lock:
                    persona_before = st.get("persona", DEFAULT_PERSONA)
                upd = summarize_and_update_persona(corpus, persona_before)

                with _state_lock:
                    st = load_state()
                    st["notes"] = (upd["notes_texts"] + st.get("notes", []))[:80]
                    st["persona"] = upd["persona"][:1600]
                    st["last_seen_row"] = start_idx + len(rows)
                    st["last_update_iso"] = datetime.now(timezone.utc).isoformat()
                    save_state(st)
                log.info("🧠 Self-learning pass complete.")

            # Update news cache occasionally
            if SERPAPI_KEY:
                items = fetch_news("technology")
                if items:
                    with _state_lock:
                        st = load_state()
                        st["last_news"] = items
                        save_state(st)

        except Exception as e:
            log.error(f"Self-learning loop error: {e}")

        time.sleep(max(60, int(interval_minutes * 60)))

# ---------------- Bot runner ----------------
def run_bot():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("ping",   cmd_ping))
    app.add_handler(CommandHandler("ai",     cmd_ai))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("learn",  cmd_learn))
    app.add_handler(CommandHandler("resetmemory", cmd_resetmemory))
    app.add_handler(CommandHandler("news",   cmd_news))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    log.info("🚀 Alex is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

# ---------------- Main (with self-heal + background learning) ----------------
def main():
    # spawn background learner
    t = threading.Thread(target=self_learning_worker, kwargs={"interval_minutes": 30}, daemon=True)
    t.start()

    while True:
        try:
            run_bot()
        except Exception as e:
            log.error(f"💥 Bot crashed: {e}")
            log.info("♻️ Restarting in 5 seconds…")
            time.sleep(5)

if __name__ == "__main__":
    main()
