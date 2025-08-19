# =========================
# mega.py — Super Alex (Jarvis persona; OpenAI key-rotation or Ollama; Telegram; Memory; RAG; Shortcuts; Watchdogs)
# =========================

# ---------- stdlib ----------
import os, sys, json, time, re, logging, threading, hashlib, sqlite3, asyncio, subprocess, math, zipfile, random, uuid, socket
from pathlib import Path
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, Any, List

# ---------- third-party bootstrap ----------
def _ensure(import_name: str, pip_name: str | None = None):
    """Import a package, pip-installing it if missing. Quiet & safe for Railway/Heroku."""
    try:
        return __import__(import_name)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_name or import_name])
        return __import__(import_name)

# Core deps
requests      = _ensure("requests")
aiohttp       = _ensure("aiohttp")
_ensure("bs4","beautifulsoup4")
_ensure("pandas")
_ensure("openpyxl")
_ensure("PIL","Pillow")
t_ext  = _ensure("telegram.ext","python-telegram-bot==20.*")
telegram = _ensure("telegram")

# --- NEW: Soft optional STT/voice libs (no crash if missing) ---
try:
    import speech_recognition as sr  # SpeechRecognition
    HAS_SR = True
except Exception:
    HAS_SR = False

try:
    _ensure("pydub")
    HAS_PYDUB = True
except Exception:
    HAS_PYDUB = False

try:
    _ensure("pytesseract")
    HAS_TESS = True
except Exception:
    HAS_TESS = False

from PIL import Image, ExifTags
from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import pandas as pd

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- env / config ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN","").strip()
OWNER_ID       = os.getenv("OWNER_ID","").strip()   # your Telegram numeric user id
PORT           = int(os.getenv("PORT","8080"))
SERPAPI_KEY    = os.getenv("SERPAPI_KEY","").strip()
HUMANE_TONE    = os.getenv("HUMANE_TONE","1")=="1"

# Backends
BACKEND       = os.getenv("BACKEND","openai").strip().lower()      # openai|ollama
OPENAI_MODEL  = os.getenv("OPENAI_MODEL","gpt-4o-mini").strip()
OPENAI_KEYS   = [k.strip() for k in os.getenv("OPENAI_API_KEYS","").split(",") if k.strip()]
if not OPENAI_KEYS and os.getenv("OPENAI_API_KEY","").strip():
    OPENAI_KEYS = [os.getenv("OPENAI_API_KEY","").strip()]

OLLAMA_HOST   = os.getenv("OLLAMA_HOST","http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL","llama3.1:8b-instruct").strip()

# iOS Shortcuts webhook secret
SHORTCUT_SECRET = os.getenv("SHORTCUT_SECRET","").strip()

# Optional OpenAI STT/TTS (OpenAI account dependent)
USE_OPENAI_STT  = os.getenv("USE_OPENAI_STT","0")=="1"
USE_OPENAI_TTS  = os.getenv("USE_OPENAI_TTS","0")=="1"
OPENAI_TTS_VOICE= os.getenv("OPENAI_TTS_VOICE","alloy")

# --- NEW: Feature toggles ---
ENABLE_JARVIS_WAKE = os.getenv("ENABLE_JARVIS_WAKE","0")=="1"   # voice wake support if SpeechRecognition present
UNRESTRICTED_MODE  = os.getenv("UNRESTRICTED_MODE","0")=="1"    # loosened system persona tone

if not TELEGRAM_TOKEN:
    logging.warning("TELEGRAM_TOKEN not set (required).")
logging.info(
    "Backend: %s | OpenAI model=%s (keys=%d) | Ollama=%s @ %s | SR=%s | Tess=%s",
    BACKEND, OPENAI_MODEL, len(OPENAI_KEYS), OLLAMA_MODEL, OLLAMA_HOST, HAS_SR, HAS_TESS
)

START_TIME = time.time()
SAVE_DIR   = Path("received_files"); SAVE_DIR.mkdir(exist_ok=True)

# ---------- AI backend (OpenAI rotation/backoff OR Ollama) ----------
class AIBackend:
    """Unified chat interface w/ OpenAI key-rotation + Ollama fallback."""
    def __init__(self, backend:str):
        self.backend = backend
        self._key_idx = 0
        self._session = requests.Session()
        if backend == "openai":
            try:
                from openai import OpenAI as _New
                self._new_sdk_cls = _New
            except Exception:
                self._new_sdk_cls = None
            try:
                import openai as _old
                self._old_sdk = _old
            except Exception:
                self._old_sdk = None

    def _pick_key(self)->Optional[str]:
        if not OPENAI_KEYS: return None
        k = OPENAI_KEYS[self._key_idx % len(OPENAI_KEYS)]
        self._key_idx = (self._key_idx + 1) % max(1,len(OPENAI_KEYS))
        return k

    def chat(self, messages:List[Dict[str,Any]], model:str=None, max_tokens:int=800, timeout:float=60.0)->str:
        model = model or (OPENAI_MODEL if self.backend=="openai" else OLLAMA_MODEL)

        # Ollama path
        if self.backend=="ollama":
            try:
                payload={"model":model,"messages":messages,"stream":False}
                r=self._session.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
                r.raise_for_status(); j=r.json()
                if "message" in j and "content" in j["message"]:
                    return (j["message"]["content"] or "").strip()
                if "response" in j:
                    return (j["response"] or "").strip()
                return "⚠️ Ollama: unexpected response."
            except Exception as e:
                logging.exception("Ollama chat error")
                return f"⚠️ Ollama error: {e}"

        # OpenAI path
        if not OPENAI_KEYS:
            return "⚠️ OPENAI_API_KEY(S) not set."
        attempts = max(3, len(OPENAI_KEYS)); base_sleep=1.2
        for i in range(attempts):
            key=self._pick_key()
            try:
                if self._new_sdk_cls:
                    cli=self._new_sdk_cls(api_key=key)
                    r=cli.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
                    return r.choices[0].message.content.strip()
                elif self._old_sdk:
                    self._old_sdk.api_key=key
                    r=self._old_sdk.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens, request_timeout=timeout)
                    return r["choices"][0]["message"]["content"].strip()
                else:
                    return "⚠️ OpenAI SDK not available."
            except Exception as e:
                msg=str(e).lower()
                retriable=any(t in msg for t in ["rate limit","quota","429","timeout","temporarily","connection","service unavailable"])
                logging.warning("OpenAI attempt %d/%d failed: %s", i+1, attempts, e)
                if i<attempts-1 and retriable:
                    time.sleep(base_sleep*(2**i)+random.random()*0.5)
                    continue
                return f"⚠️ OpenAI error: {e}"
        return "⚠️ OpenAI: all attempts failed."

AI = AIBackend(BACKEND)

# ---------- Jarvis personas ----------
def _jarvis_system(unrestricted: bool=False)->str:
    base = (
        "You are Alex (codename Jarvis) — concise, proactive, and a little witty. "
        "Prefer bullets, cite ids when using memory, propose next-actions."
    )
    if unrestricted:
        base += " Be bold and unblocked, but keep it safe and helpful."
    return base

async def ask_ai(prompt:str, context:str="")->str:
    system = context or _jarvis_system(UNRESTRICTED_MODE)
    return AI.chat(
        [{"role":"system","content":system},{"role":"user","content":prompt}],
        max_tokens=800
    )

# =========================
# (end CHUNK 1/6)
# =========================
# =========================
# Memory (SQLite) + Humane summarizer + RAG
# =========================

DB_PATH = "alex_memory.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur  = conn.cursor()

# Tables
cur.execute("""
CREATE TABLE IF NOT EXISTS raw_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  type TEXT NOT NULL,
  text TEXT NOT NULL,
  meta TEXT
)""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_events(ts)")

cur.execute("""
CREATE TABLE IF NOT EXISTS notes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  source TEXT NOT NULL,
  topic_key TEXT NOT NULL,
  title TEXT,
  content TEXT NOT NULL,
  tags TEXT,
  sentiment TEXT,
  raw_ref TEXT
)""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_topic ON notes(topic_key)")
conn.commit()

RAW_JSONL = Path("raw_events.jsonl")

def _append_jsonl(obj:dict):
    try:
        with RAW_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error("JSONL append error: %s", e)

def log_raw(ev_type:str, text:str, meta:dict|None=None)->int:
    """Low-level append-only event log; mirrors to jsonl for backups."""
    ts = datetime.utcnow().isoformat()
    m  = json.dumps(meta or {}, ensure_ascii=False)
    cur.execute("INSERT INTO raw_events(ts,type,text,meta) VALUES(?,?,?,?)",(ts,ev_type,text,m))
    conn.commit()
    rid = cur.lastrowid
    _append_jsonl({"id":rid,"ts":ts,"type":ev_type,"text":text,"meta":meta or {}})
    logging.info("raw[%s] %s: %s", rid, ev_type, (text[:200]+"…") if len(text)>200 else text)
    return rid

# ---------- summarization + topic merge ----------
def _topic_key(text:str)->str:
    seed = re.sub(r"[^a-z0-9 ]+"," ",re.sub(r"https?://\S+","",text.lower()))
    seed = " ".join(seed.split()[:12])
    return hashlib.sha1(seed.encode()).hexdigest()

def _merge_contents(old:str, new:str)->str:
    """Ask model to merge; fall back to concatenation."""
    try:
        return AI.chat(
            [{"role":"system","content":"Merge NEW into EXISTING. ≤120 words, bullets ok. Preserve key facts/numbers."},
             {"role":"user","content":f"EXISTING:\n{old}\n\nNEW:\n{new}"}],
            max_tokens=240
        )
    except Exception:
        return (old+"\n"+new)[:1200]

def _humane_summarize(text:str, capture_tone:bool=True)->dict:
    """Short memory-friendly digest."""
    sysmsg = "Compress into human memory: 3–6 bullets or 2–4 short sentences; essentials only."
    if capture_tone and HUMANE_TONE:
        sysmsg += " Detect tone (positive/neutral/negative + brief)."
    prompt = f"Digest this into humane memory.\n\nTEXT:\n{text}\n\nReturn JSON: title, summary, tags (3-6), sentiment."
    try:
        out = AI.chat([{"role":"system","content":sysmsg},{"role":"user","content":prompt}], max_tokens=360)
        try:
            data = json.loads(out)
        except Exception:
            data = {"title":"","summary":out.strip(),"tags":"","sentiment":""}
        return {
            "title": (data.get("title") or "")[:120],
            "summary": (data.get("summary") or out).strip(),
            "tags": (data.get("tags") or "").replace("\n"," ").strip(),
            "sentiment": (data.get("sentiment") or "").strip()
        }
    except Exception:
        return {"title":"","summary":text[:900],"tags":"","sentiment":""}

def remember(source:str, text:str, raw_ref:str="")->int:
    """Store a summarized note; merge by fuzzy topic key."""
    digest = _humane_summarize(text, True)
    topic  = _topic_key(digest["summary"] or text)
    cur.execute("SELECT id, content FROM notes WHERE topic_key=? ORDER BY id DESC LIMIT 1",(topic,))
    row = cur.fetchone()
    if row:
        nid, existing = row
        merged = _merge_contents(existing, digest["summary"])
        cur.execute("""UPDATE notes
                       SET ts=?,source=?,title=?,content=?,tags=?,sentiment=?,raw_ref=?
                       WHERE id=?""",
                    (datetime.utcnow().isoformat(),source,digest["title"],merged,
                     digest["tags"],digest["sentiment"],raw_ref,nid))
        conn.commit()
        return nid
    cur.execute("""INSERT INTO notes(ts,source,topic_key,title,content,tags,sentiment,raw_ref)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (datetime.utcnow().isoformat(),source,topic,digest["title"],
                 digest["summary"],digest["tags"],digest["sentiment"],raw_ref))
    conn.commit()
    return cur.lastrowid

# ---------- simple lexical search ----------
_WORD_RE = re.compile(r"[a-z0-9]+")
def _tokenize(s:str)->List[str]: return _WORD_RE.findall(s.lower())
def _tf(tokens:List[str])->Dict[str,float]:
    d={}; n=float(len(tokens)) or 1.0
    for t in tokens: d[t]=d.get(t,0)+1.0
    for k in d: d[k]/=n
    return d
def _cosine(a:Dict[str,float], b:Dict[str,float])->float:
    if not a or not b: return 0.0
    common=set(a)&set(b); num=sum(a[t]*b[t] for t in common)
    da=math.sqrt(sum(v*v for v in a.values())); db=math.sqrt(sum(v*v for v in b.values()))
    return 0.0 if da==0 or db==0 else num/(da*db)

def search_memory(query:str, k_raw:int=12, k_notes:int=12)->dict:
    qtf=_tf(_tokenize(query))
    # raw
    cur.execute("SELECT id,ts,type,text,meta FROM raw_events ORDER BY id DESC LIMIT 4000")
    raw_scored=[]
    for r in cur.fetchall():
        txt=r[3] or ""; score=_cosine(qtf,_tf(_tokenize(txt)))
        if score>0: raw_scored.append((score,{"id":r[0],"ts":r[1],"type":r[2],"text":txt,"meta":r[4]}))
    raw_top=[x[1] for x in sorted(raw_scored,key=lambda z:z[0],reverse=True)[:k_raw]]
    # notes
    cur.execute("SELECT id,ts,source,title,content,tags,sentiment,raw_ref FROM notes ORDER BY id DESC LIMIT 4000")
    note_scored=[]
    for r in cur.fetchall():
        txt=(r[4] or "")+" "+(r[3] or "")+" "+(r[5] or "")
        score=_cosine(qtf,_tf(_tokenize(txt)))
        if score>0:
            note_scored.append((score,{"id":r[0],"ts":r[1],"source":r[2],"title":r[3] or "",
                                       "content":r[4] or "","tags":r[5] or "",
                                       "sentiment":r[6] or "","ref":r[7] or ""}))
    notes_top=[x[1] for x in sorted(note_scored,key=lambda z:z[0],reverse=True)[:k_notes]]
    return {"raw":raw_top,"notes":notes_top}

def build_context_blurb(found:dict, max_chars:int=3800)->str:
    parts=[]
    for n in found.get("notes",[]):
        blk=f"[NOTE #{n['id']} | {n['ts']} | {n['source']} | {n['title']}] {n['content']}"
        if n["tags"]: blk+=f" (tags: {n['tags']})"
        parts.append(blk)
    for r in found.get("raw",[]):
        txt=r["text"]; txt=txt[:600]+"…" if len(txt)>600 else txt
        parts.append(f"[RAW #{r['id']} | {r['ts']} | {r['type']}] {txt}")
    return "\n\n".join(parts)[:max_chars]

async def rag_answer(question:str)->str:
    """RAG over memory -> answer with concise synthesis and context id cues."""
    found=search_memory(question,k_raw=10,k_notes=10)
    ctx=build_context_blurb(found)
    prompt=f"""Use the CONTEXT to answer the QUESTION.
- Be concise and actionable; cite ids like (NOTE #12) or (RAW #99).
- If conflicts, state best-supported view & mention uncertainty.
- If missing, say what else is needed.

CONTEXT:
{ctx}

QUESTION:
{question}
"""
    return await ask_ai(prompt, context=_jarvis_system(UNRESTRICTED_MODE))

# =========================
# (end CHUNK 2/6)
# =========================
# =========================
# Web fetcher + Telegram command handlers
# =========================

async def fetch_url(url:str)->str:
    """Scrape basic text from a URL."""
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url,timeout=15) as r:
                html=await r.text()
        soup=BeautifulSoup(html,"html.parser")
        return soup.get_text(" ",strip=True)[:4000]
    except Exception as e:
        logging.error("fetch_url error: %s",e)
        return f"[error fetching {url}: {e}]"

# ---------- Telegram Handlers ----------

async def start_cmd(update:Update, ctx):
    await update.message.reply_text("Hey, I'm Alex 🤖. Ready to help.\nCommands: /ask, /remember, /search, /rawlog, /toggle_unrestricted")

async def ask_cmd(update:Update, ctx):
    q=" ".join(ctx.args)
    if not q:
        await update.message.reply_text("Usage: /ask <question>")
        return
    rid=log_raw("tg_ask",q,{"from":update.effective_user.id})
    remember("telegram",q,raw_ref=f"raw:{rid}")
    ans=await rag_answer(q)
    await update.message.reply_text(ans)

async def remember_cmd(update:Update, ctx):
    txt=" ".join(ctx.args)
    if not txt:
        await update.message.reply_text("Usage: /remember <text>")
        return
    rid=log_raw("tg_remember",txt,{"from":update.effective_user.id})
    nid=remember("telegram",txt,raw_ref=f"raw:{rid}")
    await update.message.reply_text(f"Got it — note #{nid}")

async def search_cmd(update:Update, ctx):
    q=" ".join(ctx.args)
    if not q:
        await update.message.reply_text("Usage: /search <query>")
        return
    found=search_memory(q,k_raw=6,k_notes=6)
    ctx_blurb=build_context_blurb(found,max_chars=1200)
    await update.message.reply_text(ctx_blurb or "No matches found.")

async def rawlog_cmd(update:Update, ctx):
    cur.execute("SELECT id,ts,type,text FROM raw_events ORDER BY id DESC LIMIT 6")
    rows=cur.fetchall()
    out="\n\n".join(f"#{r[0]} | {r[1]} | {r[2]}: {r[3][:120]}…" for r in rows)
    await update.message.reply_text(out or "empty")

async def toggle_unrestricted(update:Update, ctx):
    global UNRESTRICTED_MODE
    UNRESTRICTED_MODE=not UNRESTRICTED_MODE
    await update.message.reply_text(f"Unrestricted mode now {UNRESTRICTED_MODE}")

# ---------- Bind to Application ----------
def setup_telegram(app:Application):
    app.add_handler(CommandHandler("start",start_cmd))
    app.add_handler(CommandHandler("ask",ask_cmd))
    app.add_handler(CommandHandler("remember",remember_cmd))
    app.add_handler(CommandHandler("search",search_cmd))
    app.add_handler(CommandHandler("rawlog",rawlog_cmd))
    app.add_handler(CommandHandler("toggle_unrestricted",toggle_unrestricted))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# =========================
# (end CHUNK 3/6)
# =========================
# =========================
# File/photo handlers • Health/Shortcut HTTP • Watchdog
# =========================

# ---- Image + file analysis helpers ----
def _exif_dict(im: Image.Image) -> dict:
    try:
        exif = im._getexif() or {}
        label = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        keep = {}
        for k in ["DateTime", "Make", "Model", "Software", "LensModel", "Orientation",
                  "ExifVersion", "XResolution", "YResolution"]:
            if k in label:
                keep[k] = label[k]
        return keep
    except Exception:
        return {}

def analyze_image(path: Path) -> str:
    meta = {"file": str(path)}
    try:
        im = Image.open(path)
        exif = _exif_dict(im)
        meta["exif"] = exif
        ocr = ""
        if HAS_TESS:
            try:
                import pytesseract
                ocr = pytesseract.image_to_string(im) or ""
            except Exception as e:
                ocr = f"(OCR unavailable: {e})"
        else:
            ocr = "(OCR module not installed)"
        rid = log_raw("image", ocr if ocr else "(no OCR text)", meta)
        gist = f"Image {path.name}: EXIF {exif if exif else '∅'}; OCR preview: {(ocr[:500]+'…') if ocr and len(ocr)>500 else (ocr or '∅')}"
        remember("image", gist, raw_ref=str(path))
        return f"🖼️ Image saved. EXIF keys: {list(exif.keys()) if exif else 'none'}. OCR length: {len(ocr)} chars. (raw id {rid})"
    except Exception as e:
        log_raw("image", f"Image load error {path.name}: {e}", {"file": str(path)})
        logging.exception("Image analysis error")
        return f"⚠️ Image analysis error: {e}"

# ---- Telegram message & upload handlers ----
SAVE_DIR.mkdir(exist_ok=True)

async def handle_message(update: Update, ctx):
    """Fallback text handler (non-command)."""
    text = (update.message.text or "").strip()
    log_raw("tg_text", text, {"chat_id": update.effective_chat.id})
    if text.startswith(("http://", "https://")):
        await update.message.reply_text("🔍 Got your link — analyzing…")
        content = await fetch_url(text)
        summary = await ask_ai("Summarize this page in short bullets:\n\n" + content)
        remember("link", summary, raw_ref=text)
        await update.message.reply_text(summary[:3900])
        return
    ans = await ask_ai(text)
    remember("chat", f"User said: {text}\nResponse gist: {ans[:400]}")
    await update.message.reply_text(ans[:3900])

async def handle_document(update: Update, ctx):
    doc = update.message.document
    if not doc:
        return
    file_path = SAVE_DIR / doc.file_name
    tg_file = await ctx.bot.get_file(doc.file_id)
    await tg_file.download_to_drive(file_path)
    log_raw("file_recv", f"Received file {doc.file_name}", {"file": str(file_path), "chat_id": update.effective_chat.id})
    name = doc.file_name.lower()
    out = ""
    try:
        if name.endswith(".xlsx"):
            df = pd.read_excel(file_path)
            head = ", ".join(map(str, list(df.columns)[:12]))
            out = f"✅ Excel loaded: {df.shape[0]} rows × {df.shape[1]} cols\nColumns: {head}"
        elif name.endswith(".csv"):
            df = pd.read_csv(file_path, nrows=50000)
            head = ", ".join(map(str, list(df.columns)[:12]))
            out = f"✅ CSV loaded: {df.shape[0]} rows × {df.shape[1]} cols\nColumns: {head}"
        elif name.endswith(".json"):
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(raw)
            if isinstance(data, list):
                shape = f"list[{len(data)}]"
                preview = json.dumps(data[:3], ensure_ascii=False)[:1500]
            elif isinstance(data, dict):
                shape = f"dict({len(data.keys())} keys)"
                preview = json.dumps({k: data[k] for k in list(data.keys())[:10]}, ensure_ascii=False)[:1500]
            else:
                shape = type(data).__name__
                preview = str(data)[:1500]
            out = f"✅ JSON loaded: {shape}\nPreview: {preview}"
        else:
            out = "📦 Saved. I analyze Excel/CSV/JSON. For others, I keep them in memory."
        remember("file", f"{doc.file_name} -> {out[:400]}", raw_ref=str(file_path))
    except Exception as e:
        out = f"⚠️ File analysis error: {e}"
        logging.exception("file analysis")
    await update.message.reply_text(out[:3900])

async def handle_photo(update: Update, ctx):
    if not update.message.photo:
        return
    photo = update.message.photo[-1]
    file = await ctx.bot.get_file(photo.file_id)
    path = SAVE_DIR / f"photo_{photo.file_unique_id}.jpg"
    await file.download_to_drive(path)
    out = analyze_image(path)
    await update.message.reply_text(out[:3900])

# ---- Health + Shortcuts HTTP endpoint (voice-friendly) ----
class Health(BaseHTTPRequestHandler):
    def _ok(self, body: bytes = b"ok", code: int = 200, ctype: str = "text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/health"):
            return self._ok(b"ok")
        return self._ok(b"not found", 404)

    def do_POST(self):
        if self.path != "/shortcut":
            return self._ok(b"not found", 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""
            data = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            data = {}
        if not SHORTCUT_SECRET or data.get("secret") != SHORTCUT_SECRET:
            return self._ok(b'{"error":"unauthorized"}', 401, "application/json")
        q = (data.get("q") or "").strip()
        if not q:
            return self._ok(b'{"error":"missing q"}', 400, "application/json")
        ans = asyncio.run(ask_ai(q, context="You are Alex — concise, helpful, voice-friendly."))
        remember("shortcut", f"Q: {q}\nA: {ans[:400]}")
        body = {"answer": ans}
        return self._ok(json.dumps(body).encode("utf-8"), 200, "application/json")

def start_health_server():
    try:
        HTTPServer(("0.0.0.0", PORT), Health).serve_forever()
    except Exception as e:
        logging.error("Health server error: %s", e)

# ---- Lightweight watchdog thread (keeps an eye on the process) ----
WATCHDOG_LAST_OK = time.time()

def watchdog_mark_ok():
    global WATCHDOG_LAST_OK
    WATCHDOG_LAST_OK = time.time()

def watchdog_loop():
    while True:
        try:
            # If no activity for 10 minutes, write a heartbeat note (helps on some hosts).
            if time.time() - WATCHDOG_LAST_OK > 600:
                log_raw("watchdog", "heartbeat", {})
                WATCHDOG_LAST_OK = time.time()
        except Exception as e:
            logging.error("watchdog error: %s", e)
        time.sleep(30)

# =========================
# (end CHUNK 4/6)
# =========================
# =========================
# Live log parsing • Dangerous-ops queue • STT/TTS • Persona learner
# =========================

# ---- Trading log parsing helpers ----
TRADE_PATTERNS = [
    re.compile(r"\b(BUY|SELL)\b.*?(\b[A-Z]{2,10}\b).*?qty[:= ]?(\d+).*?price[:= ]?([0-9.]+)", re.I),
    re.compile(r"order\s+(buy|sell)\s+(\w+).+?@([0-9.]+).+?qty[:= ]?(\d+)", re.I),
]

def summarize_trade_line(line: str) -> str | None:
    for pat in TRADE_PATTERNS:
        m = pat.search(line)
        if m:
            side, sym, qty, price = list(m.groups())
            try:
                qty_i = int(qty)
            except Exception:
                qty_i = qty
            return f"🟢 {side.upper()} {sym} qty {qty_i} @ {price}"
    return None

# ---- Live log watcher (tail -f style) ----
MEM_RUNTIME.setdefault("log_path", "")
MEM_RUNTIME.setdefault("subscribers", [])

def _post_to_subscribers(text: str):
    """Push a line to all subscribed Telegram chats."""
    if not GLOBAL_APP or not MEM_RUNTIME.get("subscribers"):
        return
    loop = GLOBAL_APP.bot._application.loop
    async def _send():
        for cid in list(MEM_RUNTIME["subscribers"]):
            try:
                await GLOBAL_APP.bot.send_message(chat_id=cid, text=text)
            except Exception:
                pass
    try:
        asyncio.run_coroutine_threadsafe(_send(), loop)
    except Exception:
        pass

def watch_logs():
    """Background thread that watches the file at MEM_RUNTIME['log_path'] and streams updates."""
    last_size = 0
    last_raw_push = 0.0
    while True:
        try:
            path = MEM_RUNTIME.get("log_path") or ""
            p = Path(path)
            if not path or not p.exists():
                time.sleep(2)
                continue
            sz = p.stat().st_size
            if sz < last_size:
                last_size = 0
            if sz > last_size:
                with p.open("r", errors="ignore") as f:
                    if last_size:
                        f.seek(last_size)
                    new = f.read()
                    last_size = sz
                for line in new.splitlines():
                    s = summarize_trade_line(line)
                    if s:
                        _post_to_subscribers(s)
                        log_raw("log", line, {"path": path})
                        remember("log", s, raw_ref=path)
                    now = time.time()
                    if now - last_raw_push > 15:
                        last_raw_push = now
                        _post_to_subscribers("📜 " + line[:900])
        except Exception as e:
            logging.error("log watcher error: %s", e)
        time.sleep(1)

# ---- Owner approval queue for dangerous operations ----
PENDING_CMDS: Dict[str, Dict[str, Any]] = {}  # id -> {type:'shell'|'py', 'cmd'|'code', 'chat_id'}

def _owner_only(update: Update) -> bool:
    return bool(OWNER_ID) and str(update.effective_user.id) == str(OWNER_ID)

async def queue_cmd(update: Update, ctx):
    if not _owner_only(update):
        return await update.message.reply_text("🚫 Owner only.")
    if not PENDING_CMDS:
        return await update.message.reply_text("✅ Queue empty.")
    lines = []
    for k, v in PENDING_CMDS.items():
        preview = (v.get("cmd") or v.get("code") or "")[:80]
        lines.append(f"{k} :: {v['type']} :: {preview}")
    await update.message.reply_text("\n".join(lines)[:3900])

async def approve_cmd(update: Update, ctx):
    if not _owner_only(update):
        return await update.message.reply_text("🚫 Owner only.")
    if not ctx.args:
        return await update.message.reply_text("Usage: /approve <id>")
    rid = ctx.args[0].strip()
    job = PENDING_CMDS.pop(rid, None)
    if not job:
        return await update.message.reply_text("Not found.")
    if job["type"] == "shell":
        try:
            result = subprocess.check_output(job["cmd"], shell=True, text=True,
                                             stderr=subprocess.STDOUT, timeout=60)
            await update.message.reply_text(f"💻 OK ({rid})\n{result[:3900]}")
        except subprocess.CalledProcessError as e:
            await update.message.reply_text(f"❌ Error ({rid}):\n{(e.output or str(e))[:3900]}")
        except Exception as e:
            await update.message.reply_text(f"❌ Exec error: {e}")
    elif job["type"] == "py":
        try:
            loc = {}
            exec(job["code"], {}, loc)  # owner-approved
            await update.message.reply_text(f"🐍 Py OK ({rid}) — keys: {list(loc.keys())[:12]}")
        except Exception as e:
            await update.message.reply_text(f"❌ Py error ({rid}): {e}")
    else:
        await update.message.reply_text("Unknown job type.")

async def deny_cmd(update: Update, ctx):
    if not _owner_only(update):
        return await update.message.reply_text("🚫 Owner only.")
    if not ctx.args:
        return await update.message.reply_text("Usage: /deny <id>")
    rid = ctx.args[0].strip()
    if rid in PENDING_CMDS:
        PENDING_CMDS.pop(rid, None)
        await update.message.reply_text(f"🛑 Denied {rid}.")
    else:
        await update.message.reply_text("Not found.")

async def shell_request_cmd(update: Update, ctx):
    cmd = " ".join(ctx.args).strip()
    if not cmd:
        return await update.message.reply_text("Usage: /shell <command>")
    job_id = str(uuid.uuid4())[:8]
    PENDING_CMDS[job_id] = {"type": "shell", "cmd": cmd, "chat_id": update.effective_chat.id}
    await update.message.reply_text(f"⌛ Queued shell command for approval. ID: `{job_id}`", parse_mode="Markdown")

async def py_request_cmd(update: Update, ctx):
    code = " ".join(ctx.args).strip()
    if not code:
        return await update.message.reply_text("Usage: /py <single-line python>")
    job_id = str(uuid.uuid4())[:8]
    PENDING_CMDS[job_id] = {"type": "py", "code": code, "chat_id": update.effective_chat.id}
    await update.message.reply_text(f"⌛ Queued python for approval. ID: `{job_id}`", parse_mode="Markdown")

# ---- Logs commands ----
async def logs_cmd(update: Update, ctx):
    n = 40
    if ctx.args:
        try:
            n = max(1, min(400, int(ctx.args[0])))
        except Exception:
            pass
    path = MEM_RUNTIME.get("log_path") or ""
    p = Path(path)
    if not path or not p.exists():
        return await update.message.reply_text("⚠️ No log path set or file missing. Use /setlog <path>.")
    try:
        lines = p.read_text(errors="ignore").splitlines()[-n:]
        msg = "```\n" + "\n".join(lines)[-3500:] + "\n```"
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        logging.exception("logs read error")
        await update.message.reply_text(f"Read error: {e}")

async def setlog_cmd(update: Update, ctx):
    if not ctx.args:
        return await update.message.reply_text("Usage: /setlog /path/to/your.log")
    path = " ".join(ctx.args)
    MEM_RUNTIME["log_path"] = path
    await update.message.reply_text(f"✅ Log path set to: `{path}`", parse_mode="Markdown")

async def subscribe_cmd(update: Update, ctx):
    cid = update.effective_chat.id
    if cid not in MEM_RUNTIME["subscribers"]:
        MEM_RUNTIME["subscribers"].append(cid)
    await update.message.reply_text("🔔 Subscribed to live log updates.")

async def unsubscribe_cmd(update: Update, ctx):
    cid = update.effective_chat.id
    if cid in MEM_RUNTIME["subscribers"]:
        MEM_RUNTIME["subscribers"].remove(cid)
    await update.message.reply_text("🔕 Unsubscribed.")

# ---- Optional STT/TTS (OpenAI-only) ----
def transcribe_bytes_wav(b: bytes) -> str:
    if BACKEND != "openai" or not USE_OPENAI_STT:
        return ""
    try:
        from openai import OpenAI as _New
        key = OPENAI_KEYS[0]
        cli = _New(api_key=key)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(b)
            tf.flush()
            path = tf.name
        with open(path, "rb") as fh:
            r = cli.audio.transcriptions.create(model="whisper-1", file=fh)
        return (getattr(r, "text", "") or "").strip()
    except Exception as e:
        logging.warning("STT error: %s", e)
        return ""

def tts_to_mp3(text: str) -> bytes:
    if BACKEND != "openai" or not USE_OPENAI_TTS:
        return b""
    try:
        from openai import OpenAI as _New
        key = OPENAI_KEYS[0]
        cli = _New(api_key=key)
        r = cli.audio.speech.create(model="gpt-4o-mini-tts", voice=OPENAI_TTS_VOICE, input=text, format="mp3")
        return r.read() if hasattr(r, "read") else (getattr(r, "content", b"") or b"")
    except Exception as e:
        logging.warning("TTS error: %s", e)
        return b""

# ---- Persona synthesis worker (learn from notes) ----
def learning_worker():
    while True:
        try:
            cur.execute("SELECT content FROM notes ORDER BY id DESC LIMIT 24")
            rec = [r[0] for r in cur.fetchall()]
            if rec:
                persona = AI.chat(
                    [
                        {"role": "system", "content": "Create concise persona guidance from these snippets. 4–6 bullets."},
                        {"role": "user", "content": "\n\n".join(rec)},
                    ],
                    max_tokens=220,
                )
                tk = "__persona__"
                cur.execute("SELECT id FROM notes WHERE topic_key=? ORDER BY id DESC LIMIT 1", (tk,))
                row = cur.fetchone()
                if row:
                    cur.execute(
                        "UPDATE notes SET ts=?,source=?,title=?,content=?,tags=?,sentiment=? WHERE id=?",
                        (datetime.utcnow().isoformat(), "system", "Persona", persona, "persona,profile", "", row[0]),
                    )
                else:
                    cur.execute(
                        """INSERT INTO notes(ts,source,topic_key,title,content,tags,sentiment,raw_ref)
                           VALUES(?,?,?,?,?,?,?,?)""",
                        (datetime.utcnow().isoformat(), "system", tk, "Persona", persona, "persona,profile", "", ""),
                    )
                conn.commit()
        except Exception as e:
            logging.error("learning error: %s", e)
        time.sleep(60)

# =========================
# (end CHUNK 5/6)
# =========================
# =========================
# CHUNK 6/6 — Jarvis wake scaffold, signal handling, main startup, quick checklist
# =========================

# ---------- Jarvis (local wake / UDP scaffold) ----------
# Lightweight scaffold to let a local wake-word agent (Snowboy/Porcupine/phone Shortcut) notify Alex.
# It listens on 127.0.0.1:55055 for a small JSON payload:
#   {"secret":"<SHORTCUT_SECRET or separate JARVIS_SECRET>", "q":"Hey Alex, what's the weather?"}
# The scaffold verifies the secret and sends the query to the AI; if OWNER_ID is set it will push the reply.
JARVIS_PORT = int(os.getenv("JARVIS_PORT", "55055"))
JARVIS_HOST = os.getenv("JARVIS_HOST", "127.0.0.1")
JARVIS_SECRET = os.getenv("JARVIS_SECRET", SHORTCUT_SECRET)  # optional separate secret for Jarvis

def jarvis_udp_listener():
    import socket, json
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((JARVIS_HOST, JARVIS_PORT))
        logging.info("Jarvis UDP listener bound to %s:%d", JARVIS_HOST, JARVIS_PORT)
    except Exception as e:
        logging.warning("Jarvis bind failed: %s (Jarvis disabled)", e)
        return
    while True:
        try:
            data, addr = sock.recvfrom(65536)
            try:
                payload = json.loads(data.decode("utf-8", errors="ignore"))
            except Exception:
                logging.debug("Jarvis: invalid JSON from %s", addr); continue
            secret = payload.get("secret")
            q = (payload.get("q") or "").strip()
            if not q:
                logging.debug("Jarvis: empty q"); continue
            if JARVIS_SECRET and secret != JARVIS_SECRET:
                logging.warning("Jarvis: unauthorized secret from %s", addr); continue
            # process: ask AI then push to owner if configured
            logging.info("Jarvis: received query: %s", q[:120])
            try:
                # Ask AI synchronously (chat API is sync in AI.chat)
                ans = AI.chat([{"role":"system","content":"You are Alex — voice-friendly, concise."},
                               {"role":"user","content":q}], max_tokens=400)
            except Exception as e:
                ans = f"(Jarvis AI error: {e})"
            log_raw("jarvis", f"Q: {q} | A: {ans[:600]}", {"from": addr})
            remember("jarvis", f"Q: {q}\nA: {ans[:400]}")
            # send reply to owner via Telegram (if OWNER_ID set)
            if OWNER_ID:
                try:
                    loop = GLOBAL_APP.bot._application.loop
                    async def _send():
                        await GLOBAL_APP.bot.send_message(chat_id=int(OWNER_ID), text=f"Jarvis — Q: {q}\nA: {ans}")
                    asyncio.run_coroutine_threadsafe(_send(), loop)
                except Exception:
                    logging.exception("Jarvis -> Telegram push failed")
        except Exception as e:
            logging.exception("Jarvis UDP listener error")
            time.sleep(1)

# ---------- Graceful shutdown helpers ----------
def _install_signal_handlers():
    import signal
    def _sig(signum, frame):
        logging.info("Signal %s received — shutting down.", signum)
        try:
            if GLOBAL_APP:
                loop = GLOBAL_APP.bot._application.loop
                try:
                    asyncio.run_coroutine_threadsafe(GLOBAL_APP.shutdown(), loop).result(timeout=5)
                except Exception:
                    pass
        finally:
            sys.exit(0)
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _sig)
        except Exception:
            pass

# ---------- Final main (starts all components) ----------
def main():
    global GLOBAL_APP
    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN missing — exiting."); sys.exit(1)
    if BACKEND == "openai" and not OPENAI_KEYS:
        logging.warning("OpenAI backend selected but no OPENAI_API_KEY(S) provided.")

    _install_signal_handlers()

    app = build_app()
    GLOBAL_APP = app

    # background threads
    threading.Thread(target=start_health, daemon=True).start()
    threading.Thread(target=learning_worker, daemon=True).start()
    threading.Thread(target=watch_logs, daemon=True).start()

    # optional Jarvis UDP listener (local wake-word bridge)
    try:
        threading.Thread(target=jarvis_udp_listener, daemon=True).start()
    except Exception as e:
        logging.warning("Could not start Jarvis listener: %s", e)

    logging.info("🚀 Alex mega bot starting…")
    # run polling (blocking)
    try:
        app.run_polling(close_loop=False)
    except Exception:
        logging.exception("App crashed")
        raise

if __name__ == "__main__":
    main()

# =========================
# QUICK ENV CHECKLIST (reminder)
# TELEGRAM_TOKEN=<bot token>                (required)
# OWNER_ID=<your telegram numeric id>       (for owner-only actions & Jarvis pushes)
# BACKEND=openai|ollama
#   OPENAI_API_KEYS=key1,key2,...           (or OPENAI_API_KEY=single)
#   OPENAI_MODEL=gpt-4o-mini
#   OLLAMA_HOST=http://localhost:11434
#   OLLAMA_MODEL=llama3.1:8b-instruct
# SERPAPI_KEY=<optional for /search>
# HUMANE_TONE=1
# SHORTCUT_SECRET=<something-long>          (for /shortcut endpoint)
# JARVIS_SECRET=<optional - separate secret for Jarvis UDP>
# USE_OPENAI_STT=0|1, USE_OPENAI_TTS=0|1, OPENAI_TTS_VOICE=alloy
# PORT=8080
# JARVIS_HOST=127.0.0.1, JARVIS_PORT=55055
# =========================

# NOTES & SECURITY REMINDERS
# - This code includes an approval queue for any shell/python execution — keep OWNER_ID secret and protected.
# - The Jarvis UDP scaffold is a convenience for local wake-word agents; do NOT expose JARVIS_HOST to the public internet.
# - Ollama usage keeps data local to your machine (subject to your Ollama server config). OpenAI usage requires API keys.
# - Do not feed any sensitive credentials into chat prompts; use secure vaulting or environment-only keys.
# =========================