# ---------- mega.py ----------
import os, sys, json, time, threading, socket, logging, subprocess, asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
)

# ✅ Safe SpeechRecognition import (won’t crash if missing)
try:
    import speech_recognition as sr
except ImportError:
    sr = None
    print("⚠️ SpeechRecognition not installed. Jarvis voice disabled.")

# Optional: for TTS/STT + wake word Jarvis mode
import speech_recognition as sr
import pyttsx3

# =========================
# Environment setup
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
OWNER_ID = os.getenv("OWNER_ID", "")
BACKEND = os.getenv("BACKEND", "openai")  # openai | ollama

OPENAI_KEYS = os.getenv("OPENAI_API_KEYS", "").split(",") if os.getenv("OPENAI_API_KEYS") else []
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
HUMANE_TONE = bool(int(os.getenv("HUMANE_TONE", "1")))

SHORTCUT_SECRET = os.getenv("SHORTCUT_SECRET", "changeme")
USE_OPENAI_STT = bool(int(os.getenv("USE_OPENAI_STT", "0")))
USE_OPENAI_TTS = bool(int(os.getenv("USE_OPENAI_TTS", "0")))
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

PORT = int(os.getenv("PORT", "8080"))

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# =========================
# Globals
# =========================
GLOBAL_APP = None
MEMORY: List[Dict[str, Any]] = []
KNOWLEDGE_FILE = Path("alex_memory.json")

def save_memory():
    try:
        with open(KNOWLEDGE_FILE, "w") as f:
            json.dump(MEMORY, f, indent=2)
    except Exception as e:
        logging.error(f"Failed saving memory: {e}")

def load_memory():
    global MEMORY
    if KNOWLEDGE_FILE.exists():
        try:
            MEMORY = json.load(open(KNOWLEDGE_FILE))
            logging.info(f"Loaded {len(MEMORY)} memories.")
        except:
            MEMORY = []
    else:
        MEMORY = []
        # =========================
# Jarvis Voice Mode
# =========================
class Jarvis:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.listener = sr.Recognizer()
        self.mic = sr.Microphone()

    def speak(self, text: str):
        logging.info(f"🗣️ Jarvis says: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self) -> str:
        with self.mic as source:
            logging.info("🎤 Listening for wake word…")
            audio = self.listener.listen(source, phrase_time_limit=5)
        try:
            return self.listener.recognize_google(audio)
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            logging.error(f"Voice recognition error: {e}")
            return ""

    def run_loop(self):
        while True:
            heard = self.listen().lower()
            if "hey alex" in heard or "jarvis" in heard:
                self.speak("Yes, Blaize?")
                # Wait for command
                cmd = self.listen().lower()
                if cmd:
                    self.speak(f"Got it: {cmd}")
                    # Store in memory
                    MEMORY.append({"timestamp": str(datetime.now()), "cmd": cmd})
                    save_memory()

                    # Run as unrestricted command if safe
                    if cmd.startswith("run "):
                        output = run_shell(cmd[4:])
                        self.speak(output[:100])  # read first 100 chars back
            time.sleep(1)

# =========================
# Shell executor (unrestricted)
# =========================
def run_shell(command: str) -> str:
    """Runs arbitrary shell commands (⚠️ powerful)."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logging.info(f"Shell OK: {command}")
            return result.stdout.strip()
        else:
            logging.warning(f"Shell ERR: {command}")
            return result.stderr.strip()
    except Exception as e:
        logging.error(f"Shell fail: {e}")
        return str(e)
        # =========================
# AI backend (OpenAI rotation/backoff OR Ollama) + ask_ai
# =========================

class AIBackend:
    def __init__(self, backend: str):
        self.backend = backend
        self._key_idx = 0
        self._session = requests.Session()

        # Prefer the new OpenAI SDK if available; fall back to legacy
        self._new_sdk_cls = None
        self._old_sdk = None
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

    def _pick_key(self) -> Optional[str]:
        if not OPENAI_KEYS:
            return None
        key = OPENAI_KEYS[self._key_idx % len(OPENAI_KEYS)]
        self._key_idx = (self._key_idx + 1) % max(1, len(OPENAI_KEYS))
        return key

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 800,
        timeout: float = 60.0,
    ) -> str:
        model = model or (OPENAI_MODEL if self.backend == "openai" else OLLAMA_MODEL)

        # ---- Ollama path ----
        if self.backend == "ollama":
            try:
                payload = {"model": model, "messages": messages, "stream": False}
                r = self._session.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
                r.raise_for_status()
                j = r.json()
                if "message" in j and "content" in j["message"]:
                    return (j["message"]["content"] or "").strip()
                if "response" in j:
                    return (j["response"] or "").strip()
                return "⚠️ Ollama: unexpected response."
            except Exception as e:
                logging.exception("Ollama chat error")
                return f"⚠️ Ollama error: {e}"

        # ---- OpenAI path ----
        if not OPENAI_KEYS:
            return "⚠️ OPENAI_API_KEY(S) not set."

        attempts = max(3, len(OPENAI_KEYS))
        base_sleep = 1.2
        for i in range(attempts):
            key = self._pick_key()
            try:
                # New SDK
                if self._new_sdk_cls:
                    cli = self._new_sdk_cls(api_key=key)
                    r = cli.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
                    return (r.choices[0].message.content or "").strip()

                # Legacy SDK
                elif self._old_sdk:
                    self._old_sdk.api_key = key
                    r = self._old_sdk.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        request_timeout=timeout,
                    )
                    return (r["choices"][0]["message"]["content"] or "").strip()

                # No SDKs
                else:
                    return "⚠️ OpenAI SDK not available."

            except Exception as e:
                msg = str(e).lower()
                retriable = any(t in msg for t in ["rate limit", "quota", "429", "timeout", "temporarily", "connection", "service unavailable"])
                logging.warning("OpenAI attempt %d/%d failed: %s", i + 1, attempts, e)
                if i < attempts - 1 and retriable:
                    time.sleep(base_sleep * (2 ** i) + random.random() * 0.5)
                    continue
                return f"⚠️ OpenAI error: {e}"

        return "⚠️ OpenAI: all attempts failed."

# Single global backend instance (configured by env BACKEND)
AI = AIBackend(BACKEND)

async def ask_ai(prompt: str, context: str = "") -> str:
    """
    Thin async helper that calls the backend synchronously under the hood.
    Keeps persona/system context tight by default.
    """
    sysmsg = context or "You are Alex — concise, helpful, a little witty."
    return AI.chat(
        [{"role": "system", "content": sysmsg},
         {"role": "user", "content": prompt}],
        max_tokens=800,
    )
# =========================
# SQLite memory + RAG + URL/File/Image analyzers
# =========================

# ----- DB bootstrap -----
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
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

def _append_jsonl(obj: dict):
    try:
        with RAW_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error("JSONL append error: %s", e)

def log_raw(ev_type: str, text: str, meta: dict | None = None) -> int:
    ts = datetime.utcnow().isoformat()
    m = json.dumps(meta or {}, ensure_ascii=False)
    cur.execute("INSERT INTO raw_events(ts,type,text,meta) VALUES(?,?,?,?)", (ts, ev_type, text, m))
    conn.commit()
    rid = cur.lastrowid
    _append_jsonl({"id": rid, "ts": ts, "type": ev_type, "text": text, "meta": meta or {}})
    logging.info("raw[%s] %s: %s", rid, ev_type, (text[:200] + "…") if len(text) > 200 else text)
    return rid

# ----- Memory helpers (summarize + topic merge) -----
def _topic_key(text: str) -> str:
    seed = re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"https?://\S+", "", text.lower()))
    seed = " ".join(seed.split()[:12])
    return hashlib.sha1(seed.encode()).hexdigest()

def _merge_contents(old: str, new: str) -> str:
    try:
        return AI.chat(
            [
                {"role": "system", "content": "Merge NEW into EXISTING. ≤120 words, bullets ok. Preserve key facts/numbers."},
                {"role": "user", "content": f"EXISTING:\n{old}\n\nNEW:\n{new}"},
            ],
            max_tokens=240,
        )
    except Exception:
        return (old + "\n" + new)[:1200]

def _humane_summarize(text: str, capture_tone: bool = True) -> dict:
    sysmsg = "Compress into human memory: 3–6 bullets or 2–4 short sentences; essentials only."
    if capture_tone and HUMANE_TONE:
        sysmsg += " Detect tone (positive/neutral/negative + brief)."
    prompt = f"Digest this into humane memory.\n\nTEXT:\n{text}\n\nReturn JSON: title, summary, tags (3-6), sentiment."
    try:
        out = AI.chat([{"role": "system", "content": sysmsg}, {"role": "user", "content": prompt}], max_tokens=360)
        try:
            data = json.loads(out)
        except Exception:
            data = {"title": "", "summary": out.strip(), "tags": "", "sentiment": ""}
        return {
            "title": (data.get("title") or "")[:120],
            "summary": (data.get("summary") or out).strip(),
            "tags": (data.get("tags") or "").replace("\n", " ").strip(),
            "sentiment": (data.get("sentiment") or "").strip(),
        }
    except Exception:
        return {"title": "", "summary": text[:900], "tags": "", "sentiment": ""}

def remember(source: str, text: str, raw_ref: str = "") -> int:
    digest = _humane_summarize(text, True)
    topic = _topic_key(digest["summary"] or text)
    cur.execute("SELECT id, content FROM notes WHERE topic_key=? ORDER BY id DESC LIMIT 1", (topic,))
    row = cur.fetchone()
    if row:
        nid, existing = row
        merged = _merge_contents(existing, digest["summary"])
        cur.execute(
            """UPDATE notes
               SET ts=?, source=?, title=?, content=?, tags=?, sentiment=?, raw_ref=?
               WHERE id=?""",
            (datetime.utcnow().isoformat(), source, digest["title"], merged, digest["tags"], digest["sentiment"], raw_ref, nid),
        )
        conn.commit()
        return nid
    cur.execute(
        """INSERT INTO notes(ts,source,topic_key,title,content,tags,sentiment,raw_ref)
           VALUES(?,?,?,?,?,?,?,?)""",
        (datetime.utcnow().isoformat(), source, topic, digest["title"], digest["summary"], digest["tags"], digest["sentiment"], raw_ref),
    )
    conn.commit()
    return cur.lastrowid

# ----- Simple lexical search + RAG -----
_WORD_RE = re.compile(r"[a-z0-9]+")

def _tokenize(s: str) -> List[str]:
    return _WORD_RE.findall(s.lower())

def _tf(tokens: List[str]) -> Dict[str, float]:
    d: Dict[str, float] = {}
    n = float(len(tokens)) or 1.0
    for t in tokens:
        d[t] = d.get(t, 0.0) + 1.0
    for k in d:
        d[k] /= n
    return d

def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if da == 0 or db == 0 else num / (da * db)

def search_memory(query: str, k_raw: int = 12, k_notes: int = 12) -> dict:
    qtf = _tf(_tokenize(query))

    cur.execute("SELECT id,ts,type,text,meta FROM raw_events ORDER BY id DESC LIMIT 4000")
    raw_scored = []
    for r in cur.fetchall():
        txt = r[3] or ""
        score = _cosine(qtf, _tf(_tokenize(txt)))
        if score > 0:
            raw_scored.append(
                (score, {"id": r[0], "ts": r[1], "type": r[2], "text": txt, "meta": r[4]})
            )
    raw_top = [x[1] for x in sorted(raw_scored, key=lambda z: z[0], reverse=True)[:k_raw]]

    cur.execute("SELECT id,ts,source,title,content,tags,sentiment,raw_ref FROM notes ORDER BY id DESC LIMIT 4000")
    note_scored = []
    for r in cur.fetchall():
        txt = (r[4] or "") + " " + (r[3] or "") + " " + (r[5] or "")
        score = _cosine(qtf, _tf(_tokenize(txt)))
        if score > 0:
            note_scored.append(
                (
                    score,
                    {
                        "id": r[0],
                        "ts": r[1],
                        "source": r[2],
                        "title": r[3] or "",
                        "content": r[4] or "",
                        "tags": r[5] or "",
                        "sentiment": r[6] or "",
                        "ref": r[7] or "",
                    },
                )
            )
    notes_top = [x[1] for x in sorted(note_scored, key=lambda z: z[0], reverse=True)[:k_notes]]
    return {"raw": raw_top, "notes": notes_top}

def build_context_blurb(found: dict, max_chars: int = 3800) -> str:
    parts: List[str] = []
    for n in found.get("notes", []):
        blk = f"[NOTE #{n['id']} | {n['ts']} | {n['source']} | {n['title']}] {n['content']}"
        if n["tags"]:
            blk += f" (tags: {n['tags']})"
        parts.append(blk)
    for r in found.get("raw", []):
        txt = r["text"]
        txt = txt[:600] + "…" if len(txt) > 600 else txt
        parts.append(f"[RAW #{r['id']} | {r['ts']} | {r['type']}] {txt}")
    return "\n\n".join(parts)[:max_chars]

async def rag_answer(question: str) -> str:
    found = search_memory(question, k_raw=10, k_notes=10)
    ctx = build_context_blurb(found)
    prompt = f"""Use the CONTEXT to answer the QUESTION.
- Be concise and actionable; cite ids like (NOTE #12) or (RAW #99).
- If conflicts, state best-supported view & mention uncertainty.
- If missing, say what else is needed.

CONTEXT:
{ctx}

QUESTION:
{question}
"""
    return await ask_ai(prompt, context="You are Alex — precise, synthesizes across memory, cites context ids.")

# ----- URL crawler + SEO summarizer -----
async def fetch_url(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"}) as r:
                if r.status != 200:
                    return f"⚠️ HTTP {r.status}"
                text = await r.text()
        soup = BeautifulSoup(text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
        desc = (soup.find("meta", {"name": "description"}) or {}).get("content", "")
        h1 = soup.h1.get_text(strip=True) if soup.h1 else ""
        words = len(soup.get_text(" ").split())
        links = len(soup.find_all("a"))
        images = len(soup.find_all("img"))
        snippet = soup.get_text(" ")[:1800]
        return f"🌐 {title}\nDesc: {desc[:200]}\nH1: {h1}\nWords:{words} Links:{links} Images:{images}\n\nSnippet:\n{snippet}"
    except Exception as e:
        logging.exception("Crawl error")
        return f"⚠️ Crawl error: {e}"

async def analyze_url(url: str) -> str:
    content = await fetch_url(url)
    if content.startswith("⚠️"):
        return content
    summary = await ask_ai("Summarize page in short bullets; key entities, actions, SEO opportunities:\n\n" + content)
    log_raw("link", summary, {"url": url})
    remember("link", summary, raw_ref=url)
    return summary

# ----- File analyzers -----
def analyze_excel(path: Path) -> str:
    try:
        df = pd.read_excel(path)
        head = ", ".join(map(str, list(df.columns)[:12]))
        info = f"✅ Excel loaded: {df.shape[0]} rows × {df.shape[1]} cols\nColumns: {head}"
        num = df.select_dtypes(include="number")
        if not num.empty:
            info += "\n\nNumeric summary:\n" + num.describe().to_string()[:1800]
        log_raw("file", f"Excel {path.name}: shape {df.shape}; columns {list(df.columns)!r}", {"file": str(path)})
        remember("file", f"Excel {path.name}: shape {df.shape}; columns {list(df.columns)!r}", raw_ref=str(path))
        return info
    except Exception as e:
        logging.exception("Excel analysis error")
        return f"⚠️ Excel analysis error: {e}"

def analyze_csv(path: Path) -> str:
    try:
        df = pd.read_csv(path, nrows=50000)
        head = ", ".join(map(str, list(df.columns)[:12]))
        info = f"✅ CSV loaded: {df.shape[0]} rows × {df.shape[1]} cols\nColumns: {head}"
        num = df.select_dtypes(include="number")
        if not num.empty:
            info += "\n\nNumeric summary:\n" + num.describe().to_string()[:1800]
        log_raw("file", f"CSV {path.name}: shape {df.shape}; columns {list(df.columns)!r}", {"file": str(path)})
        remember("file", f"CSV {path.name}: shape {df.shape}; columns {list(df.columns)!r}", raw_ref=str(path))
        return info
    except Exception as e:
        logging.exception("CSV analysis error")
        return f"⚠️ CSV analysis error: {e}"

def analyze_json(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
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
        info = f"✅ JSON loaded: {shape}\nPreview: {preview}"
        log_raw("file", f"JSON {path.name}: shape {shape}", {"file": str(path)})
        remember("file", f"JSON {path.name}: shape {shape}\nPreview: {preview}", raw_ref=str(path))
        return info
    except Exception as e:
        logging.exception("JSON analysis error")
        return f"⚠️ JSON analysis error: {e}"

# ----- Image analyzer (EXIF + optional OCR) -----
def _exif_dict(im: Image.Image) -> dict:
    try:
        exif = im._getexif() or {}
        label = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        keep = {}
        for k in ["DateTime", "Make", "Model", "Software", "LensModel", "Orientation", "ExifVersion", "XResolution", "YResolution"]:
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
        if HAS_TESS:
            try:
                import pytesseract
                ocr = pytesseract.image_to_string(im) or ""
            except Exception as e:
                ocr = f"(OCR unavailable: {e})"
        else:
            ocr = "(OCR module not installed)"
        rid = log_raw("image", ocr if ocr else "(no OCR text)", meta)
        gist = f"Image {path.name}: EXIF {exif if exif else '∅'}; OCR preview: {(ocr[:500] + '…') if ocr and len(ocr) > 500 else (ocr or '∅')}"
        remember("image", gist, raw_ref=str(path))
        return f"🖼️ Image saved. EXIF keys: {list(exif.keys()) if exif else 'none'}. OCR length: {len(ocr)} chars. (raw id {rid})"
    except Exception as e:
        log_raw("image", f"Image load error {path.name}: {e}", {"file": str(path)})
        logging.exception("Image analysis error")
        return f"⚠️ Image analysis error: {e}"
# ---------- mega.py (Part 5/6) ----------
from fastapi import FastAPI, Request
import uvicorn

# -------- SHORTCUT ENDPOINT --------
WEB_APP = FastAPI()

@WEB_APP.post("/shortcut")
async def shortcut(req: Request):
    """Secure endpoint to trigger Alex with external calls (e.g. Apple Watch)."""
    data = await req.json()
    secret = data.get("secret", "")
    if secret != SHORTCUT_SECRET:
        return {"error": "unauthorized"}
    msg = data.get("message", "")
    # forward to Telegram owner
    if GLOBAL_APP:
        try:
            await GLOBAL_APP.bot.send_message(chat_id=OWNER_ID, text=f"⌚ Shortcut: {msg}")
        except Exception as e:
            logging.error(f"Shortcut delivery failed: {e}")
    return {"ok": True}

# -------- JARVIS-STYLE HOOK --------
async def jarvis_listener():
    """
    Placeholder for voice trigger (e.g. 'Hey Alex').
    Future expansion: always-on mic (Lenovo), Apple Watch Shortcut,
    or phone integration.
    """
    logging.info("🎙️ Jarvis listener active (placeholder)")

threading.Thread(target=lambda: asyncio.run(jarvis_listener()), daemon=True).start()

# -------- NOTES FEATURE --------
NOTES_FILE = Path("alex_notes.txt")

async def save(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = " ".join(context.args)
    if not text:
        await update.message.reply_text("⚡ Usage: /save <note>")
        return
    with open(NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - {text}\n")
    await update.message.reply_text("✅ Note saved.")

async def notes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if NOTES_FILE.exists():
        content = NOTES_FILE.read_text(encoding="utf-8")[-1500:]
        await update.message.reply_text(f"📝 Notes:\n{content}")
    else:
        await update.message.reply_text("No notes yet.")
# ---------- mega.py (Part 6/6: wiring + main) ----------
# This chunk finalizes handlers, background workers, and launches the bot + web server.

# --- Optional: separate port for FastAPI to avoid clashing with /health HTTPServer ---
UVICORN_HOST = os.getenv("UVICORN_HOST", "0.0.0.0")
UVICORN_PORT = int(os.getenv("UVICORN_PORT", "9090"))

def _start_uvicorn():
    try:
        import uvicorn
        logging.info(f"🌐 Starting FastAPI on {UVICORN_HOST}:{UVICORN_PORT}")
        uvicorn.run(WEB_APP, host=UVICORN_HOST, port=UVICORN_PORT, log_level="warning")
    except Exception as e:
        logging.error(f"FastAPI/uvicorn error: {e}")

# -------- Additional commands (Jarvis toggle/status) --------
JARVIS_ENABLED = True

async def jarvis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global JARVIS_ENABLED
    if context.args:
        arg = context.args[0].lower().strip()
        if arg in ("on", "enable", "enabled"):
            JARVIS_ENABLED = True
        elif arg in ("off", "disable", "disabled"):
            JARVIS_ENABLED = False
    await update.message.reply_text(f"🎙️ Jarvis is {'ON' if JARVIS_ENABLED else 'OFF'}")

# -------- Build Telegram application & handlers --------
def build_app() -> Application:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Core commands
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("id",    id_cmd))
    app.add_handler(CommandHandler("uptime", uptime_cmd))
    app.add_handler(CommandHandler("config", config_cmd))
    app.add_handler(CommandHandler("backup", backup_cmd))

    # AI & RAG
    app.add_handler(CommandHandler("ai",  ai_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))

    # Web/SEO + Search
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("search",  search_cmd))

    # Memory
    app.add_handler(CommandHandler("remember", remember_cmd))
    app.add_handler(CommandHandler("mem",      mem_cmd))
    app.add_handler(CommandHandler("exportmem", exportmem_cmd))
    app.add_handler(CommandHandler("raw",       raw_cmd))

    # Notes
    app.add_handler(CommandHandler("save",  save))
    app.add_handler(CommandHandler("notes", notes))

    # Logs & live watchers
    app.add_handler(CommandHandler("setlog", setlog_cmd))
    app.add_handler(CommandHandler("subscribe_logs", subscribe_cmd))
    app.add_handler(CommandHandler("unsubscribe_logs", unsubscribe_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))

    # Owner approval / dangerous ops
    app.add_handler(CommandHandler("queue",   queue_cmd))
    app.add_handler(CommandHandler("approve", approve_cmd))
    app.add_handler(CommandHandler("deny",    deny_cmd))
    app.add_handler(CommandHandler("shell",   shell_request_cmd))
    app.add_handler(CommandHandler("py",      py_request_cmd))

    # Jarvis control
    app.add_handler(CommandHandler("jarvis", jarvis))

    # File & photo ingestion
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.PHOTO,        handle_photo))

    # Fallback chat
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    return app

# -------- Background threads: health, learning, logs, FastAPI --------
def _start_background_threads(app: Application):
    # Expose to global for push-backs from watchers / shortcut endpoint
    global GLOBAL_APP
    GLOBAL_APP = app

    # /health HTTP server (from earlier chunk)
    threading.Thread(target=start_health, daemon=True).start()

    # Persona synthesis worker
    threading.Thread(target=learning_worker, daemon=True).start()

    # Live log watcher
    threading.Thread(target=watch_logs, daemon=True).start()

    # FastAPI (Shortcut + future webhooks)
    threading.Thread(target=_start_uvicorn, daemon=True).start()

# -------- Main entry --------
def main():
    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN missing — exiting.")
        sys.exit(1)

    if BACKEND == "openai" and not OPENAI_KEYS:
        logging.warning("OpenAI backend selected but no OPENAI_API_KEY(S) provided.")

    app = build_app()
    _start_background_threads(app)

    logging.info("🚀 Alex mega bot starting… (Telegram polling)")
    # close_loop=False keeps the existing loop for background tasks
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
# ---------- Jarvis Voice ----------
def listen_to_voice():
    if sr is None:
        return "Voice unavailable"
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except Exception as e:
        return f"Error: {e}"