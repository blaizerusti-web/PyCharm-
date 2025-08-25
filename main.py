# =========================
# mega.py — Alex (Hybrid auto: Ollama via tunnel + OpenAI fallback; Telegram; Memory; RAG; Jarvis; Shortcuts webhook)
# =========================

import os, sys, json, time, re, logging, threading, hashlib, sqlite3, asyncio, subprocess, math, zipfile, random, uuid
from pathlib import Path
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, Any, List, Tuple

# ---------- bootstrap: install/import ----------
def _ensure(import_name: str, pip_name: str | None = None):
    try:
        return __import__(import_name)
    except ModuleNotFoundError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_name or import_name])
        return __import__(import_name)

requests=_ensure("requests")
aiohttp=_ensure("aiohttp")
_ensure("bs4","beautifulsoup4")
_ensure("pandas")
_ensure("openpyxl")
_ensure("PIL","Pillow")
t_ext=_ensure("telegram.ext","python-telegram-bot==20.*")
telegram=_ensure("telegram")

try:
    _ensure("pytesseract")
    HAS_TESS=True
except Exception:
    HAS_TESS=False

from PIL import Image, ExifTags
from bs4 import BeautifulSoup
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import pandas as pd

# ---------- config / logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Telegram & owner
TELEGRAM_TOKEN=os.getenv("TELEGRAM_TOKEN","").strip()
OWNER_ID=os.getenv("OWNER_ID","").strip()   # your Telegram numeric user id

# Server / ports (robust: accept PORT or Heroku-style lowercase 'port')
PORT=int(os.getenv("PORT") or os.getenv("port") or 8080)

# Third-party APIs
SERPAPI_KEY=os.getenv("SERPAPI_KEY","").strip()

# Personality toggles
HUMANE_TONE=os.getenv("HUMANE_TONE","1")=="1"
JARVIS_MODE_DEFAULT=os.getenv("JARVIS_MODE","0")=="1"     # can be toggled at runtime
DEV_MODE_DEFAULT=os.getenv("DEV_MODE","0")=="1"           # looser tone, still safe; owner can toggle

# Backend selection
#   auto   => try Ollama (tunnel / remote) first, fallback to OpenAI
#   openai => force OpenAI
#   ollama => force Ollama (if unavailable, graceful error)
BACKEND_MODE_DEFAULT=os.getenv("BACKEND","auto").strip().lower()
OPENAI_MODEL=os.getenv("OPENAI_MODEL","gpt-4o-mini").strip()

# OpenAI keys (rotation)
OPENAI_KEYS=[k.strip() for k in os.getenv("OPENAI_API_KEYS","").split(",") if k.strip()]
if not OPENAI_KEYS and os.getenv("OPENAI_API_KEY","").strip():
    OPENAI_KEYS=[os.getenv("OPENAI_API_KEY","").strip()]

# Ollama hybrid (Option 2 foundations for 24/7)
# Multiple candidate endpoints; first healthy is used.
# Examples:
#   OLLAMA_URLS="https://abc-123.ngrok.app,http://100.100.100.10:11434,http://localhost:11434"
OLLAMA_URLS=[u.strip().rstrip("/") for u in os.getenv("OLLAMA_URLS","").split(",") if u.strip()]
# Back-compat single host:
if not OLLAMA_URLS:
    host=os.getenv("OLLAMA_HOST","http://localhost:11434").strip().rstrip("/")
    if host: OLLAMA_URLS=[host]
OLLAMA_MODEL=os.getenv("OLLAMA_MODEL","llama3.1:8b-instruct").strip()
# iOS Shortcuts webhook
SHORTCUT_SECRET=os.getenv("SHORTCUT_SECRET","").strip()

# Optional STT/TTS flags (OpenAI only)
USE_OPENAI_STT=os.getenv("USE_OPENAI_STT","0")=="1"
USE_OPENAI_TTS=os.getenv("USE_OPENAI_TTS","0")=="1"
OPENAI_TTS_VOICE=os.getenv("OPENAI_TTS_VOICE","alloy")

logging.info("Backend mode: %s | OpenAI model=%s (keys=%d)", BACKEND_MODE_DEFAULT, OPENAI_MODEL, len(OPENAI_KEYS))
logging.info("Ollama candidates: %s | model=%s", OLLAMA_URLS, OLLAMA_MODEL)
logging.info("OCR pytesseract: %s", "available" if HAS_TESS else "not available")

START_TIME=time.time()
SAVE_DIR=Path("received_files"); SAVE_DIR.mkdir(exist_ok=True)

# ---------- AI backend (Hybrid: Ollama candidate list + OpenAI rotation/backoff) ----------
class AIBackend:
    def __init__(self, backend_mode:str, ollama_urls:List[str], ollama_model:str, openai_model:str):
        self.backend_mode=backend_mode
        self._ollama_urls=list(dict.fromkeys([u for u in ollama_urls if u]))  # de-dup, preserve order
        self.ollama_model=ollama_model
        self.openai_model=openai_model
        self._key_idx=0
        self._session = requests.Session()
        self._ollama_health: Dict[str, Tuple[bool, float]] = {}  # url -> (healthy, ts_epoch)

        # OpenAI SDKs
        self._new_sdk_cls=None
        self._old_sdk=None
        if OPENAI_KEYS:
            try:
                from openai import OpenAI as _New
                self._new_sdk_cls=_New
            except Exception:
                self._new_sdk_cls=None
            try:
                import openai as _old
                self._old_sdk=_old
            except Exception:
                self._old_sdk=None

    # ----- dynamic controls -----
    def set_backend_mode(self, mode:str):
        self.backend_mode=mode.strip().lower()

    def add_ollama_url(self, url:str):
        url=url.strip().rstrip("/")
        if url and url not in self._ollama_urls:
            self._ollama_urls.insert(0, url)  # prefer newly added
            logging.info("Added Ollama URL: %s", url)

    def list_ollama_urls(self)->List[str]:
        return list(self._ollama_urls)

    def remove_ollama_url(self, url:str):
        url=url.strip().rstrip("/")
        self._ollama_urls=[u for u in self._ollama_urls if u!=url]

    # ----- openai key rotation -----
    def _pick_key(self)->Optional[str]:
        if not OPENAI_KEYS: return None
        k=OPENAI_KEYS[self._key_idx % len(OPENAI_KEYS)]
        self._key_idx=(self._key_idx+1) % max(1,len(OPENAI_KEYS))
        return k

    # ----- ollama health & selection -----
    def _check_ollama(self, url:str, ttl:float=25.0)->bool:
        now=time.time()
        healthy, ts = self._ollama_health.get(url,(False,0.0))
        if now - ts < 15:   # cache for 15s
            return healthy
        try:
            r=self._session.get(f"{url}/api/version", timeout=ttl)
            ok = (r.status_code==200)
        except Exception:
            ok=False
        self._ollama_health[url]=(ok, now)
        return ok

    def _select_ollama(self)->Optional[str]:
        for u in self._ollama_urls:
            if self._check_ollama(u):
                return u
        return None
            # ----- chat paths -----
    def _ollama_chat(self, messages:List[Dict[str,Any]], model:str|None, timeout:float=60.0)->str:
        url=self._select_ollama()
        if not url:
            raise RuntimeError("No healthy Ollama endpoints available.")
        payload={"model": model or self.ollama_model, "messages":messages, "stream":False}
        r=self._session.post(f"{url}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        j=r.json()
        if isinstance(j, dict):
            if "message" in j and isinstance(j["message"], dict):
                return (j["message"].get("content","") or "").strip()
            if "response" in j:
                return (j["response"] or "").strip()
        return "⚠️ Ollama: unexpected response."

    def _openai_chat(self, messages:List[Dict[str,Any]], model:str|None, max_tokens:int=800, timeout:float=60.0)->str:
        if not OPENAI_KEYS: return "⚠️ OPENAI_API_KEY(S) not set."
        attempts=max(3, len(OPENAI_KEYS)); base_sleep=1.1
        last_error=None
        for i in range(attempts):
            key=self._pick_key()
            try:
                if self._new_sdk_cls:
                    cli=self._new_sdk_cls(api_key=key)
                    r=cli.chat.completions.create(model=model or self.openai_model, messages=messages, max_tokens=max_tokens)
                    return r.choices[0].message.content.strip()
                elif self._old_sdk:
                    self._old_sdk.api_key=key
                    r=self._old_sdk.ChatCompletion.create(model=model or self.openai_model, messages=messages, max_tokens=max_tokens, request_timeout=timeout)
                    return r["choices"][0]["message"]["content"].strip()
                else:
                    return "⚠️ OpenAI SDK not available."
            except Exception as e:
                last_error=e
                msg=str(e).lower()
                retriable=any(t in msg for t in ["rate limit","quota","429","timeout","temporarily","connection","service unavailable"])
                logging.warning("OpenAI attempt %d/%d failed: %s", i+1, attempts, e)
                if i<attempts-1 and retriable:
                    time.sleep(base_sleep*(2**i)+random.random()*0.4)
                    continue
                break
        return f"⚠️ OpenAI error: {last_error}"

    def chat(self, messages:List[Dict[str,Any]], model:str|None=None, max_tokens:int=800, timeout:float=60.0)->str:
        mode=self.backend_mode
        if mode=="ollama":
            try:
                return self._ollama_chat(messages, model, timeout=timeout)
            except Exception as e:
                logging.error("Ollama error: %s", e)
                return f"⚠️ Ollama error: {e}"
        elif mode=="openai":
            return self._openai_chat(messages, model, max_tokens=max_tokens, timeout=timeout)
        else:  # auto (hybrid)
            try:
                return self._ollama_chat(messages, model, timeout=timeout)
            except Exception as e:
                logging.info("Ollama unavailable, falling back to OpenAI: %s", e)
                return self._openai_chat(messages, model, max_tokens=max_tokens, timeout=timeout)

# Global backend instance (runtime-mutable)
AI=AIBackend(BACKEND_MODE_DEFAULT, OLLAMA_URLS, OLLAMA_MODEL, OPENAI_MODEL)

# ---------- Persona / runtime state ----------
GLOBAL_APP:Application|None=None
STATE={
    "jarvis_mode": JARVIS_MODE_DEFAULT,
    "dev_mode": DEV_MODE_DEFAULT,
    "backend_mode": BACKEND_MODE_DEFAULT,            # mirrors AI.backend_mode
    "force_ollama_url": "",                          # optional: pin to one URL
}
MEM_RUNTIME={"log_path":"", "subscribers": []}

def _owner_only(update:Update)->bool:
    return OWNER_ID and str(update.effective_user.id)==str(OWNER_ID)

def persona_prompt()->str:
    base = "You are Alex — concise, helpful, witty, and practical."
    if STATE["jarvis_mode"]:
        base = ("You are Alex (aka Jarvis) — proactive, succinct, voice-friendly. "
                "Offer smart defaults, anticipate needs, and keep answers tight.")
    if STATE["dev_mode"]:
        base += " Be direct and unblocked, but stay safe and lawful."
    if HUMANE_TONE:
        base += " Prefer short sentences and humane summaries."
    return base
    async def ask_ai(prompt:str, context:str="")->str:
    sysmsg=context or persona_prompt()
    try:
        return AI.chat(
            [{"role":"system","content":sysmsg},
             {"role":"user","content":prompt}],
            max_tokens=800
        )
    except Exception as e:
        logging.exception("ask_ai error")
        return f"⚠️ AI error: {e}"

# ---------- SQLite memory ----------
DB_PATH="alex_memory.db"
conn=sqlite3.connect(DB_PATH, check_same_thread=False)
cur=conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS raw_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, type TEXT NOT NULL, text TEXT NOT NULL, meta TEXT)""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_events(ts)")
cur.execute("""CREATE TABLE IF NOT EXISTS notes(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, source TEXT NOT NULL, topic_key TEXT NOT NULL,
  title TEXT, content TEXT NOT NULL, tags TEXT, sentiment TEXT, raw_ref TEXT)""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_topic ON notes(topic_key)")
conn.commit()

RAW_JSONL=Path("raw_events.jsonl")

def _append_jsonl(obj:dict):
    try:
        with RAW_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False)+"\n")
    except Exception as e:
        logging.error("JSONL append error: %s", e)

def log_raw(ev_type:str, text:str, meta:dict|None=None)->int:
    ts=datetime.utcnow().isoformat()
    m=json.dumps(meta or {}, ensure_ascii=False)
    cur.execute("INSERT INTO raw_events(ts,type,text,meta) VALUES(?,?,?,?)",(ts,ev_type,text,m))
    conn.commit()
    rid=cur.lastrowid
    _append_jsonl({"id":rid,"ts":ts,"type":ev_type,"text":text,"meta":meta or {}})
    logging.info("raw[%s] %s: %s", rid, ev_type, (text[:200]+"…") if len(text)>200 else text)
    return rid

# summarization + topic merge
def _topic_key(text:str)->str:
    seed=re.sub(r"[^a-z0-9 ]+"," ",re.sub(r"https?://\S+","",text.lower()))
    seed=" ".join(seed.split()[:12])
    return hashlib.sha1(seed.encode()).hexdigest()

def _merge_contents(old:str, new:str)->str:
    try:
        return AI.chat(
            [{"role":"system","content":"Merge NEW into EXISTING. ≤120 words, bullets ok. Preserve key facts/numbers."},
             {"role":"user","content":f"EXISTING:\n{old}\n\nNEW:\n{new}"}],
            max_tokens=240
        )
    except Exception:
        return (old+"\n"+new)[:1200]

def _humane_summarize(text:str, capture_tone:bool=True)->dict:
    sysmsg="Compress into human memory: 3–6 bullets or 2–4 short sentences; essentials only."
    if capture_tone and HUMANE_TONE:
        sysmsg+=" Detect tone (positive/neutral/negative + brief)."
    prompt=f"Digest this into humane memory.\n\nTEXT:\n{text}\n\nReturn JSON: title, summary, tags (3-6), sentiment."
    try:
        out=AI.chat([{"role":"system","content":sysmsg},{"role":"user","content":prompt}], max_tokens=360)
        try:
            data=json.loads(out)
        except Exception:
            data={"title":"","summary":out.strip(),"tags":"","sentiment":""}
        return {"title":(data.get("title") or "")[:120],
                "summary":(data.get("summary") or out).strip(),
                "tags":(data.get("tags") or "").replace("\n"," ").strip(),
                "sentiment":(data.get("sentiment") or "").strip()}
    except Exception:
        return {"title":"","summary":text[:900],"tags":"","sentiment":""}

def remember(source:str, text:str, raw_ref:str="")->int:
    digest=_humane_summarize(text, True)
    topic=_topic_key(digest["summary"] or text)
    cur.execute("SELECT id, content FROM notes WHERE topic_key=? ORDER BY id DESC LIMIT 1",(topic,))
    row=cur.fetchone()
    if row:
        nid, existing=row
        merged=_merge_contents(existing, digest["summary"])
        cur.execute("""UPDATE notes SET ts=?,source=?,title=?,content=?,tags=?,sentiment=?,raw_ref=? WHERE id=?""",
                    (datetime.utcnow().isoformat(),source,digest["title"],merged,digest["tags"],digest["sentiment"],raw_ref,nid))
        conn.commit()
        return nid
    cur.execute("""INSERT INTO notes(ts,source,topic_key,title,content,tags,sentiment,raw_ref)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (datetime.utcnow().isoformat(),source,topic,digest["title"],digest["summary"],digest["tags"],digest["sentiment"],raw_ref))
    conn.commit()
    return cur.lastrowid

# ---------- simple lexical search ----------
_WORD_RE=re.compile(r"[a-z0-9]+")
def _tokenize(s:str)->List[str]:
    return _WORD_RE.findall(s.lower())

def _tf(tokens:List[str])->Dict[str,float]:
    d={}; n=float(len(tokens)) or 1.0
    for t in tokens:
        d[t]=d.get(t,0)+1.0
    for k in d:
        d[k]/=n
    return d

def _cosine(a:Dict[str,float], b:Dict[str,float])->float:
    if not a or not b:
        return 0.0
    common=set(a)&set(b)
    num=sum(a[t]*b[t] for t in common)
    da=math.sqrt(sum(v*v for v in a.values()))
    db=math.sqrt(sum(v*v for v in b.values()))
    return 0.0 if da==0 or db==0 else num/(da*db)

def search_memory(query:str, k_raw:int=12, k_notes:int=12)->dict:
    qtf=_tf(_tokenize(query))
    cur.execute("SELECT id,ts,type,text,meta FROM raw_events ORDER BY id DESC LIMIT 4000")
    raw_scored=[]
    for r in cur.fetchall():
        txt=r[3] or ""
        score=_cosine(qtf,_tf(_tokenize(txt)))
        if score>0:
            raw_scored.append((score,{"id":r[0],"ts":r[1],"type":r[2],"text":txt,"meta":r[4]}))
    raw_top=[x[1] for x in sorted(raw_scored,key=lambda z:z[0],reverse=True)[:k_raw]]

    cur.execute("SELECT id,ts,source,title,content,tags,sentiment,raw_ref FROM notes ORDER BY id DESC LIMIT 4000")
    note_scored=[]
    for r in cur.fetchall():
        txt=(r[4] or "")+" "+(r[3] or "")+" "+(r[5] or "")
        score=_cosine(qtf,_tf(_tokenize(txt)))
        if score>0:
            note_scored.append((score,{"id":r[0],"ts":r[1],"source":r[2],
                                       "title":r[3] or "","content":r[4] or "",
                                       "tags":r[5] or "","sentiment":r[6] or "","ref":r[7] or ""}))
    notes_top=[x[1] for x in sorted(note_scored,key=lambda z:z[0],reverse=True)[:k_notes]]
    return {"raw":raw_top,"notes":notes_top}

def build_context_blurb(found:dict, max_chars:int=3800)->str:
    parts=[]
    for n in found.get("notes",[]):
        blk=f"[NOTE #{n['id']} | {n['ts']} | {n['source']} | {n['title']}] {n['content']}"
        if n["tags"]:
            blk+=f" (tags: {n['tags']})"
        parts.append(blk)
    for r in found.get("raw",[]):
        txt=r["text"]
        txt=txt[:600]+"…" if len(txt)>600 else txt
        parts.append(f"[RAW #{r['id']} | {r['ts']} | {r['type']}] {txt}")
    return "\n\n".join(parts)[:max_chars]

async def rag_answer(question:str)->str:
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
    return await ask_ai(prompt, context="You are Alex — precise, synthesizes across memory, cites context ids.")

# ---------- URL crawler + SEO summariser ----------
async def fetch_url(url:str)->str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=25, headers={"User-Agent":"Mozilla/5.0"}) as r:
                if r.status!=200:
                    return f"⚠️ HTTP {r.status}"
                text=await r.text()
        soup=BeautifulSoup(text,"html.parser")
        title=soup.title.string.strip() if soup.title and soup.title.string else "No title"
        desc=(soup.find("meta",{"name":"description"}) or {}).get("content","")
        h1=soup.h1.get_text(strip=True) if soup.h1 else ""
        words=len(soup.get_text(' ').split())
        links=len(soup.find_all("a"))
        images=len(soup.find_all("img"))
        snippet=soup.get_text(" ")[:1800]
        return f"🌐 {title}\nDesc: {desc[:200]}\nH1: {h1}\nWords:{words} Links:{links} Images:{images}\n\nSnippet:\n{snippet}"
    except Exception as e:
        logging.exception("Crawl error")
        return f"⚠️ Crawl error: {e}"

async def analyze_url(url:str)->str:
    content=await fetch_url(url)
    if content.startswith("⚠️"):
        return content
    summary=await ask_ai("Summarize page in short bullets; key entities, actions, SEO opportunities:\n\n"+content)
    log_raw("link", summary, {"url":url})
    remember("link", summary, raw_ref=url)
    return summary

# ---------- File analyzers ----------
def analyze_excel(path:Path)->str:
    try:
        df=pd.read_excel(path)
        head=", ".join(map(str,list(df.columns)[:12]))
        info=f"✅ Excel loaded: {df.shape[0]} rows × {df.shape[1]} cols\nColumns: {head}"
        num=df.select_dtypes(include="number")
        if not num.empty:
            info+="\n\nNumeric summary:\n"+num.describe().to_string()[:1800]
        log_raw("file", f"Excel {path.name}: shape {df.shape}; columns {list(df.columns)!r}", {"file":str(path)})
        remember("file", f"Excel {path.name}: shape {df.shape}; columns {list(df.columns)!r}", raw_ref=str(path))
        return info
    except Exception as e:
        logging.exception("Excel analysis error")
        return f"⚠️ Excel analysis error: {e}"

def analyze_csv(path:Path)->str:
    try:
        df=pd.read_csv(path, nrows=50000)
        head=", ".join(map(str,list(df.columns)[:12]))
        info=f"✅ CSV loaded: {df.shape[0]} rows × {df.shape[1]} cols\nColumns: {head}"
        num=df.select_dtypes(include="number")
        if not num.empty:
            info+="\n\nNumeric summary:\n"+num.describe().to_string()[:1800]
        log_raw("file", f"CSV {path.name}: shape {df.shape}; columns {list(df.columns)!r}", {"file":str(path)})
        remember("file", f"CSV {path.name}: shape {df.shape}; columns {list(df.columns)!r}", raw_ref=str(path))
        return info
    except Exception as e:
        logging.exception("CSV analysis error")
        return f"⚠️ CSV analysis error: {e}"

def analyze_json(path:Path)->str:
    try:
        raw=path.read_text(encoding="utf-8", errors="ignore")
        data=json.loads(raw)
        if isinstance(data,list):
            shape=f"list[{len(data)}]"
            preview=json.dumps(data[:3], ensure_ascii=False)[:1500]
        elif isinstance(data,dict):
            shape=f"dict({len(data.keys())} keys)"
            preview=json.dumps({k:data[k] for k in list(data.keys())[:10]}, ensure_ascii=False)[:1500]
        else:
            shape=type(data).__name__
            preview=str(data)[:1500]
        info=f"✅ JSON loaded: {shape}\nPreview: {preview}"
        log_raw("file", f"JSON {path.name}: shape {shape}", {"file":str(path)})
        remember("file", f"JSON {path.name}: shape {shape}\nPreview: {preview}", raw_ref=str(path))
        return info
    except Exception as e:
        logging.exception("JSON analysis error")
        return f"⚠️ JSON analysis error: {e}"

# ---------- Image analyzer (EXIF + optional OCR) ----------
def _exif_dict(im:Image.Image)->dict:
    try:
        exif=im._getexif() or {}
        label={ExifTags.TAGS.get(k,k):v for k,v in exif.items()}
        keep={}
        for k in ["DateTime","Make","Model","Software","LensModel","Orientation","ExifVersion","XResolution","YResolution"]:
            if k in label:
                keep[k]=label[k]
        return keep
    except Exception:
        return {}

def analyze_image(path:Path)->str:
    meta={"file":str(path)}
    try:
        im=Image.open(path)
        exif=_exif_dict(im)
        meta["exif"]=exif
        if HAS_TESS:
            try:
                import pytesseract
                ocr=pytesseract.image_to_string(im) or ""
            except Exception as e:
                ocr=f"(OCR unavailable: {e})"
        else:
            ocr="(OCR module not installed)"
        rid=log_raw("image", ocr if ocr else "(no OCR text)", meta)
        gist=f"Image {path.name}: EXIF {exif if exif else '∅'}; OCR preview: {(ocr[:500]+'…') if ocr and len(ocr)>500 else (ocr or '∅')}"
        remember("image", gist, raw_ref=str(path))
        return f"🖼️ Image saved. EXIF keys: {list(exif.keys()) if exif else 'none'}. OCR length: {len(ocr)} chars. (raw id {rid})"
    except Exception as e:
        log_raw("image", f"Image load error {path.name}: {e}", {"file": str(path)})
        logging.exception("Image analysis error")
        return f"⚠️ Image analysis error: {e}"
        # ---------- SERP (optional) ----------
def serp_search_snippet(query:str)->str:
    if not SERPAPI_KEY:
        return "⚠️ SERPAPI_KEY not set."
    try:
        r=requests.get(
            "https://serpapi.com/search",
            params={"q":query,"hl":"en","api_key":SERPAPI_KEY},
            timeout=25
        )
        j=r.json()
        snip=j.get("organic_results",[{}])[0].get("snippet","(no results)")
        return f"🔎 {query}\n{snip}"
    except Exception as e:
        logging.exception("Search error")
        return f"Search error: {e}"

# ---------- Telegram handlers ----------
async def start_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hey, I'm Alex 🤖 (hybrid brain: Ollama→OpenAI fallback)\n"
        "Core: /ai <prompt>, /ask <question>, prefix 'jarvis:' for Jarvis mode\n"
        "Ingest: /analyze <url>, send images & .xlsx/.csv/.json files\n"
        "Memory: /remember <text>, /mem [n], /exportmem, /raw [n]\n"
        "Logs: /setlog <path>, /subscribe_logs, /unsubscribe_logs, /logs [n]\n"
        "Search: /search <query>\n"
        "Voice: /speak <text> (TTS, if enabled)\n"
        "Backend: /backend <auto|openai|ollama>, /ollama_list, /ollama_add <url>, /ollama_status\n"
        "Dangerous ops (owner approval): /queue, /approve <id>, /deny <id>, /shell <cmd>, /py <code>\n"
        "Jarvis: /jarvis_on, /jarvis_off | Guardrails: /trust_on, /trust_off"
    )

async def id_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Your chat id: `{update.effective_chat.id}`", parse_mode="Markdown")

async def uptime_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    u=int(time.time()-START_TIME); h,m,s=u//3600,(u%3600)//60,u%60
    await update.message.reply_text(f"⏱️ Uptime {h}h {m}m {s}s")

async def config_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    flags={
        "BACKEND_MODE":STATE["backend_mode"],
        "OPENAI_MODEL":OPENAI_MODEL,
        "OLLAMA_MODEL":OLLAMA_MODEL,
        "HUMANE_TONE":HUMANE_TONE,
        "HAS_TESS":HAS_TESS,
        "SERPAPI_KEY_set":bool(SERPAPI_KEY),
        "DB_PATH":DB_PATH,
        "SAVE_DIR":str(SAVE_DIR),
        "PORT":PORT,
        "USE_OPENAI_STT":USE_OPENAI_STT,
        "USE_OPENAI_TTS":USE_OPENAI_TTS,
        "JARVIS_MODE":STATE["jarvis_mode"],
        "DEV_MODE":STATE["dev_mode"],
        "OLLAMA_URLS":AI.list_ollama_urls()
    }
    await update.message.reply_text("Config:\n"+json.dumps(flags, indent=2))

async def backup_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    try:
        zpath=Path("alex_backup.zip")
        with zipfile.ZipFile(zpath,"w",compression=zipfile.ZIP_DEFLATED) as z:
            if Path(DB_PATH).exists():
                z.write(DB_PATH)
            if RAW_JSONL.exists():
                z.write(RAW_JSONL)
            manifest={"files":[str(p) for p in SAVE_DIR.glob("*") if p.is_file()]}
            man_path=Path("received_manifest.json")
            man_path.write_text(json.dumps(manifest, indent=2))
            z.write(man_path)
        await update.message.reply_document(document=str(zpath), filename=zpath.name)
    except Exception as e:
        logging.exception("Backup error")
        await update.message.reply_text(f"Backup error: {e}")

# ---- App logic helpers ----
def _maybe_jarvis_text(text:str)->Tuple[bool,str]:
    t=text.strip()
    lowered=t.lower()
    if lowered.startswith("jarvis:") or lowered.startswith("jarvis,"):
        return True, (t.split(":",1)[1].strip() if ":" in t else t.split(",",1)[1].strip())
    return False, t

async def ai_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    q=" ".join(ctx.args)
    if not q:
        return await update.message.reply_text("Usage: /ai <your prompt>")
    is_jarvis, q2=_maybe_jarvis_text(q)
    persona = persona_prompt() if not is_jarvis else ("You are Alex (Jarvis) — voice-friendly, decisive, short.")
    ans=AI.chat(
        [{"role":"system","content":persona},{"role":"user","content":q2}],
        max_tokens=800
    )
    log_raw("chat_user", q, {"chat_id":update.effective_chat.id})
    log_raw("chat_alex", ans, {"chat_id":update.effective_chat.id})
    remember("chat", f"Q: {q}\nA gist: {ans[:400]}")
    await update.message.reply_text(ans)

async def ask_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    q=" ".join(ctx.args).strip()
    if not q:
        return await update.message.reply_text("Usage: /ask <question about anything I've seen/learned>")
    log_raw("chat_user", f"/ask {q}", {"chat_id":update.effective_chat.id})
    ans=await rag_answer(q)
    log_raw("chat_alex", ans, {"chat_id":update.effective_chat.id})
    await update.message.reply_text(ans)

async def search_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /search <query>")
    query=" ".join(ctx.args)
    out=serp_search_snippet(query)
    log_raw("chat_user", f"/search {query}", {"chat_id":update.effective_chat.id})
    log_raw("chat_alex", out, {"chat_id":update.effective_chat.id})
    if out.startswith("🔎"):
        remember("chat", f"SERP for '{query}': {out.splitlines()[-1]}")
    await update.message.reply_text(out)

async def analyze_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /analyze <url>")
    url=ctx.args[0]
    await update.message.reply_text("🔍 Crawling and summarizing…")
    res=await analyze_url(url)
    await update.message.reply_text(res)

async def remember_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    text=" ".join(ctx.args).strip()
    if not text:
        return await update.message.reply_text("Usage: /remember <text to add to memory>")
    rid=log_raw("chat_user", f"/remember {text}", {"chat_id":update.effective_chat.id})
    nid=remember("chat", text)
    await update.message.reply_text(f"🧠 Noted (note id {nid}, raw id {rid}). Essence kept, details in raw log.")

async def mem_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    n=8
    if ctx.args:
        try:
            n=max(1,min(40,int(ctx.args[0])))
        except:
            pass
    cur.execute("SELECT id,ts,source,title,content,tags,sentiment,raw_ref FROM notes ORDER BY id DESC LIMIT ?",(n,))
    rows=cur.fetchall()
    if not rows:
        return await update.message.reply_text("🧠 Memory is empty (for now).")
    lines=[]
    for r in rows:
        lines.append(f"#{r[0]} [{r[2]}] {r[3] or '(no title)'}")
        lines.append("  "+(r[4] or "").replace("\n","\n  ")[:700])
        if r[5]:
            lines.append(f"  tags: {r[5]}")
        if r[7]:
            lines.append(f"  ref: {r[7]}")
        lines.append("")
    await update.message.reply_text("\n".join(lines)[:3900])

async def exportmem_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    cur.execute("SELECT id,ts,source,title,content,tags,sentiment,raw_ref FROM notes ORDER BY id ASC")
    rows=cur.fetchall()
    data=[{"id":r[0],"ts":r[1],"source":r[2],"title":r[3] or "","content":r[4] or "",
           "tags":r[5] or "","sentiment":r[6] or "","ref":r[7] or ""} for r in rows]
    path=Path("memory_export.json")
    path.write_text(json.dumps(data, indent=2))
    await update.message.reply_document(document=str(path), filename="alex_memory.json")

async def raw_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    n=30
    if ctx.args:
        try:
            n=max(1,min(200,int(ctx.args[0])))
        except:
            pass
    cur.execute("SELECT id,ts,type,text,meta FROM raw_events ORDER BY id DESC LIMIT ?",(n,))
    rows=cur.fetchall()
    if not rows:
        return await update.message.reply_text("Raw memory is empty.")
    out=[]
    for r in rows:
        try:
            meta=json.loads(r[4] or "{}")
        except:
            meta={}
        txt=(r[3] or "").replace("\n"," ")[:800]
        mtxt=f" | meta: {meta}" if meta else ""
        out.append(f"#{r[0]} [{r[1]} | {r[2]}] {txt}{mtxt}")
    await update.message.reply_text("\n".join(out)[:3900])

# ----- owner approval flow for dangerous ops -----
PENDING_CMDS: Dict[str, Dict[str,Any]] = {}  # id -> {type:'shell'|'py', 'cmd'|'code', 'chat_id'}

async def queue_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not _owner_only(update):
        return await update.message.reply_text("🚫 Owner only.")
    if not PENDING_CMDS:
        return await update.message.reply_text("✅ Queue empty.")
    lines=[]
    for k,v in PENDING_CMDS.items():
        lines.append(f"{k} :: {v['type']} :: {(v.get('cmd') or v.get('code'))[:80]}")
    await update.message.reply_text("\n".join(lines)[:3900])

async def approve_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not _owner_only(update):
        return await update.message.reply_text("🚫 Owner only.")
    if not ctx.args:
        return await update.message.reply_text("Usage: /approve <id>")
    rid=ctx.args[0].strip()
    job=PENDING_CMDS.pop(rid, None)
    if not job:
        return await update.message.reply_text("Not found.")
    if job["type"]=="shell":
        try:
            result=subprocess.check_output(job["cmd"], shell=True, text=True,
                                           stderr=subprocess.STDOUT, timeout=60)
            await update.message.reply_text(f"💻 OK ({rid})\n{result[:3900]}")
        except subprocess.CalledProcessError as e:
            await update.message.reply_text(f"❌ Error ({rid}):\n{(e.output or str(e))[:3900]}")
        except Exception as e:
            await update.message.reply_text(f"❌ Exec error: {e}")
    elif job["type"]=="py":
        try:
            loc={}
            exec(job["code"], {}, loc)  # sandbox-light; trust owner only
            await update.message.reply_text(f"🐍 Py OK ({rid}) — keys: {list(loc.keys())[:12]}")
        except Exception as e:
            await update.message.reply_text(f"❌ Py error ({rid}): {e}")
    else:
        await update.message.reply_text("Unknown job type.")

async def deny_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not _owner_only(update):
        return await update.message.reply_text("🚫 Owner only.")
    if not ctx.args:
        return await update.message.reply_text("Usage: /deny <id>")
    rid=ctx.args[0].strip()
    if rid in PENDING_CMDS:
        PENDING_CMDS.pop(rid, None)
        await update.message.reply_text(f"🛑 Denied {rid}.")
    else:
        await update.message.reply_text("Not found.")
        async def shell_request_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    cmd=" ".join(ctx.args).strip()
    if not cmd:
        return await update.message.reply_text("Usage: /shell <command>")
    job_id=str(uuid.uuid4())[:8]
    PENDING_CMDS[job_id]={"type":"shell","cmd":cmd,"chat_id":update.effective_chat.id}
    await update.message.reply_text(
        f"⌛ Queued shell command for approval. ID: `{job_id}`",
        parse_mode="Markdown"
    )

async def py_request_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    code=" ".join(ctx.args).strip()
    if not code:
        return await update.message.reply_text("Usage: /py <single-line python> (multi-line not supported here)")
    job_id=str(uuid.uuid4())[:8]
    PENDING_CMDS[job_id]={"type":"py","code":code,"chat_id":update.effective_chat.id}
    await update.message.reply_text(
        f"⌛ Queued python for approval. ID: `{job_id}`",
        parse_mode="Markdown"
    )

# --- uploads ---
async def handle_file(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    doc=update.message.document
    if not doc:
        return
    file_path=SAVE_DIR/doc.file_name
    tg_file=await ctx.bot.get_file(doc.file_id)
    await tg_file.download_to_drive(file_path)
    log_raw("file", f"Received file {doc.file_name}", {"file":str(file_path), "chat_id":update.effective_chat.id})
    await update.message.reply_text(f"📂 Saved `{doc.file_name}` — analyzing…", parse_mode="Markdown")
    name=doc.file_name.lower()
    if   name.endswith(".xlsx"): out=analyze_excel(file_path)
    elif name.endswith(".csv"):  out=analyze_csv(file_path)
    elif name.endswith(".json"): out=analyze_json(file_path)
    else:
        remember("file", f"Received file {doc.file_name}", raw_ref=str(file_path))
        out="Saved. I analyze Excel/CSV/JSON."
    log_raw("chat_alex", out, {"chat_id":update.effective_chat.id})
    await update.message.reply_text(out)

async def handle_photo(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        return
    photo=update.message.photo[-1]
    file=await ctx.bot.get_file(photo.file_id)
    path=SAVE_DIR/f"photo_{photo.file_id}.jpg"
    await file.download_to_drive(path)
    out=analyze_image(path)
    log_raw("chat_alex", out, {"chat_id":update.effective_chat.id})
    await update.message.reply_text(out)

async def handle_text(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    text=(update.message.text or "").strip()
    log_raw("chat_user", text, {"chat_id":update.effective_chat.id})

    # URL auto-analyze
    if text.startswith(("http://","https://")):
        await update.message.reply_text("🔍 Got your link — analyzing…")
        res=await analyze_url(text)
        log_raw("chat_alex", res, {"chat_id":update.effective_chat.id})
        return await update.message.reply_text(res)

    # Jarvis prefix
    is_jarvis, prompt=_maybe_jarvis_text(text)
    sysmsg = ("You are Alex (Jarvis) — voice-friendly, decisive, short.") if is_jarvis else persona_prompt()
    ans=AI.chat([{"role":"system","content":sysmsg},{"role":"user","content":prompt}], max_tokens=800)
    remember("chat", f"User said: {text}\nResponse gist: {ans[:400]}")
    log_raw("chat_alex", ans, {"chat_id":update.effective_chat.id})
    await update.message.reply_text(ans)

# ---------- live log watcher ----------
def _post_to_subscribers(text:str):
    if not GLOBAL_APP or not MEM_RUNTIME.get("subscribers"):
        return
    loop=GLOBAL_APP.bot._application.loop
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

TRADE_PATTERNS=[
    re.compile(r"\b(BUY|SELL)\b.*?(\b[A-Z]{2,10}\b).*?qty[:= ]?(\d+).*?price[:= ]?([0-9.]+)", re.I),
    re.compile(r"order\s+(buy|sell)\s+(\w+).+?@([0-9.]+).+?qty[:= ]?(\d+)", re.I),
]

def summarize_trade_line(line:str)->str|None:
    for pat in TRADE_PATTERNS:
        m=pat.search(line)
        if m:
            side,sym,qty,price=list(m.groups())
            try:
                qty_i=int(qty)
            except:
                qty_i=qty
            return f"🟢 {side.upper()} {sym} qty {qty_i} @ {price}"
    return None

def watch_logs():
    last_size=0; last_raw_push=0
    while True:
        try:
            path=MEM_RUNTIME.get("log_path") or ""
            if not path or not Path(path).exists():
                time.sleep(2); continue
            p=Path(path); sz=p.stat().st_size
            if sz<last_size: last_size=0
            if sz>last_size:
                with p.open("r", errors="ignore") as f:
                    if last_size: f.seek(last_size)
                    new=f.read(); last_size=sz
                for line in new.splitlines():
                    s=summarize_trade_line(line)
                    if s:
                        _post_to_subscribers(s)
                        log_raw("log", line, {"path":path})
                        remember("log", s, raw_ref=path)
                    now=time.time()
                    if now-last_raw_push>15:
                        last_raw_push=now
                        _post_to_subscribers("📜 "+line[:900])
        except Exception as e:
            logging.error("log watcher error: %s", e)
        time.sleep(1)
        # ---------- persona synthesis worker ----------
def learning_worker():
    while True:
        try:
            cur.execute("SELECT content FROM notes ORDER BY id DESC LIMIT 24")
            rec=[r[0] for r in cur.fetchall()]
            if rec:
                persona=AI.chat(
                    [{"role":"system","content":"Create concise persona guidance from these snippets. 4–6 bullets."},
                     {"role":"user","content":"\n\n".join(rec)}],
                    max_tokens=220
                )
                tk="__persona__"
                cur.execute("SELECT id FROM notes WHERE topic_key=? ORDER BY id DESC LIMIT 1",(tk,))
                row=cur.fetchone()
                if row:
                    cur.execute("UPDATE notes SET ts=?,source=?,title=?,content=?,tags=?,sentiment=? WHERE id=?",
                                (datetime.utcnow().isoformat(),"system","Persona",persona,"persona,profile","",row[0]))
                else:
                    cur.execute("""INSERT INTO notes(ts,source,topic_key,title,content,tags,sentiment,raw_ref)
                                   VALUES(?,?,?,?,?,?,?,?)""",
                                (datetime.utcnow().isoformat(),"system",tk,"Persona",persona,"persona,profile","",""))
                conn.commit()
        except Exception as e:
            logging.error("learning error: %s", e)
        time.sleep(60)

# ---------- Simple STT/TTS scaffolding (OpenAI-only, optional) ----------
def transcribe_bytes_wav(b:bytes)->str:
    if not USE_OPENAI_STT or not OPENAI_KEYS:
        return ""
    try:
        from openai import OpenAI as _New
        key=OPENAI_KEYS[0]
        cli=_New(api_key=key)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(b); tf.flush(); path=tf.name
        with open(path,"rb") as fh:
            r=cli.audio.transcriptions.create(model="whisper-1", file=fh)
        return (r.text or "").strip()
    except Exception as e:
        logging.warning("STT error: %s", e)
        return ""

def tts_to_mp3(text:str)->bytes:
    if not USE_OPENAI_TTS or not OPENAI_KEYS:
        return b""
    try:
        from openai import OpenAI as _New
        key=OPENAI_KEYS[0]
        cli=_New(api_key=key)
        r=cli.audio.speech.create(model="gpt-4o-mini-tts", voice=OPENAI_TTS_VOICE, input=text, format="mp3")
        return r.read() if hasattr(r,"read") else (getattr(r,"content",b"") or b"")
    except Exception as e:
        logging.warning("TTS error: %s", e)
        return b""

# ---------- Keep-alive HTTP + iOS Shortcuts webhook ----------
class Health(BaseHTTPRequestHandler):
    def _ok(self, body:bytes=b"ok", code:int=200, ctype:str="text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path=urlparse(self.path).path
        if path=="/": return self._ok(b"ok")
        if path=="/health": return self._ok(b"ok")
        if path=="/ollama_status":
            status={"mode": STATE["backend_mode"], "urls": AI.list_ollama_urls()}
            return self._ok(json.dumps(status).encode("utf-8"), 200, "application/json")
        return self._ok(b"not found", 404)

    def do_POST(self):
        path=urlparse(self.path).path
        try:
            length=int(self.headers.get("Content-Length","0"))
            raw=self.rfile.read(length) if length>0 else b""
            data=json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            data={}
        # /shortcut — called from iOS/Apple Watch Shortcuts
        if path=="/shortcut":
            if not SHORTCUT_SECRET or data.get("secret")!=SHORTCUT_SECRET:
                return self._ok(b'{"error":"unauthorized"}', 401, "application/json")
            q=(data.get("q") or "").strip()
            if not q:
                return self._ok(b'{"error":"missing q"}', 400, "application/json")
            ans=AI.chat(
                [{"role":"system","content":"You are Alex — concise, helpful, voice-friendly."},
                 {"role":"user","content":q}], max_tokens=300
            )
            remember("shortcut", f"Q: {q}\nA: {ans[:400]}")
            # Optional TTS
            audio=tts_to_mp3(ans) if USE_OPENAI_TTS else b""
            body={"answer":ans, "audio_base64": (audio.decode("latin1") if audio else "")}
            return self._ok(json.dumps(body).encode("utf-8"), 200, "application/json")
        return self._ok(b"not found", 404)
        class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

def run_health_server():
    # Use dynamic port if provided (important for deployment environments like Render/Heroku)
    port = int(os.getenv("PORT", PORT))
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, HealthHandler)
    logging.info(f"Health server running on port {port}")
    httpd.serve_forever()

# Run health server in a separate thread
health_thread = threading.Thread(target=run_health_server, daemon=True)
health_thread.start()

# ---------- Backend / Jarvis / Guardrails control commands ----------
async def backend_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not _owner_only(update):
        return await update.message.reply_text("🚫 Owner only.")
    if not ctx.args:
        return await update.message.reply_text("Usage: /backend <auto|openai|ollama>")
    mode=ctx.args[0].strip().lower()
    if mode not in ("auto","openai","ollama"):
        return await update.message.reply_text("Use one of: auto | openai | ollama")
    STATE["backend_mode"]=mode
    AI.set_backend_mode(mode)
    await update.message.reply_text(f"✅ Backend mode set to: {mode}")

async def ollama_add_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not _owner_only(update):
        return await update.message.reply_text("🚫 Owner only.")
    if not ctx.args:
        return await update.message.reply_text("Usage: /ollama_add <url>")
    url=" ".join(ctx.args).strip().rstrip("/")
    AI.add_ollama_url(url)
    await update.message.reply_text(f"✅ Added Ollama URL: {url}")

async def ollama_list_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    urls=AI.list_ollama_urls()
    await update.message.reply_text(
        "Ollama URLs (priority order):\n"+"\n".join(urls) if urls else "No URLs configured."
    )

async def ollama_status_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    urls=AI.list_ollama_urls()
    statuses=[]
    for u in urls:
        try:
            r=requests.get(f"{u}/api/version", timeout=6)
            statuses.append(f"{u} :: {r.status_code}")
        except Exception as e:
            statuses.append(f"{u} :: error {e}")
    await update.message.reply_text(
        "Ollama status:\n"+"\n".join(statuses) if statuses else "No URLs configured."
    )

async def jarvis_on_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not _owner_only(update):
        return await update.message.reply_text("🚫 Owner only.")
    STATE["jarvis_mode"]=True
    await update.message.reply_text("🧠 Jarvis mode: ON")

async def jarvis_off_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not _owner_only(update):
        return await update.message.reply_text("🚫 Owner only.")
    STATE["jarvis_mode"]=False
    await update.message.reply_text("🧠 Jarvis mode: OFF")

async def trust_on_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not _owner_only(update):
        return await update.message.reply_text("🚫 Owner only.")
    STATE["dev_mode"]=True
    await update.message.reply_text("⚙️ Guardrails: relaxed (still safe).")

async def trust_off_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not _owner_only(update):
        return await update.message.reply_text("🚫 Owner only.")
    STATE["dev_mode"]=False
    await update.message.reply_text("⚙️ Guardrails: standard.")

async def speak_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /speak <text>")
    text=" ".join(ctx.args)
    if not USE_OPENAI_TTS:
        return await update.message.reply_text("TTS disabled. Set USE_OPENAI_TTS=1.")
    audio=tts_to_mp3(text)
    if not audio:
        return await update.message.reply_text("TTS failed.")
    path=Path("speech.mp3"); path.write_bytes(audio)
    await update.message.reply_audio(audio=InputFile(str(path)), title="Alex says")
    # ---------- log path & subscription ----------
async def logs_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    n=40
    if ctx.args:
        try:
            n=max(1,min(400,int(ctx.args[0])))
        except:
            pass
    path=MEM_RUNTIME.get("log_path") or ""
    p=Path(path)
    if not path or not p.exists():
        return await update.message.reply_text("⚠️ No log path set or file missing. Use /setlog <path>.")
    try:
        lines=p.read_text(errors="ignore").splitlines()[-n:]
        msg="```\n"+"\n".join(lines)[-3500:]+"\n```"
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        logging.exception("logs read error")
        await update.message.reply_text(f"Read error: {e}")

async def setlog_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Usage: /setlog /path/to/your.log")
    path=" ".join(ctx.args)
    MEM_RUNTIME["log_path"]=path
    await update.message.reply_text(f"✅ Log path set to: `{path}`", parse_mode="Markdown")

async def subscribe_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    cid=update.effective_chat.id
    if cid not in MEM_RUNTIME["subscribers"]:
        MEM_RUNTIME["subscribers"].append(cid)
    await update.message.reply_text("🔔 Subscribed to live log updates.")

async def unsubscribe_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    cid=update.effective_chat.id
    if cid in MEM_RUNTIME["subscribers"]:
        MEM_RUNTIME["subscribers"].remove(cid)
    await update.message.reply_text("🔕 Unsubscribed.")

# ---------- Wire up Telegram + background threads ----------
def build_app()->Application:
    app=Application.builder().token(TELEGRAM_TOKEN).build()
    # commands
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("id", id_cmd))
    app.add_handler(CommandHandler("uptime", uptime_cmd))
    app.add_handler(CommandHandler("config", config_cmd))
    app.add_handler(CommandHandler("backup", backup_cmd))
    app.add_handler(CommandHandler("ai", ai_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("search", search_cmd))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("remember", remember_cmd))
    app.add_handler(CommandHandler("mem", mem_cmd))
    app.add_handler(CommandHandler("exportmem", exportmem_cmd))
    app.add_handler(CommandHandler("raw", raw_cmd))
    app.add_handler(CommandHandler("setlog", setlog_cmd))
    app.add_handler(CommandHandler("subscribe_logs", subscribe_cmd))
    app.add_handler(CommandHandler("unsubscribe_logs", unsubscribe_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))

    # backend / jarvis / guardrails
    app.add_handler(CommandHandler("backend", backend_cmd))
    app.add_handler(CommandHandler("ollama_add", ollama_add_cmd))
    app.add_handler(CommandHandler("ollama_list", ollama_list_cmd))
    app.add_handler(CommandHandler("ollama_status", ollama_status_cmd))
    app.add_handler(CommandHandler("jarvis_on", jarvis_on_cmd))
    app.add_handler(CommandHandler("jarvis_off", jarvis_off_cmd))
    app.add_handler(CommandHandler("trust_on", trust_on_cmd))
    app.add_handler(CommandHandler("trust_off", trust_off_cmd))
    app.add_handler(CommandHandler("speak", speak_cmd))

    # approval / dangerous ops
    app.add_handler(CommandHandler("queue", queue_cmd))
    app.add_handler(CommandHandler("approve", approve_cmd))
    app.add_handler(CommandHandler("deny", deny_cmd))
    app.add_handler(CommandHandler("shell", shell_request_cmd))
    app.add_handler(CommandHandler("py", py_request_cmd))

    # messages
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    return app
    # ---------- main ----------
def main():
    global GLOBAL_APP
    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN missing — exiting.")
        sys.exit(1)
    if AI.backend_mode in ("auto","openai") and not OPENAI_KEYS:
        logging.warning("OpenAI path may be used but no OPENAI_API_KEY(S) provided.")

    app=build_app(); GLOBAL_APP=app

    # background threads
    threading.Thread(target=run_health_server, daemon=True).start()
    threading.Thread(target=learning_worker, daemon=True).start()
    threading.Thread(target=watch_logs, daemon=True).start()

    logging.info("🚀 Alex mega bot starting…")
    logging.info("Jarvis=%s | DevMode=%s | Backend=%s | OllamaURLs=%s",
                 STATE['jarvis_mode'], STATE['dev_mode'], STATE['backend_mode'], AI.list_ollama_urls())
    app.run_polling(close_loop=False)

if __name__=="__main__":
    main()

# =========================
# Quick env checklist:
# TELEGRAM_TOKEN=<bot token>
# OWNER_ID=<your telegram numeric id>
# BACKEND=auto|openai|ollama
#   OPENAI_API_KEYS=key1,key2,...   (or OPENAI_API_KEY=single)
#   OPENAI_MODEL=gpt-4o-mini
# OLLAMA_URLS=https://YOUR-NGROK-ID.ngrok.app,http://YOUR-LAN-IP:11434,http://localhost:11434
#   OLLAMA_MODEL=llama3.1:8b-instruct
# SERPAPI_KEY=<optional>
# HUMANE_TONE=1
# SHORTCUT_SECRET=<something-long>  (for /shortcut endpoint)
# USE_OPENAI_STT=0|1, USE_OPENAI_TTS=0|1, OPENAI_TTS_VOICE=alloy
# PORT=8080
# JARVIS_MODE=0|1  (default Jarvis persona)
# DEV_MODE=0|1     (relaxed style; still safe; owner can /trust_on)
# =========================