# =========================
# mega.py — Alex (Railway/GitHub ready, compact + multi-provider, detailed logging)
# =========================
# Features:
# - Telegram bot (python-telegram-bot v20)
# - Raw append-only memory (SQLite) + JSONL backup
# - Humane summaries/merge + topic-key clustering
# - RAG-style recall (/ask) across notes + raw
# - URL crawler + SEO summarize + ingest
# - File analyzers: Excel/CSV/JSON + ingest
# - Image ingest: EXIF + (optional) OCR + ingest
# - Live log watcher (/setlog, /subscribe_logs, /logs) + trade-line summarizer
# - Persona auto-learning worker (periodically distills notes)
# - Health server (PORT for Railway), backup/export/config
# - SerpAPI search (/search)
# - NEW: Multi-provider LLM client with automatic fallback:
#   Ollama → OpenRouter → Groq → Together → OpenAI (configurable via ENV)
# =========================

import os, sys, json, time, re, logging, threading, hashlib, sqlite3, asyncio, subprocess, math, zipfile
from pathlib import Path
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------- bootstrap: install/import (auto-installs missing deps) ----------
def _ensure(import_name: str, pip_name: str | None = None):
    try: return __import__(import_name)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_name or import_name])
        return __import__(import_name)

requests=_ensure("requests"); aiohttp=_ensure("aiohttp")
_ensure("bs4","beautifulsoup4"); _ensure("python-telegram-bot","python-telegram-bot==20.*")
telegram=_ensure("telegram"); _ensure("pandas"); _ensure("openpyxl"); _ensure("PIL","Pillow")

# Optional OCR (requires tesseract system binary to actually OCR)
try: _ensure("pytesseract"); HAS_TESS=True
except Exception: HAS_TESS=False

from PIL import Image, ExifTags
from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import pandas as pd

# ---------- config / logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
TELEGRAM_TOKEN=os.getenv("TELEGRAM_TOKEN","").strip()
SERPAPI_KEY=os.getenv("SERPAPI_KEY","").strip()
PORT=int(os.getenv("PORT","8080"))
HUMANE_TONE=os.getenv("HUMANE_TONE","1")=="1"

# Provider env (set any subset; order matters for fallback)
# 1) OLLAMA
OLLAMA_BASE=os.getenv("OLLAMA_BASE_URL","").rstrip("/")
OLLAMA_MODEL=os.getenv("OLLAMA_MODEL","")
# 2) OPENROUTER
OPENROUTER_KEY=os.getenv("OPENROUTER_API_KEY","").strip()
OPENROUTER_MODEL=os.getenv("OPENROUTER_MODEL","meta-llama/llama-3-70b-instruct")
# 3) GROQ
GROQ_KEY=os.getenv("GROQ_API_KEY","").strip()
GROQ_MODEL=os.getenv("GROQ_MODEL","llama3-70b-8192")
# 4) TOGETHER
TOGETHER_KEY=os.getenv("TOGETHER_API_KEY","").strip()
TOGETHER_MODEL=os.getenv("TOGETHER_MODEL","meta-llama/Meta-Llama-3-70B-Instruct-Turbo")
# 5) OPENAI
OPENAI_KEY=os.getenv("OPENAI_API_KEY","").strip()
OPENAI_MODEL=os.getenv("OPENAI_MODEL","gpt-4o-mini")

if not TELEGRAM_TOKEN: logging.warning("TELEGRAM_TOKEN not set.")
if not any([OLLAMA_BASE, OPENROUTER_KEY, GROQ_KEY, TOGETHER_KEY, OPENAI_KEY]):
    logging.warning("No LLM provider keys/URL set. You can still run, but AI answers will be limited.")

logging.info("OCR pytesseract: %s", "available" if HAS_TESS else "not available (EXIF only)")

START_TIME=time.time()
SAVE_DIR=Path("received_files"); SAVE_DIR.mkdir(exist_ok=True)

# ---------- Multi-provider AI client with automatic fallback ----------
class AIClient:
    """
    Tries providers in priority order. If a provider fails (timeout, rate limit, quota, network),
    it logs and falls through to the next one. Designed to work with a single key as well.
    Order (change by editing self.providers):
      1) Ollama (if OLLAMA_BASE_URL set)
      2) OpenRouter (if OPENROUTER_API_KEY set)
      3) Groq (if GROQ_API_KEY set)
      4) Together (if TOGETHER_API_KEY set)
      5) OpenAI (if OPENAI_API_KEY set)
    """
    def __init__(self):
        self.session = requests.Session()
        self.timeout = 30
        self.providers = []
        if OLLAMA_BASE and OLLAMA_MODEL:
            self.providers.append(("ollama", self._call_ollama))
        if OPENROUTER_KEY:
            self.providers.append(("openrouter", self._call_openrouter))
        if GROQ_KEY:
            self.providers.append(("groq", self._call_groq))
        if TOGETHER_KEY:
            self.providers.append(("together", self._call_together))
        if OPENAI_KEY:
            self.providers.append(("openai", self._call_openai))

    def chat(self, messages, max_tokens=800, sys_prompt=None):
        """
        Unified chat call. Will cycle providers until one succeeds.
        Returns text or a short diagnostic if none are available.
        """
        payload = {"messages": messages, "max_tokens": max_tokens, "sys_prompt": sys_prompt or ""}
        errors = []
        for name, fn in self.providers:
            try:
                logging.info("AI provider attempt: %s", name)
                out = fn(payload)
                if out and isinstance(out, str):
                    logging.info("AI provider success: %s", name)
                    return out.strip()
                errors.append(f"{name}: empty response")
            except Exception as e:
                msg = f"{name} error: {e}"
                logging.warning(msg)
                errors.append(msg)
        if not self.providers:
            return "⚠️ No AI providers configured. Add OLLAMA_BASE_URL or any *_API_KEY."
        return "⚠️ All AI providers failed:\n" + "\n".join(errors[:4])

    # ----- Provider adapters -----
    def _call_ollama(self, p):
        """
        Ollama HTTP API: POST /api/chat
        Requires OLLAMA_BASE_URL + OLLAMA_MODEL. Great when you run your own host.
        """
        url=f"{OLLAMA_BASE}/api/chat"
        data={
            "model": OLLAMA_MODEL,
            "messages": [{"role":m["role"],"content":m["content"]} for m in p["messages"]],
            "options": {"num_predict": p["max_tokens"]},
            "stream": False
        }
        r=self.session.post(url, json=data, timeout=self.timeout)
        r.raise_for_status()
        j=r.json()
        return (j.get("message",{}) or {}).get("content","")

    def _call_openrouter(self, p):
        """
        OpenRouter REST: POST https://openrouter.ai/api/v1/chat/completions
        Single API key, many models. Good 'one-key' option from phone.
        """
        url="https://openrouter.ai/api/v1/chat/completions"
        headers={"Authorization":f"Bearer {OPENROUTER_KEY}","Content-Type":"application/json"}
        data={
            "model": OPENROUTER_MODEL,
            "messages": [{"role":m["role"],"content":m["content"]} for m in p["messages"]],
            "max_tokens": p["max_tokens"]
        }
        r=self.session.post(url, headers=headers, json=data, timeout=self.timeout)
        r.raise_for_status()
        j=r.json()
        return j["choices"][0]["message"]["content"]

    def _call_groq(self, p):
        """
        Groq REST: POST https://api.groq.com/openai/v1/chat/completions
        Very fast Llama-3. Often generous free limits; great fallback.
        """
        url="https://api.groq.com/openai/v1/chat/completions"
        headers={"Authorization":f"Bearer {GROQ_KEY}","Content-Type":"application/json"}
        data={
            "model": GROQ_MODEL,
            "messages": [{"role":m["role"],"content":m["content"]} for m in p["messages"]],
            "max_tokens": p["max_tokens"]
        }
        r=self.session.post(url, headers=headers, json=data, timeout=self.timeout)
        r.raise_for_status()
        j=r.json()
        return j["choices"][0]["message"]["content"]

    def _call_together(self, p):
        """
        Together REST: POST https://api.together.xyz/v1/chat/completions
        Big frontier models hosted; good capacity.
        """
        url="https://api.together.xyz/v1/chat/completions"
        headers={"Authorization":f"Bearer {TOGETHER_KEY}","Content-Type":"application/json"}
        data={
            "model": TOGETHER_MODEL,
            "messages": [{"role":m["role"],"content":m["content"]} for m in p["messages"]],
            "max_tokens": p["max_tokens"]
        }
        r=self.session.post(url, headers=headers, json=data, timeout=self.timeout)
        r.raise_for_status()
        j=r.json()
        return j["choices"][0]["message"]["content"]

    def _call_openai(self, p):
        """
        OpenAI (optional fallback). Uses REST for zero extra deps.
        """
        url="https://api.openai.com/v1/chat/completions"
        headers={"Authorization":f"Bearer {OPENAI_KEY}","Content-Type":"application/json"}
        data={
            "model": OPENAI_MODEL,
            "messages": [{"role":m["role"],"content":m["content"]} for m in p["messages"]],
            "max_tokens": p["max_tokens"]
        }
        r=self.session.post(url, headers=headers, json=data, timeout=self.timeout)
        r.raise_for_status()
        j=r.json()
        return j["choices"][0]["message"]["content"]

AI = AIClient()

async def ask_ai(prompt:str, context:str="")->str:
    """
    Unified AI ask. If no providers configured, returns a helpful notice.
    """
    messages = [{"role":"system","content": context or "You are Alex — concise, helpful, a little witty."},
                {"role":"user","content": prompt}]
    return AI.chat(messages, max_tokens=900)

# ---------- SQLite: RAW memory (append-only) + NOTES (summaries) ----------
DB_PATH="alex_memory.db"; conn=sqlite3.connect(DB_PATH, check_same_thread=False); cur=conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS raw_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, type TEXT NOT NULL, text TEXT NOT NULL, meta TEXT)""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_events(ts)")
cur.execute("""CREATE TABLE IF NOT EXISTS notes(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, source TEXT NOT NULL, topic_key TEXT NOT NULL,
  title TEXT, content TEXT NOT NULL, tags TEXT, sentiment TEXT, raw_ref TEXT)""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_topic ON notes(topic_key)"); conn.commit()

RAW_JSONL=Path("raw_events.jsonl")
def _append_jsonl(obj:dict):
    try:
        with RAW_JSONL.open("a", encoding="utf-8") as f: f.write(json.dumps(obj, ensure_ascii=False)+"\n")
    except Exception as e: logging.error("JSONL append error: %s", e)

def log_raw(ev_type:str, text:str, meta:dict|None=None)->int:
    ts=datetime.utcnow().isoformat(); m=json.dumps(meta or {}, ensure_ascii=False)
    cur.execute("INSERT INTO raw_events(ts,type,text,meta) VALUES(?,?,?,?)",(ts,ev_type,text,m)); conn.commit()
    rid=cur.lastrowid; _append_jsonl({"id":rid,"ts":ts,"type":ev_type,"text":text,"meta":meta or {}})
    logging.info("raw[%s] %s: %s", rid, ev_type, (text[:160]+"…") if len(text)>160 else text); return rid

# ---------- Summarisation + topic merge ----------
def _topic_key(text:str)->str:
    seed=re.sub(r"[^a-z0-9 ]+"," ",re.sub(r"https?://\S+","",text.lower())); seed=" ".join(seed.split()[:12])
    return hashlib.sha1(seed.encode()).hexdigest()

def _merge_contents(old:str, new:str)->str:
    try:
        return AI.chat(
            [{"role":"system","content":"Merge NEW into EXISTING. ≤120 words, bullet style OK. Preserve key facts/numbers."},
             {"role":"user","content":f"EXISTING:\n{old}\n\nNEW:\n{new}"}],
            max_tokens=240
        )
    except Exception: return (old+"\n"+new)[:1200]

def _humane_summarize(text:str, capture_tone:bool=True)->dict:
    sysmsg="Compress into human memory: 3–6 bullets or 2–4 short sentences; essentials only."
    if capture_tone and HUMANE_TONE: sysmsg+=" Detect tone (positive/neutral/negative + short descriptor)."
    prompt=f"Digest this into humane memory.\n\nTEXT:\n{text}\n\nReturn JSON: title, summary, tags (3-6), sentiment."
    try:
        out=AI.chat([{"role":"system","content":sysmsg},{"role":"user","content":prompt}], max_tokens=360)
        try: data=json.loads(out)
        except Exception: data={"title":"","summary":out.strip(),"tags":"","sentiment":""}
        return {"title":(data.get("title") or "")[:120],"summary":(data.get("summary") or out).strip(),
                "tags":(data.get("tags") or "").replace("\n"," ").strip(),"sentiment":(data.get("sentiment") or "").strip()}
    except Exception: return {"title":"","summary":text[:900],"tags":"","sentiment":""}

def remember(source:str, text:str, raw_ref:str="")->int:
    digest=_humane_summarize(text, True); topic=_topic_key(digest["summary"] or text)
    cur.execute("SELECT id, content FROM notes WHERE topic_key=? ORDER BY id DESC LIMIT 1",(topic,)); row=cur.fetchone()
    if row:
        nid, existing=row; merged=_merge_contents(existing, digest["summary"])
        cur.execute("""UPDATE notes SET ts=?,source=?,title=?,content=?,tags=?,sentiment=?,raw_ref=? WHERE id=?""",
                    (datetime.utcnow().isoformat(),source,digest["title"],merged,digest["tags"],digest["sentiment"],raw_ref,nid))
        conn.commit(); logging.info("notes merge id=%s topic=%s", nid, topic); return nid
    cur.execute("""INSERT INTO notes(ts,source,topic_key,title,content,tags,sentiment,raw_ref)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (datetime.utcnow().isoformat(),source,topic,digest["title"],digest["summary"],digest["tags"],digest["sentiment"],raw_ref))
    conn.commit(); nid=cur.lastrowid; logging.info("notes add id=%s topic=%s", nid, topic); return nid

def recent_notes(n:int=10)->list[dict]:
    cur.execute("SELECT id,ts,source,title,content,tags,sentiment,raw_ref FROM notes ORDER BY id DESC LIMIT ?",(n,))
    return [{"id":r[0],"ts":r[1],"source":r[2],"title":r[3] or "","content":r[4],
             "tags":r[5] or "","sentiment":r[6] or "","ref":r[7] or ""} for r in cur.fetchall()]

def export_all_notes()->list[dict]:
    cur.execute("SELECT id,ts,source,title,content,tags,sentiment,raw_ref FROM notes ORDER BY id ASC")
    rows=cur.fetchall()
    return [{"id":r[0],"ts":r[1],"source":r[2],"title":r[3] or "","content":r[4],
             "tags":r[5] or "","sentiment":r[6] or "","ref":r[7] or ""} for r in rows]

def recent_raw(n:int=50)->list[dict]:
    cur.execute("SELECT id,ts,type,text,meta FROM raw_events ORDER BY id DESC LIMIT ?",(n,))
    out=[]
    for r in cur.fetchall():
        try: meta=json.loads(r[4] or "{}")
        except: meta={}
        out.append({"id":r[0],"ts":r[1],"type":r[2],"text":r[3],"meta":meta})
    return out

# ---------- Simple lexical search (no extra deps) ----------
_WORD_RE=re.compile(r"[a-z0-9]+")
def _tokenize(s:str)->list[str]: return _WORD_RE.findall(s.lower())
def _tf(tokens:list[str])->dict[str,float]:
    d={}; n=float(len(tokens)) or 1.0
    for t in tokens: d[t]=d.get(t,0)+1.0
    for k in d: d[k]/=n
    return d
def _cosine(a:dict[str,float], b:dict[str,float])->float:
    if not a or not b: return 0.0
    common=set(a)&set(b); num=sum(a[t]*b[t] for t in common)
    da=math.sqrt(sum(v*v for v in a.values())); db=math.sqrt(sum(v*v for v in b.values()))
    return 0.0 if da==0 or db==0 else num/(da*db)

def search_memory(query:str, k_raw:int=12, k_notes:int=12)->dict:
    qtf=_tf(_tokenize(query))
    cur.execute("SELECT id,ts,type,text,meta FROM raw_events ORDER BY id DESC LIMIT 4000")
    raw_scored=[]
    for r in cur.fetchall():
        txt=r[3] or ""; score=_cosine(qtf,_tf(_tokenize(txt)))
        if score>0: raw_scored.append((score,{"id":r[0],"ts":r[1],"type":r[2],"text":txt,"meta":r[4]}))
    raw_top=[x[1] for x in sorted(raw_scored,key=lambda z:z[0],reverse=True)[:k_raw]]

    cur.execute("SELECT id,ts,source,title,content,tags,sentiment,raw_ref FROM notes ORDER BY id DESC LIMIT 4000")
    note_scored=[]
    for r in cur.fetchall():
        txt=(r[4] or "")+" "+(r[3] or "")+" "+(r[5] or "")
        score=_cosine(qtf,_tf(_tokenize(txt)))
        if score>0:
            note_scored.append((score,{"id":r[0],"ts":r[1],"source":r[2],"title":r[3] or "","content":r[4] or "",
                                       "tags":r[5] or "","sentiment":r[6] or "","ref":r[7] or ""}))
    notes_top=[x[1] for x in sorted(note_scored,key=lambda z:z[0],reverse=True)[:k_notes]]
    return {"raw":raw_top,"notes":notes_top}

# ---------- RAG-style answer ----------
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
    found=search_memory(question,k_raw=10,k_notes=10); ctx=build_context_blurb(found)
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
                if r.status!=200: return f"⚠️ HTTP {r.status}"
                text=await r.text()
        soup=BeautifulSoup(text,"html.parser")
        title=soup.title.string.strip() if soup.title and soup.title.string else "No title"
        desc=(soup.find("meta",{"name":"description"}) or {}).get("content","")
        h1=soup.h1.get_text(strip=True) if soup.h1 else ""
        words=len(soup.get_text(' ').split()); links=len(soup.find_all("a")); images=len(soup.find_all("img"))
        snippet=soup.get_text(" ")[:1800]
        return f"🌐 {title}\nDesc: {desc[:200]}\nH1: {h1}\nWords:{words} Links:{links} Images:{images}\n\nSnippet:\n{snippet}"
    except Exception as e:
        logging.exception("Crawl error"); return f"⚠️ Crawl error: {e}"

async def analyze_url(url:str)->str:
    content=await fetch_url(url)
    if content.startswith("⚠️"): return content
    summary=await ask_ai("Summarize page in short bullets; key entities, actions, SEO opportunities:\n\n"+content)
    log_raw("link", summary, {"url":url}); remember("link", summary, raw_ref=url); return summary

# ---------- File analyzers ----------
def analyze_excel(path:Path)->str:
    try:
        df=pd.read_excel(path); head=", ".join(map(str,list(df.columns)[:12]))
        info=f"✅ Excel loaded: {df.shape[0]} rows × {df.shape[1]} cols\nColumns: {head}"
        num=df.select_dtypes(include="number")
        if not num.empty: info+="\n\nNumeric summary:\n"+num.describe().to_string()[:1800]
        log_raw("file", f"Excel {path.name}: shape {df.shape}; columns {list(df.columns)!r}", {"file":str(path)})
        remember("file", f"Excel {path.name}: shape {df.shape}; columns {list(df.columns)!r}", raw_ref=str(path))
        return info
    except Exception as e: logging.exception("Excel analysis error"); return f"⚠️ Excel analysis error: {e}"

def analyze_csv(path:Path)->str:
    try:
        df=pd.read_csv(path, nrows=50000); head=", ".join(map(str,list(df.columns)[:12]))
        info=f"✅ CSV loaded: {df.shape[0]} rows × {df.shape[1]} cols\nColumns: {head}"
        num=df.select_dtypes(include="number")
        if not num.empty: info+="\n\nNumeric summary:\n"+num.describe().to_string()[:1800]
        log_raw("file", f"CSV {path.name}: shape {df.shape}; columns {list(df.columns)!r}", {"file":str(path)})
        remember("file", f"CSV {path.name}: shape {df.shape}; columns {list(df.columns)!r}", raw_ref=str(path))
        return info
    except Exception as e: logging.exception("CSV analysis error"); return f"⚠️ CSV analysis error: {e}"

def analyze_json(path:Path)->str:
    try:
        raw=path.read_text(encoding="utf-8", errors="ignore"); data=json.loads(raw)
        if isinstance(data,list): shape=f"list[{len(data)}]"; preview=json.dumps(data[:3], ensure_ascii=False)[:1500]
        elif isinstance(data,dict): shape=f"dict({len(data.keys())} keys)"; preview=json.dumps({k:data[k] for k in list(data.keys())[:10]}, ensure_ascii=False)[:1500]
        else: shape=type(data).__name__; preview=str(data)[:1500]
        info=f"✅ JSON loaded: {shape}\nPreview: {preview}"
        log_raw("file", f"JSON {path.name}: shape {shape}", {"file":str(path)})
        remember("file", f"JSON {path.name}: shape {shape}\nPreview: {preview}", raw_ref=str(path))
        return info
    except Exception as e: logging.exception("JSON analysis error"); return f"⚠️ JSON analysis error: {e}"

# ---------- Image analyzer (EXIF + optional OCR) ----------
def _exif_dict(im:Image.Image)->dict:
    try:
        exif=im._getexif() or {}; label={ExifTags.TAGS.get(k,k):v for k,v in exif.items()}
        keep={}
        for k in ["DateTime","Make","Model","Software","LensModel","Orientation","ExifVersion","XResolution","YResolution"]:
            if k in label: keep[k]=label[k]
        return keep
    except Exception: return {}

def analyze_image(path:Path)->str:
    meta={"file":str(path)}
    try:
        im=Image.open(path); exif=_exif_dict(im); meta["exif"]=exif
        if HAS_TESS:
            try:
                import pytesseract; ocr=pytesseract.image_to_string(im) or ""
            except Exception as e: ocr=f"(OCR unavailable: {e})"
        else: ocr="(OCR module not installed)"
        rid=log_raw("image", ocr if ocr else "(no OCR text)", meta)
        gist=f"Image {path.name}: EXIF {exif if exif else '∅'}; OCR preview: {(ocr[:500]+'…') if ocr and len(ocr)>500 else (ocr or '∅')}"
        remember("image", gist, raw_ref=str(path))
        return f"🖼️ Image saved. EXIF keys: {list(exif.keys()) if exif else 'none'}. OCR length: {len(ocr)} chars. (raw id {rid})"
    except Exception as e:
        log_raw("image", f"Image load error {path.name}: {e}", {"file":str(path)}); logging.exception("Image analysis error")
        return f"⚠️ Image analysis error: {e}"

# ---------- Trading log parsing ----------
TRADE_PATTERNS=[re.compile(r"\b(BUY|SELL)\b.*?(\b[A-Z]{2,10}\b).*?qty[:= ]?(\d+).*?price[:= ]?([0-9.]+)", re.I),
                re.compile(r"order\s+(buy|sell)\s+(\w+).+?@([0-9.]+).+?qty[:= ]?(\d+)", re.I)]
def summarize_trade_line(line:str)->str|None:
    for pat in TRADE_PATTERNS:
        m=pat.search(line)
        if m:
            side,sym,qty,price=list(m.groups())
            try: qty_i=int(qty)
            except: qty_i=qty
            return f"🟢 {side.upper()} {sym} qty {qty_i} @ {price}"
    return None

# ---------- Telegram handlers ----------
GLOBAL_APP:Application|None=None
MEM_RUNTIME={"log_path":"", "subscribers": []}

async def start_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hey, I'm Alex 🤖\n"
        "I keep a perfect raw log and recall with concise insights.\n\n"
        "Core: /ask <question>\n"
        "Ingest: /analyze <url>, send images & .xlsx/.csv/.json files\n"
        "Memory: /remember <text>, /mem [n], /exportmem, /raw [n]\n"
        "Logs: /setlog <path>, /subscribe_logs, /unsubscribe_logs, /logs [n]\n"
        "Utils: /ai <prompt>, /search <query>, /id, /uptime, /backup, /config"
    )

async def id_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Your chat id: `{update.effective_chat.id}`", parse_mode="Markdown")

async def uptime_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    u=int(time.time()-START_TIME); h,m,s=u//3600,(u%3600)//60,u%60
    await update.message.reply_text(f"⏱️ Uptime {h}h {m}m {s}s")

async def config_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    flags={
        "HUMANE_TONE":HUMANE_TONE,
        "HAS_TESS":HAS_TESS,
        "SERPAPI_KEY_set":bool(SERPAPI_KEY),
        "DB_PATH":DB_PATH,"SAVE_DIR":str(SAVE_DIR),"PORT":PORT,
        "Providers":{
            "ollama":bool(OLLAMA_BASE and OLLAMA_MODEL),
            "openrouter":bool(OPENROUTER_KEY),
            "groq":bool(GROQ_KEY),
            "together":bool(TOGETHER_KEY),
            "openai":bool(OPENAI_KEY),
        }
    }
    await update.message.reply_text("Config:\n"+json.dumps(flags, indent=2))

async def backup_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    try:
        zpath=Path("alex_backup.zip")
        with zipfile.ZipFile(zpath,"w",compression=zipfile.ZIP_DEFLATED) as z:
            if Path(DB_PATH).exists(): z.write(DB_PATH)
            if RAW_JSONL.exists(): z.write(RAW_JSONL)
            manifest={"files":[str(p) for p in SAVE_DIR.glob("*") if p.is_file()]}
            man_path=Path("received_manifest.json"); man_path.write_text(json.dumps(manifest, indent=2)); z.write(man_path)
        await update.message.reply_document(document=str(zpath), filename=zpath.name)
    except Exception as e: logging.exception("Backup error"); await update.message.reply_text(f"Backup error: {e}")

async def ai_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    q=" ".join(ctx.args)
    if not q: return await update.message.reply_text("Usage: /ai <your prompt>")
    ans=await ask_ai(q)
    log_raw("chat_user", q, {"chat_id":update.effective_chat.id}); log_raw("chat_alex", ans, {"chat_id":update.effective_chat.id})
    remember("chat", f"Q: {q}\nA gist: {ans[:400]}"); await update.message.reply_text(ans)

async def ask_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    q=" ".join(ctx.args).strip()
    if not q: return await update.message.reply_text("Usage: /ask <question about anything I've seen/learned>")
    log_raw("chat_user", f"/ask {q}", {"chat_id":update.effective_chat.id})
    ans=await rag_answer(q); log_raw("chat_alex", ans, {"chat_id":update.effective_chat.id}); await update.message.reply_text(ans)

async def search_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not ctx.args: return await update.message.reply_text("Usage: /search <query>")
    if not SERPAPI_KEY: return await update.message.reply_text("⚠️ SERPAPI_KEY not set.")
    query=" ".join(ctx.args)
    try:
        r=requests.get("https://serpapi.com/search", params={"q":query,"hl":"en","api_key":SERPAPI_KEY}, timeout=25)
        j=r.json(); snip=j.get("organic_results",[{}])[0].get("snippet","(no results)")
        out=f"🔎 {query}\n{snip}"
        log_raw("chat_user", f"/search {query}", {"chat_id":update.effective_chat.id})
        log_raw("chat_alex", out, {"chat_id":update.effective_chat.id}); remember("chat", f"SERP for '{query}': {snip}")
        await update.message.reply_text(out)
    except Exception as e: logging.exception("Search error"); await update.message.reply_text(f"Search error: {e}")

async def analyze_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not ctx.args: return await update.message.reply_text("Usage: /analyze <url>")
    url=ctx.args[0]; await update.message.reply_text("🔍 Crawling and summarizing…")
    res=await analyze_url(url); await update.message.reply_text(res)

async def remember_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    text=" ".join(ctx.args).strip()
    if not text: return await update.message.reply_text("Usage: /remember <text to add to memory>")
    rid=log_raw("chat_user", f"/remember {text}", {"chat_id":update.effective_chat.id})
    nid=remember("chat", text); await update.message.reply_text(f"🧠 Noted (note id {nid}, raw id {rid}). Essence kept, details in raw log.")

async def mem_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    n=8
    if ctx.args:
        try: n=max(1,min(40,int(ctx.args[0])))
        except: pass
    items=recent_notes(n)
    if not items: return await update.message.reply_text("🧠 Memory is empty (for now).")
    lines=[]
    for it in items:
        lines.append(f"#{it['id']} [{it['source']}] {it['title'] or '(no title)'}")
        lines.append("  "+it["content"].replace("\n","\n  ")[:700])
        if it["tags"]: lines.append(f"  tags: {it['tags']}")
        if it["ref"]:  lines.append(f"  ref: {it['ref']}")
        lines.append("")
    await update.message.reply_text("\n".join(lines)[:3900])

async def exportmem_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    data=export_all_notes(); path=Path("memory_export.json"); path.write_text(json.dumps(data, indent=2))
    await update.message.reply_document(document=str(path), filename="alex_memory.json")

async def raw_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    n=30
    if ctx.args:
        try: n=max(1,min(200,int(ctx.args[0])))
        except: pass
    rows=recent_raw(n)
    if not rows: return await update.message.reply_text("Raw memory is empty.")
    out=[]
    for r in rows:
        meta=r.get("meta",{}); mtxt=f" | meta: {meta}" if meta else ""; txt=r["text"].replace("\n"," ")[:800]
        out.append(f"#{r['id']} [{r['ts']} | {r['type']}] {txt}{mtxt}")
    await update.message.reply_text("\n".join(out)[:3900])

async def logs_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    n=40
    if ctx.args:
        try: n=max(1,min(400,int(ctx.args[0])))
        except: pass
    path=MEM_RUNTIME.get("log_path") or ""; p=Path(path)
    if not path or not p.exists(): return await update.message.reply_text("⚠️ No log path set or file missing. Use /setlog <path>.")
    try:
        lines=p.read_text(errors="ignore").splitlines()[-n:]; msg="```\n"+"\n".join(lines)[-3500:]+"\n```"
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e: logging.exception("logs read error"); await update.message.reply_text(f"Read error: {e}")

async def setlog_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not ctx.args: return await update.message.reply_text("Usage: /setlog /path/to/your.log")
    path=" ".join(ctx.args); MEM_RUNTIME["log_path"]=path
    await update.message.reply_text(f"✅ Log path set to: `{path}`", parse_mode="Markdown")

async def subscribe_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    cid=update.effective_chat.id
    if cid not in MEM_RUNTIME["subscribers"]: MEM_RUNTIME["subscribers"].append(cid)
    await update.message.reply_text("🔔 Subscribed to live log updates.")

async def unsubscribe_cmd(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    cid=update.effective_chat.id
    if cid in MEM_RUNTIME["subscribers"]: MEM_RUNTIME["subscribers"].remove(cid)
    await update.message.reply_text("🔕 Unsubscribed.")

# --- text messages (general) ---
async def handle_text(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    text=(update.message.text or "").strip()
    log_raw("chat_user", text, {"chat_id":update.effective_chat.id})
    if text.lower()=="analyse alex_profile":
        msg="Back in the zone. What’s next?"; log_raw("chat_alex", msg, {"chat_id":update.effective_chat.id})
        return await update.message.reply_text(msg)
    if text.startswith(("http://","https://")):
        await update.message.reply_text("🔍 Got your link — analyzing…"); res=await analyze_url(text)
        log_raw("chat_alex", res, {"chat_id":update.effective_chat.id}); return await update.message.reply_text(res)
    ans=await ask_ai(text); remember("chat", f"User said: {text}\nResponse gist: {ans[:400]}")
    log_raw("chat_alex", ans, {"chat_id":update.effective_chat.id}); await update.message.reply_text(ans)

# --- document uploads ---
async def handle_file(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    doc=update.message.document
    if not doc: return
    file_path=SAVE_DIR/doc.file_name; tg_file=await ctx.bot.get_file(doc.file_id); await tg_file.download_to_drive(file_path)
    log_raw("file", f"Received file {doc.file_name}", {"file":str(file_path), "chat_id":update.effective_chat.id})
    await update.message.reply_text(f"📂 Saved `{doc.file_name}` — analyzing…", parse_mode="Markdown")
    name=doc.file_name.lower()
    if   name.endswith(".xlsx"): out=analyze_excel(file_path)
    elif name.endswith(".csv"):  out=analyze_csv(file_path)
    elif name.endswith(".json"): out=analyze_json(file_path)
    else:
        remember("file", f"Received file {doc.file_name}", raw_ref=str(file_path)); out="Saved. I currently analyze Excel (.xlsx), CSV, and JSON."
    log_raw("chat_alex", out, {"chat_id":update.effective_chat.id}); await update.message.reply_text(out)

# --- photo uploads ---
async def handle_photo(update:Update, ctx:ContextTypes.DEFAULT_TYPE):
    if not update.message.photo: return
    photo=update.message.photo[-1]; file=await ctx.bot.get_file(photo.file_id)
    path=SAVE_DIR/f"photo_{photo.file_id}.jpg"; await file.download_to_drive(path)
    out=analyze_image(path); log_raw("chat_alex", out, {"chat_id":update.effective_chat.id}); await update.message.reply_text(out)

# ---------- live log watcher (push to subscribers) ----------
def _post_to_subscribers(text:str):
    if not GLOBAL_APP or not MEM_RUNTIME.get("subscribers"): return
    loop=GLOBAL_APP.bot._application.loop
    async def _send():
        for cid in list(MEM_RUNTIME["subscribers"]):
            try: await GLOBAL_APP.bot.send_message(chat_id=cid, text=text)
            except Exception: pass
    try: asyncio.run_coroutine_threadsafe(_send(), loop)
    except Exception: pass

def watch_logs():
    last_size=0; last_raw_push=0
    while True:
        try:
            path=MEM_RUNTIME.get("log_path") or ""
            if not path or not Path(path).exists(): time.sleep(2); continue
            p=Path(path); sz=p.stat().st_size
            if sz<last_size: last_size=0  # rotation
            if sz>last_size:
                with p.open("r", errors="ignore") as f:
                    if last_size: f.seek(last_size)
                    new=f.read(); last_size=sz
                for line in new.splitlines():
                    s=summarize_trade_line(line)
                    if s:
                        _post_to_subscribers(s); log_raw("log", line, {"path":path}); remember("log", s, raw_ref=path)
                    now=time.time()
                    if now-last_raw_push>15:
                        last_raw_push=now; _post_to_subscribers("📜 "+line[:900])
        except Exception as e:
            logging.error("log watcher error: %s", e)
        time.sleep(1)

# ---------- persona synthesis worker ----------
def learning_worker():
    while True:
        try:
            cur.execute("SELECT content FROM notes ORDER BY id DESC LIMIT 24"); rec=[r[0] for r in cur.fetchall()]
            if rec:
                persona=AI.chat(
                    [{"role":"system","content":"Create concise persona guidance from these snippets. 4–6 bullets."},
                     {"role":"user","content":"\n\n".join(rec)}],
                    max_tokens=220
                )
                tk="__persona__"; cur.execute("SELECT id FROM notes WHERE topic_key=? ORDER BY id DESC LIMIT 1",(tk,)); row=cur.fetchone()
                if row:
                    cur.execute("UPDATE notes SET ts=?,source=?,title=?,content=?,tags=?,sentiment=? WHERE id=?",
                                (datetime.utcnow().isoformat(),"system","Persona",persona,"persona,profile","",row[0]))
                else:
                    cur.execute("""INSERT INTO notes(ts,source,topic_key,title,content,tags,sentiment,raw_ref)
                                   VALUES(?,?,?,?,?,?,?,?)""",
                                (datetime.utcnow().isoformat(),"system",tk,"Persona",persona,"persona,profile","",""))
                conn.commit()
        except Exception as e: logging.error("learning error: %s", e)
        time.sleep(60)

# ---------- health server ----------
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type","text/plain"); self.end_headers(); self.wfile.write(b"ok")
def start_health(): HTTPServer(("0.0.0.0", PORT), Health).serve_forever()

# ---------- main ----------
def main():
    global GLOBAL_APP
    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN missing — exiting."); sys.exit(1)

    app=Application.builder().token(TELEGRAM_TOKEN).build(); GLOBAL_APP=app
    # commands
    app.add_handler(CommandHandler("start", start_cmd)); app.add_handler(CommandHandler("id", id_cmd))
    app.add_handler(CommandHandler("uptime", uptime_cmd)); app.add_handler(CommandHandler("config", config_cmd))
    app.add_handler(CommandHandler("backup", backup_cmd)); app.add_handler(CommandHandler("ai", ai_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd)); app.add_handler(CommandHandler("search", search_cmd))
    app.add_handler(CommandHandler("analyze", analyze_cmd)); app.add_handler(CommandHandler("remember", remember_cmd))
    app.add_handler(CommandHandler("mem", mem_cmd)); app.add_handler(CommandHandler("exportmem", exportmem_cmd))
    app.add_handler(CommandHandler("raw", raw_cmd)); app.add_handler(CommandHandler("setlog", setlog_cmd))
    app.add_handler(CommandHandler("subscribe_logs", subscribe_cmd)); app.add_handler(CommandHandler("unsubscribe_logs", unsubscribe_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))
    # messages
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    # background threads
    threading.Thread(target=start_health, daemon=True).start()
    threading.Thread(target=learning_worker, daemon=True).start()
    threading.Thread(target=watch_logs, daemon=True).start()
    logging.info("Alex mega bot starting…"); app.run_polling(close_loop=False)

if __name__=="__main__": main()