"""
Market Intelligence & Corporate Announcement Screening Engine (GitHub Actions Edition)
=======================================================================================
Architecture & New Features:
- Dynamic Payload Router: Intelligently switches between pypdf (10k chars) and pdfplumber (50k chars) based on headline context.
- Full-PDF Passthrough: Qwen 2.5 7B acts as a fast gatekeeper (4.5k chars), while Sieve 2 gets the full 50k-char raw text.
- Peak Deluge Throttling: Thread-safe token bucket pacing prevents API 429 errors during earnings season.
- Direct Alpha Quick-Links: Embedded Screener & TradingView URLs.
- Type-Safe Quant Ledger: NumPy bool conversion prevents Supabase crash.

Note: Ensure `pdfplumber` is added to your requirements.txt
"""

import os
import io
import re
import sys
import json
import time
import queue
import logging
import itertools
import threading
import argparse
import requests
import feedparser
import psycopg2
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from anthropic import Anthropic
from youtube_transcript_api import YouTubeTranscriptApi
from pypdf import PdfReader
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# 0. LOGGING & SDK SUPPRESSION
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# 1. CONFIGURATION & ENVIRONMENT SETUP
# ---------------------------------------------------------------------------
load_dotenv()

BSE_API_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?pageno={page}&strCat=-1&strPrevDate=&strScrip=&strSearch=P&strToDate=&strType=C"
BSE_PDF_BASE_URL = "https://www.bseindia.com/xml-data/corpfiling/AttachLive/{attachment}"
NSE_BASE_URL = "https://www.nseindia.com"
NSE_API_URL = "https://www.nseindia.com/api/corporate-announcements?index=equities"
VALUEPICKR_API_URL = "https://forum.valuepickr.com/search/query.json?term={term}"
YOUTUBE_RSS_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*"
}
BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "origin": "https://www.bseindia.com",
    "referer": "https://www.bseindia.com/"
}
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com/"
}

DEFAULT_CHUNK_SIZE = 50
DEFAULT_YOUTUBE_CHANNEL_ID = "UCb5hMTAFjG5j79V6nL3_YCQ"

# Verbosity & Sieve Logic
PUBLISH_WHEN_SIEVE_PASSED = float(os.getenv("PUBLISH_WHEN_SIEVE_PASSED", "1.5"))
DEFAULT_MAX_SIEVE2 = int(os.getenv("MAX_SIEVE2_ITEMS", "0"))
SIEVE_1_5_MIN_SCORE = int(os.getenv("SIEVE_1_5_MIN_SCORE", "5"))
ALERT_ON_SINGLE_MODEL_IGNORE = os.getenv("ALERT_ON_SINGLE_MODEL_IGNORE", "false").lower() in ("true", "1", "yes")

parser = argparse.ArgumentParser(description="Market Intelligence Screening Engine (GitHub Actions Edition)")
parser.add_argument("--ignore-cache", "-f", action="store_true", help="Bypass Supabase cache and re-evaluate all filings")
parser.add_argument("--max-pages", type=int, default=100, help="Maximum BSE announcement pages to fetch")
parser.add_argument("--max-sieve2", type=int, default=DEFAULT_MAX_SIEVE2, help="Max candidates sent to Sieve 2 (0 for all)")
args, _ = parser.parse_known_args()

IGNORE_CACHE = args.ignore_cache or os.getenv("IGNORE_CACHE", "false").lower() in ("true", "1", "yes")

USE_LOCAL_EXTRACTOR = os.getenv("USE_LOCAL_EXTRACTOR", "true").lower() in ("true", "1", "yes")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", DEFAULT_OLLAMA_URL)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_PAID") or os.getenv("GEMINI_API_KEY_FREE")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SUPABASE_URL = os.getenv("SUPABASE_URL")

TIER1_MODEL = os.getenv("GEMINI_TIER1_MODEL", "gemini-3.5-flash-lite")
TIER2_GEMINI_MODEL = os.getenv("GEMINI_TIER2_MODEL", "gemini-3.1-pro-preview")
TIER2_CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
claude_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Throttling Locks
api_lock = threading.Lock()
last_api_call = 0.0
MIN_API_DELAY = 1.5  # Prevents 429 Too Many Requests across threaded workers

def throttle_api():
    """Implements thread-safe pacing for peak earnings deluge spikes."""
    global last_api_call
    with api_lock:
        now = time.time()
        elapsed = now - last_api_call
        if elapsed < MIN_API_DELAY:
            time.sleep(MIN_API_DELAY - elapsed)
        last_api_call = time.time()


# ---------------------------------------------------------------------------
# 2. SUPABASE DATABASE LAYER
# ---------------------------------------------------------------------------
def get_db_connection():
    if not SUPABASE_URL: return None
    try: return psycopg2.connect(SUPABASE_URL)
    except Exception as e:
        print(f"[Database Error] Connection failed: {e}")
        return None

def filter_unprocessed_announcements(filings):
    if IGNORE_CACHE: return filings
    conn = get_db_connection()
    if not conn: return filings
    try:
        cursor = conn.cursor()
        attachments = [item['id'] for item in filings if item.get('id')]
        if not attachments:
            conn.close(); return []
        query = f"SELECT bse_attachment_name FROM bse_announcements WHERE bse_attachment_name IN ({','.join(['%s'] * len(attachments))})"
        cursor.execute(query, tuple(attachments))
        existing = {row[0] for row in cursor.fetchall()}
        conn.close()
        return [item for item in filings if item['id'] not in existing]
    except Exception as e:
        print(f"[Database Error] Deduplication query failed: {e}")
        return filings

def group_filings_by_company(filings):
    grouped = {}
    for item in filings:
        key = (item['exchange'], item['scrip'])
        if key not in grouped:
            grouped[key] = {
                'id': item['id'], 'all_ids': [item['id']], 'company': item['company'],
                'scrip': item['scrip'], 'headline': item['headline'], 'isin': item['isin'],
                'all_links': [item['link']], 'exchange': item['exchange']
            }
        else:
            existing = grouped[key]
            if item['id'] not in existing['all_ids']: existing['all_ids'].append(item['id'])
            if item['headline'] not in existing['headline']: existing['headline'] += f" | {item['headline']}"
            if item['link'] not in existing['all_links']: existing['all_links'].append(item['link'])
    return list(grouped.values())

def log_announcements_batch(decisions_list):
    if not decisions_list: return
    conn = get_db_connection()
    if not conn: return
    try:
        cursor = conn.cursor()
        query = """INSERT INTO bse_announcements (bse_attachment_name, company_name, headline, ai_decision) 
                   VALUES (%s, %s, %s, %s) ON CONFLICT (bse_attachment_name) DO UPDATE SET ai_decision = EXCLUDED.ai_decision"""
        cursor.executemany(query, decisions_list)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Database Error] Batch cache insertion failed: {e}")

def log_permanent_ledger(item, market_data, evals, audit, extracted_text):
    conn = get_db_connection()
    if not conn: return
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO ai_recommendation_ledger (
                bse_attachment_name, company_name, scrip_code, exchange, isin, headline, catalyst_category,
                alert_price, market_cap_cr, vol_multiple_20d, above_50dma, above_200dma,
                final_score, consensus_status, high_conviction,
                claude_catalyst_score, gemini_catalyst_score, claude_company_score, gemini_company_score,
                llama_extracted_facts, claude_analysis, gemini_analysis
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (bse_attachment_name) DO UPDATE SET
                final_score = EXCLUDED.final_score, consensus_status = EXCLUDED.consensus_status,
                claude_analysis = EXCLUDED.claude_analysis, gemini_analysis = EXCLUDED.gemini_analysis;
        """
        cursor.execute(query, (
            item['id'], item['company'], item['scrip'], item['exchange'], item.get('isin', 'N/A'),
            item.get('headline', ''), item.get('catalyst_reason', 'Actionable corporate action'),
            market_data.get('price', 0.0), market_data.get('market_cap_cr', 0.0), market_data.get('vol_multiple', 1.0),
            market_data.get('above_50dma', False), market_data.get('above_200dma', False),
            audit.get('final_score', 1), audit.get('consensus_status', 'NEUTRAL_MIX'), audit.get('high_conviction', False),
            audit.get('claude_catalyst_score'), audit.get('gemini_catalyst_score'),
            audit.get('claude_company_score'), audit.get('gemini_company_score'),
            extracted_text, evals.get('claude_analysis', ''), evals.get('gemini_analysis', '')
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Database Error] Comprehensive ledger logging failed: {e}")


# ---------------------------------------------------------------------------
# 3. QUANT METRICS & SCUTTLEBUTT CONTEXT
# ---------------------------------------------------------------------------
def fetch_market_metrics(scrip_code, exchange):
    default_payload = {
        "price": 0.0, "vol_multiple": 1.0, "above_50dma": False, "above_200dma": False,
        "dist_52w_high": 0.0, "price_feed_sync": True, "market_cap_cr": 0.0
    }
    if not scrip_code: return default_payload
    try:
        ticker = f"{scrip_code}.NS" if exchange == "NSE" else f"{scrip_code}.BO"
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < 20: return default_payload

        current_price = round(float(hist['Close'].iloc[-1]), 2)
        avg_20_volume = float(hist['Volume'].iloc[-21:-1].mean()) if len(hist) >= 21 else float(hist['Volume'].iloc[-1])
        vol_multiple = round((float(hist['Volume'].iloc[-1]) / avg_20_volume), 2) if avg_20_volume > 0 else 1.0

        dma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else current_price
        dma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else current_price
        high_52w = float(hist['High'].max())
        dist_52w_high = round(((current_price - high_52w) / high_52w) * 100, 1) if high_52w > 0 else 0.0

        market_cap_cr = 0.0
        try:
            mkt_cap = getattr(stock.fast_info, 'market_cap', 0)
            if mkt_cap: market_cap_cr = round(mkt_cap / 1e7, 2)
        except Exception: pass

        return {
            "price": current_price, "vol_multiple": vol_multiple,
            "above_50dma": bool(current_price > dma_50), "above_200dma": bool(current_price > dma_200),
            "dist_52w_high": dist_52w_high, "price_feed_sync": bool(current_price == 0.0), "market_cap_cr": market_cap_cr
        }
    except Exception as e:
        print(f"[Market Data Notice] Scrip {scrip_code} ({exchange}): {e}")
        return default_payload

def fetch_valuepickr_sentiment(company_name):
    try:
        clean_name = company_name.split()[0].replace("Ltd", "").strip()
        res = requests.get(VALUEPICKR_API_URL.format(term=clean_name), headers=DEFAULT_HEADERS, timeout=5)
        if res.status_code != 200: return "No active forum discussion found."
        posts = res.json().get('posts', [])
        return " ".join([p.get('blurb', '') for p in posts[:5]])[:1500] if posts else "No active forum discussion found."
    except Exception: return "Forum search bypassed."


# ---------------------------------------------------------------------------
# 4. DYNAMIC PAYLOAD ROUTER (PDF EXTRACTION)
# ---------------------------------------------------------------------------
def sanitize_filing_text(text):
    if not text: return ""
    match = re.search(r'(?i)(?:Sub(?:ject)?\s*:|Ref\s*:)', text)
    if match: text = text[match.start():]
    text = re.sub(r'(?i)(?:\bDisclaimer\b|CAREEDGE RATINGS DISCLAIMS|S&P Global Ratings Terms and Conditions).*$', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F₹]+', '', text)
    return text.strip()

def extract_text_from_pdf_url(pdf_url, headline):
    """
    Dynamic Router:
    - Standard Catalysts: pypdf (4 pages, 10k chars) -> Lightning fast.
    - Financial Tables: pdfplumber (15 pages, 50k chars) -> High fidelity tabular rendering.
    """
    if not pdf_url or not pdf_url.startswith("http"):
        return "No valid PDF URL provided."

    headline_lower = headline.lower()
    financial_keywords = ["financial result", "outcome of board meeting", "earnings", "annual report", "financial statement"]
    is_heavy_financial = any(kw in headline_lower for kw in financial_keywords)

    max_pages = 15 if is_heavy_financial else 4
    max_chars = 50000 if is_heavy_financial else 10000

    try:
        response = requests.get(pdf_url, headers=DEFAULT_HEADERS, timeout=15)
        if response.status_code == 200:
            with io.BytesIO(response.content) as pdf_buffer:
                if is_heavy_financial and pdfplumber:
                    extracted_pages = []
                    with pdfplumber.open(pdf_buffer) as pdf:
                        for i, page in enumerate(pdf.pages):
                            if i >= max_pages: break
                            # layout=True preserves spatial tabular alignment crucial for financial P&L
                            text = page.extract_text(layout=True) or page.extract_text()
                            if text: extracted_pages.append(text)
                    raw_text = "\n".join(extracted_pages)
                else:
                    reader = PdfReader(pdf_buffer)
                    raw_text = "\n".join([page.extract_text() for page in reader.pages[:max_pages] if page.extract_text()])

                clean_text = sanitize_filing_text(raw_text)
                return clean_text[:max_chars] if clean_text else "PDF contains scanned imagery or unextractable text."
        return f"Failed to download PDF (HTTP {response.status_code})"
    except Exception as e:
        return f"PDF extraction error: {e}"

def extract_score(text, label):
    if not text: return None
    for line in text.splitlines():
        if label.lower() in line.lower():
            match = re.search(r'(\b10|[1-9])\s*/\s*10', line)
            if match: return int(match.group(1))
            digits = re.findall(r'\b(10|[1-9])\b', line)
            if digits: return int(digits[0])
    return None


# ---------------------------------------------------------------------------
# 5. SIEVE 1 (FLASH LITE) & SIEVE 1.5 (LOCAL QWEN 2.5)
# ---------------------------------------------------------------------------
def run_tier1_batch_sieve(announcements):
    if not announcements or not gemini_client:
        return announcements, []

    hits, rejections = [], []
    for i in range(0, len(announcements), DEFAULT_CHUNK_SIZE):
        chunk = announcements[i:i + DEFAULT_CHUNK_SIZE]
        items_payload = [{"index": idx, "exchange": a['exchange'], "scrip": a['scrip'], "company": a['company'], "headline": a['headline']} for idx, a in enumerate(chunk)]

        prompt = f"""
        You are an objective exchange filing intake filter. Categorize EACH announcement as either "HIT" or "REJECT".

        HIGH-MATERIALITY CATALYSTS (FLAG AS "HIT"):
        1. Financial Results & Guidance (Quarterly/annual statements, revenue/margin guidance updates).
        2. Commercial Order Wins & Contract Awards (Defense, Railways, OEM agreements, LoA).
        3. Fundraisings & M&A (Preferential allotments, strategic warrants, QIP, acquisitions, demergers, open offers).
        4. Credit Rating Actions (Upgrades or major revisions).
        5. Capex & Commercial Production (Commissioning of new plants, capacity expansions).
        6. Balance Sheet Deleveraging (Prepayments, net debt-free milestones, one-time settlements).
        7. Regulatory Clearances (USFDA EIR/approvals, CDSCO, PLI scheme subsidies, patent grants).

        CRITICAL EXCLUSIONS (FLAG AS "REJECT"):
        - Loss of share certificates / duplicate requests (Reg 39(3)).
        - Trading window closure notices for board meetings / financial results.
        - Advance intimations of Board Meeting dates (prior notices).
        - Routine shareholding patterns (Reg 31), Corporate Governance (Reg 27(2)), Secretarial Compliance (Reg 24A).
        - Newspaper publication clippings, routine ESOP allotments.

        Respond with ONLY a valid JSON array of objects in this exact structure:
        [
          {{"index": 0, "status": "HIT", "reason": "Commercial contract win of INR 250 Cr"}},
          {{"index": 1, "status": "REJECT", "reason": "Trading window closure notice"}}
        ]

        Announcements to screen:
        {json.dumps(items_payload, indent=2)}
        """
        for attempt in range(3):
            try:
                response = gemini_client.models.generate_content(model=TIER1_MODEL, contents=prompt)
                parsed_decisions = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
                for dec in parsed_decisions:
                    idx = dec.get("index")
                    if idx is not None and idx < len(chunk):
                        ann_item = chunk[idx]
                        status, reason = dec.get("status", "REJECT").upper(), dec.get("reason", "Routine filing")
                        if status == "HIT":
                            ann_item["catalyst_reason"] = reason
                            hits.append(ann_item)
                            print(f" [SIEVE 1 HIT] {ann_item['company']} | {reason}")
                        else:
                            ann_item["rejection_reason"] = reason
                            rejections.append(ann_item)
                            print(f" [SIEVE 1 REJECT] {ann_item['company']} | {reason}")
                break
            except Exception as e:
                if "503" in str(e) or "429" in str(e) or "quota" in str(e): time.sleep(2 ** attempt); continue
                print(f"[Tier 1 Error] Chunk evaluation failed: {e}. Defaulting chunk to HIT.")
                hits.extend(chunk); break

    return hits, rejections


def sieve_1_5_local_qwen_extraction(cleaned_pdf_text, headline):
    """Sieve 1.5 strictly reads the first 4,500 chars to save CPU time on the GitHub runner."""
    if not USE_LOCAL_EXTRACTOR or not cleaned_pdf_text or len(cleaned_pdf_text) < 80:
        return "No local extraction.", 5

    prompt = f"""
    You are a strict financial analyst pre-screening corporate disclosures.
    Extract key facts and assign an objective PreScore from 1 to 10 based on concrete economic materiality.

    CRITICAL SCORING RULES:
    - Score 1 to 4: Trade expo, PR marketing, non-binding MoUs, minor updates.
    - Score 5 to 6: Small/routine purchase orders, incremental business progress.
    - Score 7 to 10: Hard confirmed contract wins (>INR 50 Cr+), net-debt reduction, major capex commissioning, or strong financial beats.

    Document Text:
    Headline: {headline}
    Filing Body:
    {cleaned_pdf_text[:4500]}

    Output format:
    Summary: [150-word tight summary of the core trigger event]
    Value: [Exact deal value or financial figure]
    Client: [Entity name or domestic/international status]
    PreScore: [Strictly an integer from 1 to 10, e.g., 7/10]
    """.strip()

    try:
        res = requests.post(OLLAMA_API_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}}, timeout=90)
        if res.status_code == 200:
            extracted_output = res.json().get("response", "").strip()
            if extracted_output:
                pre_score = extract_score(extracted_output, "PreScore:")
                return extracted_output, (pre_score if pre_score is not None else 5)
    except Exception as e:
        print(f" [Sieve 1.5 Warning] Qwen inference bypassed ({e}).")
    return "Local inference failed.", 5


# ---------------------------------------------------------------------------
# 6. TIER 2 (SIEVE 2): RAW PASSTHROUGH TO CLAUDE & GEMINI
# ---------------------------------------------------------------------------
def build_sieve2_prompt(item, market_data, forum_text, raw_pdf_text, qwen_summary):
    price_display = f"INR {market_data['price']}" if market_data['price'] > 0 else "Data Feed Sync"
    mkt_cap_display = f"INR {market_data['market_cap_cr']} Cr" if market_data.get('market_cap_cr', 0) > 0 else "Not specified"

    return f"""You are an institutional equity research analyst evaluating corporate exchange filings.
Assess how strongly this event will impact the company's future earnings power, business trajectory, and institutional market re-rating.

EVALUATION PRINCIPLES:
- Disregard market cap. Evaluate the PROPORTIONAL impact of the event on the company's business scale.
- Base your analysis on both the extracted summary and the FULL raw document below.

COMPANY DETAILS:
Company: {item['company']} ({item['exchange']}: {item['scrip']} | ISIN: {item['isin']})
Price: {price_display} | 20D Vol: {market_data['vol_multiple']}x | 52W High Dist: {market_data['dist_52w_high']}% | MktCap: {mkt_cap_display}
Headline: {item['headline']}
Flagged Catalyst: {item.get('catalyst_reason', 'Actionable corporate action')}
Forum Context: {forum_text}

==================== PRE-SCREENER SUMMARY ====================
{qwen_summary}

==================== FULL RAW REGULATORY FILING ====================
{raw_pdf_text}
======================================================================

OUTPUT FORMAT:
Reasoning: <2-3 sentences on proportional commercial impact, execution capability, and market re-rating probability>
Hype / Red Flag Check: <Clean / Warning flags>
Catalyst Score: <Strictly an integer from 1 to 10, e.g., 8/10>
Company Quality Score: <Strictly an integer from 1 to 10 evaluating core business franchise durability>
"""


def evaluate_with_claude(prompt):
    if not claude_client: return "Claude evaluation skipped: API key missing."
    for attempt in range(3):
        throttle_api()
        try:
            res = claude_client.messages.create(model=TIER2_CLAUDE_MODEL, max_tokens=500, messages=[{"role": "user", "content": prompt}])
            text_blocks = [b.text for b in res.content if getattr(b, "type", None) == "text"]
            return "\n".join(text_blocks).strip() if text_blocks else "Claude returned no text."
        except Exception as e:
            if "503" in str(e) or "429" in str(e) or "overloaded" in str(e): time.sleep(2 ** attempt); continue
            return f"Claude analysis error: {e}"
    return "Claude analysis error: API unavailable after retries."


def evaluate_with_gemini(prompt):
    if not gemini_client: return "Gemini Pro evaluation skipped: API key missing."
    for attempt in range(3):
        throttle_api()
        try:
            res = gemini_client.models.generate_content(model=TIER2_GEMINI_MODEL, contents=prompt)
            return res.text.strip()
        except Exception as e:
            if "503" in str(e) or "429" in str(e) or "quota" in str(e): time.sleep(2 ** attempt); continue
            return f"Gemini Pro analysis error: {e}"
    return "Gemini Pro analysis error: API unavailable after retries."


# ---------------------------------------------------------------------------
# 7. TELEGRAM DISPATCHER (WITH QUICK LINKS)
# ---------------------------------------------------------------------------
def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n[Telegram Output Preview]\n" + message)
        return
    try:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",  # Switched to HTML to bulletproof links and formatting
            "disable_web_page_preview": True
        }
        requests.post(TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN), json=payload, timeout=8)
    except Exception as e:
        print(f"[Telegram Dispatch Error] {e}")


def build_telegram_message(company, exchange, scrip_code, isin, market_data, audit, all_links):
    status = audit.get('consensus_status', 'NEUTRAL_MIX')

    if status == "SIEVE_1_PASS":
        banner = "🔍 <b>INITIAL RADAR: SIEVE 1 (FLASH) PASSED</b>"
    elif status == "SIEVE_1_5_PASS":
        banner = "⚡ <b>EARLY RADAR: SIEVE 1.5 (QWEN) PASSED</b>"
    elif status == "MODEL_DIVERGENCE":
        banner = "⚠️ <b>MODEL DIVERGENCE DETECTED</b>"
    elif status == "SINGLE_MODEL_IGNORE":
        banner = "🚫 <b>FILTERED / LOW CONVICTION (EARLY EXIT)</b>"
    elif audit.get('high_conviction'):
        banner = "🚨 <b>HIGH CONVICTION CATALYST CONCURRENCE</b>"
    else:
        banner = "📢 <b>CORPORATE ACTION RE-RATING CATALYST</b>"

    price_str = f"₹{market_data['price']}" if market_data['price'] > 0 else "₹0.0 (Data Feed Sync)"

    c_cat, g_cat = audit.get('claude_catalyst_score'), audit.get('gemini_catalyst_score')
    cat_scores = []
    if c_cat is not None: cat_scores.append(f"Claude: {c_cat}/10")
    if g_cat is not None: cat_scores.append(f"Gemini: {g_cat}/10")
    fs = audit.get('final_score', 'N/A')
    cat_line = " | ".join(cat_scores) if cat_scores else (
        f"Score: {fs}/10" if isinstance(fs, (int, float)) else "Score: N/A")

    c_comp, g_comp = audit.get('claude_company_score'), audit.get('gemini_company_score')
    comp_scores = []
    if c_comp is not None: comp_scores.append(f"Claude: {c_comp}/10")
    if g_comp is not None: comp_scores.append(f"Gemini: {g_comp}/10")
    comp_line = " | ".join(comp_scores) if comp_scores else "N/A"

    screener_link = f"https://www.screener.in/company/{scrip_code}/"
    tv_symbol = f"NSE:{scrip_code}" if exchange == "NSE" else f"BSE:{scrip_code}"
    tv_link = f"https://in.tradingview.com/chart/?symbol={tv_symbol}"

    msg = (
        f"{banner}\n"
        f"<b>Company:</b> {company}\n"
        f"<b>{exchange}:</b> <code>{scrip_code}</code> | <b>ISIN:</b> <code>{isin}</code>\n"
        f"<b>Price:</b> {price_str} | <b>20D Vol:</b> {market_data['vol_multiple']}x | <b>52W High:</b> {market_data['dist_52w_high']}%\n"
        f"<b>Est. MktCap:</b> ₹{market_data.get('market_cap_cr', 0)} Cr\n"
    )

    if status not in ["SIEVE_1_PASS", "SIEVE_1_5_PASS"]:
        msg += f"<b>Consensus Status:</b> <code>{status}</code>\n\n🎯 <b>Catalyst Score:</b> {cat_line}\n🏢 <b>Company Quality:</b> {comp_line}\n"

    # --- CASCADING REASONING TRAIL ---
    # 1. Sieve 1 Reason
    s1_reason = audit.get('sieve1_reason')
    if s1_reason:
        msg += f"\n🔍 <b>Sieve 1 (Flash Lite) Intake Rationale:</b>\n{s1_reason}\n"

    # 2. Sieve 1.5 Qwen Summary
    s15_summary = audit.get('sieve15_summary')
    if s15_summary and status != "SIEVE_1_PASS":
        msg += f"\n🤖 <b>Sieve 1.5 (Qwen 7B) Extracted Summary & Pre-Score ({audit.get('qwen_pre_score', 'N/A')}/10):</b>\n{s15_summary[:600]}...\n"

    # 3. Sieve 2 Deep Dive Analysis
    if status == "SIEVE_1_PASS":
        msg += f"\n🔍 <i>Catalyst Reason:</i>\n{audit.get('claude_analysis', 'N/A')}\n"
    elif status == "SIEVE_1_5_PASS":
        msg += f"\n🤖 <i>Qwen Extracted Facts:</i>\n{audit.get('claude_analysis', 'N/A')}\n"
    else:
        if audit.get('claude_analysis'):
            msg += f"\n🧠 <b>Claude Analysis:</b>\n{audit['claude_analysis']}\n"
        if audit.get('gemini_analysis'):
            msg += f"\n🤖 <b>Gemini Analysis:</b>\n{audit['gemini_analysis']}\n"

    pdf_links_str = " | ".join([f'<a href="{link}">PDF {i + 1}</a>' for i, link in enumerate(all_links)])
    msg += f"\n🔗 <b>Quick Links:</b> <a href=\"{screener_link}\">Screener.in</a> | <a href=\"{tv_link}\">TradingView</a> | {pdf_links_str}"
    return msg

# ---------------------------------------------------------------------------
# 8. STAGGERED PRIORITY WORKERS & FINALIZATION
# ---------------------------------------------------------------------------
def finalize_dual_evaluation(item, evals, market_data, raw_pdf_text):
    c_cat, g_cat = evals.get('claude_catalyst_score', 1), evals.get('gemini_catalyst_score', 1)
    c_comp, g_comp = evals.get('claude_company_score', 5), evals.get('gemini_company_score', 5)

    is_divergent = (abs(c_cat - g_cat) >= 4 or (c_cat >= 7 and g_cat <= 4) or (g_cat >= 7 and c_cat <= 4))
    if is_divergent: consensus_status = "MODEL_DIVERGENCE"
    elif c_cat >= 7 and g_cat >= 7: consensus_status = "CONSENSUS_HIT"
    elif c_cat <= 4 and g_cat <= 4: consensus_status = "CONSENSUS_IGNORE"
    else: consensus_status = "NEUTRAL_MIX"

    final_score = round((c_cat + g_cat) / 2)
    is_high_conviction = (final_score >= 8 and market_data['vol_multiple'] >= 2.0 and market_data['above_50dma'])

    audit = {
        'consensus_status': consensus_status, 'final_score': final_score, 'high_conviction': is_high_conviction,
        'claude_catalyst_score': c_cat, 'gemini_catalyst_score': g_cat, 'claude_company_score': c_comp,
        'gemini_company_score': g_comp,
        'claude_analysis': evals.get('claude_analysis', ''), 'gemini_analysis': evals.get('gemini_analysis', ''),
        'sieve1_reason': item.get('catalyst_reason'),
        'sieve15_summary': item.get('qwen_summary'),
        'qwen_pre_score': item.get('qwen_pre_score')
    }

    if PUBLISH_WHEN_SIEVE_PASSED <= 2.0 and final_score >= 6:
        send_telegram_alert(build_telegram_message(item['company'], item['exchange'], item['scrip'], item['isin'], market_data, audit, item['all_links']))

    log_permanent_ledger(item, market_data, evals, audit, raw_pdf_text)
    log_announcements_batch([(att_id, item['company'], item['headline'], "HIT") for att_id in item['all_ids']])


def finalize_single_model_ignore(item, evals, market_data, source_model, raw_pdf_text):
    score = evals.get(f'{source_model.lower()}_catalyst_score', 1)
    print(
        f" -> [{source_model.upper()} Gatekeeper] Score {score}/10 < 5. Terminating Sieve 2 early for {item['company']}.")

    audit = {
        'consensus_status': "SINGLE_MODEL_IGNORE", 'final_score': score, 'high_conviction': False,
        'claude_catalyst_score': evals.get('claude_catalyst_score'),
        'gemini_catalyst_score': evals.get('gemini_catalyst_score'),
        'claude_company_score': evals.get('claude_company_score', 5),
        'gemini_company_score': evals.get('gemini_company_score', 5),
        'claude_analysis': evals.get('claude_analysis', ''), 'gemini_analysis': evals.get('gemini_analysis', ''),
        'sieve1_reason': item.get('catalyst_reason'),
        'sieve15_summary': item.get('qwen_summary'),
        'qwen_pre_score': item.get('qwen_pre_score')
    }

    if ALERT_ON_SINGLE_MODEL_IGNORE:
        send_telegram_alert(
            build_telegram_message(item['company'], item['exchange'], item['scrip'], item['isin'], market_data, audit,
                                   item['all_links']))

    log_announcements_batch([(att_id, item['company'], item['headline'], "IGNORE") for att_id in item['all_ids']])

def run_staggered_sieve2_workers(hits, num_workers=4):
    if not hits: return
    counter = itertools.count()
    claude_queue, gemini_queue = queue.PriorityQueue(), queue.PriorityQueue()
    total_items = len(hits)
    completed_lock = threading.Lock()
    completed_count = 0
    stop_event = threading.Event()

    for idx, hit in enumerate(hits):
        task_payload = {'item': hit, 'stage': 1, 'evals': {}}
        if idx % 2 == 0: claude_queue.put((20, next(counter), task_payload))
        else: gemini_queue.put((20, next(counter), task_payload))

    def claude_worker(worker_id):
        nonlocal completed_count
        while not stop_event.is_set():
            try: priority, seq, task = claude_queue.get(timeout=1.0)
            except queue.Empty:
                with completed_lock:
                    if completed_count >= total_items: break
                continue

            item, stage, evals = task['item'], task['stage'], task['evals']
            market_data = task.get('market_data') or fetch_market_metrics(item['scrip'], item['exchange'])
            forum_data = task.get('forum_data') or fetch_valuepickr_sentiment(item['company'])

            print(f"\n[Claude Worker-{worker_id} Stage {stage}] Analyzing {item['company']} ({item['exchange']}:{item['scrip']})...")
            prompt = build_sieve2_prompt(item, market_data, forum_data, item.get('raw_pdf_text', ''), item.get('qwen_summary', ''))
            claude_output = evaluate_with_claude(prompt)

            cat_score, comp_score = extract_score(claude_output, "Catalyst Score:"), extract_score(claude_output, "Company Quality Score:")
            evals.update({'claude_catalyst_score': cat_score if isinstance(cat_score, int) else 1, 'claude_company_score': comp_score if isinstance(comp_score, int) else 5, 'claude_analysis': claude_output})

            if stage == 1:
                if evals['claude_catalyst_score'] >= 5:
                    print(f" -> [Claude Worker-{worker_id}] Score {evals['claude_catalyst_score']}/10 >= 5. Handoff to Gemini (Priority {20 - evals['claude_catalyst_score']}).")
                    gemini_queue.put((20 - evals['claude_catalyst_score'], next(counter), {'item': item, 'stage': 2, 'evals': evals, 'market_data': market_data, 'forum_data': forum_data}))
                else:
                    finalize_single_model_ignore(item, evals, market_data, "claude", item.get('raw_pdf_text', ''))
                    with completed_lock:
                        completed_count += 1
                        if completed_count >= total_items: stop_event.set()
            elif stage == 2:
                finalize_dual_evaluation(item, evals, market_data, item.get('raw_pdf_text', ''))
                with completed_lock:
                    completed_count += 1
                    if completed_count >= total_items: stop_event.set()
            claude_queue.task_done()

    def gemini_worker(worker_id):
        nonlocal completed_count
        while not stop_event.is_set():
            try: priority, seq, task = gemini_queue.get(timeout=1.0)
            except queue.Empty:
                with completed_lock:
                    if completed_count >= total_items: break
                continue

            item, stage, evals = task['item'], task['stage'], task['evals']
            market_data = task.get('market_data') or fetch_market_metrics(item['scrip'], item['exchange'])
            forum_data = task.get('forum_data') or fetch_valuepickr_sentiment(item['company'])

            print(f"\n[Gemini Worker-{worker_id} Stage {stage}] Analyzing {item['company']} ({item['exchange']}:{item['scrip']})...")
            prompt = build_sieve2_prompt(item, market_data, forum_data, item.get('raw_pdf_text', ''), item.get('qwen_summary', ''))
            gemini_output = evaluate_with_gemini(prompt)

            cat_score, comp_score = extract_score(gemini_output, "Catalyst Score:"), extract_score(gemini_output, "Company Quality Score:")
            evals.update({'gemini_catalyst_score': cat_score if isinstance(cat_score, int) else 1, 'gemini_company_score': comp_score if isinstance(comp_score, int) else 5, 'gemini_analysis': gemini_output})

            if stage == 1:
                if evals['gemini_catalyst_score'] >= 5:
                    print(f" -> [Gemini Worker-{worker_id}] Score {evals['gemini_catalyst_score']}/10 >= 5. Handoff to Claude (Priority {20 - evals['gemini_catalyst_score']}).")
                    claude_queue.put((20 - evals['gemini_catalyst_score'], next(counter), {'item': item, 'stage': 2, 'evals': evals, 'market_data': market_data, 'forum_data': forum_data}))
                else:
                    finalize_single_model_ignore(item, evals, market_data, "gemini", item.get('raw_pdf_text', ''))
                    with completed_lock:
                        completed_count += 1
                        if completed_count >= total_items: stop_event.set()
            elif stage == 2:
                finalize_dual_evaluation(item, evals, market_data, item.get('raw_pdf_text', ''))
                with completed_lock:
                    completed_count += 1
                    if completed_count >= total_items: stop_event.set()
            gemini_queue.task_done()

    threads = []
    for i in range(num_workers):
        t_c, t_g = threading.Thread(target=claude_worker, args=(i + 1,), daemon=True), threading.Thread(target=gemini_worker, args=(i + 1,), daemon=True)
        threads.extend([t_c, t_g])
        t_c.start(); t_g.start()
    for t in threads: t.join()

def send_scan_digest(total_ingested, total_new, hits, rejections, duration_seconds):
    mode_text = "🔄 *Manual Refresh*" if IGNORE_CACHE else "⏰ *Scheduled GitHub Action Scan*"
    sast_count = sum(1 for r in rejections if any(k in r.get('rejection_reason', '').lower() for k in ['sast', 'pit', 'insider', 'transfer']))
    admin_count = sum(1 for r in rejections if any(k in r.get('rejection_reason', '').lower() for k in ['certificate', 'meeting', 'governance', 'window', 'newspaper']))
    other_count = len(rejections) - (sast_count + admin_count)

    hit_lines = ""
    if hits:
        hit_lines = "\n\n🎯 *High-Materiality Hits Passed to Sieve 2:*\n" + "\n".join([f"• *{h['company']}* (`{h['exchange']}:{h['scrip']}`) - Score: {h.get('qwen_pre_score', 'N/A')}/10" for h in hits])

    send_telegram_alert(
        f"📊 *EXCHANGE SCAN COMPLETE (GitHub Runner)*\n• *Mode:* {mode_text}\n• *Total Ingested:* {total_ingested} filings\n"
        f"• *New Entities Screened:* {total_new}\n• *Passed Sieve 1.5:* {len(hits)} out of {total_new} ({round((len(hits) / max(total_new, 1)) * 100, 1)}%)\n"
        f"• *Execution Latency:* {round(duration_seconds, 1)}s\n{hit_lines}\n\n🚫 *Noise Filter Breakdown:* ({len(rejections)})\n"
        f"• SAST / PIT / Insider Transfers: {sast_count}\n• Share Certificates / Board Meetings: {admin_count}\n• Routine Disclosures: {max(0, other_count)}"
    )


# ---------------------------------------------------------------------------
# 9. YOUTUBE TV INTERVIEWS & DUAL INGESTION
# ---------------------------------------------------------------------------
def process_youtube_interviews(channel_id=DEFAULT_YOUTUBE_CHANNEL_ID):
    if not gemini_client: return
    feed = feedparser.parse(YOUTUBE_RSS_URL.format(channel_id=channel_id))
    cutoff = datetime.now(timezone.utc) - timedelta(days=15)
    for entry in feed.entries[:4]:
        if datetime(*entry.published_parsed[:6], tzinfo=timezone.utc) < cutoff: continue
        try:
            transcript = YouTubeTranscriptApi.get_transcript(entry.yt_videoid)
            text = " ".join([t['text'] for t in transcript[:120]])
            res = gemini_client.models.generate_content(model=TIER1_MODEL, contents=f"Does this interview discuss capex expansion, guidance revisions, or major order pipeline?\nTitle: {entry.title}\nTranscript Preview: {text}\nReturn strictly the COMPANY_NAME if YES, or return IGNORE if routine.")
            if "IGNORE" not in res.text: send_telegram_alert(f"📺 *MANAGEMENT INTERVIEW CATALYST*\n\n**Title:** {entry.title}\n🔗 [Watch Interview]({entry.link})")
        except Exception: continue

def fetch_live_nse_filings():
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    standardized_filings = []
    try:
        session.get(NSE_BASE_URL, timeout=10)
        resp = session.get(NSE_API_URL, timeout=10)
        if resp.status_code == 200:
            data = resp.json() if isinstance(resp.json(), list) else resp.json().get('data', [])
            for item in data:
                attachment = item.get('attchmntFile') or item.get('attchmntText') or str(item.get('seq_id', ''))
                if not attachment: continue
                pdf_link = item.get('attchmntText', '')
                if pdf_link and not pdf_link.startswith('http'): pdf_link = f"{NSE_BASE_URL}{pdf_link}"
                standardized_filings.append({'id': str(attachment), 'company': item.get('sm_name', item.get('symbol', 'Unknown NSE Company')), 'scrip': item.get('symbol', ''), 'headline': item.get('subject', item.get('desc', '')), 'isin': 'N/A', 'link': pdf_link, 'exchange': 'NSE'})
    except Exception as e: print(f"[NSE Ingestion Notice] {e}")
    return standardized_filings

def fetch_live_bse_filings(max_pages=100):
    standardized_filings = []
    page = 1
    while page <= max_pages:
        try:
            resp = requests.get(BSE_API_URL.format(page=page), headers=BSE_HEADERS, timeout=10)
            if resp.status_code == 200:
                table = resp.json().get('Table', [])
                if not table: break
                for item in table:
                    if not item.get('ATTACHMENTNAME'): continue
                    raw_isin = item.get('ISIN_CODE', '').strip()
                    standardized_filings.append({'id': item['ATTACHMENTNAME'], 'company': item.get('SLONGNAME', 'Unknown Company'), 'scrip': str(item.get('SCRIP_CD', '')).strip(), 'headline': item.get('NEWSSUB', ''), 'isin': raw_isin if raw_isin else 'N/A', 'link': BSE_PDF_BASE_URL.format(attachment=item['ATTACHMENTNAME']), 'exchange': 'BSE'})
                page += 1; time.sleep(0.2)
            else: break
        except Exception as e:
            print(f"[BSE Ingestion Notice on Page {page}] {e}"); break
    return standardized_filings


# ---------------------------------------------------------------------------
# 10. MAIN PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    print(f"[{datetime.now()}] Initializing Market Intelligence Pipeline (GitHub Runner)...")

    unified_filings = fetch_live_bse_filings(max_pages=args.max_pages) + fetch_live_nse_filings()
    unprocessed_filings = filter_unprocessed_announcements(unified_filings)

    if not unprocessed_filings:
        print("No new filings found in this scan cycle. Exiting.")
        return

    grouped_filings = group_filings_by_company(unprocessed_filings)
    print(f"Consolidated into {len(grouped_filings)} distinct company events.")

    # Tier 1 Sieve
    hits, rejections = run_tier1_batch_sieve(grouped_filings)

    if rejections:
        log_announcements_batch([(att_id, r['company'], r['headline'], "IGNORE") for r in rejections for att_id in r['all_ids']])

    if hits and PUBLISH_WHEN_SIEVE_PASSED <= 1.0:
        for h in hits:
            mkt = fetch_market_metrics(h['scrip'], h['exchange'])
            alert_msg = build_telegram_message(h['company'], h['exchange'], h['scrip'], h['isin'], mkt, {'consensus_status': 'SIEVE_1_PASS', 'final_score': 'N/A', 'high_conviction': False, 'claude_catalyst_score': None, 'gemini_catalyst_score': None, 'claude_analysis': h.get('catalyst_reason', ''), 'gemini_analysis': ''}, h['all_links'])
            send_telegram_alert(alert_msg)

    # Tier 1.5 Sieve (Extracts via Router & Evaluates via Qwen)
    sieve_1_5_passed = []
    if hits:
        print(f"\n=======================================================")
        print(f"🤖 [SIEVE 1.5] Executing Dynamic Extraction & Qwen Pre-Scoring on {len(hits)} hits...")
        print(f"=======================================================\n")

        for h in hits:
            primary_link = h['all_links'][0] if h.get('all_links') else h.get('link', '')
            raw_pdf_text = extract_text_from_pdf_url(primary_link, h['headline'])
            qwen_summary, pre_score = sieve_1_5_local_qwen_extraction(raw_pdf_text, h['headline'])

            h['raw_pdf_text'] = raw_pdf_text
            h['qwen_summary'] = qwen_summary
            h['qwen_pre_score'] = pre_score

            print(f" -> [{h['company']}] Qwen Pre-Score: {pre_score}/10 (Threshold: >= {SIEVE_1_5_MIN_SCORE})")

            if pre_score >= SIEVE_1_5_MIN_SCORE:
                sieve_1_5_passed.append(h)
                if PUBLISH_WHEN_SIEVE_PASSED <= 1.5:
                    mkt = fetch_market_metrics(h['scrip'], h['exchange'])
                    alert_msg = build_telegram_message(h['company'], h['exchange'], h['scrip'], h['isin'], mkt, {'consensus_status': 'SIEVE_1_5_PASS', 'final_score': pre_score, 'high_conviction': False, 'claude_catalyst_score': None, 'gemini_catalyst_score': None, 'claude_analysis': qwen_summary[:800], 'gemini_analysis': ''}, h['all_links'])
                    send_telegram_alert(alert_msg)
            else:
                log_announcements_batch([(att_id, h['company'], h['headline'], "IGNORE") for att_id in h['all_ids']])
                print(f"    🚫 Filtered out locally by Sieve 1.5 (Score {pre_score} < {SIEVE_1_5_MIN_SCORE}).")

    # Tier 2 Sieve (Full PDF Passthrough)
    if sieve_1_5_passed:
        candidates = sieve_1_5_passed[:args.max_sieve2] if args.max_sieve2 > 0 else sieve_1_5_passed
        run_staggered_sieve2_workers(candidates, num_workers=4)

    duration = time.time() - start_time
    send_scan_digest(len(unified_filings), len(grouped_filings), sieve_1_5_passed, rejections, duration)
    process_youtube_interviews()
    print(f"[{datetime.now()}] Pipeline execution finished in {round(duration, 1)}s.")

if __name__ == "__main__":
    main()