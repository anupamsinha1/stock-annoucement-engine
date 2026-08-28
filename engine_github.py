"""
Market Intelligence & Corporate Announcement Screening Engine
=======================================================================================
Architecture & Features:
- State Machine Tracker: Defers all Telegram alerts to the end of the pipeline.
- Cascading Verbosity (20, 40, 60, 1000): Dictates pipeline telemetry via Threshold Gatekeeper.
- Sieve 20: Noise filter (Upgraded: Detects Open Market Promoter buying vs routine SAST).
- Sieve 40: Local Fluff Gatekeeper (Upgraded: Parses deal values dynamically).
- Sieve 60: Deep Dive Consensus (Upgraded: Computes Deal-to-Market-Cap ratios).
- Document Processor: (Upgraded: Gemini Flash Vision OCR fallback for scanned SMEs).
- Dispatcher: (Upgraded: Live Upper Circuit tags + Telegram Inline Keyboards).
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
from google.genai import types  # Required for Vision PDF fallback
from anthropic import Anthropic
from youtube_transcript_api import YouTubeTranscriptApi
from pypdf import PdfReader
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# 0. LOGGING & SDK CONFIGURATION
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# We suppress Google's internal API warnings so our console stays clean
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

# Loads local variables from .env file (if it exists)
load_dotenv()

# ---------------------------------------------------------------------------
# 1. EXECUTION ENVIRONMENT & CONFIGURATION
# ---------------------------------------------------------------------------
# [ENVIRONMENT SWITCH]
# This pulls the environment dynamically.
# - Locally, it reads from your .env file.
# - On GitHub, it reads from your scan.yml file.
# - If not found anywhere, it defaults to "GITHUB".
#
# Possible values:
#   "GITHUB" -> Uses 'qwen2.5:7b' (Optimized for GitHub Actions 4-core runners)
#   "LOCAL"  -> Uses 'llama3' (Perfect for local testing on your personal machine)
# ---------------------------------------------------------------------------
EXECUTION_ENVIRONMENT = os.getenv("EXECUTION_ENVIRONMENT", "GITHUB")

# [DECISION] Set the default local AI model based on the environment chosen above
if EXECUTION_ENVIRONMENT.upper() == "LOCAL":
    DEFAULT_LOCAL_MODEL = "llama3"
else:
    DEFAULT_LOCAL_MODEL = "qwen2.5:7b"

# Pull API Keys and model names from the .env file (or use the defaults we just set)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_LOCAL_MODEL)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
USE_LOCAL_EXTRACTOR = os.getenv("USE_LOCAL_EXTRACTOR", "true").lower() in ("true", "1", "yes")

# Define the source URLs where we will scrape the data
BSE_API_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?pageno={page}&strCat=-1&strPrevDate=&strScrip=&strSearch=P&strToDate=&strType=C"
BSE_PDF_BASE_URL = "https://www.bseindia.com/xml-data/corpfiling/AttachLive/{attachment}"
NSE_BASE_URL = "https://www.nseindia.com"
NSE_API_URL = "https://www.nseindia.com/api/corporate-announcements?index=equities"
VALUEPICKR_API_URL = "https://forum.valuepickr.com/search/query.json?term={term}"
YOUTUBE_RSS_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

# Headers make our Python script look like a real web browser to avoid getting blocked
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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/"
}

# ---------------------------------------------------------------------------
# VERBOSITY GATEKEEPER CONSTANT
# 20   = Debug Mode: Sends alerts for everything (Sieve 20, 40, and 60 passes AND rejects)
# 40   = Sends Sieve 40 and 60 results (Ignores the Sieve 20 noise)
# 60   = Sends Sieve 60 results only (Passes and early exits)
# 1000 = Production Mode: Sends ONLY fully passed Sieve 60 filings (Score >= 6)
# ---------------------------------------------------------------------------
VERBOSITY_LEVEL = float(os.getenv("VERBOSITY_LEVEL", "40.0"))

DEFAULT_CHUNK_SIZE = 50
DEFAULT_YOUTUBE_CHANNEL_ID = "UCb5hMTAFjG5j79V6nL3_YCQ"
DEFAULT_MAX_SIEVE60 = int(os.getenv("MAX_SIEVE60_ITEMS", "0"))
SIEVE_40_MIN_SCORE = int(os.getenv("SIEVE_40_MIN_SCORE", "5"))

# Parse command line arguments
parser = argparse.ArgumentParser(description="Corporate Announcement Screening Engine")
parser.add_argument("--ignore-cache", "-f", action="store_true", help="Bypass Supabase cache and re-evaluate all filings")
parser.add_argument("--max-pages", type=int, default=100, help="Maximum BSE announcement pages to fetch")
parser.add_argument("--max-sieve60", type=int, default=DEFAULT_MAX_SIEVE60, help="Max candidates sent to Sieve 60")
args, _ = parser.parse_known_args()

IGNORE_CACHE = args.ignore_cache or os.getenv("IGNORE_CACHE", "false").lower() in ("true", "1", "yes")

# Pull Cloud API Keys from the .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_PAID") or os.getenv("GEMINI_API_KEY_FREE")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SUPABASE_URL = os.getenv("SUPABASE_URL")

TIER1_MODEL = os.getenv("GEMINI_TIER1_MODEL", "gemini-3.5-flash-lite")
TIER2_GEMINI_MODEL = os.getenv("GEMINI_TIER2_MODEL", "gemini-3.1-pro-preview")
TIER2_CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")

# Initialize Cloud AI clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
claude_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Threading locks to prevent APIs from being overwhelmed when running in parallel
api_lock = threading.Lock()
master_list_lock = threading.Lock()
last_api_call = 0.0
MIN_API_DELAY = 1.5

def throttle_api():
    """[ROUTING] Forces a 1.5-second pause between heavy API calls to avoid 429 Too Many Requests errors."""
    global last_api_call
    with api_lock:
        now = time.time()
        elapsed = now - last_api_call
        if elapsed < MIN_API_DELAY:
            time.sleep(MIN_API_DELAY - elapsed)
        last_api_call = time.time()

# ---------------------------------------------------------------------------
# 2. DATABASE / LEDGER CACHE
# ---------------------------------------------------------------------------
def get_db_connection():
    if not SUPABASE_URL: return None
    try:
        return psycopg2.connect(SUPABASE_URL)
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Database Error] Connection failed: {e}")
        return None

def filter_unprocessed_announcements(filings):
    """[DECISION] Queries the DB. If the filing ID already exists, we drop it to avoid re-evaluating old news."""
    if IGNORE_CACHE:
        print(" [Cache] Ignoring DB cache based on CLI flags. Processing everything.")
        return filings

    conn = get_db_connection()
    if not conn:
        return filings

    try:
        cursor = conn.cursor()
        attachments = [item['id'] for item in filings if item.get('id')]
        if not attachments:
            conn.close()
            return []

        # Create a dynamic SQL query based on how many attachments we have
        query = f"SELECT bse_attachment_name FROM bse_announcements WHERE bse_attachment_name IN ({','.join(['%s'] * len(attachments))})"
        cursor.execute(query, tuple(attachments))
        existing = {row[0] for row in cursor.fetchall()}
        conn.close()

        # Keep only the items that were NOT found in the database
        filtered = [item for item in filings if item['id'] not in existing]
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Cache] Filtered out {len(filings) - len(filtered)} already processed filings.")
        return filtered
    except Exception as e:
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Database Error] Deduplication query failed: {e}")
        return filings

def group_filings_by_company(filings):
    """[DECISION] If a company releases 3 PDFs at the same time, we merge them into 1 master record here."""
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
            # Combine the headlines and links of subsequent filings into the first one
            existing = grouped[key]
            if item['id'] not in existing['all_ids']:
                existing['all_ids'].append(item['id'])
            if item['headline'] not in existing['headline']:
                existing['headline'] += f" | {item['headline']}"
            if item['link'] not in existing['all_links']:
                existing['all_links'].append(item['link'])

    return list(grouped.values())

def log_announcements_batch(decisions_list):
    """[STATE UPDATE] Saves basic triage decisions (HIT/IGNORE) so we don't process them again."""
    if not decisions_list:
        return
    conn = get_db_connection()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        query = """INSERT INTO bse_announcements (bse_attachment_name, company_name, headline, ai_decision) 
                   VALUES (%s, %s, %s, %s) ON CONFLICT (bse_attachment_name) DO UPDATE SET ai_decision = EXCLUDED.ai_decision"""
        cursor.executemany(query, decisions_list)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Database Error] Batch cache insertion failed: {e}")

def log_permanent_ledger(item, market_data):
    """[STATE UPDATE] Saves the massive, deep-dive AI analysis into our permanent historical tracker."""
    conn = get_db_connection()
    if not conn:
        return
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
            item.get('headline', ''), item.get('sieve20_reason', 'Actionable corporate action'),
            market_data.get('price', 0.0), market_data.get('market_cap_cr', 0.0), market_data.get('vol_multiple', 1.0),
            market_data.get('above_50dma', False), market_data.get('above_200dma', False),
            item.get('final_score', 1), item.get('status', 'NEUTRAL_MIX'), item.get('high_conviction', False),
            item.get('sieve60_claude_score'), item.get('sieve60_gemini_score'),
            item.get('sieve60_claude_company'), item.get('sieve60_gemini_company'),
            item.get('sieve40_summary', ''), item.get('sieve60_claude_analysis', ''), item.get('sieve60_gemini_analysis', '')
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Database Error] Comprehensive ledger logging failed: {e}")

# ---------------------------------------------------------------------------
# 3. METRICS & EXTRACTION ROUTER
# ---------------------------------------------------------------------------
def fetch_market_metrics(scrip_code, exchange):
    """
    [CALCULATION] Fetches live stock price, 20-day volume surge, and moving averages.
    [IMPROVEMENT 3] Upgraded to detect Live Upper Circuit Locks.
    """
    default_payload = {"price": 0.0, "vol_multiple": 1.0, "above_50dma": False, "above_200dma": False, "dist_52w_high": 0.0, "market_cap_cr": 0.0, "is_upper_circuit": False}
    if not scrip_code:
        return default_payload

    try:
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Metrics] Fetching market data for {exchange}:{scrip_code}...")
        ticker = f"{scrip_code}.NS" if exchange == "NSE" else f"{scrip_code}.BO"
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        if hist.empty or len(hist) < 20:
            return default_payload

        current_price = round(float(hist['Close'].iloc[-1]), 2)

        # Compare today's volume to the average of the last 20 days
        avg_20_volume = float(hist['Volume'].iloc[-21:-1].mean()) if len(hist) >= 21 else float(hist['Volume'].iloc[-1])
        vol_multiple = round((float(hist['Volume'].iloc[-1]) / avg_20_volume), 2) if avg_20_volume > 0 else 1.0

        dma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else current_price
        dma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else current_price
        high_52w = float(hist['High'].max())

        # How far away is the stock from its 52-week peak?
        dist_52w_high = round(((current_price - high_52w) / high_52w) * 100, 1) if high_52w > 0 else 0.0
        market_cap_cr = round(getattr(stock.fast_info, 'market_cap', 0) / 1e7, 2)

        # [UPPER CIRCUIT CALCULATION]
        # Detect if the stock is locked at the upper circuit to notify the user of illiquidity
        is_uc = False
        if len(hist) >= 2:
            today_high = float(hist['High'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2])
            if prev_close > 0:
                pct_change = (current_price - prev_close) / prev_close
                # If price is at the absolute day's high AND up more than 4.5%, it's almost certainly locked in a circuit
                if current_price >= (today_high * 0.998) and pct_change >= 0.045:
                    is_uc = True
                    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Metrics] UPPER CIRCUIT LOCK DETECTED for {scrip_code}!")

        return {
            "price": current_price, "vol_multiple": vol_multiple,
            "above_50dma": bool(current_price > dma_50), "above_200dma": bool(current_price > dma_200),
            "dist_52w_high": dist_52w_high, "market_cap_cr": market_cap_cr, "is_upper_circuit": is_uc
        }
    except Exception as e:
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Metrics Error] Failed to fetch data for {scrip_code}: {e}")
        return default_payload

def fetch_valuepickr_sentiment(company_name):
    """[EXTERNAL API] Scrapes the ValuePickr forum to see if retailers are discussing this stock."""
    try:
        clean_name = company_name.split()[0].replace("Ltd", "").replace("Limited", "").strip()
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Scuttlebutt] Searching ValuePickr forum for '{clean_name}'...")
        res = requests.get(VALUEPICKR_API_URL.format(term=clean_name), headers=DEFAULT_HEADERS, timeout=5)

        if res.status_code != 200:
            return "No active forum discussion found."

        posts = res.json().get('posts', [])
        if posts:
            print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Scuttlebutt] Found {len(posts)} recent posts for {clean_name}.")
            return " ".join([p.get('blurb', '') for p in posts[:5]])[:1500]
        else:
            print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Scuttlebutt] No posts found for {clean_name}.")
            return "No active forum discussion found."
    except Exception as e:
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Scuttlebutt Error] ValuePickr search failed: {e}")
        return "Forum search bypassed."

def extract_text_from_pdf_url(pdf_url, headline):
    """
    [DECISION: DYNAMIC ROUTER] Routes PDFs to the right extraction engine based on context.
    [IMPROVEMENT 1] Integrates Gemini Flash Vision OCR for unextractable scanned SME PDFs.
    """
    if not pdf_url or not pdf_url.startswith("http"):
        return "No valid PDF URL."

    headline_lower = headline.lower()
    financial_keywords = ["financial result", "outcome of board meeting", "earnings", "annual report", "financial statement"]
    is_heavy_financial = any(kw in headline_lower for kw in financial_keywords)

    max_pages, max_chars = (15, 20000) if is_heavy_financial else (4, 10000)

    if is_heavy_financial:
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [PDF Router] Financial terms detected. Using heavy pdfplumber extractor (Limit: {max_chars} chars)...")
    else:
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [PDF Router] Standard filing detected. Using fast pypdf extractor (Limit: {max_chars} chars)...")

    try:
        res = requests.get(pdf_url, headers=DEFAULT_HEADERS, timeout=15)
        if res.status_code != 200:
            return f"Failed to download PDF ({res.status_code})"

        pdf_bytes = res.content
        raw_text = ""

        with io.BytesIO(pdf_bytes) as pdf_buffer:
            if is_heavy_financial and pdfplumber:
                extracted_pages = []
                with pdfplumber.open(pdf_buffer) as pdf:
                    for i, page in enumerate(pdf.pages):
                        if i >= max_pages: break
                        text = page.extract_text(layout=True) or page.extract_text()
                        if text: extracted_pages.append(text)
                raw_text = "\n".join(extracted_pages)
            else:
                reader = PdfReader(pdf_buffer)
                raw_text = "\n".join([page.extract_text() for page in reader.pages[:max_pages] if page.extract_text()])

        clean_text = re.sub(r'(?i)(?:\bDisclaimer\b|CAREEDGE RATINGS DISCLAIMS).*$', '', raw_text, flags=re.DOTALL)
        clean_text = re.sub(r'[ \t]+', ' ', re.sub(r'\n{2,}', '\n', re.sub(r'[^\x00-\x7F₹]+', '', clean_text))).strip()

        # [VISION OCR FALLBACK CHECK]
        # If text is nearly empty, it's a flat image scan (very common in SME court orders/factory docs)
        if len(clean_text) < 80 and gemini_client:
            print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Vision Fallback] Blank/Scanned PDF detected. Passing bytes to Gemini Flash Vision OCR...")
            try:
                throttle_api()
                resp = gemini_client.models.generate_content(
                    model=TIER1_MODEL,
                    contents=[
                        types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
                        "Extract all readable text, tables, and financial figures from this scanned document. Return as clean, unformatted text."
                    ]
                )
                clean_text = resp.text.strip()
                print(" [Vision Fallback] Successfully OCR'd scanned document.")
            except Exception as e:
                print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Vision Fallback Error] Unable to read scanned document: {e}")

        return clean_text[:max_chars] if clean_text else "Unextractable text."
    except Exception as e:
        return f"PDF error: {e}"


# ---------------------------------------------------------------------------
# 4. SIEVES 20, 40, 60 LOGIC (STATE TRACKERS)
# ---------------------------------------------------------------------------
def run_sieve20_batch(announcements, master_results):
    """
    [SIEVE 20] Reads just the headline of 50 filings at once using fast Gemini Flash.
    [IMPROVEMENT 4] Explicit instructions to HIT on Open Market Promoter buying while rejecting normal SAST noise.
    """
    if not announcements or not gemini_client:
        return []
    hits = []

    for i in range(0, len(announcements), DEFAULT_CHUNK_SIZE):
        chunk = announcements[i:i + DEFAULT_CHUNK_SIZE]
        items_payload = [{"index": idx, "exchange": a['exchange'], "scrip": a['scrip'], "company": a['company'], "headline": a['headline']} for idx, a in enumerate(chunk)]

        prompt = f"""You are an objective exchange filing intake filter. Categorize EACH announcement as either "HIT" or "REJECT".
        HIGH-MATERIALITY CATALYSTS (HIT): Financial Results, Commercial Orders, Fundraisings/M&A, Capex, FDA Clearances, AND Promoter Open Market Acquisitions (Explicit Creeping/Insider Buying).
        CRITICAL EXCLUSIONS (REJECT): Loss of shares, trading window closures, shareholding patterns, ESOPs, newspaper clippings, and Routine SAST (Pledge/Encumbrance releases).
        Respond with ONLY a valid JSON array. Announcements: {json.dumps(items_payload, indent=2)}"""

        for attempt in range(3):
            try:
                throttle_api()
                response = gemini_client.models.generate_content(model=TIER1_MODEL, contents=prompt)
                parsed = json.loads(response.text.strip().replace("```json", "").replace("```", ""))

                for dec in parsed:
                    idx = dec.get("index")
                    if idx is not None and idx < len(chunk):
                        ann_item = chunk[idx]
                        status, reason = dec.get("status", "REJECT").upper(), dec.get("reason", "Routine filing")
                        ann_item["sieve20_reason"] = reason

                        if status == "HIT":
                            hits.append(ann_item)
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [SIEVE 20 HIT] {ann_item['company']} | {reason}")
                        else:
                            # [STATE UPDATE] Document died at Sieve 20. Mark terminal stage and save to master list.
                            ann_item["terminal_stage"] = 20
                            ann_item["status"] = "REJECTED_SIEVE20"
                            master_results.append(ann_item)
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [SIEVE 20 REJECT] {ann_item['company']} | {reason}")
                break
            except Exception as e:
                if "503" in str(e) or "429" in str(e):
                    time.sleep(2 ** attempt)
                    continue
                print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Sieve 20 Error] Chunk failed: {e}. Defaulting to HIT.")
                hits.extend(chunk)
                break

    return hits


def run_sieve40_extraction(item):
    """
    [SIEVE 40] Downloads the PDF, extracts text, and uses the Groq API to score it.
    [IMPROVEMENT 2: PART A] Regex parser successfully grabs numeric deal value from the summary block.
    """
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [SIEVE 40] Downloading and parsing PDF for {item['company']}...")
    text = extract_text_from_pdf_url(item['all_links'][0] if item.get('all_links') else item.get('link', ''),
                                     item['headline'])
    item['raw_pdf_text'] = text
    item['deal_value_cr'] = 0.0  # Default value

    if not text or len(text) < 80:
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [SIEVE 40] No readable text found for {item['company']}. Defaulting score to 5.")
        item['sieve40_summary'], item['sieve40_score'] = "No local extraction.", 5
        return item

    # Dynamic text slicing: 20k chars for heavy financials, 10k for standard announcements
    is_heavy_financial = any(kw in item.get('headline', '').lower() for kw in
                             ['financial results', 'outcome of board meeting', 'audited', 'unaudited'])
    max_chars = 20000 if is_heavy_financial else 10000

    prompt = f"""Extract key facts and assign a PreScore (1 to 10).
    Score 1-4: Trade expo, generic PR, minor updates.
    Score 5-6: Small/routine purchase orders, incremental progress.
    Score 7-10: Hard confirmed wins (>50Cr), net-debt reduction, major capex, strong financial beats.
    Document Text: {item['headline']}\n{text[:max_chars]}\nOutput format:\nSummary: [150 words]\nValue: [Exact value]\nClient: [Entity]\nPreScore: [Integer 1-10]"""

    headers = {
        "Authorization": f"Bearer {os.getenv("GROQ_API_KEY")}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system",
             "content": "You are a precise financial analyst extracting metrics from corporate announcements."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "stream": False
    }

    try:
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [SIEVE 40] Triggering Groq API ({OLLAMA_MODEL}) for {item['company']}...")
        # Inside your loop or right before triggering the API:
        time.sleep(3)
        res = requests.post(OLLAMA_API_URL, headers=headers, json=payload, timeout=60)

        if res.status_code == 200:
            response_data = res.json()
            extracted = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            score_match = re.search(r'PreScore.*?(\b10|[1-9])\b', extracted, re.IGNORECASE)
            item['sieve40_score'] = int(score_match.group(1)) if score_match else 5
            item['sieve40_summary'] = extracted

            # [DEAL VALUE EXTRACTION FOR RATIO CALCULATION]
            val_match = re.search(r'(?i)Value:\s*.*?(\d+(?:\.\d+)?)\s*(Cr|Crore|Million|Mn|Billion|Bn)', extracted)
            if val_match:
                val = float(val_match.group(1))
                unit = val_match.group(2).lower()
                if 'm' in unit:
                    item['deal_value_cr'] = round(val / 10, 2)
                elif 'b' in unit:
                    item['deal_value_cr'] = round(val * 100, 2)
                else:
                    item['deal_value_cr'] = val

            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [SIEVE 40] Scored {item['sieve40_score']}/10. Extracted Deal Value: ~{item['deal_value_cr']} Cr.")
            return item
        else:
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [SIEVE 40] Groq API HTTP error {res.status_code}: {res.text}")
    except Exception as e:
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [SIEVE 40] Groq connection failed ({e}). Defaulting score to 5.")

    item['sieve40_summary'], item['sieve40_score'] = "Groq inference failed.", 5
    return item


def build_sieve60_prompt(item, market_data, forum_text):
    """
    [IMPROVEMENT 2: PART B] Anchors the deep-dive models with an exact quantitative deal-to-market-cap ratio.
    """
    anchor_text = ""
    deal_cr = item.get('deal_value_cr', 0.0)
    mcap_cr = market_data.get('market_cap_cr', 0.0)

    if deal_cr > 0 and mcap_cr > 0:
        ratio = (deal_cr / mcap_cr) * 100
        anchor_text = f"\nQUANTITATIVE ANCHOR: The extracted deal value is ~INR {deal_cr} Cr, representing exactly {ratio:.1f}% of the current market cap.\n"
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Prompt Builder] Injected Quantitative Anchor: {ratio:.1f}% of Market Cap.")

    return f"""Assess how strongly this event will impact future earnings power. Disregard market cap (evaluate proportional impact).
    Company: {item['company']} | Price: {market_data['price']} | 20D Vol: {market_data['vol_multiple']}x | 52W High Dist: {market_data['dist_52w_high']}%
    Headline: {item['headline']} | Forum Context: {forum_text}{anchor_text}
    Pre-Screen: {item.get('sieve40_summary','')}
    Raw Document: {item.get('raw_pdf_text','')}
    OUTPUT FORMAT:
    Reasoning: <2-3 sentences on commercial impact>
    Catalyst Score: <Integer 1-10>
    Company Quality Score: <Integer 1-10>"""

def run_staggered_sieve60_workers(candidates, master_results):
    """
    [SIEVE 60] The heavy-lifters (Claude and Gemini Pro).
    Uses a priority queue to hand off filings between models dynamically.
    """
    if not candidates:
        return

    counter = itertools.count()
    claude_queue, gemini_queue = queue.PriorityQueue(), queue.PriorityQueue()
    stop_event = threading.Event()
    total_items, completed_count = len(candidates), 0

    # [ROUTING] Assign half the filings to Claude first, and half to Gemini first.
    for idx, hit in enumerate(candidates):
        if idx % 2 == 0:
            claude_queue.put((20, next(counter), {'item': hit, 'stage': 1}))
        else:
            gemini_queue.put((20, next(counter), {'item': hit, 'stage': 1}))

    def extract_scores(text):
        c_m = re.search(r'Catalyst Score.*?(\b10|[1-9])\b', text, re.IGNORECASE)
        q_m = re.search(r'Company Quality Score.*?(\b10|[1-9])\b', text, re.IGNORECASE)
        return int(c_m.group(1)) if c_m else 1, int(q_m.group(1)) if q_m else 5

    def worker_logic(queue_in, queue_out, model_name, client_func):
        nonlocal completed_count
        while not stop_event.is_set():
            try:
                _, _, task = queue_in.get(timeout=1.0)
            except queue.Empty:
                if completed_count >= total_items:
                    break
                continue

            item, stage = task['item'], task['stage']
            market_data = item.get('market_data') or fetch_market_metrics(item['scrip'], item['exchange'])
            forum_data = item.get('forum_data') or fetch_valuepickr_sentiment(item['company'])
            item.update({'market_data': market_data, 'forum_data': forum_data})

            print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{model_name.upper()} Worker - Stage {stage}] Analyzing {item['company']}...")
            output = client_func(build_sieve60_prompt(item, market_data, forum_data))
            cat_score, comp_score = extract_scores(output)

            item[f'sieve60_{model_name}_score'] = cat_score
            item[f'sieve60_{model_name}_company'] = comp_score
            item[f'sieve60_{model_name}_analysis'] = output

            if stage == 1:
                # [DECISION] If the first model scores it >= 5, pass to the second model to get a consensus.
                if cat_score >= 5:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   -> [{model_name.upper()}] Score {cat_score}/10 >= 5. Handing off to Stage 2.")
                    queue_out.put((20 - cat_score, next(counter), {'item': item, 'stage': 2}))
                else:
                    # [STATE UPDATE] First model hated it. Kill it early to save API tokens.
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   -> [{model_name.upper()}] Early Exit! Score {cat_score}/10 < 5. Terminating.")
                    item['terminal_stage'] = 60
                    item['status'] = "SINGLE_MODEL_IGNORE"
                    item['final_score'] = cat_score
                    with master_list_lock:
                        master_results.append(item)
                        completed_count += 1

            elif stage == 2:
                # [DECISION] Both models have scored it. Calculate the consensus average.
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   -> [{model_name.upper()}] Stage 2 Complete. Finalizing consensus.")
                c_cat = item.get('sieve60_claude_score', 1)
                g_cat = item.get('sieve60_gemini_score', 1)

                is_divergent = (abs(c_cat - g_cat) >= 4 or (c_cat >= 7 and g_cat <= 4) or (g_cat >= 7 and c_cat <= 4))

                if is_divergent:
                    status = "MODEL_DIVERGENCE"
                elif c_cat >= 7 and g_cat >= 7:
                    status = "CONSENSUS_HIT"
                elif c_cat <= 4 and g_cat <= 4:
                    status = "CONSENSUS_IGNORE"
                else:
                    status = "NEUTRAL_MIX"

                final_score = round((c_cat + g_cat) / 2)
                item['terminal_stage'] = 60
                item['status'] = status
                item['final_score'] = final_score
                item['high_conviction'] = (final_score >= 8 and market_data['vol_multiple'] >= 2.0 and market_data['above_50dma'])

                with master_list_lock:
                    master_results.append(item)
                    completed_count += 1

            queue_in.task_done()

    def claude_func(prompt):
        throttle_api()
        res = claude_client.messages.create(model=TIER2_CLAUDE_MODEL, max_tokens=500, messages=[{"role": "user", "content": prompt}])
        return "\n".join([b.text for b in res.content if getattr(b, "type", None) == "text"]).strip()

    def gemini_func(prompt):
        throttle_api()
        return gemini_client.models.generate_content(model=TIER2_GEMINI_MODEL, contents=prompt).text.strip()

    # Start 8 parallel workers (4 Claude, 4 Gemini)
    threads = []
    for _ in range(4):
        t_c = threading.Thread(target=worker_logic, args=(claude_queue, gemini_queue, "claude", claude_func))
        t_g = threading.Thread(target=worker_logic, args=(gemini_queue, claude_queue, "gemini", gemini_func))
        threads.extend([t_c, t_g])
        t_c.start()
        t_g.start()

    for t in threads:
        t.join()


# ---------------------------------------------------------------------------
# 5. MASTER DISPATCHER (GATEKEEPER)
# ---------------------------------------------------------------------------
def dispatch_deferred_alerts(pipeline_results):
    """
    [DECISION: VERBOSITY ROUTER]
    Loops through every single filing processed today and decides whether to alert you
    based on the VERBOSITY_LEVEL constant.
    [IMPROVEMENT 5] Generates Interactive Telegram Inline Keyboards instead of text links.
    """
    print(f"\n=======================================================")
    print(f"📡 DISPATCHING DEFERRED ALERTS (VERBOSITY: {VERBOSITY_LEVEL})")
    print(f"=======================================================\n")

    for item in pipeline_results:
        stage = item.get('terminal_stage', 0)
        status = item.get('status', '')
        final_score = item.get('final_score', 0)
        should_send = False

        if VERBOSITY_LEVEL <= 20:
            should_send = True
        elif VERBOSITY_LEVEL <= 40 and stage >= 40:
            should_send = True
        elif VERBOSITY_LEVEL <= 60 and stage >= 60:
            should_send = True
        elif VERBOSITY_LEVEL >= 1000:
            if stage == 60 and status not in ["SINGLE_MODEL_IGNORE", "REJECTED_SIEVE20", "REJECTED_SIEVE40"] and final_score >= 6:
                should_send = True

        if should_send and TELEGRAM_BOT_TOKEN:
            try:
                market_data = item.get('market_data', fetch_market_metrics(item['scrip'], item['exchange']))
                uc_flag = " 🔒 [UPPER CIRCUIT]" if market_data.get('is_upper_circuit') else ""

                # Set the headline banner based on where it stopped
                if stage == 20: banner = "🗑️ <b>FILTERED: SIEVE 20 REJECT</b>"
                elif stage == 40: banner = "🚫 <b>FILTERED: SIEVE 40 REJECT</b>"
                elif status == "SINGLE_MODEL_IGNORE": banner = "🚫 <b>FILTERED: SIEVE 60 LOW CONVICTION</b>"
                elif status == "MODEL_DIVERGENCE": banner = f"⚠️ <b>MODEL DIVERGENCE DETECTED</b>{uc_flag}"
                elif item.get('high_conviction'): banner = f"🚨 <b>HIGH CONVICTION CATALYST</b>{uc_flag}"
                else: banner = f"📢 <b>CORPORATE ACTION CATALYST</b>{uc_flag}"

                price_str = f"₹{market_data.get('price', 0.0)}" if market_data.get('price', 0.0) > 0 else "₹0.0 (Feed Sync)"

                msg = (f"{banner}\n<b>Company:</b> {item['company']}\n"
                       f"<b>{item['exchange']}:</b> <code>{item['scrip']}</code> | <b>ISIN:</b> <code>{item.get('isin', 'N/A')}</code>\n"
                       f"<b>Price:</b> {price_str} | <b>20D Vol:</b> {market_data.get('vol_multiple', 1.0)}x | <b>52W High:</b> {market_data.get('dist_52w_high', 0.0)}%\n"
                       f"<b>Est. MktCap:</b> ₹{market_data.get('market_cap_cr', 0)} Cr\n")

                if stage == 60:
                    c_cat, g_cat = item.get('sieve60_claude_score'), item.get('sieve60_gemini_score')
                    cat_scores = []
                    if c_cat is not None: cat_scores.append(f"Claude: {c_cat}/10")
                    if g_cat is not None: cat_scores.append(f"Gemini: {g_cat}/10")
                    cat_line = " | ".join(cat_scores) if cat_scores else f"Score: {item.get('final_score', 'N/A')}/10"
                    msg += f"<b>Consensus Status:</b> <code>{status}</code>\n\n🎯 <b>Catalyst Score:</b> {cat_line}\n"

                # Cascading Trail
                if item.get('sieve20_reason'):
                    msg += f"\n🔍 <b>Sieve 20 Rationale:</b>\n{item['sieve20_reason']}\n"
                if stage >= 40 and item.get('sieve40_summary'):
                    msg += f"\n🤖 <b>Sieve 40 Score ({item.get('sieve40_score', 'N/A')}/10):</b>\n{item['sieve40_summary'][:600]}...\n"
                if stage == 60:
                    if item.get('sieve60_claude_analysis'):
                        msg += f"\n🧠 <b>Claude (Sieve 60):</b>\n{item['sieve60_claude_analysis']}\n"
                    if item.get('sieve60_gemini_analysis'):
                        msg += f"\n🤖 <b>Gemini (Sieve 60):</b>\n{item['sieve60_gemini_analysis']}\n"

                # [TELEGRAM BUTTONS BUILDER] Build the interactive grid below the message
                inline_keyboard = [[
                    {"text": "📊 Screener.in", "url": f"https://www.screener.in/company/{item['scrip']}/"},
                    {"text": "📈 TradingView", "url": f"https://in.tradingview.com/chart/?symbol={'NSE' if item['exchange'] == 'NSE' else 'BSE'}:{item['scrip']}"}
                ]]
                for i, url in enumerate(item.get('all_links', [])):
                    inline_keyboard.append([{"text": f"📑 Read Original PDF {i+1}", "url": url}])

                payload = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": msg,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                    "reply_markup": {"inline_keyboard": inline_keyboard}
                }

                print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Telegram] Dispatching payload for {item['company']}...")
                requests.post(TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN), json=payload, timeout=8)
                time.sleep(0.5) # Anti-spam buffer
            except Exception as e:
                print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Telegram Error] Dispatch failed for {item['company']}: {e}")

        # Ledger & DB Caching for all processed items
        if stage >= 60:
            log_permanent_ledger(item, item.get('market_data', {}))

        decision = "HIT" if stage == 60 and final_score >= 6 else "IGNORE"
        log_announcements_batch([(att_id, item['company'], item['headline'], decision) for att_id in item['all_ids']])


# ---------------------------------------------------------------------------
# 6. RAW DATA INGESTION METHODS
# ---------------------------------------------------------------------------
def process_youtube_interviews(channel_id=DEFAULT_YOUTUBE_CHANNEL_ID):
    """[EXTERNAL API] Downloads recent TV interviews and asks Gemini to look for capex/guidance mentions."""
    if not gemini_client: return
    try:
        print(" [YouTube] Scanning recent management interviews...")
        feed = feedparser.parse(YOUTUBE_RSS_URL.format(channel_id=channel_id))
        cutoff = datetime.now(timezone.utc) - timedelta(days=15)

        for entry in feed.entries[:4]:
            if datetime(*entry.published_parsed[:6], tzinfo=timezone.utc) < cutoff: continue
            transcript = YouTubeTranscriptApi.get_transcript(entry.yt_videoid)
            text = " ".join([t['text'] for t in transcript[:120]])

            prompt = f"Does this interview discuss capex expansion, guidance revisions, or major order pipeline?\nTitle: {entry.title}\nTranscript Preview: {text}\nReturn strictly the COMPANY_NAME if YES, or return IGNORE if routine."
            res = gemini_client.models.generate_content(model=TIER1_MODEL, contents=prompt)

            if "IGNORE" not in res.text:
                print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [YouTube Hit] Catalyst found in: {entry.title}")
                if VERBOSITY_LEVEL <= 1000:
                    btn = {"inline_keyboard": [[{"text": "📺 Watch Interview", "url": entry.link}]]}
                    requests.post(TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN), json={"chat_id": TELEGRAM_CHAT_ID, "text": f"📺 <b>MANAGEMENT INTERVIEW CATALYST</b>\n\n<b>Title:</b> {entry.title}", "parse_mode": "HTML", "reply_markup": btn}, timeout=8)
    except Exception as e:
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [YouTube Scanner Error] {e}")

def fetch_live_nse_filings():
    """[EXTERNAL API] Hits the NSE API using a Session to preserve cookies and bypass basic bot protection."""
    print(" [NSE] Fetching latest announcements from NSE India...")
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
                if pdf_link and not pdf_link.startswith('http'):
                    pdf_link = f"{NSE_BASE_URL}{pdf_link}"
                standardized_filings.append({'id': str(attachment), 'company': item.get('sm_name', item.get('symbol', 'Unknown')), 'scrip': item.get('symbol', ''), 'headline': item.get('subject', item.get('desc', '')), 'isin': 'N/A', 'link': pdf_link, 'exchange': 'NSE'})
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [NSE] Successfully fetched {len(standardized_filings)} filings.")
    except Exception as e:
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [NSE Ingestion Error] {e}")
    return standardized_filings

def fetch_live_bse_filings(max_pages=100):
    """[EXTERNAL API] Paginates backward through the live BSE announcement feed."""
    print(" [BSE] Fetching latest announcements from BSE India...")
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
                    standardized_filings.append({'id': item['ATTACHMENTNAME'], 'company': item.get('SLONGNAME', 'Unknown'), 'scrip': str(item.get('SCRIP_CD', '')).strip(), 'headline': item.get('NEWSSUB', ''), 'isin': item.get('ISIN_CODE', '').strip() or 'N/A', 'link': BSE_PDF_BASE_URL.format(attachment=item['ATTACHMENTNAME']), 'exchange': 'BSE'})
                page += 1
                time.sleep(0.2)
            else:
                break
        except Exception as e:
            print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [BSE Ingestion Notice on Page {page}] {e}")
            break

    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [BSE] Successfully fetched {len(standardized_filings)} filings across {page - 1} pages.")
    return standardized_filings


def print_stage_telemetry(stage_name, total_input, survived_count, pipeline_results, terminal_stage_num):
    """[TELEMETRY] Reusable method to print drop-off stats immediately after any sieve."""
    rejects_count = len([item for item in pipeline_results if item.get('terminal_stage') == terminal_stage_num])
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  =======================================================")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  🛡️ {stage_name.upper()} COMPLETE")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  =======================================================")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  • Evaluated Input : {total_input} entities")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  • Rejected / Dead : {rejects_count}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  • Survived to Next: {survived_count}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  =======================================================\n")

# ---------------------------------------------------------------------------
# 7. MAIN PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------
def main():
    print(f"Value of sieve 40 model being used is : {OLLAMA_MODEL}")
    start_time = time.time()
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{datetime.now()}] Initializing Pipeline (Verbosity: {VERBOSITY_LEVEL} | Env: {EXECUTION_ENVIRONMENT})...")

    unified = fetch_live_bse_filings(max_pages=args.max_pages) + fetch_live_nse_filings()
    unprocessed = filter_unprocessed_announcements(unified)

    if not unprocessed:
        print("No new filings found in this scan cycle.")
        return

    grouped = group_filings_by_company(unprocessed)
    pipeline_results = []

    print(f"\n {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [SIEVE 20] Broad-net noise filtering on {len(grouped)} filings...")
    sieve20_hits = run_sieve20_batch(grouped, pipeline_results)

    print_stage_telemetry(
        stage_name="Sieve 20 (Flash Noise Filter)",
        total_input=len(grouped),
        survived_count=len(sieve20_hits),
        pipeline_results=pipeline_results,
        terminal_stage_num=20
    )

    sieve40_hits = []
    if sieve20_hits:
        print(f"\n {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [SIEVE 40] Executing Dynamic Extraction & Pre-Scoring on {len(sieve20_hits)} hits...")
        for item in sieve20_hits:
            item = run_sieve40_extraction(item)
            if item.get('sieve40_score', 5) >= SIEVE_40_MIN_SCORE:
                sieve40_hits.append(item)
            else:
                print(f" -> [{item['company']}] Sieve 40 Score: {item.get('sieve40_score', 5)}/10. Rejected.")
                item['terminal_stage'] = 40
                item['status'] = "REJECTED_SIEVE40"
                pipeline_results.append(item)

    print_stage_telemetry(
        stage_name="Sieve 40 (Local LLM Pre-Score)",
        total_input=len(sieve20_hits),
        survived_count=len(sieve40_hits),
        pipeline_results=pipeline_results,
        terminal_stage_num=40
    )

    if sieve40_hits:
        print(f"\n {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [SIEVE 60] Handing off {len(sieve40_hits)} candidates to Claude & Gemini for deep reasoning...")
        run_staggered_sieve60_workers(sieve40_hits[:args.max_sieve60] if args.max_sieve60 > 0 else sieve40_hits, pipeline_results)

    sieve60_passed = [item for item in pipeline_results if
                      item.get('terminal_stage') == 60 and item.get('final_score', 0) >= 6]
    print_stage_telemetry(
        stage_name="Sieve 60 (Claude & Gemini Consensus)",
        total_input=len(sieve40_hits),
        survived_count=len(sieve60_passed),
        pipeline_results=pipeline_results,
        terminal_stage_num=60
    )

    # ---------------------------------------------------------------------------
    # FINAL DISPATCH & TELEMETRY
    # ---------------------------------------------------------------------------
    dispatch_deferred_alerts(pipeline_results)

    duration = time.time() - start_time
    sieve60_passed = [item for item in pipeline_results if item.get('terminal_stage') == 60 and item.get('final_score', 0) >= 6]
    sieve20_rejects = [item for item in pipeline_results if item.get('terminal_stage') == 20]

    sast_count = sum(1 for r in sieve20_rejects if any(k in r.get('sieve20_reason', '').lower() for k in ['sast', 'pit', 'insider', 'transfer']))
    admin_count = sum(1 for r in sieve20_rejects if any(k in r.get('sieve20_reason', '').lower() for k in ['certificate', 'meeting', 'governance', 'window', 'newspaper']))
    other_count = max(0, len(sieve20_rejects) - (sast_count + admin_count))

    print(f"\n=======================================================")
    print(f"📊 PIPELINE EXECUTION SUMMARY")
    print(f"=======================================================")
    print(f"• Total Filings Ingested : {len(unified)}")
    print(f"• Passed Sieve 20 (Flash): {len(sieve20_hits)}")
    print(f"• Passed Sieve 40 (Local): {len(sieve40_hits)}")
    print(f"• Passed Sieve 60 (Final): {len(sieve60_passed)}")
    print(f"• Execution Latency      : {round(duration, 1)}s")
    print(f"=======================================================\n")

    if TELEGRAM_BOT_TOKEN:
        digest_msg = (
            f"📊 <b>EXCHANGE SCAN COMPLETE</b>\n• <b>Mode:</b> {'🔄 Manual' if IGNORE_CACHE else '⏰ Scheduled'}\n"
            f"• <b>Total Screened:</b> {len(grouped)} entities\n• <b>Passed Sieve 20:</b> {len(sieve20_hits)}\n"
            f"• <b>Passed Sieve 40:</b> {len(sieve40_hits)}\n• <b>Final Sieve 60 Hits:</b> {len(sieve60_passed)}\n"
            f"• <b>Execution Latency:</b> {round(duration, 1)}s\n\n🚫 <b>Sieve 20 Noise Breakdown:</b>\n"
            f"• SAST / Insider Transfers: {sast_count}\n• Admin / Board Meetings: {admin_count}\n• Routine Disclosures: {other_count}"
        )
        try:
            requests.post(TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN), json={"chat_id": TELEGRAM_CHAT_ID, "text": digest_msg, "parse_mode": "HTML"}, timeout=8)
        except Exception as e:
            print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [Telegram Digest Error] {e}")

    process_youtube_interviews()
    print(f"[{datetime.now()}] Execution finished successfully in {round(duration, 1)}s.")

if __name__ == "__main__":
    main()