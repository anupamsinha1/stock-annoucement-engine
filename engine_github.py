"""
Market Intelligence & Corporate Announcement Screening Engine
=======================================================================================
Architecture & Features:
- State Machine Tracker: Defers all Telegram alerts to the end of the pipeline.
- Cascading Verbosity (20, 40, 60, 1000): Dictates pipeline telemetry via Threshold Gatekeeper.
- Sieve 20 (Flash Lite): Administrative Noise Gatekeeper.
- Sieve 40 (Local LLM): Local Fluff/Pre-Score Gatekeeper (4.5k chars extraction).
- Sieve 60 (Claude & Gemini): Deep Dive Consensus (Full 50k chars pdfplumber payload).
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

# Set the default local AI model based on the environment chosen above
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
VERBOSITY_LEVEL = float(os.getenv("VERBOSITY_LEVEL", "1000.0"))

DEFAULT_CHUNK_SIZE = 50
DEFAULT_YOUTUBE_CHANNEL_ID = "UCb5hMTAFjG5j79V6nL3_YCQ"
DEFAULT_MAX_SIEVE60 = int(os.getenv("MAX_SIEVE60_ITEMS", "0"))
SIEVE_40_MIN_SCORE = int(os.getenv("SIEVE_40_MIN_SCORE", "5"))

# Parse command line arguments (like python script.py --ignore-cache)
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
    try: return psycopg2.connect(SUPABASE_URL)
    except Exception as e:
        print(f"[Database Error] Connection failed: {e}")
        return None

def filter_unprocessed_announcements(filings):
    """[DECISION] Queries the DB. If the filing ID already exists, we drop it to avoid re-evaluating old news."""
    if IGNORE_CACHE: return filings
    conn = get_db_connection()
    if not conn: return filings
    try:
        cursor = conn.cursor()
        attachments = [item['id'] for item in filings if item.get('id')]
        if not attachments:
            conn.close(); return []

        # Create a dynamic SQL query based on how many attachments we have
        query = f"SELECT bse_attachment_name FROM bse_announcements WHERE bse_attachment_name IN ({','.join(['%s'] * len(attachments))})"
        cursor.execute(query, tuple(attachments))
        existing = {row[0] for row in cursor.fetchall()}
        conn.close()

        # Keep only the items that were NOT found in the database
        return [item for item in filings if item['id'] not in existing]
    except Exception as e:
        print(f"[Database Error] Deduplication query failed: {e}")
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
            if item['id'] not in existing['all_ids']: existing['all_ids'].append(item['id'])
            if item['headline'] not in existing['headline']: existing['headline'] += f" | {item['headline']}"
            if item['link'] not in existing['all_links']: existing['all_links'].append(item['link'])
    return list(grouped.values())

def log_announcements_batch(decisions_list):
    """[STATE UPDATE] Saves basic triage decisions (HIT/IGNORE) so we don't process them again for 7 days."""
    if not decisions_list: return
    conn = get_db_connection()
    if not conn: return
    try:
        cursor = conn.cursor()
        query = """INSERT INTO bse_announcements (bse_attachment_name, company_name, headline, ai_decision) 
                   VALUES (%s, %s, %s, %s) ON CONFLICT (bse_attachment_name) DO UPDATE SET ai_decision = EXCLUDED.ai_decision"""
        cursor.executemany(query, decisions_list)
        conn.commit(); conn.close()
    except Exception: pass

def log_permanent_ledger(item, market_data):
    """[STATE UPDATE] Saves the massive, deep-dive AI analysis into our permanent historical tracker."""
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
            item.get('headline', ''), item.get('sieve20_reason', 'Actionable corporate action'),
            market_data.get('price', 0.0), market_data.get('market_cap_cr', 0.0), market_data.get('vol_multiple', 1.0),
            market_data.get('above_50dma', False), market_data.get('above_200dma', False),
            item.get('final_score', 1), item.get('status', 'NEUTRAL_MIX'), item.get('high_conviction', False),
            item.get('sieve60_claude_score'), item.get('sieve60_gemini_score'),
            item.get('sieve60_claude_company'), item.get('sieve60_gemini_company'),
            item.get('sieve40_summary', ''), item.get('sieve60_claude_analysis', ''), item.get('sieve60_gemini_analysis', '')
        ))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"[Database Error] Comprehensive ledger logging failed: {e}")

# ---------------------------------------------------------------------------
# 3. METRICS & EXTRACTION ROUTER
# ---------------------------------------------------------------------------
def fetch_market_metrics(scrip_code, exchange):
    """[CALCULATION] Fetches live stock price, 20-day volume surge, and moving averages using Yahoo Finance."""
    default_payload = {"price": 0.0, "vol_multiple": 1.0, "above_50dma": False, "above_200dma": False, "dist_52w_high": 0.0, "market_cap_cr": 0.0}
    if not scrip_code: return default_payload
    try:
        print(f" [Metrics] Fetching market data for {exchange}:{scrip_code}...")
        ticker = f"{scrip_code}.NS" if exchange == "NSE" else f"{scrip_code}.BO"
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty or len(hist) < 20: return default_payload

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
        return {"price": current_price, "vol_multiple": vol_multiple, "above_50dma": bool(current_price > dma_50), "above_200dma": bool(current_price > dma_200), "dist_52w_high": dist_52w_high, "market_cap_cr": market_cap_cr}
    except Exception as e:
        print(f" [Metrics Error] Failed to fetch data for {scrip_code}: {e}")
        return default_payload

def fetch_valuepickr_sentiment(company_name):
    """[EXTERNAL API] Scrapes the ValuePickr forum to see if retailers are discussing this stock."""
    try:
        clean_name = company_name.split()[0].replace("Ltd", "").replace("Limited", "").strip()
        print(f" [Scuttlebutt] Searching ValuePickr forum for '{clean_name}'...")
        res = requests.get(VALUEPICKR_API_URL.format(term=clean_name), headers=DEFAULT_HEADERS, timeout=5)

        if res.status_code != 200:
            return "No active forum discussion found."

        posts = res.json().get('posts', [])
        if posts:
            print(f" [Scuttlebutt] Found {len(posts)} recent posts for {clean_name}.")
            return " ".join([p.get('blurb', '') for p in posts[:5]])[:1500]
        else:
            return "No active forum discussion found."
    except Exception as e:
        print(f" [Scuttlebutt Error] ValuePickr search failed: {e}")
        return "Forum search bypassed."

def extract_text_from_pdf_url(pdf_url, headline):
    """
    [DECISION: DYNAMIC ROUTER]
    If the headline mentions financials/earnings, we fire up pdfplumber to extract 50,000 characters and 15 pages.
    Otherwise, we use lightweight pypdf to just grab the first 10,000 characters to keep things fast.
    """
    if not pdf_url or not pdf_url.startswith("http"): return "No valid PDF URL."
    headline_lower = headline.lower()
    financial_keywords = ["financial result", "outcome of board meeting", "earnings", "annual report", "financial statement"]
    is_heavy_financial = any(kw in headline_lower for kw in financial_keywords)

    max_pages, max_chars = (15, 50000) if is_heavy_financial else (4, 10000)

    if is_heavy_financial:
        print(f" [PDF Router] Financial terms detected in headline. Using heavy pdfplumber extractor (Limit: {max_chars} chars)...")
    else:
        print(f" [PDF Router] Standard filing detected. Using fast pypdf extractor (Limit: {max_chars} chars)...")

    try:
        res = requests.get(pdf_url, headers=DEFAULT_HEADERS, timeout=15)
        if res.status_code == 200:
            with io.BytesIO(res.content) as pdf_buffer:
                if is_heavy_financial and pdfplumber:
                    extracted_pages = []
                    with pdfplumber.open(pdf_buffer) as pdf:
                        for i, page in enumerate(pdf.pages):
                            if i >= max_pages: break
                            # layout=True keeps financial tables in their correct columns
                            text = page.extract_text(layout=True) or page.extract_text()
                            if text: extracted_pages.append(text)
                    raw_text = "\n".join(extracted_pages)
                else:
                    reader = PdfReader(pdf_buffer)
                    raw_text = "\n".join([page.extract_text() for page in reader.pages[:max_pages] if page.extract_text()])

                # Clean up boilerplate disclaimers
                text = re.sub(r'(?i)(?:\bDisclaimer\b|CAREEDGE RATINGS DISCLAIMS).*$', '', raw_text, flags=re.DOTALL)
                clean_text = re.sub(r'[ \t]+', ' ', re.sub(r'\n{2,}', '\n', re.sub(r'[^\x00-\x7F₹]+', '', text))).strip()
                return clean_text[:max_chars] if clean_text else "Unextractable text."
        return f"Failed to download PDF ({res.status_code})"
    except Exception as e: return f"PDF error: {e}"

def extract_score(text, label):
    """[PARSER] Helper function to find a score like 'Catalyst Score: 8/10' inside AI text."""
    if not text: return None
    for line in text.splitlines():
        if label.lower() in line.lower():
            match = re.search(r'(\b10|[1-9])\s*/\s*10', line)
            if match: return int(match.group(1))
            digits = re.findall(r'\b(10|[1-9])\b', line)
            if digits: return int(digits[0])
    return None

# ---------------------------------------------------------------------------
# 4. SIEVES 20, 40, 60 LOGIC (STATE TRACKERS)
# ---------------------------------------------------------------------------
def run_sieve20_batch(announcements, master_results):
    """
    [SIEVE 20] Reads just the headline of 50 filings at once using fast Gemini Flash.
    Any routine noise (newspaper clippings, lost shares) gets rejected immediately.
    """
    if not announcements or not gemini_client: return []
    hits = []

    for i in range(0, len(announcements), DEFAULT_CHUNK_SIZE):
        chunk = announcements[i:i + DEFAULT_CHUNK_SIZE]
        items_payload = [{"index": idx, "exchange": a['exchange'], "scrip": a['scrip'], "company": a['company'], "headline": a['headline']} for idx, a in enumerate(chunk)]

        prompt = f"""You are an objective exchange filing intake filter. Categorize EACH announcement as either "HIT" or "REJECT".
        HIGH-MATERIALITY CATALYSTS (HIT): Financial Results, Commercial Orders, Fundraisings/M&A, Rating Upgrades, Capex, Deleveraging, FDA Clearances.
        CRITICAL EXCLUSIONS (REJECT): Loss of shares, trading window closures, meeting intimations, shareholding patterns, ESOPs, newspaper clippings.
        Respond with ONLY a valid JSON array. Announcements: {json.dumps(items_payload, indent=2)}"""

        for attempt in range(3):
            try:
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
                            print(f" [SIEVE 20 HIT] {ann_item['company']} | {reason}")
                        else:
                            # [STATE UPDATE] Document died at Sieve 20. Mark terminal stage and save to master list.
                            ann_item["terminal_stage"] = 20
                            ann_item["status"] = "REJECTED_SIEVE20"
                            master_results.append(ann_item)
                            print(f" [SIEVE 20 REJECT] {ann_item['company']} | {reason}")
                break
            except Exception as e:
                if "503" in str(e) or "429" in str(e): time.sleep(2 ** attempt); continue
                print(f"[Sieve 20 Error] Chunk failed: {e}. Defaulting to HIT.")
                hits.extend(chunk); break
    return hits

def run_sieve40_extraction(item):
    """
    [SIEVE 40] Downloads the PDF, extracts text, and uses the local model to score it.
    If it's marketing fluff disguised as a contract, the local model will score it low and kill it here.
    """
    print(f" [SIEVE 40] Downloading and parsing PDF for {item['company']}...")
    text = extract_text_from_pdf_url(item['all_links'][0] if item.get('all_links') else item.get('link', ''), item['headline'])
    item['raw_pdf_text'] = text

    # [DECISION] We only pass the first 4,500 chars to local model to save CPU cycles.
    if not text or len(text) < 80:
        print(f" [SIEVE 40] No readable text found. Bypassing extraction.")
        item['sieve40_summary'], item['sieve40_score'] = "No local extraction.", 5
        return item

    prompt = f"""Extract key facts and assign a PreScore (1 to 10).
    Score 1-4: Trade expo, generic PR, minor updates.
    Score 5-6: Small/routine purchase orders, incremental progress.
    Score 7-10: Hard confirmed wins (>50Cr), net-debt reduction, major capex, strong financial beats.
    Document Text: {item['headline']}\n{text[:4500]}\nOutput format:\nSummary: [150 words]\nValue: [Exact value]\nClient: [Entity]\nPreScore: [Integer 1-10]"""

    try:
        print(f" [SIEVE 40] Triggering local {OLLAMA_MODEL} model for {item['company']}...")
        res = requests.post(OLLAMA_API_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}}, timeout=90)
        if res.status_code == 200:
            extracted = res.json().get("response", "").strip()
            score_match = re.search(r'PreScore.*?(\b10|[1-9])\b', extracted, re.IGNORECASE)
            item['sieve40_score'] = int(score_match.group(1)) if score_match else 5
            item['sieve40_summary'] = extracted
            return item
    except Exception as e:
        print(f" [SIEVE 40 Warning] Local Ollama failed: {e}")

    item['sieve40_summary'], item['sieve40_score'] = "Local inference failed.", 5
    return item

def build_sieve60_prompt(item, market_data, forum_text):
    return f"""Assess how strongly this event will impact future earnings power. Disregard market cap (evaluate proportional impact).
    Company: {item['company']} | Price: {market_data['price']} | 20D Vol: {market_data['vol_multiple']}x | 52W High Dist: {market_data['dist_52w_high']}%
    Headline: {item['headline']} | Forum Context: {forum_text}
    Pre-Screen: {item.get('sieve40_summary','')}
    Raw Document: {item.get('raw_pdf_text','')}
    OUTPUT FORMAT:
    Reasoning: <2-3 sentences on commercial impact>
    Catalyst Score: <Integer 1-10>
    Company Quality Score: <Integer 1-10>"""

def run_staggered_sieve60_workers(candidates, master_results):
    """
    [SIEVE 60] The heavy-lifters (Claude and Gemini Pro).
    Uses a priority queue to hand off filings between models.
    """
    if not candidates: return
    counter = itertools.count()
    claude_queue, gemini_queue = queue.PriorityQueue(), queue.PriorityQueue()
    stop_event = threading.Event()
    total_items, completed_count = len(candidates), 0

    # [ROUTING] Assign half the filings to Claude first, and half to Gemini first.
    for idx, hit in enumerate(candidates):
        if idx % 2 == 0: claude_queue.put((20, next(counter), {'item': hit, 'stage': 1}))
        else: gemini_queue.put((20, next(counter), {'item': hit, 'stage': 1}))

    def extract_scores(text):
        c_m = re.search(r'Catalyst Score.*?(\b10|[1-9])\b', text, re.IGNORECASE)
        q_m = re.search(r'Company Quality Score.*?(\b10|[1-9])\b', text, re.IGNORECASE)
        return int(c_m.group(1)) if c_m else 1, int(q_m.group(1)) if q_m else 5

    def worker_logic(queue_in, queue_out, model_name, client_func):
        nonlocal completed_count
        while not stop_event.is_set():
            try: _, _, task = queue_in.get(timeout=1.0)
            except queue.Empty:
                if completed_count >= total_items: break
                continue

            item, stage = task['item'], task['stage']
            market_data = item.get('market_data') or fetch_market_metrics(item['scrip'], item['exchange'])
            forum_data = item.get('forum_data') or fetch_valuepickr_sentiment(item['company'])
            item.update({'market_data': market_data, 'forum_data': forum_data})

            print(f"\n[{model_name.upper()} Worker - Stage {stage}] Analyzing {item['company']}...")
            output = client_func(build_sieve60_prompt(item, market_data, forum_data))
            cat_score, comp_score = extract_scores(output)

            item[f'sieve60_{model_name}_score'] = cat_score
            item[f'sieve60_{model_name}_company'] = comp_score
            item[f'sieve60_{model_name}_analysis'] = output

            if stage == 1:
                # [DECISION] If the first model scores it >= 5, pass to the second model to get a consensus.
                if cat_score >= 5:
                    print(f" -> [{model_name.upper()}] Score {cat_score}/10 >= 5. Handing off to Stage 2.")
                    queue_out.put((20 - cat_score, next(counter), {'item': item, 'stage': 2}))
                else:
                    # [STATE UPDATE] First model hated it. Kill it early to save API tokens.
                    print(f" -> [{model_name.upper()}] Early Exit! Score {cat_score}/10 < 5. Terminating.")
                    item['terminal_stage'] = 60
                    item['status'] = "SINGLE_MODEL_IGNORE"
                    item['final_score'] = cat_score
                    with master_list_lock:
                        master_results.append(item)
                        completed_count += 1
            elif stage == 2:
                # [DECISION] Both models have scored it. Calculate the consensus average.
                print(f" -> [{model_name.upper()}] Stage 2 Complete. Finalizing consensus.")
                c_cat, g_cat = item.get('sieve60_claude_score', 1), item.get('sieve60_gemini_score', 1)
                is_divergent = (abs(c_cat - g_cat) >= 4 or (c_cat >= 7 and g_cat <= 4) or (g_cat >= 7 and c_cat <= 4))

                if is_divergent: status = "MODEL_DIVERGENCE"
                elif c_cat >= 7 and g_cat >= 7: status = "CONSENSUS_HIT"
                elif c_cat <= 4 and g_cat <= 4: status = "CONSENSUS_IGNORE"
                else: status = "NEUTRAL_MIX"

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
        t_c.start(); t_g.start()
    for t in threads: t.join()


# ---------------------------------------------------------------------------
# 5. MASTER DISPATCHER (GATEKEEPER)
# ---------------------------------------------------------------------------
def build_html_telegram_message(item):
    """Constructs the heavily formatted Telegram message based on which stage the document died at."""
    stage, status = item.get('terminal_stage'), item.get('status', 'N/A')
    market_data = item.get('market_data', fetch_market_metrics(item['scrip'], item['exchange']))

    # Set the headline banner based on where it stopped
    if stage == 20: banner = "🗑️ <b>FILTERED: SIEVE 20 REJECT</b>"
    elif stage == 40: banner = "🚫 <b>FILTERED: SIEVE 40 REJECT</b>"
    elif status == "SINGLE_MODEL_IGNORE": banner = "🚫 <b>FILTERED: SIEVE 60 LOW CONVICTION</b>"
    elif status == "MODEL_DIVERGENCE": banner = "⚠️ <b>MODEL DIVERGENCE DETECTED</b>"
    elif item.get('high_conviction'): banner = "🚨 <b>HIGH CONVICTION CATALYST CONCURRENCE</b>"
    else: banner = "📢 <b>CORPORATE ACTION RE-RATING CATALYST</b>"

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

    # Cascading Trail: Always append the history of what prior sieves thought
    if item.get('sieve20_reason'):
        msg += f"\n🔍 <b>Sieve 20 Rationale:</b>\n{item['sieve20_reason']}\n"
    if stage >= 40 and item.get('sieve40_summary'):
        msg += f"\n🤖 <b>Sieve 40 Score ({item.get('sieve40_score', 'N/A')}/10):</b>\n{item['sieve40_summary'][:600]}...\n"
    if stage == 60:
        if item.get('sieve60_claude_analysis'): msg += f"\n🧠 <b>Claude (Sieve 60):</b>\n{item['sieve60_claude_analysis']}\n"
        if item.get('sieve60_gemini_analysis'): msg += f"\n🤖 <b>Gemini (Sieve 60):</b>\n{item['sieve60_gemini_analysis']}\n"

    # HTML Links
    links = " | ".join([f'<a href="{url}">PDF {i+1}</a>' for i, url in enumerate(item.get('all_links', []))])
    screener = f"https://www.screener.in/company/{item['scrip']}/"
    tv = f"https://in.tradingview.com/chart/?symbol={'NSE' if item['exchange'] == 'NSE' else 'BSE'}:{item['scrip']}"
    msg += f"\n🔗 <b>Quick Links:</b> <a href=\"{screener}\">Screener.in</a> | <a href=\"{tv}\">TradingView</a> | {links}"
    return msg

def dispatch_deferred_alerts(pipeline_results):
    """
    [DECISION: VERBOSITY ROUTER]
    Loops through every single filing processed today and decides whether to alert you
    based on the VERBOSITY_LEVEL constant.
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
            should_send = True  # Send absolutely everything
        elif VERBOSITY_LEVEL <= 40 and stage >= 40:
            should_send = True  # Ignore Sieve 20 rejects
        elif VERBOSITY_LEVEL <= 60 and stage >= 60:
            should_send = True  # Ignore Sieve 20 and 40 rejects
        elif VERBOSITY_LEVEL >= 1000:
            # Production max: Only send if it fully completed Sieve 60 AND scored 6 or higher
            if stage == 60 and status not in ["SINGLE_MODEL_IGNORE", "REJECTED_SIEVE20", "REJECTED_SIEVE40"] and final_score >= 6:
                should_send = True

        if should_send and TELEGRAM_BOT_TOKEN:
            try:
                msg = build_html_telegram_message(item)
                requests.post(TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN), json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=8)
                time.sleep(0.5) # Anti-spam buffer
            except Exception as e: print(f"Telegram dispatch failed for {item['company']}: {e}")

        # Ledger & DB Caching for all processed items
        if stage >= 60: log_permanent_ledger(item, item.get('market_data', {}))
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
                print(f" [YouTube Hit] Catalyst found in: {entry.title}")
                if VERBOSITY_LEVEL <= 1000:  # Always send TV hits
                    requests.post(TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN), json={"chat_id": TELEGRAM_CHAT_ID, "text": f"📺 <b>MANAGEMENT INTERVIEW CATALYST</b>\n\n<b>Title:</b> {entry.title}\n🔗 <a href=\"{entry.link}\">Watch Interview</a>", "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=8)
    except Exception as e:
        print(f"[YouTube Scanner Error] {e}")

def fetch_live_nse_filings():
    """[EXTERNAL API] Hits the NSE API using a Session to preserve cookies and bypass basic bot protection."""
    print(" [NSE] Fetching latest announcements from NSE India...")
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    standardized_filings = []
    try:
        # Step 1: Hit homepage to acquire required cookies
        session.get(NSE_BASE_URL, timeout=10)
        # Step 2: Hit actual API endpoint
        resp = session.get(NSE_API_URL, timeout=10)

        if resp.status_code == 200:
            data = resp.json() if isinstance(resp.json(), list) else resp.json().get('data', [])
            for item in data:
                attachment = item.get('attchmntFile') or item.get('attchmntText') or str(item.get('seq_id', ''))
                if not attachment: continue

                pdf_link = item.get('attchmntText', '')
                if pdf_link and not pdf_link.startswith('http'):
                    pdf_link = f"{NSE_BASE_URL}{pdf_link}"

                standardized_filings.append({
                    'id': str(attachment),
                    'company': item.get('sm_name', item.get('symbol', 'Unknown NSE Company')),
                    'scrip': item.get('symbol', ''),
                    'headline': item.get('subject', item.get('desc', '')),
                    'isin': 'N/A',
                    'link': pdf_link,
                    'exchange': 'NSE'
                })
            print(f" [NSE] Successfully fetched {len(standardized_filings)} filings.")
    except Exception as e: print(f"[NSE Ingestion Notice] {e}")
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
                if not table: break  # Reached the end of available filings

                for item in table:
                    if not item.get('ATTACHMENTNAME'): continue
                    raw_isin = item.get('ISIN_CODE', '').strip()
                    standardized_filings.append({
                        'id': item['ATTACHMENTNAME'],
                        'company': item.get('SLONGNAME', 'Unknown Company'),
                        'scrip': str(item.get('SCRIP_CD', '')).strip(),
                        'headline': item.get('NEWSSUB', ''),
                        'isin': raw_isin if raw_isin else 'N/A',
                        'link': BSE_PDF_BASE_URL.format(attachment=item['ATTACHMENTNAME']),
                        'exchange': 'BSE'
                    })
                page += 1
                time.sleep(0.2) # Politeness delay
            else:
                break
        except Exception as e:
            print(f"[BSE Ingestion Notice on Page {page}] {e}"); break
    print(f" [BSE] Successfully fetched {len(standardized_filings)} filings across {page - 1} pages.")
    return standardized_filings


# ---------------------------------------------------------------------------
# 7. MAIN PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    print(f"[{datetime.now()}] Initializing Pipeline (Verbosity Threshold: {VERBOSITY_LEVEL})...")

    unified = fetch_live_bse_filings(max_pages=args.max_pages) + fetch_live_nse_filings()
    unprocessed = filter_unprocessed_announcements(unified)
    if not unprocessed:
        print("No new filings found in this scan cycle.")
        return

    grouped = group_filings_by_company(unprocessed)
    print(f"Consolidated into {len(grouped)} distinct company events.")

    pipeline_results = []

    # --- SIEVE 20 ---
    print(f"\n[SIEVE 20] Broad-net noise filtering...")
    sieve20_hits = run_sieve20_batch(grouped, pipeline_results)

    # --- SIEVE 40 ---
    sieve40_hits = []
    if sieve20_hits:
        print(f"\n[SIEVE 40] Executing Dynamic Extraction & Pre-Scoring on {len(sieve20_hits)} hits...")
        for item in sieve20_hits:
            item = run_sieve40_extraction(item)
            score = item.get('sieve40_score', 5)
            print(f" -> [{item['company']}] Sieve 40 Score: {score}/10")
            if score >= SIEVE_40_MIN_SCORE:
                sieve40_hits.append(item)
            else:
                item['terminal_stage'] = 40
                item['status'] = "REJECTED_SIEVE40"
                pipeline_results.append(item)

    # --- SIEVE 60 ---
    if sieve40_hits:
        print(f"\n[SIEVE 60] Handing off to Claude & Gemini for deep reasoning...")
        candidates = sieve40_hits[:args.max_sieve60] if args.max_sieve60 > 0 else sieve40_hits
        run_staggered_sieve60_workers(candidates, pipeline_results)

    # --- DISPATCH ---
    dispatch_deferred_alerts(pipeline_results)

    # --- GLOBAL SUMMARY DIGEST ---
    duration = time.time() - start_time

    # Calculate exactly how many passed the final Sieve 60 successfully
    sieve60_passed = [
        item for item in pipeline_results
        if item.get('terminal_stage') == 60 and item.get('final_score', 0) >= 6
    ]

    # Break down the Sieve 20 Rejections for the telemetry report
    sieve20_rejects = [item for item in pipeline_results if item.get('terminal_stage') == 20]
    sast_count = sum(1 for r in sieve20_rejects if any(k in r.get('sieve20_reason', '').lower() for k in ['sast', 'pit', 'insider', 'transfer']))
    admin_count = sum(1 for r in sieve20_rejects if any(k in r.get('sieve20_reason', '').lower() for k in ['certificate', 'meeting', 'governance', 'window', 'newspaper']))
    other_count = len(sieve20_rejects) - (sast_count + admin_count)

    # Print a clean, formatted summary to the GitHub Actions Console
    print(f"\n=======================================================")
    print(f"📊 PIPELINE EXECUTION SUMMARY")
    print(f"=======================================================")
    print(f"• Total Filings Ingested : {len(unified)}")
    print(f"• Unique Company Events  : {len(grouped)}")
    print(f"• Passed Sieve 20 (Flash): {len(sieve20_hits)}")
    print(f"• Passed Sieve 40 (Local): {len(sieve40_hits)}")
    print(f"• Passed Sieve 60 (Final): {len(sieve60_passed)}")
    print(f"• Execution Latency      : {round(duration, 1)}s")
    print(f"=======================================================\n")

    # Build the rich Telegram Digest
    mode_text = "🔄 <b>Manual Refresh</b>" if IGNORE_CACHE else "⏰ <b>Scheduled GitHub Action Scan</b>"

    digest_msg = (
        f"📊 <b>EXCHANGE SCAN COMPLETE</b>\n"
        f"• <b>Mode:</b> {mode_text}\n"
        f"• <b>Verbosity Level:</b> {VERBOSITY_LEVEL}\n"
        f"• <b>Total Screened:</b> {len(grouped)} entities\n"
        f"• <b>Passed Sieve 20:</b> {len(sieve20_hits)}\n"
        f"• <b>Passed Sieve 40:</b> {len(sieve40_hits)}\n"
        f"• <b>Final Sieve 60 Hits:</b> {len(sieve60_passed)}\n"
        f"• <b>Execution Latency:</b> {round(duration, 1)}s\n\n"
        f"🚫 <b>Noise Filter Breakdown (Sieve 20):</b>\n"
        f"• SAST / Insider Transfers: {sast_count}\n"
        f"• Admin / Board Meetings: {admin_count}\n"
        f"• Routine Disclosures: {max(0, other_count)}"
    )

    # Dispatch the final summary to Telegram
    if TELEGRAM_BOT_TOKEN:
        try:
            requests.post(TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN), json={"chat_id": TELEGRAM_CHAT_ID, "text": digest_msg, "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=8)
        except Exception as e:
            print(f"[Telegram Digest Error] {e}")

    process_youtube_interviews()
    print(f"[{datetime.now()}] Execution finished successfully in {round(duration, 1)}s.")

if __name__ == "__main__":
    main()