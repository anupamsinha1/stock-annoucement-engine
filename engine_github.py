"""
Market Intelligence & Corporate Announcement Screening Engine (GitHub Actions Edition)
=======================================================================================
Target Environment:
- Designed specifically for automated execution inside GitHub Actions Ubuntu runners.
- Leverages GitHub's public runner hardware (4 vCPUs / 16 GB RAM) to execute local
  quantized LLM inference ('qwen2.5:7b') via background Ollama service at $0 compute cost.

Architecture Overview:
1. Ingestion: Live BSE & NSE corporate filings APIs + YouTube TV interview feeds.
2. Deduplication & Grouping: Supabase PostgreSQL 7-day cache; groups simultaneous company filings.
3. Tier 1 Sieve (Gemini 3.5 Flash Lite): Wide-net commercial catalyst filter.
4. Document Extraction: In-memory PDF streaming, boilerplate stripping, and cleaning.
5. Tier 1.5 Sieve (Local Qwen 2.5 7B via Ollama): On-runner extraction & strict pre-scoring.
6. Tier 2 Staggered Sieve: 8-Worker Pool (4 Claude Sonnet 5 + 4 Gemini 3.1 Pro Preview)
   with reasoning-first prompt architecture and priority hand-off for scores >= 5.
7. Dispatcher & Ledger: Real-time Telegram alerts, scan summary digest, and permanent audit ledger.
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
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# 0. LOGGING & SDK WARNING SUPPRESSION
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# Suppress Google GenAI automatic function calling (AFC) warning noise
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# 1. CONFIGURATION & ENVIRONMENT SETUP
# ---------------------------------------------------------------------------
load_dotenv()

# API Endpoints & Base URLs
BSE_API_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?pageno={page}&strCat=-1&strPrevDate=&strScrip=&strSearch=P&strToDate=&strType=C"
BSE_PDF_BASE_URL = "https://www.bseindia.com/xml-data/corpfiling/AttachLive/{attachment}"
NSE_BASE_URL = "https://www.nseindia.com"
NSE_API_URL = "https://www.nseindia.com/api/corporate-announcements?index=equities"
VALUEPICKR_API_URL = "https://forum.valuepickr.com/search/query.json?term={term}"
YOUTUBE_RSS_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Global Headers
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

# Extraction Limits
DEFAULT_CHUNK_SIZE = 50
MAX_PDF_PAGES = 4
MAX_PDF_CHARS = 10000
DEFAULT_YOUTUBE_CHANNEL_ID = "UCb5hMTAFjG5j79V6nL3_YCQ"

# CLI & Environment Flags
DEFAULT_MAX_SIEVE2 = int(os.getenv("MAX_SIEVE2_ITEMS", "0"))
SIEVE_1_5_MIN_SCORE = int(os.getenv("SIEVE_1_5_MIN_SCORE", "5"))
PUBLISH_WHEN_SIEVE_PASSED = float(os.getenv("PUBLISH_WHEN_SIEVE_PASSED", "2.0"))
ALERT_ON_SINGLE_MODEL_IGNORE = os.getenv("ALERT_ON_SINGLE_MODEL_IGNORE", "false").lower() in ("true", "1", "yes")

parser = argparse.ArgumentParser(description="Market Intelligence Screening Engine (GitHub Actions Edition)")
parser.add_argument("--ignore-cache", "-f", action="store_true", help="Bypass Supabase cache and re-evaluate all filings")
parser.add_argument("--max-pages", type=int, default=100, help="Maximum BSE announcement pages to fetch")
parser.add_argument("--max-sieve2", type=int, default=DEFAULT_MAX_SIEVE2, help="Max candidates sent to Sieve 2 (0 for all)")
args, _ = parser.parse_known_args()

IGNORE_CACHE = args.ignore_cache or os.getenv("IGNORE_CACHE", "false").lower() in ("true", "1", "yes")

# Local Model (Sieve 1.5) - Defaulted to Qwen 2.5 7B on GitHub Runners
USE_LOCAL_EXTRACTOR = os.getenv("USE_LOCAL_EXTRACTOR", "true").lower() in ("true", "1", "yes")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", DEFAULT_OLLAMA_URL)

# Cloud Model Credentials & Endpoints
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_PAID") or os.getenv("GEMINI_API_KEY_FREE")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SUPABASE_URL = os.getenv("SUPABASE_URL")

# Cloud Models (Locked to calibrated versions)
TIER1_MODEL = os.getenv("GEMINI_TIER1_MODEL", "gemini-3.5-flash-lite")
TIER2_GEMINI_MODEL = os.getenv("GEMINI_TIER2_MODEL", "gemini-3.1-pro-preview")
TIER2_CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")

# Initialize SDK Clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
claude_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


# ---------------------------------------------------------------------------
# 2. SUPABASE DATABASE LAYER (CACHE & PERMANENT LEDGER)
# ---------------------------------------------------------------------------
def get_db_connection():
    """Establishes connection to Supabase PostgreSQL."""
    if not SUPABASE_URL:
        return None
    try:
        return psycopg2.connect(SUPABASE_URL)
    except Exception as e:
        print(f"[Database Error] Connection failed: {e}")
        return None


def filter_unprocessed_announcements(filings):
    """Filters out already-evaluated filings using the Supabase 7-day cache."""
    if IGNORE_CACHE:
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

        format_strings = ','.join(['%s'] * len(attachments))
        query = f"SELECT bse_attachment_name FROM bse_announcements WHERE bse_attachment_name IN ({format_strings})"
        cursor.execute(query, tuple(attachments))

        existing = {row[0] for row in cursor.fetchall()}
        conn.close()
        return [item for item in filings if item['id'] not in existing]
    except Exception as e:
        print(f"[Database Error] Deduplication query failed: {e}")
        return filings


def group_filings_by_company(filings):
    """Consolidates simultaneous filings for the same company into a single unified event."""
    grouped = {}
    for item in filings:
        key = (item['exchange'], item['scrip'])
        if key not in grouped:
            grouped[key] = {
                'id': item['id'],
                'all_ids': [item['id']],
                'company': item['company'],
                'scrip': item['scrip'],
                'headline': item['headline'],
                'isin': item['isin'],
                'all_links': [item['link']],
                'exchange': item['exchange']
            }
        else:
            existing = grouped[key]
            if item['id'] not in existing['all_ids']:
                existing['all_ids'].append(item['id'])
            if item['headline'] not in existing['headline']:
                existing['headline'] += f" | {item['headline']}"
            if item['link'] not in existing['all_links']:
                existing['all_links'].append(item['link'])

    return list(grouped.values())


def log_announcements_batch(decisions_list):
    """Bulk inserts triage decisions to prevent connection pool exhaustion."""
    if not decisions_list:
        return
    conn = get_db_connection()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO bse_announcements (bse_attachment_name, company_name, headline, ai_decision) 
            VALUES (%s, %s, %s, %s) 
            ON CONFLICT (bse_attachment_name) DO UPDATE SET ai_decision = EXCLUDED.ai_decision
        """
        cursor.executemany(query, decisions_list)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Database Error] Batch cache insertion failed: {e}")


def log_permanent_ledger(item, market_data, evals, audit, extracted_text):
    """Permanently logs the complete market and AI context for multi-horizon backtesting."""
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
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (bse_attachment_name) DO UPDATE SET
                final_score = EXCLUDED.final_score,
                consensus_status = EXCLUDED.consensus_status,
                claude_analysis = EXCLUDED.claude_analysis,
                gemini_analysis = EXCLUDED.gemini_analysis;
        """
        cursor.execute(
            query,
            (
                item['id'],
                item['company'],
                item['scrip'],
                item['exchange'],
                item.get('isin', 'N/A'),
                item.get('headline', ''),
                item.get('catalyst_reason', 'Actionable corporate action'),
                market_data.get('price', 0.0),
                market_data.get('market_cap_cr', 0.0),
                market_data.get('vol_multiple', 1.0),
                market_data.get('above_50dma', False),
                market_data.get('above_200dma', False),
                audit.get('final_score', 1),
                audit.get('consensus_status', 'NEUTRAL_MIX'),
                audit.get('high_conviction', False),
                audit.get('claude_catalyst_score'),
                audit.get('gemini_catalyst_score'),
                audit.get('claude_company_score'),
                audit.get('gemini_company_score'),
                extracted_text,
                evals.get('claude_analysis', ''),
                evals.get('gemini_analysis', '')
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Database Error] Comprehensive ledger logging failed: {e}")


# ---------------------------------------------------------------------------
# 3. QUANT METRICS & SCUTTLEBUTT CONTEXT
# ---------------------------------------------------------------------------
def fetch_market_metrics(scrip_code, exchange):
    """Fetches live price, 20D volume surge multiple, and moving averages via Yahoo Finance."""
    default_payload = {
        "price": 0.0,
        "vol_multiple": 1.0,
        "above_50dma": False,
        "above_200dma": False,
        "price_feed_sync": True,
        "market_cap_cr": 0.0
    }
    if not scrip_code:
        return default_payload

    try:
        ticker = f"{scrip_code}.NS" if exchange == "NSE" else f"{scrip_code}.BO"
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")

        if hist.empty or len(hist) < 20:
            return default_payload

        current_price = round(float(hist['Close'].iloc[-1]), 2)
        today_volume = float(hist['Volume'].iloc[-1])
        avg_20_volume = float(hist['Volume'].iloc[-21:-1].mean()) if len(hist) >= 21 else today_volume

        vol_multiple = round((today_volume / avg_20_volume), 2) if avg_20_volume > 0 else 1.0
        dma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else current_price
        dma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else current_price

        market_cap_cr = 0.0
        try:
            mkt_cap = getattr(stock.fast_info, 'market_cap', 0)
            if mkt_cap:
                market_cap_cr = round(mkt_cap / 1e7, 2)
        except Exception:
            pass

        return {
            "price": current_price,
            "vol_multiple": vol_multiple,
            "above_50dma": current_price > dma_50,
            "above_200dma": current_price > dma_200,
            "price_feed_sync": current_price == 0.0,
            "market_cap_cr": market_cap_cr
        }
    except Exception as e:
        print(f"[Market Data Notice] Scrip {scrip_code} ({exchange}): {e}")
        return default_payload


def fetch_valuepickr_sentiment(company_name):
    """Queries ValuePickr Discourse API for community discussion and background flags."""
    try:
        clean_name = company_name.split()[0].replace("Ltd", "").strip()
        url = VALUEPICKR_API_URL.format(term=clean_name)
        res = requests.get(url, headers=DEFAULT_HEADERS, timeout=5)
        if res.status_code != 200:
            return "No active forum discussion found."

        posts = res.json().get('posts', [])
        if not posts:
            return "No active forum discussion found."

        combined = " ".join([p.get('blurb', '') for p in posts[:5]])
        return combined[:1500]
    except Exception:
        return "Forum search bypassed."


# ---------------------------------------------------------------------------
# 4. PDF EXTRACTION & SANITIZATION
# ---------------------------------------------------------------------------
def sanitize_filing_text(text):
    """Strips regulatory boilerplate, addresses, and agency disclaimers."""
    if not text:
        return ""
    match = re.search(r'(?i)(?:Sub(?:ject)?\s*:|Ref\s*:)', text)
    if match:
        text = text[match.start():]

    text = re.sub(r'(?i)(?:\bDisclaimer\b|CAREEDGE RATINGS DISCLAIMS|S&P Global Ratings Terms and Conditions).*$', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F₹]+', '', text)
    return text.strip()


def extract_text_from_pdf_url(pdf_url, max_pages=MAX_PDF_PAGES, max_chars=MAX_PDF_CHARS):
    """Downloads and extracts raw text directly from exchange PDF filings."""
    if not pdf_url or not pdf_url.startswith("http"):
        return "No valid PDF URL provided."

    try:
        response = requests.get(pdf_url, headers=DEFAULT_HEADERS, timeout=15)
        if response.status_code == 200:
            with io.BytesIO(response.content) as pdf_buffer:
                reader = PdfReader(pdf_buffer)
                extracted_pages = []
                total_pages = min(len(reader.pages), max_pages)

                for i in range(total_pages):
                    text = reader.pages[i].extract_text()
                    if text:
                        extracted_pages.append(text)

                raw_text = "\n".join(extracted_pages)
                clean_text = sanitize_filing_text(raw_text)
                return clean_text[:max_chars] if clean_text else "PDF contains scanned imagery or unextractable text."
        return f"Failed to download PDF (HTTP {response.status_code})"
    except Exception as e:
        return f"PDF extraction error: {e}"


def extract_score(text, label):
    """Safely extracts integer score (1-10) for a given label in model output."""
    if not text:
        return None
    for line in text.splitlines():
        if label.lower() in line.lower():
            match = re.search(r'(\b10|[1-9])\s*/\s*10', line)
            if match:
                return int(match.group(1))
            digits = re.findall(r'\b(10|[1-9])\b', line)
            if digits:
                return int(digits[0])
    return None


# ---------------------------------------------------------------------------
# 5. SIEVE 1 (FLASH LITE) & SIEVE 1.5 (LOCAL QWEN 2.5 7B)
# ---------------------------------------------------------------------------
def run_tier1_batch_sieve(announcements):
    """
    Sieve 1: Broad-net filter using Gemini Flash Lite.
    Only eliminates unambiguous administrative noise while passing all commercial activity.
    """
    if not announcements or not gemini_client:
        return announcements, []

    hits = []
    rejections = []
    chunk_size = DEFAULT_CHUNK_SIZE

    for i in range(0, len(announcements), chunk_size):
        chunk = announcements[i:i + chunk_size]
        items_payload = [
            {"index": idx, "exchange": ann['exchange'], "scrip": ann['scrip'], "company": ann['company'], "headline": ann['headline']}
            for idx, ann in enumerate(chunk)
        ]

        prompt = f"""
        You are an objective exchange filing intake filter.
        Categorize EACH announcement as either "HIT" or "REJECT".

        STRICT REJECTION RULES (REJECT ONLY IF UNAMBIGUOUS NOISE):
        - Loss of share certificates / duplicate certificate requests (Reg 39(3)).
        - Trading window closure notices for board meetings / financial results.
        - Advance intimations of Board Meeting dates (prior notices).
        - Routine shareholding patterns (Reg 31), Corporate Governance (Reg 27(2)), Secretarial Compliance (Reg 24A).
        - Newspaper publication clippings, routine ESOP allotments.

        BROAD INTAKE RULES (ALWAYS FLAG AS "HIT"):
        - Any commercial order win, contract award, letter of award (LoA), or supply agreement.
        - Financial results, earnings releases, revenue/margin guidance updates.
        - Fundraisings, preferential issues, QIPs, strategic warrant allotments, M&A.
        - Plant capex, commercial production commissioning, plant expansions.
        - Balance sheet deleveraging / debt-free milestones / one-time settlements.
        - Regulatory approvals (USFDA, CDSCO, PLI scheme subsidies, patent grants).
        - Credit rating upgrades or substantial revisions.

        Respond with ONLY a valid JSON array of objects:
        [
          {{"index": 0, "status": "HIT", "reason": "Commercial contract win"}},
          {{"index": 1, "status": "REJECT", "reason": "Trading window closure notice"}}
        ]

        Announcements:
        {json.dumps(items_payload, indent=2)}
        """

        for attempt in range(3):
            try:
                response = gemini_client.models.generate_content(model=TIER1_MODEL, contents=prompt)
                raw_text = response.text.strip().replace("```json", "").replace("```", "")
                parsed_decisions = json.loads(raw_text)

                for dec in parsed_decisions:
                    idx = dec.get("index")
                    if idx is not None and idx < len(chunk):
                        ann_item = chunk[idx]
                        status = dec.get("status", "REJECT").upper()
                        reason = dec.get("reason", "Routine filing")

                        if status == "HIT":
                            ann_item["catalyst_reason"] = reason
                            hits.append(ann_item)
                            print(f" [SIEVE 1 HIT] {ann_item['company']} ({ann_item['exchange']}:{ann_item['scrip']}) | {reason}")
                        else:
                            ann_item["rejection_reason"] = reason
                            rejections.append(ann_item)
                            print(f" [SIEVE 1 REJECT] {ann_item['company']} ({ann_item['exchange']}:{ann_item['scrip']}) | {reason}")
                break
            except Exception as e:
                err_str = str(e).lower()
                if "503" in err_str or "429" in err_str or "quota" in err_str:
                    time.sleep(2 ** attempt)
                    continue
                print(f"[Tier 1 Error] Chunk evaluation failed: {e}. Defaulting chunk to HIT.")
                hits.extend(chunk)
                break

    return hits, rejections


def sieve_1_5_local_qwen_extraction(cleaned_pdf_text, headline):
    """
    Sieve 1.5: Executes local Qwen 2.5 7B extraction and rigorous pre-scoring inside the GitHub Runner.
    Filters out marketing fluff (trade expos, non-binding MoUs) while escalating structural growth.
    """
    if not USE_LOCAL_EXTRACTOR or not cleaned_pdf_text or len(cleaned_pdf_text) < 80:
        return cleaned_pdf_text, 5

    prompt = f"""
    You are a strict financial analyst pre-screening corporate disclosures.
    Extract the key facts and assign an objective PreScore from 1 to 10 based on concrete economic materiality.

    CRITICAL SCORING RULES:
    - Score 1 to 4 (Routine / Fluff): Trade expo participation, generic PR marketing, non-binding MoUs, routine minor updates.
    - Score 5 to 6 (Moderate): Small/routine purchase orders, incremental business progress.
    - Score 7 to 10 (High Impact): Hard confirmed contract wins (>INR 50 Cr+), net-debt reduction, major capacity commissioning, or strong financial beats.

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

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }

    try:
        res = requests.post(OLLAMA_API_URL, json=payload, timeout=90)
        if res.status_code == 200:
            extracted_output = res.json().get("response", "").strip()
            if extracted_output:
                pre_score = extract_score(extracted_output, "PreScore:")
                return extracted_output, (pre_score if pre_score is not None else 5)
    except Exception as e:
        print(f" [Sieve 1.5 Warning] Qwen inference bypassed ({e}). Falling back to raw text.")

    return cleaned_pdf_text, 5


# ---------------------------------------------------------------------------
# 6. TIER 2 (SIEVE 2): UNIFIED REASONING-FIRST CLAUDE & GEMINI DEEP DIVE
# ---------------------------------------------------------------------------
def build_sieve2_prompt(item, market_data, forum_text, extracted_text):
    """
    Unified Reasoning-First Prompt for Claude and Gemini Pro.
    Evaluates proportional economic impact regardless of market cap.
    """
    price_display = f"INR {market_data['price']}" if market_data['price'] > 0 else "Data Feed Sync (Evaluate on strategic merits)"
    mkt_cap_display = f"INR {market_data['market_cap_cr']} Cr" if market_data.get('market_cap_cr', 0) > 0 else "Not specified"

    return f"""You are an institutional equity research analyst evaluating corporate exchange filings.
Assess how strongly this event will impact the company's future earnings power, business trajectory, and institutional market re-rating.

EVALUATION PRINCIPLES:
- Disregard market cap (whether Microcap, SME, Midcap, or Large-cap). Evaluate the PROPORTIONAL impact of the event on the company's business scale.
- Base your analysis directly on the extracted filing details and metrics provided below.
- If Price is listed as 0.00 or "Data Feed Sync", treat this purely as external quote latency.

SCORING BENCHMARKS (1 to 10):
• 1 to 3 (Administrative / Noise): Share certificate losses, trading window notices, routine secretarial compliance.
• 4 to 6 (Moderate Operational Update): Small/routine purchase orders, non-binding MoUs, standard conference participations.
• 7 to 8 (High-Impact Structural Catalyst): Large firm order wins, full balance sheet deleveraging (net debt-free), major capacity commissioning, or substantial earnings beats (>40% YoY).
• 9 to 10 (Transformational Breakout): Landmark multi-year global contracts, explosive earnings surges (>100%), game-changing strategic partnerships, or institutional warrant allotments at premium.

COMPANY DETAILS:
Company: {item['company']} ({item['exchange']}: {item['scrip']} | ISIN: {item['isin']})
Price: {price_display} | 20D Vol: {market_data['vol_multiple']}x | MktCap: {mkt_cap_display}
Headline: {item['headline']}
Flagged Catalyst: {item.get('catalyst_reason', 'Actionable corporate action')}
Forum/Scuttlebutt Context: {forum_text}

==================== EXTRACTED DISCLOSURE METRICS ====================
{extracted_text}
======================================================================

OUTPUT FORMAT (Follow this structure strictly):
Reasoning: <2-3 sentences on proportional commercial impact, execution capability, and market re-rating probability>
Hype / Red Flag Check: <Clean / Warning flags>
Catalyst Score: <Strictly an integer from 1 to 10, e.g., 8/10>
Company Quality Score: <Strictly an integer from 1 to 10 evaluating core business franchise durability>
"""


def evaluate_with_claude(prompt):
    """Executes qualitative reasoning via Claude Sonnet with exponential backoff."""
    if not claude_client:
        return "Claude evaluation skipped: API key missing."

    for attempt in range(3):
        try:
            res = claude_client.messages.create(
                model=TIER2_CLAUDE_MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            text_blocks = [b.text for b in res.content if getattr(b, "type", None) == "text"]
            return "\n".join(text_blocks).strip() if text_blocks else "Claude returned no text."
        except Exception as e:
            err_str = str(e).lower()
            if "503" in err_str or "429" in err_str or "overloaded" in err_str or "rate_limit" in err_str:
                time.sleep(2 ** attempt)
                continue
            return f"Claude analysis error: {e}"
    return "Claude analysis error: API unavailable after retries."


def evaluate_with_gemini(prompt):
    """Executes structured assessment via Gemini Pro with exponential backoff."""
    if not gemini_client:
        return "Gemini Pro evaluation skipped: API key missing."

    for attempt in range(3):
        try:
            res = gemini_client.models.generate_content(
                model=TIER2_GEMINI_MODEL,
                contents=prompt
            )
            return res.text.strip()
        except Exception as e:
            err_str = str(e).lower()
            if "503" in err_str or "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                time.sleep(2 ** attempt)
                continue
            return f"Gemini Pro analysis error: {e}"
    return "Gemini Pro analysis error: API unavailable after retries."


# ---------------------------------------------------------------------------
# 7. STAGGERED PRIORITY QUEUE WORKER POOL & FINALIZATION
# ---------------------------------------------------------------------------
def finalize_dual_evaluation(item, evals, market_data, extracted_text):
    """Calculates consensus scores across both models and dispatches alerts."""
    c_cat = evals.get('claude_catalyst_score', 1)
    g_cat = evals.get('gemini_catalyst_score', 1)
    c_comp = evals.get('claude_company_score', 5)
    g_comp = evals.get('gemini_company_score', 5)

    is_divergent = (abs(c_cat - g_cat) >= 4 or (c_cat >= 7 and g_cat <= 4) or (g_cat >= 7 and c_cat <= 4))

    if is_divergent:
        consensus_status = "MODEL_DIVERGENCE"
    elif c_cat >= 7 and g_cat >= 7:
        consensus_status = "CONSENSUS_HIT"
    elif c_cat <= 4 and g_cat <= 4:
        consensus_status = "CONSENSUS_IGNORE"
    else:
        consensus_status = "NEUTRAL_MIX"

    final_score = round((c_cat + g_cat) / 2)
    is_high_conviction = (final_score >= 8 and market_data['vol_multiple'] >= 2.0 and market_data['above_50dma'])

    audit = {
        'consensus_status': consensus_status,
        'final_score': final_score,
        'high_conviction': is_high_conviction,
        'claude_catalyst_score': c_cat,
        'gemini_catalyst_score': g_cat,
        'claude_company_score': c_comp,
        'gemini_company_score': g_comp,
        'claude_analysis': evals.get('claude_analysis', ''),
        'gemini_analysis': evals.get('gemini_analysis', '')
    }

    if PUBLISH_WHEN_SIEVE_PASSED >= 2.0 and final_score >= 6:
        alert_msg = build_telegram_message(
            company=item['company'], exchange=item['exchange'], scrip_code=item['scrip'],
            isin=item['isin'], market_data=market_data, audit=audit, all_links=item['all_links']
        )
        send_telegram_alert(alert_msg)

    log_permanent_ledger(item, market_data, evals, audit, extracted_text)
    hit_tuples = [(att_id, item['company'], item['headline'], "HIT") for att_id in item['all_ids']]
    log_announcements_batch(hit_tuples)


def finalize_single_model_ignore(item, evals, market_data, source_model, extracted_text):
    """Terminates low-scoring filings early (< 5 on Model A) to save downstream API tokens."""
    score = evals.get(f'{source_model.lower()}_catalyst_score', 1)
    print(f" -> [{source_model.upper()} Gatekeeper] Score {score}/10 < 5. Terminating Sieve 2 early for {item['company']}.")

    audit = {
        'consensus_status': "SINGLE_MODEL_IGNORE",
        'final_score': score,
        'high_conviction': False,
        'claude_catalyst_score': evals.get('claude_catalyst_score'),
        'gemini_catalyst_score': evals.get('gemini_catalyst_score'),
        'claude_company_score': evals.get('claude_company_score', 5),
        'gemini_company_score': evals.get('gemini_company_score', 5),
        'claude_analysis': evals.get('claude_analysis', ''),
        'gemini_analysis': evals.get('gemini_analysis', '')
    }

    if ALERT_ON_SINGLE_MODEL_IGNORE:
        alert_msg = build_telegram_message(
            company=item['company'], exchange=item['exchange'], scrip_code=item['scrip'],
            isin=item['isin'], market_data=market_data, audit=audit, all_links=item['all_links']
        )
        send_telegram_alert(alert_msg)

    ignore_tuples = [(att_id, item['company'], item['headline'], "IGNORE") for att_id in item['all_ids']]
    log_announcements_batch(ignore_tuples)


def run_staggered_sieve2_workers(hits, num_workers=4):
    """
    Shards candidates into 4 Claude + 4 Gemini worker threads with a staggered priority queue.
    Filings scoring >= 5 on stage 1 hand off to the counterpart model with priority (20 - score).
    """
    if not hits:
        return

    counter = itertools.count()
    claude_queue = queue.PriorityQueue()
    gemini_queue = queue.PriorityQueue()
    total_items = len(hits)
    completed_lock = threading.Lock()
    completed_count = 0
    stop_event = threading.Event()

    for idx, hit in enumerate(hits):
        task_payload = {'item': hit, 'stage': 1, 'evals': {}}
        if idx % 2 == 0:
            claude_queue.put((20, next(counter), task_payload))
        else:
            gemini_queue.put((20, next(counter), task_payload))

    def claude_worker(worker_id):
        nonlocal completed_count
        while not stop_event.is_set():
            try:
                priority, seq, task = claude_queue.get(timeout=1.0)
            except queue.Empty:
                with completed_lock:
                    if completed_count >= total_items:
                        break
                continue

            item = task['item']
            stage = task['stage']
            evals = task['evals']

            market_data = task.get('market_data') or fetch_market_metrics(item['scrip'], item['exchange'])
            forum_data = task.get('forum_data') or fetch_valuepickr_sentiment(item['company'])
            extracted_text = task.get('extracted_text') or item.get('extracted_text', '')

            task['market_data'] = market_data
            task['forum_data'] = forum_data
            task['extracted_text'] = extracted_text

            print(f"\n[Claude Worker-{worker_id} Stage {stage}] Analyzing {item['company']} ({item['exchange']}:{item['scrip']})...")
            prompt = build_sieve2_prompt(item, market_data, forum_data, extracted_text)
            claude_output = evaluate_with_claude(prompt)

            cat_score = extract_score(claude_output, "Catalyst Score:")
            comp_score = extract_score(claude_output, "Company Quality Score:")

            cat_score_safe = cat_score if isinstance(cat_score, int) else 1
            comp_score_safe = comp_score if isinstance(comp_score, int) else 5

            evals['claude_catalyst_score'] = cat_score_safe
            evals['claude_company_score'] = comp_score_safe
            evals['claude_analysis'] = claude_output

            if stage == 1:
                if cat_score_safe >= 5:
                    print(f" -> [Claude Worker-{worker_id}] Score {cat_score_safe}/10 >= 5. Handoff to Gemini Queue (Priority {20 - cat_score_safe})...")
                    gemini_queue.put((20 - cat_score_safe, next(counter), {
                        'item': item, 'stage': 2, 'evals': evals, 'market_data': market_data, 'forum_data': forum_data, 'extracted_text': extracted_text
                    }))
                else:
                    finalize_single_model_ignore(item, evals, market_data, "claude", extracted_text)
                    with completed_lock:
                        completed_count += 1
                        if completed_count >= total_items: stop_event.set()
            elif stage == 2:
                finalize_dual_evaluation(item, evals, market_data, extracted_text)
                with completed_lock:
                    completed_count += 1
                    if completed_count >= total_items: stop_event.set()

            claude_queue.task_done()

    def gemini_worker(worker_id):
        nonlocal completed_count
        while not stop_event.is_set():
            try:
                priority, seq, task = gemini_queue.get(timeout=1.0)
            except queue.Empty:
                with completed_lock:
                    if completed_count >= total_items:
                        break
                continue

            item = task['item']
            stage = task['stage']
            evals = task['evals']

            market_data = task.get('market_data') or fetch_market_metrics(item['scrip'], item['exchange'])
            forum_data = task.get('forum_data') or fetch_valuepickr_sentiment(item['company'])
            extracted_text = task.get('extracted_text') or item.get('extracted_text', '')

            task['market_data'] = market_data
            task['forum_data'] = forum_data
            task['extracted_text'] = extracted_text

            print(f"\n[Gemini Worker-{worker_id} Stage {stage}] Analyzing {item['company']} ({item['exchange']}:{item['scrip']})...")
            prompt = build_sieve2_prompt(item, market_data, forum_data, extracted_text)
            gemini_output = evaluate_with_gemini(prompt)

            cat_score = extract_score(gemini_output, "Catalyst Score:")
            comp_score = extract_score(gemini_output, "Company Quality Score:")

            cat_score_safe = cat_score if isinstance(cat_score, int) else 1
            comp_score_safe = comp_score if isinstance(comp_score, int) else 5

            evals['gemini_catalyst_score'] = cat_score_safe
            evals['gemini_company_score'] = comp_score_safe
            evals['gemini_analysis'] = gemini_output

            if stage == 1:
                if cat_score_safe >= 5:
                    print(f" -> [Gemini Worker-{worker_id}] Score {cat_score_safe}/10 >= 5. Handoff to Claude Queue (Priority {20 - cat_score_safe})...")
                    claude_queue.put((20 - cat_score_safe, next(counter), {
                        'item': item, 'stage': 2, 'evals': evals, 'market_data': market_data, 'forum_data': forum_data, 'extracted_text': extracted_text
                    }))
                else:
                    finalize_single_model_ignore(item, evals, market_data, "gemini", extracted_text)
                    with completed_lock:
                        completed_count += 1
                        if completed_count >= total_items: stop_event.set()
            elif stage == 2:
                finalize_dual_evaluation(item, evals, market_data, extracted_text)
                with completed_lock:
                    completed_count += 1
                    if completed_count >= total_items: stop_event.set()

            gemini_queue.task_done()

    threads = []
    for i in range(num_workers):
        t_c = threading.Thread(target=claude_worker, args=(i + 1,), daemon=True)
        t_g = threading.Thread(target=gemini_worker, args=(i + 1,), daemon=True)
        threads.extend([t_c, t_g])
        t_c.start()
        t_g.start()

    for t in threads:
        t.join()


# ---------------------------------------------------------------------------
# 8. TELEGRAM DISPATCHER & MESSAGE FORMATTER
# ---------------------------------------------------------------------------
def send_telegram_alert(message):
    """Sends structured Markdown messages to the configured Telegram channel."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n[Telegram Output Preview]\n" + message)
        return

    url = TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN)
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False
    }
    try:
        requests.post(url, json=payload, timeout=8)
    except Exception as e:
        print(f"[Telegram Dispatch Error] {e}")


def build_telegram_message(company, exchange, scrip_code, isin, market_data, audit, all_links):
    """Formats individual stock alerts according to visual hierarchy."""
    status = audit['consensus_status']
    if status == "MODEL_DIVERGENCE":
        banner = "⚠️ *MODEL DIVERGENCE DETECTED*"
    elif status == "SINGLE_MODEL_IGNORE":
        banner = "🚫 *FILTERED / LOW CONVICTION (SINGLE MODEL)*"
    elif audit.get('high_conviction'):
        banner = "🚨 *HIGH CONVICTION CATALYST CONCURRENCE*"
    else:
        banner = "📢 *CORPORATE ACTION RE-RATING CATALYST*"

    price_str = f"₹{market_data['price']}" if market_data['price'] > 0 else "₹0.0 (Data Feed Sync)"

    c_cat = audit.get('claude_catalyst_score')
    g_cat = audit.get('gemini_catalyst_score')
    cat_scores = []
    if c_cat is not None: cat_scores.append(f"Claude: {c_cat}/10")
    if g_cat is not None: cat_scores.append(f"Gemini: {g_cat}/10")
    cat_line = " | ".join(cat_scores) if cat_scores else f"Score: {audit.get('final_score', 'N/A')}/10"

    c_comp = audit.get('claude_company_score')
    g_comp = audit.get('gemini_company_score')
    comp_scores = []
    if c_comp is not None: comp_scores.append(f"Claude: {c_comp}/10")
    if g_comp is not None: comp_scores.append(f"Gemini: {g_comp}/10")
    comp_line = " | ".join(comp_scores) if comp_scores else "N/A"

    msg = (
        f"{banner}\n"
        f"**Company:** {company}\n"
        f"**{exchange}:** `{scrip_code}` | **ISIN:** `{isin}`\n"
        f"**Price:** {price_str} | **20D Vol:** {market_data['vol_multiple']}x | **Est. MktCap:** ₹{market_data.get('market_cap_cr', 0)} Cr\n"
        f"**Consensus Status:** `{status}`\n\n"
        f"🎯 **Catalyst Score:** {cat_line}\n"
        f"🏢 **Company Quality:** {comp_line}\n"
    )

    if audit.get('claude_analysis'):
        msg += f"\n🧠 *Claude Analysis:*\n{audit['claude_analysis']}\n"
    if audit.get('gemini_analysis'):
        msg += f"\n🤖 *Gemini Analysis:*\n{audit['gemini_analysis']}\n"

    msg += "\n" + "\n".join([f"📄 [View Official Filing PDF {i + 1}]({link})" for i, link in enumerate(all_links)])
    return msg


def send_scan_digest(total_ingested, total_new, hits, rejections, duration_seconds):
    """Dispatches executive summary digest of the screening run to Telegram."""
    mode_text = "🔄 *Manual Refresh*" if IGNORE_CACHE else "⏰ *Scheduled GitHub Action Scan*"

    sast_count = sum(1 for r in rejections if any(k in r.get('rejection_reason', '').lower() for k in ['sast', 'pit', 'insider', 'transfer']))
    admin_count = sum(1 for r in rejections if any(k in r.get('rejection_reason', '').lower() for k in ['certificate', 'meeting', 'governance', 'window', 'newspaper']))
    other_count = len(rejections) - (sast_count + admin_count)

    hit_lines = ""
    if hits:
        hit_lines = "\n\n🎯 *High-Materiality Hits Passed to Sieve 2:*\n" + "\n".join([
            f"• *{h['company']}* (`{h['exchange']}:{h['scrip']}`) - {h.get('catalyst_reason', '')[:65]}"
            for h in hits
        ])

    digest_msg = (
        f"📊 *EXCHANGE SCAN COMPLETE (GitHub Runner)*\n"
        f"• *Mode:* {mode_text}\n"
        f"• *Total Ingested:* {total_ingested} filings\n"
        f"• *New Entities Screened:* {total_new}\n"
        f"• *Passed to Sieve 2:* {len(hits)} out of {total_new} ({round((len(hits) / max(total_new, 1)) * 100, 1)}%)\n"
        f"• *Execution Latency:* {round(duration_seconds, 1)}s\n"
        f"{hit_lines}\n\n"
        f"🚫 *Noise Filter Breakdown:* ({len(rejections)})\n"
        f"• SAST / PIT / Insider Transfers: {sast_count}\n"
        f"• Share Certificates / Board Meetings: {admin_count}\n"
        f"• Routine Disclosures: {max(0, other_count)}"
    )
    send_telegram_alert(digest_msg)


# ---------------------------------------------------------------------------
# 9. YOUTUBE TV INTERVIEW SCANNER
# ---------------------------------------------------------------------------
def process_youtube_interviews(channel_id=DEFAULT_YOUTUBE_CHANNEL_ID):
    """Scans TV interview transcripts for management guidance or capacity revisions."""
    if not gemini_client:
        return

    feed = feedparser.parse(YOUTUBE_RSS_URL.format(channel_id=channel_id))
    cutoff = datetime.now(timezone.utc) - timedelta(days=15)

    for entry in feed.entries[:4]:
        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        if published < cutoff:
            continue

        try:
            video_id = entry.yt_videoid
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([t['text'] for t in transcript[:120]])

            filter_prompt = f"""
            Does this management interview discuss capex expansion, guidance revisions, or major order pipeline?
            Title: {entry.title}
            Transcript Preview: {text}
            Return strictly the COMPANY_NAME if YES, or return IGNORE if routine.
            """
            res = gemini_client.models.generate_content(model=TIER1_MODEL, contents=filter_prompt)
            if "IGNORE" not in res.text:
                msg = f"📺 *MANAGEMENT INTERVIEW CATALYST*\n\n**Title:** {entry.title}\n🔗 [Watch Interview]({entry.link})"
                send_telegram_alert(msg)
        except Exception:
            continue


# ---------------------------------------------------------------------------
# 10. DUAL-EXCHANGE INGESTION MODULES
# ---------------------------------------------------------------------------
def fetch_live_nse_filings():
    """Fetches NSE announcements using a cookie-primed session."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    standardized_filings = []

    try:
        session.get(NSE_BASE_URL, timeout=10)
        resp = session.get(NSE_API_URL, timeout=10)

        if resp.status_code == 200:
            json_payload = resp.json()
            data = json_payload if isinstance(json_payload, list) else json_payload.get('data', [])

            for item in data:
                attachment = item.get('attchmntFile') or item.get('attchmntText') or str(item.get('seq_id', ''))
                if not attachment:
                    continue

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
    except Exception as e:
        print(f"[NSE Ingestion Notice] {e}")

    return standardized_filings


def fetch_live_bse_filings(max_pages=100):
    """Paginates through the BSE API for corporate announcements."""
    standardized_filings = []
    page = 1

    while page <= max_pages:
        url = BSE_API_URL.format(page=page)
        try:
            resp = requests.get(url, headers=BSE_HEADERS, timeout=10)
            if resp.status_code == 200:
                table = resp.json().get('Table', [])
                if not table:
                    break

                for item in table:
                    attachment = item.get('ATTACHMENTNAME')
                    if not attachment:
                        continue

                    pdf_link = BSE_PDF_BASE_URL.format(attachment=attachment)
                    raw_isin = item.get('ISIN_CODE', '').strip()

                    standardized_filings.append({
                        'id': attachment,
                        'company': item.get('SLONGNAME', 'Unknown Company'),
                        'scrip': str(item.get('SCRIP_CD', '')).strip(),
                        'headline': item.get('NEWSSUB', ''),
                        'isin': raw_isin if raw_isin else 'N/A',
                        'link': pdf_link,
                        'exchange': 'BSE'
                    })

                page += 1
                time.sleep(0.2)
            else:
                break
        except Exception as e:
            print(f"[BSE Ingestion Notice on Page {page}] {e}")
            break

    return standardized_filings


# ---------------------------------------------------------------------------
# 11. MAIN PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    print(f"[{datetime.now()}] Initializing Market Intelligence Pipeline (GitHub Runner)...")

    raw_bse = fetch_live_bse_filings(max_pages=args.max_pages)
    raw_nse = fetch_live_nse_filings()
    unified_filings = raw_bse + raw_nse

    print(f"Ingested {len(raw_bse)} BSE filings and {len(raw_nse)} NSE filings ({len(unified_filings)} total).")

    unprocessed_filings = filter_unprocessed_announcements(unified_filings)
    print(f"Filtering complete. {len(unprocessed_filings)} new individual filings found.")

    if not unprocessed_filings:
        print("No new filings found in this scan cycle. Exiting.")
        return

    grouped_filings = group_filings_by_company(unprocessed_filings)
    print(f"Consolidated into {len(grouped_filings)} distinct company events.")

    # Tier 1 Sieve (Gemini Flash Lite)
    hits, rejections = run_tier1_batch_sieve(grouped_filings)

    if rejections:
        rejection_tuples = []
        for r in rejections:
            for att_id in r['all_ids']:
                rejection_tuples.append((att_id, r['company'], r['headline'], "IGNORE"))
        log_announcements_batch(rejection_tuples)

    # Tier 1.5 Sieve (Local Qwen 2.5 7B on GitHub Runner)
    sieve_1_5_passed = []
    if hits:
        print(f"\n=======================================================")
        print(f"🤖 [SIEVE 1.5] Executing Qwen 2.5 7B extraction on {len(hits)} hits...")
        print(f"=======================================================\n")

        for h in hits:
            primary_link = h['all_links'][0] if h.get('all_links') else h.get('link', '')
            raw_pdf_text = extract_text_from_pdf_url(primary_link)
            extracted_text, pre_score = sieve_1_5_local_qwen_extraction(raw_pdf_text, h['headline'])

            h['extracted_text'] = extracted_text
            h['qwen_pre_score'] = pre_score

            print(f" -> [{h['company']}] Qwen Pre-Score: {pre_score}/10 (Threshold: >= {SIEVE_1_5_MIN_SCORE})")

            if pre_score >= SIEVE_1_5_MIN_SCORE:
                sieve_1_5_passed.append(h)
            else:
                ignore_tuples = [(att_id, h['company'], h['headline'], "IGNORE") for att_id in h['all_ids']]
                log_announcements_batch(ignore_tuples)
                print(f"    🚫 Filtered out locally by Sieve 1.5 (Score {pre_score} < {SIEVE_1_5_MIN_SCORE}). Saved cloud tokens.")

    print(f"\n=======================================================")
    print(f"📊 SIEVE 1.5 SUMMARY: {len(sieve_1_5_passed)} out of {len(hits)} passed to Sieve 2.")
    print(f"=======================================================\n")

    # Tier 2 Staggered Workers (Claude Sonnet 5 + Gemini 3.1 Pro Preview)
    if sieve_1_5_passed:
        sieve2_limit = args.max_sieve2
        candidates = sieve_1_5_passed[:sieve2_limit] if sieve2_limit > 0 else sieve_1_5_passed
        run_staggered_sieve2_workers(candidates, num_workers=4)

    duration = time.time() - start_time

    # End-of-Scan Telegram Summary Digest
    send_scan_digest(
        total_ingested=len(unified_filings),
        total_new=len(grouped_filings),
        hits=sieve_1_5_passed,
        rejections=rejections,
        duration_seconds=duration
    )

    process_youtube_interviews()
    print(f"[{datetime.now()}] Pipeline execution finished in {round(duration, 1)}s.")


if __name__ == "__main__":
    main()