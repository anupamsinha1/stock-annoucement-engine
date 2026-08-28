"""
Market Intelligence & Corporate Announcement Screening Engine (GitHub Actions Edition)
=======================================================================================
Target Environment:
- Designed specifically for automated execution inside GitHub Actions Ubuntu runners.
- Leverages GitHub's public runner hardware (4 vCPUs / 16 GB RAM) to execute local
  quantized LLM inference ('qwen2.5:7b') via background Ollama service at $0 compute cost.

Filename: engine_github.py
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

# ---------------------------------------------------------------------------
# VERBOSITY & ALERTING THRESHOLDS
# 1.0 = Ping for Sieve 1 (Headline), Sieve 1.5 (Qwen), and Sieve 2 (Claude/Gemini)
# 1.5 = Ping for Sieve 1.5 (Qwen) and Sieve 2 (Claude/Gemini)
# 2.0 = ONLY Ping for Sieve 2 Final Consensus (Default & Recommended)
# ---------------------------------------------------------------------------
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

# Local Model (Sieve 1.5) - Qwen 2.5 7B on GitHub Runners
USE_LOCAL_EXTRACTOR = os.getenv("USE_LOCAL_EXTRACTOR", "true").lower() in ("true", "1", "yes")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", DEFAULT_OLLAMA_URL)

# Cloud Model Credentials & Endpoints
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_PAID") or os.getenv("GEMINI_API_KEY_FREE")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SUPABASE_URL = os.getenv("SUPABASE_URL")

# Cloud Models
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
    if not SUPABASE_URL:
        return None
    try:
        return psycopg2.connect(SUPABASE_URL)
    except Exception as e:
        print(f"[Database Error] Connection failed: {e}")
        return None

def filter_unprocessed_announcements(filings):
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
                item['id'], item['company'], item['scrip'], item['exchange'], item.get('isin', 'N/A'),
                item.get('headline', ''), item.get('catalyst_reason', 'Actionable corporate action'),
                market_data.get('price', 0.0), market_data.get('market_cap_cr', 0.0), market_data.get('vol_multiple', 1.0),
                market_data.get('above_50dma', False), market_data.get('above_200dma', False),
                audit.get('final_score', 1), audit.get('consensus_status', 'NEUTRAL_MIX'), audit.get('high_conviction', False),
                audit.get('claude_catalyst_score'), audit.get('gemini_catalyst_score'),
                audit.get('claude_company_score'), audit.get('gemini_company_score'),
                extracted_text, evals.get('claude_analysis', ''), evals.get('gemini_analysis', '')
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Database Error] Comprehensive ledger logging failed: {e}")


# ---------------------------------------------------------------------------
# 3. QUANT METRICS (FIXED NUMPY BOOL ADAPTION)
# ---------------------------------------------------------------------------
def fetch_market_metrics(scrip_code, exchange):
    default_payload = {
        "price": 0.0, "vol_multiple": 1.0, "above_50dma": False,
        "above_200dma": False, "price_feed_sync": True, "market_cap_cr": 0.0
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
            "above_50dma": bool(current_price > dma_50),
            "above_200dma": bool(current_price > dma_200),
            "price_feed_sync": bool(current_price == 0.0),
            "market_cap_cr": market_cap_cr
        }
    except Exception as e:
        print(f"[Market Data Notice] Scrip {scrip_code} ({exchange}): {e}")
        return default_payload

def fetch_valuepickr_sentiment(company_name):
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
    if not text:
        return ""
    match = re.search(r'(?i)(?:Sub(?:ject)?\s*:|Ref\s*:)', text)
    if match: text = text[match.start():]
    text = re.sub(r'(?i)(?:\bDisclaimer\b|CAREEDGE RATINGS DISCLAIMS|S&P Global Ratings Terms and Conditions).*$', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F₹]+', '', text)
    return text.strip()

def extract_text_from_pdf_url(pdf_url, max_pages=MAX_PDF_PAGES, max_chars=MAX_PDF_CHARS):
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
                    if text: extracted_pages.append(text)
                raw_text = "\n".join(extracted_pages)
                clean_text = sanitize_filing_text(raw_text)
                return clean_text[:max_chars] if clean_text else "PDF contains scanned imagery or unextractable text."
        return f"Failed to download PDF (HTTP {response.status_code})"
    except Exception as e:
        return f"PDF extraction error: {e}"

def extract_score(text, label):
    if not text:
        return None
    for line in text.splitlines():
        if label.lower() in line.lower():
            match = re.search(r'(\b10|[1-9])\s*/\s*10', line)
            if match: return int(match.group(1))
            digits = re.findall(r'\b(10|[1-9])\b', line)
            if digits: return int(digits[0])
    return None


# ---------------------------------------------------------------------------
# 5. SIEVE 1 & SIEVE 1.5
# ---------------------------------------------------------------------------
def run_tier1_batch_sieve(announcements):
    if not announcements or not gemini_client:
        return announcements, []
    hits = []
    rejections = []
    chunk_size = DEFAULT_CHUNK_SIZE
    for i in range(0, len(announcements), chunk_size):
        chunk = announcements[i:i + chunk_size]
        items_payload = [{"index": idx, "exchange": ann['exchange'], "scrip": ann['scrip'], "company": ann['company'], "headline": ann['headline']} for idx, ann in enumerate(chunk)]
        prompt = f"""
        You are an objective exchange filing intake filter. Categorize EACH announcement as either "HIT" or "REJECT".
        STRICT REJECTION RULES (REJECT ONLY IF UNAMBIGUOUS NOISE):
        - Loss of share certificates, trading window closures, meeting intimations, shareholding patterns, ESOP allotments, newspaper clippings.
        BROAD INTAKE RULES (ALWAYS FLAG AS "HIT"):
        - Commercial orders, financial results, guidance, fundraisings, M&A, capex, deleveraging, ratings upgrades, FDA/CDSCO approvals.
        Respond with ONLY a valid JSON array. Announcements:
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
                            print(f" [SIEVE 1 HIT] {ann_item['company']} | {reason}")
                        else:
                            ann_item["rejection_reason"] = reason
                            rejections.append(ann_item)
                            print(f" [SIEVE 1 REJECT] {ann_item['company']} | {reason}")
                break
            except Exception as e:
                err_str = str(e).lower()
                if "503" in err_str or "429" in err_str or "quota" in err_str:
                    time.sleep(2 ** attempt)
                    continue
                print(f"[Tier 1 Error] Chunk failed: {e}. Defaulting chunk to HIT.")
                hits.extend(chunk)
                break
    return hits, rejections

def sieve_1_5_local_qwen_extraction(cleaned_pdf_text, headline):
    if not USE_LOCAL_EXTRACTOR or not cleaned_pdf_text or len(cleaned_pdf_text) < 80:
        return cleaned_pdf_text, 5
    prompt = f"""
    You are a strict financial analyst pre-screening corporate disclosures. Extract key facts and assign an objective PreScore from 1 to 10.
    CRITICAL SCORING RULES:
    - Score 1 to 4: Trade expo, generic PR marketing, non-binding MoUs.
    - Score 5 to 6: Small/routine purchase orders, incremental progress.
    - Score 7 to 10: Confirmed contract wins (>INR 50 Cr+), net-debt reduction, major capex, strong financial beats.
    Headline: {headline}
    Filing Body:
    {cleaned_pdf_text[:4500]}
    Output format:
    Summary: [150-word summary]
    Value: [Exact deal value]
    Client: [Entity name]
    PreScore: [Strictly an integer from 1 to 10, e.g., 7/10]
    """.strip()
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}}
    try:
        res = requests.post(OLLAMA_API_URL, json=payload, timeout=90)
        if res.status_code == 200:
            extracted_output = res.json().get("response", "").strip()
            if extracted_output:
                pre_score = extract_score(extracted_output, "PreScore:")
                return extracted_output, (pre_score if pre_score is not None else 5)
    except Exception as e:
        print(f" [Sieve 1.5 Warning] Qwen bypassed ({e}). Falling back to raw text.")
    return cleaned_pdf_text, 5


# ---------------------------------------------------------------------------
# 6. TIER 2 (SIEVE 2): CLAUDE & GEMINI DEEP DIVE
# ---------------------------------------------------------------------------
def build_sieve2_prompt(item, market_data, forum_text, extracted_text):
    price_display = f"INR {market_data['price']}" if market_data['price'] > 0 else "Data Feed Sync (Evaluate on strategic merits)"
    mkt_cap_display = f"INR {market_data['market_cap_cr']} Cr" if market_data.get('market_cap_cr', 0) > 0 else "Not specified"
    return f"""You are an institutional equity research analyst evaluating corporate exchange filings.
Assess how strongly this event will impact the company's future earnings power and trajectory.
EVALUATION PRINCIPLES: Disregard market cap. Evaluate the PROPORTIONAL impact of the event on the company's business scale.
SCORING BENCHMARKS (1 to 10):
• 1 to 3: Administrative / Noise.
• 4 to 6: Moderate Operational Update.
• 7 to 8: High-Impact Structural Catalyst (Large firm order wins, full balance sheet deleveraging).
• 9 to 10: Transformational Breakout.

Company: {item['company']} ({item['exchange']}: {item['scrip']} | ISIN: {item['isin']})
Price: {price_display} | 20D Vol: {market_data['vol_multiple']}x | MktCap: {mkt_cap_display}
Headline: {item['headline']}
Flagged Catalyst: {item.get('catalyst_reason', 'Actionable corporate action')}
Forum/Scuttlebutt Context: {forum_text}

==================== EXTRACTED DISCLOSURE METRICS ====================
{extracted_text}
======================================================================

OUTPUT FORMAT:
Reasoning: <2-3 sentences on proportional commercial impact>
Hype / Red Flag Check: <Clean / Warning flags>
Catalyst Score: <Strictly an integer from 1 to 10>
Company Quality Score: <Strictly an integer from 1 to 10>
"""

def evaluate_with_claude(prompt):
    if not claude_client: return "Claude evaluation skipped."
    for attempt in range(3):
        try:
            res = claude_client.messages.create(model=TIER2_CLAUDE_MODEL, max_tokens=500, messages=[{"role": "user", "content": prompt}])
            text_blocks = [b.text for b in res.content if getattr(b, "type", None) == "text"]
            return "\n".join(text_blocks).strip() if text_blocks else "Claude returned no text."
        except Exception as e:
            err_str = str(e).lower()
            if "503" in err_str or "429" in err_str or "overloaded" in err_str or "rate_limit" in err_str:
                time.sleep(2 ** attempt); continue
            return f"Claude analysis error: {e}"
    return "Claude analysis error: API unavailable after retries."

def evaluate_with_gemini(prompt):
    if not gemini_client: return "Gemini Pro evaluation skipped."
    for attempt in range(3):
        try:
            res = gemini_client.models.generate_content(model=TIER2_GEMINI_MODEL, contents=prompt)
            return res.text.strip()
        except Exception as e:
            err_str = str(e).lower()
            if "503" in err_str or "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                time.sleep(2 ** attempt); continue
            return f"Gemini Pro analysis error: {e}"
    return "Gemini Pro analysis error: API unavailable after retries."


# ---------------------------------------------------------------------------
# 7. TELEGRAM DISPATCHER & MESSAGE FORMATTER
# ---------------------------------------------------------------------------
def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n[Telegram Output Preview]\n" + message)
        return
    url = TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN)
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown", "disable_web_page_preview": False}
    try:
        requests.post(url, json=payload, timeout=8)
    except Exception as e:
        print(f"[Telegram Dispatch Error] {e}")

def build_telegram_message(company, exchange, scrip_code, isin, market_data, audit, all_links):
    status = audit['consensus_status']
    if status == "SIEVE_1_PASS":
        banner = "🔍 *INITIAL RADAR: SIEVE 1 (FLASH) PASSED*"
    elif status == "SIEVE_1_5_PASS":
        banner = "⚡ *EARLY RADAR: SIEVE 1.5 (QWEN) PASSED*"
    elif status == "MODEL_DIVERGENCE":
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

    fs = audit.get('final_score', 'N/A')
    cat_line = " | ".join(cat_scores) if cat_scores else (f"Score: {fs}/10" if isinstance(fs, (int, float)) else "Score: N/A")

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
    )

    if status not in ["SIEVE_1_PASS", "SIEVE_1_5_PASS"]:
        msg += f"**Consensus Status:** `{status}`\n\n🎯 **Catalyst Score:** {cat_line}\n🏢 **Company Quality:** {comp_line}\n"

    if status == "SIEVE_1_PASS":
        msg += f"\n🔍 *Sieve 1 Catalyst Reason:*\n{audit.get('claude_analysis', 'N/A')}\n"
    elif status == "SIEVE_1_5_PASS":
        msg += f"\n🤖 *Qwen 2.5 Extracted Facts (Score {fs}/10):*\n{audit.get('claude_analysis', 'N/A')}\n"
    else:
        if audit.get('claude_analysis'):
            msg += f"\n🧠 *Claude Analysis:*\n{audit['claude_analysis']}\n"
        if audit.get('gemini_analysis'):
            msg += f"\n🤖 *Gemini Analysis:*\n{audit['gemini_analysis']}\n"

    msg += "\n" + "\n".join([f"📄 [View Official Filing PDF {i + 1}]({link})" for i, link in enumerate(all_links)])
    return msg


# ---------------------------------------------------------------------------
# 8. STAGGERED PRIORITY QUEUE WORKER POOL & FINALIZATION
# ---------------------------------------------------------------------------
def finalize_dual_evaluation(item, evals, market_data, extracted_text):
    c_cat = evals.get('claude_catalyst_score', 1)
    g_cat = evals.get('gemini_catalyst_score', 1)
    c_comp = evals.get('claude_company_score', 5)
    g_comp = evals.get('gemini_company_score', 5)

    is_divergent = (abs(c_cat - g_cat) >= 4 or (c_cat >= 7 and g_cat <= 4) or (g_cat >= 7 and c_cat <= 4))
    if is_divergent: consensus_status = "MODEL_DIVERGENCE"
    elif c_cat >= 7 and g_cat >= 7: consensus_status = "CONSENSUS_HIT"
    elif c_cat <= 4 and g_cat <= 4: consensus_status = "CONSENSUS_IGNORE"
    else: consensus_status = "NEUTRAL_MIX"

    final_score = round((c_cat + g_cat) / 2)
    is_high_conviction = (final_score >= 8 and market_data['vol_multiple'] >= 2.0 and market_data['above_50dma'])

    audit = {
        'consensus_status': consensus_status, 'final_score': final_score, 'high_conviction': is_high_conviction,
        'claude_catalyst_score': c_cat, 'gemini_catalyst_score': g_cat,
        'claude_company_score': c_comp, 'gemini_company_score': g_comp,
        'claude_analysis': evals.get('claude_analysis', ''), 'gemini_analysis': evals.get('gemini_analysis', '')
    }

    # Always publish if threshold is 2.0 or lower (cascading logic)
    if PUBLISH_WHEN_SIEVE_PASSED <= 2.0 and final_score >= 6:
        alert_msg = build_telegram_message(company=item['company'], exchange=item['exchange'], scrip_code=item['scrip'], isin=item['isin'], market_data=market_data, audit=audit, all_links=item['all_links'])
        send_telegram_alert(alert_msg)

    log_permanent_ledger(item, market_data, evals, audit, extracted_text)
    log_announcements_batch([(att_id, item['company'], item['headline'], "HIT") for att_id in item['all_ids']])

def finalize_single_model_ignore(item, evals, market_data, source_model, extracted_text):
    score = evals.get(f'{source_model.lower()}_catalyst_score', 1)
    print(f" -> [{source_model.upper()} Gatekeeper] Score {score}/10 < 5. Terminating Sieve 2 early for {item['company']}.")

    audit = {
        'consensus_status': "SINGLE_MODEL_IGNORE", 'final_score': score, 'high_conviction': False,
        'claude_catalyst_score': evals.get('claude_catalyst_score'), 'gemini_catalyst_score': evals.get('gemini_catalyst_score'),
        'claude_company_score': evals.get('claude_company_score', 5), 'gemini_company_score': evals.get('gemini_company_score', 5),
        'claude_analysis': evals.get('claude_analysis', ''), 'gemini_analysis': evals.get('gemini_analysis', '')
    }

    if ALERT_ON_SINGLE_MODEL_IGNORE:
        alert_msg = build_telegram_message(company=item['company'], exchange=item['exchange'], scrip_code=item['scrip'], isin=item['isin'], market_data=market_data, audit=audit, all_links=item['all_links'])
        send_telegram_alert(alert_msg)

    log_announcements_batch([(att_id, item['company'], item['headline'], "IGNORE") for att_id in item['all_ids']])

def run_staggered_sieve2_workers(hits, num_workers=4):
    if not hits: return
    counter = itertools.count()
    claude_queue = queue.PriorityQueue()
    gemini_queue = queue.PriorityQueue()
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
            extracted_text = task.get('extracted_text') or item.get('extracted_text', '')

            task.update({'market_data': market_data, 'forum_data': forum_data, 'extracted_text': extracted_text})
            print(f"\n[Claude Worker-{worker_id} Stage {stage}] Analyzing {item['company']} ({item['exchange']}:{item['scrip']})...")
            prompt = build_sieve2_prompt(item, market_data, forum_data, extracted_text)
            claude_output = evaluate_with_claude(prompt)

            cat_score, comp_score = extract_score(claude_output, "Catalyst Score:"), extract_score(claude_output, "Company Quality Score:")
            cat_score_safe, comp_score_safe = cat_score if isinstance(cat_score, int) else 1, comp_score if isinstance(comp_score, int) else 5

            evals.update({'claude_catalyst_score': cat_score_safe, 'claude_company_score': comp_score_safe, 'claude_analysis': claude_output})

            if stage == 1:
                if cat_score_safe >= 5:
                    print(f" -> [Claude Worker-{worker_id}] Score {cat_score_safe}/10 >= 5. Handoff to Gemini Queue (Priority {20 - cat_score_safe})...")
                    gemini_queue.put((20 - cat_score_safe, next(counter), {'item': item, 'stage': 2, 'evals': evals, 'market_data': market_data, 'forum_data': forum_data, 'extracted_text': extracted_text}))
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
            try: priority, seq, task = gemini_queue.get(timeout=1.0)
            except queue.Empty:
                with completed_lock:
                    if completed_count >= total_items: break
                continue

            item, stage, evals = task['item'], task['stage'], task['evals']
            market_data = task.get('market_data') or fetch_market_metrics(item['scrip'], item['exchange'])
            forum_data = task.get('forum_data') or fetch_valuepickr_sentiment(item['company'])
            extracted_text = task.get('extracted_text') or item.get('extracted_text', '')

            task.update({'market_data': market_data, 'forum_data': forum_data, 'extracted_text': extracted_text})
            print(f"\n[Gemini Worker-{worker_id} Stage {stage}] Analyzing {item['company']} ({item['exchange']}:{item['scrip']})...")
            prompt = build_sieve2_prompt(item, market_data, forum_data, extracted_text)
            gemini_output = evaluate_with_gemini(prompt)

            cat_score, comp_score = extract_score(gemini_output, "Catalyst Score:"), extract_score(gemini_output, "Company Quality Score:")
            cat_score_safe, comp_score_safe = cat_score if isinstance(cat_score, int) else 1, comp_score if isinstance(comp_score, int) else 5

            evals.update({'gemini_catalyst_score': cat_score_safe, 'gemini_company_score': comp_score_safe, 'gemini_analysis': gemini_output})

            if stage == 1:
                if cat_score_safe >= 5:
                    print(f" -> [Gemini Worker-{worker_id}] Score {cat_score_safe}/10 >= 5. Handoff to Claude Queue (Priority {20 - cat_score_safe})...")
                    claude_queue.put((20 - cat_score_safe, next(counter), {'item': item, 'stage': 2, 'evals': evals, 'market_data': market_data, 'forum_data': forum_data, 'extracted_text': extracted_text}))
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

    digest_msg = (
        f"📊 *EXCHANGE SCAN COMPLETE (GitHub Runner)*\n"
        f"• *Mode:* {mode_text}\n"
        f"• *Total Ingested:* {total_ingested} filings\n"
        f"• *New Entities Screened:* {total_new}\n"
        f"• *Passed Sieve 1.5:* {len(hits)} out of {total_new} ({round((len(hits) / max(total_new, 1)) * 100, 1)}%)\n"
        f"• *Execution Latency:* {round(duration_seconds, 1)}s\n"
        f"{hit_lines}\n\n"
        f"🚫 *Noise Filter Breakdown:* ({len(rejections)})\n"
        f"• SAST / PIT / Insider Transfers: {sast_count}\n"
        f"• Share Certificates / Board Meetings: {admin_count}\n"
        f"• Routine Disclosures: {max(0, other_count)}"
    )
    send_telegram_alert(digest_msg)


# ---------------------------------------------------------------------------
# 9. YOUTUBE TV INTERVIEW SCANNER & INGESTION
# ---------------------------------------------------------------------------
def process_youtube_interviews(channel_id=DEFAULT_YOUTUBE_CHANNEL_ID):
    if not gemini_client: return
    feed = feedparser.parse(YOUTUBE_RSS_URL.format(channel_id=channel_id))
    cutoff = datetime.now(timezone.utc) - timedelta(days=15)
    for entry in feed.entries[:4]:
        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        if published < cutoff: continue
        try:
            video_id = entry.yt_videoid
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([t['text'] for t in transcript[:120]])
            res = gemini_client.models.generate_content(model=TIER1_MODEL, contents=f"Does this interview discuss capex expansion, guidance revisions, or major order pipeline?\nTitle: {entry.title}\nTranscript Preview: {text}\nReturn strictly the COMPANY_NAME if YES, or return IGNORE if routine.")
            if "IGNORE" not in res.text:
                send_telegram_alert(f"📺 *MANAGEMENT INTERVIEW CATALYST*\n\n**Title:** {entry.title}\n🔗 [Watch Interview]({entry.link})")
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
                    attachment = item.get('ATTACHMENTNAME')
                    if not attachment: continue
                    raw_isin = item.get('ISIN_CODE', '').strip()
                    standardized_filings.append({'id': attachment, 'company': item.get('SLONGNAME', 'Unknown Company'), 'scrip': str(item.get('SCRIP_CD', '')).strip(), 'headline': item.get('NEWSSUB', ''), 'isin': raw_isin if raw_isin else 'N/A', 'link': BSE_PDF_BASE_URL.format(attachment=attachment), 'exchange': 'BSE'})
                page += 1
                time.sleep(0.2)
            else: break
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

    unified_filings = fetch_live_bse_filings(max_pages=args.max_pages) + fetch_live_nse_filings()
    unprocessed_filings = filter_unprocessed_announcements(unified_filings)

    if not unprocessed_filings:
        print("No new filings found in this scan cycle. Exiting.")
        return

    grouped_filings = group_filings_by_company(unprocessed_filings)
    print(f"Consolidated into {len(grouped_filings)} distinct company events.")

    # ---------------------------------------------------------
    # TIER 1: FLASH LITE BROAD-NET FILTER
    # ---------------------------------------------------------
    hits, rejections = run_tier1_batch_sieve(grouped_filings)

    if rejections:
        log_announcements_batch([(att_id, r['company'], r['headline'], "IGNORE") for r in rejections for att_id in r['all_ids']])

    # -> EARLY RADAR DISPATCH: SIEVE 1 (If Verbosity is 1.0)
    if hits and PUBLISH_WHEN_SIEVE_PASSED <= 1.0:
        for h in hits:
            dummy_audit = {
                'consensus_status': 'SIEVE_1_PASS', 'final_score': 'N/A', 'high_conviction': False,
                'claude_catalyst_score': None, 'gemini_catalyst_score': None,
                'claude_analysis': h.get('catalyst_reason', ''), 'gemini_analysis': ''
            }
            mkt = fetch_market_metrics(h['scrip'], h['exchange'])
            alert_msg = build_telegram_message(company=h['company'], exchange=h['exchange'], scrip_code=h['scrip'], isin=h['isin'], market_data=mkt, audit=dummy_audit, all_links=h['all_links'])
            send_telegram_alert(alert_msg)

    # ---------------------------------------------------------
    # TIER 1.5: QWEN 2.5 LOCAL LLM
    # ---------------------------------------------------------
    sieve_1_5_passed = []
    if hits:
        print(f"\n=======================================================")
        print(f"🤖 [SIEVE 1.5] Executing Qwen 2.5 7B extraction on {len(hits)} hits...")
        print(f"=======================================================\n")

        for h in hits:
            primary_link = h['all_links'][0] if h.get('all_links') else h.get('link', '')
            raw_pdf_text = extract_text_from_pdf_url(primary_link)
            extracted_text, pre_score = sieve_1_5_local_qwen_extraction(raw_pdf_text, h['headline'])

            h['extracted_text'], h['qwen_pre_score'] = extracted_text, pre_score
            print(f" -> [{h['company']}] Qwen Pre-Score: {pre_score}/10 (Threshold: >= {SIEVE_1_5_MIN_SCORE})")

            if pre_score >= SIEVE_1_5_MIN_SCORE:
                sieve_1_5_passed.append(h)

                # -> EARLY RADAR DISPATCH: SIEVE 1.5 (If Verbosity is 1.5 or lower)
                if PUBLISH_WHEN_SIEVE_PASSED <= 1.5:
                    dummy_audit = {
                        'consensus_status': 'SIEVE_1_5_PASS', 'final_score': pre_score, 'high_conviction': False,
                        'claude_catalyst_score': None, 'gemini_catalyst_score': None,
                        'claude_analysis': extracted_text[:800], 'gemini_analysis': ''
                    }
                    mkt = fetch_market_metrics(h['scrip'], h['exchange'])
                    alert_msg = build_telegram_message(company=h['company'], exchange=h['exchange'], scrip_code=h['scrip'], isin=h['isin'], market_data=mkt, audit=dummy_audit, all_links=h['all_links'])
                    send_telegram_alert(alert_msg)
            else:
                log_announcements_batch([(att_id, h['company'], h['headline'], "IGNORE") for att_id in h['all_ids']])
                print(f"    🚫 Filtered out locally by Sieve 1.5 (Score {pre_score} < {SIEVE_1_5_MIN_SCORE}).")

    # ---------------------------------------------------------
    # TIER 2: CLAUDE & GEMINI CONCURRENCE (Publishes if <= 2.0)
    # ---------------------------------------------------------
    if sieve_1_5_passed:
        candidates = sieve_1_5_passed[:args.max_sieve2] if args.max_sieve2 > 0 else sieve_1_5_passed
        run_staggered_sieve2_workers(candidates, num_workers=4)

    duration = time.time() - start_time
    send_scan_digest(len(unified_filings), len(grouped_filings), sieve_1_5_passed, rejections, duration)
    process_youtube_interviews()
    print(f"[{datetime.now()}] Pipeline execution finished in {round(duration, 1)}s.")

if __name__ == "__main__":
    main()