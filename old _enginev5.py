"""
Market Intelligence & Corporate Announcement Screening Engine
============================================================
Architecture:
- Ingestion: Live BSE & NSE corporate filings APIs (limitless loops) + YouTube TV
- Deduplication: Supabase PostgreSQL 7-day rolling cache (with optional bypass flag)
- Tier 1 Sieve: Batch filtering via Gemini Flash with structured rejection explanations
- Tier 2 Deep-Dive: Concurrent dual-model evaluation (Claude Sonnet 5 + Gemini 3.1 Pro)
- Financial Scoring: Dedicated operational growth & margin audit for quarterly results
- Consensus & Divergence: Automated detection of conflicting model convictions
- Tier 3 Quant Audit: 20-day delivery volume surges & 50/200 DMA trend checks
- Dispatcher: Real-time action alerts + End-of-Scan Telegram Digest
- Permanent Ledger: Full audit trail for multi-horizon backtesting & prompt optimization
"""

import os
import re
import sys
import json
import time
import argparse
from datetime import datetime, timedelta, timezone
import concurrent.futures
import requests
import feedparser
import psycopg2
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from anthropic import Anthropic
from youtube_transcript_api import YouTubeTranscriptApi

# ---------------------------------------------------------------------------
# 0. CONFIGURATION & CLI ARGUMENT SETUP
# ---------------------------------------------------------------------------
load_dotenv()

# Parse CLI arguments for manual runs / GitHub Actions
parser = argparse.ArgumentParser(description="Corporate Announcement Screening Engine")
parser.add_argument("--ignore-cache", "-f", action="store_true", help="Bypass Supabase cache and re-evaluate all filings")
parser.add_argument("--max-pages", type=int, default=100, help="Maximum BSE announcement pages to fetch")
args, _ = parser.parse_known_args()

# Check both CLI flag and environment variable for cache bypass
IGNORE_CACHE = args.ignore_cache or os.getenv("IGNORE_CACHE", "false").lower() in ("true", "1", "yes")

# API Keys & Endpoints
GEMINI_API_KEY_FREE = os.getenv("GEMINI_API_KEY_FREE")
GEMINI_API_KEY_PAID = os.getenv("GEMINI_API_KEY_PAID") or GEMINI_API_KEY_FREE
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SUPABASE_URL = os.getenv("SUPABASE_URL")

# Model Configuration
TIER1_MODEL = os.getenv("GEMINI_TIER1_MODEL", "gemini-3.5-flash-lite")  # Ultra-cheap batch sieve
TIER2_GEMINI_MODEL = os.getenv("GEMINI_TIER2_MODEL", "gemini-3.1-pro-preview")
TIER2_CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")

# Initialize SDK Clients
gemini_tier1_client = genai.Client(api_key=GEMINI_API_KEY_PAID) if GEMINI_API_KEY_PAID else None
gemini_paid_client = genai.Client(api_key=GEMINI_API_KEY_PAID) if GEMINI_API_KEY_PAID else None
claude_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


# ---------------------------------------------------------------------------
# 1. SUPABASE DATABASE LAYER (CACHE & PERMANENT LEDGER)
# ---------------------------------------------------------------------------
def get_db_connection():
    """Establishes connection to Supabase PostgreSQL."""
    if not SUPABASE_URL:
        print("[Database] SUPABASE_URL is not configured in .env.")
        return None
    try:
        return psycopg2.connect(SUPABASE_URL)
    except Exception as e:
        print(f"[Database Error] Connection failed: {e}")
        return None


def filter_unprocessed_announcements(filings):
    """Filters out already-evaluated filings using the Supabase 7-day cache."""
    if IGNORE_CACHE:
        print("[Cache Bypass] IGNORE_CACHE is active. Evaluating all incoming announcements freshly.")
        return filings

    conn = get_db_connection()
    if not conn:
        return filings  # Fail-open strategy

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


def log_permanent_ledger(attachment_id, company, scrip_code, isin, price, final_score,
                         thesis, high_conviction, gemini_score, claude_score, consensus_status):
    """Permanently logs an evaluated hit for multi-horizon backtesting."""
    conn = get_db_connection()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO ai_recommendation_ledger 
            (bse_attachment_name, company_name, scrip_code, isin, alert_price, bullishness_score, 
             thesis_summary, high_conviction, gemini_score, claude_score, consensus_status) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
            ON CONFLICT (bse_attachment_name) DO NOTHING
            """,
            (attachment_id, company, scrip_code, isin, price, final_score,
             thesis, high_conviction, gemini_score, claude_score, consensus_status)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Database Error] Permanent ledger insert failed: {e}")


# ---------------------------------------------------------------------------
# 2. MARKET DATA & QUANT CHECKS (PRICE, VOLUME SURGE, DMA)
# ---------------------------------------------------------------------------
def fetch_market_metrics(scrip_code, exchange):
    """Fetches real-time price, 20D volume surge multiple, and moving averages via Yahoo Finance."""
    default_payload = {"price": 0.0, "vol_multiple": 1.0, "above_50dma": False, "above_200dma": False}
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

        return {
            "price": current_price,
            "vol_multiple": vol_multiple,
            "above_50dma": current_price > dma_50,
            "above_200dma": current_price > dma_200
        }
    except Exception as e:
        print(f"[Market Data Warning] Scrip {scrip_code} ({exchange}): {e}")
        return default_payload


# ---------------------------------------------------------------------------
# 3. SCUTTLEBUTT: VALUEPICKR FORUM PARSER
# ---------------------------------------------------------------------------
def fetch_valuepickr_sentiment(company_name):
    """Queries ValuePickr Discourse API for community discussion and red flags."""
    try:
        clean_name = company_name.split()[0].replace("Ltd", "").strip()
        url = f"https://forum.valuepickr.com/search/query.json?term={clean_name}"
        headers = {'User-Agent': 'Mozilla/5.0'}

        res = requests.get(url, headers=headers, timeout=5)
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
# 4. TIER 1: BATCH FILTER SIEVE WITH EXPLAINABLE REJECTIONS
# ---------------------------------------------------------------------------
def run_tier1_batch_sieve(announcements):
    """
    Evaluates filings in chunks of 50 via Gemini Flash.
    Returns:
      hits (list[dict]): Filings meeting material catalyst criteria.
      rejections (list[dict]): Discarded filings with explicit reason explanations.
    """
    if not announcements or not gemini_tier1_client:
        return [], []

    hits = []
    rejections = []
    chunk_size = 50

    for i in range(0, len(announcements), chunk_size):
        chunk = announcements[i:i + chunk_size]

        items_payload = [
            {"index": idx, "exchange": ann['exchange'], "scrip": ann['scrip'], "company": ann['company'], "headline": ann['headline']}
            for idx, ann in enumerate(chunk)
        ]

        prompt = f"""
        You are a cynical Indian equity analyst screening corporate exchange announcements.
        Categorize EACH announcement as either "HIT" (high-materiality catalyst) or "REJECT" (routine noise).

        HIGH-MATERIALITY CATALYSTS (FLAG AS "HIT"):
        1. Financial Results & Guidance (Quarterly/Annual financial statements, revenue/margin guidance updates).
        2. Order Wins & Significant Contracts (Defense, Railways, Capex, OEM supply contracts).
        3. Fundraisings & M&A (Preferential issues, strategic warrants, QIP, acquisitions, demergers, open offers).
        4. Promoter Pledging (Substantial creation, revocation, or release of pledged shares).
        5. Regulatory Clearances & Approvals (USFDA EIR/approvals, CDSCO clearances, PLI scheme subsidies, patent grants).
        6. Credit Rating Actions (Upgrades or major downgrades by CRISIL, ICRA, CARE, India Ratings).
        7. Capex & Commercial Production (Commissioning of new plants, commercial production starts).
        8. Auditor Resignations & Forensics (Statutory auditor mid-term resignations, forensic audit reports).
        9. Debt Reduction Milestones (Company turning net debt-free, One-Time Settlement approvals).
        10. Tech Transfers & Foreign JVs (Technology licensing, global OEM tie-ups).
        11. Share Buybacks (Tender route buyback approvals).
        12. Environmental Clearances (EC / Consent to Operate approvals for major plants).

        CRITICAL EXCLUSIONS (FLAG AS "REJECT"):
        - Routine SEBI SAST Reg 29(1)/29(2) shareholding threshold crossings.
        - Routine SEBI PIT Reg 7(2) insider transaction disclosures.
        - Inter-promoter share transfers, family gifts, or transmission of shares.
        - Loss of share certificates / duplicate certificate requests.
        - Shareholding Patterns (Reg 31), Corporate Governance (Reg 27(2)), Secretarial Compliance (Reg 24A).
        - Intimations of upcoming Board Meeting dates (prior notices).
        - Trading window closure notices, newspaper clipping uploads, routine ESOP allotments.

        Respond with ONLY a valid JSON array of objects in this exact structure:
        [
          {{"index": 0, "status": "HIT", "reason": "Order win of INR 250 Cr from Ministry of Defense"}},
          {{"index": 1, "status": "REJECT", "reason": "Routine SEBI SAST 29(2) compliance disclosure"}}
        ]

        Announcements to screen:
        {json.dumps(items_payload, indent=2)}
        """

        try:
            response = gemini_tier1_client.models.generate_content(
                model=TIER1_MODEL,
                contents=prompt
            )
            raw_text = response.text.strip()

            # Clean markdown JSON block formatting if present
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\n", "", raw_text)
                raw_text = re.sub(r"\n```$", "", raw_text)

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
                    else:
                        ann_item["rejection_reason"] = reason
                        rejections.append(ann_item)
                        # Print rejection log to stdout for GitHub/Terminal inspection
                        print(f" [REJECTED] {ann_item['company']} ({ann_item['exchange']}:{ann_item['scrip']}) -> {reason}")

        except Exception as e:
            print(f"[Tier 1 Error] Sieve chunk evaluation failed: {e}")

    return hits, rejections


# ---------------------------------------------------------------------------
# 5. TIER 2 & TIER 3: CONCURRENT DUAL-MODEL AUDIT & FINANCIAL SCORING
# ---------------------------------------------------------------------------
def extract_numerical_score(text, score_label="Bullishness Score:"):
    """Extracts numerical ratings (1-10) from structured model outputs."""
    for line in text.split('\n'):
        if score_label in line:
            digits = [s for s in line.split() if s.isdigit() or '/' in s]
            if digits:
                try:
                    return int(digits[0].split('/')[0])
                except Exception:
                    pass
    return None


def evaluate_with_claude(prompt):
    """Executes deep qualitative reasoning and financial extraction via Claude Sonnet."""
    if not claude_client:
        return "Claude evaluation skipped: API key missing."
    try:
        response = claude_client.messages.create(
            model=TIER2_CLAUDE_MODEL,
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}]
        )
        text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return "\n".join(text_blocks).strip() if text_blocks else "Claude returned no text."
    except Exception as e:
        return f"Claude analysis error: {e}"


def evaluate_with_gemini(prompt):
    """Executes structured corporate assessment via Gemini Pro with progressive 503 retries."""
    if not gemini_paid_client:
        return "Gemini Pro evaluation skipped: API key missing."

    wait_times = [60, 120, 180]  # 1 min, 2 min, 3 min

    for attempt, wait_time in enumerate(wait_times + [None]):
        try:
            response = gemini_paid_client.models.generate_content(
                model=TIER2_GEMINI_MODEL,
                contents=prompt
            )
            return response.text.strip()

        except Exception as e:
            error_msg = str(e)

            # Check if it's a server overload error
            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                if wait_time is not None:
                    print(
                        f" [Gemini 503 Overload] High demand. Waiting {wait_time} seconds before Retry {attempt + 1}/3...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "Gemini Pro analysis error: Google Gemini was not able to perform the analysis due to high demand after 3 retries."

            # If it's a different error (e.g., 400 Bad Request), fail immediately
            return f"Gemini Pro analysis error: {e}"


def run_tier2_consensus_audit(company, exchange, scrip_code, isin, headline, catalyst_reason, forum_text, market_metrics):
    """Concurrently audits actionable announcements using Claude Sonnet and Gemini Pro."""
    prompt = f"""
    You are a cynical microcap portfolio manager. Analyze this corporate event:

    Company: {company} ({exchange}: {scrip_code} | ISIN: {isin})
    Current Price: INR {market_metrics['price']} | 20D Volume Multiple: {market_metrics['vol_multiple']}x
    Headline: {headline}
    Flagged Catalyst: {catalyst_reason}
    ValuePickr Sentiment / Forum Context: {forum_text}

    Evaluate:
    1. Strategic Thesis: Core economic rationale.
    2. Executive/Domain Value: Execution feasibility, order margin protection, or domain capability.
    3. Hype / AI-Washing Check: Check for buzzwords without committed capital or genuine cash flow.
    4. Forum Scuttlebutt: Synthesize community skepticism.
    5. Financial Results Audit (If applicable): Assess YoY/QoQ revenue, operating EBITDA margin expansion, and earnings quality.
    6. Financial Result Score (1 to 10): If financial results, provide numerical score. Otherwise state N/A.
    7. Bullishness Score (1 to 10): Provide a strict numerical score. IF the announcement is poor, routine, lacks economic substance, or looks like a red flag, you MUST default to a rating of 1/10.

    Output Format:
    **Core Thesis:** <2 sentences>
    **Domain & Operational Impact:** <Assessment N/A or>
    **Hype / Red Flag Check:** <Clean / Warning flags>
    **Community Sentiment:** <Summary>
    **Financial Result Score:** <X/10 1-sentence N/A breakdown or with>
    **Bullishness Score:** <X/10 - 1-sentence rationale>
    """

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_claude = executor.submit(evaluate_with_claude, prompt)
        future_gemini = executor.submit(evaluate_with_gemini, prompt)

        claude_output = future_claude.result()
        gemini_output = future_gemini.result()

    claude_score = extract_numerical_score(claude_output, "Bullishness Score:")
    gemini_score = extract_numerical_score(gemini_output, "Bullishness Score:")

    # FIX: Change the failure fallback from 5 (neutral) to 1 (bearish)
    c_score_safe = claude_score if claude_score is not None else 1
    g_score_safe = gemini_score if gemini_score is not None else 1

    is_divergent = (
        abs(c_score_safe - g_score_safe) >= 4 or
        (c_score_safe >= 7 and g_score_safe <= 4) or
        (g_score_safe >= 7 and c_score_safe <= 4)
    )

    if claude_score is None or gemini_score is None:
        consensus_status = "ANALYSIS_ERROR"
    elif is_divergent:
        consensus_status = "MODEL_DIVERGENCE"
    elif c_score_safe >= 7 and g_score_safe >= 7:
        consensus_status = "CONSENSUS_HIT"
    elif c_score_safe <= 4 and g_score_safe <= 4:
        consensus_status = "CONSENSUS_IGNORE"
    else:
        consensus_status = "NEUTRAL_MIX"

    final_score = round((c_score_safe + g_score_safe) / 2)
    is_high_conviction = (final_score >= 8 and market_metrics['vol_multiple'] >= 2.0 and market_metrics['above_50dma'])

    return {
        "final_score": final_score,
        "claude_score": claude_score or "N/A",
        "gemini_score": gemini_score or "N/A",
        "consensus_status": consensus_status,
        "claude_analysis": claude_output,
        "gemini_analysis": gemini_output,
        "high_conviction": is_high_conviction
    }


# ---------------------------------------------------------------------------
# 6. YOUTUBE TV INTERVIEW MODULE
# ---------------------------------------------------------------------------
def process_youtube_interviews(channel_id="UCb5hMTAFjG5j79V6nL3_YCQ"):
    """Scans TV interview transcripts for management guidance or capex updates."""
    if not gemini_tier1_client:
        return

    feed = feedparser.parse(f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}")
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)

    for entry in feed.entries[:5]:
        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        if published < cutoff:
            continue

        try:
            video_id = entry.yt_videoid
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([t['text'] for t in transcript[:120]])

            filter_prompt = f"""
            Does this corporate management interview contain guidance, margin revisions, or concrete capex plans?
            Headline: {entry.title}
            Transcript Preview: {text}
            Return strictly the COMPANY_NAME if YES, or return IGNORE if routine.
            """
            res = gemini_tier1_client.models.generate_content(
                model=TIER1_MODEL,
                contents=filter_prompt
            )
            if "IGNORE" not in res.text:
                msg = f"📺 *MANAGEMENT INTERVIEW CATALYST*\n\n**Title:** {entry.title}\n🔗 [Watch Interview]({entry.link})"
                send_telegram_alert(msg)
        except Exception:
            continue


# ---------------------------------------------------------------------------
# 7. TELEGRAM DISPATCHER & SCAN DIGEST
# ---------------------------------------------------------------------------
def send_telegram_alert(message):
    """Sends structured Markdown messages to the private Telegram channel."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram Alert Output]\n" + message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"[Telegram Dispatch Error] {e}")


def build_telegram_message(company, exchange, scrip_code, isin, market_data, audit, pdf_link):
    """Formats individual stock alerts based on model consensus."""
    if audit['consensus_status'] == "ANALYSIS_ERROR":
        banner = "⚠️ *MODEL ANALYSIS ERROR*"
    elif audit['consensus_status'] == "MODEL_DIVERGENCE":
        banner = "⚠️ *MODEL DIVERGENCE DETECTED*"
    elif audit['high_conviction']:
        banner = "🚨 *HIGH CONVICTION CONCURRENCE ALERT*"
    else:
        banner = "📢 *CORPORATE ACTION HIT*"

    return (
        f"{banner}\n"
        f"**Company:** {company}\n"
        f"**{exchange}:** `{scrip_code}` | **ISIN:** `{isin}`\n"
        f"**Price:** ₹{market_data['price']} | **20D Vol:** {market_data['vol_multiple']}x\n"
        f"**Consensus Status:** `{audit['consensus_status']}`\n\n"
        f"🧠 *Claude Analysis (Score: {audit['claude_score']}/10):*\n{audit['claude_analysis']}\n\n"
        f"🤖 *Gemini Analysis (Score: {audit['gemini_score']}/10):*\n{audit['gemini_analysis']}\n\n"
        f"📄 [View Official Filing PDF]({pdf_link})"
    )


def send_scan_digest(total_ingested, total_new, hits, rejections, duration_seconds):
    """Dispatches an executive summary of the screening run to Telegram."""
    mode_text = "🔄 *Manual Forced Refresh*" if IGNORE_CACHE else "⏰ *Scheduled Scan*"

    # Categorize rejections
    sast_count = sum(1 for r in rejections if any(k in r.get('rejection_reason', '').lower() for k in ['sast', 'pit', 'insider', 'shareholding', 'transfer']))
    admin_count = sum(1 for r in rejections if any(k in r.get('rejection_reason', '').lower() for k in ['certificate', 'meeting', 'governance', 'window', 'newspaper', 'secretarial', 'loss']))
    other_count = len(rejections) - (sast_count + admin_count)

    hit_lines = ""
    if hits:
        hit_lines = "\n\n🎯 *Actionable Candidates Passed to Sieve 2:*\n" + "\n".join([
            f"• *{h['company']}* (`{h['exchange']}:{h['scrip']}`) - {h.get('catalyst_reason', '')[:60]}"
            for h in hits
        ])

    digest_msg = (
        f"📊 *EXCHANGE SCAN COMPLETE*\n"
        f"• *Mode:* {mode_text}\n"
        f"• *Total Ingested:* {total_ingested} filings\n"
        f"• *New Screened:* {total_new}\n"
        f"• *Passed to Sieve 2:* {len(hits)} out of {total_new} ({round((len(hits)/max(total_new, 1))*100, 1)}%)\n"
        f"• *Execution Time:* {round(duration_seconds, 1)}s\n"
        f"{hit_lines}\n\n"
        f"🚫 *Discarded Noise Breakdown:* ({len(rejections)})\n"
        f"• SAST / PIT / Insider Transfers: {sast_count}\n"
        f"• Share Certificates / Meetings / Governance: {admin_count}\n"
        f"• Routine Compliance / Misc: {max(0, other_count)}"
    )
    send_telegram_alert(digest_msg)


# ---------------------------------------------------------------------------
# 8. BUSINESS WORKFLOW & DUAL-EXCHANGE INGESTION MODULES
# ---------------------------------------------------------------------------
def fetch_live_nse_filings():
    """Fetches NSE corporate announcements using a cookie-primed session."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.nseindia.com/'
    }
    session = requests.Session()
    session.headers.update(headers)
    standardized_filings = []

    try:
        session.get("https://www.nseindia.com", timeout=10)
        url = "https://www.nseindia.com/api/corporate-announcements?index=equities"
        resp = session.get(url, timeout=10)

        if resp.status_code == 200:
            data = resp.json().get('data', [])
            for item in data:
                attachment = item.get('attchmntFile') or item.get('attchmntText') or str(item.get('seq_id', ''))
                if not attachment:
                    continue

                pdf_link = item.get('attchmntText', '')
                if pdf_link and not pdf_link.startswith('http'):
                    pdf_link = f"https://www.nseindia.com{pdf_link}"

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
        print(f"[NSE Ingestion Error] {e}")

    return standardized_filings


def fetch_live_bse_filings(max_pages=100):
    """Paginates through the BSE API until the day's payload ends."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'origin': 'https://www.bseindia.com',
        'referer': 'https://www.bseindia.com/'
    }
    standardized_filings = []
    page = 1

    while page <= max_pages:
        url = (
            f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?"
            f"pageno={page}&strCat=-1&strPrevDate=&strScrip=&strSearch=P&strToDate=&strType=C"
        )
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                table = resp.json().get('Table', [])
                if not table:
                    break

                for item in table:
                    attachment = item.get('ATTACHMENTNAME')
                    if not attachment:
                        continue

                    pdf_link = f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{attachment}"
                    raw_isin = item.get('ISIN_CODE', '').strip()
                    clean_isin = raw_isin if raw_isin else 'N/A'

                    standardized_filings.append({
                        'id': attachment,
                        'company': item.get('SLONGNAME', 'Unknown Company'),
                        'scrip': str(item.get('SCRIP_CD', '')).strip(),
                        'headline': item.get('NEWSSUB', ''),
                        'isin': clean_isin,
                        'link': pdf_link,
                        'exchange': 'BSE'
                    })

                page += 1
                time.sleep(0.3)
            else:
                break
        except Exception as e:
            print(f"[BSE Ingestion Error on Page {page}] {e}")
            break

    return standardized_filings

def process_single_announcement(item):
    """Executes Sieve-2 dual-model audit and commits to permanent ledger and cache upon success."""
    print(f"\n[Deep Dive Audit] {item['company']} ({item['exchange']}: {item['scrip']})...")

    market_data = fetch_market_metrics(item['scrip'], item['exchange'])
    forum_scuttlebutt = fetch_valuepickr_sentiment(item['company'])

    audit = run_tier2_consensus_audit(
        company=item['company'],
        exchange=item['exchange'],
        scrip_code=item['scrip'],
        isin=item['isin'],
        headline=item['headline'],
        catalyst_reason=item.get('catalyst_reason', 'Actionable catalyst'),
        forum_text=forum_scuttlebutt,
        market_metrics=market_data
    )

    alert_msg = build_telegram_message(
        company=item['company'],
        exchange=item['exchange'],
        scrip_code=item['scrip'],
        isin=item['isin'],
        market_data=market_data,
        audit=audit,
        pdf_link=item['link']
    )

    send_telegram_alert(alert_msg)

    # Commit to DB only AFTER Sieve 2 succeeds
    log_permanent_ledger(
        attachment_id=item['id'],
        company=item['company'],
        scrip_code=item['scrip'],
        isin=item['isin'],
        price=market_data['price'],
        final_score=audit['final_score'],
        thesis=audit['claude_analysis'][:500],
        high_conviction=audit['high_conviction'],
        gemini_score=audit['gemini_score'] if audit['gemini_score'] != "N/A" else 0,
        claude_score=audit['claude_score'] if audit['claude_score'] != "N/A" else 0,
        consensus_status=audit['consensus_status']
    )
    log_announcements_batch([(item['id'], item['company'], item['headline'], "HIT")])


def evaluate_and_dispatch_filings(unprocessed_items):
    """Orchestrates Sieve 1 screening, logging, and Sieve 2 deep-dive dispatch."""
    start_time = time.time()

    hits, rejections = run_tier1_batch_sieve(unprocessed_items)

    # Stage 1 Persistence: Discarded items are immediately bulk-inserted to cache
    if rejections:
        rejection_tuples = [(r['id'], r['company'], r['headline'], "IGNORE") for r in rejections]
        log_announcements_batch(rejection_tuples)

    # Print upfront candidate ratio and names
    print(f"\n=======================================================")
    print(f"📊 SIEVE 1 SUMMARY: {len(hits)} out of {len(unprocessed_items)} passed to Sieve 2:")
    for h in hits:
        print(f" -> {h['company']} ({h['exchange']}:{h['scrip']}) | Catalyst: {h.get('catalyst_reason', '')}")
    print(f"=======================================================\n")

    # Stage 2: Deep Dive on Hits
    for item in hits:
        process_single_announcement(item)

    duration = time.time() - start_time

    # Send End-of-Scan Telegram Digest
    send_scan_digest(
        total_ingested=len(unprocessed_items),
        total_new=len(unprocessed_items),
        hits=hits,
        rejections=rejections,
        duration_seconds=duration
    )


# ---------------------------------------------------------------------------
# 9. MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------
def main():
    print(f"[{datetime.now()}] Initializing Market Intelligence Scan...")

    raw_bse = fetch_live_bse_filings(max_pages=args.max_pages)
    raw_nse = fetch_live_nse_filings()
    unified_filings = raw_bse + raw_nse

    print(f"Ingested {len(raw_bse)} BSE filings and {len(raw_nse)} NSE filings ({len(unified_filings)} total).")

    unprocessed_filings = filter_unprocessed_announcements(unified_filings)
    print(f"Evaluating {len(unprocessed_filings)} announcement(s).")

    if unprocessed_filings:
        evaluate_and_dispatch_filings(unprocessed_filings)
    else:
        print("No new announcements to evaluate.")

    process_youtube_interviews()
    print(f"[{datetime.now()}] Market intelligence scan completed successfully.")


if __name__ == "__main__":
    main()