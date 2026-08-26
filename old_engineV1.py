"""
Market Intelligence & Corporate Announcement Screening Engine
============================================================
Architecture:
- Ingestion: Live BSE corporate filings API (paginated) & YouTube TV management interviews
- Deduplication: Supabase PostgreSQL 7-day rolling cache
- Tier 1 Sieve: Batch filtering via Gemini 3.6 Flash (chunked for accuracy)
- Tier 2 Deep-Dive: Concurrent dual-model evaluation (Claude Sonnet 5 + Gemini 3.1 Pro)
- Consensus & Divergence: Automated detection of conflicting model convictions
- Tier 3 Quant Audit: 20-day delivery volume surges & 50/200 DMA trend checks
- Dispatcher: Structured Telegram alerts with rich Markdown formatting
- Permanent Ledger: Full audit trail for multi-horizon backtesting & prompt optimization
"""

import os
import sys
import re
import json
import time
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
# 0. CONFIGURATION & ENVIRONMENT SETUP
# ---------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY_FREE = os.getenv("GEMINI_API_KEY_FREE")
GEMINI_API_KEY_PAID = os.getenv("GEMINI_API_KEY_PAID")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SUPABASE_URL = os.getenv("SUPABASE_URL")

# Initialize SDK Clients
gemini_free_client = genai.Client(api_key=GEMINI_API_KEY_FREE) if GEMINI_API_KEY_FREE else None
gemini_paid_client = genai.Client(api_key=GEMINI_API_KEY_PAID or GEMINI_API_KEY_FREE) if (
    GEMINI_API_KEY_PAID or GEMINI_API_KEY_FREE
) else None
claude_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


# ---------------------------------------------------------------------------
# 1. SUPABASE DATABASE LAYER (CACHE & PERMANENT LEDGER)
# ---------------------------------------------------------------------------
def get_db_connection():
    """
    Establishes a connection to the Supabase PostgreSQL database.
    """
    if not SUPABASE_URL:
        print("[Database] SUPABASE_URL is not configured in .env.")
        return None
    try:
        return psycopg2.connect(SUPABASE_URL)
    except Exception as e:
        print(f"[Database Error] Connection failed: {e}")
        return None


def filter_unprocessed_announcements(bse_items):
    """
    Deduplicates incoming filings against the Supabase 7-day rolling cache table.
    """
    conn = get_db_connection()
    if not conn:
        return bse_items  # Fail-open strategy if database is unreachable

    try:
        cursor = conn.cursor()
        attachments = [item.get('ATTACHMENTNAME') for item in bse_items if item.get('ATTACHMENTNAME')]
        if not attachments:
            conn.close()
            return []

        format_strings = ','.join(['%s'] * len(attachments))
        query = f"SELECT bse_attachment_name FROM bse_announcements WHERE bse_attachment_name IN ({format_strings})"
        cursor.execute(query, tuple(attachments))

        existing = {row[0] for row in cursor.fetchall()}
        conn.close()
        return [item for item in bse_items if item.get('ATTACHMENTNAME') not in existing]
    except Exception as e:
        print(f"[Database Error] Deduplication query failed: {e}")
        return bse_items


def log_announcement_decision(attachment_name, company, headline, decision):
    """Records the triage decision ('HIT' or 'IGNORE') into the short-term cache table."""
    conn = get_db_connection()
    if not conn:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO bse_announcements (bse_attachment_name, company_name, headline, ai_decision) 
            VALUES (%s, %s, %s, %s) 
            ON CONFLICT (bse_attachment_name) DO NOTHING
            """,
            (attachment_name, company, headline, decision)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Database Error] Cache insertion failed for {attachment_name}: {e}")


def log_permanent_ledger(attachment_name, company, scrip_code, isin, price, final_score,
                         thesis, high_conviction, gemini_score, claude_score, consensus_status):
    """Inserts an evaluated HIT permanently into the audit ledger for historical tracking."""
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
            (attachment_name, company, scrip_code, isin, price, final_score,
             thesis, high_conviction, gemini_score, claude_score, consensus_status)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Database Error] Permanent ledger insert failed: {e}")


# ---------------------------------------------------------------------------
# 2. MARKET DATA & QUANT CHECKS (PRICE, VOLUME SURGE, DMA)
# ---------------------------------------------------------------------------
def fetch_market_metrics(scrip_code):
    """Fetches market metrics (price, volume surge multiple, moving averages) using Yahoo Finance."""
    default_payload = {"price": 0.0, "vol_multiple": 1.0, "above_50dma": False, "above_200dma": False}
    if not scrip_code:
        return default_payload

    try:
        ticker = f"{scrip_code}.BO"
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
        print(f"[Market Data Warning] Scrip {scrip_code}: {e}")
        return default_payload


# ---------------------------------------------------------------------------
# 3. SCUTTLEBUTT: VALUEPICKR FORUM PARSER
# ---------------------------------------------------------------------------
def fetch_valuepickr_sentiment(company_name):
    """Searches the ValuePickr Discourse forum API for retail discussion and community skepticism."""
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
# 4. TIER 1: BATCH FILTER SIEVE (GEMINI FLASH)
# ---------------------------------------------------------------------------
def run_tier1_batch_sieve(announcements):
    """
    Chunks filings into sets of 50 and makes high-speed calls to Gemini Flash.
    Uses regex to safely extract the returned indices.
    """
    if not announcements or not gemini_free_client:
        return []

    all_hits = []
    chunk_size = 50

    for i in range(0, len(announcements), chunk_size):
        chunk = announcements[i:i + chunk_size]

        numbered_list = "\n".join([
            f"{idx}. [{ann.get('SCRIP_CD', 'N/A')}] {ann.get('NEWSSUB', '')}"
            for idx, ann in enumerate(chunk)
        ])

        prompt = f"""
        You are an equity analyst screening Indian corporate exchange announcements.
        Identify items that match ANY of these high-materiality catalysts:
        1. Open Offers, Mergers, Downstream/Upstream Acquisitions.
        2. Preferential Allotments, Warrants to strategic investors.
        3. Promoter Open Market Buying or Creeping Acquisition.
        4. Strategic Leadership Hires (Executive hires from DRDO, Railways, Defense, PSUs, Tier-1 MNCs).
        5. Auditor Resignations (Severe Red Flag).

        Discard routine compliance notices, standard earnings calendars, and investor presentation uploads.

        Output Format: Return ONLY a comma-separated list of the matching integer indices (e.g., 0, 4, 12). 
        If no announcements qualify, return strictly: NONE.

        Announcements:
        {numbered_list}
        """

        try:
            response = gemini_free_client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt
            )
            output = response.text.strip().upper()

            if "NONE" in output or not output:
                continue

            # Robust regex parsing: Extracts only integers from the response
            indices = [int(x) for x in re.findall(r'\d+', output)]
            all_hits.extend([chunk[idx] for idx in indices if idx < len(chunk)])

        except Exception as e:
            print(f"[Tier 1 Error] Batch sieve chunk failed: {e}")

    return all_hits


# ---------------------------------------------------------------------------
# 5. TIER 2 & TIER 3: CONCURRENT DUAL-MODEL AUDIT
# ---------------------------------------------------------------------------
def extract_numerical_score(text):
    """Extracts the 1-10 numerical bullishness score from model structured responses."""
    for line in text.split('\n'):
        if "Bullishness Score:" in line:
            digits = [s for s in line.split() if s.isdigit() or '/' in s]
            if digits:
                try:
                    return int(digits[0].split('/')[0])
                except Exception:
                    pass
    return 5


def evaluate_with_claude(prompt):
    """Executes deep qualitative reasoning and sentiment extraction via Claude Sonnet."""
    if not claude_client:
        return "Claude evaluation skipped: API key missing."
    try:
        response = claude_client.messages.create(
            model="claude-sonnet-5",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"Claude analysis error: {e}"


def evaluate_with_gemini(prompt):
    """Executes structured corporate assessment via Gemini Pro."""
    if not gemini_paid_client:
        return "Gemini Pro evaluation skipped: API key missing."
    try:
        response = gemini_paid_client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Gemini Pro analysis error: {e}"


def run_tier2_consensus_audit(company, scrip_code, isin, headline, forum_text, market_metrics):
    """Runs Claude and Gemini concurrently, and checks quantitative conviction triggers."""
    prompt = f"""
    You are a cynical microcap portfolio manager. Analyze this corporate event:

    Company: {company} (BSE: {scrip_code} | ISIN: {isin})
    Current Price: INR {market_metrics['price']} | 20D Volume Multiple: {market_metrics['vol_multiple']}x
    Headline: {headline}
    ValuePickr Sentiment / Forum Context: {forum_text}

    Evaluate:
    1. Strategic Thesis: Core economic rationale.
    2. Executive/Domain Value: Procurement power or domain capability of key hires (if applicable).
    3. Hype / AI-Washing Check: Check for buzzwords without capital commitment.
    4. Forum Scuttlebutt: Synthesize community skepticism.
    5. Bullishness Score (1 to 10): Provide a strict numerical score.

    Output Format:
    **Core Thesis:** <2 sentences>
    **Domain & Executive Impact:** <Assessment or N/A>
    **Hype / Red Flag Check:** <Clean / Warning flags>
    **Community Sentiment:** <Summary>
    **Bullishness Score:** <X/10 - with 1-sentence rationale>
    """

    # Parallel asynchronous model invocation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_claude = executor.submit(evaluate_with_claude, prompt)
        future_gemini = executor.submit(evaluate_with_gemini, prompt)

        claude_output = future_claude.result()
        gemini_output = future_gemini.result()

    claude_score = extract_numerical_score(claude_output)
    gemini_score = extract_numerical_score(gemini_output)

    # Divergence Classification
    is_divergent = (
        abs(claude_score - gemini_score) >= 4 or
        (claude_score >= 7 and gemini_score <= 4) or
        (gemini_score >= 7 and claude_score <= 4)
    )

    if is_divergent:
        consensus_status = "MODEL_DIVERGENCE"
    elif claude_score >= 7 and gemini_score >= 7:
        consensus_status = "CONSENSUS_HIT"
    elif claude_score <= 4 and gemini_score <= 4:
        consensus_status = "CONSENSUS_IGNORE"
    else:
        consensus_status = "NEUTRAL_MIX"

    final_score = round((claude_score + gemini_score) / 2)

    is_high_conviction = (
        final_score >= 8 and
        market_metrics['vol_multiple'] >= 2.0 and
        market_metrics['above_50dma']
    )

    return {
        "final_score": final_score,
        "claude_score": claude_score,
        "gemini_score": gemini_score,
        "consensus_status": consensus_status,
        "claude_analysis": claude_output,
        "gemini_analysis": gemini_output,
        "high_conviction": is_high_conviction
    }


# ---------------------------------------------------------------------------
# 6. YOUTUBE TV INTERVIEW MODULE
# ---------------------------------------------------------------------------
def process_youtube_interviews(channel_id="UCb5hMTAFjG5j79V6nL3_YCQ"):
    """Parses recent management TV interviews for guidance revisions or capex updates."""
    if not gemini_free_client:
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
            res = gemini_free_client.models.generate_content(
                model='gemini-3.6-flash',
                contents=filter_prompt
            )

            if "IGNORE" not in res.text:
                msg = f"📺 *MANAGEMENT INTERVIEW CATALYST*\n\n**Title:** {entry.title}\n🔗 [Watch Interview]({entry.link})"
                send_telegram_alert(msg)
        except Exception:
            continue


# ---------------------------------------------------------------------------
# 7. TELEGRAM DISPATCHER & MESSAGE FORMATTER
# ---------------------------------------------------------------------------
def send_telegram_alert(message):
    """Sends a formatted Markdown payload to the designated private Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram Alert Output (Local Fallback)]\n" + message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False
    }
    try:
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code != 200:
            print(f"[Telegram Error] HTTP {res.status_code}: {res.text}")
    except Exception as e:
        print(f"[Telegram Dispatch Error] {e}")


def build_telegram_message(company, scrip_code, isin, market_data, audit, pdf_link):
    """Constructs a structured Markdown alert based on consensus and conviction levels."""
    if audit['consensus_status'] == "MODEL_DIVERGENCE":
        banner = "⚠️ *MODEL DIVERGENCE DETECTED*"
    elif audit['high_conviction']:
        banner = "🚨 *HIGH CONVICTION CONCURRENCE ALERT*"
    else:
        banner = "📢 *BSE CORPORATE ACTION HIT*"

    return (
        f"{banner}\n"
        f"**Company:** {company}\n"
        f"**Scrip:** `{scrip_code}` | **ISIN:** `{isin}`\n"
        f"**Price:** ₹{market_data['price']} | **20D Vol:** {market_data['vol_multiple']}x\n"
        f"**Consensus Status:** `{audit['consensus_status']}`\n\n"
        f"🧠 *Claude Analysis (Score: {audit['claude_score']}/10):*\n{audit['claude_analysis']}\n\n"
        f"🤖 *Gemini Analysis (Score: {audit['gemini_score']}/10):*\n{audit['gemini_analysis']}\n\n"
        f"📄 [View Official BSE Filing PDF]({pdf_link})"
    )


# ---------------------------------------------------------------------------
# 8. BUSINESS WORKFLOW & PIPELINE METHODS
# ---------------------------------------------------------------------------
def old_Not_used_fetch_live_bse_filings(max_pages=3):
    """Paginates through recent BSE corporate announcement pages to ingest live filings."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'origin': 'https://www.bseindia.com',
        'referer': 'https://www.bseindia.com/'
    }
    all_filings = []

    for page in range(1, max_pages + 1):
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
                all_filings.extend(table)
                time.sleep(0.3)
            else:
                break
        except Exception as e:
            print(f"[BSE Ingestion Error on Page {page}] {e}")
            break

    return all_filings


def fetch_live_bse_filings(max_pages=100):
    """
    Dynamically paginates through BSE corporate announcements.
    Continues fetching until the API returns an empty table, ensuring all recent filings are captured.
    Includes a fail-safe ceiling (max_pages=100, approx 5,000 filings) to prevent infinite loops.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'origin': 'https://www.bseindia.com',
        'referer': 'https://www.bseindia.com/'
    }
    all_filings = []
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

                # Dynamic Break: If the payload is empty, we have hit the end of the filings
                if not table:
                    print(f"Reached end of BSE filings at page {page}.")
                    break

                all_filings.extend(table)
                page += 1
                time.sleep(0.3)  # Respectful rate limiting
            else:
                print(f"BSE API returned HTTP {resp.status_code} on page {page}.")
                break
        except Exception as e:
            print(f"[BSE Ingestion Error on Page {page}] {e}")
            break

    return all_filings


def process_single_announcement(item):
    """Executes the full pipeline for an identified actionable filing."""
    attachment = item.get('ATTACHMENTNAME')
    company = item.get('SLONGNAME', 'Unknown Company')
    scrip_code = str(item.get('SCRIP_CD', '')).strip()
    headline = item.get('NEWSSUB', '')
    isin = item.get('ISIN_CODE', 'N/A')
    pdf_link = f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{attachment}"

    print(f"\n[Deep Dive Audit] {company} ({scrip_code})...")

    # Quant metrics & community sentiment
    market_data = fetch_market_metrics(scrip_code)
    forum_scuttlebutt = fetch_valuepickr_sentiment(company)

    # Dual-model Tier 2 & Tier 3 consensus evaluation
    audit = run_tier2_consensus_audit(company, scrip_code, isin, headline, forum_scuttlebutt, market_data)

    # Dispatch Alert
    alert_msg = build_telegram_message(company, scrip_code, isin, market_data, audit, pdf_link)
    send_telegram_alert(alert_msg)

    # Database Persistence
    log_announcement_decision(attachment, company, headline, "HIT")
    log_permanent_ledger(
        attachment_name=attachment,
        company=company,
        scrip_code=scrip_code,
        isin=isin,
        price=market_data['price'],
        final_score=audit['final_score'],
        thesis=audit['claude_analysis'][:500],
        high_conviction=audit['high_conviction'],
        gemini_score=audit['gemini_score'],
        claude_score=audit['claude_score'],
        consensus_status=audit['consensus_status']
    )


def evaluate_and_dispatch_filings(unprocessed_items):
    """Runs the Tier 1 batch sieve and routes items to the Deep-Dive or Cache."""
    hits = run_tier1_batch_sieve(unprocessed_items)
    hit_attachments = {h.get('ATTACHMENTNAME') for h in hits}

    print(f"Tier 1 Sieve flagged {len(hits)} actionable catalyst(s).")

    for item in unprocessed_items:
        attachment = item.get('ATTACHMENTNAME')
        company = item.get('SLONGNAME', 'Unknown Company')
        headline = item.get('NEWSSUB', '')

        if attachment in hit_attachments:
            process_single_announcement(item)
        else:
            log_announcement_decision(attachment, company, headline, "IGNORE")


# ---------------------------------------------------------------------------
# 9. MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------
def main():
    print(f"[{datetime.now()}] Initializing Market Intelligence Scan...")

    # Step 1: Ingest live filings across pages
    raw_filings = fetch_live_bse_filings(max_pages=3)
    print(f"Ingested {len(raw_filings)} total corporate filings from BSE.")

    # Step 2: Deduplicate against Supabase short-term cache
    unprocessed_filings = filter_unprocessed_announcements(raw_filings)
    print(f"Found {len(unprocessed_filings)} new, unprocessed announcement(s).")

    if not unprocessed_filings:
        print("No new announcements to evaluate. Exiting scan.")
        return

    # Step 3: Run filtering, consensus analysis, and alerting
    evaluate_and_dispatch_filings(unprocessed_filings)

    # Step 4: Scan YouTube management interview transcripts
    process_youtube_interviews()
    print(f"[{datetime.now()}] Market intelligence scan completed successfully.")


if __name__ == "__main__":
    main()