"""
Market Intelligence & Corporate Announcement Screening Engine
============================================================
Architecture:
- Ingestion: Live BSE & NSE corporate filings APIs (limitless loops) + YouTube TV
- Anti-Scraping: Persistent cookie sessions for NSE Cloudflare bypass
- Deduplication: Supabase PostgreSQL 7-day rolling cache
- Tier 1 Sieve: Batch filtering via Gemini 3.6 Flash (ruthlessly tuned against SEBI noise)
- Tier 2 Deep-Dive: Concurrent dual-model evaluation (Claude Sonnet 5 + Gemini 3.1 Pro)
- Consensus & Divergence: Automated detection of conflicting model convictions
- Tier 3 Quant Audit: 20-day delivery volume surges & 50/200 DMA trend checks
- Dispatcher: Structured Telegram alerts with rich Markdown formatting
- Permanent Ledger: Full audit trail for multi-horizon backtesting & prompt optimization
"""

import os
import re
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
    if not SUPABASE_URL:
        print("[Database] SUPABASE_URL is not configured in .env.")
        return None
    try:
        return psycopg2.connect(SUPABASE_URL)
    except Exception as e:
        print(f"[Database Error] Connection failed: {e}")
        return None


def filter_unprocessed_announcements(filings):
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
            ON CONFLICT (bse_attachment_name) DO NOTHING
        """
        cursor.executemany(query, decisions_list)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Database Error] Batch cache insertion failed: {e}")


def log_permanent_ledger(attachment_id, company, scrip_code, isin, price, final_score,
                         thesis, high_conviction, gemini_score, claude_score, consensus_status):
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
    if not announcements or not gemini_free_client:
        return []

    all_hits = []
    chunk_size = 50

    for i in range(0, len(announcements), chunk_size):
        chunk = announcements[i:i + chunk_size]

        numbered_list = "\n".join([
            f"{idx}. [{ann['exchange']}:{ann['scrip']}] {ann['headline']}"
            for idx, ann in enumerate(chunk)
        ])

        # FIX: Heavily tuned prompt to ban SEBI SAST and routine regulatory disclosures
        prompt = f"""
        You are a ruthless equity analyst screening Indian corporate exchange announcements.
        Your job is to identify high-materiality catalysts while STRICTLY IGNORING routine compliance filings.

        Identify items that match ANY of these high-materiality catalysts:
        1. Open Offers, Mergers, Demergers, Downstream/Upstream Acquisitions.
        2. Preferential Allotments, Warrants issuance to strategic investors.
        3. SIGNIFICANT Promoter Open Market Buying (Must be a major strategic buy, NOT a minor compliance disclosure).
        4. Strategic Leadership Hires (Executive hires from DRDO, Railways, Defense, PSUs, Tier-1 MNCs).
        5. Auditor Resignations (Severe Red Flag).

        CRITICAL EXCLUSIONS - You MUST DISCARD the following routine noise:
        - Routine SEBI SAST Regulation 29(1) and 29(2) disclosures (small threshold crossings).
        - Routine SEBI PIT Regulation 7(2) disclosures (minor insider trades).
        - Inter-promoter share transfers, family gifts, or transmission of shares.
        - Loss of share certificates / Issue of duplicate certificates.
        - ESOP (Employee Stock Option) allotments or exercises.
        - Closure of trading window, earnings calendars, investor presentation uploads, or AGM/EGM notices.

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

            indices = [int(x) for x in re.findall(r'\d+', output)]
            all_hits.extend([chunk[idx] for idx in indices if idx < len(chunk)])

        except Exception as e:
            print(f"[Tier 1 Error] Batch sieve chunk failed: {e}")

    return all_hits


# ---------------------------------------------------------------------------
# 5. TIER 2 & TIER 3: CONCURRENT DUAL-MODEL AUDIT
# ---------------------------------------------------------------------------
def extract_numerical_score(text):
    for line in text.split('\n'):
        if "Bullishness Score:" in line:
            digits = [s for s in line.split() if s.isdigit() or '/' in s]
            if digits:
                try:
                    return int(digits[0].split('/')[0])
                except Exception:
                    pass
    return None


def evaluate_with_claude(prompt):
    if not claude_client:
        return "Claude evaluation skipped: API key missing."
    try:
        response = claude_client.messages.create(
            model="claude-sonnet-5",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return "\n".join(text_blocks).strip() if text_blocks else "Claude returned no text."
    except Exception as e:
        return f"Claude analysis error: {e}"


def evaluate_with_gemini(prompt):
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


def run_tier2_consensus_audit(company, exchange, scrip_code, isin, headline, forum_text, market_metrics):
    prompt = f"""
    You are a cynical microcap portfolio manager. Analyze this corporate event:

    Company: {company} ({exchange}: {scrip_code} | ISIN: {isin})
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

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_claude = executor.submit(evaluate_with_claude, prompt)
        future_gemini = executor.submit(evaluate_with_gemini, prompt)

        claude_output = future_claude.result()
        gemini_output = future_gemini.result()

    claude_score = extract_numerical_score(claude_output)
    gemini_score = extract_numerical_score(gemini_output)

    c_score_safe = claude_score if claude_score is not None else 5
    g_score_safe = gemini_score if gemini_score is not None else 5

    is_divergent = (
            abs(c_score_safe - g_score_safe) >= 4 or
            (c_score_safe >= 7 and g_score_safe <= 4) or
            (g_score_safe >= 7 and c_score_safe <= 4)
    )

    if claude_score is None or gemini_score is None:
        consensus_status = "ANALYSIS_ERROR"
        final_score = c_score_safe if gemini_score is None else (g_score_safe if claude_score is None else 5)
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
# 7. TELEGRAM DISPATCHER
# ---------------------------------------------------------------------------
def send_telegram_alert(message):
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
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        pass


def build_telegram_message(company, exchange, scrip_code, isin, market_data, audit, pdf_link):
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


# ---------------------------------------------------------------------------
# 8. BUSINESS WORKFLOW & DUAL-EXCHANGE INGESTION MODULES
# ---------------------------------------------------------------------------
def fetch_live_nse_filings():
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
                    print(f"Reached end of BSE filings at page {page}.")
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
            print(f"[BSE Ingestion Error] {e}")
            break

    return standardized_filings


def process_single_announcement(item):
    print(f"\n[Deep Dive Audit] {item['company']} ({item['exchange']}: {item['scrip']})...")

    market_data = fetch_market_metrics(item['scrip'], item['exchange'])
    forum_scuttlebutt = fetch_valuepickr_sentiment(item['company'])

    audit = run_tier2_consensus_audit(
        company=item['company'],
        exchange=item['exchange'],
        scrip_code=item['scrip'],
        isin=item['isin'],
        headline=item['headline'],
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


def evaluate_and_dispatch_filings(unprocessed_items):
    hits = run_tier1_batch_sieve(unprocessed_items)
    hit_ids = {h['id'] for h in hits}

    # FIX: Upfront printing of companies that passed the sieve
    print(f"\n=======================================================")
    print(f"Tier 1 Sieve flagged {len(hits)} actionable catalyst(s):")
    for h in hits:
        print(f" -> {h['company']} ({h['exchange']}:{h['scrip']})")
        print(f"    {h['headline'][:120]}...")
    print(f"=======================================================\n")

    batch_cache_data = []

    for item in unprocessed_items:
        if item['id'] in hit_ids:
            process_single_announcement(item)
            batch_cache_data.append((item['id'], item['company'], item['headline'], "HIT"))
        else:
            batch_cache_data.append((item['id'], item['company'], item['headline'], "IGNORE"))

    if batch_cache_data:
        log_announcements_batch(batch_cache_data)


# ---------------------------------------------------------------------------
# 9. MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------
def main():
    print(f"[{datetime.now()}] Initializing Market Intelligence Scan...")

    raw_bse = fetch_live_bse_filings()
    raw_nse = fetch_live_nse_filings()
    unified_filings = raw_bse + raw_nse

    print(f"Ingested {len(raw_bse)} BSE filings and {len(raw_nse)} NSE filings.")

    unprocessed_filings = filter_unprocessed_announcements(unified_filings)
    print(f"Found {len(unprocessed_filings)} new, unprocessed announcement(s).")

    if unprocessed_filings:
        evaluate_and_dispatch_filings(unprocessed_filings)
    else:
        print("No new announcements to evaluate.")

    process_youtube_interviews()
    print(f"[{datetime.now()}] Market intelligence scan completed successfully.")


if __name__ == "__main__":
    main()