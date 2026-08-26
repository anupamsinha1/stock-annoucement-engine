"""
Market Intelligence & Corporate Announcement Screening Engine
============================================================
Architecture:
- Ingestion: Live BSE & NSE corporate filings APIs (safe sequential loops) + YouTube TV
- Grouping: Pre-groups simultaneous filings by company for holistic evaluation
- Deduplication: Supabase PostgreSQL 7-day rolling cache (with bypass flag)
- Tier 1 Sieve: Batch filtering via Gemini 3.5 Flash Lite with rejection reason logs
- Tier 2 Deep-Dive: 8-Worker Sharded Priority Queues (4 Claude + 4 Gemini)
- Configurable Alerting: Optional Telegram alerts on single-model filtered items (default: true)
- Rate Limit Resilience: Exponential backoff retries for Claude (429/529) & Gemini (503/429)
- Dual Scoring: Independent 'Catalyst Impact Score' & 'Company Quality Score' (1-10)
- Price Artifact Guard: Automatic neutralization of 0.00 price feed sync latency
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
import queue
import itertools
import threading
import argparse
from datetime import datetime, timedelta, timezone
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

parser = argparse.ArgumentParser(description="Corporate Announcement Screening Engine")
parser.add_argument("--ignore-cache", "-f", action="store_true", help="Bypass Supabase cache and re-evaluate all filings")
parser.add_argument("--max-pages", type=int, default=100, help="Maximum BSE announcement pages to fetch")
args, _ = parser.parse_known_args()

IGNORE_CACHE = args.ignore_cache or os.getenv("IGNORE_CACHE", "false").lower() in ("true", "1", "yes")

# Configurable flag: Default to True if missing or set to true
ALERT_ON_SINGLE_MODEL_IGNORE = os.getenv("ALERT_ON_SINGLE_MODEL_IGNORE", "true").lower() in ("true", "1", "yes")

GEMINI_API_KEY_FREE = os.getenv("GEMINI_API_KEY_FREE")
GEMINI_API_KEY_PAID = os.getenv("GEMINI_API_KEY_PAID") or GEMINI_API_KEY_FREE
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SUPABASE_URL = os.getenv("SUPABASE_URL")

# Models
TIER1_MODEL = os.getenv("GEMINI_TIER1_MODEL", "gemini-3.5-flash-lite")
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
    """
    Consolidates simultaneous filings for the same company into a single unified event.
    Prevents redundant model evaluations and provides holistic context.
    """
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
                        print(f" [REJECTED] {ann_item['company']} ({ann_item['exchange']}:{ann_item['scrip']}) -> {reason}")

        except Exception as e:
            print(f"[Tier 1 Error] Sieve chunk evaluation failed: {e}")

    return hits, rejections


# ---------------------------------------------------------------------------
# 5. TIER 2: MODEL EVALUATORS & PROMPT HARDENING
# ---------------------------------------------------------------------------
def extract_score(text, label):
    """Safely extracts integer score (1-10) for a given label in the response text."""
    if not text:
        return None
    for line in text.splitlines():
        if label.lower() in line.lower():
            match = re.search(r'(\b10|[1-9])\s*/\s*10', line)
            if match:
                try:
                    return int(match.group(1))
                except Exception:
                    pass
            digits = re.findall(r'\b(10|[1-9])\b', line)
            if digits:
                try:
                    return int(digits[0])
                except Exception:
                    pass
    return None


def build_sieve2_prompt(item, market_data, forum_text):
    """Constructs hardened evaluation prompt with data-feed artifact protection."""
    price_display = f"INR {market_data['price']}" if market_data['price'] > 0 else "Data Feed In Sync (Live quote unavailable - evaluate purely on strategic/economic merits)"
    mkt_cap_display = f"INR {market_data['market_cap_cr']} Cr" if market_data.get('market_cap_cr', 0) > 0 else "Not specified"

    return f"""
    You are a cynical microcap portfolio manager. Analyze this corporate event:

    Company: {item['company']} ({item['exchange']}: {item['scrip']} | ISIN: {item['isin']})
    Current Price: {price_display} | 20D Volume Multiple: {market_data['vol_multiple']}x | Est. Market Cap: {mkt_cap_display}
    Headline: {item['headline']}
    Flagged Catalyst: {item.get('catalyst_reason', 'Actionable corporate action')}
    ValuePickr Sentiment / Forum Context: {forum_text}

    GUIDELINES:
    - If Price is listed as 0.00 or "Data Feed In Sync", treat this strictly as an external data feed latency artifact. DO NOT assume illiquidity or suspension.
    - Evaluate both the specific announcement AND the underlying business strength independently.

    Evaluate:
    1. Strategic Thesis: Core economic rationale of the announcement.
    2. Executive & Operational Feasibility: Execution capability, margin protection, or technical validation.
    3. Hype / Red Flag Check: Check for buzzwords without committed capital or genuine cash flow.
    4. Community Sentiment: Synthesize forum skepticism.
    5. Financial Results Audit (If applicable): Assess YoY/QoQ revenue, operating EBITDA margin expansion, and earnings quality.
    6. Financial Result Score: If financial results, provide strictly X/10. Otherwise state N/A.
    7. Catalyst Score (MANDATORY): Strictly an integer from 1 to 10 (e.g., 8/10). If the event has no merit, lacks economic substance, or is routine, you MUST output strictly 1/10. Do NOT output N/A or text.
    8. Company Quality Score (MANDATORY): Strictly an integer from 1 to 10 (e.g., 6/10) evaluating core business durability, governance, and capital efficiency. If unknown, output 5/10. Do NOT output N/A or text.

    Output Format:
    **Core Thesis:** <2 sentences>
    **Domain & Operational Impact:** <Assessment N/A or>
    **Hype / Red Flag Check:** <Clean / Warning flags>
    **Community Sentiment:** <Summary>
    **Financial Result Score:** <X/10 1-sentence N/A breakdown or with>
    **Catalyst Score:** <Strictly - 1-sentence X/10 rationale>
    **Company Quality Score:** <Strictly - 1-sentence X/10 business core evaluation of strength>
    """


def evaluate_with_claude(prompt):
    """Executes deep qualitative reasoning via Claude Sonnet with progressive 429/529 retries."""
    if not claude_client:
        return "Claude evaluation skipped: API key missing."

    wait_times = [5, 15, 30, 60]

    for attempt, wait_time in enumerate(wait_times + [None]):
        try:
            response = claude_client.messages.create(
                model=TIER2_CLAUDE_MODEL,
                max_tokens=900,
                messages=[{"role": "user", "content": prompt}]
            )
            text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            return "\n".join(text_blocks).strip() if text_blocks else "Claude returned no text."
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower() or "529" in error_msg or "overloaded" in error_msg.lower():
                if wait_time is not None:
                    print(f" [Claude Rate Limit/Overload] Waiting {wait_time}s before Retry {attempt + 1}/4...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Claude analysis error: Rate limit or overload sustained after retries ({e})"
            return f"Claude analysis error: {e}"


def evaluate_with_gemini(prompt):
    """Executes structured corporate assessment via Gemini Pro with progressive 503/429 retries."""
    if not gemini_paid_client:
        return "Gemini Pro evaluation skipped: API key missing."

    wait_times = [15, 30, 60, 120]

    for attempt, wait_time in enumerate(wait_times + [None]):
        try:
            response = gemini_paid_client.models.generate_content(
                model=TIER2_GEMINI_MODEL,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if wait_time is not None:
                    print(f" [Gemini 503/429 Overload] High demand. Waiting {wait_time}s before Retry {attempt + 1}/4...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "Gemini Pro analysis error: Google Gemini was not able to perform the analysis due to high demand after retries."
            return f"Gemini Pro analysis error: {e}"


# ---------------------------------------------------------------------------
# 6. 8-WORKER SHARDED PRIORITY QUEUE WORKER POOL & CROSS-EVALUATION
# ---------------------------------------------------------------------------
def finalize_dual_evaluation(item, evals, market_data):
    """Finalizes dual-model consensus calculation, logs to permanent ledger, and dispatches alert."""
    c_cat = evals.get('claude_catalyst_score', 1)
    g_cat = evals.get('gemini_catalyst_score', 1)
    c_comp = evals.get('claude_company_score', 1)
    g_comp = evals.get('gemini_company_score', 1)

    is_divergent = (
        abs(c_cat - g_cat) >= 4 or
        (c_cat >= 7 and g_cat <= 4) or
        (g_cat >= 7 and c_cat <= 4)
    )

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

    alert_msg = build_telegram_message(
        company=item['company'],
        exchange=item['exchange'],
        scrip_code=item['scrip'],
        isin=item['isin'],
        market_data=market_data,
        audit=audit,
        all_links=item['all_links']
    )
    send_telegram_alert(alert_msg)

    # Log primary ID to permanent ledger
    log_permanent_ledger(
        attachment_id=item['id'],
        company=item['company'],
        scrip_code=item['scrip'],
        isin=item['isin'],
        price=market_data['price'],
        final_score=final_score,
        thesis=evals.get('claude_analysis', evals.get('gemini_analysis', ''))[:500],
        high_conviction=is_high_conviction,
        gemini_score=g_cat,
        claude_score=c_cat,
        consensus_status=consensus_status
    )

    # Log ALL associated filing IDs for this company as HIT
    hit_tuples = [(att_id, item['company'], item['headline'], "HIT") for att_id in item['all_ids']]
    log_announcements_batch(hit_tuples)


def finalize_single_model_ignore(item, evals, market_data, source_model):
    """
    Handles filings scoring <= 4 on the first model.
    Optionally sends a Telegram notification based on ALERT_ON_SINGLE_MODEL_IGNORE.
    """
    audit = {
        'consensus_status': "SINGLE_MODEL_IGNORE",
        'final_score': evals.get(f'{source_model.lower()}_catalyst_score', 1),
        'high_conviction': False,
        'claude_catalyst_score': evals.get('claude_catalyst_score'),
        'gemini_catalyst_score': evals.get('gemini_catalyst_score'),
        'claude_company_score': evals.get('claude_company_score'),
        'gemini_company_score': evals.get('gemini_company_score'),
        'claude_analysis': evals.get('claude_analysis', ''),
        'gemini_analysis': evals.get('gemini_analysis', '')
    }

    if ALERT_ON_SINGLE_MODEL_IGNORE:
        alert_msg = build_telegram_message(
            company=item['company'],
            exchange=item['exchange'],
            scrip_code=item['scrip'],
            isin=item['isin'],
            market_data=market_data,
            audit=audit,
            all_links=item['all_links']
        )
        send_telegram_alert(alert_msg)

    # Log to cache as IGNORE
    ignore_tuples = [(att_id, item['company'], item['headline'], "IGNORE") for att_id in item['all_ids']]
    log_announcements_batch(ignore_tuples)


def run_sharded_sieve2_workers(hits, num_workers=4):
    """
    Shards Sieve-2 candidates into two priority worker pools (4 Claude + 4 Gemini threads).
    Cross-evaluates filing via second model ONLY if primary score > 4.
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

    # Round-robin initial assignment (Priority 1 = Standard)
    for idx, hit in enumerate(hits):
        task_payload = {'item': hit, 'stage': 1, 'evals': {}}
        if idx % 2 == 0:
            claude_queue.put((1, next(counter), task_payload))
        else:
            gemini_queue.put((1, next(counter), task_payload))

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
            task['market_data'] = market_data
            task['forum_data'] = forum_data

            print(f"\n[Claude Worker-{worker_id} Stage {stage}] Evaluating {item['company']} ({item['exchange']}:{item['scrip']})...")
            prompt = build_sieve2_prompt(item, market_data, forum_data)
            claude_output = evaluate_with_claude(prompt)

            cat_score = extract_score(claude_output, "Catalyst Score:")
            comp_score = extract_score(claude_output, "Company Quality Score:")

            cat_score_safe = cat_score if cat_score is not None else 1
            comp_score_safe = comp_score if comp_score is not None else 1

            evals['claude_catalyst_score'] = cat_score_safe
            evals['claude_company_score'] = comp_score_safe
            evals['claude_analysis'] = claude_output

            if stage == 1:
                if cat_score_safe > 4:
                    print(f" -> [Claude Worker-{worker_id} Catalyst Score {cat_score_safe}/10 > 4] Pushing {item['company']} to Gemini Queue (Priority 0 URGENT)...")
                    gemini_queue.put((0, next(counter), {'item': item, 'stage': 2, 'evals': evals, 'market_data': market_data, 'forum_data': forum_data}))
                else:
                    print(f" -> [Claude Worker-{worker_id} Catalyst Score {cat_score_safe}/10 <= 4] Finalizing {item['company']} as SINGLE_MODEL_IGNORE.")
                    finalize_single_model_ignore(item, evals, market_data, source_model="claude")

                    with completed_lock:
                        completed_count += 1
                        if completed_count >= total_items:
                            stop_event.set()
            elif stage == 2:
                print(f" -> [Dual Evaluation Complete] {item['company']} finalized by Claude Worker-{worker_id}.")
                finalize_dual_evaluation(item, evals, market_data)
                with completed_lock:
                    completed_count += 1
                    if completed_count >= total_items:
                        stop_event.set()

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
            task['market_data'] = market_data
            task['forum_data'] = forum_data

            print(f"\n[Gemini Worker-{worker_id} Stage {stage}] Evaluating {item['company']} ({item['exchange']}:{item['scrip']})...")
            prompt = build_sieve2_prompt(item, market_data, forum_data)
            gemini_output = evaluate_with_gemini(prompt)

            cat_score = extract_score(gemini_output, "Catalyst Score:")
            comp_score = extract_score(gemini_output, "Company Quality Score:")

            cat_score_safe = cat_score if cat_score is not None else 1
            comp_score_safe = comp_score if comp_score is not None else 1

            evals['gemini_catalyst_score'] = cat_score_safe
            evals['gemini_company_score'] = comp_score_safe
            evals['gemini_analysis'] = gemini_output

            if stage == 1:
                if cat_score_safe > 4:
                    print(f" -> [Gemini Worker-{worker_id} Catalyst Score {cat_score_safe}/10 > 4] Pushing {item['company']} to Claude Queue (Priority 0 URGENT)...")
                    claude_queue.put((0, next(counter), {'item': item, 'stage': 2, 'evals': evals, 'market_data': market_data, 'forum_data': forum_data}))
                else:
                    print(f" -> [Gemini Worker-{worker_id} Catalyst Score {cat_score_safe}/10 <= 4] Finalizing {item['company']} as SINGLE_MODEL_IGNORE.")
                    finalize_single_model_ignore(item, evals, market_data, source_model="gemini")

                    with completed_lock:
                        completed_count += 1
                        if completed_count >= total_items:
                            stop_event.set()
            elif stage == 2:
                print(f" -> [Dual Evaluation Complete] {item['company']} finalized by Gemini Worker-{worker_id}.")
                finalize_dual_evaluation(item, evals, market_data)
                with completed_lock:
                    completed_count += 1
                    if completed_count >= total_items:
                        stop_event.set()

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
# 7. TELEGRAM DISPATCHER & MESSAGE FORMATTER
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


def build_telegram_message(company, exchange, scrip_code, isin, market_data, audit, all_links):
    """Formats individual stock alerts strictly adhering to the specified visual hierarchy."""
    status = audit['consensus_status']
    if status == "ANALYSIS_ERROR":
        banner = "⚠️ *MODEL ANALYSIS ERROR*"
    elif status == "MODEL_DIVERGENCE":
        banner = "⚠️ *MODEL DIVERGENCE DETECTED*"
    elif status == "SINGLE_MODEL_IGNORE":
        banner = "🚫 *FILTERED / LOW CONVICTION (SINGLE MODEL)*"
    elif audit['high_conviction']:
        banner = "🚨 *HIGH CONVICTION CONCURRENCE ALERT*"
    else:
        banner = "📢 *CORPORATE ACTION HIT*"

    price_str = f"₹{market_data['price']}"
    if market_data.get('price_feed_sync') or market_data['price'] == 0.0:
        price_str = "₹0.0 (Data Feed In Sync)"

    c_cat = audit.get('claude_catalyst_score')
    g_cat = audit.get('gemini_catalyst_score')
    cat_scores = []
    if c_cat is not None:
        cat_scores.append(f"Claude: {c_cat}/10")
    if g_cat is not None:
        cat_scores.append(f"Gemini: {g_cat}/10")
    cat_line = " | ".join(cat_scores) if cat_scores else "N/A"

    c_comp = audit.get('claude_company_score')
    g_comp = audit.get('gemini_company_score')
    comp_scores = []
    if c_comp is not None:
        comp_scores.append(f"Claude: {c_comp}/10")
    if g_comp is not None:
        comp_scores.append(f"Gemini: {g_comp}/10")
    comp_line = " | ".join(comp_scores) if comp_scores else "N/A"

    msg = (
        f"{banner}\n"
        f"**Company:** {company}\n"
        f"**{exchange}:** `{scrip_code}` | **ISIN:** `{isin}`\n"
        f"**Price:** {price_str} | **20D Vol:** {market_data['vol_multiple']}x\n"
        f"**Consensus Status:** `{status}`\n\n"
        f"🎯 **Catalyst Score:** {cat_line}\n"
        f"🏢 **Company Quality:** {comp_line}\n"
    )

    if audit.get('claude_analysis'):
        msg += f"\n🧠 *Claude Detailed Analysis:*\n{audit['claude_analysis']}\n"
    if audit.get('gemini_analysis'):
        msg += f"\n🤖 *Gemini Detailed Analysis:*\n{audit['gemini_analysis']}\n"

    msg += "\n" + "\n".join([f"📄 [View Official Filing PDF {i + 1}]({link})" for i, link in enumerate(all_links)])
    return msg


def send_scan_digest(total_ingested, total_new, hits, rejections, duration_seconds):
    """Dispatches an executive summary of the screening run to Telegram."""
    mode_text = "🔄 *Manual Forced Refresh*" if IGNORE_CACHE else "⏰ *Scheduled Scan*"

    sast_count = sum(1 for r in rejections if any(
        k in r.get('rejection_reason', '').lower() for k in ['sast', 'pit', 'insider', 'shareholding', 'transfer']))
    admin_count = sum(1 for r in rejections if any(k in r.get('rejection_reason', '').lower() for k in
                                                   ['certificate', 'meeting', 'governance', 'window', 'newspaper',
                                                    'secretarial', 'loss']))
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
        f"• *Total Ingested:* {total_ingested} unique filings\n"
        f"• *New Screened:* {total_new} grouped entity updates\n"
        f"• *Passed to Sieve 2:* {len(hits)} out of {total_new} ({round((len(hits) / max(total_new, 1)) * 100, 1)}%)\n"
        f"• *Execution Time:* {round(duration_seconds, 1)}s\n"
        f"{hit_lines}\n\n"
        f"🚫 *Discarded Noise Breakdown:* ({len(rejections)})\n"
        f"• SAST / PIT / Insider Transfers: {sast_count}\n"
        f"• Share Certificates / Meetings / Governance: {admin_count}\n"
        f"• Routine Compliance / Misc: {max(0, other_count)}"
    )
    send_telegram_alert(digest_msg)


# ---------------------------------------------------------------------------
# 8. YOUTUBE TV INTERVIEW MODULE
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
# 9. DUAL-EXCHANGE INGESTION MODULES
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
            json_payload = resp.json()
            data = json_payload if isinstance(json_payload, list) else json_payload.get('data', [])

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


# ---------------------------------------------------------------------------
# 10. MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    print(f"[{datetime.now()}] Initializing Market Intelligence Scan...")

    raw_bse = fetch_live_bse_filings(max_pages=args.max_pages)
    raw_nse = fetch_live_nse_filings()
    unified_filings = raw_bse + raw_nse

    print(f"Ingested {len(raw_bse)} BSE filings and {len(raw_nse)} NSE filings ({len(unified_filings)} total).")

    unprocessed_filings = filter_unprocessed_announcements(unified_filings)
    print(f"Filtering complete. {len(unprocessed_filings)} new individual filings found.")

    if not unprocessed_filings:
        print("No new announcements to evaluate. Exiting.")
        return

    grouped_filings = group_filings_by_company(unprocessed_filings)
    print(f"Consolidated into {len(grouped_filings)} distinct company events for evaluation.")

    # Tier 1 Sieve Execution
    hits, rejections = run_tier1_batch_sieve(grouped_filings)

    # Discarded items logged to cache immediately (extracting all consolidated IDs)
    if rejections:
        rejection_tuples = []
        for r in rejections:
            for att_id in r['all_ids']:
                rejection_tuples.append((att_id, r['company'], r['headline'], "IGNORE"))
        log_announcements_batch(rejection_tuples)

    print(f"\n=======================================================")
    print(f"📊 SIEVE 1 SUMMARY: {len(hits)} out of {len(grouped_filings)} passed to Sieve 2:")
    for h in hits:
        print(f" -> {h['company']} ({h['exchange']}:{h['scrip']}) | Catalyst: {h.get('catalyst_reason', '')}")
    print(f"=======================================================\n")

    # Tier 2 Sharded Queue Execution (4 Claude Workers + 4 Gemini Workers)
    if hits:
        run_sharded_sieve2_workers(hits, num_workers=4)

    duration = time.time() - start_time

    # Send End-of-Scan Telegram Digest
    send_scan_digest(
        total_ingested=len(unified_filings),
        total_new=len(grouped_filings),
        hits=hits,
        rejections=rejections,
        duration_seconds=duration
    )

    process_youtube_interviews()
    print(f"[{datetime.now()}] Market intelligence scan completed successfully.")


if __name__ == "__main__":
    main()