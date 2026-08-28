"""
GitHub Actions CI/CD Pipeline Calibration Suite
===========================================================================
Functionality:
- Optimized for GitHub's ubuntu-latest public runners (4 vCPU / 16GB RAM).
- Sieve 1.5 is upgraded to 'qwen2.5:7b' for superior on-runner CPU reasoning.
- Sieve 1.5 prompt is strict: heavily penalizes PR fluff and non-binding updates.
- Complete exception handling and exponential backoff for cloud rate limits.
"""

import os
import re
import json
import time
import requests
from dotenv import load_dotenv
from google import genai
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# 0. CONFIGURATION & ENVIRONMENT SETUP
# ---------------------------------------------------------------------------
load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
# Upgraded to 7B class for GitHub's 16GB RAM runners
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_PAID") or os.getenv("GEMINI_API_KEY_FREE")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
claude_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

TIER1_MODEL = os.getenv("GEMINI_TIER1_MODEL", "gemini-3.5-flash-lite")
TIER2_CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
TIER2_GEMINI_MODEL = os.getenv("GEMINI_TIER2_MODEL", "gemini-3.1-pro-preview")

# ---------------------------------------------------------------------------
# 1. THE 10 CALIBRATION TEST CASES (1/10 TO 10/10)
# ---------------------------------------------------------------------------
CALIBRATION_TEST_CASES = [
    {
        "id": "TEST_01",
        "company": "Zenith Fibres Ltd",
        "expected_score": 1,
        "headline": "Intimation regarding loss of share certificates under Regulation 39(3)",
        "body": "Pursuant to Regulation 39(3) of SEBI (LODR) Regulations 2015, we wish to inform you that our registrar has received request for issue of duplicate share certificates from shareholders holding 50 shares due to loss of physical certificates."
    },
    {
        "id": "TEST_02",
        "company": "Kaveri Seed Company Ltd",
        "expected_score": 2,
        "headline": "Trading Window Closure Notice for Board Meeting",
        "body": "Notice is hereby given that pursuant to SEBI (Prohibition of Insider Trading) Regulations 2015, the trading window for dealing in securities of the company shall remain closed from October 1 2026 until 48 hours after the declaration of financial results."
    },
    {
        "id": "TEST_03",
        "company": "Siyaram Silk Mills Ltd",
        "expected_score": 3,
        "headline": "Disclosure under Regulation 29(2) of SEBI SAST Regulations",
        "body": "Disclosure by promoter group regarding acquisition of 15,000 equity shares (0.03% stake) via open market purchase on September 15 2026, bringing total promoter holding to 52.41%."
    },
    {
        "id": "TEST_04",
        "company": "Ducon Infratechnologies Ltd",
        "expected_score": 4,
        "headline": "Press Release: Company participates in global industrial trade fair in Hanover, Germany",
        "body": "We are pleased to announce that our executive management team participated as delegates at the international clean-tech exhibition in Hanover, showcasing our proprietary flue gas desulfurization technologies to global industry leaders."
    },
    {
        "id": "TEST_05",
        "company": "Titagarh Rail Systems Ltd",
        "expected_score": 5,
        "headline": "Secured domestic export order worth INR 22 Crores for specialized industrial components",
        "body": "The company has formally bagged a purchase order worth INR 22 Crores (inclusive of taxes) from an established domestic manufacturing enterprise for the supply of heavy engineering components, to be executed over a 6-month period."
    },
    {
        "id": "TEST_06",
        "company": "PNC Infratech Ltd",
        "expected_score": 6,
        "headline": "Received Letter of Award (LoA) for highway construction project worth INR 450 Crores from NHAI",
        "body": "We are pleased to inform that the company has been declared as L-1 bidder by National Highways Authority of India (NHAI) for an EPC road infrastructure project in Uttar Pradesh valued at INR 450 Crores, execution timeline 24 months."
    },
    {
        "id": "TEST_07",
        "company": "Gravita India Ltd",
        "expected_score": 7,
        "headline": "Debt Reduction Milestone: Company turns net debt-free ahead of institutional schedule",
        "body": "The company has successfully prepaid its remaining long-term institutional term loans amounting to INR 135 Crores out of internal accruals. With this prepayment, the company has officially achieved a net debt-free status on its balance sheet."
    },
    {
        "id": "TEST_08",
        "company": "KPI Green Energy Ltd",
        "expected_score": 8,
        "headline": "Strategic Expansion: Commercial production commences at new 100MW solar park 2 months ahead of schedule",
        "body": "Management is thrilled to announce the successful grid synchronization and commencement of commercial power generation at our new 100MW solar energy plant in Gujarat, expanding total operational capacity by 45% ahead of projected deadlines."
    },
    {
        "id": "TEST_09",
        "company": "Cochin Shipyard Ltd",
        "expected_score": 9,
        "headline": "Q3 Financial Results: Consolidated Revenue jumps 70% YoY, Net Profit surges 150% with guidance upgrade",
        "body": "Financial Results for Q3 FY27: Consolidated Revenue from Operations reached INR 1,420 Crores (up 70% YoY). Net Profit skyrocketed 150% YoY to INR 310 Crores. Board of Directors has also upgraded full-year revenue growth guidance from 30% to 50%."
    },
    {
        "id": "TEST_10",
        "company": "Kaynes Technology India Ltd",
        "expected_score": 10,
        "headline": "Definitive Agreement: Multi-year global EMS contract worth INR 1,500 Cr with Fortune 500 tech major + Warrant Allotment",
        "body": "The company has entered into a strategic 5-year definitive master services agreement worth INR 1,500 Crores with a leading US-based Fortune 500 semiconductor firm. Additionally, the board approved a preferential warrant issue of 4% equity stake at a 15% market premium."
    }
]


# ---------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def extract_score(text, label):
    if not text: return "ERR"
    for line in text.splitlines():
        if label.lower() in line.lower():
            match = re.search(r'(\b10|[1-9])\s*/\s*10', line)
            if match: return int(match.group(1))
            digits = re.findall(r'\b(10|[1-9])\b', line)
            if digits: return int(digits[0])
    return "ERR"


def get_sieve2_prompt(test_case):
    return f"""You are an institutional equity research analyst evaluating corporate exchange filings.
Assess how strongly this event will impact the company's future earnings power, business trajectory, and institutional market re-rating.

EVALUATION PRINCIPLES:
- Disregard market cap. Evaluate the PROPORTIONAL impact of the event on the company's business scale.
- Look for concrete economic catalysts: order inflows, capacity additions, balance sheet deleveraging, margin improvements, or revenue beats.

SCORING BENCHMARKS (1 to 10):
• 1 to 3: Share certificate losses, trading window notices, routine secretarial compliance.
• 4 to 6: Small/routine purchase orders, non-binding MoUs, standard conference participations.
• 7 to 8: Large firm order wins, full balance sheet deleveraging, major capacity commissioning, or substantial earnings beats (>40% growth).
• 9 to 10: Landmark multi-year global contracts, explosive earnings surges (>100%), game-changing strategic partnerships.

Company: {test_case['company']}
Headline: {test_case['headline']}
Filing Details: {test_case['body']}

OUTPUT FORMAT:
Reasoning: <1-2 sentences on proportional business impact and market reaction>
Catalyst Score: <Strictly an integer from 1 to 10>
"""


# ---------------------------------------------------------------------------
# 3. SIEVE EXECUTION WRAPPERS
# ---------------------------------------------------------------------------
def run_test_sieve1(test_case):
    if not gemini_client: return "ERR", 0.0
    start_time = time.perf_counter()
    prompt = f"""
    You are an objective exchange filing intake filter. 
    Classify this announcement as "REJECT" strictly if it is routine paperwork like share certificate loss, trading window closure, or board meeting date notice.
    If it mentions any commercial activity, contract win, financial result, order, capex, or business update, classify it as "HIT".
    Headline: {test_case['headline']}
    Respond strictly in JSON format: {{"status": "HIT" or "REJECT"}}
    """
    for attempt in range(3):
        try:
            res = gemini_client.models.generate_content(model=TIER1_MODEL, contents=prompt)
            text = res.text.strip().replace("```json", "").replace("```", "")
            return json.loads(text).get("status", "HIT").upper(), time.perf_counter() - start_time
        except Exception as e:
            if "503" in str(e) or "429" in str(e): time.sleep(2 ** attempt); continue
            return "ERR", time.perf_counter() - start_time
    return "ERR", time.perf_counter() - start_time


def run_test_sieve1_5(test_case):
    start_time = time.perf_counter()
    prompt = f"""
    You are a strict financial data extraction and pre-screening agent.
    Your job is to read the corporate filing and assign an objective PreScore from 1 to 10 based on concrete economic magnitude.

    CRITICAL RULES:
    - Penalize and score LOW (1-4): Routine press releases, trade expo participation, non-binding MoUs, and generic marketing fluff.
    - Score HIGH (7-10): Hard financial metrics, confirmed large contract wins, major capacity expansions, or net-debt reduction.

    Headline: {test_case['headline']}
    Body: {test_case['body']}

    Output format:
    Summary: [1 sentence]
    PreScore: [X/10]
    """
    try:
        res = requests.post(OLLAMA_API_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                                                  "options": {"temperature": 0.0}}, timeout=60)
        if res.status_code == 200:
            return extract_score(res.json().get("response", ""), "PreScore:"), time.perf_counter() - start_time
    except Exception:
        pass
    return "ERR", time.perf_counter() - start_time


def run_test_sieve2_claude(test_case):
    if not claude_client: return "ERR", 0.0
    start_time = time.perf_counter()
    for attempt in range(3):
        try:
            res = claude_client.messages.create(model=TIER2_CLAUDE_MODEL, max_tokens=400,
                                                messages=[{"role": "user", "content": get_sieve2_prompt(test_case)}])
            return extract_score("\n".join([b.text for b in res.content if getattr(b, "type", None) == "text"]).strip(),
                                 "Catalyst Score:"), time.perf_counter() - start_time
        except Exception as e:
            if "503" in str(e) or "429" in str(e) or "overloaded" in str(e): time.sleep(2 ** attempt); continue
            return "ERR", time.perf_counter() - start_time
    return "ERR", time.perf_counter() - start_time


def run_test_sieve2_gemini(test_case):
    if not gemini_client: return "ERR", 0.0
    start_time = time.perf_counter()
    for attempt in range(3):
        try:
            res = gemini_client.models.generate_content(model=TIER2_GEMINI_MODEL, contents=get_sieve2_prompt(test_case))
            return extract_score(res.text.strip(), "Catalyst Score:"), time.perf_counter() - start_time
        except Exception as e:
            if "503" in str(e) or "429" in str(e) or "quota" in str(e): time.sleep(2 ** attempt); continue
            return "ERR", time.perf_counter() - start_time
    return "ERR", time.perf_counter() - start_time


# ---------------------------------------------------------------------------
# 4. ORCHESTRATOR
# ---------------------------------------------------------------------------
def run_calibration_suite():
    print("=" * 125)
    print(f"🚀 GITHUB RUNNER CALIBRATION TEST (Ollama: {OLLAMA_MODEL})")
    print("=" * 125)
    print(
        f"{'Test ID & Company':<28} | {'Exp':<4} | {'S1 Status':<9} | {'Llama(1.5)':<11} | {'Claude S2':<11} | {'Gemini S2':<11} | {'Delta':<5} | {'Diagnostic Status'}")
    print("-" * 125)

    abs_errors = []

    for tc in CALIBRATION_TEST_CASES:
        expected = tc["expected_score"]
        s1_status, s1_time = run_test_sieve1(tc)
        llama_score, llama_time = run_test_sieve1_5(tc)
        claude_score, claude_time = run_test_sieve2_claude(tc)
        gemini_score, gemini_time = run_test_sieve2_gemini(tc)

        valid_scores = [s for s in (claude_score, gemini_score) if isinstance(s, int)]
        if valid_scores:
            avg = round(sum(valid_scores) / len(valid_scores))
            delta = avg - expected
            abs_errors.append(abs(delta))
            diag = "🚨 FALSE NEGATIVE" if expected >= 7 and avg <= 4 else "⚠️ FALSE POSITIVE" if expected <= 3 and avg >= 5 else "✅ Perfect Calibration" if abs(
                delta) <= 1 else "🔸 Moderate Variance"
        else:
            delta, diag = "N/A", "❌ API FAILURE"

        print(
            f"{tc['id']} - {tc['company'][:18]:<18} | {expected}/10{' ':<1} | {s1_status:<9} | {str(llama_score) + '/10':<11} | {str(claude_score) + '/10':<11} | {str(gemini_score) + '/10':<11} | {(f'{delta:+d}' if isinstance(delta, int) else str(delta)):<5} | {diag}")
        print(
            f"{' ':<28} | {' ':<4} | {' ':<9} | {' ':<11} | {' ':<11} | {' ':<11} | {' ':<5} | Total Latency: {(s1_time + llama_time + claude_time + gemini_time):.1f}s\n" + "-" * 125)


if __name__ == "__main__":
    import logging

    logging.getLogger("google").setLevel(logging.ERROR)  # Suppress AFC warnings
    run_calibration_suite()