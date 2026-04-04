#!/usr/bin/env python3
"""
AXIS_PMS Scanner v2.0 — Gamma API + DeepSeek AI Divergence Scanner
Rebuilt: March 19, 2026
Replaces v1.0 which used PolymarketScan ai-vs-humans endpoint (20 market cap, near-zero divergence).

New approach:
  - Pull markets directly from Polymarket Gamma API (up to 200 markets)
  - Filter by volume, liquidity, active status, time-to-resolution
  - Batch markets and send to DeepSeek for AI probability estimation
  - Compare AI estimate to crowd price — flag divergences >= threshold
  - Check PolymarketScan whales endpoint for secondary confirmation

Usage:
  python3 axis_pms_scan_v2.py               # Full scan, log paper trades for RED signals
  python3 axis_pms_scan_v2.py --dry-run     # Scan but don't log trades
  python3 axis_pms_scan_v2.py --volume-report  # Weekly volume tracker (separate function)

Data sources:
  - Primary: Polymarket Gamma API (gamma-api.polymarket.com) — REQUIRES User-Agent header
  - AI: DeepSeek API (api.deepseek.com) — env var DEEPSEEK_API_KEY
  - Secondary: PolymarketScan whales (Supabase endpoint, no auth)

Cost target: < $0.05 per scan when no signals found (batched LLM calls)
"""

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# API endpoints
GAMMA_API = "https://gamma-api.polymarket.com/markets"
PMS_WHALES = "https://gzydspfquuaudqeztorw.supabase.co/functions/v1/agent-api"
DEEPSEEK_API = "https://api.deepseek.com/chat/completions"

# Required headers for Gamma API (403 without User-Agent)
GAMMA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Signal thresholds (percentage points)
RED_GAP_MIN = 15.0          # RED signal: >= 15% gap (with whale confirmation)
RED_GAP_NO_WHALE = 20.0     # RED signal without whale confirmation needs higher gap
YELLOW_GAP_MIN = 10.0       # YELLOW signal: >= 10% gap
GREEN_GAP_MIN = 7.0         # GREEN: log only, no alert

# Market filters
VOLUME_24H_MIN = 10000      # Minimum $10K 24h volume
LIQUIDITY_MIN = 1000        # Minimum $1K liquidity
CROWD_CEILING = 0.95        # Skip markets with crowd > 95%
CROWD_FLOOR = 0.05          # Skip markets with crowd < 5%
SPORTS_CROWD_MAX = 0.75     # Don't invert sports when crowd is >75% confident on one side
NOISE_SLUGS = [
    "elon-musk-of-tweets",
    "elon-musk-tweet",
    "backpack-fdv",
    "gensyn-fdv",
    "fdv-above",
    "fdv-below",
    "token-fdv",
    "mrbeasts-next-video",
    "mrbeast-video",
    "views-on-day-1",
    # Crypto price markets — 0% WR on 6 resolved trades, no AI edge
    "will-bitcoin-", "will-btc-", "bitcoin-above", "bitcoin-below",
    "will-ethereum-", "will-eth-", "ethereum-above", "ethereum-below",
    "will-solana-", "will-sol-", "solana-above", "solana-below",
    "will-crypto-", "crypto-above", "crypto-below",
    "bitcoin-dip", "bitcoin-reach", "ethereum-dip", "ethereum-reach",
    "ethereum-above-", "ethereum-price",
]
NOISE_QUESTION_PATTERNS = [
    r"elon musk.*tweet",
    r"tweets? from .* to .*202",
    r"how many tweets",
    r"fdv above",
    r"fdv below",
    r"one day after launch",
    r"mrbeast.*video.*views",
    r"views on day 1",
    # Crypto price patterns — blocked due to 0% WR
    r"will bitcoin (reach|dip|hit|drop)",
    r"will ethereum (reach|dip|hit|drop|be above)",
    r"will solana (reach|dip|hit|drop)",
    r"bitcoin (above|below|price|at) \$",
    r"ethereum (above|below|price|at) \$",
    r"get between .* million views",
    r"token.*fdv",
    r"market cap.*one day after",
]
BLOCKED_SPORTS_IN_PMS = True  # Sports routed to AXIS_CROW as of 2026-04-01
SPORTS_SLUG_PREFIXES = [
    "nba-", "nfl-", "nhl-", "mlb-", "cbb-", "cfb-", "epl-",
    "atp-", "wta-", "crint-", "mma-", "boxing-", "f1-",
    "nascar-", "soccer-", "rugby-",
]
MIN_HOURS_TO_RESOLVE = 24   # Skip markets resolving within 24h
MAX_RESOLUTION_DAYS = 90    # Skip markets resolving more than 90 days out
THEME_DEDUP_WINDOW_H = 48   # Suppress themes with 2+ signals in this window
THEME_DEDUP_MAX = 2          # Max signals per theme in the window

# Scanning parameters
MARKETS_PER_PAGE = 50       # Gamma API page size
MAX_PAGES = 4               # Up to 200 markets total
BATCH_SIZE = 5              # Markets per DeepSeek call (reduce API cost)
REQUEST_DELAY = 2.0         # Seconds between API calls

# DeepSeek model
DEEPSEEK_MODEL = "deepseek-chat"

# File paths (inside OpenClaw container on Hetzner)
WORKSPACE = os.environ.get("AXIS_WORKSPACE", "/home/node/.openclaw/workspace")
HEALTH_FILE = os.path.join(WORKSPACE, "memory", "scan-health-axis_pms.json")
PAPER_TRADES = os.path.join(WORKSPACE, "memory", "paper-trades.md")
SCAN_LOG = os.path.join(WORKSPACE, "memory", "axis_pms_scan.log")
THEME_LOG = os.path.join(WORKSPACE, "memory", "theme_dedup.json")


# ─── API HELPERS ──────────────────────────────────────────────────────────────

def http_get(url, headers=None, timeout=15):
    """HTTP GET with error handling. Returns parsed JSON or None."""
    hdrs = dict(GAMMA_HEADERS)
    if headers:
        hdrs.update(headers)
    try:
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        log(f"HTTP {e.code} from {url[:80]}")
        return None
    except Exception as e:
        log(f"GET failed: {e} — {url[:80]}")
        return None


def deepseek_estimate(markets, api_key):
    """
    Send a batch of market questions to DeepSeek and get probability estimates.
    Returns list of floats (0.0-1.0) or None on failure.
    """
    if not api_key:
        log("ERROR: DEEPSEEK_API_KEY not set")
        return None

    # Build numbered question list
    questions = "\n".join(
        f"{i+1}. {m['question']}" for i, m in enumerate(markets)
    )

    system_prompt = (
        "You are a calibrated prediction market probability estimator. "
        "Today's date is " + datetime.now(timezone.utc).strftime("%Y-%m-%d") + ". "
        "For each numbered market question below, estimate the TRUE probability of the YES outcome "
        "as a decimal between 0.0 and 1.0. "
        "Use base rates, current events knowledge, and logical reasoning. "
        "IMPORTANT: If you are genuinely uncertain about a market (e.g. niche elections, "
        "obscure sports matchups, specific YouTube metrics, or events you have no data on), "
        "return null for that position — do NOT guess 0.65 or any default value. "
        "Only return a probability if you have real signal. "
        "Return ONLY a JSON array in the same order as the questions. "
        "Use null for uncertain markets. "
        "Example for 3 questions where you are uncertain about #2: [0.45, null, 0.67]\n"
        "Do NOT include any text, explanation, or markdown — just the JSON array."
    )

    payload = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions}
        ],
        "temperature": 0.3,
        "max_tokens": 200
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "AXIS-Scanner/2.0"
    }

    try:
        req = urllib.request.Request(DEEPSEEK_API, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Strip markdown fences if present
        content = content.strip().strip("`").strip()
        if content.startswith("json"):
            content = content[4:].strip()

        probs = json.loads(content)
        if isinstance(probs, list) and len(probs) == len(markets):
            # Clamp to 0.0-1.0, preserve None for uncertain markets
            result = []
            for idx, p in enumerate(probs):
                if p is None:
                    result.append(None)
                    continue
                try:
                    result.append(max(0.0, min(1.0, float(p))))
                except (TypeError, ValueError):
                    log(f"SKIP (no AI estimate): market index {idx} in batch")
                    result.append(None)
            return result
        else:
            log(f"DeepSeek returned {len(probs) if isinstance(probs, list) else 'non-list'}, expected {len(markets)}")
            return None

    except json.JSONDecodeError as e:
        log(f"DeepSeek JSON parse failed: {e}")
        return None
    except Exception as e:
        log(f"DeepSeek API error: {e}")
        return None


# ─── MARKET FETCHING ──────────────────────────────────────────────────────────

def fetch_markets():
    """
    Fetch active, open markets from Gamma API sorted by 24h volume descending.
    Returns list of market dicts, up to MAX_PAGES * MARKETS_PER_PAGE.
    """
    all_markets = []

    for page in range(MAX_PAGES):
        offset = page * MARKETS_PER_PAGE
        url = (
            f"{GAMMA_API}?limit={MARKETS_PER_PAGE}&offset={offset}"
            f"&active=true&closed=false"
            f"&order=volume24hr&ascending=false"
        )
        data = http_get(url)
        if not data or not isinstance(data, list):
            log(f"Gamma API page {page} returned no data, stopping pagination")
            break

        all_markets.extend(data)
        log(f"Fetched page {page}: {len(data)} markets")

        if len(data) < MARKETS_PER_PAGE:
            break  # Last page
        time.sleep(REQUEST_DELAY)

    return all_markets


def fetch_whale_markets():
    """
    Fetch currently whale-active market slugs from PolymarketScan.
    Returns a set of slug strings.
    """
    url = f"{PMS_WHALES}?action=whales&limit=50"
    data = http_get(url)
    if not data:
        return set()

    # Handle both list and {ok, data} response formats
    markets = data if isinstance(data, list) else data.get("data", [])
    slugs = set()
    for m in markets:
        slug = m.get("slug") or m.get("conditionId") or ""
        if slug:
            slugs.add(slug)
    return slugs


# ─── MARKET FILTERING ─────────────────────────────────────────────────────────

def filter_market(m):
    """
    Apply all filters to a single market. Returns (pass: bool, reason: str).
    """
    # Volume check
    vol24h = float(m.get("volume24hr") or 0)
    if vol24h < VOLUME_24H_MIN:
        return False, f"low_volume ({vol24h:.0f})"

    # Liquidity check
    liq = float(m.get("liquidity") or m.get("liquidityNum") or 0)
    if liq < LIQUIDITY_MIN:
        return False, f"low_liquidity ({liq:.0f})"

    # Parse outcome prices
    try:
        prices = json.loads(m.get("outcomePrices", "[]"))
        crowd_price = float(prices[0]) if prices else 0
    except (json.JSONDecodeError, ValueError, IndexError):
        return False, "price_parse_error"

    # Near-certain markets
    if crowd_price >= CROWD_CEILING:
        return False, f"near_certain_yes ({crowd_price:.2f})"
    if crowd_price <= CROWD_FLOOR:
        return False, f"near_certain_no ({crowd_price:.2f})"

    # Already resolved (price exactly 0 or 1)
    if crowd_price == 0.0 or crowd_price == 1.0:
        return False, "already_resolved"

    # Time to resolution
    end_date_str = m.get("endDate") or m.get("endDateIso")
    if end_date_str:
        try:
            # Handle various date formats
            if "T" in end_date_str:
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            else:
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            hours_left = (end_date - datetime.now(timezone.utc)).total_seconds() / 3600
            if hours_left < MIN_HOURS_TO_RESOLVE:
                return False, f"resolving_soon ({hours_left:.0f}h)"
            days_left = hours_left / 24
            if days_left > MAX_RESOLUTION_DAYS:
                return False, f"RESOLUTION_GATE: {m.get('slug', 'unknown')} — {days_left:.0f} days exceeds limit"
        except (ValueError, TypeError):
            pass  # Can't parse date, don't filter

    # Noise market check — categories where AI has no structural edge
    slug_lower = m.get("slug", "").lower()
    q_lower_fm = m.get("question", "").lower()
    for pattern in NOISE_SLUGS:
        if pattern in slug_lower:
            return False, f"noise_market ({pattern})"
    for pattern in NOISE_QUESTION_PATTERNS:
        if pattern in q_lower_fm:
            return False, f"noise_market_question ({pattern})"
    # Sports routed to AXIS_CROW — killed from PMS 2026-04-01
    if BLOCKED_SPORTS_IN_PMS:
        for prefix in SPORTS_SLUG_PREFIXES:
            if slug_lower.startswith(prefix) or f"-{prefix.rstrip('-')}-" in slug_lower:
                return False, f"sports_routed_to_crow ({prefix})"
    return True, "pass"


def parse_crowd_price(m):
    """Extract YES price from outcomePrices field."""
    try:
        prices = json.loads(m.get("outcomePrices", "[]"))
        return float(prices[0]) if prices else None
    except (json.JSONDecodeError, ValueError, IndexError):
        return None


# ─── SIGNAL CLASSIFICATION ────────────────────────────────────────────────────

def classify_signal(ai_estimate, crowd_price, whale_confirmed):
    """
    Classify divergence into RED/YELLOW/GREEN/SKIP.
    Returns (tier, gap_pct) or (None, gap_pct) for skip.
    """
    gap_pct = abs(ai_estimate - crowd_price) * 100

    if gap_pct >= RED_GAP_NO_WHALE:
        return "RED", gap_pct
    elif gap_pct >= RED_GAP_MIN and whale_confirmed:
        return "RED", gap_pct
    elif gap_pct >= YELLOW_GAP_MIN:
        return "YELLOW", gap_pct
    elif gap_pct >= GREEN_GAP_MIN:
        return "GREEN", gap_pct
    else:
        return None, gap_pct


def determine_direction(ai_estimate, crowd_price):
    """Determine trade direction based on divergence."""
    if ai_estimate > crowd_price:
        return "BUY YES"
    else:
        return "BUY NO"


# ─── PAPER TRADE LOGGING ─────────────────────────────────────────────────────

def load_existing_slugs():
    """Load all slugs already recorded in paper-trades.md to prevent duplicates."""
    slugs = set()
    try:
        with open(PAPER_TRADES, "r") as f:
            for line in f:
                if line.startswith("- Slug:"):
                    slug = line.split(":", 1)[1].strip()
                    if slug:
                        slugs.add(slug)
    except FileNotFoundError:
        pass
    except Exception as e:
        log(f"WARNING: Could not read paper-trades.md for dedup: {e}")
    return slugs


def log_paper_trade(signal):
    """Append a paper trade entry to paper-trades.md using deterministic Python."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = (
        f"\n## Paper Trade — {now}\n"
        f"- Scanner: AXIS_PMS v2.1\n"
        f"- Market: {signal['question']}\n"
        f"- Slug: {signal['slug']}\n"
        f"- Direction: {signal['direction']}\n"
        f"- AI Estimate: {signal['ai_estimate']:.1%}\n"
        f"- Crowd Price: {signal['crowd_price']:.1%}\n"
        f"- Divergence: {signal['gap_pct']:.1f}%\n"
        f"- Signal Tier: {signal['tier']}\n"
        f"- Whale Confirmed: {signal.get('whale_confirmed', False)}\n"
        f"- Volume 24h: ${signal.get('volume24h', 0):,.0f}\n"
        f"- Resolution Criteria: {signal.get('description', '')[:300]}\n"
        f"- Outcome: PENDING\n"
        f"\n"
    )

    try:
        os.makedirs(os.path.dirname(PAPER_TRADES), exist_ok=True)
        with open(PAPER_TRADES, "a") as f:
            f.write(entry)
        log(f"Paper trade logged: {signal['direction']} on {signal['slug']}")
    except Exception as e:
        log(f"ERROR: Failed to log paper trade: {e}")


# ─── HEALTH FILE ──────────────────────────────────────────────────────────────

def write_health(markets_scanned, signals_found, red_count, yellow_count, status="OK", error_msg=None):
    """Write health JSON for Mother watchdog compatibility."""
    health = {
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "scanner": "axis_pms_v2",
        "markets_scanned": markets_scanned,
        "signals_found": signals_found,
        "red_signals": red_count,
        "yellow_signals": yellow_count,
        "scan_status": status,
    }
    if error_msg:
        health["error"] = error_msg

    try:
        os.makedirs(os.path.dirname(HEALTH_FILE), exist_ok=True)
        with open(HEALTH_FILE, "w") as f:
            json.dump(health, f, indent=2)
    except Exception as e:
        log(f"ERROR: Failed to write health file: {e}")


# ─── LOGGING ──────────────────────────────────────────────────────────────────

def log(msg):
    """Print timestamped log line to stderr (wrapper captures stdout for signals)."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"LOG|[{ts}] {msg}", file=sys.stderr)


def signal_output(tier, ai_est, crowd, gap, question, slug, direction, volume24h, whale):
    """Print structured signal line to stdout for wrapper parsing."""
    print(f"SIGNAL|{tier}|{ai_est:.4f}|{crowd:.4f}|{gap:.1f}|{question}|{slug}|{direction}|{volume24h:.0f}|{whale}")


# ─── THEMATIC CLUSTER DEDUP ───────────────────────────────────────────────────

THEME_PATTERNS = [
    (r"iran|iranian|tehran|irgc|khamenei", "iran-conflict"),
    (r"ukraine|ukrainian|kyiv|zelensky|donbas|crimea", "ukraine-conflict"),
    (r"russia|russian|putin|kremlin|moscow", "russia-geopolitics"),
    (r"israel|israeli|idf|netanyahu|gaza|hamas|hezbollah", "israel-conflict"),
    (r"china|chinese|beijing|xi jinping|taiwan|pla", "china-geopolitics"),
    (r"trump.*president|trump.*2028|trump.*term", "trump-presidency"),
    (r"democrat.*nomin|2028.*dem|dem.*primary", "2028-dem-nom"),
    (r"republican.*nomin|gop.*primary|2028.*rep", "2028-rep-nom"),
    (r"hungar|orban|fidesz", "hungarian-election"),
    (r"french.*election|macron|le pen|france.*vote", "french-election"),
    (r"german.*election|bundestag|scholz|merz", "german-election"),
    (r"uk.*election|labour|conservative|starmer|sunak", "uk-election"),
    (r"bitcoin|btc.*price|btc.*above|btc.*below", "bitcoin-price"),
    (r"ethereum|eth.*price|eth.*above|eth.*below", "ethereum-price"),
    (r"fed.*rate|fomc.*rate|interest rate.*fed", "fed-rates"),
    (r"tariff|trade war|import duty", "tariff-trade"),
    (r"ceasefire", "ceasefire-talks"),
    (r"nuclear|nuke", "nuclear-risk"),
    (r"recession|gdp.*contract|economic.*downturn", "recession-risk"),
    (r"ai.*regulation|ai.*ban|ai.*executive order", "ai-regulation"),
]


def extract_theme(slug, question):
    """Extract a thematic key from slug/question. Returns theme string or None."""
    text = f"{slug} {question}".lower()
    for pattern, theme_key in THEME_PATTERNS:
        if re.search(pattern, text):
            return theme_key
    return None


def load_theme_log():
    """Load theme signal timestamps from disk."""
    try:
        with open(THEME_LOG, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_theme_log(data):
    """Persist theme signal timestamps."""
    try:
        os.makedirs(os.path.dirname(THEME_LOG), exist_ok=True)
        with open(THEME_LOG, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log(f"WARNING: Could not write theme log: {e}")


def check_theme_dedup(theme_key, theme_log):
    """Return True if this theme should be suppressed (already has THEME_DEDUP_MAX signals in window)."""
    if not theme_key:
        return False
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=THEME_DEDUP_WINDOW_H)).isoformat()
    entries = theme_log.get(theme_key, [])
    recent = [ts for ts in entries if ts >= cutoff]
    return len(recent) >= THEME_DEDUP_MAX


def record_theme_signal(theme_key, theme_log):
    """Record a signal timestamp for this theme."""
    if not theme_key:
        return
    now = datetime.now(timezone.utc).isoformat()
    if theme_key not in theme_log:
        theme_log[theme_key] = []
    theme_log[theme_key].append(now)
    # Prune old entries (older than 2x window)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=THEME_DEDUP_WINDOW_H * 2)).isoformat()
    theme_log[theme_key] = [ts for ts in theme_log[theme_key] if ts >= cutoff]


# ─── MAIN SCAN ────────────────────────────────────────────────────────────────

def run_scan(dry_run=False):
    """Main scan loop: fetch markets, filter, estimate, classify, output."""

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        log("FATAL: DEEPSEEK_API_KEY not set in environment")
        write_health(0, 0, 0, 0, status="ERROR", error_msg="No DEEPSEEK_API_KEY")
        print("SCAN_COMPLETE|0|0|0|0")
        return

    log("=== AXIS_PMS Scanner v2.1 starting ===")
    log(f"Mode: {'DRY RUN' if dry_run else 'LIVE (paper trades will be logged)'}")

    # Step 1: Fetch markets from Gamma API
    log("Fetching markets from Gamma API...")
    raw_markets = fetch_markets()
    log(f"Fetched {len(raw_markets)} total markets")

    if not raw_markets:
        log("No markets returned from Gamma API")
        write_health(0, 0, 0, 0, status="ERROR", error_msg="Gamma API returned no data")
        print("SCAN_COMPLETE|0|0|0|0")
        return

    # Step 2: Filter markets
    candidates = []
    for m in raw_markets:
        passed, reason = filter_market(m)
        if passed:
            crowd_price = parse_crowd_price(m)
            if crowd_price is not None:
                candidates.append({
                    "question": m.get("question", "Unknown"),
                    "slug": m.get("slug", ""),
                    "crowd_price": crowd_price,
                    "volume24h": float(m.get("volume24hr") or 0),
                    "liquidity": float(m.get("liquidity") or m.get("liquidityNum") or 0),
                    "end_date": m.get("endDate", ""),
                    "description": m.get("description", ""),
                })

    log(f"After filtering: {len(candidates)} candidates from {len(raw_markets)} markets")

    if not candidates:
        log("No candidates passed filters")
        write_health(len(raw_markets), 0, 0, 0, status="OK")
        print(f"SCAN_COMPLETE|{len(raw_markets)}|0|0|0")
        return

    # Step 3: Fetch whale-active markets for confirmation
    log("Fetching whale data from PolymarketScan...")
    whale_slugs = fetch_whale_markets()
    log(f"Found {len(whale_slugs)} whale-active markets")

    # Load existing slugs for dedup
    existing_slugs = load_existing_slugs()
    log(f"Loaded {len(existing_slugs)} existing slugs for dedup")

    # Load theme dedup log
    theme_log = load_theme_log()

    # Step 4: Batch candidates and get AI estimates from DeepSeek
    signals = []
    batches = [candidates[i:i+BATCH_SIZE] for i in range(0, len(candidates), BATCH_SIZE)]
    log(f"Processing {len(batches)} batches of up to {BATCH_SIZE} markets each")

    for batch_idx, batch in enumerate(batches):
        log(f"Batch {batch_idx+1}/{len(batches)}: {len(batch)} markets")

        ai_probs = deepseek_estimate(batch, api_key)
        if ai_probs is None:
            log(f"Batch {batch_idx+1} failed — skipping")
            time.sleep(REQUEST_DELAY)
            continue

        for i, market in enumerate(batch):
            ai_est = ai_probs[i]
            if ai_est is None:
                log(f"SKIP (no AI estimate): {market['question']}")
                continue
            crowd = market["crowd_price"]
            whale = market["slug"] in whale_slugs

            tier, gap = classify_signal(ai_est, crowd, whale)
            if tier is not None:
                direction = determine_direction(ai_est, crowd)

                SPORTS_PREFIXES = [
                    "nba-", "nfl-", "nhl-", "mlb-", "cbb-", "cfb-", "epl-",
                    "atp-", "wta-", "crint-", "mma-", "boxing-", "f1-",
                    "nascar-", "soccer-", "rugby-",
                ]
                GEOPOLITICS_KEYWORDS = {
                    "iran", "israel", "ukraine", "russia", "military",
                    "ceasefire", "invasion", "offensive", "troops", "forces",
                    "war", "conflict", "sanctions", "nuclear", "regime",
                }
                slug_lower = market["slug"].lower()
                q_lower = market["question"].lower()
                # Primary: slug prefix match
                is_sports = any(slug_lower.startswith(p) for p in SPORTS_PREFIXES)
                # Secondary: "X vs Y" pattern WITHOUT geopolitics keywords
                if not is_sports:
                    has_vs = " vs " in q_lower or " vs. " in q_lower
                    has_geo = any(kw in q_lower for kw in GEOPOLITICS_KEYWORDS)
                    if has_vs and not has_geo:
                        is_sports = True
                if is_sports:
                    # Don't invert when crowd is near-certain (>75% on either side)
                    crowd_for_check = crowd if hasattr(crowd, '__float__') else 0.5
                    try:
                        crowd_for_check = float(crowd)
                    except (TypeError, ValueError):
                        crowd_for_check = 0.5
                    crowd_dominant = max(crowd_for_check, 1 - crowd_for_check)
                    if crowd_dominant > SPORTS_CROWD_MAX:
                        log(f"SKIP sports inversion (crowd too certain: {crowd_dominant:.0%}): {market['slug']}")
                        continue
                    log(f"INVERSION: crowd-bias-sports — {market['slug']} (AI={ai_est:.2f} crowd={crowd:.2f})")
                    if direction == "BUY YES":
                        direction = "BUY NO (INVERTED-SPORTS)"
                    else:
                        direction = "BUY YES (INVERTED-SPORTS)"

                # Thematic cluster dedup
                theme_key = extract_theme(market["slug"], market["question"])
                if check_theme_dedup(theme_key, theme_log):
                    log(f"THEME_DEDUP: {market['slug']}")
                    continue

                sig = {
                    "tier": tier,
                    "ai_estimate": ai_est,
                    "crowd_price": crowd,
                    "gap_pct": gap,
                    "question": market["question"],
                    "slug": market["slug"],
                    "direction": direction,
                    "volume24h": market["volume24h"],
                    "whale_confirmed": whale,
                    "description": market.get("description", ""),
                }
                signals.append(sig)
                record_theme_signal(theme_key, theme_log)

                # Output signal line for wrapper
                signal_output(
                    tier, ai_est, crowd, gap,
                    market["question"], market["slug"],
                    direction, market["volume24h"], whale
                )

                # Log paper trade for RED signals (unless dry run)
                if tier == "RED" and not dry_run:
                    if market["slug"] in existing_slugs:
                        log(f"SKIP (already traded): {market['slug']}")
                    else:
                        log_paper_trade(sig)
                        existing_slugs.add(market["slug"])

        time.sleep(REQUEST_DELAY)

    # Step 5: Summary
    red_count = sum(1 for s in signals if s["tier"] == "RED")
    yellow_count = sum(1 for s in signals if s["tier"] == "YELLOW")
    green_count = sum(1 for s in signals if s["tier"] == "GREEN")

    log(f"=== Scan complete: {len(candidates)} scanned, {len(signals)} signals "
        f"(RED:{red_count} YELLOW:{yellow_count} GREEN:{green_count}) ===")

    write_health(len(candidates), len(signals), red_count, yellow_count)
    save_theme_log(theme_log)
    print(f"SCAN_COMPLETE|{len(candidates)}|{len(signals)}|{red_count}|{yellow_count}")


# ─── VOLUME TRACKER (Component 2 — called with --volume-report) ──────────────

def run_volume_tracker():
    """
    Weekly volume report: top 50 markets by 24h volume, grouped by category.
    Prints Discord-ready summary to stdout.
    Cron: Sundays 6AM UTC — separate entry from main scan.
    """
    log("=== Polymarket Volume Tracker starting ===")

    url = (
        f"{GAMMA_API}?limit=50&active=true&closed=false"
        f"&order=volume24hr&ascending=false"
    )
    markets = http_get(url)
    if not markets or not isinstance(markets, list):
        print("ERROR: Failed to fetch markets from Gamma API")
        return

    # Group by category — Polymarket doesn't have a clean category field,
    # so we infer from question content and groupItemTitle
    categories = {}
    top_market = None
    top_volume = 0

    for m in markets:
        vol = float(m.get("volume24hr") or 0)
        question = m.get("question", "")
        group_title = m.get("groupItemTitle", "")

        # Category inference from question content
        cat = infer_category(question, group_title)

        if cat not in categories:
            categories[cat] = {"volume": 0, "count": 0}
        categories[cat]["volume"] += vol
        categories[cat]["count"] += 1

        if vol > top_volume:
            top_volume = vol
            top_market = question

    # Sort by total volume
    sorted_cats = sorted(categories.items(), key=lambda x: x[1]["volume"], reverse=True)

    # Format output
    lines = [
        "TOP POLYMARKET CATEGORIES THIS WEEK",
        "=====================================",
    ]
    for rank, (cat, data) in enumerate(sorted_cats[:10], 1):
        vol_str = format_volume(data["volume"])
        lines.append(f"{rank:>2}. {cat:<18} {vol_str:>8}  ({data['count']} markets)")

    lines.append(f"\nTop market: {top_market[:60]}  ${top_volume:,.0f}")
    lines.append("=====================================")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    output = "\n".join(lines)
    print(output)
    log("Volume report complete")


def infer_category(question, group_title):
    """Infer market category from question text and group title."""
    q_lower = question.lower()
    g_lower = group_title.lower()

    # Sports patterns
    sports_keywords = [
        "vs.", "vs ", "win the", "qualify for", "world cup", "super bowl",
        "nba", "nfl", "nhl", "mlb", "epl", "premier league", "champions league",
        "serie a", "bundesliga", "la liga", "ncaa", "march madness",
        "lakers", "celtics", "warriors", "yankees", "dodgers",
        "match winner", "game winner", "tournament", "playoffs",
        "counter-strike", "esports", "league of legends",
    ]
    if any(kw in q_lower or kw in g_lower for kw in sports_keywords):
        return "Sports"

    # Politics
    politics_keywords = [
        "trump", "biden", "president", "congress", "senate", "election",
        "republican", "democrat", "governor", "nomination", "impeach",
        "supreme court", "vance", "desantis", "greenland",
    ]
    if any(kw in q_lower for kw in politics_keywords):
        return "Politics"

    # Geopolitics
    geo_keywords = [
        "iran", "ukraine", "russia", "china", "taiwan", "israel",
        "ceasefire", "invade", "invasion", "regime", "nato", "sanctions",
        "war", "missile", "nuclear",
    ]
    if any(kw in q_lower for kw in geo_keywords):
        return "Geopolitics"

    # Crypto
    crypto_keywords = [
        "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
        "crypto", "token", "blockchain", "nft", "defi",
    ]
    if any(kw in q_lower for kw in crypto_keywords):
        return "Crypto"

    # Macro/Economics
    macro_keywords = [
        "fed", "interest rate", "inflation", "cpi", "fomc",
        "oil", "gdp", "unemployment", "recession", "tariff",
    ]
    if any(kw in q_lower for kw in macro_keywords):
        return "Macro/Fed"

    # Tech
    tech_keywords = [
        "ai ", "artificial intelligence", "openai", "google", "apple",
        "tesla", "spacex", "gta", "release", "launch",
    ]
    if any(kw in q_lower for kw in tech_keywords):
        return "Tech"

    # Entertainment / Pop Culture
    pop_keywords = [
        "oscar", "grammy", "movie", "film", "netflix", "spotify",
        "jesus", "pope", "celebrity",
    ]
    if any(kw in q_lower for kw in pop_keywords):
        return "Pop Culture"

    return "Other"


def format_volume(vol):
    """Format volume as $X.XM or $X.XK."""
    if vol >= 1_000_000:
        return f"${vol/1_000_000:.1f}M"
    elif vol >= 1_000:
        return f"${vol/1_000:.0f}K"
    else:
        return f"${vol:.0f}"


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--volume-report" in args:
        run_volume_tracker()
    elif "--dry-run" in args:
        run_scan(dry_run=True)
    else:
        run_scan(dry_run=False)
