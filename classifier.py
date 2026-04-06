"""
Task Classifier — 5-Tier LLM Routing System
Evaluates incoming tasks and assigns them to the appropriate tier/model.

Tier 0 — Deterministic bypass (no LLM, no cost, <1ms)
Tier 1 — Local/Ollama (free, simple tasks)
Tier 2 — Budget Cloud / Llama Cloud ($20/mo fixed)
Tier 3 — Standard Cloud / Claude Sonnet (pay-per-token, prompt cached)
Tier 4 — Premium / Scheduled ONLY (never fires from real-time triggers)
"""

import re
import json
import anthropic
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

client = anthropic.Anthropic()


# ─────────────────────────────────────────────
# Tier & Role Definitions
# ─────────────────────────────────────────────

class Tier(Enum):
    TIER_0 = "tier_0"   # Deterministic — no LLM needed
    TIER_1 = "tier_1"   # Simple classification/extraction → Ollama (free, local)
    TIER_2 = "tier_2"   # Light reasoning/summarization → Llama Cloud ($20/mo fixed)
    TIER_3 = "tier_3"   # Complex analysis/risk scoring → Claude Sonnet (cached)
    TIER_4 = "tier_4"   # Strategic/premium → Scheduled ONLY, never real-time

class BotRole(Enum):
    TRADER = "trader"
    SENTINEL = "sentinel"
    RESEARCHER = "researcher"
    SUPERVISOR = "supervisor"

# Default bias per bot role — helps the Haiku classifier calibrate
BOT_TIER_BIAS = {
    BotRole.RESEARCHER:  Tier.TIER_1,   # Data gathering
    BotRole.SENTINEL:    Tier.TIER_2,   # Pattern monitoring
    BotRole.TRADER:      Tier.TIER_3,   # Execution decisions
    BotRole.SUPERVISOR:  Tier.TIER_3,   # Strategic oversight (never below Tier 3)
}

# Hard floor per role — these tiers CANNOT be routed below this
ROLE_FLOOR = {
    BotRole.TRADER:      Tier.TIER_3,
    BotRole.SUPERVISOR:  Tier.TIER_3,
    BotRole.SENTINEL:    Tier.TIER_1,
    BotRole.RESEARCHER:  Tier.TIER_0,   # Researcher CAN go all the way to Tier 0
}

# ─────────────────────────────────────────────
# Extraction Enforcement Rules (Tier 2 & 3)
# ─────────────────────────────────────────────
# Injected into every Tier 2/3 execution prompt to enforce honesty-first extraction.
# Reference: ~/.claude/CLAUDE.md honesty rules — never infer, never guess.
#
# These rules are also defined in llmroute.py where they are actively injected
# into call_llama_cloud() (Tier 2) and call_sonnet() (Tier 3).

EXTRACTION_ENFORCEMENT_RULES = """
## Extraction Enforcement Rules

- If your confidence in an extracted value is below 95%, return an empty string for that field — never infer or guess.
- Prefix every returned field with [EXTRACTED] if taken directly from the source text, or [INFERRED] if derived from context. Flag all [INFERRED] fields clearly.
- If you are uncertain, leave the field blank. A blank answer is always better than a wrong answer.
"""

# Per-tier token limits — enforced in router.py, defined here as the contract
TIER_MAX_TOKENS = {
    Tier.TIER_0: 0,      # No LLM call
    Tier.TIER_1: 100,
    Tier.TIER_2: 300,
    Tier.TIER_3: 500,
    Tier.TIER_4: 2000,
}


# ─────────────────────────────────────────────
# ClassificationResult
# ─────────────────────────────────────────────

@dataclass
class ClassificationResult:
    tier: Tier
    model: str
    confidence: float           # 0.0–1.0 (1.0 for Tier 0/4 — deterministic)
    reasoning: str
    use_cache: bool             # Whether Sonnet prompt caching applies
    estimated_cost_usd: float
    max_tokens: int             # Token budget for the routed call
    bypass_reason: Optional[str] = None   # Set for Tier 0 (why no LLM needed)
    scheduled_only: bool = False          # True for Tier 4


# ─────────────────────────────────────────────
# TIER 0 — Deterministic Bypass
# ─────────────────────────────────────────────
#
# Rules for what qualifies as Tier 0:
#   ✅ Pure data fetch — the task IS the API call, no processing of response
#   ✅ Format conversion — input structure → output structure, no judgment
#   ✅ Binary state check — market open/closed, position exists, flag set/unset
#   ✅ Arithmetic — sum, percentage, ratio, threshold comparison with exact numbers
#   ✅ Template fill — fixed format output with known slot values
#
#   ❌ NOT Tier 0 — "Is this volume unusual?" (requires judgment)
#   ❌ NOT Tier 0 — "Check if RSI is above 70 and flag" (requires interpretation)
#   ❌ NOT Tier 0 — "Format the best trades from today" (requires selection)
#   ❌ NOT Tier 0 — Anything with "should", "evaluate", "assess", "analyze"

# Each entry: (pattern_regex, bypass_reason)
# Patterns are matched case-insensitively against the task description.
_TIER_0_PATTERNS: list[tuple[str, str]] = [

    # ── Pure API fetches ──────────────────────────────────────────────────────
    (r"\bfetch\b.{0,40}\bprice\b",           "fetch_price: pure API call, no processing"),
    (r"\bget\b.{0,40}\bprice\b",             "get_price: pure API call, no processing"),
    (r"\bretrieve\b.{0,40}\b(price|ohlcv|candle|tick|quote)\b",
                                              "retrieve_market_data: pure API call"),
    (r"\bpull\b.{0,40}\b(price|ohlcv|candle|tick|quote)\b",
                                              "pull_market_data: pure API call"),
    (r"\bfetch.{0,30}\b(ohlcv|candle|bar|tick)\b",
                                              "fetch_ohlcv: pure API call, no processing"),
    (r"\blist\b.{0,40}\b(open orders?|positions?|trades?|fills?)\b",
                                              "list_orders: pure API call"),
    (r"\bget\b.{0,40}\b(open orders?|positions?|portfolio|balance|equity)\b",
                                              "get_portfolio: pure API call"),
    (r"\bretrieve\b.{0,40}\b(order|trade|fill|log)\b",
                                              "retrieve_records: pure API call"),

    # ── Parse / format / convert ──────────────────────────────────────────────
    (r"\bparse\b.{0,60}\b(json|payload|webhook|response|string)\b",
                                              "parse_structure: format conversion, no judgment"),
    (r"\bformat\b.{0,60}\b(as |to |into ).{0,30}\b(json|csv|string|list|table|dict)\b",
                                              "format_output: structure conversion"),
    (r"\bconvert\b.{0,60}\b(to |into ).{0,30}\b(json|csv|string|list|table|dict)\b",
                                              "convert_format: structure conversion"),
    (r"\bserialize\b",                        "serialize: structure conversion"),
    (r"\bdeserialize\b",                      "deserialize: structure conversion"),
    (r"\bformat.{0,40}\b(positions?|orders?|trades?|logs?).{0,20}\b(as |into |to )\b",
                                              "format_records: structure conversion"),

    # ── Binary state checks ───────────────────────────────────────────────────
    (r"\bcheck\b.{0,30}\bmarket\b.{0,30}\b(open|closed|hours)\b",
                                              "check_market_hours: binary state, no judgment"),
    (r"\bis\b.{0,30}\bmarket\b.{0,30}\b(open|closed)\b",
                                              "check_market_open: binary state"),
    (r"\bcheck\b.{0,30}\b(position|order)\b.{0,30}\b(exists?|active|open|filled)\b",
                                              "check_position_state: binary state"),
    (r"\bverify\b.{0,40}\b(connection|api|endpoint|auth|key)\b",
                                              "verify_connectivity: binary state"),
    (r"\b(ping|health.?check|heartbeat)\b",   "health_check: binary state"),

    # ── Pure arithmetic / threshold with exact numbers ────────────────────────
    (r"\bcalculate\b.{0,40}\b(percentage|percent|ratio|sum|total|average|mean)\b",
                                              "calculate_metric: arithmetic, no judgment"),
    (r"\bcompute\b.{0,40}\b(percentage|percent|ratio|sum|total|average|mean)\b",
                                              "compute_metric: arithmetic, no judgment"),
    (r"\b(sum|total)\b.{0,40}\b(positions?|p&l|pnl|balance)\b",
                                              "sum_positions: arithmetic"),

    # ── Log retrieval / record lookup ─────────────────────────────────────────
    (r"\b(retrieve|fetch|get|pull)\b.{0,30}\b(last|recent|latest)\b.{0,20}\b(logs?|records?|entries|history)\b",
                                              "retrieve_logs: pure data fetch"),
    (r"\b(retrieve|fetch|get)\b.{0,40}\btrade logs?\b",
                                              "retrieve_trade_logs: pure data fetch"),
]

# Pre-compile all patterns once at import time
_COMPILED_T0 = [
    (re.compile(pat, re.IGNORECASE), reason)
    for pat, reason in _TIER_0_PATTERNS
]


def check_tier_0(task_description: str) -> Optional[str]:
    """
    Scan task description against Tier 0 bypass patterns.

    Returns the bypass_reason string if matched, None otherwise.

    Design intent:
    - False negatives (missing a Tier 0 task) → routed to Tier 1, costs ~$0, safe
    - False positives (wrongly marking as Tier 0) → task runs without LLM, BAD
    - Therefore: patterns must be conservative. When ambiguous, return None.
    """
    for pattern, reason in _COMPILED_T0:
        if pattern.search(task_description):
            # Secondary guard: reject if task contains judgment words
            # These override any Tier 0 match
            judgment_words = re.compile(
                r"\b(should|evaluate|assess|analyze|decide|recommend|"
                r"unusual|abnormal|appropriate|best|worst|better|"
                r"why|whether|compare|prioritize|strategy)\b",
                re.IGNORECASE
            )
            if judgment_words.search(task_description):
                return None  # Has judgment component — not Tier 0
            return reason
    return None


# ─────────────────────────────────────────────
# TIER 4 — Scheduled-Only Gate
# ─────────────────────────────────────────────
#
# Tier 4 is NEVER triggered from real-time request paths.
# The only valid callers:
#   1. A weekly/daily cron job (is_scheduled=True)
#   2. An explicit human request via the API (is_human_requested=True)
#
# If a task looks like it needs Tier 4 reasoning but isn't scheduled/human-requested,
# it falls back to Tier 3 (best available real-time option).
#
# Examples of Tier 4 tasks:
#   - "Generate weekly strategy review across all positions"
#   - "Comprehensive portfolio rebalancing analysis"
#   - "Deep dive on macro regime change and its implications for our strategy"
#   - "Full risk audit of all open positions and correlation matrix"

_TIER_4_INDICATORS = re.compile(
    r"\b(weekly|monthly|quarterly|comprehensive|deep.?dive|"
    r"full.{0,10}(audit|review|analysis)|strategic.{0,10}review|"
    r"portfolio.{0,10}rebalanc|regime.{0,10}(change|shift|analysis)|"
    r"correlation.{0,10}matrix|risk.{0,10}audit)\b",
    re.IGNORECASE
)

def check_tier_4_candidate(task_description: str) -> bool:
    """Returns True if task looks like Tier 4 material (still need schedule/human gate)."""
    return bool(_TIER_4_INDICATORS.search(task_description))


# ─────────────────────────────────────────────
# Cost Model
# ─────────────────────────────────────────────

MODEL_COSTS = {
    "deterministic":   0.0000,   # Tier 0 — no model
    "ollama":          0.0000,   # Tier 1 — local
    "llama_cloud":     0.0000,   # Tier 2 — fixed $20/mo, marginal ~$0
    "claude_sonnet":   0.0003,   # Tier 3 — cached Sonnet (estimated per call)
    "claude_sonnet_uncached": 0.003,  # Tier 3 without cache hit
    "claude_opus":     0.015,    # Tier 4 — premium (scheduled only)
    "claude_haiku":    0.0003,   # Cost of THIS classifier call
}

def estimate_cost(model: str, use_cache: bool = False) -> float:
    if model == "claude_sonnet" and not use_cache:
        return MODEL_COSTS["claude_sonnet_uncached"]
    return MODEL_COSTS.get(model, 0.001)


# ─────────────────────────────────────────────
# Haiku Classifier Prompt (Tiers 1–3 only)
# ─────────────────────────────────────────────
# Note: Tier 0 and Tier 4 are handled BEFORE this prompt fires.
# Haiku only needs to distinguish Tier 1 / 2 / 3.

CLASSIFIER_SYSTEM_PROMPT = """You are a task complexity classifier for an automated trading system. 
Your role is to evaluate incoming task descriptions and assign them to the correct processing tier to minimize compute costs while preserving decision quality.

Tier 0 (deterministic bypass) has already been checked before reaching you.
Tier 4 (premium/scheduled) is handled separately. You only assign Tier 1, 2, or 3.

## Tier Definitions

**TIER_1 — Simple Classification / Extraction (route to: Ollama/local)**
Tasks requiring minimal reasoning: single-step extraction, basic pattern matching, 
simple yes/no classification, straightforward template completion.
max_tokens budget: 100
Examples:
- "Extract all ticker symbols mentioned in this text"
- "Is this alert message a warning or info level?"
- "Tag this news headline by sector (tech/finance/energy/other)"
- "Does this trade log entry contain an error code?"
- "Classify this order type (market/limit/stop)"

**TIER_2 — Light Analysis / Summarization (route to: Llama Cloud)**
Tasks requiring multi-step but predictable reasoning: summarization, pattern detection
across multiple data points, report generation from structured inputs, threshold evaluation.
Low-to-medium financial stakes. Output drives alerts, not direct execution.
max_tokens budget: 300
Extraction enforcement: All Tier 2 extraction prompts enforce honesty-first rules —
95% confidence threshold, [EXTRACTED]/[INFERRED] field prefixing, blank-over-wrong policy.
Examples:
- "Summarize overnight news sentiment for tech sector"
- "Check if RSI is above 70 and generate an overbought alert"
- "Compare today's volume to 30-day average and note if anomalous"
- "Detect if price crossed the 200-day moving average"
- "Draft a brief market summary for the daily log"
- "Screen these 20 stocks for momentum criteria and return a ranked list"
- "Evaluate if a Sentinel alert warrants escalation"
- "Summarize this vendor risk report into 3 bullet points"

**TIER_3 — Critical Decision / Complex Reasoning (route to: Claude Sonnet)**
Tasks involving financial risk, novel situations, multi-factor synthesis,
or where an error causes significant monetary loss. Always Tier 3 for trade decisions.
max_tokens budget: 500
Extraction enforcement: All Tier 3 extraction prompts enforce honesty-first rules —
95% confidence threshold, [EXTRACTED]/[INFERRED] field prefixing, blank-over-wrong policy.
Examples:
- "Should I enter a long position on NVDA given current conditions?"
- "Evaluate risk/reward of this options trade setup"
- "Determine position sizing for this trade"
- "Assess whether to override a stop-loss given macro context"
- "Synthesize earnings report + technical setup + macro backdrop into a trade thesis"
- "Approve or reject this trade proposed by the Trader bot"
- "Decide whether to halt trading given unusual market conditions"
- "Score the severity of this vendor risk event (0-10 with justification)"

## Classification Rules

1. **When in doubt, escalate** — Tier 2 costs ~$0, Tier 3 is still cheap with caching.
2. **Bot role matters** — Trader or Supervisor asking anything decision-adjacent → Tier 3.
3. **Irreversibility flag** — Any task that triggers a real order or modifies a position → Tier 3.
4. **Novelty flag** — Unusual market conditions (extreme VIX, circuit breakers, earnings surprises) → escalate one tier.
5. **Compound tasks** — If a task bundles retrieval + decision, classify by its highest-tier component.
6. **Output stakes** — If the output feeds directly into trade execution → Tier 3.

## Output Format

Respond ONLY with a valid JSON object. No preamble, no explanation outside the JSON.

{
  "tier": "tier_1" | "tier_2" | "tier_3",
  "confidence": 0.0–1.0,
  "reasoning": "One concise sentence explaining the classification.",
  "flags": {
    "irreversible": true | false,
    "novel_conditions": true | false,
    "compound_task": true | false,
    "feeds_execution": true | false
  },
  "suggested_model": "ollama" | "llama_cloud" | "claude_sonnet"
}"""


# ─────────────────────────────────────────────
# Main Classifier (5-tier)
# ─────────────────────────────────────────────

def classify_task(
    task_description: str,
    bot_role: BotRole,
    market_conditions: Optional[str] = None,
    is_live_trading: bool = True,
    is_scheduled: bool = False,        # True = cron job / batch, enables Tier 4
    is_human_requested: bool = False,  # True = explicit human API call, enables Tier 4
) -> ClassificationResult:
    """
    Classify a task across 5 tiers and return routing metadata.

    Flow:
      1. Tier 0 check (deterministic bypass) — no API call
      2. Tier 4 check (scheduled/human gate) — no API call
      3. Role floor enforcement (pre-Haiku) — no API call
      4. Haiku classification (Tier 1 / 2 / 3)
      5. Post-classification safety overrides

    Args:
        task_description:    Natural language description of the task.
        bot_role:            Which bot is requesting classification.
        market_conditions:   Optional context ("VIX > 30", "earnings week", etc.)
        is_live_trading:     If False, paper trading — allows tier relaxation.
        is_scheduled:        True if this is called from a cron/batch job.
        is_human_requested:  True if a human explicitly triggered this.

    Returns:
        ClassificationResult
    """

    # ── STAGE 1: Tier 0 deterministic bypass ─────────────────────────────────
    bypass_reason = check_tier_0(task_description)
    if bypass_reason:
        # Still enforce role floor — Trader/Supervisor can't go below Tier 3
        role_floor = ROLE_FLOOR.get(bot_role, Tier.TIER_0)
        if role_floor.value > Tier.TIER_0.value:
            # Floor is higher than Tier 0 — can't bypass for this role
            # Fall through to full classification
            pass
        else:
            return ClassificationResult(
                tier=Tier.TIER_0,
                model="deterministic",
                confidence=1.0,
                reasoning=f"Tier 0 bypass: {bypass_reason}",
                use_cache=False,
                estimated_cost_usd=0.0,
                max_tokens=0,
                bypass_reason=bypass_reason,
            )

    # ── STAGE 2: Tier 4 gate ──────────────────────────────────────────────────
    is_tier4_candidate = check_tier_4_candidate(task_description)
    if is_tier4_candidate and (is_scheduled or is_human_requested):
        return ClassificationResult(
            tier=Tier.TIER_4,
            model="claude_opus",
            confidence=1.0,
            reasoning="Tier 4: matches strategic/comprehensive pattern + authorized caller (scheduled or human-requested)",
            use_cache=False,
            estimated_cost_usd=estimate_cost("claude_opus"),
            max_tokens=TIER_MAX_TOKENS[Tier.TIER_4],
            scheduled_only=True,
        )
    elif is_tier4_candidate and not (is_scheduled or is_human_requested):
        # Task looks like Tier 4 but arrived via real-time path — demote to Tier 3
        return ClassificationResult(
            tier=Tier.TIER_3,
            model="claude_sonnet",
            confidence=0.85,
            reasoning="Tier 4 candidate demoted to Tier 3: real-time trigger not permitted for premium tier. "
                      "Re-submit with is_scheduled=True or is_human_requested=True for full analysis.",
            use_cache=True,
            estimated_cost_usd=estimate_cost("claude_sonnet", use_cache=True),
            max_tokens=TIER_MAX_TOKENS[Tier.TIER_3],
        )

    # ── STAGE 3: Role floor pre-check ─────────────────────────────────────────
    # If the role floor is already Tier 3, skip the Haiku call and go straight there.
    # This saves $0.0003 per call for Trader/Supervisor tasks we KNOW are Tier 3.
    role_floor = ROLE_FLOOR.get(bot_role, Tier.TIER_1)
    if role_floor == Tier.TIER_3:
        # Only skip Haiku if the task description also looks decision-adjacent
        # (don't skip if a Trader asks a pure data question — Haiku might give Tier 2)
        decision_keywords = re.compile(
            r"\b(should|approve|reject|enter|exit|execute|trade|position|"
            r"sizing|stop.?loss|halt|override|buy|sell|risk|rebalanc)\b",
            re.IGNORECASE
        )
        if decision_keywords.search(task_description):
            return ClassificationResult(
                tier=Tier.TIER_3,
                model="claude_sonnet",
                confidence=1.0,
                reasoning=f"Fast-path Tier 3: {bot_role.value} role floor + decision keyword detected, "
                          "Haiku call skipped.",
                use_cache=True,
                estimated_cost_usd=estimate_cost("claude_sonnet", use_cache=True),
                max_tokens=TIER_MAX_TOKENS[Tier.TIER_3],
            )

    # ── STAGE 4: Haiku classification (Tier 1 / 2 / 3) ───────────────────────
    user_content = f"""Bot Role: {bot_role.value.upper()}
Task: {task_description}
Live Trading: {is_live_trading}"""

    if market_conditions:
        user_content += f"\nMarket Conditions: {market_conditions}"

    default_tier = BOT_TIER_BIAS.get(bot_role, Tier.TIER_2)
    user_content += f"\nDefault tier bias for this bot role: {default_tier.value}"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=CLASSIFIER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    parsed = json.loads(raw)

    tier = Tier(parsed["tier"])
    model = parsed["suggested_model"]
    confidence = float(parsed["confidence"])
    reasoning = parsed["reasoning"]
    flags = parsed.get("flags", {})

    use_cache = (model == "claude_sonnet")

    # ── STAGE 5: Post-classification safety overrides ─────────────────────────

    # 5a. Irreversible action in live trading → force Tier 3
    if flags.get("irreversible") and is_live_trading:
        tier = Tier.TIER_3
        model = "claude_sonnet"
        use_cache = True
        reasoning += " [Escalated: irreversible action in live trading]"

    # 5b. Feeds execution directly → force Tier 3
    if flags.get("feeds_execution") and is_live_trading:
        tier = Tier.TIER_3
        model = "claude_sonnet"
        use_cache = True
        reasoning += " [Escalated: output feeds trade execution]"

    # 5c. Novel market conditions → bump one tier
    if flags.get("novel_conditions"):
        if tier == Tier.TIER_1:
            tier = Tier.TIER_2
            model = "llama_cloud"
            reasoning += " [Escalated: novel market conditions]"
        elif tier == Tier.TIER_2:
            tier = Tier.TIER_3
            model = "claude_sonnet"
            use_cache = True
            reasoning += " [Escalated: novel market conditions]"

    # 5d. Role floor enforcement (final safety net)
    tier_order = [Tier.TIER_0, Tier.TIER_1, Tier.TIER_2, Tier.TIER_3, Tier.TIER_4]
    floor_index = tier_order.index(role_floor)
    tier_index = tier_order.index(tier)
    if tier_index < floor_index:
        tier = role_floor
        model = {
            Tier.TIER_1: "ollama",
            Tier.TIER_2: "llama_cloud",
            Tier.TIER_3: "claude_sonnet",
        }.get(role_floor, "claude_sonnet")
        use_cache = (model == "claude_sonnet")
        reasoning += f" [Floor applied: {bot_role.value} cannot go below {role_floor.value}]"

    return ClassificationResult(
        tier=tier,
        model=model,
        confidence=confidence,
        reasoning=reasoning,
        use_cache=use_cache,
        estimated_cost_usd=estimate_cost(model, use_cache),
        max_tokens=TIER_MAX_TOKENS[tier],
    )


# ─────────────────────────────────────────────
# Batch Classifier
# ─────────────────────────────────────────────

def classify_batch(tasks: list[dict]) -> list[ClassificationResult]:
    """
    Classify multiple tasks. Each dict should have:
      - task_description: str
      - bot_role: BotRole
      - market_conditions: Optional[str]
      - is_live_trading: bool
      - is_scheduled: bool (optional, default False)
      - is_human_requested: bool (optional, default False)
    """
    return [classify_task(**t) for t in tasks]


# ─────────────────────────────────────────────
# CLI Test Harness — 20 tasks across all 5 tiers
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        # ── Expected Tier 0 (deterministic bypass) ────────────────────────────
        {
            "task_description": "Fetch current BTC/USD price from Binance API",
            "bot_role": BotRole.RESEARCHER,
            "is_live_trading": True,
            "_expected_tier": "tier_0",
        },
        {
            "task_description": "Parse webhook payload from exchange",
            "bot_role": BotRole.RESEARCHER,
            "is_live_trading": True,
            "_expected_tier": "tier_0",
        },
        {
            "task_description": "List all open orders",
            "bot_role": BotRole.RESEARCHER,
            "is_live_trading": True,
            "_expected_tier": "tier_0",
        },
        {
            "task_description": "Check if market is currently open",
            "bot_role": BotRole.SENTINEL,
            "is_live_trading": True,
            "_expected_tier": "tier_0",
        },
        {
            "task_description": "Retrieve last 50 trade logs and format as CSV",
            "bot_role": BotRole.RESEARCHER,
            "is_live_trading": False,
            "_expected_tier": "tier_0",
        },
        {
            "task_description": "Calculate percentage gain: entry 100, current 115",
            "bot_role": BotRole.RESEARCHER,
            "is_live_trading": True,
            "_expected_tier": "tier_0",
        },

        # ── Expected Tier 1 (Ollama — simple classification) ─────────────────
        {
            "task_description": "Extract all ticker symbols mentioned in this news headline",
            "bot_role": BotRole.RESEARCHER,
            "is_live_trading": True,
            "_expected_tier": "tier_1",
        },
        {
            "task_description": "Classify this order as market, limit, or stop",
            "bot_role": BotRole.RESEARCHER,
            "is_live_trading": True,
            "_expected_tier": "tier_1",
        },
        {
            "task_description": "Tag this alert as info, warning, or critical",
            "bot_role": BotRole.SENTINEL,
            "is_live_trading": True,
            "_expected_tier": "tier_1",
        },

        # ── Expected Tier 2 (Llama Cloud — light reasoning) ───────────────────
        {
            "task_description": "Summarize overnight news sentiment for the tech sector",
            "bot_role": BotRole.RESEARCHER,
            "is_live_trading": True,
            "_expected_tier": "tier_2",
        },
        {
            "task_description": "Check if RSI is above 70 and generate an overbought alert",
            "bot_role": BotRole.SENTINEL,
            "is_live_trading": True,
            "_expected_tier": "tier_2",
        },
        {
            "task_description": "Compare today's SPY volume to 30-day average and flag if anomalous",
            "bot_role": BotRole.SENTINEL,
            "is_live_trading": True,
            "_expected_tier": "tier_2",
        },
        {
            "task_description": "Screen these 20 stocks for basic momentum criteria and rank them",
            "bot_role": BotRole.RESEARCHER,
            "is_live_trading": True,
            "_expected_tier": "tier_2",
        },

        # ── Expected Tier 3 (Sonnet — critical decisions) ─────────────────────
        {
            "task_description": "Given RSI=28, MACD bullish cross, positive earnings — should I enter long AAPL at 2% allocation?",
            "bot_role": BotRole.TRADER,
            "market_conditions": "VIX=18, normal conditions",
            "is_live_trading": True,
            "_expected_tier": "tier_3",
        },
        {
            "task_description": "Approve or reject the Trader's proposed NVDA position given today's macro backdrop",
            "bot_role": BotRole.SUPERVISOR,
            "market_conditions": "Earnings week, elevated volatility",
            "is_live_trading": True,
            "_expected_tier": "tier_3",
        },
        {
            "task_description": "Evaluate risk/reward of this options trade setup and determine position sizing",
            "bot_role": BotRole.TRADER,
            "is_live_trading": True,
            "_expected_tier": "tier_3",
        },
        {
            "task_description": "Decide whether to halt trading given circuit breaker conditions",
            "bot_role": BotRole.SUPERVISOR,
            "market_conditions": "VIX=45, circuit breaker triggered on SPY",
            "is_live_trading": True,
            "_expected_tier": "tier_3",
        },

        # ── Expected Tier 4 (Premium — scheduled only) ────────────────────────
        {
            "task_description": "Generate weekly strategy review across all open positions",
            "bot_role": BotRole.SUPERVISOR,
            "is_live_trading": True,
            "is_scheduled": True,
            "_expected_tier": "tier_4",
        },
        {
            "task_description": "Full risk audit of portfolio correlation matrix and concentration risk",
            "bot_role": BotRole.SUPERVISOR,
            "is_live_trading": True,
            "is_scheduled": True,
            "_expected_tier": "tier_4",
        },
        # Same task WITHOUT is_scheduled → should demote to Tier 3
        {
            "task_description": "Generate weekly strategy review across all open positions",
            "bot_role": BotRole.SUPERVISOR,
            "is_live_trading": True,
            "is_scheduled": False,
            "_expected_tier": "tier_3",  # Demoted — not scheduled
        },
    ]

    print("=" * 70)
    print("TASK CLASSIFIER — 5-TIER TEST RUN (20 tasks)")
    print("=" * 70)

    haiku_calls = 0
    total_cost = 0.0
    correct = 0
    wrong = 0

    for i, tc in enumerate(test_cases, 1):
        expected = tc.pop("_expected_tier")
        result = classify_task(**tc)

        # Count Haiku calls (Tier 0, fast-path Tier 3, Tier 4 don't use Haiku)
        used_haiku = result.tier not in (Tier.TIER_0, Tier.TIER_4) and result.confidence != 1.0
        if used_haiku:
            haiku_calls += 1
            total_cost += MODEL_COSTS["claude_haiku"]

        total_cost += result.estimated_cost_usd

        status = "✅" if result.tier.value == expected else f"❌ (expected {expected})"
        if result.tier.value == expected:
            correct += 1
        else:
            wrong += 1

        print(f"\n[{i:02d}] {tc['task_description'][:60]}...")
        print(f"  Bot: {tc['bot_role'].value:<12}  {status}")
        print(f"  → Tier: {result.tier.value}  Model: {result.model}  "
              f"max_tokens: {result.max_tokens}  conf: {result.confidence:.0%}")
        print(f"  → Cost: ${result.estimated_cost_usd:.5f}  Cache: {result.use_cache}")
        print(f"  → Reason: {result.reasoning[:90]}")

    print(f"\n{'=' * 70}")
    print(f"Results:     {correct}/{len(test_cases)} correct  ({wrong} misroutes)")
    print(f"Haiku calls: {haiku_calls}/{len(test_cases)} tasks  "
          f"({len(test_cases) - haiku_calls} bypassed)")
    print(f"Total cost:  ${total_cost:.5f} for {len(test_cases)} classifications")
    print(f"Per-task avg: ${total_cost / len(test_cases):.5f}")
    print("=" * 70)
