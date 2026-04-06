#!/usr/bin/env python3
"""
LLMRoute — Intelligent LLM Router (v2 — with Prompt Caching)
Classifies incoming tasks via Claude Haiku, then dispatches
to the correct tier model for execution.

Tier 1: Ollama (local, free) — data retrieval, formatting
Tier 2: Llama Cloud API ($20/mo flat) — light reasoning
Tier 3: Claude Sonnet (pay-per-use) — critical decisions

NEW in v2: Anthropic prompt caching for classifier + Tier 3.
Static system prompts are cached, reducing repeat call costs by ~90%.

Usage:
    from llmroute import route_task
    result = route_task("Fetch the current BTC price and return as JSON")

Environment variables required:
    ANTHROPIC_API_KEY  — for Haiku classifier + Tier 3
    LLAMA_CLOUD_API_KEY — for Tier 2 (optional, falls back to Tier 3)

Optional:
    OLLAMA_HOST — default: http://localhost:11434
    LLMROUTE_LOG — path to log file (default: ~/llmroute/route_log.jsonl)
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

# Configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LOG_FILE = os.environ.get("LLMROUTE_LOG", os.path.expanduser("~/llmroute/route_log.jsonl"))

# ─────────────────────────────────────────────
# Extraction Enforcement Rules (Tier 2 & 3)
# ─────────────────────────────────────────────
# Applied to every Tier 2/3 prompt to enforce honesty-first extraction.
# Reference: ~/.claude/CLAUDE.md honesty rules — never infer, never guess.

EXTRACTION_ENFORCEMENT_RULES = """
## Extraction Enforcement Rules

- If your confidence in an extracted value is below 95%, return an empty string for that field — never infer or guess.
- Prefix every returned field with [EXTRACTED] if taken directly from the source text, or [INFERRED] if derived from context. Flag all [INFERRED] fields clearly.
- If you are uncertain, leave the field blank. A blank answer is always better than a wrong answer.
"""

# Load classifier prompt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PROMPT_PATH = os.path.join(SCRIPT_DIR, "classifier_prompt.txt")

with open(CLASSIFIER_PROMPT_PATH, "r") as f:
    CLASSIFIER_PROMPT = f.read()

# Lazy imports for anthropic
_anthropic_client = None


def _get_anthropic_client():
    """Lazy-load the Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
        except ImportError:
            os.system(f"{sys.executable} -m pip install anthropic -q")
            import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


# Token Tracking

_session_tokens = {
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_calls": 0,
}


def get_session_stats():
    """Return token usage stats for this session."""
    stats = _session_tokens.copy()
    stats["est_cost_without_cache"] = (
        (stats["input_tokens"] + stats["cache_read_input_tokens"] + stats["cache_creation_input_tokens"])
        * 2.0 / 1_000_000
        + stats["output_tokens"] * 8.0 / 1_000_000
    )
    stats["est_cost_with_cache"] = (
        stats["input_tokens"] * 2.0 / 1_000_000
        + stats["cache_read_input_tokens"] * 0.20 / 1_000_000
        + stats["cache_creation_input_tokens"] * 2.50 / 1_000_000
        + stats["output_tokens"] * 8.0 / 1_000_000
    )
    stats["savings_pct"] = (
        (1 - stats["est_cost_with_cache"] / stats["est_cost_without_cache"]) * 100
        if stats["est_cost_without_cache"] > 0 else 0
    )
    return stats


def _track_usage(response):
    """Extract and track token usage from Anthropic response."""
    usage = response.usage
    _session_tokens["total_calls"] += 1
    _session_tokens["input_tokens"] += getattr(usage, "input_tokens", 0)
    _session_tokens["output_tokens"] += getattr(usage, "output_tokens", 0)
    _session_tokens["cache_creation_input_tokens"] += getattr(usage, "cache_creation_input_tokens", 0)
    _session_tokens["cache_read_input_tokens"] += getattr(usage, "cache_read_input_tokens", 0)


# Classifier (with caching)

def classify_task(task):
    """Send task to Haiku classifier with prompt caching."""
    client = _get_anthropic_client()

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        system=[
            {
                "type": "text",
                "text": CLASSIFIER_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": task}],
    )

    _track_usage(response)
    text = response.content[0].text.strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(text)


# Tier 1: Ollama (Local)

def call_ollama(task, model="qwen2.5:3b", system_prompt=None):
    """Call local Ollama instance."""
    url = f"{OLLAMA_HOST}/api/generate"
    payload_data = {
        "model": model,
        "prompt": task,
        "stream": False,
    }
    if system_prompt:
        payload_data["system"] = system_prompt

    payload = json.dumps(payload_data).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        raise ConnectionError(f"Ollama not reachable at {OLLAMA_HOST}: {e}")
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}")


# Tier 2: Llama Cloud API

def call_llama_cloud(task, system_prompt=None):
    """Call Llama Cloud API. Falls back to Sonnet if no key."""
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")

    # Inject extraction enforcement rules into Tier 2 system prompt
    if system_prompt:
        system_prompt = system_prompt + "\n" + EXTRACTION_ENFORCEMENT_RULES
    else:
        system_prompt = EXTRACTION_ENFORCEMENT_RULES

    if not api_key:
        return call_sonnet(task, system_prompt=system_prompt, fallback_from_tier=2)

    url = "https://api.llama-api.com/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": task})

    payload = json.dumps({
        "model": "llama3.1-70b",
        "messages": messages,
        "max_tokens": 500,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[LLMRoute] Llama Cloud error, falling back to Sonnet: {e}")
        return call_sonnet(task, system_prompt=system_prompt, fallback_from_tier=2)


# Tier 3: Claude Sonnet (with caching)

def call_sonnet(task, system_prompt=None, fallback_from_tier=0):
    """Call Claude Sonnet with prompt caching."""
    client = _get_anthropic_client()

    # Inject extraction enforcement rules into Tier 3 system prompt
    # (skip if already injected — e.g. Tier 2 fallback path)
    if system_prompt and EXTRACTION_ENFORCEMENT_RULES.strip() not in system_prompt:
        system_prompt = system_prompt + "\n" + EXTRACTION_ENFORCEMENT_RULES
    elif not system_prompt:
        system_prompt = EXTRACTION_ENFORCEMENT_RULES

    system = None
    if system_prompt:
        system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    kwargs = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": task}],
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    _track_usage(response)

    result = response.content[0].text.strip()

    if fallback_from_tier:
        print(f"[LLMRoute] Tier {fallback_from_tier} -> Sonnet fallback used")

    return result


# Dispatcher

def route_task(task, system_prompt=None, log=True):
    """Main entry point. Classifies a task and routes to correct tier."""
    total_start = time.time()

    # Step 1: Classify
    classify_start = time.time()
    cache_read_before = _session_tokens["cache_read_input_tokens"]
    try:
        classification = classify_task(task)
    except Exception as e:
        print(f"[LLMRoute] Classifier error, defaulting to Tier 3: {e}")
        classification = {
            "tier": 3,
            "model": "anthropic/claude-sonnet",
            "reason": f"Classifier error fallback: {e}",
        }
    classify_ms = int((time.time() - classify_start) * 1000)

    cache_read_after = _session_tokens["cache_read_input_tokens"]
    cache_status = "hit" if cache_read_after > cache_read_before else "miss"

    tier = classification["tier"]

    # Step 2: Execute on correct tier
    execute_start = time.time()

    try:
        if tier == 1:
            result = call_ollama(task, system_prompt=system_prompt)
        elif tier == 2:
            result = call_llama_cloud(task, system_prompt=system_prompt)
        else:
            result = call_sonnet(task, system_prompt=system_prompt)
    except Exception as e:
        if tier != 3:
            print(f"[LLMRoute] Tier {tier} failed, falling back to Tier 3: {e}")
            result = call_sonnet(task, system_prompt=system_prompt)
            classification["reason"] += f" | FALLBACK: Tier {tier} failed"
            tier = 3
        else:
            raise

    execute_ms = int((time.time() - execute_start) * 1000)
    total_ms = int((time.time() - total_start) * 1000)

    output = {
        "tier": tier,
        "model": classification["model"],
        "reason": classification["reason"],
        "response": result,
        "latency_ms": total_ms,
        "classify_ms": classify_ms,
        "execute_ms": execute_ms,
        "cache_status": cache_status if tier != 1 else "n/a",
    }

    if log:
        _log_route(task, output)

    return output


def _log_route(task, output):
    """Append routing decision to JSONL log file."""
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": task[:200],
            "tier": output["tier"],
            "model": output["model"],
            "reason": output["reason"],
            "latency_ms": output["latency_ms"],
            "classify_ms": output["classify_ms"],
            "execute_ms": output["execute_ms"],
            "cache_status": output["cache_status"],
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[LLMRoute] Log write failed: {e}")


# CLI Mode

def main():
    """Run tasks from command line with cache performance tracking."""
    if len(sys.argv) < 2:
        print('Usage: python3 llmroute.py "Your task here"')
        print("       python3 llmroute.py --interactive")
        print("       python3 llmroute.py --cache-test")
        sys.exit(1)

    if sys.argv[1] == "--cache-test":
        print("=" * 60)
        print("LLMRoute Cache Performance Test")
        print("=" * 60)
        print("Running 5 classifier calls to demonstrate caching...\n")

        test_tasks = [
            "Convert this to JSON: BTC price is 97500",
            "What is the sentiment of: Bitcoin rallies hard",
            "Format this log entry: Trade #3 opened",
            "Summarize: Markets are up 5% today on strong ETF flows",
            "Check status of the Ollama service",
        ]

        for i, task in enumerate(test_tasks, 1):
            start = time.time()
            classification = classify_task(task)
            elapsed = int((time.time() - start) * 1000)

            stats = get_session_stats()
            print(f"[{i}] Tier {classification['tier']} | {elapsed}ms | "
                  f"Cache reads: {stats['cache_read_input_tokens']} tokens")

        stats = get_session_stats()
        print(f"\n{'=' * 60}")
        print("CACHE PERFORMANCE SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Total API calls:           {stats['total_calls']}")
        print(f"  Input tokens (uncached):   {stats['input_tokens']}")
        print(f"  Cache creation tokens:     {stats['cache_creation_input_tokens']}")
        print(f"  Cache read tokens:         {stats['cache_read_input_tokens']}")
        print(f"  Output tokens:             {stats['output_tokens']}")
        print(f"  Est. cost WITHOUT cache:   ${stats['est_cost_without_cache']:.6f}")
        print(f"  Est. cost WITH cache:      ${stats['est_cost_with_cache']:.6f}")
        print(f"  Savings:                   {stats['savings_pct']:.1f}%")
        print(f"{'=' * 60}")
        return

    if sys.argv[1] == "--interactive":
        print("LLMRoute Interactive Mode (type 'quit' to exit, 'stats' for cache stats)")
        print("=" * 50)
        while True:
            try:
                task = input("\nTask: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if task.lower() in ("quit", "exit", "q"):
                break
            if task.lower() == "stats":
                stats = get_session_stats()
                print(f"\nSession Stats:")
                print(f"   Calls: {stats['total_calls']} | "
                      f"Cache reads: {stats['cache_read_input_tokens']} tokens | "
                      f"Savings: {stats['savings_pct']:.1f}%")
                continue
            if not task:
                continue

            result = route_task(task)
            print(f"\nTier {result['tier']} -> {result['model']} | Cache: {result['cache_status']}")
            print(f"Reason: {result['reason']}")
            print(f"Classify: {result['classify_ms']}ms | Execute: {result['execute_ms']}ms | Total: {result['latency_ms']}ms")
            print(f"\nResponse:\n{result['response']}")
    else:
        task = " ".join(sys.argv[1:])
        result = route_task(task)

        print(f"\n{'=' * 50}")
        print(f"Routed to: Tier {result['tier']} -> {result['model']} | Cache: {result['cache_status']}")
        print(f"Reason: {result['reason']}")
        print(f"Classify: {result['classify_ms']}ms | Execute: {result['execute_ms']}ms | Total: {result['latency_ms']}ms")
        print(f"{'=' * 50}")
        print(f"\n{result['response']}")

        stats = get_session_stats()
        print(f"\nTokens -- Input: {stats['input_tokens']} | "
              f"Cache created: {stats['cache_creation_input_tokens']} | "
              f"Cache read: {stats['cache_read_input_tokens']} | "
              f"Output: {stats['output_tokens']}")


if __name__ == "__main__":
    main()
