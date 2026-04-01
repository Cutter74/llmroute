# LLMRoute — Project Claude Code Rules
# 5-tier cost optimization routing layer. Shared across AXIS, Bulwark, future projects.

## Infrastructure
- Production: Hetzner (midas-prod-01), /root/llmroute/
- Ollama: cutter74 localhost:11434 (CPU mode, Qwen 2.5 3B)
- Cost log: /root/llmroute/cost_log.jsonl

## Key File Paths
- Router: /root/llmroute/llmroute.py
- Classifier prompt: /root/llmroute/classifier_prompt.txt
- AXIS scan wrapper: /root/llmroute/axis_scan_wrapper.py
- Env vars: /root/llmroute/axis_env.sh ← source before running anything
- Route log: /root/llmroute/route_log.jsonl
- Cost log: /root/llmroute/cost_log.jsonl

## Tier Definitions — NEVER CHANGE WITHOUT REVIEW
- Tier 1: openai-codex (GPT OAuth, zero marginal cost) — data retrieval, formatting, status checks
- Tier 2: DeepSeek V3 (~$0.26/$0.38 per 1M) — light reasoning, signal scoring, sentiment tagging
- Tier 3: Claude Sonnet (pay-per-use) — trade approvals, Supervisor reviews, risk calls
- Tier 4 (Opus): HARD BLOCKED for automatic fallback — manual/scheduled only
- Never route trading decisions below Tier 3

## Critical Rules
- Opus is HARD BLOCKED in router.py _dispatch() — do NOT re-enable as automatic fallback
- Codex retry-with-backoff: 3 attempts at 0s/5s/15s before falling to Tier 2
- Fallback chain: Codex (3 retries) → DeepSeek → Sonnet → hard stop
- is_scheduled=True flag unlocks Tier 4 — only use for intentional scheduled jobs
- Duplicate cron jobs = runaway cost — always verify crontab -l before adding new jobs

## Cost Targets
- No-signal scans: $0.00 (Tier 1 only)
- Signal scans: ~$0.01 (Tier 3 call)
- Daily target: $0.02-0.12
- Monthly target: $1-3
- Anthropic hard cap: $30/month (set at console.anthropic.com/settings/limits)

## Incident History
- March 19-20: $20.05 runaway — duplicate cron + is_scheduled=True + Opus fallback
- Fix: duplicate cron disabled, Opus hard-blocked, Codex retry added
- Lesson: ALWAYS verify crontab -l shows exactly one scan job before leaving a session

## Deployment Pattern
- Test on cutter74 first (AMD GPU, faster Ollama)
- Deploy to Hetzner via scp after testing
- Source axis_env.sh before any live run
