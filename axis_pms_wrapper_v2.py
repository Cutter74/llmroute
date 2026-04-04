#!/usr/bin/env python3
"""
AXIS_PMS Wrapper v2.0
Runs axis_pms_scan_v2.py as a subprocess, parses pipe-delimited stdout signals,
posts RED/YELLOW signals and scan summaries to Discord.
Cron: */15 * * * * via axis_env.sh
"""

import os
import sys
import subprocess
import json
import time
import urllib.request
from datetime import datetime, timezone

SCANNER_SCRIPT = "/root/llmroute/axis_pms_scan_v2.py"
SUBPROCESS_TIMEOUT = 300  # 5 minutes

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")

TIER_EMOJI = {"RED": "\U0001f534", "YELLOW": "\U0001f7e1"}


def log(msg):
    """Timestamped log to stderr (cron captures to /var/log/axis-pms-v2.log)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", file=sys.stderr)


def send_discord(msg, dry_run=False):
    """Post a message to the Discord webhook. No-op in dry-run mode."""
    if dry_run:
        log(f"DRY-RUN Discord (suppressed): {msg[:120]}...")
        return
    if not DISCORD_WEBHOOK:
        log("WARNING: DISCORD_WEBHOOK_URL not set, skipping post")
        return
    try:
        payload = json.dumps({"content": msg}).encode("utf-8")
        req = urllib.request.Request(
            DISCORD_WEBHOOK,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "AXIS-PMS/2.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as e:
        log(f"Discord post failed: {e}")


def format_signal_message(parts):
    """Format a SIGNAL| line into a Discord message.

    Expected parts (index 0 is 'SIGNAL'):
      1=tier, 2=ai_est, 3=crowd, 4=gap, 5=question, 6=slug, 7=direction, 8=volume24h, 9=whale
    """
    tier = parts[1] if len(parts) > 1 else "?"
    ai_est = parts[2] if len(parts) > 2 else "?"
    crowd = parts[3] if len(parts) > 3 else "?"
    gap = parts[4] if len(parts) > 4 else "?"
    question = parts[5] if len(parts) > 5 else "Unknown"
    slug = parts[6] if len(parts) > 6 else ""
    direction = parts[7] if len(parts) > 7 else ""
    volume = parts[8] if len(parts) > 8 else "0"
    whale = parts[9] if len(parts) > 9 else "False"

    emoji = TIER_EMOJI.get(tier, "\u26aa")

    try:
        ai_pct = f"{float(ai_est) * 100:.1f}%"
        crowd_pct = f"{float(crowd) * 100:.1f}%"
    except (ValueError, TypeError):
        ai_pct = ai_est
        crowd_pct = crowd

    try:
        vol_fmt = f"${float(volume):,.0f}"
    except (ValueError, TypeError):
        vol_fmt = f"${volume}"

    whale_str = "\u2705 Whale" if whale.strip().lower() == "true" else ""

    msg = (
        f"{emoji} **[AXIS_PMS] {tier} Signal**\n"
        f"**{question}**\n"
        f"Direction: **{direction}** | Gap: {gap}%\n"
        f"AI: {ai_pct} vs Crowd: {crowd_pct} | Vol: {vol_fmt}"
    )
    if whale_str:
        msg += f" | {whale_str}"
    msg += f"\nhttps://polymarket.com/event/{slug}"
    return msg


def main():
    dry_run = "--dry-run" in sys.argv
    log(f"=== AXIS_PMS Wrapper v2.0 starting ({'DRY RUN' if dry_run else 'LIVE'}) ===")

    if not os.path.exists(SCANNER_SCRIPT):
        log(f"FATAL: Scanner not found at {SCANNER_SCRIPT}")
        send_discord(
            "\u26a0\ufe0f **[AXIS_PMS] Scanner script not found!** Check deployment.",
            dry_run=False,
        )
        return 1

    cmd = [sys.executable, SCANNER_SCRIPT]
    if dry_run:
        cmd.append("--dry-run")

    log(f"Running: {' '.join(cmd)}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        log(f"FATAL: Scanner timed out after {SUBPROCESS_TIMEOUT}s")
        send_discord(
            f"\u26a0\ufe0f **[AXIS_PMS] Scanner TIMEOUT** \u2014 killed after {SUBPROCESS_TIMEOUT // 60} minutes",
            dry_run=dry_run,
        )
        return 1
    except Exception as e:
        log(f"FATAL: Scanner failed to start: {e}")
        send_discord(
            f"\u26a0\ufe0f **[AXIS_PMS] Scanner CRASH:** {str(e)[:200]}",
            dry_run=dry_run,
        )
        return 1

    elapsed = time.time() - start
    log(f"Scanner finished in {elapsed:.1f}s (exit code: {result.returncode})")

    # Relay scanner stderr (LOG| lines) to our stderr for cron log capture
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            print(f"  {line}", file=sys.stderr)

    # Parse stdout for SIGNAL| and SCAN_COMPLETE| lines
    signals_posted = 0
    scan_summary = None

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("SIGNAL|"):
                parts = line.split("|")
                tier = parts[1] if len(parts) > 1 else ""
                if tier in ("RED", "YELLOW"):
                    msg = format_signal_message(parts)
                    send_discord(msg, dry_run=dry_run)
                    signals_posted += 1
                    log(f"Signal: {tier} — {parts[6] if len(parts) > 6 else '?'}")
                elif tier == "GREEN":
                    log(f"Signal (no post): GREEN — {parts[6] if len(parts) > 6 else '?'}")

            elif line.startswith("SCAN_COMPLETE|"):
                parts = line.split("|")
                try:
                    scanned = int(parts[1])
                    total_sig = int(parts[2])
                    reds = int(parts[3])
                    yellows = int(parts[4])
                    scan_summary = (scanned, total_sig, reds, yellows)
                except (IndexError, ValueError) as e:
                    log(f"Failed to parse SCAN_COMPLETE: {e}")

    # Post scan summary to Discord only if signals were found
    if scan_summary:
        scanned, total_sig, reds, yellows = scan_summary
        if total_sig > 0:
            ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
            summary = (
                f"\U0001f4ca **[AXIS_PMS] Scan Complete** ({ts})\n"
                f"Markets scanned: {scanned} | Signals: {total_sig} "
                f"(\U0001f534 {reds} | \U0001f7e1 {yellows})"
            )
            send_discord(summary, dry_run=dry_run)
        log(f"Scan summary: {scanned} scanned, {total_sig} signals (RED:{reds} YELLOW:{yellows})")

    if result.returncode != 0:
        log(f"Scanner exited with code {result.returncode}")
        send_discord(
            f"\u26a0\ufe0f **[AXIS_PMS] Scanner error** \u2014 exit code {result.returncode}",
            dry_run=dry_run,
        )
        return result.returncode

    log(f"=== AXIS_PMS Wrapper complete | {signals_posted} signals posted ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
