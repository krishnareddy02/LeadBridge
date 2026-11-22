"""Live Stage 1 -> Stage 2 End-to-End Integration Test.

Usage:
  1. $0 Apify credit check (HN + Bluesky live):
     python scripts/test_stage1_2_live.py

  2. Full Lean Run (Google SERP + Reddit + HN + Bluesky, ~$0.09 Apify credits):
     python scripts/test_stage1_2_live.py --full-lean
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.collectors import harvest_complaints
from src.profiler import build_product_profile
from src.schemas import ActorInput

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


async def run_live_integration(full_lean: bool) -> None:
    sources = (
        ["google_serp", "reddit", "hacker_news", "bluesky"]
        if full_lean
        else ["hacker_news", "bluesky"]
    )
    mode_label = "FULL LEAN MODE (4 Sources)" if full_lean else "FREE LIVE MODE (HN + Bluesky, $0)"

    actor_input = ActorInput(
        productName="Heroku",
        productUrl="https://www.heroku.com",
        productDescription=(
            "Fully managed PaaS cloud platform for deploying Python, Node.js, Go, and Ruby apps "
            "via Git push with managed Postgres, Redis, Review Apps, and zero DevOps setup."
        ),
        competitors=[],
        sources=sources,
        lookbackDays=14,
    )

    print(f"\n{'=' * 74}\nSTAGE 1 -> STAGE 2 LIVE INTEGRATION | {mode_label}\n{'=' * 74}")

    # --- STAGE 1 ---
    profile = await build_product_profile(actor_input=actor_input, enable_apify_fallback=False)
    print(
        f"[Stage 1] Provider={profile.llm_provider_used} | Crawl={profile.crawl_diagnostics.crawl_method} "
        f"| Competitors={profile.competitors}"
    )

    # --- STAGE 2 ---
    stage2 = await harvest_complaints(profile=profile, actor_input=actor_input)

    print("\n[Stage 2 Telemetry]")
    for t in stage2.telemetry:
        print(
            f"  - {t.source:<12} | status={t.status:<7} | items={t.items_collected:<3} "
            f"| fallback={str(t.fallback_used):<5} | error={t.error}"
        )

    print(
        f"\n[Stage 2 Totals] pass1={stage2.pass1_unique_count} | "
        f"pass2_added={stage2.pass2_added_count} | "
        f"fallback_triggered={stage2.fallback_triggered} | "
        f"total_unique={stage2.total_unique_count}"
    )

    print("\n[Sample Harvested Items (First 5)]")
    for idx, item in enumerate(stage2.complaints[:5], start=1):
        preview = item.text.replace("\n", " ")[:180]
        print(
            f"  {idx}. [{item.source} / {item.sub_source}] {item.author_handle}\n"
            f"     URL:  {item.post_url}\n"
            f"     Text: {preview}..."
        )

    # Save snapshot for zero-credit Stage 3 development & testing
    out_dir = Path("storage")
    out_dir.mkdir(exist_ok=True)
    snapshot_path = out_dir / "stage1_2_snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "input": actor_input.model_dump(mode="json"),
                "stage1_profile": profile.model_dump(mode="json"),
                "stage2_result": stage2.model_dump(mode="json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved full Stage 1+2 snapshot to: {snapshot_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-lean",
        action="store_true",
        help="Include Google SERP + Reddit Apify Actors (~$0.09 Apify credits).",
    )
    args = parser.parse_args()
    asyncio.run(run_live_integration(full_lean=args.full_lean))