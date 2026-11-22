"""Run Stage 3 + Stage 4 live against the saved Stage 1+2 snapshot ($0 Apify credits!).

Usage:
  python scripts/test_stage3_4_from_snapshot.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.brain import score_complaints_batched
from src.filters import filter_complaints
from src.schemas import ActorInput, ProductProfile, Stage2Result

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


async def run_from_snapshot(snapshot_path: Path) -> None:
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Snapshot not found at {snapshot_path}. Run scripts/test_stage1_2_live.py first."
        )

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    actor_input = ActorInput.model_validate(payload["input"])
    profile = ProductProfile.model_validate(payload["stage1_profile"])
    stage2 = Stage2Result.model_validate(payload["stage2_result"])

    print("=" * 76)
    print(f"STAGE 3 -> STAGE 4 LIVE RUN FROM SNAPSHOT | Product: {profile.product_name}")
    print("=" * 76)

    # --- STAGE 3: Deterministic Filter ($0) ---
    stage3 = filter_complaints(
        complaints=stage2.complaints,
        profile=profile,
        actor_input=actor_input,
    )
    print(
        f"\n[Stage 3 Deterministic Filter] Input={stage3.input_count} -> "
        f"Kept={stage3.kept_count} | Dropped={stage3.dropped_count}"
    )
    print(f"  Drop breakdown: {stage3.drop_reason_counts}")
    for d in stage3.dropped_items:
        print(f"   - DROPPED ({d.reason:<19} | rule={d.matched_rule}): [{d.source}/{d.sub_source}] {d.title or d.post_url}")

    # --- STAGE 4: Batched LLM Solvability Gate & Scorer ---
    stage4 = await score_complaints_batched(
        complaints=stage3.clean_complaints,
        profile=profile,
        actor_input=actor_input,
        batch_size=15,
    )
    print(
        f"\n[Stage 4 Batched LLM Brain] Evaluated={stage4.total_evaluated} | "
        f"Batches={stage4.llm_batches_sent} (failed={stage4.llm_batches_failed}) | "
        f"Providers={stage4.providers_used}"
    )
    print(
        f"  Buckets: HOT_LEADS={len(stage4.hot_leads)} | WARM_SIGNALS={len(stage4.warm_signals)} | "
        f"MARKET_GAPS={len(stage4.unresolved_market_gaps)} | DISCARDED={len(stage4.discarded)}"
    )

    print("\n--- HOT LEADS (Score >= 70 -> Ready for Stage 5 Outreach Drafting) ---")
    if not stage4.hot_leads:
        print("  (None)")
    for idx, lead in enumerate(stage4.hot_leads, start=1):
        print(
            f"  {idx}. SCORE {lead.winability_score}/100 "
            f"(Solv={lead.solvability_score}/50 + Urg={lead.urgency_score}/30 + Rec={lead.recency_score}/20)\n"
            f"     Source: [{lead.complaint.source}/{lead.complaint.sub_source}] {lead.complaint.author_handle}\n"
            f"     Comp:   {lead.competitor_mentioned} | Category: {lead.pain_category}\n"
            f"     Pain:   {lead.pain_summary}\n"
            f"     Proof:  {lead.matched_capabilities}\n"
            f"     URL:    {lead.complaint.post_url}"
        )

    print("\n--- WARM / SEO SIGNALS (Score 40-69 -> Market Insights) ---")
    if not stage4.warm_signals:
        print("  (None)")
    for idx, item in enumerate(stage4.warm_signals, start=1):
        print(
            f"  {idx}. SCORE {item.winability_score}/100 | [{item.complaint.source}/{item.complaint.sub_source}] "
            f"{item.competitor_mentioned} -> {item.pain_summary}"
        )

    print("\n--- UNRESOLVED MARKET GAPS (Genuine Pain, Not Solved by Ledger) ---")
    if not stage4.unresolved_market_gaps:
        print("  (None)")
    for idx, item in enumerate(stage4.unresolved_market_gaps, start=1):
        print(
            f"  {idx}. [{item.complaint.source}/{item.complaint.sub_source}] {item.competitor_mentioned}: "
            f"{item.pain_summary} | Gap: {item.rejection_or_gap_reason}"
        )

    print("\n--- DISCARDED BY HARD GATE (Sample First 6) ---")
    for idx, item in enumerate(stage4.discarded[:6], start=1):
        print(
            f"  {idx}. [{item.complaint.source}/{item.complaint.sub_source}] "
            f"{item.pain_category} -> {item.rejection_or_gap_reason}"
        )

    # Save Stage 3+4 snapshot for zero-credit Stage 5 building & testing
    out_path = Path("storage/stage3_4_snapshot.json")
    out_path.write_text(
        json.dumps(
            {
                "input": actor_input.model_dump(mode="json"),
                "stage1_profile": profile.model_dump(mode="json"),
                "stage2_telemetry": [t.model_dump(mode="json") for t in stage2.telemetry],
                "stage3_result": stage3.model_dump(mode="json"),
                "stage4_result": stage4.model_dump(mode="json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved Stage 3+4 snapshot to: {out_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(run_from_snapshot(Path("storage/stage1_2_snapshot.json")))