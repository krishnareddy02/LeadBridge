"""Run Stage 5 live against saved Stage 3+4 snapshot ($0 Apify credits -> generates REPORT.html).

Usage:
  python scripts/test_stage5_from_snapshot.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.drafter import draft_hot_leads
from src.report import render_executive_report_html
from src.schemas import (
    ActorInput,
    CollectorTelemetry,
    ProductProfile,
    Stage3Result,
    Stage4Result,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


async def run_stage5_from_snapshot(snapshot_path: Path) -> None:
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Snapshot not found at {snapshot_path}. Run scripts/test_stage3_4_from_snapshot.py first."
        )

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    actor_input = ActorInput.model_validate(payload["input"])
    profile = ProductProfile.model_validate(payload["stage1_profile"])
    telemetry = [CollectorTelemetry.model_validate(t) for t in payload["stage2_telemetry"]]
    stage3 = Stage3Result.model_validate(payload["stage3_result"])
    stage4 = Stage4Result.model_validate(payload["stage4_result"])

    print("=" * 76)
    print(f"STAGE 5 LIVE RUN FROM SNAPSHOT | Product: {profile.product_name}")
    print("=" * 76)

    stage5 = await draft_hot_leads(stage4=stage4, profile=profile, actor_input=actor_input)
    report_html = render_executive_report_html(
        profile=profile,
        actor_input=actor_input,
        stage2_telemetry=telemetry,
        stage3=stage3,
        stage4=stage4,
        stage5=stage5,
    )

    out_dir = Path("storage")
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "REPORT.html"
    report_path.write_text(report_html, encoding="utf-8")

    dataset_preview_path = out_dir / "stage5_dataset_preview.json"
    dataset_records = [
        r.model_dump(mode="json")
        for r in (
            stage5.hot_lead_records
            + stage5.warm_signal_records
            + stage5.market_gap_records
        )
    ]
    dataset_preview_path.write_text(json.dumps(dataset_records, indent=2), encoding="utf-8")

    print(
        f"\n[Stage 5 Complete] Hot Drafted={len(stage5.hot_lead_records)} | "
        f"Warm={len(stage5.warm_signal_records)} | Batches={stage5.llm_batches_sent} | "
        f"Providers={stage5.providers_used}"
    )

    for idx, lead in enumerate(stage5.hot_lead_records, start=1):
        print(f"\n--- HOT LEAD #{idx} ({lead.winabilityScore}/100 | {lead.competitorMentioned}) ---")
        print(f"URL:    {lead.postUrl}")
        print(f"Safety: {lead.platformSafetyRule}")
        print(f"Proof:  {lead.claimEvidenceSummary}")
        print(f"DRAFT A (Public):\n  {lead.draftPublicReply}")
        print(f"DRAFT B (Direct/DM):\n  {lead.draftDirectPitch}")

    print(f"\nSaved Dataset Preview : {dataset_preview_path.resolve()}")
    print(f"Saved HTML Dashboard  : {report_path.resolve()}")
    print("Tip: Open storage\\REPORT.html in your browser to inspect the Copy Draft buttons!")


if __name__ == "__main__":
    asyncio.run(run_stage5_from_snapshot(Path("storage/stage3_4_snapshot.json")))