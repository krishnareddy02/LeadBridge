"""Pure-Python Local End-to-End Tester for All 5 Stages (No `apify run` needed).

Usage examples (run from D:\Apify):
  # 1. Run using local_test_input.json (Full Lean 4-source run + open REPORT.html):
  python scripts/test_local_e2e.py --open

  # 2. Run using local_test_input.json at $0 Apify cost (HN + Bluesky only):
  python scripts/test_local_e2e.py --free-only --open

  # 3. Run a specific product JSON file:
  python scripts/test_local_e2e.py --input my_other_product.json --open

  # 4. Type new product details interactively right in the terminal:
  python scripts/test_local_e2e.py --interactive --open

  # 5. Include G2 Critical Reviews Deep Mode (+~$0.18 Apify credits):
  python scripts/test_local_e2e.py --deep-g2 --open
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from pydantic import ValidationError

from src.brain import score_complaints_batched
from src.collectors import harvest_complaints
from src.drafter import draft_hot_leads
from src.filters import filter_complaints
from src.profiler import build_product_profile
from src.report import render_executive_report_html
from src.schemas import ActorInput

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "product"


def _prompt_interactive_input() -> dict:
    print("\n--- Enter Product Details for Real-World E2E Test ---")
    product_name = input("Product Name (e.g. Plausible Analytics): ").strip()
    product_url = input("Product URL (e.g. https://plausible.io): ").strip()
    product_desc = input("Product Description & Core Advantages: ").strip()
    comp_raw = input("Known Competitors (comma-separated, or press Enter to auto-discover): ").strip()
    lookback_raw = input("Lookback Days [default 14]: ").strip() or "14"
    min_score_raw = input("Min Hot Lead Score 40-95 [default 70]: ").strip() or "70"
    tone = (
        input("Tone (founder_candid / helpful_technical / executive_concise) [default founder_candid]: ").strip()
        or "founder_candid"
    )

    competitors = [c.strip() for c in comp_raw.split(",") if c.strip()] if comp_raw else []
    return {
        "productName": product_name,
        "productUrl": product_url,
        "productDescription": product_desc,
        "competitors": competitors,
        "sources": ["google_serp", "reddit", "hacker_news", "bluesky"],
        "lookbackDays": int(lookback_raw),
        "minWinabilityScore": int(min_score_raw),
        "tone": tone,
    }


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


async def run_local_e2e(
    input_data: dict,
    free_only: bool = False,
    deep_g2: bool = False,
    enable_spa_fallback: bool = False,
    open_report: bool = False,
) -> None:
    # Validate founder input first (catches bad URLs / SSRF before any API call)
    try:
        actor_input = ActorInput.model_validate(input_data)
    except ValidationError as exc:
        print("\n[BLOCKED AT INPUT VALIDATION]")
        print(exc)
        return

    if free_only:
        actor_input.sources = ["hacker_news", "bluesky"]
    elif deep_g2 and "g2" not in actor_input.sources:
        actor_input.sources.append("g2")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("storage") / "local_runs" / f"{_slug(actor_input.productName)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "00_input.json", actor_input.model_dump(mode="json"))

    t_total = time.perf_counter()
    print("\n" + "=" * 78)
    print(f"LOCAL 5-STAGE E2E RUN | Product: {actor_input.productName}")
    print(
        f"URL: {actor_input.productUrl} | Sources: {actor_input.sources} | "
        f"Lookback: {actor_input.lookbackDays}d | Tone: {actor_input.tone}"
    )
    print("=" * 78)

    # =========================================================================
    # STAGE 1: Auto-Profiler
    # =========================================================================
    t1 = time.perf_counter()
    profile = await build_product_profile(
        actor_input=actor_input,
        enable_apify_fallback=enable_spa_fallback,
    )
    s1_sec = time.perf_counter() - t1
    _write_json(run_dir / "01_stage1_profile.json", profile.model_dump(mode="json"))

    print(f"\n[STAGE 1: AUTO-PROFILER] ({s1_sec:.1f}s | {profile.llm_provider_used})")
    print(
        f"  Crawl Method : {profile.crawl_diagnostics.crawl_method} "
        f"(chars={profile.crawl_diagnostics.extracted_char_count}, urls={profile.crawl_diagnostics.urls_fetched})"
    )
    if profile.crawl_diagnostics.warning:
        print(f"  Crawl Warning: {profile.crawl_diagnostics.warning}")
    print(f"  Category     : {profile.product_category}")
    print(f"  Competitors  : {profile.competitors}")
    print(f"  Domain Terms : {profile.domain_context_terms}")
    print(f"  Ledger ({len(profile.verified_capabilities)} verified capabilities):")
    for idx, cap in enumerate(profile.verified_capabilities, start=1):
        print(f"    {idx}. {cap}")

    # =========================================================================
    # STAGE 2: Multi-Source Async Complaint Harvester
    # =========================================================================
    t2 = time.perf_counter()
    stage2 = await harvest_complaints(profile=profile, actor_input=actor_input)
    s2_sec = time.perf_counter() - t2
    _write_json(run_dir / "02_stage2_harvested.json", stage2.model_dump(mode="json"))

    print(
        f"\n[STAGE 2: MULTI-SOURCE HARVESTER] ({s2_sec:.1f}s | "
        f"pass1={stage2.pass1_unique_count}, pass2_added={stage2.pass2_added_count}, "
        f"fallback={stage2.fallback_triggered}, total_unique={stage2.total_unique_count})"
    )
    for t in stage2.telemetry:
        print(
            f"  - {t.source:<12} | status={t.status:<7} | items={t.items_collected:<3} "
            f"| fallback={str(t.fallback_used):<5} | err={t.error}"
        )

    # =========================================================================
    # STAGE 3: Deterministic Noise & Own-Brand Filter ($0 Compute)
    # =========================================================================
    t3 = time.perf_counter()
    stage3 = filter_complaints(
        complaints=stage2.complaints,
        profile=profile,
        actor_input=actor_input,
    )
    s3_sec = time.perf_counter() - t3
    _write_json(run_dir / "03_stage3_filtered.json", stage3.model_dump(mode="json"))

    print(
        f"\n[STAGE 3: DETERMINISTIC FILTER ($0)] ({s3_sec:.3f}s | "
        f"input={stage3.input_count} -> kept={stage3.kept_count}, dropped={stage3.dropped_count})"
    )
    if stage3.drop_reason_counts:
        print(f"  Drop Reasons : {stage3.drop_reason_counts}")
        for d in stage3.dropped_items[:5]:
            print(f"   - dropped ({d.reason} | {d.matched_rule}): {d.title or d.post_url}")

    # =========================================================================
    # STAGE 4: Batched LLM Solvability Hard-Gate & Winability Scorer
    # =========================================================================
    t4 = time.perf_counter()
    stage4 = await score_complaints_batched(
        complaints=stage3.clean_complaints,
        profile=profile,
        actor_input=actor_input,
    )
    s4_sec = time.perf_counter() - t4
    _write_json(run_dir / "04_stage4_scored.json", stage4.model_dump(mode="json"))

    print(
        f"\n[STAGE 4: BATCHED SOLVABILITY GATE & SCORER] ({s4_sec:.1f}s | "
        f"batches={stage4.llm_batches_sent}, failed={stage4.llm_batches_failed}, providers={stage4.providers_used})"
    )
    print(
        f"  Hot Leads={len(stage4.hot_leads)} | Warm Signals={len(stage4.warm_signals)} | "
        f"Market Gaps={len(stage4.unresolved_market_gaps)} | Discarded={len(stage4.discarded)}"
    )

    # =========================================================================
    # STAGE 5: Conditional Dual Ban-Safe Drafter & REPORT.html
    # =========================================================================
    t5 = time.perf_counter()
    stage5 = await draft_hot_leads(
        stage4=stage4,
        profile=profile,
        actor_input=actor_input,
    )
    report_html = render_executive_report_html(
        profile=profile,
        actor_input=actor_input,
        stage2_telemetry=stage2.telemetry,
        stage3=stage3,
        stage4=stage4,
        stage5=stage5,
    )
    s5_sec = time.perf_counter() - t5
    total_sec = time.perf_counter() - t_total

    dataset_records = [
        r.model_dump(mode="json")
        for r in (
            stage5.hot_lead_records
            + stage5.warm_signal_records
            + stage5.market_gap_records
        )
    ]
    _write_json(run_dir / "05_stage5_dataset.json", dataset_records)
    report_path = run_dir / "REPORT.html"
    report_path.write_text(report_html, encoding="utf-8")

    print(
        f"\n[STAGE 5: DUAL DRAFTER & REPORT.HTML] ({s5_sec:.1f}s | "
        f"hot_drafted={len(stage5.hot_lead_records)}, warm={len(stage5.warm_signal_records)}, "
        f"gaps={len(stage5.market_gap_records)}, providers={stage5.providers_used})"
    )

    if stage5.hot_lead_records:
        print("\n--- TOP HOT LEADS & READY-TO-SEND DRAFTS ---")
        for i, lead in enumerate(stage5.hot_lead_records[:5], start=1):
            print(
                f"\n  #{i} SCORE {lead.winabilityScore}/100 "
                f"(Solv={lead.solvabilityScore}/50 + Urg={lead.urgencyScore}/30 + Rec={lead.recencyScore}/20)"
            )
            print(f"     Source : {lead.source} ({lead.subSource}) | {lead.authorHandle}")
            print(f"     Comp   : {lead.competitorMentioned} | Theme: {lead.painCategory}")
            print(f"     Pain   : {lead.painSummary}")
            print(f"     Proof  : {lead.matchedCapabilities}")
            print(f"     Safety : {lead.platformSafetyRule}")
            print(f"     URL    : {lead.postUrl}")
            print(f"     Draft A (Public):\n       {lead.draftPublicReply}")
            print(f"     Draft B (Direct/DM):\n       {lead.draftDirectPitch}")
    else:
        print("\n  No Hot Leads crossed the minWinabilityScore threshold in this run.")

    if stage5.warm_signal_records:
        print("\n--- WARM / SEO SIGNALS (No Outreach Drafted) ---")
        for w in stage5.warm_signal_records[:5]:
            print(f"  - [{w.winabilityScore}/100] {w.competitorMentioned} ({w.subSource}): {w.painSummary}")

    if stage4.discarded:
        print("\n--- HARD-GATE REJECTED NOISE (Sample First 4) ---")
        for d in stage4.discarded[:4]:
            print(f"  - [{d.complaint.source}/{d.complaint.sub_source}] {d.rejection_or_gap_reason}")

    # =========================================================================
    # AUTOMATED END-TO-END QUALITY VERIFICATION
    # =========================================================================
    own_low = actor_input.productName.lower()
    own_pat = re.compile(rf"\b{re.escape(actor_input.productName)}\b", re.IGNORECASE)
    ledger_set = set(profile.verified_capabilities)

    checks = [
        (
            "Stage 1: Own brand excluded from competitors",
            all(c.lower() != own_low for c in profile.competitors),
        ),
        (
            "Stage 1: Own brand excluded from search queries",
            all(not own_pat.search(q) for q in profile.competitor_pain_queries),
        ),
        (
            "Stage 3: Zero own-brand mentions survive into Stage 4",
            all(not own_pat.search(f"{c.title} {c.text} {c.post_url}") for c in stage3.clean_complaints),
        ),
        (
            "Stage 4/5: Every Hot Lead cites ONLY exact Verified Capabilities",
            all(
                len(r.matchedCapabilities) > 0 and set(r.matchedCapabilities).issubset(ledger_set)
                for r in stage5.hot_lead_records
            ),
        ),
        (
            "Stage 5: Warm signals have empty drafts (no wasted drafting)",
            all(r.draftPublicReply == "" and r.draftDirectPitch == "" for r in stage5.warm_signal_records),
        ),
        (
            "Stage 5: REPORT.html compiled cleanly",
            len(report_html) > 1500,
        ),
    ]

    print("\n" + "=" * 78)
    print(f"END-TO-END VERIFICATION SCORECARD | Total Runtime: {total_sec:.1f}s")
    print("=" * 78)
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    print(f"\nAll stage outputs saved in:\n  {run_dir.resolve()}")
    print("  - 01_stage1_profile.json")
    print("  - 02_stage2_harvested.json")
    print("  - 03_stage3_filtered.json")
    print("  - 04_stage4_scored.json")
    print("  - 05_stage5_dataset.json")
    print("  - REPORT.html")

    if open_report:
        webbrowser.open(report_path.resolve().as_uri())


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-Python local 5-stage E2E tester.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("local_test_input.json"),
        help="Path to product JSON input file (default: local_test_input.json).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for product name, URL, description, and competitors in terminal.",
    )
    parser.add_argument(
        "--free-only",
        action="store_true",
        help="Run $0 APIs only (Hacker News + Bluesky, zero Apify credits used).",
    )
    parser.add_argument(
        "--deep-g2",
        action="store_true",
        help="Also include G2 critical reviews (+~$0.18 Apify credits).",
    )
    parser.add_argument(
        "--spa-fallback",
        action="store_true",
        help="Allow Stage 1 apify/website-content-crawler fallback if URL is a JS SPA.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open generated REPORT.html in your browser.",
    )
    args = parser.parse_args()

    if args.interactive:
        data = _prompt_interactive_input()
    else:
        if not args.input.exists():
            raise FileNotFoundError(
                f"Input file not found: {args.input}. Create local_test_input.json or pass --interactive."
            )
        data = json.loads(args.input.read_text(encoding="utf-8"))

    asyncio.run(
        run_local_e2e(
            input_data=data,
            free_only=args.free_only,
            deep_g2=args.deep_g2,
            enable_spa_fallback=args.spa_fallback,
            open_report=args.open,
        )
    )


if __name__ == "__main__":
    main()