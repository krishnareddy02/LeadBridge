"""Stage 5 unit & failure-case tests: drafting, XSS-safe REPORT.html, and PPE budget safety."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.drafter import draft_hot_leads, get_platform_safety_rule
from src.llm_router import LLMRouter, LLMRouterError
from src.main import save_qualified_hot_lead
from src.report import render_executive_report_html
from src.schemas import (
    ActorInput,
    CollectorTelemetry,
    CrawlDiagnostics,
    DatasetLeadRecord,
    LLMBatchDraftResponse,
    LLMLeadDraft,
    ProductProfile,
    RawComplaint,
    ScoredComplaint,
    Stage3Result,
    Stage4Result,
)


def _sample_input() -> ActorInput:
    return ActorInput(
        productName="InvoiceSpark",
        productUrl="https://invoicespark.io",
        productDescription="Flat $15/mo invoicing for freelancers with unlimited clients and automated overdue reminders.",
        lookbackDays=14,
        minWinabilityScore=70,
        tone="founder_candid",
    )


def _sample_profile() -> ProductProfile:
    return ProductProfile(
        product_name="InvoiceSpark",
        product_url="https://invoicespark.io",
        root_domain="invoicespark.io",
        product_category="Freelance Invoicing SaaS",
        target_audience="Freelancers and small agencies",
        verified_capabilities=[
            "Flat $15/mo pricing with unlimited clients",
            "Automated polite overdue payment reminders",
            "1-click CSV import from FreshBooks/QuickBooks",
        ],
        pain_points_solved=["Per-client tier jumps", "Manual overdue chasing"],
        competitors=["FreshBooks", "QuickBooks Online", "Wave"],
        competitor_pain_queries=[
            "FreshBooks alternative",
            "QuickBooks pricing increase",
            "Wave invoicing issues",
        ],
        category_problem_queries=["flat pricing invoicing", "overdue invoice reminders"],
        g2_product_slugs=["freshbooks", "quickbooks-online"],
        domain_context_terms=["invoicing", "invoice", "billing", "clients"],
        exclude_terms=["InvoiceSpark", "invoicespark.io", "coupon"],
        llm_provider_used="gemini:gemini-3.5-flash-lite",
        crawl_diagnostics=CrawlDiagnostics(
            crawl_method="httpx_direct",
            urls_fetched=["https://invoicespark.io"],
            http_status=200,
            extracted_char_count=1200,
        ),
    )


def _sample_stage4(inject_xss: bool = False) -> Stage4Result:
    title = '<script>alert("xss")</script> FreshBooks cap' if inject_xss else "FreshBooks 50-client cap"
    hot = ScoredComplaint(
        complaint=RawComplaint(
            id="hot1",
            source="reddit",
            sub_source="r/freelance",
            author_handle="u/agency_dev",
            post_url="https://www.reddit.com/r/freelance/comments/hot1/cap/",
            published_at=datetime(2026, 11, 21, 9, 0, tzinfo=timezone.utc),
            title=title,
            text="FreshBooks bumped my tier at 50 clients and I hate chasing overdue invoices manually.",
        ),
        lead_tier="hot_lead",
        is_genuine_pain=True,
        can_client_resolve=True,
        competitor_mentioned="FreshBooks",
        pain_category="per_client_pricing_cap",
        pain_summary="Forced FreshBooks tier upgrade at 50 clients."
        + (' <img src=x onerror="alert(1)">' if inject_xss else ""),
        matched_capabilities=[
            "Flat $15/mo pricing with unlimited clients",
            "Automated polite overdue payment reminders",
        ],
        solvability_score=48,
        urgency_score=27,
        recency_score=20,
        winability_score=95,
    )
    warm = ScoredComplaint(
        complaint=RawComplaint(
            id="warm1",
            source="hacker_news",
            sub_source="Hacker News",
            author_handle="hn_user",
            post_url="https://news.ycombinator.com/item?id=55555",
            title="Wave vs FreshBooks",
            text="Comparing Wave and FreshBooks for small agency billing.",
        ),
        lead_tier="warm_signal",
        is_genuine_pain=True,
        can_client_resolve=True,
        competitor_mentioned="Wave",
        pain_category="tool_comparison",
        pain_summary="Comparing Wave and FreshBooks.",
        matched_capabilities=["Flat $15/mo pricing with unlimited clients"],
        solvability_score=35,
        urgency_score=15,
        recency_score=10,
        winability_score=60,
    )
    rejected = ScoredComplaint(
        complaint=RawComplaint(
            id="rej1",
            source="reddit",
            sub_source="r/smallbusiness",
            author_handle="u/payroll_user",
            post_url="https://www.reddit.com/r/smallbusiness/comments/rej1/payroll/",
            title="QuickBooks payroll W-2 broken",
            text="QuickBooks broke quarterly W-2 payroll filings.",
        ),
        lead_tier="discarded",
        is_genuine_pain=False,
        can_client_resolve=False,
        competitor_mentioned="QuickBooks Online",
        pain_category="off_topic",
        pain_summary="W-2 payroll tax filing failure.",
        rejection_or_gap_reason="InvoiceSpark does not offer W-2 payroll tax filing.",
    )
    return Stage4Result(
        hot_leads=[hot],
        warm_signals=[warm],
        unresolved_market_gaps=[],
        discarded=[rejected],
        total_evaluated=3,
        llm_batches_sent=1,
        providers_used=["gemini:gemini-3.5-flash-lite"],
    )


# =====================================================================
# VALID CASES
# =====================================================================


@pytest.mark.asyncio
async def test_stage5_valid_dual_drafts_and_report_html_rendering():
    """Valid Case: Hot Leads receive grounded Draft A/B; Warm Signals do not; REPORT.html compiles."""
    mock_router = AsyncMock(spec=LLMRouter)
    mock_router.generate_structured.return_value = (
        LLMBatchDraftResponse(
            drafts=[
                LLMLeadDraft(
                    idx=0,
                    draft_public_reply=(
                        "Quick tip: archiving inactive one-off clients in FreshBooks often keeps you under "
                        "the active cap. Full disclosure: I'm the founder of InvoiceSpark, which offers "
                        "flat $15/mo pricing with unlimited clients and automated overdue reminders."
                    ),
                    draft_direct_pitch=(
                        "Hey u/agency_dev! Saw you hit FreshBooks' 50-client tier jump. "
                        "InvoiceSpark keeps pricing flat at $15/mo for unlimited clients with automated "
                        "overdue payment reminders—happy to walk you through our CSV import."
                    ),
                    claim_evidence_summary="Cited: Flat $15/mo unlimited clients; Automated overdue reminders.",
                )
            ]
        ),
        "gemini:gemini-3.5-flash-lite",
    )

    stage4 = _sample_stage4()
    stage5 = await draft_hot_leads(stage4, _sample_profile(), _sample_input(), llm_router=mock_router)

    assert len(stage5.hot_lead_records) == 1
    assert len(stage5.warm_signal_records) == 1
    assert stage5.hot_lead_records[0].draftPublicReply.startswith("Quick tip:")
    assert "Reddit (r/freelance)" in stage5.hot_lead_records[0].platformSafetyRule
    # Warm signals must NOT have outreach drafts generated
    assert stage5.warm_signal_records[0].draftPublicReply == ""
    assert stage5.warm_signal_records[0].draftDirectPitch == ""

    html_doc = render_executive_report_html(
        profile=_sample_profile(),
        actor_input=_sample_input(),
        stage2_telemetry=[CollectorTelemetry(source="reddit", status="ok", items_collected=3)],
        stage3=Stage3Result(input_count=5, kept_count=3, dropped_count=2),
        stage4=stage4,
        stage5=stage5,
    )
    assert "InvoiceSpark" in html_doc
    assert "Copy Draft A" in html_doc
    assert "Copy Draft B" in html_doc
    assert "W-2 payroll tax filing" in html_doc  # Hard-Gate rejection proof included


# =====================================================================
# FAILURE & SAFETY CASES
# =====================================================================


@pytest.mark.asyncio
async def test_stage5_failure_case_all_llms_down_uses_deterministic_fallback_draft():
    """Failure Case: Total LLM outage during Stage 5 never loses a qualified Hot Lead."""
    mock_router = AsyncMock(spec=LLMRouter)
    mock_router.generate_structured.side_effect = LLMRouterError("All LLMs down")

    stage5 = await draft_hot_leads(
        _sample_stage4(), _sample_profile(), _sample_input(), llm_router=mock_router
    )

    assert len(stage5.hot_lead_records) == 1
    lead = stage5.hot_lead_records[0]
    assert lead.draftProvider == "deterministic_fallback"
    assert "Flat $15/mo pricing with unlimited clients" in lead.draftPublicReply
    assert "InvoiceSpark" in lead.draftDirectPitch


@pytest.mark.asyncio
async def test_stage5_failure_case_xss_payload_escaped_in_report_html():
    """Failure Case: Malicious <script> / onerror HTML in scraped Reddit post is safely escaped."""
    stage4 = _sample_stage4(inject_xss=True)
    mock_router = AsyncMock(spec=LLMRouter)
    mock_router.generate_structured.side_effect = LLMRouterError("Use fallback")

    stage5 = await draft_hot_leads(stage4, _sample_profile(), _sample_input(), llm_router=mock_router)
    html_doc = render_executive_report_html(
        profile=_sample_profile(),
        actor_input=_sample_input(),
        stage2_telemetry=[],
        stage3=Stage3Result(input_count=3, kept_count=3, dropped_count=0),
        stage4=stage4,
        stage5=stage5,
    )

    assert '<script>alert("xss")</script>' not in html_doc
    assert '<img src=x onerror="alert(1)">' not in html_doc
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html_doc


@pytest.mark.asyncio
async def test_stage5_failure_case_ppe_spending_limit_stops_pushing_gracefully(monkeypatch):
    """Failure Case: When PPE event_charge_limit_reached=True and charged_count=0, do NOT push data."""
    monkeypatch.setenv("ACTOR_TEST_PAY_PER_EVENT", "true")

    lead_record = DatasetLeadRecord(
        recordType="hot_lead",
        winabilityScore=92,
        source="reddit",
        subSource="r/freelance",
        authorHandle="u/test",
        competitorMentioned="FreshBooks",
        painCategory="pricing",
        painSummary="Too expensive",
        postUrl="https://reddit.com/r/freelance/comments/1/x/",
    )

    mock_charge = AsyncMock(
        return_value=SimpleNamespace(charged_count=0, event_charge_limit_reached=True)
    )
    mock_push = AsyncMock()

    with (
        patch("src.main.Actor.charge", mock_charge),
        patch("src.main.Actor.push_data", mock_push),
    ):
        should_continue = await save_qualified_hot_lead(lead_record)

    assert should_continue is False
    mock_push.assert_not_awaited()


def test_platform_safety_rules_cover_all_sources():
    assert "Reddit (r/webhosting)" in get_platform_safety_rule(
        "google_serp", "r/webhosting", "https://www.reddit.com/r/webhosting/comments/1/x/"
    )
    assert "Hacker News" in get_platform_safety_rule(
        "hacker_news", "Hacker News", "https://news.ycombinator.com/item?id=123"
    )
    assert "Bluesky" in get_platform_safety_rule(
        "bluesky", "Bluesky", "https://bsky.app/profile/dev/post/123"
    )
    assert "GDPR" in get_platform_safety_rule(
        "g2", "G2 (FreshBooks)", "https://www.g2.com/products/freshbooks/reviews/1"
    )