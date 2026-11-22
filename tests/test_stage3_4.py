"""Stage 3 & Stage 4 unit and failure-case test suite ($0 cost)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from src.brain import compute_recency_score, score_complaints_batched
from src.filters import filter_complaints
from src.llm_router import LLMRouter
from src.schemas import (
    ActorInput,
    CrawlDiagnostics,
    LLMBatchEvaluationResponse,
    LLMComplaintEvaluation,
    ProductProfile,
    RawComplaint,
)

NOW = datetime(2026, 11, 21, 12, 0, 0, tzinfo=timezone.utc)


def _sample_input() -> ActorInput:
    return ActorInput(
        productName="InvoiceSpark",
        productUrl="https://invoicespark.io",
        productDescription="Flat $15/mo invoicing for freelancers with unlimited clients and automated overdue reminders.",
        lookbackDays=14,
        minWinabilityScore=70,
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
            "0% markup on Stripe/Wise/UPI",
            "1-click CSV import from FreshBooks/QuickBooks",
        ],
        pain_points_solved=["Per-client tier jumps", "Chasing unpaid invoices"],
        competitors=["FreshBooks", "QuickBooks Online", "Wave"],
        competitor_pain_queries=[
            "FreshBooks alternative",
            "QuickBooks pricing increase",
            "Wave invoicing issues",
        ],
        category_problem_queries=[
            "flat pricing invoicing",
            "automated overdue invoice reminders",
        ],
        g2_product_slugs=["freshbooks", "quickbooks-online"],
        domain_context_terms=["invoicing", "invoice", "billing", "clients"],
        exclude_terms=["InvoiceSpark", "invoicespark.io", "coupon", "promo code", "we are hiring"],
        llm_provider_used="gemini:gemini-3.5-flash-lite",
        crawl_diagnostics=CrawlDiagnostics(
            crawl_method="httpx_direct",
            urls_fetched=["https://invoicespark.io"],
            http_status=200,
            extracted_char_count=1200,
        ),
    )


def _make_complaint(
    cid: str,
    text: str,
    url: str | None = None,
    source: str = "reddit",
    title: str = "Test post",
    age_hours: float | None = 4.0,
) -> RawComplaint:
    pub = None if age_hours is None else NOW - timedelta(hours=age_hours)
    return RawComplaint(
        id=cid,
        source=source,  # type: ignore[arg-type]
        sub_source="r/freelance",
        author_handle="u/tester",
        post_url=url or f"https://www.reddit.com/r/freelance/comments/{cid}/post/",
        published_at=pub,
        title=title,
        text=text,
    )


# =====================================================================
# STAGE 3 VALID & FAILURE CASES
# =====================================================================


def test_stage3_drops_own_brand_spam_profiles_old_and_crosspost_dupes():
    """Stage 3 deterministically drops all 6 noise categories while preserving clean leads."""
    complaints = [
        # KEEP 1: Valid competitor complaint
        _make_complaint(
            "keep1",
            "FreshBooks bumped my tier after 50 clients and I hate chasing late invoices.",
            age_hours=5,
        ),
        # KEEP 2: Google SERP item with published_at=None (already after:-bounded)
        _make_complaint(
            "keep2",
            "QuickBooks Online raised prices again, looking for a simple flat-rate alternative.",
            source="google_serp",
            url="https://www.reddit.com/r/smallbusiness/comments/keep2/qb/",
            age_hours=None,
        ),
        # DROP: Exact duplicate URL with UTM params
        _make_complaint(
            "dup_url",
            "FreshBooks bumped my tier after 50 clients and I hate chasing late invoices.",
            url="https://www.reddit.com/r/freelance/comments/keep1/post/?utm_source=share",
        ),
        # DROP: Cross-posted identical body on different URL
        _make_complaint(
            "dup_body",
            "FreshBooks bumped my tier after 50 clients and I hate chasing late invoices.",
            url="https://www.reddit.com/r/smallbusiness/comments/other99/repost/",
        ),
        # DROP: Own-brand mention
        _make_complaint(
            "own_brand",
            "Has anyone compared InvoiceSpark vs FreshBooks for small agencies?",
        ),
        # DROP: Spam / student discount / promo
        _make_complaint(
            "spam1",
            "Grab this FreshBooks promo code for 50% off during our Black Friday deal!",
        ),
        # DROP: Bluesky profile homepage (not a /post/ URL)
        _make_complaint(
            "bsky_profile",
            "Dave Smart profile bio discussing FreshBooks invoicing tools and billing.",
            source="google_serp",
            url="https://bsky.app/profile/davesmart.bsky.social",
        ),
        # DROP: Older than 14-day lookback (+1d buffer -> > 15 days old)
        _make_complaint(
            "too_old",
            "FreshBooks pricing annoyed me back in September when I had 60 clients.",
            age_hours=24 * 25,
        ),
        # DROP: Too short fragment
        _make_complaint("short1", "FreshBooks sucks"),
    ]

    result = filter_complaints(complaints, _sample_profile(), _sample_input(), now=NOW)

    assert result.input_count == 9
    assert result.kept_count == 2
    assert [c.id for c in result.clean_complaints] == ["keep1", "keep2"]
    assert result.drop_reason_counts == {
        "duplicate_url": 1,
        "duplicate_content": 1,
        "own_brand_mention": 1,
        "spam_or_promo": 1,
        "non_post_url": 1,
        "older_than_lookback": 1,
        "too_short": 1,
    }


# =====================================================================
# STAGE 4 VALID & FAILURE CASES
# =====================================================================


@pytest.mark.asyncio
async def test_stage4_valid_hot_lead_warm_signal_market_gap_and_discarded():
    """Stage 4 routes into all 4 buckets and computes deterministic 0-100 scores."""
    c_hot = _make_complaint(
        "hot1",
        "FreshBooks forced a tier upgrade at 50 clients and I chase overdue invoices manually. Need flat pricing!",
        age_hours=4,  # Recency = 20
    )
    c_warm = _make_complaint(
        "warm1",
        "Curious how Wave compares to FreshBooks for simple freelancer invoicing.",
        age_hours=24 * 10,  # Recency = 9
    )
    c_gap = _make_complaint(
        "gap1",
        "QuickBooks Online completely broke my quarterly W-2 payroll tax filings. Need payroll software!",
        age_hours=12,
    )
    c_noise = _make_complaint(
        "noise1",
        "Just shipped v2.0 of my open-source CLI tool built over the weekend.",
        age_hours=2,
    )

    mock_router = AsyncMock(spec=LLMRouter)
    mock_router.generate_structured.return_value = (
        LLMBatchEvaluationResponse(
            evaluations=[
                LLMComplaintEvaluation(
                    idx=0,
                    is_genuine_pain_or_seeking_alternative=True,
                    competitor_mentioned="FreshBooks",
                    pain_category="per_client_pricing_cap",
                    pain_summary="Forced FreshBooks tier upgrade at 50 clients and manual late-invoice chasing.",
                    can_client_resolve=True,
                    matched_capability_indices=[1, 2, 4],
                    solvability_score=48,
                    urgency_score=27,
                ),
                LLMComplaintEvaluation(
                    idx=1,
                    is_genuine_pain_or_seeking_alternative=True,
                    competitor_mentioned="Wave",
                    pain_category="tool_comparison",
                    pain_summary="Comparing Wave and FreshBooks for simple invoicing.",
                    can_client_resolve=True,
                    matched_capability_indices=[1],
                    solvability_score=30,
                    urgency_score=12,
                ),
                LLMComplaintEvaluation(
                    idx=2,
                    is_genuine_pain_or_seeking_alternative=True,
                    competitor_mentioned="QuickBooks Online",
                    pain_category="payroll_tax_filing",
                    pain_summary="QuickBooks Online messed up quarterly W-2 payroll tax filings.",
                    can_client_resolve=False,
                    matched_capability_indices=[],
                    solvability_score=0,
                    urgency_score=0,
                    rejection_or_gap_reason="InvoiceSpark does not offer W-2 payroll tax filing.",
                ),
                LLMComplaintEvaluation(
                    idx=3,
                    is_genuine_pain_or_seeking_alternative=False,
                    competitor_mentioned="None",
                    pain_category="self_promo",
                    pain_summary="Developer announcing their own open-source release.",
                    can_client_resolve=False,
                    matched_capability_indices=[],
                    solvability_score=0,
                    urgency_score=0,
                    rejection_or_gap_reason="Self-promotion announcement, not a complaint.",
                ),
            ]
        ),
        "gemini:gemini-3.5-flash-lite",
    )

    result = await score_complaints_batched(
        complaints=[c_hot, c_warm, c_gap, c_noise],
        profile=_sample_profile(),
        actor_input=_sample_input(),
        llm_router=mock_router,
        batch_size=15,
        now=NOW,
    )

    assert result.llm_batches_sent == 1  # All 4 scored in ONE batched LLM call
    assert len(result.hot_leads) == 1
    assert result.hot_leads[0].winability_score == 95  # 48 + 27 + 20
    assert result.hot_leads[0].matched_capabilities == [
        "Flat $15/mo pricing with unlimited clients",
        "Automated polite overdue payment reminders",
        "1-click CSV import from FreshBooks/QuickBooks",
    ]
    assert len(result.warm_signals) == 1
    assert result.warm_signals[0].winability_score == 51  # 30 + 12 + 9
    assert len(result.unresolved_market_gaps) == 1
    assert result.unresolved_market_gaps[0].winability_score == 0
    assert len(result.discarded) == 1


@pytest.mark.asyncio
async def test_stage4_failure_case_hallucinated_capability_index_rejected_by_guardrail():
    """Failure Case: LLM says can_client_resolve=True with invalid/empty ledger index -> Python rejects."""
    c1 = _make_complaint(
        "halluc1",
        "QuickBooks broke my payroll filings and I need a replacement.",
        age_hours=3,
    )
    mock_router = AsyncMock(spec=LLMRouter)
    mock_router.generate_structured.return_value = (
        LLMBatchEvaluationResponse(
            evaluations=[
                LLMComplaintEvaluation(
                    idx=0,
                    is_genuine_pain_or_seeking_alternative=True,
                    competitor_mentioned="QuickBooks Online",
                    pain_category="payroll_tax",
                    pain_summary="Needs payroll filing software.",
                    can_client_resolve=True,  # Hallucinated True!
                    matched_capability_indices=[99],  # Out-of-bounds invalid index!
                    solvability_score=45,
                    urgency_score=28,
                )
            ]
        ),
        "groq:openai/gpt-oss-20b",
    )

    result = await score_complaints_batched(
        complaints=[c1],
        profile=_sample_profile(),
        actor_input=_sample_input(),
        llm_router=mock_router,
        now=NOW,
    )

    assert len(result.hot_leads) == 0
    assert len(result.unresolved_market_gaps) == 1
    assert result.unresolved_market_gaps[0].grounding_override_applied is True
    assert result.unresolved_market_gaps[0].winability_score == 0


@pytest.mark.asyncio
async def test_stage4_failure_case_batch_split_retry_recovers_on_truncation():
    """Failure Case: If a 6-item batch throws an error, Stage 4 splits into 3+3 and recovers cleanly."""
    complaints = [
        _make_complaint(
            f"c{i}",
            f"FreshBooks pricing jumped again on client #{i}, looking for flat-rate alternative.",
            age_hours=6,
        )
        for i in range(6)
    ]

    call_count = 0

    async def fake_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Simulated truncated JSON on large batch")
        # Succeed for each split half (3 items each)
        return (
            LLMBatchEvaluationResponse(
                evaluations=[
                    LLMComplaintEvaluation(
                        idx=j,
                        is_genuine_pain_or_seeking_alternative=True,
                        competitor_mentioned="FreshBooks",
                        pain_category="pricing",
                        pain_summary="FreshBooks pricing jump.",
                        can_client_resolve=True,
                        matched_capability_indices=[1],
                        solvability_score=45,
                        urgency_score=25,
                    )
                    for j in range(3)
                ]
            ),
            "gemini:gemini-3.5-flash-lite",
        )

    mock_router = AsyncMock(spec=LLMRouter)
    mock_router.generate_structured.side_effect = fake_generate

    result = await score_complaints_batched(
        complaints=complaints,
        profile=_sample_profile(),
        actor_input=_sample_input(),
        llm_router=mock_router,
        batch_size=6,
        now=NOW,
    )

    assert call_count == 3  # 1 failed full batch + 2 recovered halves
    assert result.llm_batches_failed == 0
    assert len(result.hot_leads) == 6
    assert compute_recency_score(None, now=NOW) == 10