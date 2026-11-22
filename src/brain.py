"""Stage 4: Batched LLM Solvability Hard-Gate & Winability Scorer.

Evaluates 15 clean complaints per LLM request (saving ~93% of free-tier RPD quota),
enforces strict Verified Capabilities Ledger grounding in Python so the LLM can
never hallucinate product advantages, computes deterministic recency + total
Winability scores (0-100), and routes every complaint into:
  - hot_lead              (score >= minWinabilityScore, ready for Stage 5 drafting)
  - warm_signal           (40 <= score < minWinabilityScore, SEO/market insight)
  - unresolved_market_gap (genuine competitor pain we cannot solve yet -> roadmap signal)
  - discarded             (off-topic noise, self-promo, praise, or non-market issue)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.llm_router import LLMRouter
from src.schemas import (
    ActorInput,
    LLMBatchEvaluationResponse,
    LLMComplaintEvaluation,
    ProductProfile,
    RawComplaint,
    ScoredComplaint,
    Stage4Result,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 15
MAX_ITEM_CHARS_FOR_LLM = 550

STAGE4_SYSTEM_PROMPT = """You are a strict, skeptical B2B revenue-qualification analyst.
You will receive:
1. A Founder's Product Profile containing a NUMBERED "Verified Capabilities Ledger" (1..N).
2. A batch of public social/forum posts indexed 0..M.

For EVERY post in the batch (do NOT skip any index), evaluate:

1. is_genuine_pain_or_seeking_alternative:
   - True ONLY if the author is complaining about a competitor/workflow in the founder's product category, hitting a technical/pricing limit, or actively asking for recommendations/alternatives.
   - False for: general news, self-promotion/product launches ("Show HN", "I built X", "v3.6.0 shipped"), pure praise/apologies, casual metaphors, or complaints about an UNRELATED product (e.g., complaining about a retail store website that merely happens to run on a cloud provider).

2. HARD SOLVABILITY GATE (can_client_resolve):
   - Set can_client_resolve = True ONLY if BOTH:
     (a) is_genuine_pain_or_seeking_alternative is True, AND
     (b) One or more EXPLICIT items in the numbered Verified Capabilities Ledger factually fix the user's specific stated pain or requirement.
   - NEVER assume or invent features not explicitly listed in the Verified Capabilities Ledger.
   - Example REJECT: If a user wants a free-forever tier with zero spin-down and the ledger only offers "$5/mo paid tier", check whether the $5/mo tier is a valid low-cost alternative (moderate solvability) vs a hard mismatch (e.g., user needs W-2 payroll tax filing and product is simple invoicing -> MUST be can_client_resolve = False!).
   - If can_client_resolve is False: set matched_capability_indices = [], solvability_score = 0, urgency_score = 0, and write a concise rejection_or_gap_reason.

3. matched_capability_indices:
   - List the 1-based integers (e.g. [1, 7]) from the Verified Capabilities Ledger that directly address the user's pain.

4. Scoring (ONLY when can_client_resolve is True):
   - solvability_score (0-50):
     * 40-50: Direct, indisputable match (ledger explicitly fixes the exact pain).
     * 25-39: Strong partial match (fixes core problem, minor tradeoff like $5/mo vs free).
     * 10-24: Weak/indirect match.
   - urgency_score (0-30):
     * 24-30: Explicitly asking for alternatives, actively migrating, or blocked right now.
     * 14-23: Strong frustration, pricing shock, or evaluating competitors.
     * 5-13: Mild/exploratory comparison or casual mention of pain.
"""


def compute_recency_score(
    published_at: datetime | None,
    now: datetime | None = None,
) -> int:
    """Deterministic 0-20 recency score based on post age."""
    if published_at is None:
        # Google SERP results bounded by after:YYYY-MM-DD often omit exact timestamp -> neutral mid score
        return 10

    ref_now = now or datetime.now(timezone.utc)
    pub = (
        published_at
        if published_at.tzinfo is not None
        else published_at.replace(tzinfo=timezone.utc)
    )
    age_hours = max(0.0, (ref_now - pub).total_seconds() / 3600.0)

    if age_hours <= 24:
        return 20
    if age_hours <= 72:
        return 17
    if age_hours <= 168:
        return 13
    if age_hours <= 336:
        return 9
    return 4


def _build_batch_user_prompt(
    profile: ProductProfile,
    batch: list[RawComplaint],
) -> str:
    ledger_lines = "\n".join(
        f"  {i}. {cap}" for i, cap in enumerate(profile.verified_capabilities, start=1)
    )
    competitors_str = ", ".join(profile.competitors)

    post_blocks: list[str] = []
    for idx, item in enumerate(batch):
        compact_text = item.text.replace("\n", " ").strip()[:MAX_ITEM_CHARS_FOR_LLM]
        title_line = f"Title: {item.title[:160]}\n" if item.title else ""
        post_blocks.append(
            f"[POST idx={idx}]\n"
            f"Source: {item.source} ({item.sub_source}) | Author: {item.author_handle}\n"
            f"{title_line}"
            f"Text: {compact_text}"
        )

    posts_section = "\n\n".join(post_blocks)
    return (
        f"FOUNDER PRODUCT: {profile.product_name} ({profile.product_category})\n"
        f"Target Audience: {profile.target_audience}\n"
        f"Direct Competitors: {competitors_str}\n\n"
        f"VERIFIED CAPABILITIES LEDGER (ONLY cite these 1-based numbers):\n"
        f"{ledger_lines}\n\n"
        f"POSTS TO EVALUATE ({len(batch)} items, return evaluation for EVERY idx 0..{len(batch) - 1}):\n\n"
        f"{posts_section}"
    )


def _apply_grounding_and_build_scored(
    item: RawComplaint,
    ev: LLMComplaintEvaluation,
    profile: ProductProfile,
    min_winability_score: int,
    now: datetime,
) -> ScoredComplaint:
    """Enforce hard grounding invariants and deterministic score arithmetic in Python."""
    max_cap_idx = len(profile.verified_capabilities)
    valid_indices: list[int] = []
    seen: set[int] = set()
    for raw_idx in ev.matched_capability_indices:
        if 1 <= raw_idx <= max_cap_idx and raw_idx not in seen:
            seen.add(raw_idx)
            valid_indices.append(raw_idx)

    matched_caps = [profile.verified_capabilities[i - 1] for i in valid_indices]

    can_resolve = bool(ev.can_client_resolve)
    is_pain = bool(ev.is_genuine_pain_or_seeking_alternative)
    grounding_override = False
    gap_reason = (ev.rejection_or_gap_reason or "").strip()

    # Guardrail 1: Non-pain posts cannot be marked resolvable leads
    if can_resolve and not is_pain:
        can_resolve = False
        grounding_override = True
        gap_reason = gap_reason or "Not a genuine complaint or alternative-seeking post"

    # Guardrail 2: Strict Grounding Gate — must cite at least 1 valid Verified Capability!
    if can_resolve and not matched_caps:
        can_resolve = False
        grounding_override = True
        gap_reason = (
            gap_reason
            or "Rejected by strict grounding gate: no Verified Capability Ledger match cited"
        )

    if not can_resolve:
        # Distinguish genuine unresolved competitor pain (roadmap insight) from irrelevant noise
        is_market_gap = (
            is_pain
            and ev.pain_category.lower() not in {"off_topic", "none", "spam", "self_promo"}
        )
        return ScoredComplaint(
            complaint=item,
            lead_tier="unresolved_market_gap" if is_market_gap else "discarded",
            is_genuine_pain=is_pain,
            can_client_resolve=False,
            competitor_mentioned=ev.competitor_mentioned.strip() or "None",
            pain_category=ev.pain_category.strip() or "off_topic",
            pain_summary=ev.pain_summary.strip(),
            matched_capabilities=[],
            solvability_score=0,
            urgency_score=0,
            recency_score=0,
            winability_score=0,
            rejection_or_gap_reason=gap_reason or "Does not match verified capabilities",
            grounding_override_applied=grounding_override,
        )

    # Deterministic Winability Score = Solvability (0-50) + Urgency (0-30) + Recency (0-20)
    solvability = max(0, min(50, int(ev.solvability_score)))
    urgency = max(0, min(30, int(ev.urgency_score)))
    recency = compute_recency_score(item.published_at, now=now)
    total_score = min(100, solvability + urgency + recency)

    if total_score >= min_winability_score:
        tier = "hot_lead"
    elif total_score >= 40:
        tier = "warm_signal"
    else:
        tier = "low_priority"

    return ScoredComplaint(
        complaint=item,
        lead_tier=tier,
        is_genuine_pain=True,
        can_client_resolve=True,
        competitor_mentioned=ev.competitor_mentioned.strip() or "Category-Level",
        pain_category=ev.pain_category.strip() or "general_pain",
        pain_summary=ev.pain_summary.strip(),
        matched_capabilities=matched_caps,
        solvability_score=solvability,
        urgency_score=urgency,
        recency_score=recency,
        winability_score=total_score,
        rejection_or_gap_reason="",
        grounding_override_applied=False,
    )


async def _evaluate_batch_with_split_retry(
    batch: list[RawComplaint],
    profile: ProductProfile,
    router: LLMRouter,
) -> tuple[dict[int, LLMComplaintEvaluation], list[str], int, int]:
    """Evaluate a batch; if it fails and size > 4, split in half and retry cleanly.

    Returns: (eval_by_local_idx, providers_used, batches_sent, batches_failed)
    """
    if not batch:
        return {}, [], 0, 0

    user_prompt = _build_batch_user_prompt(profile, batch)
    try:
        parsed, provider = await router.generate_structured(
            system_prompt=STAGE4_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=LLMBatchEvaluationResponse,
            temperature=0.1,
        )
        mapping: dict[int, LLMComplaintEvaluation] = {}
        for ev in parsed.evaluations:
            if 0 <= ev.idx < len(batch):
                mapping[ev.idx] = ev
        return mapping, [provider], 1, 0
    except Exception as exc:
        logger.warning(
            "Stage 4 batch (size=%d) failed (%s: %s)",
            len(batch),
            type(exc).__name__,
            exc,
        )
        if len(batch) <= 4:
            return {}, [], 1, 1

        # Split-retry in two halves to recover from output-length truncation or transient errors
        mid = len(batch) // 2
        left_map, left_prov, left_sent, left_fail = await _evaluate_batch_with_split_retry(
            batch[:mid], profile, router
        )
        right_map, right_prov, right_sent, right_fail = await _evaluate_batch_with_split_retry(
            batch[mid:], profile, router
        )
        combined = dict(left_map)
        for r_idx, ev in right_map.items():
            combined[mid + r_idx] = ev
        return (
            combined,
            left_prov + right_prov,
            1 + left_sent + right_sent,
            left_fail + right_fail,
        )


async def score_complaints_batched(
    complaints: list[RawComplaint],
    profile: ProductProfile,
    actor_input: ActorInput,
    llm_router: LLMRouter | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    now: datetime | None = None,
) -> Stage4Result:
    """Execute Stage 4 batched solvability gate & winability scoring."""
    if not complaints:
        return Stage4Result()

    ref_now = now or datetime.now(timezone.utc)
    router = llm_router or LLMRouter(gemini_api_key=actor_input.llmApiKey)
    effective_batch_size = max(1, min(20, batch_size))

    hot_leads: list[ScoredComplaint] = []
    warm_signals: list[ScoredComplaint] = []
    market_gaps: list[ScoredComplaint] = []
    discarded: list[ScoredComplaint] = []

    providers_seen: list[str] = []
    total_batches_sent = 0
    total_batches_failed = 0

    for start in range(0, len(complaints), effective_batch_size):
        chunk = complaints[start : start + effective_batch_size]
        eval_map, chunk_providers, sent, failed = await _evaluate_batch_with_split_retry(
            chunk, profile, router
        )
        total_batches_sent += sent
        total_batches_failed += failed
        for p in chunk_providers:
            if p not in providers_seen:
                providers_seen.append(p)

        for local_idx, item in enumerate(chunk):
            ev = eval_map.get(local_idx)
            if ev is None:
                # Defensive fallback if LLM omitted an index or batch failed permanently
                discarded.append(
                    ScoredComplaint(
                        complaint=item,
                        lead_tier="discarded",
                        is_genuine_pain=False,
                        can_client_resolve=False,
                        competitor_mentioned="None",
                        pain_category="evaluation_omitted",
                        pain_summary="Item omitted or batch failed during Stage 4 LLM scoring.",
                        rejection_or_gap_reason="Batch evaluation missing for this item",
                    )
                )
                continue

            scored = _apply_grounding_and_build_scored(
                item=item,
                ev=ev,
                profile=profile,
                min_winability_score=actor_input.minWinabilityScore,
                now=ref_now,
            )

            if scored.lead_tier == "hot_lead":
                hot_leads.append(scored)
            elif scored.lead_tier == "warm_signal":
                warm_signals.append(scored)
            elif scored.lead_tier == "unresolved_market_gap":
                market_gaps.append(scored)
            else:
                discarded.append(scored)

    hot_leads.sort(key=lambda x: x.winability_score, reverse=True)
    warm_signals.sort(key=lambda x: x.winability_score, reverse=True)

    logger.info(
        "Stage 4 complete: evaluated=%d batches=%d hot=%d warm=%d market_gaps=%d discarded=%d providers=%s",
        len(complaints),
        total_batches_sent,
        len(hot_leads),
        len(warm_signals),
        len(market_gaps),
        len(discarded),
        providers_seen,
    )

    return Stage4Result(
        hot_leads=hot_leads,
        warm_signals=warm_signals,
        unresolved_market_gaps=market_gaps,
        discarded=discarded,
        total_evaluated=len(complaints),
        llm_batches_sent=total_batches_sent,
        llm_batches_failed=total_batches_failed,
        providers_used=providers_seen,
    )