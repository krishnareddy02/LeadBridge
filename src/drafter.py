# """Stage 5A: Conditional Batched Dual Outreach Drafter & Platform Compliance Tagger.

# Runs ONLY on qualified Hot Leads (score >= minWinabilityScore). Batches up to 5
# Hot Leads per LLM call, enforces strict grounding against matched_capabilities,
# attaches deterministic platform anti-ban rules, and includes a safe fallback
# template if all LLM providers are unavailable.
# """

# from __future__ import annotations

# import logging
# from urllib.parse import urlparse

# from src.llm_router import LLMRouter
# from src.schemas import (
#     ActorInput,
#     DatasetLeadRecord,
#     LLMBatchDraftResponse,
#     LLMLeadDraft,
#     ProductProfile,
#     ScoredComplaint,
#     SourceType,
#     Stage4Result,
#     Stage5Result,
#     ToneType,
# )

# logger = logging.getLogger(__name__)

# DRAFT_BATCH_SIZE = 5

# TONE_GUIDANCE: dict[ToneType, str] = {
#     "helpful_technical": (
#         "Peer engineer tone: concise, concrete technical details, zero marketing buzzwords, "
#         "candidly acknowledge tradeoffs."
#     ),
#     "founder_candid": (
#         "Authentic indie founder tone: warm, empathetic, practical tip first, transparent "
#         "('Full disclosure: I'm building X'), zero hard-sell pressure."
#     ),
#     "executive_concise": (
#         "Crisp B2B executive tone: direct value mapping, clear ROI/operational impact, "
#         "respectful and brief."
#     ),
# }

# STAGE5_SYSTEM_PROMPT = """You are an ethical B2B founder outreach specialist trained in community-safe engagement (Reddit, Hacker News, Bluesky, G2).

# For EVERY Hot Lead in the batch (indexed 0..M), generate:
# 1. draft_public_reply (Draft A — Value-First Public Reply):
#    - Start with a genuinely helpful, non-promotional tip or workaround for their current tool/problem (so your comment adds value even if they never switch).
#    - Transition smoothly and transparently disclose affiliation (e.g., "Full disclosure: I work on [ProductName]" or "Disclosure: I'm the founder of [ProductName]").
#    - Mention ONLY factual capabilities explicitly present in that lead's Matched Verified Capabilities. NEVER invent pricing, free tiers, uptime SLAs, or integrations not explicitly listed.
#    - Do NOT include raw promotional URLs or pushy calls-to-action (prevents Reddit/HN bans).
#    - Keep it crisp (3-5 sentences; if Bluesky, keep under 280 characters).

# 2. draft_direct_pitch (Draft B — Direct / DM Follow-Up):
#    - Written assuming the user replied asking for details or opted into a DM.
#    - Reference their exact frustration, map it cleanly to the Matched Verified Capabilities, and offer a low-friction next step (e.g., quick setup walkthrough or migration check).
#    - Keep it concise (3-4 sentences).

# 3. claim_evidence_summary:
#    - One brief line quoting the exact Verified Capabilities relied on so the founder can verify 100% factual accuracy before sending.
# """


# def get_platform_safety_rule(source: SourceType, sub_source: str, post_url: str) -> str:
#     """Deterministic anti-ban and privacy guidance tailored to the exact destination platform."""
#     host = (urlparse(post_url).hostname or "").lower()
#     sub_low = (sub_source or "").lower()

#     if source == "reddit" or "reddit.com" in host or sub_low.startswith("r/"):
#         sub_label = sub_source if sub_low.startswith("r/") else "Reddit"
#         return (
#             f"Reddit ({sub_label}): Strict anti-self-promo community. Lead with helpful advice, "
#             "disclose affiliation clearly, and omit raw promo links unless OP asks for them."
#         )
#     if source == "hacker_news" or "news.ycombinator.com" in host:
#         return (
#             "Hacker News: High sensitivity to marketing speak. Stay technical and candid, "
#             "explicitly disclose your affiliation, and acknowledge real tradeoffs."
#         )
#     if source == "bluesky" or "bsky.app" in host:
#         return (
#             "Bluesky: Keep replies brief (<300 chars), conversational, and genuinely helpful. "
#             "Never send automated or templated pitches."
#         )
#     if source == "g2" or any(d in host for d in ("g2.com", "trustpilot.com", "capterra.com")):
#         return (
#             "Review Site (G2/Trustpilot/Capterra): Do not scrape or cold-DM private contact info "
#             "(GDPR/privacy compliance). Use for public positioning or compliant vendor replies."
#         )
#     return (
#         "Public Forum: Human-in-the-loop review required. Lead with value, disclose affiliation "
#         "transparently, and follow local community self-promotion guidelines."
#     )


# def _build_draft_batch_prompt(
#     profile: ProductProfile,
#     leads: list[ScoredComplaint],
#     tone: ToneType,
# ) -> str:
#     tone_desc = TONE_GUIDANCE.get(tone, TONE_GUIDANCE["founder_candid"])
#     blocks: list[str] = []
#     for idx, lead in enumerate(leads):
#         safety_rule = get_platform_safety_rule(
#             lead.complaint.source, lead.complaint.sub_source, lead.complaint.post_url
#         )
#         matched_str = "\n".join(f"    - {cap}" for cap in lead.matched_capabilities)
#         snippet = lead.complaint.text.replace("\n", " ").strip()[:550]
#         blocks.append(
#             f"[HOT LEAD idx={idx}]\n"
#             f"Platform: {lead.complaint.source} ({lead.complaint.sub_source}) | Author: {lead.complaint.author_handle}\n"
#             f"Safety Rule: {safety_rule}\n"
#             f"Competitor Mentioned: {lead.competitor_mentioned}\n"
#             f"Pain Summary: {lead.pain_summary}\n"
#             f"Original Post Excerpt: {snippet}\n"
#             f"ALLOWED Matched Verified Capabilities (ONLY claim facts from this list):\n{matched_str}"
#         )

#     return (
#         f"FOUNDER PRODUCT: {profile.product_name} ({profile.product_category})\n"
#         f"Website: {profile.product_url}\n"
#         f"Requested Tone ({tone}): {tone_desc}\n\n"
#         f"HOT LEADS TO DRAFT ({len(leads)} items, return drafts for EVERY idx 0..{len(leads) - 1}):\n\n"
#         + "\n\n".join(blocks)
#     )


# def _deterministic_fallback_draft(
#     profile: ProductProfile,
#     lead: ScoredComplaint,
# ) -> LLMLeadDraft:
#     """Safe zero-hallucination fallback if every LLM provider fails during drafting."""
#     caps_joined = "; ".join(lead.matched_capabilities[:2]) or profile.product_category
#     draft_a = (
#         f"Totally hear the frustration around {lead.competitor_mentioned} ({lead.pain_category.replace('_', ' ')}). "
#         f"Before migrating, double-check whether adjusting your current plan configuration resolves the immediate bottleneck. "
#         f"If you do evaluate alternatives, full disclosure: I'm with {profile.product_name}, which directly supports {caps_joined}."
#     )
#     draft_b = (
#         f"Hi {lead.complaint.author_handle}, saw your post regarding {lead.competitor_mentioned}. "
#         f"{profile.product_name} addresses this through {caps_joined}. "
#         f"Happy to share a quick walkthrough if you want to compare options."
#     )
#     return LLMLeadDraft(
#         idx=0,
#         draft_public_reply=draft_a,
#         draft_direct_pitch=draft_b,
#         claim_evidence_summary=f"Grounded in Verified Capabilities: {caps_joined}",
#     )


# def _to_dataset_record(
#     scored: ScoredComplaint,
#     record_type: str,
#     draft: LLMLeadDraft | None = None,
#     draft_provider: str = "",
# ) -> DatasetLeadRecord:
#     safety = get_platform_safety_rule(
#         scored.complaint.source,
#         scored.complaint.sub_source,
#         scored.complaint.post_url,
#     )
#     pub_iso = (
#         scored.complaint.published_at.isoformat() if scored.complaint.published_at else None
#     )
#     return DatasetLeadRecord(
#         recordType=record_type,  # type: ignore[arg-type]
#         winabilityScore=scored.winability_score,
#         solvabilityScore=scored.solvability_score,
#         urgencyScore=scored.urgency_score,
#         recencyScore=scored.recency_score,
#         source=scored.complaint.source,
#         subSource=scored.complaint.sub_source,
#         authorHandle=scored.complaint.author_handle,
#         competitorMentioned=scored.competitor_mentioned,
#         painCategory=scored.pain_category,
#         painSummary=scored.pain_summary,
#         matchedCapabilities=scored.matched_capabilities,
#         claimEvidenceSummary=draft.claim_evidence_summary if draft else "",
#         platformSafetyRule=safety,
#         draftPublicReply=draft.draft_public_reply if draft else "",
#         draftDirectPitch=draft.draft_direct_pitch if draft else "",
#         gapOrRejectionReason=scored.rejection_or_gap_reason,
#         postTitle=scored.complaint.title,
#         postUrl=scored.complaint.post_url,
#         publishedAt=pub_iso,
#         draftProvider=draft_provider,
#     )


# async def draft_hot_leads(
#     stage4: Stage4Result,
#     profile: ProductProfile,
#     actor_input: ActorInput,
#     llm_router: LLMRouter | None = None,
#     batch_size: int = DRAFT_BATCH_SIZE,
# ) -> Stage5Result:
#     """Generate dual ban-safe drafts for Hot Leads and assemble all Dataset records."""
#     router = llm_router or LLMRouter(gemini_api_key=actor_input.llmApiKey)
#     hot_records: list[DatasetLeadRecord] = []
#     providers_seen: list[str] = []
#     batches_sent = 0

#     hot_leads = stage4.hot_leads
#     eff_batch = max(1, min(8, batch_size))

#     for start in range(0, len(hot_leads), eff_batch):
#         chunk = hot_leads[start : start + eff_batch]
#         prompt = _build_draft_batch_prompt(profile, chunk, actor_input.tone)
#         batches_sent += 1

#         draft_map: dict[int, LLMLeadDraft] = {}
#         provider_label = ""
#         try:
#             parsed, provider_label = await router.generate_structured(
#                 system_prompt=STAGE5_SYSTEM_PROMPT,
#                 user_prompt=prompt,
#                 response_model=LLMBatchDraftResponse,
#                 temperature=0.35,
#             )
#             if provider_label not in providers_seen:
#                 providers_seen.append(provider_label)
#             for d in parsed.drafts:
#                 if 0 <= d.idx < len(chunk):
#                     draft_map[d.idx] = d
#         except Exception as exc:
#             logger.warning(
#                 "Stage 5 drafting batch failed (%s: %s); applying deterministic fallback templates.",
#                 type(exc).__name__,
#                 exc,
#             )
#             provider_label = "deterministic_fallback"
#             if provider_label not in providers_seen:
#                 providers_seen.append(provider_label)

#         for local_idx, lead in enumerate(chunk):
#             draft = draft_map.get(local_idx)
#             used_provider = provider_label
#             if draft is None:
#                 draft = _deterministic_fallback_draft(profile, lead)
#                 used_provider = "deterministic_fallback"
#             hot_records.append(
#                 _to_dataset_record(
#                     scored=lead,
#                     record_type="hot_lead",
#                     draft=draft,
#                     draft_provider=used_provider,
#                 )
#             )

#     warm_records = [
#         _to_dataset_record(scored=item, record_type="warm_signal")
#         for item in stage4.warm_signals
#     ]
#     gap_records = [
#         _to_dataset_record(scored=item, record_type="unresolved_market_gap")
#         for item in stage4.unresolved_market_gaps
#     ]

#     logger.info(
#         "Stage 5A drafting complete: hot_drafted=%d warm=%d market_gaps=%d batches=%d providers=%s",
#         len(hot_records),
#         len(warm_records),
#         len(gap_records),
#         batches_sent,
#         providers_seen,
#     )

#     return Stage5Result(
#         hot_lead_records=hot_records,
#         warm_signal_records=warm_records,
#         market_gap_records=gap_records,
#         llm_batches_sent=batches_sent,
#         providers_used=providers_seen,
#     )

"""Stage 5A: Conditional Batched Dual Outreach Drafter & Platform Compliance Tagger.

Runs on qualified Hot Leads (score >= minWinabilityScore) with an Adaptive Top-3
Fallback when 0 items cross the threshold (auto-drafts top warm opportunities >= 55
so users never receive empty draft columns). Batches up to 5 leads per LLM call,
enforces strict grounding against matched_capabilities, attaches deterministic
platform anti-ban rules, and formats flat CRM/CSV-ready records.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from src.llm_router import LLMRouter
from src.schemas import (
    ActorInput,
    DatasetLeadRecord,
    LLMBatchDraftResponse,
    LLMLeadDraft,
    ProductProfile,
    ScoredComplaint,
    SourceType,
    Stage4Result,
    Stage5Result,
    ToneType,
)

logger = logging.getLogger(__name__)

DRAFT_BATCH_SIZE = 5
AUTO_DRAFT_FALLBACK_MIN_SCORE = 55
AUTO_DRAFT_FALLBACK_LIMIT = 3

TONE_GUIDANCE: dict[ToneType, str] = {
    "helpful_technical": (
        "Peer engineer tone: concise, concrete technical details, zero marketing buzzwords, "
        "candidly acknowledge tradeoffs."
    ),
    "founder_candid": (
        "Authentic indie founder tone: warm, empathetic, practical tip first, transparent "
        "('Full disclosure: I'm building X'), zero hard-sell pressure."
    ),
    "executive_concise": (
        "Crisp B2B executive tone: direct value mapping, clear ROI/operational impact, "
        "respectful and brief."
    ),
}

STAGE5_SYSTEM_PROMPT = """You are an ethical B2B founder outreach specialist trained in community-safe engagement (Reddit, Hacker News, Bluesky, G2).

For EVERY Hot Lead in the batch (indexed 0..M), generate:
1. draft_public_reply (Draft A — Value-First Public Reply):
   - Start with a genuinely helpful, non-promotional tip or workaround for their current tool/problem (so your comment adds value even if they never switch).
   - Transition smoothly and transparently disclose affiliation (e.g., "Full disclosure: I work on [ProductName]" or "Disclosure: I'm the founder of [ProductName]").
   - Mention ONLY factual capabilities explicitly present in that lead's Matched Verified Capabilities. NEVER invent pricing, free tiers, uptime SLAs, or integrations not explicitly listed.
   - Do NOT include raw promotional URLs or pushy calls-to-action (prevents Reddit/HN bans).
   - Keep it crisp (3-5 sentences; if Bluesky, keep under 280 characters).

2. draft_direct_pitch (Draft B — Direct / DM Follow-Up):
   - Written assuming the user replied asking for details or opted into a DM.
   - Reference their exact frustration, map it cleanly to the Matched Verified Capabilities, and offer a low-friction next step (e.g., quick setup walkthrough or migration check).
   - Keep it concise (3-4 sentences).

3. claim_evidence_summary:
   - One brief line quoting the exact Verified Capabilities relied on so the founder can verify 100% factual accuracy before sending.
"""


def get_platform_safety_rule(source: SourceType, sub_source: str, post_url: str) -> str:
    """Deterministic anti-ban and privacy guidance tailored to the exact destination platform."""
    host = (urlparse(post_url).hostname or "").lower()
    sub_clean = (sub_source or "").strip()

    if source == "reddit" or "reddit.com" in host or sub_clean.lower().startswith("r/"):
        sub_label = sub_clean if sub_clean.lower().startswith("r/") else ""
        if not sub_label:
            m = re.search(r"reddit\.com/(r/[A-Za-z0-9_]+)", post_url or "", re.IGNORECASE)
            sub_label = m.group(1) if m else "Community"
        return (
            f"Reddit ({sub_label}): Strict anti-self-promo rules. Lead with helpful advice, "
            "disclose affiliation clearly, and omit raw promo links unless OP asks."
        )
    if source == "hacker_news" or "news.ycombinator.com" in host:
        return (
            "Hacker News: High sensitivity to marketing speak. Stay technical and candid, "
            "explicitly disclose your affiliation, and acknowledge real tradeoffs."
        )
    if source == "bluesky" or "bsky.app" in host:
        return (
            "Bluesky: Keep replies brief (<300 chars), conversational, and genuinely helpful. "
            "Never send automated or templated pitches."
        )
    if source == "g2" or any(d in host for d in ("g2.com", "trustpilot.com", "capterra.com")):
        return (
            "Review Site (G2/Trustpilot/Capterra): Do not scrape or cold-DM private contact info "
            "(GDPR/privacy compliance). Use for public positioning or compliant vendor replies."
        )
    return (
        "Public Forum: Human-in-the-loop review required. Lead with value, disclose affiliation "
        "transparently, and follow local community self-promotion guidelines."
    )


def _recommended_action_for(
    record_type: str,
    has_drafts: bool,
    source: SourceType,
    post_url: str,
) -> str:
    host = (urlparse(post_url).hostname or "").lower()
    if any(d in host for d in ("g2.com", "trustpilot.com", "capterra.com")) or source == "g2":
        return "Use for comparison page / sales battlecard (no cold DM)"
    if has_drafts:
        return "Review & post Draft A publicly; use Draft B only if OP replies"
    if record_type == "unresolved_market_gap":
        return "Log in product roadmap / feature backlog"
    return "Monitor thread or repurpose into SEO / landing-page copy"


def _build_draft_batch_prompt(
    profile: ProductProfile,
    leads: list[ScoredComplaint],
    tone: ToneType,
) -> str:
    tone_desc = TONE_GUIDANCE.get(tone, TONE_GUIDANCE["founder_candid"])
    blocks: list[str] = []
    for idx, lead in enumerate(leads):
        safety_rule = get_platform_safety_rule(
            lead.complaint.source, lead.complaint.sub_source, lead.complaint.post_url
        )
        matched_str = "\n".join(f"    - {cap}" for cap in lead.matched_capabilities)
        snippet = lead.complaint.text.replace("\n", " ").strip()[:550]
        blocks.append(
            f"[HOT LEAD idx={idx}]\n"
            f"Platform: {lead.complaint.source} ({lead.complaint.sub_source}) | Author: {lead.complaint.author_handle}\n"
            f"Safety Rule: {safety_rule}\n"
            f"Competitor Mentioned: {lead.competitor_mentioned}\n"
            f"Pain Summary: {lead.pain_summary}\n"
            f"Original Post Excerpt: {snippet}\n"
            f"ALLOWED Matched Verified Capabilities (ONLY claim facts from this list):\n{matched_str}"
        )

    return (
        f"FOUNDER PRODUCT: {profile.product_name} ({profile.product_category})\n"
        f"Website: {profile.product_url}\n"
        f"Requested Tone ({tone}): {tone_desc}\n\n"
        f"HOT LEADS TO DRAFT ({len(leads)} items, return drafts for EVERY idx 0..{len(leads) - 1}):\n\n"
        + "\n\n".join(blocks)
    )


def _deterministic_fallback_draft(
    profile: ProductProfile,
    lead: ScoredComplaint,
) -> LLMLeadDraft:
    """Safe zero-hallucination fallback if every LLM provider fails during drafting."""
    caps_joined = "; ".join(lead.matched_capabilities[:2]) or profile.product_category
    draft_a = (
        f"Totally hear the frustration around {lead.competitor_mentioned} ({lead.pain_category.replace('_', ' ')}). "
        f"Before migrating, double-check whether adjusting your current plan configuration resolves the immediate bottleneck. "
        f"If you do evaluate alternatives, full disclosure: I'm with {profile.product_name}, which directly supports {caps_joined}."
    )
    draft_b = (
        f"Hi {lead.complaint.author_handle}, saw your post regarding {lead.competitor_mentioned}. "
        f"{profile.product_name} addresses this through {caps_joined}. "
        f"Happy to share a quick walkthrough if you want to compare options."
    )
    return LLMLeadDraft(
        idx=0,
        draft_public_reply=draft_a,
        draft_direct_pitch=draft_b,
        claim_evidence_summary=f"Grounded in Verified Capabilities: {caps_joined}",
    )


def _to_dataset_record(
    scored: ScoredComplaint,
    record_type: str,
    draft: LLMLeadDraft | None = None,
    draft_provider: str = "",
    priority_label: str | None = None,
    report_url: str = "",
) -> DatasetLeadRecord:
    safety = get_platform_safety_rule(
        scored.complaint.source,
        scored.complaint.sub_source,
        scored.complaint.post_url,
    )
    pub_iso = (
        scored.complaint.published_at.isoformat() if scored.complaint.published_at else None
    )
    pub_date = (
        scored.complaint.published_at.strftime("%Y-%m-%d")
        if scored.complaint.published_at
        else "Within lookback window"
    )
    verified_proof = " | ".join(scored.matched_capabilities) if scored.matched_capabilities else "None"
    default_evidence = (
        f"Grounded in: {'; '.join(scored.matched_capabilities[:3])}"
        if scored.matched_capabilities
        else (scored.rejection_or_gap_reason or "No direct capability match")
    )

    if priority_label:
        tier_label = priority_label
    elif record_type == "hot_lead":
        tier_label = "Hot Lead"
    elif record_type == "warm_signal":
        tier_label = "Warm / SEO Signal"
    else:
        tier_label = "Unresolved Market Gap"

    author = scored.complaint.author_handle
    if author == "public_web":
        author = "Public Web Post"

    has_drafts = draft is not None
    return DatasetLeadRecord(
        priorityTier=tier_label,
        recordType=record_type,  # type: ignore[arg-type]
        winabilityScore=scored.winability_score,
        solvabilityScore=scored.solvability_score,
        urgencyScore=scored.urgency_score,
        recencyScore=scored.recency_score,
        recommendedAction=_recommended_action_for(
            record_type, has_drafts, scored.complaint.source, scored.complaint.post_url
        ),
        outreachStatus="Pending Review" if has_drafts else "Insight Only",
        source=scored.complaint.source,
        subSource=scored.complaint.sub_source,
        authorHandle=author,
        competitorMentioned=scored.competitor_mentioned,
        painCategory=scored.pain_category,
        painSummary=scored.pain_summary,
        verifiedProof=verified_proof,
        matchedCapabilities=scored.matched_capabilities,
        claimEvidenceSummary=(draft.claim_evidence_summary if draft else default_evidence),
        platformSafetyRule=safety,
        draftPublicReply=draft.draft_public_reply if draft else "",
        draftDirectPitch=draft.draft_direct_pitch if draft else "",
        gapOrRejectionReason=scored.rejection_or_gap_reason,
        postTitle=scored.complaint.title,
        postUrl=scored.complaint.post_url,
        publishedDate=pub_date,
        publishedAt=pub_iso,
        reportUrl=report_url,
        draftProvider=draft_provider,
    )


async def draft_hot_leads(
    stage4: Stage4Result,
    profile: ProductProfile,
    actor_input: ActorInput,
    llm_router: LLMRouter | None = None,
    batch_size: int = DRAFT_BATCH_SIZE,
    report_url: str = "",
) -> Stage5Result:
    """Generate dual ban-safe drafts for Hot Leads (or Top-3 fallback) and assemble Dataset records."""
    router = llm_router or LLMRouter(gemini_api_key=actor_input.llmApiKey)
    hot_records: list[DatasetLeadRecord] = []
    providers_seen: list[str] = []
    batches_sent = 0

    leads_to_draft: list[tuple[ScoredComplaint, str, str]] = [
        (lead, "hot_lead", "Hot Lead") for lead in stage4.hot_leads
    ]
    remaining_warm = list(stage4.warm_signals)

    # Adaptive Fallback: if 0 leads crossed minWinabilityScore, draft up to top 3
    # high-scoring warm opportunities (>= 55) so users never receive blank draft columns.
    if not leads_to_draft and remaining_warm:
        promoted = [
            w for w in remaining_warm if w.winability_score >= AUTO_DRAFT_FALLBACK_MIN_SCORE
        ][:AUTO_DRAFT_FALLBACK_LIMIT]
        promoted_ids = {w.complaint.id for w in promoted}
        leads_to_draft = [(w, "hot_lead", "Top Opportunity (Auto-Drafted)") for w in promoted]
        remaining_warm = [w for w in remaining_warm if w.complaint.id not in promoted_ids]
        if promoted:
            logger.info(
                "0 items crossed minWinabilityScore=%d; auto-drafting top %d warm opportunities (>=%d).",
                actor_input.minWinabilityScore,
                len(promoted),
                AUTO_DRAFT_FALLBACK_MIN_SCORE,
            )

    eff_batch = max(1, min(8, batch_size))

    for start in range(0, len(leads_to_draft), eff_batch):
        chunk_tuples = leads_to_draft[start : start + eff_batch]
        chunk_leads = [t[0] for t in chunk_tuples]
        prompt = _build_draft_batch_prompt(profile, chunk_leads, actor_input.tone)
        batches_sent += 1

        draft_map: dict[int, LLMLeadDraft] = {}
        provider_label = ""
        try:
            parsed, provider_label = await router.generate_structured(
                system_prompt=STAGE5_SYSTEM_PROMPT,
                user_prompt=prompt,
                response_model=LLMBatchDraftResponse,
                temperature=0.35,
            )
            if provider_label not in providers_seen:
                providers_seen.append(provider_label)
            for d in parsed.drafts:
                if 0 <= d.idx < len(chunk_leads):
                    draft_map[d.idx] = d
        except Exception as exc:
            logger.warning(
                "Stage 5 drafting batch failed (%s: %s); applying deterministic fallback templates.",
                type(exc).__name__,
                exc,
            )
            provider_label = "deterministic_fallback"
            if provider_label not in providers_seen:
                providers_seen.append(provider_label)

        for local_idx, (lead, rec_type, prio_label) in enumerate(chunk_tuples):
            draft = draft_map.get(local_idx)
            used_provider = provider_label
            if draft is None:
                draft = _deterministic_fallback_draft(profile, lead)
                used_provider = "deterministic_fallback"
            hot_records.append(
                _to_dataset_record(
                    scored=lead,
                    record_type=rec_type,
                    draft=draft,
                    draft_provider=used_provider,
                    priority_label=prio_label,
                    report_url=report_url,
                )
            )

    warm_records = [
        _to_dataset_record(scored=item, record_type="warm_signal", report_url=report_url)
        for item in remaining_warm
    ]
    gap_records = [
        _to_dataset_record(
            scored=item, record_type="unresolved_market_gap", report_url=report_url
        )
        for item in stage4.unresolved_market_gaps
    ]

    logger.info(
        "Stage 5A drafting complete: drafted=%d warm=%d market_gaps=%d batches=%d providers=%s",
        len(hot_records),
        len(warm_records),
        len(gap_records),
        batches_sent,
        providers_seen,
    )

    return Stage5Result(
        hot_lead_records=hot_records,
        warm_signal_records=warm_records,
        market_gap_records=gap_records,
        llm_batches_sent=batches_sent,
        providers_used=providers_seen,
    )