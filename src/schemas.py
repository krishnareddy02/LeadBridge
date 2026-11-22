"""Pydantic schemas shared across Stage 1 (Profiler) and Stage 2 (Collectors)."""

from __future__ import annotations

import ipaddress
import re
from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


SourceType = Literal["google_serp", "reddit", "hacker_news", "bluesky", "g2"]
ToneType = Literal["helpful_technical", "founder_candid", "executive_concise"]


class ActorInput(BaseModel):
    """Validated founder input from Apify Input Schema or local JSON."""

    productName: str = Field(..., min_length=1, max_length=120)
    productUrl: str = Field(..., min_length=4, max_length=500)
    productDescription: str = Field(..., min_length=15, max_length=4000)
    competitors: list[str] = Field(default_factory=list)
    sources: list[SourceType] = Field(
        default_factory=lambda: ["google_serp", "reddit", "hacker_news", "bluesky"]
    )
    lookbackDays: int = Field(default=14, ge=1, le=90)
    minWinabilityScore: int = Field(default=70, ge=40, le=95)
    tone: ToneType = Field(default="founder_candid")
    llmApiKey: str | None = Field(default=None)

    @field_validator("productName", "productDescription", mode="before")
    @classmethod
    def strip_text(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Must be a string")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be blank")
        return cleaned

    @field_validator("productUrl", mode="before")
    @classmethod
    def validate_and_normalize_url(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("productUrl cannot be empty")
        raw = v.strip()
        if not re.match(r"^https?://", raw, flags=re.IGNORECASE):
            raw = f"https://{raw}"

        parsed = urlparse(raw)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only http:// and https:// URLs are allowed")
        if not parsed.hostname or "." not in parsed.hostname:
            raise ValueError(f"Invalid public hostname in URL: {raw}")

        host = parsed.hostname.lower()
        # SSRF protection: block localhost and private/reserved IPs
        if host in {"localhost", "0.0.0.0"} or host.endswith(".local"):
            raise ValueError(f"Local/private hostnames are forbidden: {host}")
        try:
            ip = ipaddress.ip_address(host)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            ):
                raise ValueError(f"Private or reserved IP addresses are forbidden: {host}")
        except ValueError as exc:
            # Raised when host is a domain name (not an IP) -> safe to continue;
            # re-raise only if it was our own forbidden IP error.
            if "forbidden" in str(exc):
                raise

        return raw

    @field_validator("competitors", mode="before")
    @classmethod
    def clean_competitors(cls, v: list[str] | None) -> list[str]:
        if not v:
            return []
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in v:
            name = str(item).strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                cleaned.append(name)
        return cleaned[:10]

    @property
    def root_domain(self) -> str:
        host = (urlparse(self.productUrl).hostname or "").lower()
        return host[4:] if host.startswith("www.") else host


class LLMProfileExtraction(BaseModel):
    """Strict JSON schema the LLM must return during Stage 1."""

    product_category: str = Field(
        ...,
        description="Concise 2-6 word product category (e.g., 'Freelancer Invoicing SaaS').",
    )
    target_audience: str = Field(
        ...,
        description="Primary buyer persona (e.g., 'Freelancers and small creative agencies').",
    )
    verified_capabilities: list[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="5-8 concrete, factual capabilities/pricing advantages explicitly stated in the website or description.",
    )
    pain_points_solved: list[str] = Field(
        ...,
        min_length=2,
        max_length=8,
        description="4-6 specific customer frustrations this product resolves.",
    )
    competitors: list[str] = Field(
        ...,
        min_length=2,
        max_length=8,
        description="Top 4-6 direct market competitors (exact brand names).",
    )
    competitor_pain_queries: list[str] = Field(
        ...,
        min_length=3,
        max_length=8,
        description="5 concise, high-intent search queries targeting unhappy competitor users (e.g., 'FreshBooks alternative', 'switching from QuickBooks').",
    )
    category_problem_queries: list[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="3 brand-agnostic category problem queries for fallback search if competitor volume is low.",
    )
    g2_product_slugs: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Lowercase hyphenated G2 product slugs for the competitors (e.g., 'freshbooks', 'quickbooks-online').",
    )
    domain_context_terms: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "5-8 single-word lowercase industry/domain anchor terms "
            "(e.g., ['hosting', 'deploy', 'paas', 'cloud', 'server', 'backend'] or "
            "['invoicing', 'invoice', 'billing', 'accounting', 'clients']) used to "
            "disambiguate common-word competitor names like Render, Wave, or Linear."
        ),
    )


class CrawlDiagnostics(BaseModel):
    """Observability metadata explaining how Stage 1 acquired product context."""

    crawl_method: Literal["httpx_direct", "apify_wcc_fallback", "description_only_fallback"]
    urls_fetched: list[str] = Field(default_factory=list)
    http_status: int | None = None
    extracted_char_count: int = 0
    warning: str | None = None


class ProductProfile(LLMProfileExtraction):
    """Final grounded output of Stage 1, handed directly to Stage 2, 3, 4, and 5."""

    product_name: str
    product_url: str
    root_domain: str
    exclude_terms: list[str]
    llm_provider_used: str
    crawl_diagnostics: CrawlDiagnostics


# --- Stage 2 Handoff Contracts (Pre-defined so Stage 2 plugs in seamlessly) ---

class RawComplaint(BaseModel):
    """Uniform complaint schema emitted by every Stage 2 collector."""

    id: str
    source: SourceType
    sub_source: str = ""  # e.g., "r/freelance", "Hacker News", "G2"
    author_handle: str = "anonymous"
    post_url: str
    published_at: datetime | None = None
    title: str = ""
    text: str
    matched_query: str = ""
    engagement_score: int = 0


class CollectorTelemetry(BaseModel):
    """Per-source health & cost tracking for Stage 2 observability."""

    source: SourceType
    status: Literal["ok", "partial", "skipped", "failed"]
    items_collected: int = 0
    fallback_used: bool = False
    error: str | None = None

class Stage2Result(BaseModel):
    """Complete output of Stage 2 handed directly to Stage 3 (Deterministic Filter)."""

    complaints: list[RawComplaint] = Field(default_factory=list)
    telemetry: list[CollectorTelemetry] = Field(default_factory=list)
    fallback_triggered: bool = False
    pass1_unique_count: int = 0
    pass2_added_count: int = 0
    total_unique_count: int = 0

# --- Stage 3 Contracts (Deterministic Noise & Own-Brand Filter) ---

DropReasonType = Literal[
    "duplicate_url",
    "duplicate_content",
    "non_post_url",
    "own_brand_mention",
    "spam_or_promo",
    "older_than_lookback",
    "too_short",
]


class DroppedComplaint(BaseModel):
    """Audit record for a complaint dropped deterministically in Stage 3."""

    id: str
    source: SourceType
    sub_source: str = ""
    post_url: str
    title: str = ""
    reason: DropReasonType
    matched_rule: str = ""


class Stage3Result(BaseModel):
    """Output of Stage 3 handed directly to Stage 4 (Batched LLM Scorer)."""

    clean_complaints: list[RawComplaint] = Field(default_factory=list)
    input_count: int = 0
    kept_count: int = 0
    dropped_count: int = 0
    drop_reason_counts: dict[str, int] = Field(default_factory=dict)
    dropped_items: list[DroppedComplaint] = Field(default_factory=list)


# --- Stage 4 Contracts (Batched LLM Solvability Hard-Gate & Winability Scorer) ---

LeadTierType = Literal[
    "hot_lead",
    "warm_signal",
    "low_priority",
    "unresolved_market_gap",
    "discarded",
]


class LLMComplaintEvaluation(BaseModel):
    """Per-item structured evaluation returned inside a Stage 4 LLM batch."""

    idx: int = Field(..., ge=0, description="0-based index matching the complaint in this batch.")
    is_genuine_pain_or_seeking_alternative: bool = Field(
        ...,
        description="True ONLY if the author expresses frustration, limitations, pricing pain, or asks for alternatives/recommendations in the target product category. False for news, self-promo, praise, or unrelated topics.",
    )
    competitor_mentioned: str = Field(
        default="None",
        description="Primary competitor brand mentioned (or 'Category-Level' / 'None').",
    )
    pain_category: str = Field(
        default="none",
        description="Short 2-4 word pain theme (e.g., 'pricing_jump', 'reliability_outage', 'ram_cpu_limits', 'spin_down_latency', 'complex_devops', 'off_topic').",
    )
    pain_summary: str = Field(
        ...,
        description="1 concise sentence summarizing what the user is struggling with or looking for.",
    )
    can_client_resolve: bool = Field(
        ...,
        description="HARD GATE: True ONLY if at least one numbered capability in the Founder's Verified Capabilities Ledger factually solves this user's stated problem.",
    )
    matched_capability_indices: list[int] = Field(
        default_factory=list,
        description="1-based indices from the Verified Capabilities Ledger that factually fix this pain. MUST be empty if can_client_resolve is False.",
    )
    solvability_score: int = Field(
        default=0,
        ge=0,
        le=50,
        description="0-50: How directly and completely the cited verified capabilities resolve the complaint. MUST be 0 if can_client_resolve is False.",
    )
    urgency_score: int = Field(
        default=0,
        ge=0,
        le=30,
        description="0-30: Buying/switching intent (24-30 actively asking for alternatives, 14-23 high frustration/churn risk, 5-13 passive comparison, 0 no intent).",
    )
    rejection_or_gap_reason: str = Field(
        default="",
        description="If can_client_resolve is False, explain briefly why (e.g. 'Off-topic store bug', 'Self-promo release', or 'Missing feature: GPU video transcoding').",
    )


class LLMBatchEvaluationResponse(BaseModel):
    """Top-level JSON object wrapper required for Gemini + Groq/OpenRouter JSON mode."""

    evaluations: list[LLMComplaintEvaluation] = Field(default_factory=list)


class ScoredComplaint(BaseModel):
    """Fully grounded, deterministically scored complaint ready for Stage 5 drafting."""

    complaint: RawComplaint
    lead_tier: LeadTierType
    is_genuine_pain: bool
    can_client_resolve: bool
    competitor_mentioned: str
    pain_category: str
    pain_summary: str
    matched_capabilities: list[str] = Field(default_factory=list)
    solvability_score: int = Field(default=0, ge=0, le=50)
    urgency_score: int = Field(default=0, ge=0, le=30)
    recency_score: int = Field(default=0, ge=0, le=20)
    winability_score: int = Field(default=0, ge=0, le=100)
    rejection_or_gap_reason: str = ""
    grounding_override_applied: bool = False


class Stage4Result(BaseModel):
    """Complete output of Stage 4 handed directly to Stage 5 (Drafting, PPE & Report)."""

    hot_leads: list[ScoredComplaint] = Field(default_factory=list)
    warm_signals: list[ScoredComplaint] = Field(default_factory=list)
    unresolved_market_gaps: list[ScoredComplaint] = Field(default_factory=list)
    discarded: list[ScoredComplaint] = Field(default_factory=list)
    total_evaluated: int = 0
    llm_batches_sent: int = 0
    llm_batches_failed: int = 0
    providers_used: list[str] = Field(default_factory=list)

# --- Stage 5 Contracts (Dual Drafting, Flat Dataset Schema & Report) ---

RecordType = Literal["hot_lead", "warm_signal", "unresolved_market_gap"]


class LLMLeadDraft(BaseModel):
    """Per-lead dual drafts generated by Stage 5 LLM batch."""

    idx: int = Field(..., ge=0, description="0-based index matching the hot lead in this batch.")
    draft_public_reply: str = Field(
        ...,
        min_length=30,
        description="Draft A: Value-first, transparent, anti-ban public community reply. Lead with helpful advice, disclose affiliation candidly, never drop raw spammy links.",
    )
    draft_direct_pitch: str = Field(
        ...,
        min_length=30,
        description="Draft B: Concise direct/DM follow-up if the user replies asking for details, mapping their exact pain to the verified capabilities.",
    )
    claim_evidence_summary: str = Field(
        ...,
        description="1 short line explicitly citing which verified capabilities support the claims made in Draft A and Draft B.",
    )


class LLMBatchDraftResponse(BaseModel):
    """Wrapper object for batched Stage 5 outreach generation."""

    drafts: list[LLMLeadDraft] = Field(default_factory=list)


# class DatasetLeadRecord(BaseModel):
#     """Flat camelCase record pushed to Apify Dataset (maps 1:1 to .actor/dataset_schema.json)."""

#     recordType: RecordType
#     winabilityScore: int = Field(default=0, ge=0, le=100)
#     solvabilityScore: int = Field(default=0, ge=0, le=50)
#     urgencyScore: int = Field(default=0, ge=0, le=30)
#     recencyScore: int = Field(default=0, ge=0, le=20)
#     source: SourceType
#     subSource: str
#     authorHandle: str
#     competitorMentioned: str
#     painCategory: str
#     painSummary: str
#     matchedCapabilities: list[str] = Field(default_factory=list)
#     claimEvidenceSummary: str = ""
#     platformSafetyRule: str = ""
#     draftPublicReply: str = ""
#     draftDirectPitch: str = ""
#     gapOrRejectionReason: str = ""
#     postTitle: str = ""
#     postUrl: str
#     publishedAt: str | None = None
#     draftProvider: str = ""

class DatasetLeadRecord(BaseModel):
    """Flat camelCase record pushed to Apify Dataset (maps 1:1 to .actor/dataset_schema.json)."""

    priorityTier: str = "Hot Lead"
    recordType: RecordType
    winabilityScore: int = Field(default=0, ge=0, le=100)
    solvabilityScore: int = Field(default=0, ge=0, le=50)
    urgencyScore: int = Field(default=0, ge=0, le=30)
    recencyScore: int = Field(default=0, ge=0, le=20)
    recommendedAction: str = ""
    outreachStatus: str = "Pending Review"
    source: SourceType
    subSource: str
    authorHandle: str
    competitorMentioned: str
    painCategory: str
    painSummary: str
    verifiedProof: str = ""
    matchedCapabilities: list[str] = Field(default_factory=list)
    claimEvidenceSummary: str = ""
    platformSafetyRule: str = ""
    draftPublicReply: str = ""
    draftDirectPitch: str = ""
    gapOrRejectionReason: str = ""
    postTitle: str = ""
    postUrl: str
    publishedDate: str = "Within lookback window"
    publishedAt: str | None = None
    reportUrl: str = ""
    draftProvider: str = ""

class Stage5Result(BaseModel):
    """Final output summary produced by Stage 5."""

    hot_lead_records: list[DatasetLeadRecord] = Field(default_factory=list)
    warm_signal_records: list[DatasetLeadRecord] = Field(default_factory=list)
    market_gap_records: list[DatasetLeadRecord] = Field(default_factory=list)
    llm_batches_sent: int = 0
    providers_used: list[str] = Field(default_factory=list)
    report_html: str = ""