"""Stage 3: Deterministic Noise, Own-Brand & Recency Filter ($0 Compute).

Eliminates duplicates, non-post directory/profile URLs, own-brand mentions,
spam/promo/hiring/student-perk noise, and stale posts BEFORE any LLM tokens
are spent in Stage 4.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from src.collectors import normalize_url
from src.schemas import (
    ActorInput,
    DroppedComplaint,
    DropReasonType,
    ProductProfile,
    RawComplaint,
    Stage3Result,
)

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 30

EXTRA_SPAM_PHRASES: tuple[str, ...] = (
    "student id benefits",
    "student discount",
    "github student pack",
    "free steam keys",
    "promo code",
    "discount code",
    "coupon code",
    "referral link",
    "use my referral",
    "we are hiring",
    "now hiring",
    "job opening",
    "apply now",
    "black friday deal",
    "cyber monday deal",
    "dm for rate",
    "for sale",
)

NON_POST_URL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("g2_tracking_or_docs", re.compile(r"^https?://(ai|documentation)\.g2\.com/", re.IGNORECASE)),
    ("review_directory_category", re.compile(r"g2\.com/categories/|capterra\.com/sem-compare/", re.IGNORECASE)),
)


def _content_fingerprint(title: str, text: str) -> str:
    """Create a near-duplicate hash over normalized alphanumeric text to catch cross-posts."""
    normalized = re.sub(r"[^a-z0-9]+", " ", f"{title} {text}".lower()).strip()
    # Skip ultra-short prefixes; hash first 260 normalized chars
    snippet = normalized[:260]
    return hashlib.sha256(snippet.encode("utf-8")).hexdigest()[:16]


def _check_non_post_url(url: str) -> str | None:
    """Detect profile pages, docs, and ad-tracking links that are never actionable complaints."""
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")

    # Bluesky profile homepages (bsky.app/profile/<handle>) without /post/<rkey>
    if host.endswith("bsky.app") and "/post/" not in path.lower():
        return "bluesky_profile_not_post"

    for rule_name, pattern in NON_POST_URL_PATTERNS:
        if pattern.search(url):
            return rule_name

    return None


def _build_own_brand_patterns(profile: ProductProfile) -> list[tuple[str, re.Pattern[str]]]:
    """Compile case-insensitive whole-word matchers for the founder's brand and domain."""
    tokens = [
        profile.product_name.strip(),
        profile.root_domain.strip(),
    ]
    # Also include bare domain stem if >= 4 chars (e.g. 'heroku' from 'heroku.com')
    stem = profile.root_domain.split(".")[0].strip()
    if len(stem) >= 4 and stem.lower() not in {t.lower() for t in tokens}:
        tokens.append(stem)

    patterns: list[tuple[str, re.Pattern[str]]] = []
    seen: set[str] = set()
    for tok in tokens:
        key = tok.lower()
        if not tok or len(tok) < 3 or key in seen:
            continue
        seen.add(key)
        patterns.append((tok, re.compile(rf"\b{re.escape(tok)}\b", re.IGNORECASE)))
    return patterns


def _build_spam_patterns(profile: ProductProfile, own_brand_lower: set[str]) -> list[tuple[str, re.Pattern[str]]]:
    """Combine Stage 1 exclude_terms (minus own brand) with deterministic spam phrases."""
    raw_terms: list[str] = list(EXTRA_SPAM_PHRASES)
    for term in profile.exclude_terms:
        cleaned = term.strip()
        if cleaned and cleaned.lower() not in own_brand_lower:
            raw_terms.append(cleaned)

    patterns: list[tuple[str, re.Pattern[str]]] = []
    seen: set[str] = set()
    for term in raw_terms:
        key = term.lower()
        if len(key) < 4 or key in seen:
            continue
        seen.add(key)
        patterns.append((term, re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)))
    return patterns


def filter_complaints(
    complaints: list[RawComplaint],
    profile: ProductProfile,
    actor_input: ActorInput,
    now: datetime | None = None,
) -> Stage3Result:
    """Run all deterministic Stage 3 filters at $0 compute cost."""
    ref_now = now or datetime.now(timezone.utc)
    # 1-day buffer accounts for SERP date rounding to midnight UTC
    cutoff_dt = ref_now - timedelta(days=actor_input.lookbackDays + 1)

    own_brand_patterns = _build_own_brand_patterns(profile)
    own_brand_lower = {tok.lower() for tok, _ in own_brand_patterns}
    spam_patterns = _build_spam_patterns(profile, own_brand_lower)

    seen_urls: set[str] = set()
    seen_content_hashes: set[str] = set()

    clean: list[RawComplaint] = []
    dropped: list[DroppedComplaint] = []
    reason_counts: Counter[str] = Counter()

    def _drop(item: RawComplaint, reason: DropReasonType, rule: str) -> None:
        reason_counts[reason] += 1
        dropped.append(
            DroppedComplaint(
                id=item.id,
                source=item.source,
                sub_source=item.sub_source,
                post_url=item.post_url,
                title=item.title,
                reason=reason,
                matched_rule=rule,
            )
        )

    for item in complaints:
        # 1. Exact Canonical URL Deduplication
        norm_url = normalize_url(item.post_url)
        if norm_url and norm_url in seen_urls:
            _drop(item, "duplicate_url", norm_url)
            continue
        if norm_url:
            seen_urls.add(norm_url)

        # 2. Minimum Text Length Check
        combined_text = f"{item.title}\n{item.text}".strip()
        if len(item.text.strip()) < MIN_TEXT_LENGTH:
            _drop(item, "too_short", f"len={len(item.text.strip())}<{MIN_TEXT_LENGTH}")
            continue

        # 3. Non-Post / Profile / Directory URL Check
        non_post_rule = _check_non_post_url(item.post_url)
        if non_post_rule:
            _drop(item, "non_post_url", non_post_rule)
            continue

        # 4. Own-Brand / Own-Domain Exclusion (Competitor Pain Only!)
        brand_haystack = f"{combined_text}\n{item.post_url}"
        matched_brand = next(
            (tok for tok, pat in own_brand_patterns if pat.search(brand_haystack)),
            None,
        )
        if matched_brand:
            _drop(item, "own_brand_mention", matched_brand)
            continue

        # 5. Spam / Promo / Affiliate / Hiring / Student-Perk Exclusion
        matched_spam = next(
            (term for term, pat in spam_patterns if pat.search(combined_text)),
            None,
        )
        if matched_spam:
            _drop(item, "spam_or_promo", matched_spam)
            continue

        # 6. Lookback Window Recency Exclusion (preserve None dates from after:-bounded SERP)
        if item.published_at is not None:
            pub = (
                item.published_at
                if item.published_at.tzinfo is not None
                else item.published_at.replace(tzinfo=timezone.utc)
            )
            if pub < cutoff_dt:
                _drop(
                    item,
                    "older_than_lookback",
                    f"{pub.date().isoformat()}<{cutoff_dt.date().isoformat()}",
                )
                continue

        # 7. Near-Duplicate Content Deduplication (catches cross-posted spam/questions)
        fp = _content_fingerprint(item.title, item.text)
        if fp in seen_content_hashes:
            _drop(item, "duplicate_content", f"fp={fp}")
            continue
        seen_content_hashes.add(fp)

        clean.append(item)

    logger.info(
        "Stage 3 filter complete: input=%d kept=%d dropped=%d reasons=%s",
        len(complaints),
        len(clean),
        len(dropped),
        dict(reason_counts),
    )

    return Stage3Result(
        clean_complaints=clean,
        input_count=len(complaints),
        kept_count=len(clean),
        dropped_count=len(dropped),
        drop_reason_counts=dict(reason_counts),
        dropped_items=dropped,
    )