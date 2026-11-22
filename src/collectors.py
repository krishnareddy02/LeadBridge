"""Stage 2: Multi-Source Async Complaint Harvester with Agentic Fallback.

Consumes Stage 1's ProductProfile and harvests fresh public competitor complaints
in parallel across:
  - Hacker News (HN Algolia Official API - $0)
  - Bluesky (AT Protocol API - $0, with automatic 403 -> Google SERP rescue)
  - Google SERP Dorking (apify/google-search-scraper)
  - Reddit Fresh Posts (harshmaur/reddit-scraper-pro)
  - G2 Critical Reviews (automation-lab/g2-scraper, optional Deep Mode)

If Pass 1 yields fewer than MIN_COMPLAINTS_FOR_FALLBACK unique items, Stage 2
automatically triggers Pass 2 using profile.category_problem_queries.
"""

from __future__ import annotations

import asyncio
import hashlib
import html as html_lib
import inspect
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from src.schemas import (
    ActorInput,
    CollectorTelemetry,
    ProductProfile,
    RawComplaint,
    SourceType,
    Stage2Result,
)

logger = logging.getLogger(__name__)

MIN_COMPLAINTS_FOR_FALLBACK = 8

PAIN_SIGNAL_WORDS = (
    "alternative",
    "switching",
    "expensive",
    "broken",
    "pricing",
    "issues",
    "problem",
    "slow",
    "frustrated",
    "sucks",
    "cancel",
    "fees",
    "limit",
)

TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "fbclid",
    "gclid",
    "share_id",
}

CATEGORY_STOPWORDS = {
    "alternative",
    "alternatives",
    "issues",
    "issue",
    "problem",
    "problems",
    "software",
    "tool",
    "tools",
    "platform",
    "service",
    "with",
    "from",
    "that",
    "this",
    "best",
    "easy",
    "good",
    "affordable",
    "simple",
}

HN_SOFTWARE_OR_PAIN_SIGNALS = re.compile(
    r"\b("
    r"alternative|switch\w*|migrat\w*|pricing|price|cost|expensive|bill\w*|tier|limit\w*|"
    r"deploy\w*|host\w*|paas|saas|cloud|server|infra\w*|devops|database|postgres|redis|"
    r"app|tool|software|platform|service|api|vendor|support|outage|downtime|broken|"
    r"slow|sucks|jank|frustrat\w*|cancel\w*|issue\w*|problem\w*|bug\w*|vs\.?|compare|"
    r"invoic\w*|payment\w*|subscription|plan|dashboard|integration|workflow"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Normalization, Deduplication & Relevance Helpers
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """Canonicalize URLs so cross-source duplicates (e.g. SERP + Reddit) collapse cleanly."""
    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    clean_query = urlencode(
        [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if k.lower() not in TRACKING_QUERY_PARAMS
        ]
    )
    path = parsed.path.rstrip("/") or "/"
    return urlunparse(
        (
            parsed.scheme.lower() or "https",
            parsed.netloc.lower(),
            path,
            "",
            clean_query,
            "",
        )
    )


def make_complaint_id(source: str, post_url: str, title: str, text: str) -> str:
    """Generate a deterministic short hash ID for deduplication across passes."""
    norm_url = normalize_url(post_url)
    fingerprint = (
        norm_url
        if norm_url
        else f"{source}:{title.strip().lower()}:{text[:140].strip().lower()}"
    )
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


def strip_html_snippet(raw: str | None) -> str:
    """Convert HN/G2 HTML fragments and entities into clean plain text."""
    if not raw:
        return ""
    unescaped = html_lib.unescape(str(raw))
    if "<" in unescaped and ">" in unescaped:
        text = BeautifulSoup(unescaped, "lxml").get_text(separator=" ", strip=True)
    else:
        text = unescaped
    return re.sub(r"\s+", " ", text).strip()


def parse_timestamp(value: Any) -> datetime | None:
    """Safely parse Unix timestamps or ISO-8601 strings into UTC datetimes."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            # Handle millisecond timestamps if > year 2100 in seconds
            ts = value / 1000.0 if value > 4_102_444_800 else float(value)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        s = str(value).strip()
        if s.isdigit():
            return parse_timestamp(int(s))
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _clean_competitor_for_search(name: str) -> str:
    """Trim generic '.com' suffixes (e.g. 'Render.com' -> 'Render') while keeping '.io' brands."""
    cleaned = name.strip()
    if cleaned.lower().endswith(".com"):
        return cleaned[:-4].strip()
    return cleaned


def _is_ambiguous_single_word(comp: str) -> bool:
    """Detect single-word plain dictionary names (Render, Wave, Linear) vs multi-word/CamelCase brands."""
    cleaned = comp.strip()
    if not cleaned or " " in cleaned or "." in cleaned or "-" in cleaned:
        return False
    # Internal CamelCase (FreshBooks, ClickUp, HubSpot, PostHog) or digits are unambiguous brand names
    if any(ch.isupper() for ch in cleaned[1:]) or any(ch.isdigit() for ch in cleaned):
        return False
    return True


def _ensure_pain_intent(category_query: str) -> str:
    """Ensure fallback category queries target frustration/alternatives, not generic docs."""
    lower = category_query.lower()
    if any(word in lower for word in PAIN_SIGNAL_WORDS):
        return category_query.strip()
    return f"{category_query.strip()} alternative OR issues"


def _matches_competitor_and_domain(
    text: str,
    competitors: list[str],
    domain_terms: list[str],
    require_brand: bool = True,
) -> bool:
    """Deterministic relevance gate: prevents 'Render'->3D graphics and 'limits'->Warhammer40k."""
    if not text:
        return False

    matched_comp: str | None = None
    for comp in competitors:
        clean_c = _clean_competitor_for_search(comp)
        if not clean_c:
            continue
        if re.search(rf"\b{re.escape(clean_c)}\b", text, re.IGNORECASE):
            matched_comp = clean_c
            break

    if require_brand and not matched_comp:
        return False

    # For plain single-word brands that collide with common English (e.g. Render, Railway, Wave):
    if matched_comp and _is_ambiguous_single_word(matched_comp):
        low_brand = matched_comp.lower()
        # Reject lowercase English verb usage (e.g. "to render", "will render", "can render"),
        # while still allowing capitalized brand mentions like "alternative to Render".
        verb_collision = re.search(
            rf"\b(?:to|will|can|could|would|must|should|won't|can't|doesn't|not)\s+{re.escape(low_brand)}\b",
            text,  # intentionally case-sensitive
        )
        if verb_collision and not re.search(rf"\b{re.escape(matched_comp)}\b", text):
            return False

        if domain_terms:
            domain_pat = re.compile(
                r"\b(" + "|".join(re.escape(t) for t in domain_terms[:8]) + r")\w*\b",
                re.IGNORECASE,
            )
            if not domain_pat.search(text):
                return False
        else:
            return bool(HN_SOFTWARE_OR_PAIN_SIGNALS.search(text))

    return True


def _matches_category_context(
    text: str,
    category_queries: list[str],
    domain_terms: list[str],
) -> bool:
    """Lightweight topical guard for Pass 2 brand-agnostic fallback results."""
    if not text:
        return False
    tokens: set[str] = set()
    for term in domain_terms[:8]:
        if len(term) >= 3:
            tokens.add(term.lower())
    for q in category_queries:
        for tok in re.findall(r"[a-z0-9]{4,}", q.lower()):
            if tok not in CATEGORY_STOPWORDS:
                tokens.add(tok)
    if not tokens:
        return True
    low = text.lower()
    return any(re.search(rf"\b{re.escape(tok)}", low) for tok in tokens)


def _get_dataset_id(run_obj: Any) -> str | None:
    if run_obj is None:
        return None
    if isinstance(run_obj, dict):
        return run_obj.get("defaultDatasetId") or run_obj.get("default_dataset_id")
    return getattr(run_obj, "default_dataset_id", None) or getattr(
        run_obj, "defaultDatasetId", None
    )


async def _run_apify_actor_and_fetch_items(
    actor_id: str,
    run_input: dict[str, Any],
    apify_token: str,
    timeout_secs: int = 90,
    max_items: int = 100,
) -> list[dict[str, Any]]:
    """Run an Apify Store sub-Actor and return dataset items (apify-client v1/v2/v3 safe)."""
    from apify_client import ApifyClientAsync

    client = ApifyClientAsync(token=apify_token)
    actor_client = client.actor(actor_id)

    # Adapt cleanly to apify-client v2/v3 (timeout=timedelta) vs v1 (timeout_secs=int)
    call_params = inspect.signature(actor_client.call).parameters
    timeout_kw: dict[str, Any] = {}
    if "timeout" in call_params:
        timeout_kw["timeout"] = timedelta(seconds=timeout_secs)
    elif "timeout_secs" in call_params:
        timeout_kw["timeout_secs"] = timeout_secs

    run = await actor_client.call(run_input=run_input, **timeout_kw)
    dataset_id = _get_dataset_id(run)
    if not dataset_id:
        raise RuntimeError(f"Actor {actor_id} completed without a defaultDatasetId")

    page = await client.dataset(dataset_id).list_items(limit=max_items)
    items = getattr(page, "items", None)
    if items is None and isinstance(page, dict):
        items = page.get("items", [])
    return list(items or [])


def deduplicate_complaints(items: list[RawComplaint]) -> list[RawComplaint]:
    """Deduplicate by canonical URL (preferred) and fallback complaint ID."""
    seen_keys: set[str] = set()
    unique: list[RawComplaint] = []
    for item in items:
        url_key = normalize_url(item.post_url)
        dedupe_key = url_key or item.id
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        unique.append(item)
    return unique


# ---------------------------------------------------------------------------
# Collector 1: Hacker News (HN Algolia Official Public API - $0)
# ---------------------------------------------------------------------------


def _hn_hit_matches_query(
    query: str,
    title: str,
    body: str,
    is_brand_query: bool = True,
    domain_terms: list[str] | None = None,
) -> bool:
    """Eliminate Algolia typo-tolerance noise (e.g. 'Railway'->'always', 'Render'->'lenders')."""
    haystack = f"{title}\n{body}"
    clean_q = query.strip().strip('"')
    if not clean_q:
        return False

    if is_brand_query:
        return _matches_competitor_and_domain(
            haystack, [clean_q], domain_terms or [], require_brand=True
        )

    # Category fallback query: require at least 2 meaningful tokens from query to appear
    tokens = [
        t
        for t in re.findall(r"[a-z0-9]{3,}", clean_q.lower())
        if t not in {"for", "and", "the", "with"}
    ]
    if not tokens:
        return True
    low = haystack.lower()
    matched = sum(1 for t in tokens if re.search(rf"\b{re.escape(t)}", low))
    return matched >= min(2, len(tokens))


async def collect_hacker_news(
    queries: list[str],
    lookback_days: int,
    http_client: httpx.AsyncClient,
    hits_per_query: int = 25,
    is_brand_query: bool = True,
    domain_terms: list[str] | None = None,
) -> tuple[list[RawComplaint], CollectorTelemetry]:
    cutoff_unix = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())
    endpoint = "https://hn.algolia.com/api/v1/search_by_date"
    complaints: list[RawComplaint] = []
    errors: list[str] = []
    d_terms = domain_terms or []

    async def _fetch_one(q: str) -> None:
        clean_q = q.strip().strip('"')
        # Pair ambiguous single-word brands (Render, Railway, Wave) with primary domain anchor
        # so Algolia doesn't fill all 25 slots with 3D graphics or verb comments.
        if is_brand_query and _is_ambiguous_single_word(clean_q) and d_terms:
            search_q = f'"{clean_q}" {d_terms[0]}'
        elif is_brand_query:
            search_q = f'"{clean_q}"'
        else:
            search_q = clean_q

        resp = await http_client.get(
            endpoint,
            params={
                "query": search_q,
                "tags": "(story,comment)",
                "numericFilters": f"created_at_i>{cutoff_unix}",
                "hitsPerPage": hits_per_query,
                "advancedSyntax": "true",
                "restrictSearchableAttributes": "title,comment_text,story_text",
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        for hit in payload.get("hits", []):
            obj_id = str(hit.get("objectID") or "").strip()
            if not obj_id:
                continue
            title = strip_html_snippet(hit.get("title") or hit.get("story_title") or "")
            body = strip_html_snippet(hit.get("comment_text") or hit.get("story_text") or "")
            combined_text = body or title
            if len(combined_text) < 25:
                continue
            if not _hn_hit_matches_query(
                clean_q,
                title,
                body,
                is_brand_query=is_brand_query,
                domain_terms=d_terms,
            ):
                continue

            post_url = f"https://news.ycombinator.com/item?id={obj_id}"
            complaints.append(
                RawComplaint(
                    id=make_complaint_id("hacker_news", post_url, title, combined_text),
                    source="hacker_news",
                    sub_source="Hacker News",
                    author_handle=str(hit.get("author") or "anonymous"),
                    post_url=post_url,
                    published_at=parse_timestamp(hit.get("created_at_i") or hit.get("created_at")),
                    title=title,
                    text=combined_text[:1500],
                    matched_query=clean_q,
                    engagement_score=int(hit.get("points") or 0),
                )
            )

    results = await asyncio.gather(*[_fetch_one(q) for q in queries[:5]], return_exceptions=True)
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            errors.append(f"HN query '{queries[idx]}': {type(res).__name__}: {res}")

    deduped = deduplicate_complaints(complaints)
    if errors and not deduped:
        return [], CollectorTelemetry(
            source="hacker_news", status="failed", items_collected=0, error="; ".join(errors)
        )
    status = "partial" if errors else "ok"
    return deduped, CollectorTelemetry(
        source="hacker_news",
        status=status,
        items_collected=len(deduped),
        error="; ".join(errors) if errors else None,
    )


# ---------------------------------------------------------------------------
# Collector 2: Bluesky (AT Protocol Public API - $0 + Auth/403 Resilience)
# ---------------------------------------------------------------------------


def _at_uri_to_bsky_web_url(uri: str, author_handle: str) -> str:
    """Convert 'at://did:plc:.../app.bsky.feed.post/<rkey>' into clickable bsky.app URL."""
    match = re.search(r"/app\.bsky\.feed\.post/([^/]+)$", uri or "")
    if match and author_handle:
        return f"https://bsky.app/profile/{author_handle}/post/{match.group(1)}"
    return uri or ""


async def collect_bluesky(
    queries: list[str],
    lookback_days: int,
    http_client: httpx.AsyncClient,
    limit_per_query: int = 20,
    competitors: list[str] | None = None,
    domain_terms: list[str] | None = None,
) -> tuple[list[RawComplaint], CollectorTelemetry, bool]:
    """Returns (complaints, telemetry, hit_403_needs_serp_rescue)."""
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    since_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    bsky_handle = os.getenv("BLUESKY_HANDLE")
    bsky_app_pw = os.getenv("BLUESKY_APP_PASSWORD")
    headers: dict[str, str] = {"Accept": "application/json"}
    endpoint = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"

    # Tier 1: Optional authenticated Bluesky session (avoids public CDN 403s)
    if bsky_handle and bsky_app_pw:
        try:
            auth_resp = await http_client.post(
                "https://bsky.social/xrpc/com.atproto.server.createSession",
                json={"identifier": bsky_handle, "password": bsky_app_pw},
            )
            if auth_resp.status_code == 200:
                jwt = auth_resp.json().get("accessJwt")
                if jwt:
                    headers["Authorization"] = f"Bearer {jwt}"
                    endpoint = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"
        except Exception as auth_exc:
            logger.debug("Bluesky auth optional step failed; trying public API: %s", auth_exc)

    complaints: list[RawComplaint] = []
    errors: list[str] = []
    hit_403 = False

    for q in queries[:3]:
        try:
            resp = await http_client.get(
                endpoint,
                params={"q": q, "limit": limit_per_query, "sort": "latest", "since": since_iso},
                headers=headers,
            )
            if resp.status_code in (401, 403):
                hit_403 = True
                errors.append(f"Bluesky HTTP {resp.status_code} (public CDN rate/auth block)")
                break  # Don't hammer blocked CDN with remaining queries
            resp.raise_for_status()

            for post in resp.json().get("posts", []):
                record = post.get("record") or {}
                text = strip_html_snippet(record.get("text") or "")
                if len(text) < 20:
                    continue
                if competitors and not _matches_competitor_and_domain(
                    text, competitors, domain_terms or [], require_brand=True
                ):
                    continue
                author = (post.get("author") or {}).get("handle") or "anonymous"
                post_url = _at_uri_to_bsky_web_url(str(post.get("uri") or ""), author)
                published_at = parse_timestamp(record.get("createdAt") or post.get("indexedAt"))
                if published_at and published_at < cutoff_dt:
                    continue
                likes = int(post.get("likeCount") or 0) + int(post.get("replyCount") or 0)
                complaints.append(
                    RawComplaint(
                        id=make_complaint_id("bluesky", post_url, "", text),
                        source="bluesky",
                        sub_source="Bluesky",
                        author_handle=f"@{author}",
                        post_url=post_url,
                        published_at=published_at,
                        title="",
                        text=text[:1500],
                        matched_query=q,
                        engagement_score=likes,
                    )
                )
        except Exception as exc:
            errors.append(f"Bluesky query '{q}': {type(exc).__name__}: {exc}")

    deduped = deduplicate_complaints(complaints)
    if hit_403 and not deduped:
        return (
            [],
            CollectorTelemetry(
                source="bluesky",
                status="skipped",
                items_collected=0,
                fallback_used=False,
                error="; ".join(errors),
            ),
            True,
        )
    if errors and not deduped:
        return (
            [],
            CollectorTelemetry(
                source="bluesky", status="failed", items_collected=0, error="; ".join(errors)
            ),
            False,
        )
    return (
        deduped,
        CollectorTelemetry(
            source="bluesky",
            status="partial" if errors else "ok",
            items_collected=len(deduped),
            error="; ".join(errors) if errors else None,
        ),
        hit_403,
    )


# ---------------------------------------------------------------------------
# Collector 3: Google SERP Dorking (apify/google-search-scraper)
# ---------------------------------------------------------------------------


def _compact_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def build_google_serp_dorks(
    profile: ProductProfile,
    lookback_days: int,
    use_category_fallback: bool = False,
    include_bsky_rescue: bool = False,
) -> list[str]:
    """Generate high-precision, domain-disambiguated, whitespace-clean Google dorks."""
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    own_exclusion = f'-"{profile.product_name}" -site:{profile.root_domain}'
    d_terms = list(getattr(profile, "domain_context_terms", None) or [])[:4]
    domain_clause = "(" + " OR ".join(f'"{t}"' for t in d_terms) + ")" if d_terms else ""

    if use_category_fallback:
        dorks: list[str] = []
        for cq in profile.category_problem_queries[:3]:
            pain_q = _ensure_pain_intent(cq)
            dorks.append(
                _compact_spaces(
                    f"( site:reddit.com OR site:news.ycombinator.com OR site:indiehackers.com ) "
                    f"({pain_q}) {own_exclusion} after:{cutoff_date}"
                )
            )
        return dorks

    clean_comps = [_clean_competitor_for_search(c) for c in profile.competitors[:5] if c.strip()]
    unambiguous = [c for c in clean_comps if not _is_ambiguous_single_word(c)]
    ambiguous = [c for c in clean_comps if _is_ambiguous_single_word(c)]

    pain_clause = '("alternative" OR "switching" OR "expensive" OR "broken" OR "pricing" OR "issues")'

    # Keep unambiguous brands separate from dictionary-word brands so "Render" / "Wave"
    # never starve out "AWS Elastic Beanstalk" or "QuickBooks Online".
    dorks: list[str] = []
    if unambiguous:
        u_clause = " OR ".join(f'"{c}"' for c in unambiguous[:3])
        dorks.append(
            _compact_spaces(
                f"site:reddit.com ({u_clause}) {pain_clause} {own_exclusion} after:{cutoff_date}"
            )
        )
    if ambiguous:
        a_clause = " OR ".join(f'"{c}"' for c in ambiguous[:3])
        dorks.append(
            _compact_spaces(
                f"site:reddit.com ({a_clause}) {domain_clause} {pain_clause} {own_exclusion} after:{cutoff_date}"
            )
        )

    # Review sites & tech forums (always domain-anchored to block off-topic pages)
    all_clause = " OR ".join(f'"{c}"' for c in clean_comps[:4]) or f'"{profile.product_category}"'
    dorks.append(
        _compact_spaces(
            f"( site:g2.com OR site:trustpilot.com OR site:capterra.com OR site:news.ycombinator.com ) "
            f"({all_clause}) {domain_clause} "
            f'("alternative" OR "switching" OR "terrible" OR "cancel" OR "pricing" OR "support") '
            f"{own_exclusion} after:{cutoff_date}"
        )
    )

    if include_bsky_rescue:
        dorks.append(
            _compact_spaces(
                f"site:bsky.app ({all_clause}) {domain_clause} {pain_clause} {own_exclusion} after:{cutoff_date}"
            )
        )
    return dorks


def build_reddit_search_terms(profile: ProductProfile) -> list[str]:
    """Build exact-quoted boolean Reddit search queries that resist Reddit's loose sort=new matching."""
    d_terms = list(getattr(profile, "domain_context_terms", None) or [])[:3]
    domain_or = " OR ".join(d_terms)
    terms: list[str] = []

    for comp in profile.competitors[:4]:
        clean_c = _clean_competitor_for_search(comp)
        if not clean_c:
            continue
        if _is_ambiguous_single_word(clean_c) and domain_or:
            terms.append(f'"{clean_c}" ({domain_or}) (alternative OR switching OR pricing OR issues)')
        else:
            terms.append(f'"{clean_c}" (alternative OR switching OR pricing OR issues OR broken)')

    return terms or profile.competitor_pain_queries[:4]


async def collect_google_serp(
    dorks: list[str],
    apify_token: str | None,
    results_per_page: int = 10,
    competitors: list[str] | None = None,
    domain_terms: list[str] | None = None,
    require_brand_match: bool = False,
    category_queries: list[str] | None = None,
) -> tuple[list[RawComplaint], CollectorTelemetry]:
    if not apify_token:
        return [], CollectorTelemetry(
            source="google_serp",
            status="skipped",
            items_collected=0,
            error="APIFY_TOKEN not configured",
        )

    try:
        pages = await _run_apify_actor_and_fetch_items(
            actor_id="apify/google-search-scraper",
            run_input={
                "queries": "\n".join(dorks),
                "maxPagesPerQuery": 1,
                "resultsPerPage": results_per_page,
                "countryCode": "us",
                "languageCode": "en",
                "mobileResults": False,
            },
            apify_token=apify_token,
            timeout_secs=90,
            max_items=20,
        )
    except Exception as exc:
        return [], CollectorTelemetry(
            source="google_serp",
            status="failed",
            items_collected=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    complaints: list[RawComplaint] = []
    for page in pages:
        matched_q = str((page.get("searchQuery") or {}).get("term") or "")
        for res in page.get("organicResults") or []:
            url = str(res.get("url") or "").strip()
            title = strip_html_snippet(res.get("title") or "")
            desc = strip_html_snippet(res.get("description") or "")
            text = f"{title}. {desc}".strip(". ")
            if not url or len(text) < 25:
                continue

            if require_brand_match and competitors:
                if not _matches_competitor_and_domain(
                    text, competitors, domain_terms or [], require_brand=True
                ):
                    continue
            elif category_queries:
                if not _matches_category_context(text, category_queries, domain_terms or []):
                    continue

            host = (urlparse(url).hostname or "").replace("www.", "")
            complaints.append(
                RawComplaint(
                    id=make_complaint_id("google_serp", url, title, text),
                    source="google_serp",
                    sub_source=host or "Google SERP",
                    author_handle="public_web",
                    post_url=url,
                    published_at=parse_timestamp(res.get("date")),
                    title=title,
                    text=text[:1500],
                    matched_query=matched_q,
                    engagement_score=0,
                )
            )

    deduped = deduplicate_complaints(complaints)
    return deduped, CollectorTelemetry(
        source="google_serp",
        status="ok",
        items_collected=len(deduped),
    )


# ---------------------------------------------------------------------------
# Collector 4: Reddit Fresh Threads (harshmaur/reddit-scraper-pro)
# ---------------------------------------------------------------------------


async def collect_reddit(
    search_terms: list[str],
    lookback_days: int,
    apify_token: str | None,
    max_posts_count: int = 15,
    competitors: list[str] | None = None,
    domain_terms: list[str] | None = None,
    require_brand_match: bool = True,
    category_queries: list[str] | None = None,
) -> tuple[list[RawComplaint], CollectorTelemetry]:
    if not apify_token:
        return [], CollectorTelemetry(
            source="reddit",
            status="skipped",
            items_collected=0,
            error="APIFY_TOKEN not configured",
        )

    try:
        items = await _run_apify_actor_and_fetch_items(
            actor_id="harshmaur/reddit-scraper-pro",
            run_input={
                "searchTerms": search_terms[:4],
                "sort": "new",
                "postedAfter": f"{lookback_days} days",
                "searchPosts": True,
                "searchComments": False,  # Critical: prevents credit burn crawling nested comments
                "crawlCommentsPerPost": False,
                "maxPostsCount": max_posts_count,
            },
            apify_token=apify_token,
            timeout_secs=120,
            max_items=80,
        )
    except Exception as exc:
        return [], CollectorTelemetry(
            source="reddit",
            status="failed",
            items_collected=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    complaints: list[RawComplaint] = []
    for item in items:
        # Filter strictly to posts if dataType metadata is present
        data_type = str(item.get("dataType") or "post").lower()
        if data_type != "post":
            continue

        post_url = str(item.get("postUrl") or item.get("url") or "").strip()
        title = strip_html_snippet(item.get("title") or "")
        body = strip_html_snippet(item.get("body") or item.get("selftext") or "")
        combined = f"{title}\n{body}".strip()
        if not post_url or len(combined) < 20:
            continue

        # Kill loose Reddit sort=new junk (e.g. r/Warhammer40k, r/PhilippineRealEstate)
        if require_brand_match and competitors:
            if not _matches_competitor_and_domain(
                combined, competitors, domain_terms or [], require_brand=True
            ):
                continue
        elif category_queries:
            if not _matches_category_context(combined, category_queries, domain_terms or []):
                continue

        community = str(item.get("communityName") or item.get("subreddit") or "reddit")
        if community and not community.startswith("r/") and community != "reddit":
            community = f"r/{community}"

        author = str(item.get("authorName") or item.get("username") or "redditor")
        upvotes = int(item.get("upVotes") or item.get("score") or 0)

        complaints.append(
            RawComplaint(
                id=make_complaint_id("reddit", post_url, title, combined),
                source="reddit",
                sub_source=community,
                author_handle=f"u/{author}" if not author.startswith("u/") else author,
                post_url=post_url,
                published_at=parse_timestamp(item.get("createdAt") or item.get("created_utc")),
                title=title,
                text=combined[:1800],
                matched_query=str(item.get("searchTerm") or ""),
                engagement_score=upvotes,
            )
        )

    deduped = deduplicate_complaints(complaints)
    return deduped, CollectorTelemetry(
        source="reddit",
        status="ok",
        items_collected=len(deduped),
    )


# ---------------------------------------------------------------------------
# Collector 5: Optional Deep Mode — G2 Critical Reviews (automation-lab/g2-scraper)
# ---------------------------------------------------------------------------


async def collect_g2_reviews(
    g2_slugs: list[str],
    lookback_days: int,
    apify_token: str | None,
    max_reviews: int = 15,
) -> tuple[list[RawComplaint], CollectorTelemetry]:
    if not apify_token:
        return [], CollectorTelemetry(
            source="g2",
            status="skipped",
            items_collected=0,
            error="APIFY_TOKEN not configured",
        )
    if not g2_slugs:
        return [], CollectorTelemetry(
            source="g2",
            status="skipped",
            items_collected=0,
            error="No G2 product slugs available in ProductProfile",
        )

    cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    product_urls = [f"https://www.g2.com/products/{slug}/reviews" for slug in g2_slugs[:3]]

    try:
        items = await _run_apify_actor_and_fetch_items(
            actor_id="automation-lab/g2-scraper",
            run_input={
                "mode": "product_reviews",
                "productUrls": product_urls,
                "maxRating": 6,  # NPS 0-10 scale: <=6 captures 1-3 star detractors
                "sortReviews": "newest",
                "publishedAfter": cutoff_iso,
                "maxReviews": max_reviews,
            },
            apify_token=apify_token,
            timeout_secs=150,
            max_items=50,
        )
    except Exception as exc:
        return [], CollectorTelemetry(
            source="g2",
            status="failed",
            items_collected=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    complaints: list[RawComplaint] = []
    for item in items:
        post_url = str(item.get("reviewUrl") or item.get("url") or "").strip()
        title = strip_html_snippet(item.get("title") or item.get("reviewTitle") or "")
        dislike = strip_html_snippet(
            item.get("dislike")
            or item.get("whatDoYouDislike")
            or item.get("cons")
            or item.get("text")
            or ""
        )
        combined = f"{title}\nDislikes: {dislike}".strip() if dislike else title
        if len(combined) < 20:
            continue

        product_name = str(item.get("productName") or item.get("product") or "G2")
        author = str(item.get("author") or item.get("reviewerName") or "Verified G2 User")
        complaints.append(
            RawComplaint(
                id=make_complaint_id("g2", post_url, title, combined),
                source="g2",
                sub_source=f"G2 ({product_name})",
                author_handle=author,
                post_url=post_url or product_urls[0],
                published_at=parse_timestamp(item.get("publishedAt") or item.get("date")),
                title=title,
                text=combined[:1800],
                matched_query=product_name,
                engagement_score=0,
            )
        )

    deduped = deduplicate_complaints(complaints)
    return deduped, CollectorTelemetry(
        source="g2",
        status="ok",
        items_collected=len(deduped),
    )


# ---------------------------------------------------------------------------
# Stage 2 Master Orchestrator (Pass 1 Parallel + Conditional Pass 2 Fallback)
# ---------------------------------------------------------------------------


async def harvest_complaints(
    profile: ProductProfile,
    actor_input: ActorInput,
    apify_token: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    min_complaints_threshold: int = MIN_COMPLAINTS_FOR_FALLBACK,
) -> Stage2Result:
    """Execute Stage 2 across all selected sources with fault isolation & agentic fallback."""
    token = apify_token if apify_token is not None else os.getenv("APIFY_TOKEN")
    selected_sources: set[SourceType] = set(actor_input.sources)
    domain_terms = list(getattr(profile, "domain_context_terms", None) or [])

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=5.0),
        follow_redirects=True,
    )

    try:
        # Prepare domain-disambiguated competitor queries for Pass 1
        clean_comps = [_clean_competitor_for_search(c) for c in profile.competitors[:5] if c.strip()]
        hn_queries = clean_comps or profile.competitor_pain_queries[:4]
        bsky_queries = profile.competitor_pain_queries[:3] or clean_comps[:3]
        reddit_queries = build_reddit_search_terms(profile)

        # --- PASS 1: Run all enabled collectors concurrently ---
        tasks: dict[SourceType, asyncio.Future[Any]] = {}

        if "hacker_news" in selected_sources:
            tasks["hacker_news"] = asyncio.ensure_future(
                collect_hacker_news(
                    hn_queries,
                    actor_input.lookbackDays,
                    client,
                    is_brand_query=True,
                    domain_terms=domain_terms,
                )
            )
        if "bluesky" in selected_sources:
            tasks["bluesky"] = asyncio.ensure_future(
                collect_bluesky(
                    bsky_queries,
                    actor_input.lookbackDays,
                    client,
                    competitors=profile.competitors,
                    domain_terms=domain_terms,
                )
            )
        if "google_serp" in selected_sources:
            dorks = build_google_serp_dorks(profile, actor_input.lookbackDays)
            tasks["google_serp"] = asyncio.ensure_future(
                collect_google_serp(
                    dorks,
                    token,
                    competitors=profile.competitors,
                    domain_terms=domain_terms,
                    require_brand_match=True,
                )
            )
        if "reddit" in selected_sources:
            tasks["reddit"] = asyncio.ensure_future(
                collect_reddit(
                    reddit_queries,
                    actor_input.lookbackDays,
                    token,
                    competitors=profile.competitors,
                    domain_terms=domain_terms,
                    require_brand_match=True,
                )
            )
        if "g2" in selected_sources:
            tasks["g2"] = asyncio.ensure_future(
                collect_g2_reviews(profile.g2_product_slugs, actor_input.lookbackDays, token)
            )

        raw_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        all_complaints: list[RawComplaint] = []
        telemetry_map: dict[SourceType, CollectorTelemetry] = {}
        bsky_needs_serp_rescue = False

        for src, result in zip(tasks.keys(), raw_results):
            if isinstance(result, Exception):
                logger.warning("Stage 2 collector %s raised unhandled exception: %s", src, result)
                telemetry_map[src] = CollectorTelemetry(
                    source=src,
                    status="failed",
                    items_collected=0,
                    error=f"{type(result).__name__}: {result}",
                )
                continue

            if src == "bluesky":
                items, tel, hit_403 = result
                all_complaints.extend(items)
                telemetry_map[src] = tel
                bsky_needs_serp_rescue = hit_403
            else:
                items, tel = result
                all_complaints.extend(items)
                telemetry_map[src] = tel

        # Automatic Bluesky 403 -> 1-page Google SERP site:bsky.app rescue
        if bsky_needs_serp_rescue and "google_serp" in selected_sources and token:
            logger.info("Bluesky returned 403; running lightweight site:bsky.app SERP rescue")
            bsky_dork = build_google_serp_dorks(
                profile, actor_input.lookbackDays, include_bsky_rescue=True
            )[-1:]
            rescue_items, rescue_tel = await collect_google_serp(
                bsky_dork,
                token,
                results_per_page=10,
                competitors=profile.competitors,
                domain_terms=domain_terms,
                require_brand_match=True,
            )
            if rescue_items:
                for item in rescue_items:
                    item.sub_source = "Bluesky (via SERP)"
                all_complaints.extend(rescue_items)
                telemetry_map["bluesky"].fallback_used = True
                telemetry_map["bluesky"].status = "partial"
                telemetry_map["bluesky"].items_collected += len(rescue_items)
            elif rescue_tel.error:
                logger.debug("Bluesky SERP rescue produced no items: %s", rescue_tel.error)

        pass1_deduped = deduplicate_complaints(all_complaints)
        pass1_count = len(pass1_deduped)

        # --- CONDITIONAL AGENTIC PASS 2 FALLBACK ---
        fallback_triggered = False
        final_complaints = pass1_deduped

        if pass1_count < min_complaints_threshold and profile.category_problem_queries:
            fallback_triggered = True
            logger.info(
                "Pass 1 returned only %d unique items (< %d). Triggering Agentic Pass 2 fallback with category queries.",
                pass1_count,
                min_complaints_threshold,
            )
            cat_queries = [_ensure_pain_intent(q) for q in profile.category_problem_queries[:3]]
            p2_tasks: list[asyncio.Future[Any]] = []

            if "hacker_news" in selected_sources:
                # HN works best with concise 2-3 word queries without boolean OR syntax
                hn_cat = [q.replace(" alternative OR issues", "") for q in cat_queries]
                p2_tasks.append(
                    asyncio.ensure_future(
                        collect_hacker_news(
                            hn_cat,
                            actor_input.lookbackDays,
                            client,
                            hits_per_query=15,
                            is_brand_query=False,
                            domain_terms=domain_terms,
                        )
                    )
                )
            if "google_serp" in selected_sources and token:
                cat_dorks = build_google_serp_dorks(
                    profile, actor_input.lookbackDays, use_category_fallback=True
                )
                p2_tasks.append(
                    asyncio.ensure_future(
                        collect_google_serp(
                            cat_dorks,
                            token,
                            domain_terms=domain_terms,
                            require_brand_match=False,
                            category_queries=profile.category_problem_queries,
                        )
                    )
                )
            if "reddit" in selected_sources and token:
                reddit_cat = [q.replace(" OR issues", "") for q in cat_queries]
                p2_tasks.append(
                    asyncio.ensure_future(
                        collect_reddit(
                            reddit_cat,
                            actor_input.lookbackDays,
                            token,
                            max_posts_count=10,
                            domain_terms=domain_terms,
                            require_brand_match=False,
                            category_queries=profile.category_problem_queries,
                        )
                    )
                )

            if p2_tasks:
                p2_results = await asyncio.gather(*p2_tasks, return_exceptions=True)
                p2_items: list[RawComplaint] = []
                for res in p2_results:
                    if isinstance(res, Exception):
                        logger.warning("Pass 2 fallback collector error: %s", res)
                        continue
                    items, tel = res
                    p2_items.extend(items)
                    if tel.source in telemetry_map:
                        telemetry_map[tel.source].fallback_used = True
                        telemetry_map[tel.source].items_collected += len(items)

                final_complaints = deduplicate_complaints(pass1_deduped + p2_items)

        pass2_added = max(0, len(final_complaints) - pass1_count)
        return Stage2Result(
            complaints=final_complaints,
            telemetry=list(telemetry_map.values()),
            fallback_triggered=fallback_triggered,
            pass1_unique_count=pass1_count,
            pass2_added_count=pass2_added,
            total_unique_count=len(final_complaints),
        )
    finally:
        if owns_client:
            await client.aclose()