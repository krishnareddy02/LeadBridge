"""Stage 1: Founder Input & Product Auto-Profiler.

Crawls the founder's product URL (direct httpx + BeautifulSoup with automatic
pricing/features sub-page enrichment, falling back conditionally to
apify/website-content-crawler for JS-rendered SPAs, or gracefully to
productDescription if the site is unreachable), then extracts a strictly
grounded ProductProfile via LLMRouter.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from src.llm_router import LLMRouter
from src.schemas import (
    ActorInput,
    CrawlDiagnostics,
    LLMProfileExtraction,
    ProductProfile,
)

import inspect
from datetime import timedelta
logger = logging.getLogger(__name__)

MAX_WEBSITE_CHARS = 4500
MIN_MEANINGFUL_CHARS = 300

DEFAULT_SPAM_EXCLUSIONS: tuple[str, ...] = (
    "coupon",
    "promo code",
    "discount code",
    "black friday",
    "cyber monday",
    "affiliate",
    "we are hiring",
    "job opening",
    "for sale",
    "referral link",
)

BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

PROFILER_SYSTEM_PROMPT = """You are a senior B2B competitive intelligence analyst.
Your job is to create a STRICTLY GROUNDED ProductProfile for a founder's software/service product.

NON-NEGOTIABLE GROUNDING RULES:
1. verified_capabilities: List 5-8 CONCRETE, FACTUAL capabilities, pricing models, integrations, or workflow advantages EXPLICITLY supported by the provided Website Text or Founder Description.
   - NEVER invent features, compliance certs (SOC2/HIPAA), pricing, or integrations not present in the source text.
   - Be specific (e.g., "Flat $15/mo pricing with unlimited clients" instead of "Affordable pricing").
2. pain_points_solved: List 4-6 specific customer frustrations directly solved by those verified capabilities.
3. competitors: Identify 4-6 real, direct competitors users complain about and switch away from.
   - If the founder supplied known competitors, ALWAYS keep them first and add missing direct competitors up to 6 total.
   - NEVER include the founder's own product name in competitors.
4. competitor_pain_queries: Generate 5 short, realistic search queries (2-5 words each) that real frustrated users post on Reddit/HN/forums when looking to leave those competitors (e.g., "FreshBooks alternative", "QuickBooks pricing increase", "switching from Wave").
5. category_problem_queries: Generate 3 brand-agnostic problem queries (3-6 words) describing the core pain without competitor names (used as an agentic fallback if competitor volume is low).
6. g2_product_slugs: Provide lowercase hyphenated G2 review slugs matching the top competitors (e.g., "freshbooks", "quickbooks-online").
7. domain_context_terms: Provide 5-8 single-word lowercase industry nouns/verbs specific to this product's market (e.g., for cloud hosting: ["hosting", "deploy", "paas", "server", "backend", "postgres", "cloud"]; for invoicing: ["invoicing", "invoice", "billing", "accounting", "clients", "payments"]). Avoid ultra-generic words like "app", "software", "tool", or "platform".
"""


def extract_clean_text_from_html(html: str) -> str:
    """Strip HTML boilerplate and extract title, meta description, and readable copy."""
    soup = BeautifulSoup(html, "lxml")

    parts: list[str] = []
    if soup.title and soup.title.string:
        parts.append(f"Title: {soup.title.string.strip()}")

    meta_desc = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if meta_desc and meta_desc.get("content"):
        parts.append(f"Meta Description: {str(meta_desc['content']).strip()}")

    # Remove non-content elements
    for tag in soup(
        ["script", "style", "noscript", "svg", "iframe", "canvas", "footer", "nav", "aside"]
    ):
        tag.decompose()

    body = soup.body or soup
    raw_lines = body.get_text(separator="\n", strip=True).splitlines()

    # Deduplicate consecutive identical menu/CTA lines and skip 1-char artifacts
    cleaned_lines: list[str] = []
    prev = ""
    for line in raw_lines:
        compact = re.sub(r"\s+", " ", line).strip()
        if len(compact) < 2 or compact == prev:
            continue
        cleaned_lines.append(compact)
        prev = compact

    parts.append("\n".join(cleaned_lines))
    return "\n\n".join(p for p in parts if p).strip()


def is_js_spa_shell(html: str, cleaned_text: str) -> bool:
    """Detect client-side rendered SPA shells that require headless browser crawling."""
    if len(cleaned_text) < MIN_MEANINGFUL_CHARS:
        return True
    lowered = cleaned_text.lower()
    spa_markers = (
        "you need to enable javascript to run this app",
        "please enable javascript",
        "javascript is required",
        "checking your browser before accessing",
        "just a moment...",
    )
    if any(marker in lowered for marker in spa_markers) and len(cleaned_text) < 800:
        return True
    # Detect empty root containers with negligible body text
    if re.search(r'<div\s+id=["\'](root|app|__next)["\']\s*>\s*</div>', html, re.I):
        if len(cleaned_text) < 500:
            return True
    return False


def discover_high_value_subpage(base_url: str, html: str) -> str | None:
    """Find one same-origin /pricing or /features page to enrich factual capabilities."""
    soup = BeautifulSoup(html, "lxml")
    base_host = (urlparse(base_url).hostname or "").lower()
    priority_patterns = (
        re.compile(r"/(pricing|plans)\b", re.I),
        re.compile(r"/(features|product|why-us|compare)\b", re.I),
    )

    candidates: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if parsed.scheme not in ("http", "https"):
            continue
        if (parsed.hostname or "").lower() != base_host:
            continue
        if full_url.rstrip("/") == base_url.rstrip("/"):
            continue
        candidates.append(full_url)

    for pattern in priority_patterns:
        for url in candidates:
            if pattern.search(urlparse(url).path):
                return url
    return None


def _get_dataset_id(run_obj: Any) -> str | None:
    """SDK-version-safe helper supporting both apify-client v2 dicts and v3 objects."""
    if run_obj is None:
        return None
    if isinstance(run_obj, dict):
        return run_obj.get("defaultDatasetId") or run_obj.get("default_dataset_id")
    return getattr(run_obj, "default_dataset_id", None) or getattr(
        run_obj, "defaultDatasetId", None
    )


# async def crawl_with_apify_fallback(url: str) -> tuple[str, list[str]]:
#     """Fallback crawler for JS-rendered SPAs using apify/website-content-crawler."""
#     token = os.getenv("APIFY_TOKEN")
#     if not token:
#         raise RuntimeError("APIFY_TOKEN not set; cannot invoke apify/website-content-crawler fallback")

#     from apify_client import ApifyClientAsync

#     client = ApifyClientAsync(token=token)
#     run = await client.actor("apify/website-content-crawler").call(
#         run_input={
#             "startUrls": [{"url": url}],
#             "maxCrawlPages": 2,
#             "maxCrawlDepth": 1,
#             "crawlerType": "playwright:adaptive",
#         },
#         timeout_secs=90,
#     )
#     dataset_id = _get_dataset_id(run)
#     if not dataset_id:
#         raise RuntimeError("apify/website-content-crawler finished without a dataset ID")

#     chunks: list[str] = []
#     fetched_urls: list[str] = []
#     dataset_client = client.dataset(dataset_id)
#     list_page = await dataset_client.list_items(limit=5)
#     items = getattr(list_page, "items", None) or (
#         list_page.get("items", []) if isinstance(list_page, dict) else []
#     )
#     for item in items:
#         page_url = item.get("url")
#         text = (item.get("text") or item.get("markdown") or "").strip()
#         if page_url:
#             fetched_urls.append(page_url)
#         if text:
#             chunks.append(text)

#     combined = "\n\n---\n\n".join(chunks)[:MAX_WEBSITE_CHARS]
#     return combined, fetched_urls

async def crawl_with_apify_fallback(url: str) -> tuple[str, list[str]]:
    """Fallback crawler for JS-rendered SPAs using apify/website-content-crawler."""
    token = os.getenv("APIFY_TOKEN")
    if not token:
        raise RuntimeError("APIFY_TOKEN not set; cannot invoke apify/website-content-crawler fallback")

    from apify_client import ApifyClientAsync

    client = ApifyClientAsync(token=token)
    actor_client = client.actor("apify/website-content-crawler")
    call_params = inspect.signature(actor_client.call).parameters
    timeout_kw: dict[str, Any] = {}
    if "timeout" in call_params:
        timeout_kw["timeout"] = timedelta(seconds=90)
    elif "timeout_secs" in call_params:
        timeout_kw["timeout_secs"] = 90

    run = await actor_client.call(
        run_input={
            "startUrls": [{"url": url}],
            "maxCrawlPages": 2,
            "maxCrawlDepth": 1,
            "crawlerType": "playwright:adaptive",
        },
        **timeout_kw,
    )
    dataset_id = _get_dataset_id(run)
    if not dataset_id:
        raise RuntimeError("apify/website-content-crawler finished without a dataset ID")

    chunks: list[str] = []
    fetched_urls: list[str] = []
    dataset_client = client.dataset(dataset_id)
    list_page = await dataset_client.list_items(limit=5)
    items = getattr(list_page, "items", None) or (
        list_page.get("items", []) if isinstance(list_page, dict) else []
    )
    for item in items:
        page_url = item.get("url")
        text = (item.get("text") or item.get("markdown") or "").strip()
        if page_url:
            fetched_urls.append(page_url)
        if text:
            chunks.append(text)

    combined = "\n\n---\n\n".join(chunks)[:MAX_WEBSITE_CHARS]
    return combined, fetched_urls

async def fetch_product_website_text(
    product_url: str,
    enable_apify_fallback: bool = True,
    http_client: httpx.AsyncClient | None = None,
) -> tuple[str, CrawlDiagnostics]:
    """Fetch product copy via fast httpx + BS4, falling back cleanly on SPA or network failure."""
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=5.0),
        follow_redirects=True,
        headers=BROWSER_HEADERS,
    )

    urls_fetched: list[str] = []
    status_code: int | None = None
    direct_text = ""
    needs_spa_fallback = False
    failure_reason: str | None = None

    try:
        resp = await client.get(product_url)
        status_code = resp.status_code
        if resp.status_code >= 400:
            failure_reason = f"Direct HTTP fetch returned status {resp.status_code}"
            needs_spa_fallback = resp.status_code in (403, 503)
        else:
            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                failure_reason = f"Unsupported content-type: {content_type}"
            else:
                html = resp.text
                urls_fetched.append(str(resp.url))
                homepage_text = extract_clean_text_from_html(html)

                if is_js_spa_shell(html, homepage_text):
                    failure_reason = (
                        f"Page appears JS-rendered or thin ({len(homepage_text)} chars)"
                    )
                    needs_spa_fallback = True
                else:
                    direct_text = homepage_text
                    # Optional $0 enrichment: grab /pricing or /features if linked
                    subpage_url = discover_high_value_subpage(str(resp.url), html)
                    if subpage_url:
                        try:
                            sub_resp = await client.get(subpage_url)
                            if sub_resp.status_code == 200 and "text/html" in sub_resp.headers.get(
                                "content-type", ""
                            ).lower():
                                sub_text = extract_clean_text_from_html(sub_resp.text)
                                if len(sub_text) >= 150:
                                    urls_fetched.append(str(sub_resp.url))
                                    direct_text = (
                                        f"{homepage_text[:2800]}\n\n"
                                        f"--- SUBPAGE ({sub_resp.url}) ---\n"
                                        f"{sub_text[:1700]}"
                                    )
                        except Exception as sub_exc:
                            logger.debug("Non-fatal subpage fetch error (%s): %s", subpage_url, sub_exc)

                    direct_text = direct_text[:MAX_WEBSITE_CHARS]
                    return direct_text, CrawlDiagnostics(
                        crawl_method="httpx_direct",
                        urls_fetched=urls_fetched,
                        http_status=status_code,
                        extracted_char_count=len(direct_text),
                    )
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        failure_reason = f"Direct HTTP request error: {type(exc).__name__}: {exc}"
        logger.warning(failure_reason)
    finally:
        if owns_client:
            await client.aclose()

    # Conditional Apify SPA Fallback
    if needs_spa_fallback and enable_apify_fallback:
        logger.info("Triggering apify/website-content-crawler fallback for %s", product_url)
        try:
            fallback_text, fallback_urls = await crawl_with_apify_fallback(product_url)
            if len(fallback_text) >= 150:
                return fallback_text, CrawlDiagnostics(
                    crawl_method="apify_wcc_fallback",
                    urls_fetched=fallback_urls or [product_url],
                    http_status=status_code,
                    extracted_char_count=len(fallback_text),
                    warning=failure_reason,
                )
            failure_reason = f"{failure_reason}; Apify fallback returned thin content"
        except Exception as fb_exc:
            failure_reason = f"{failure_reason}; Apify fallback error: {fb_exc}"
            logger.warning(failure_reason)

    # Ultimate Graceful Degradation: proceed with Founder Description alone
    return "", CrawlDiagnostics(
        crawl_method="description_only_fallback",
        urls_fetched=urls_fetched,
        http_status=status_code,
        extracted_char_count=0,
        warning=failure_reason or "Website text unavailable; grounded using productDescription only.",
    )


def _slugify_g2(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug


def _dedupe_preserve_order(items: list[str], exclude_lower: set[str] | None = None) -> list[str]:
    blocked = exclude_lower or set()
    seen: set[str] = set()
    output: list[str] = []
    for raw in items:
        cleaned = re.sub(r"\s+", " ", str(raw)).strip()
        key = cleaned.lower()
        if not cleaned or key in seen or key in blocked:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


async def build_product_profile(
    actor_input: ActorInput,
    llm_router: LLMRouter | None = None,
    enable_apify_fallback: bool = True,
    http_client: httpx.AsyncClient | None = None,
) -> ProductProfile:
    """Execute Stage 1 end-to-end and return a strictly grounded ProductProfile."""
    website_text, diagnostics = await fetch_product_website_text(
        product_url=actor_input.productUrl,
        enable_apify_fallback=enable_apify_fallback,
        http_client=http_client,
    )

    router = llm_router or LLMRouter(gemini_api_key=actor_input.llmApiKey)

    known_comp_str = (
        ", ".join(actor_input.competitors)
        if actor_input.competitors
        else "(None supplied — auto-discover top 4-6 direct competitors)"
    )

    user_prompt = (
        f"Product Name: {actor_input.productName}\n"
        f"Product URL: {actor_input.productUrl}\n"
        f"Founder Description:\n{actor_input.productDescription}\n\n"
        f"Founder-Supplied Competitors: {known_comp_str}\n\n"
        f"Extracted Website Text ({diagnostics.crawl_method}):\n"
        f"{website_text if website_text else '[No website text available — rely strictly on Founder Description above]'}"
    )

    extracted, provider_used = await router.generate_structured(
        system_prompt=PROFILER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=LLMProfileExtraction,
        temperature=0.15,
    )

    # --- Deterministic Guardrails (Guaranteed invariants regardless of LLM quirks) ---
    own_brand_tokens = {
        actor_input.productName.lower(),
        actor_input.root_domain.lower(),
        actor_input.root_domain.split(".")[0].lower(),
    }

    # Merge user-supplied competitors first, then LLM-discovered, excluding own brand
    merged_competitors = _dedupe_preserve_order(
        actor_input.competitors + extracted.competitors,
        exclude_lower=own_brand_tokens,
    )[:6]

    # Ensure G2 slugs cover our top competitors cleanly
    raw_slugs = extracted.g2_product_slugs or [_slugify_g2(c) for c in merged_competitors]
    clean_slugs = _dedupe_preserve_order(
        [_slugify_g2(s) for s in raw_slugs if _slugify_g2(s)]
    )[:6]

    # Always guarantee founder brand + domain + spam triggers are in exclude_terms
    exclude_terms = _dedupe_preserve_order(
        [actor_input.productName, actor_input.root_domain, *DEFAULT_SPAM_EXCLUSIONS]
    )

    # pain_queries = _dedupe_preserve_order(extracted.competitor_pain_queries)[:6]
    # category_queries = _dedupe_preserve_order(extracted.category_problem_queries)[:4]

    def _contains_own_brand(text: str) -> bool:
        low = text.lower()
        return any(
            re.search(rf"\b{re.escape(tok)}\b", low)
            for tok in own_brand_tokens
            if len(tok) >= 3
        )

    raw_pain_queries = [
        q for q in extracted.competitor_pain_queries if not _contains_own_brand(q)
    ]
    # Backfill clean competitor queries if LLM generated an own-brand query
    for comp in merged_competitors:
        if len(raw_pain_queries) >= 5:
            break
        raw_pain_queries.append(f"{comp} alternative")

    pain_queries = _dedupe_preserve_order(raw_pain_queries)[:6]
    category_queries = _dedupe_preserve_order(
        [q for q in extracted.category_problem_queries if not _contains_own_brand(q)]
    )[:4]
    generic_stopwords = {"app", "apps", "software", "tool", "tools", "platform", "service", "system"}
    raw_domain_terms = [
        re.sub(r"[^a-z0-9]", "", t.lower())
        for t in extracted.domain_context_terms
        if len(t.strip().split()) == 1
    ]
    clean_domain_terms = _dedupe_preserve_order(
        [t for t in raw_domain_terms if len(t) >= 3 and t not in generic_stopwords]
    )[:8]
    return ProductProfile(
        product_name=actor_input.productName,
        product_url=actor_input.productUrl,
        root_domain=actor_input.root_domain,
        product_category=extracted.product_category.strip(),
        target_audience=extracted.target_audience.strip(),
        verified_capabilities=_dedupe_preserve_order(extracted.verified_capabilities)[:8],
        pain_points_solved=_dedupe_preserve_order(extracted.pain_points_solved)[:6],
        competitors=merged_competitors,
        competitor_pain_queries=pain_queries,
        category_problem_queries=category_queries,
        g2_product_slugs=clean_slugs,
        exclude_terms=exclude_terms,
        llm_provider_used=provider_used,
        crawl_diagnostics=diagnostics,
        domain_context_terms=clean_domain_terms
    )