"""Comprehensive Stage 1 test suite covering Valid Cases & Failure/Edge Cases."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import ValidationError

from src.llm_router import LLMRouter, LLMRouterError
from src.profiler import (
    build_product_profile,
    discover_high_value_subpage,
    extract_clean_text_from_html,
    fetch_product_website_text,
    is_js_spa_shell,
)
from src.schemas import ActorInput, LLMProfileExtraction


# --- Shared Fixtures ---

SAMPLE_HOME_HTML = """
<!doctype html>
<html>
<head>
  <title>InvoiceSpark - Flat-Rate Invoicing for Freelancers</title>
  <meta name="description" content="Flat $15/mo invoicing with unlimited clients and 0% payment markup.">
</head>
<body>
  <nav>Home Login Signup</nav>
  <main>
    <h1>Stop Paying Per-Client Tiers for Simple Invoicing</h1>
    <p>InvoiceSpark gives freelancers and small agencies unlimited clients for a flat $15/mo.</p>
    <p>Connect your own Stripe, Wise, or UPI account with 0% platform transaction markup.</p>
    <p>Automated polite overdue payment reminders chase late invoices while you sleep.</p>
    <p>Migrate from FreshBooks or QuickBooks in 60 seconds using our 1-click CSV importer.</p>
    <a href="/pricing">View Pricing</a>
  </main>
  <footer>Copyright 2026 InvoiceSpark</footer>
</body>
</html>
"""

SAMPLE_PRICING_HTML = """
<!doctype html>
<html>
<head><title>InvoiceSpark Pricing</title></head>
<body>
  <h1>Simple Pricing</h1>
  <p>Pro Plan: $15/month flat. Unlimited active and archived clients. Unlimited invoices.</p>
  <p>No per-seat penalties. Cancel anytime. Includes 1-click FreshBooks CSV migration and Slack alerts.</p>
</body>
</html>
"""

SPA_SHELL_HTML = """
<!doctype html>
<html>
<head><title>App</title></head>
<body>
  <div id="root"></div>
  <noscript>You need to enable JavaScript to run this app.</noscript>
</body>
</html>
"""


def _mock_llm_extraction() -> LLMProfileExtraction:
    return LLMProfileExtraction(
        product_category="Freelancer Invoicing SaaS",
        target_audience="Freelancers and small creative agencies",
        verified_capabilities=[
            "Flat $15/mo pricing with unlimited active and archived clients",
            "0% platform transaction markup on Stripe, Wise, and UPI",
            "Automated polite overdue payment reminders",
            "1-click CSV migration from FreshBooks and QuickBooks",
        ],
        pain_points_solved=[
            "Forced pricing tier upgrades based on billable client caps",
            "High payment gateway percentage markups",
            "Manual awkward chasing of unpaid overdue invoices",
        ],
        # Intentionally include founder's own brand to verify guardrail strips it!
        competitors=["InvoiceSpark", "FreshBooks", "QuickBooks Online", "Wave", "Harvest"],
        competitor_pain_queries=[
            "FreshBooks alternative",
            "FreshBooks client limit upgrade",
            "QuickBooks pricing increase",
            "switching from Wave invoicing",
            "Harvest invoicing alternative",
        ],
        category_problem_queries=[
            "invoicing tool flat pricing unlimited clients",
            "automated overdue invoice reminders freelancer",
            "invoicing software no transaction markup",
        ],
        g2_product_slugs=["freshbooks", "quickbooks-online", "wave", "harvest"],
    )


# =====================================================================
# VALID CASES
# =====================================================================


@pytest.mark.asyncio
async def test_valid_case_1_direct_crawl_with_pricing_subpage_and_brand_guardrail():
    """Valid Case: Healthy HTML site + auto /pricing enrichment + own-brand exclusion."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/pricing":
            return httpx.Response(
                200, text=SAMPLE_PRICING_HTML, headers={"content-type": "text/html"}
            )
        return httpx.Response(
            200, text=SAMPLE_HOME_HTML, headers={"content-type": "text/html"}
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    mock_router = AsyncMock(spec=LLMRouter)
    mock_router.generate_structured.return_value = (
        _mock_llm_extraction(),
        "gemini:gemini-3.5-flash-lite",
    )

    actor_input = ActorInput(
        productName="InvoiceSpark",
        productUrl="invoicespark.io",  # tests automatic https:// prefixing
        productDescription="Flat $15/mo invoicing for freelancers with unlimited clients and automated overdue reminders.",
    )

    profile = await build_product_profile(
        actor_input=actor_input,
        llm_router=mock_router,
        http_client=mock_client,
    )
    await mock_client.aclose()

    assert profile.product_url == "https://invoicespark.io"
    assert profile.root_domain == "invoicespark.io"
    assert profile.crawl_diagnostics.crawl_method == "httpx_direct"
    assert len(profile.crawl_diagnostics.urls_fetched) == 2  # homepage + /pricing!
    assert "InvoiceSpark" not in profile.competitors  # Guardrail stripped self-mention
    assert "FreshBooks" in profile.competitors
    assert "InvoiceSpark" in profile.exclude_terms
    assert "invoicespark.io" in profile.exclude_terms
    assert "coupon" in profile.exclude_terms
    assert len(profile.verified_capabilities) >= 4
    assert len(profile.category_problem_queries) >= 2


@pytest.mark.asyncio
async def test_valid_case_2_founder_competitors_merged_first_and_deduped():
    """Valid Case: User-provided competitors take priority and deduplicate cleanly."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text=SAMPLE_HOME_HTML, headers={"content-type": "text/html"}
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    mock_router = AsyncMock(spec=LLMRouter)
    mock_router.generate_structured.return_value = (
        _mock_llm_extraction(),
        "gemini:gemini-3.5-flash-lite",
    )

    actor_input = ActorInput(
        productName="InvoiceSpark",
        productUrl="https://www.invoicespark.io",
        productDescription="Flat $15/mo invoicing for agencies with 1-click CSV migration.",
        competitors=["Zoho Invoice", "FreshBooks", "freshbooks"],  # duplicate case
    )

    profile = await build_product_profile(
        actor_input=actor_input,
        llm_router=mock_router,
        http_client=mock_client,
    )
    await mock_client.aclose()

    assert profile.root_domain == "invoicespark.io"
    assert profile.competitors[0] == "Zoho Invoice"
    assert profile.competitors[1] == "FreshBooks"
    assert len([c for c in profile.competitors if c.lower() == "freshbooks"]) == 1


# =====================================================================
# FAILURE & EDGE CASES
# =====================================================================


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1:8000/admin",
        "http://localhost/secret",
        "http://169.254.169.254/latest/meta-data/",
        "http://192.168.1.10",
        "ftp://example.com",
        "not-a-domain",
    ],
)
def test_failure_case_1_ssrf_and_invalid_urls_rejected(bad_url: str):
    """Failure Case: Private IPs, cloud metadata SSRF, and bad schemes are rejected."""
    with pytest.raises(ValidationError):
        ActorInput(
            productName="TestApp",
            productUrl=bad_url,
            productDescription="A valid product description with at least fifteen characters.",
        )


@pytest.mark.asyncio
async def test_failure_case_2_dead_url_404_degrades_gracefully_to_description():
    """Failure Case: HTTP 404 or timeout never crashes Stage 1; falls back to description."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Not Found", headers={"content-type": "text/html"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    text, diag = await fetch_product_website_text(
        product_url="https://stealth-unlaunched-app.example.com",
        enable_apify_fallback=False,
        http_client=mock_client,
    )
    await mock_client.aclose()

    assert text == ""
    assert diag.crawl_method == "description_only_fallback"
    assert diag.http_status == 404
    assert diag.warning is not None


@pytest.mark.asyncio
async def test_failure_case_3_js_spa_triggers_apify_crawler_fallback():
    """Failure Case: Empty React SPA shell triggers apify/website-content-crawler fallback."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=SPA_SHELL_HTML, headers={"content-type": "text/html"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with patch(
        "src.profiler.crawl_with_apify_fallback",
        new_callable=AsyncMock,
        return_value=(
            "Rendered React SaaS page with flat $19/mo pricing and instant Zapier webhooks. " * 5,
            ["https://spa-product.io"],
        ),
    ) as mock_apify_crawl:
        text, diag = await fetch_product_website_text(
            product_url="https://spa-product.io",
            enable_apify_fallback=True,
            http_client=mock_client,
        )

    await mock_client.aclose()
    mock_apify_crawl.assert_awaited_once()
    assert diag.crawl_method == "apify_wcc_fallback"
    assert diag.extracted_char_count >= 150
    assert "flat $19/mo pricing" in text


@pytest.mark.asyncio
async def test_failure_case_4_gemini_429_fails_over_to_groq_automatically():
    """Failure Case: Primary Gemini 429 rate-limit seamlessly fails over to Groq."""
    router = LLMRouter(
        gemini_api_key="fake-gemini-key",
        groq_api_key="fake-groq-key",
        openrouter_api_key=None,
    )
    expected = _mock_llm_extraction()

    with (
        patch.object(
            router,
            "_call_gemini",
            new_callable=AsyncMock,
            side_effect=RuntimeError("429 RESOURCE_EXHAUSTED"),
        ),
        patch.object(
            router,
            "_call_openai_compat",
            new_callable=AsyncMock,
            return_value=expected,
        ) as mock_groq,
    ):
        result, provider = await router.generate_structured(
            system_prompt="System",
            user_prompt="User",
            response_model=LLMProfileExtraction,
        )

    assert provider == "groq:openai/gpt-oss-20b"
    assert result.product_category == expected.product_category
    mock_groq.assert_awaited_once()


@pytest.mark.asyncio
async def test_failure_case_5_all_llms_fail_raises_clear_error():
    """Failure Case: When every provider fails, LLMRouterError lists all root causes."""
    router = LLMRouter(
        gemini_api_key="fake-gemini",
        groq_api_key="fake-groq",
        openrouter_api_key=None,
    )
    with (
        patch.object(router, "_call_gemini", side_effect=RuntimeError("Gemini 429")),
        patch.object(router, "_call_openai_compat", side_effect=RuntimeError("Groq 503")),
    ):
        with pytest.raises(LLMRouterError) as exc_info:
            await router.generate_structured(
                system_prompt="System",
                user_prompt="User",
                response_model=LLMProfileExtraction,
            )

    err_msg = str(exc_info.value)
    assert "gemini:gemini-3.5-flash-lite" in err_msg
    assert "groq:openai/gpt-oss-20b" in err_msg


def test_html_helpers_clean_and_detect_spa():
    """Unit check: clean text strips nav/footer and detects SPA shells accurately."""
    cleaned = extract_clean_text_from_html(SAMPLE_HOME_HTML)
    assert "Copyright 2026" not in cleaned
    assert "Flat $15/mo" in cleaned
    assert not is_js_spa_shell(SAMPLE_HOME_HTML, cleaned)
    assert is_js_spa_shell(SPA_SHELL_HTML, extract_clean_text_from_html(SPA_SHELL_HTML))
    assert discover_high_value_subpage("https://invoicespark.io", SAMPLE_HOME_HTML) == (
        "https://invoicespark.io/pricing"
    )