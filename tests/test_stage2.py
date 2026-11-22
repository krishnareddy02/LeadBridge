"""Stage 2 + Stage 1->2 integration tests covering Valid Cases & Failure Cases ($0 cost)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.collectors import (
    build_google_serp_dorks,
    harvest_complaints,
    normalize_url,
)
from src.schemas import ActorInput, CrawlDiagnostics, ProductProfile


def _sample_profile() -> ProductProfile:
    return ProductProfile(
        product_name="InvoiceSpark",
        product_url="https://invoicespark.io",
        root_domain="invoicespark.io",
        product_category="Freelance Invoicing SaaS",
        target_audience="Freelancers and small agencies",
        verified_capabilities=[
            "Flat $15/mo invoicing with unlimited clients",
            "Automated polite overdue payment reminders",
            "0% markup on Stripe/Wise/UPI",
            "1-click CSV import from FreshBooks/QuickBooks",
        ],
        pain_points_solved=[
            "Per-client tier price jumps",
            "Manual overdue invoice chasing",
        ],
        competitors=["FreshBooks", "QuickBooks Online", "Wave", "Render.com"],
        competitor_pain_queries=[
            "FreshBooks alternative",
            "QuickBooks pricing increase",
            "Wave invoicing issues",
        ],
        category_problem_queries=[
            "affordable invoicing software",
            "automated overdue invoice reminders",
        ],
        g2_product_slugs=["freshbooks", "quickbooks-online", "wave"],
        exclude_terms=["InvoiceSpark", "invoicespark.io", "coupon", "promo code"],
        llm_provider_used="groq:openai/gpt-oss-20b",
        crawl_diagnostics=CrawlDiagnostics(
            crawl_method="httpx_direct",
            urls_fetched=["https://invoicespark.io"],
            http_status=200,
            extracted_char_count=2000,
        ),
        domain_context_terms=["invoicing", "invoice", "billing", "clients", "payments"],
    )


def _mock_hn_payload(count: int = 3) -> dict:
    now_ts = int(datetime.now(timezone.utc).timestamp())
    return {
        "hits": [
            {
                "objectID": f"9000{i}",
                "story_title": f"FreshBooks pricing jump discussion #{i}",
                "comment_text": f"<p>I&#x27;m tired of FreshBooks bumping my tier after 50 clients! Need alternative #{i}.</p>",
                "author": f"hn_dev_{i}",
                "created_at_i": now_ts - 3600,
                "points": 12 + i,
            }
            for i in range(count)
        ]
    }


def _mock_bsky_payload(count: int = 2) -> dict:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "posts": [
            {
                "uri": f"at://did:plc:abc123/app.bsky.feed.post/rkey{i}",
                "author": {"handle": f"freelancer{i}.bsky.social"},
                "record": {
                    "text": f"Anyone migrating off QuickBooks Online? Their new pricing is ridiculous #{i}.",
                    "createdAt": now_iso,
                },
                "likeCount": 5,
                "replyCount": 2,
            }
            for i in range(count)
        ]
    }


def _mock_serp_pages() -> list[dict]:
    return [
        {
            "searchQuery": {"term": 'site:reddit.com ("FreshBooks")'},
            "organicResults": [
                {
                    "title": "FreshBooks alternative with unlimited clients?",
                    "url": "https://www.reddit.com/r/freelance/comments/abc123/freshbooks_alt/?utm_source=google",
                    "description": "FreshBooks bumped me up a tier for archived clients. Looking to switch ASAP.",
                    "date": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "title": "Wave invoicing stopped syncing my bank",
                    "url": "https://www.trustpilot.com/reviews/wave-bad-1",
                    "description": "Support took 2 weeks and still broken, looking for simple invoicing.",
                },
            ],
        }
    ]


def _mock_reddit_posts() -> list[dict]:
    return [
        # Duplicate URL of SERP item above (different query params) -> must deduplicate!
        {
            "dataType": "post",
            "postUrl": "https://www.reddit.com/r/freelance/comments/abc123/freshbooks_alt/",
            "title": "FreshBooks alternative with unlimited clients?",
            "body": "FreshBooks bumped me up a tier for archived clients. Looking to switch ASAP.",
            "communityName": "freelance",
            "authorName": "agency_owner",
            "upVotes": 42,
            "createdAt": datetime.now(timezone.utc).isoformat(),
        },
        # Unique Reddit post
        {
            "dataType": "post",
            "postUrl": "https://www.reddit.com/r/smallbusiness/comments/xyz999/qb_sucks/",
            "title": "QuickBooks Online raised prices again",
            "body": "Paying $90/mo just to send 10 invoices and chase overdue payments manually.",
            "communityName": "r/smallbusiness",
            "authorName": "solo_founder",
            "upVotes": 19,
            "createdAt": datetime.now(timezone.utc).isoformat(),
        },
        # Comment item that should be ignored by dataType filter
        {
            "dataType": "comment",
            "postUrl": "https://www.reddit.com/r/smallbusiness/comments/xyz999/c1",
            "title": "",
            "body": "Ignore nested comment objects",
        },
    ]


# =====================================================================
# VALID CASES
# =====================================================================


@pytest.mark.asyncio
async def test_valid_case_1_parallel_lean_run_normalization_and_dedup():
    """Valid Case: All 4 Lean collectors run in parallel, clean HTML/AT-URIs, and dedupe cross-source URLs."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "hn.algolia.com" in str(request.url):
            return httpx.Response(200, json=_mock_hn_payload(count=4))
        if "bsky.app" in str(request.url):
            return httpx.Response(200, json=_mock_bsky_payload(count=3))
        return httpx.Response(404)

    async def fake_apify_call(actor_id: str, run_input: dict, apify_token: str, **kwargs):
        if actor_id == "apify/google-search-scraper":
            return _mock_serp_pages()
        if actor_id == "harshmaur/reddit-scraper-pro":
            # Verify credit-safety flags are strictly False!
            assert run_input["searchComments"] is False
            assert run_input["crawlCommentsPerPost"] is False
            return _mock_reddit_posts()
        raise AssertionError(f"Unexpected actor_id: {actor_id}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    actor_input = ActorInput(
        productName="InvoiceSpark",
        productUrl="https://invoicespark.io",
        productDescription="Flat $15/mo invoicing for freelancers with unlimited clients.",
        sources=["google_serp", "reddit", "hacker_news", "bluesky"],
        lookbackDays=14,
    )

    with patch("src.collectors._run_apify_actor_and_fetch_items", side_effect=fake_apify_call):
        result = await harvest_complaints(
            profile=_sample_profile(),
            actor_input=actor_input,
            apify_token="fake-apify-token",
            http_client=client,
        )
    await client.aclose()

    # 4 HN + 3 Bsky + 2 SERP + 1 unique Reddit (1 Reddit URL deduped with SERP) = 10 unique
    assert result.total_unique_count == 10
    assert result.fallback_triggered is False

    # Verify HN HTML tags & entities stripped cleanly
    hn_items = [c for c in result.complaints if c.source == "hacker_news"]
    assert all("<p>" not in c.text and "&#x27;" not in c.text for c in hn_items)

    # Verify Bluesky AT URI converted into clickable https://bsky.app URL
    bsky_items = [c for c in result.complaints if c.source == "bluesky"]
    assert all(c.post_url.startswith("https://bsky.app/profile/") for c in bsky_items)

    # Verify duplicate reddit/SERP URL exists exactly once
    reddit_dup_matches = [
        c for c in result.complaints if "abc123/freshbooks_alt" in c.post_url
    ]
    assert len(reddit_dup_matches) == 1


@pytest.mark.asyncio
async def test_valid_case_2_deep_mode_g2_critical_reviews():
    """Valid Case: Deep Mode G2 collector passes maxRating=6 & publishedAfter and parses dislikes."""

    async def fake_apify_call(actor_id: str, run_input: dict, apify_token: str, **kwargs):
        assert actor_id == "automation-lab/g2-scraper"
        assert run_input["maxRating"] == 6
        assert "publishedAfter" in run_input
        return [
            {
                "reviewUrl": "https://www.g2.com/products/freshbooks/reviews/freshbooks-review-1",
                "title": "Way too expensive once you grow",
                "dislike": "Forced upgrades for archived clients and slow customer support.",
                "productName": "FreshBooks",
                "author": "Verified Small Biz",
                "publishedAt": datetime.now(timezone.utc).isoformat(),
            }
        ]

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})))
    actor_input = ActorInput(
        productName="InvoiceSpark",
        productUrl="https://invoicespark.io",
        productDescription="Flat $15/mo invoicing for freelancers with unlimited clients.",
        sources=["g2"],
        lookbackDays=14,
    )

    with patch("src.collectors._run_apify_actor_and_fetch_items", side_effect=fake_apify_call):
        result = await harvest_complaints(
            profile=_sample_profile(),
            actor_input=actor_input,
            apify_token="fake-apify-token",
            http_client=client,
            min_complaints_threshold=1,
        )
    await client.aclose()

    assert result.total_unique_count == 1
    assert result.complaints[0].source == "g2"
    assert "Forced upgrades" in result.complaints[0].text


@pytest.mark.asyncio
async def test_valid_case_3_agentic_pass2_fallback_triggers_when_low_volume():
    """Valid Case: Pass 1 returning < 8 items automatically triggers Pass 2 category search."""
    hn_call_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        q = request.url.params.get("query", "")
        hn_call_queries.append(q)
        now_ts = int(datetime.now(timezone.utc).timestamp())
        # Pass 1 competitor queries return only 1 item; Pass 2 category queries return 5 more
        if "invoicing" in q or "reminders" in q:
            return httpx.Response(
                200,
                json={
                    "hits": [
                        {
                            "objectID": f"cat_{abs(hash(q)) % 10000}_{i}",
                            "title": f"Looking for simple invoicing tool #{i}",
                            "comment_text": f"Tired of chasing overdue invoices manually every month #{i}.",
                            "author": "founder_x",
                            "created_at_i": now_ts - 1800,
                        }
                        for i in range(3)
                    ]
                },
            )
        return httpx.Response(200, json=_mock_hn_payload(count=1))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    actor_input = ActorInput(
        productName="InvoiceSpark",
        productUrl="https://invoicespark.io",
        productDescription="Flat $15/mo invoicing for freelancers with unlimited clients.",
        sources=["hacker_news"],
        lookbackDays=14,
    )

    result = await harvest_complaints(
        profile=_sample_profile(),
        actor_input=actor_input,
        apify_token="",
        http_client=client,
        min_complaints_threshold=8,
    )
    await client.aclose()

    assert result.fallback_triggered is True
    assert result.pass1_unique_count == 1
    assert result.pass2_added_count > 0
    assert result.total_unique_count > result.pass1_unique_count
    assert any("invoicing" in q for q in hn_call_queries)


# =====================================================================
# FAILURE & EDGE CASES
# =====================================================================


@pytest.mark.asyncio
async def test_failure_case_1_bluesky_403_triggers_automatic_serp_rescue():
    """Failure Case: Bluesky HTTP 403 does not crash and automatically rescues via site:bsky.app SERP."""
    serp_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "bsky.app" in str(request.url):
            return httpx.Response(403, text="Forbidden by CDN")
        return httpx.Response(200, json={})

    async def fake_apify_call(actor_id: str, run_input: dict, apify_token: str, **kwargs):
        serp_calls.append(run_input["queries"])
        if "site:bsky.app" in run_input["queries"]:
            return [
                {
                    "searchQuery": {"term": "site:bsky.app"},
                    "organicResults": [
                        {
                            "title": "Frustrated with FreshBooks pricing",
                            "url": "https://bsky.app/profile/dev.bsky.social/post/rescued1",
                            "description": "FreshBooks pricing jumped again, anyone have a flat-rate alternative?",
                        }
                    ],
                }
            ]
        return _mock_serp_pages()

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    actor_input = ActorInput(
        productName="InvoiceSpark",
        productUrl="https://invoicespark.io",
        productDescription="Flat $15/mo invoicing for freelancers with unlimited clients.",
        sources=["bluesky", "google_serp"],
        lookbackDays=14,
    )

    with patch("src.collectors._run_apify_actor_and_fetch_items", side_effect=fake_apify_call):
        result = await harvest_complaints(
            profile=_sample_profile(),
            actor_input=actor_input,
            apify_token="fake-token",
            http_client=client,
            min_complaints_threshold=1,
        )
    await client.aclose()

    assert any("site:bsky.app" in q for q in serp_calls)
    bsky_tel = next(t for t in result.telemetry if t.source == "bluesky")
    assert bsky_tel.fallback_used is True
    assert any(c.sub_source == "Bluesky (via SERP)" for c in result.complaints)


@pytest.mark.asyncio
async def test_failure_case_2_partial_collector_crash_isolated():
    """Failure Case: Reddit sub-Actor crashing never stops HN from succeeding."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "hn.algolia.com" in str(request.url):
            return httpx.Response(200, json=_mock_hn_payload(count=3))
        return httpx.Response(500)

    async def failing_reddit(*args, **kwargs):
        raise RuntimeError("Apify Reddit actor timed out (504)")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    actor_input = ActorInput(
        productName="InvoiceSpark",
        productUrl="https://invoicespark.io",
        productDescription="Flat $15/mo invoicing for freelancers with unlimited clients.",
        sources=["hacker_news", "reddit"],
        lookbackDays=14,
    )

    with patch("src.collectors._run_apify_actor_and_fetch_items", side_effect=failing_reddit):
        result = await harvest_complaints(
            profile=_sample_profile(),
            actor_input=actor_input,
            apify_token="fake-token",
            http_client=client,
            min_complaints_threshold=1,
        )
    await client.aclose()

    assert result.total_unique_count == 3
    reddit_tel = next(t for t in result.telemetry if t.source == "reddit")
    hn_tel = next(t for t in result.telemetry if t.source == "hacker_news")
    assert reddit_tel.status == "failed"
    assert "504" in (reddit_tel.error or "")
    assert hn_tel.status == "ok"


@pytest.mark.asyncio
async def test_failure_case_3_missing_apify_token_skips_paid_keeps_free():
    """Failure Case: Missing APIFY_TOKEN gracefully skips Apify collectors while $0 HN still works."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_mock_hn_payload(count=2))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    actor_input = ActorInput(
        productName="InvoiceSpark",
        productUrl="https://invoicespark.io",
        productDescription="Flat $15/mo invoicing for freelancers with unlimited clients.",
        sources=["hacker_news", "google_serp", "reddit", "g2"],
        lookbackDays=14,
    )

    result = await harvest_complaints(
        profile=_sample_profile(),
        actor_input=actor_input,
        apify_token="",  # Explicitly empty
        http_client=client,
        min_complaints_threshold=1,
    )
    await client.aclose()

    assert result.total_unique_count == 2
    by_src = {t.source: t for t in result.telemetry}
    assert by_src["hacker_news"].status == "ok"
    assert by_src["google_serp"].status == "skipped"
    assert by_src["reddit"].status == "skipped"
    assert by_src["g2"].status == "skipped"


def test_dork_builder_excludes_own_brand_and_strips_dot_com():
    """Unit Check: Dorks enforce own-brand exclusion, date filter, domain anchors, and clean '.com'."""
    dorks = build_google_serp_dorks(_sample_profile(), lookback_days=14)
    joined = "\n".join(dorks)
    assert len(dorks) == 3
    assert '-"InvoiceSpark" -site:invoicespark.io' in dorks[0]
    assert "after:" in dorks[0]
    assert '"Render"' in joined and "Render.com" not in joined
    assert '"invoicing"' in joined
    assert "(site:" not in joined  # Whitespace-clean :site operator for Apify Google SERP
    assert (
        normalize_url("https://Reddit.com/r/test/post1/?utm_source=x&fbclid=123")
        == "https://reddit.com/r/test/post1"
    )

def test_failure_case_5_off_topic_verb_and_reddit_junk_dropped():
    """Failure Case: Common-word 'render' verb & unrelated Reddit posts are dropped deterministically."""
    from src.collectors import _matches_competitor_and_domain

    comps = ["AWS Elastic Beanstalk", "Render", "Railway"]
    domain = ["hosting", "deploy", "paas", "cloud", "server", "backend", "postgres"]

    # Must REJECT: English verb "render" in graphics/email posts & Warhammer40k Reddit noise
    assert not _matches_competitor_and_domain(
        "My Houdini renderer is broken when trying to render frames.", comps, domain
    )
    assert not _matches_competitor_and_domain(
        "Mailchimp must render emails in every mail client.", comps, domain
    )
    assert not _matches_competitor_and_domain(
        "1k custodes army roster with 1000 point deployment limits.", comps, domain
    )

    # Must ACCEPT: Real Render hosting complaint & AWS Elastic Beanstalk complaint
    assert _matches_competitor_and_domain(
        "Looking for a free alternative to Render for backend hosting that doesn't spin down.",
        comps,
        domain,
    )
    assert _matches_competitor_and_domain(
        "Switching away from AWS Elastic Beanstalk because deployments keep breaking.",
        comps,
        domain,
    )

@pytest.mark.asyncio
async def test_failure_case_4_hn_algolia_fuzzy_typo_noise_filtered_out():
    """Failure Case: Algolia matching 'Railway'->'always' or 'Render'->'lenders' is dropped."""
    now_ts = int(datetime.now(timezone.utc).timestamp())

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "objectID": "noise_1",
                        "story_title": "NIN live archive",
                        "comment_text": "It is always a very impressive live experience.",
                        "author": "music_fan",
                        "created_at_i": now_ts - 600,
                    },
                    {
                        "objectID": "noise_2",
                        "story_title": "SpaceX lenders article",
                        "comment_text": "Lenders are concerned about collateral rendered useless.",
                        "author": "Rendello",
                        "created_at_i": now_ts - 600,
                    },
                    {
                        "objectID": "real_1",
                        "story_title": "Moving off Render after pricing hike",
                        "comment_text": "Render pricing jumped on our Postgres + web app deploy. Looking for an alternative.",
                        "author": "real_dev",
                        "created_at_i": now_ts - 600,
                    },
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    from src.collectors import collect_hacker_news

    items, tel = await collect_hacker_news(
        queries=["Render"],
        lookback_days=14,
        http_client=client,
        is_brand_query=True,
    )
    await client.aclose()

    assert tel.status == "ok"
    assert len(items) == 1
    assert items[0].author_handle == "real_dev"