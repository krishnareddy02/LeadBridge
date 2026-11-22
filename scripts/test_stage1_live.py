"""Live smoke test for Stage 1 against real URLs + real Gemini/Groq keys.

Run: python scripts/test_stage1_live.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Ensure repo root is importable when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.profiler import build_product_profile
from src.schemas import ActorInput

load_dotenv()


async def run_live_tests() -> None:
    test_cases = [
        # (
        #     "VALID LIVE CASE (Real SaaS URL + Auto-Discover Competitors)",
        #     ActorInput(
        #         productName="Linear",
        #         productUrl="https://linear.app",
        #         productDescription=(
        #             "High-speed issue tracking and project planning built for modern software teams. "
        #             "Keyboard-first UI, sub-50ms interactions, cycles, roadmaps, and native GitHub/GitLab sync."
        #         ),
        #         competitors=[],
        #     ),
        # ),
        (
            "VALID LIVE CASE (Real SaaS URL + Auto-Discover Competitors)",
            ActorInput(
                productName="Heroku",
                productUrl="https://www.heroku.com",
                productDescription=(
                    "Heroku is a fully managed cloud Platform as a Service (PaaS) "
            "that enables developers, startups, and enterprises to build, "
            "deploy, manage, and scale modern applications without managing "
            "underlying infrastructure. It provides container-based "
            "application hosting through dynos, supports programming "
            "languages including Python, Node.js, Ruby, Java, PHP, Go, "
            "and .NET, and offers managed data services such as Heroku "
            "Postgres, Heroku Key-Value Store, and Apache Kafka. "
            "Key capabilities include Git and GitHub-based deployments, "
            "buildpacks, Heroku Pipelines, Review Apps, application "
            "monitoring, automated scaling, deployment rollbacks, "
            "third-party add-ons, and enterprise collaboration features. "
            "Heroku also provides AI application infrastructure through "
            "Managed Inference and Agents, enabling developers to build "
            "and deploy AI-powered applications. Its primary value "
            "proposition is simplifying application deployment and "
            "infrastructure management while improving developer productivity."
                ),
                competitors=[],
            ),
        ),
        (
            "FAILURE LIVE CASE (Non-existent domain -> Description-Only Graceful Fallback)",
            ActorInput(
                productName="InvoiceSpark",
                productUrl="https://this-domain-does-not-exist-99281.io",
                productDescription=(
                    "Flat $15/mo invoicing for freelancers and small agencies. Unlimited clients, "
                    "automated polite overdue payment reminders, 0% markup on Stripe/Wise/UPI, "
                    "and 1-click CSV import from FreshBooks/QuickBooks."
                ),
                competitors=[],
            ),
        ),
    ]

    for label, actor_input in test_cases:
        print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
        profile = await build_product_profile(
            actor_input=actor_input,
            enable_apify_fallback=False,  # Set True if you want to test Apify WCC on SPAs
        )
        print(json.dumps(profile.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    asyncio.run(run_live_tests())