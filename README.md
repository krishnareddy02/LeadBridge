# LeadBridge — Competitor Complaint-to-Revenue Agent

**LeadBridge** turns live public complaints about your competitors into qualified, factually grounded buyer leads, ban-safe outreach drafts, and a single-file interactive Executive Dashboard (`REPORT.html`)—for **any product in any industry**.

## What LeadBridge Does (5-Stage Agentic Pipeline)

1. **Stage 1 — Product Auto-Profiler:** Crawls your homepage  and builds a strictly grounded **Verified Capabilities Ledger**, auto-discovers top direct competitors, and generates industry disambiguation anchors.
2. **Stage 2 — Parallel Multi-Source Harvester:** Searches **Google SERP** (Reddit, G2, Trustpilot, Capterra, HN), **Reddit** (fresh posts), **Hacker News**, **Bluesky** (with automatic SERP rescue), and optional **G2 Critical Reviews**. If Pass 1 returns fewer than 8 items, LeadBridge automatically triggers an agentic Pass 2 category-level fallback.
3. **Stage 3 — Deterministic Filter :** Removes exact/cross-post duplicates, own-brand mentions, promo/coupon/hiring spam, subreddit homepages, and stale posts before spending a single LLM token.
4. **Stage 4 — Batched Solvability Hard-Gate & Winability Scorer:** Evaluates complaints in batches of 15 and checks every complaint against your Verified Capabilities Ledger. Irrelevant complaints are rejected (`0/100`) and never billed as leads. Qualified leads are scored `0–100` (`Solvability /50 + Urgency /30 + Recency /20`).
5. **Stage 5 — Ban-Safe Dual Outreach & `REPORT.html`:** Generates **Draft A (Value-First Public Reply)** and **Draft B (Direct / DM Follow-Up)** backed strictly by cited product capabilities, attaches platform anti-ban rules, and compiles an interactive `REPORT.html` executive brief with one-click Copy buttons.

## Input

Provide just three required fields:
* **`productName`** — Your product name (automatically excluded from competitor results).
* **`productUrl`** — Your public website URL.
* **`productDescription`** — What your product does, pricing advantages, and core differentiators.

Optional fields: `competitors` (leave empty to auto-discover), `sources`, `lookbackDays` (default `14`), `minWinabilityScore` (default `70`), `tone` (`founder_candid`, `helpful_technical`, `executive_concise`), and optional BYOK `llmApiKey`.

## Output & CRM-Ready Exports

* **Actionable Outreach & CRM Export (Default View):** Clean spreadsheet-ready columns (`Priority`, `Score /100`, `Next Step`, `Competitor`, `Community`, `Customer Pain`, `Verified Proof`, `Draft A`, `Draft B`, `Anti-Ban Rule`, `Post Link`, `Date`, `Status`) — export directly to CSV, Excel, Google Sheets, Notion, or HubSpot with zero split array columns.
* **Competitor Pain & SEO Insights View:** Score breakdowns and pain themes for homepage copy and SEO comparison pages.
* **Interactive Executive Brief (`REPORT.html`):** Visual KPI funnel, competitor pain breakdown chart, hard-gate rejection proof, and one-click Copy Draft buttons.

## Compliance & Human-in-the-Loop Safety

LeadBridge **never auto-posts** to Reddit, Hacker News, Bluesky, or review platforms, and **never scrapes private/logged-in personal data**. All drafts are generated for human review before posting.