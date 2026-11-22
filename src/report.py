"""Stage 5C: Self-Contained HTML Executive Brief Generator (REPORT.html).

Compiles a single-file, zero-CDN, XSS-safe executive dashboard saved to Apify
Key-Value Store (REPORT.html) featuring:
  - Pipeline Funnel KPIs (Raw -> Clean -> Hot Leads -> Warm Signals -> Gaps)
  - Competitor Pain Breakdown horizontal bar chart
  - Hot Lead Action Cards with one-click Copy Draft A & Copy Draft B buttons
  - Platform Anti-Ban Guidance & Exact Verified Capability Evidence pills
  - Hard-Gate Noise Rejection Proof (demonstrating false-positive filtering)
  - Human-in-the-loop & GDPR privacy compliance footer
"""

from __future__ import annotations

import html
import json
from collections import Counter
from datetime import datetime, timezone

from src.schemas import (
    ActorInput,
    CollectorTelemetry,
    DatasetLeadRecord,
    ProductProfile,
    Stage3Result,
    Stage4Result,
    Stage5Result,
)


def _esc(text: object) -> str:
    return html.escape(str(text or ""), quote=True)


def _safe_href(url: str) -> str:
    cleaned = (url or "").strip()
    if cleaned.lower().startswith(("http://", "https://")):
        return html.escape(cleaned, quote=True)
    return "#"


def _build_competitor_breakdown_html(
    hot_leads: list[DatasetLeadRecord],
    warm_signals: list[DatasetLeadRecord],
    market_gaps: list[DatasetLeadRecord],
) -> str:
    all_pain = hot_leads + warm_signals + market_gaps
    if not all_pain:
        return '<p class="muted">No competitor complaints qualified in this lookback window.</p>'

    comp_counts = Counter(
        item.competitorMentioned for item in all_pain if item.competitorMentioned and item.competitorMentioned != "None"
    )
    cat_counts = Counter(item.painCategory.replace("_", " ").title() for item in all_pain)

    max_comp = max(comp_counts.values(), default=1)
    comp_bars = []
    for comp, count in comp_counts.most_common(6):
        pct = max(12, int((count / max_comp) * 100))
        comp_bars.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{_esc(comp)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
              <div class="bar-val">{count}</div>
            </div>
            """
        )

    cat_pills = "".join(
        f'<span class="pill pill-amber">{_esc(cat)} <strong>({cnt})</strong></span>'
        for cat, cnt in cat_counts.most_common(6)
    )

    return f"""
    <div class="grid-2">
      <div>
        <h3 class="sub-h">Complaints by Competitor</h3>
        {''.join(comp_bars)}
      </div>
      <div>
        <h3 class="sub-h">Top Pain Themes</h3>
        <div class="pill-wrap">{cat_pills}</div>
      </div>
    </div>
    """


def _build_hot_lead_cards_html(hot_leads: list[DatasetLeadRecord]) -> str:
    if not hot_leads:
        return (
            '<div class="card empty-state">'
            "No complaints crossed the Hot Lead threshold in this lookback window. "
            "Check Warm / SEO Signals below or increase <code>lookbackDays</code>."
            "</div>"
        )

    cards: list[str] = []
    for idx, lead in enumerate(hot_leads, start=1):
        caps_html = "".join(
            f'<span class="pill pill-teal">&#10003; {_esc(c)}</span>'
            for c in lead.matchedCapabilities
        )
        draft_a_id = f"draft-a-{idx}"
        draft_b_id = f"draft-b-{idx}"

        cards.append(
            f"""
            <article class="card lead-card">
              <div class="lead-header">
                <div class="lead-meta">
                  <span class="score-badge">{lead.winabilityScore}<span>/100</span></span>
                  <div>
                    <div class="lead-title">#{idx} &middot; {_esc(lead.competitorMentioned)} &mdash; {_esc(lead.painCategory.replace('_', ' ').title())}</div>
                    <div class="lead-sub">
                      <strong>{_esc(lead.source.upper())}</strong> ({_esc(lead.subSource)}) &middot;
                      {_esc(lead.authorHandle)} &middot;
                      Solvability {lead.solvabilityScore}/50 &middot; Urgency {lead.urgencyScore}/30 &middot; Recency {lead.recencyScore}/20
                    </div>
                  </div>
                </div>
                <a class="btn-link" href="{_safe_href(lead.postUrl)}" target="_blank" rel="noopener noreferrer">
                  Open Original Post &#8599;
                </a>
              </div>

              <p class="pain-summary"><strong>Customer Pain:</strong> {_esc(lead.painSummary)}</p>

              <div class="section-mini">
                <div class="mini-label">Verified Capability Evidence (Zero-Hallucination Grounding)</div>
                <div class="pill-wrap">{caps_html}</div>
              </div>

              <div class="safety-box">
                <strong>Platform Anti-Ban Guidance:</strong> {_esc(lead.platformSafetyRule)}
              </div>

              <div class="drafts-grid">
                <div class="draft-box">
                  <div class="draft-top">
                    <span class="draft-label">Draft A &mdash; Value-First Public Reply</span>
                    <button class="copy-btn" type="button" onclick="copyText('{draft_a_id}', this)">Copy Draft A</button>
                  </div>
                  <pre id="{draft_a_id}" class="draft-text">{_esc(lead.draftPublicReply)}</pre>
                </div>

                <div class="draft-box">
                  <div class="draft-top">
                    <span class="draft-label">Draft B &mdash; Direct / DM Follow-Up</span>
                    <button class="copy-btn" type="button" onclick="copyText('{draft_b_id}', this)">Copy Draft B</button>
                  </div>
                  <pre id="{draft_b_id}" class="draft-text">{_esc(lead.draftDirectPitch)}</pre>
                </div>
              </div>
            </article>
            """
        )

    return "\n".join(cards)


def _build_warm_table_html(warm_signals: list[DatasetLeadRecord]) -> str:
    if not warm_signals:
        return '<p class="muted">No Warm / SEO signals scored between 40 and the Hot Lead threshold.</p>'

    rows = []
    for item in warm_signals:
        rows.append(
            f"""
            <tr>
              <td><span class="pill pill-purple">{item.winabilityScore}/100</span></td>
              <td>{_esc(item.source)} / {_esc(item.subSource)}</td>
              <td><strong>{_esc(item.competitorMentioned)}</strong></td>
              <td>{_esc(item.painSummary)}</td>
              <td><a href="{_safe_href(item.postUrl)}" target="_blank" rel="noopener noreferrer">View &#8599;</a></td>
            </tr>
            """
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Score</th><th>Source</th><th>Competitor</th><th>Pain / Signal</th><th>Link</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _build_rejected_proof_html(stage4: Stage4Result) -> str:
    sample = stage4.discarded[:8]
    if not sample:
        return '<p class="muted">No items were rejected by the Solvability Gate.</p>'

    rows = []
    for item in sample:
        rows.append(
            f"""
            <tr>
              <td>{_esc(item.complaint.source)} / {_esc(item.complaint.sub_source)}</td>
              <td>{_esc(item.complaint.title or item.complaint.text[:80])}</td>
              <td><span class="pill pill-coral">Rejected (0/100)</span></td>
              <td>{_esc(item.rejection_or_gap_reason)}</td>
            </tr>
            """
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Source</th><th>Post</th><th>Gate Verdict</th><th>Why Rejected ($0 Lead Charge)</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_executive_report_html(
    profile: ProductProfile,
    actor_input: ActorInput,
    stage2_telemetry: list[CollectorTelemetry],
    stage3: Stage3Result,
    stage4: Stage4Result,
    stage5: Stage5Result,
) -> str:
    """Compile the complete single-file HTML Executive Brief."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    raw_total = stage3.input_count

    source_badges = "".join(
        f'<span class="pill pill-neutral">{_esc(t.source)}: <strong>{t.items_collected}</strong> ({_esc(t.status)})</span>'
        for t in stage2_telemetry
    )

    ledger_items = "".join(
        f"<li>{_esc(cap)}</li>" for cap in profile.verified_capabilities
    )

    hot_cards_html = _build_hot_lead_cards_html(stage5.hot_lead_records)
    breakdown_html = _build_competitor_breakdown_html(
        stage5.hot_lead_records, stage5.warm_signal_records, stage5.market_gap_records
    )
    warm_html = _build_warm_table_html(stage5.warm_signal_records)
    rejected_html = _build_rejected_proof_html(stage4)

    # Safe JSON metadata for debugging / copy tooling
    meta_json = json.dumps(
        {
            "product": profile.product_name,
            "hotLeads": len(stage5.hot_lead_records),
            "warmSignals": len(stage5.warm_signal_records),
            "generatedAt": generated_at,
        }
    ).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Complaint-to-Revenue Executive Brief &mdash; {_esc(profile.product_name)}</title>
  <style>
    :root {{
      --bg: #f8f7f4;
      --surface: #ffffff;
      --ink: #1e1b2e;
      --muted: #5c586e;
      --border: #e6e2ee;
      --purple: #3b1d6c;
      --teal: #0f766e;
      --teal-soft: #e6f6f4;
      --coral: #e05a47;
      --coral-soft: #fdecea;
      --amber: #b45309;
      --amber-soft: #fef3c7;
      --purple-soft: #efe9fa;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.55;
    }}
    .container {{ max-width: 1120px; margin: 0 auto; padding: 28px 20px 56px; }}
    .hero {{
      background: linear-gradient(135deg, var(--purple) 0%, #241046 100%);
      color: #fff;
      border-radius: 16px;
      padding: 28px 32px;
      margin-bottom: 24px;
    }}
    .hero-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }}
    .hero h1 {{ margin: 0 0 6px; font-size: 26px; letter-spacing: -0.02em; }}
    .hero p {{ margin: 0; color: rgba(255,255,255,0.82); font-size: 14px; }}
    .badge-live {{
      background: rgba(15, 118, 110, 0.25);
      border: 1px solid rgba(94, 234, 212, 0.45);
      color: #99f6e4;
      padding: 5px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }}
    .kpi {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px 18px;
    }}
    .kpi-num {{ font-size: 28px; font-weight: 800; color: var(--purple); line-height: 1.1; }}
    .kpi-num.teal {{ color: var(--teal); }}
    .kpi-num.coral {{ color: var(--coral); }}
    .kpi-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-top: 4px; }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 22px 24px;
      margin-bottom: 18px;
      box-shadow: 0 2px 8px rgba(30, 27, 46, 0.04);
    }}
    h2 {{ font-size: 19px; margin: 28px 0 12px; color: var(--purple); }}
    .sub-h {{ font-size: 14px; margin: 0 0 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 22px; }}
    .bar-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 13px; }}
    .bar-label {{ width: 150px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .bar-track {{ flex: 1; height: 10px; background: #eeeaf5; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: linear-gradient(90deg, var(--teal), var(--purple)); border-radius: 999px; }}
    .bar-val {{ width: 24px; text-align: right; font-weight: 700; color: var(--purple); }}
    .pill-wrap {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .pill {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }}
    .pill-teal {{ background: var(--teal-soft); color: var(--teal); }}
    .pill-coral {{ background: var(--coral-soft); color: var(--coral); }}
    .pill-amber {{ background: var(--amber-soft); color: var(--amber); }}
    .pill-purple {{ background: var(--purple-soft); color: var(--purple); }}
    .pill-neutral {{ background: #f1eff6; color: var(--ink); }}
    .lead-header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; flex-wrap: wrap; margin-bottom: 12px; }}
    .lead-meta {{ display: flex; align-items: center; gap: 14px; }}
    .score-badge {{
      background: var(--purple);
      color: #fff;
      font-size: 20px;
      font-weight: 800;
      padding: 8px 12px;
      border-radius: 10px;
      min-width: 78px;
      text-align: center;
    }}
    .score-badge span {{ font-size: 12px; font-weight: 500; opacity: 0.78; }}
    .lead-title {{ font-size: 16px; font-weight: 700; color: var(--ink); }}
    .lead-sub {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
    .btn-link {{
      text-decoration: none;
      background: var(--purple-soft);
      color: var(--purple);
      font-weight: 600;
      font-size: 13px;
      padding: 7px 13px;
      border-radius: 8px;
    }}
    .btn-link:hover {{ background: #e3d7f7; }}
    .pain-summary {{ margin: 10px 0 14px; font-size: 14px; }}
    .section-mini {{ margin-bottom: 14px; }}
    .mini-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 700; margin-bottom: 6px; }}
    .safety-box {{
      background: var(--amber-soft);
      border-left: 4px solid var(--amber);
      padding: 10px 14px;
      border-radius: 6px;
      font-size: 13px;
      color: #78350f;
      margin-bottom: 16px;
    }}
    .drafts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }}
    .draft-box {{
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fbfafc;
      padding: 12px 14px;
    }}
    .draft-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; gap: 8px; }}
    .draft-label {{ font-size: 12px; font-weight: 700; color: var(--purple); }}
    .copy-btn {{
      background: var(--teal);
      color: #fff;
      border: none;
      border-radius: 6px;
      padding: 5px 11px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }}
    .copy-btn:hover {{ background: #0c5f59; }}
    .draft-text {{
      margin: 0;
      white-space: pre-wrap;
      font-family: inherit;
      font-size: 13px;
      color: var(--ink);
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }}
    th {{ background: #f4f1fa; color: var(--purple); font-size: 12px; text-transform: uppercase; }}
    details summary {{ cursor: pointer; font-weight: 700; color: var(--purple); }}
    .ledger-list {{ margin: 10px 0 0; padding-left: 20px; font-size: 13px; color: var(--muted); }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .footer {{
      margin-top: 32px;
      padding: 16px 20px;
      border-radius: 10px;
      background: #f1eff6;
      font-size: 12px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="container">
    <header class="hero">
      <div class="hero-top">
        <div>
          <h1>{_esc(profile.product_name)} &mdash; Competitor Churn &amp; Outreach Brief</h1>
          <p>{_esc(profile.product_category)} &middot; Lookback: {actor_input.lookbackDays} days &middot; Min Hot Score: {actor_input.minWinabilityScore}/100 &middot; Generated {generated_at}</p>
        </div>
        <span class="badge-live">Human-in-the-Loop Safe</span>
      </div>
    </header>

    <section class="kpi-grid">
      <div class="kpi"><div class="kpi-num">{raw_total}</div><div class="kpi-label">Harvested (Stage 2)</div></div>
      <div class="kpi"><div class="kpi-num">{stage3.dropped_count}</div><div class="kpi-label">Filtered Free (Stage 3)</div></div>
      <div class="kpi"><div class="kpi-num teal">{len(stage5.hot_lead_records)}</div><div class="kpi-label">Hot Leads Drafted</div></div>
      <div class="kpi"><div class="kpi-num">{len(stage5.warm_signal_records)}</div><div class="kpi-label">Warm / SEO Signals</div></div>
      <div class="kpi"><div class="kpi-num coral">{len(stage4.discarded)}</div><div class="kpi-label">Noise Rejected ($0)</div></div>
    </section>

    <section class="card">
      <div class="mini-label">Source Harvest Telemetry</div>
      <div class="pill-wrap" style="margin-bottom: 16px;">{source_badges}</div>
      {breakdown_html}
    </section>

    <h2>Actionable Hot Leads ({len(stage5.hot_lead_records)})</h2>
    {hot_cards_html}

    <h2>Warm / SEO Market Signals ({len(stage5.warm_signal_records)})</h2>
    <section class="card">{warm_html}</section>

    <section class="card">
      <details>
        <summary>Solvability Hard-Gate Quality Proof ({len(stage4.discarded)} Rejected False Positives &mdash; $0 Lead Charge)</summary>
        <p class="muted" style="margin: 10px 0;">Every item below failed factual solvability against the Verified Capabilities Ledger and was discarded before drafting or billing.</p>
        {rejected_html}
      </details>
    </section>

    <section class="card">
      <details>
        <summary>Stage 1 Grounding Anchor: Verified Capabilities Ledger ({len(profile.verified_capabilities)} Facts)</summary>
        <ul class="ledger-list">{ledger_items}</ul>
      </details>
    </section>

    <footer class="footer">
      <strong>Compliance &amp; Privacy Notice:</strong> This Actor never auto-posts to Reddit, Hacker News, Bluesky, or review platforms and never scrapes private/logged-in personal data. Always review drafts before posting and ensure GDPR/local privacy compliance if you enrich any public handle later.
    </footer>
  </div>

  <script>
    window.__REPORT_META__ = {meta_json};
    function copyText(elementId, btn) {{
      const el = document.getElementById(elementId);
      if (!el) return;
      const text = el.innerText || el.textContent || "";
      if (navigator.clipboard && navigator.clipboard.writeText) {{
        navigator.clipboard.writeText(text).then(() => {{
          const orig = btn.innerText;
          btn.innerText = "Copied!";
          setTimeout(() => {{ btn.innerText = orig; }}, 1600);
        }});
      }}
    }}
  </script>
</body>
</html>
"""