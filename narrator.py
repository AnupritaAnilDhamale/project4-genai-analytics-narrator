"""
narrator.py — GenAI Analytics Narrator
LLM-Powered Automated Insights | Anuprita Dhamale

Takes a KPI dataframe, detects anomalies, then calls OpenAI GPT-4
to generate a crisp, executive-ready narrative summary.
"""

import os
import json
import textwrap
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior data analyst at a healthcare company.
    You receive structured KPI data and anomaly alerts.
    Your job: write a concise, executive-ready business narrative (5–8 sentences).

    Rules:
    - Lead with the most important finding
    - Use plain business language, no jargon
    - Quantify every claim using the numbers provided
    - Flag anomalies clearly with urgency
    - End with one actionable recommendation
    - Tone: confident, direct, professional
    - Do NOT say "the data shows" — just state the finding
""").strip()


# ── Anomaly Detection ─────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, metric_col: str, date_col: str,
                     z_threshold: float = 2.0) -> list[dict]:
    """Flag data points more than z_threshold standard deviations from rolling mean."""
    df = df.copy().sort_values(date_col)
    rolling = df[metric_col].rolling(window=4, min_periods=2)
    df["rolling_mean"] = rolling.mean()
    df["rolling_std"]  = rolling.std().fillna(1)
    df["z_score"]      = (df[metric_col] - df["rolling_mean"]) / df["rolling_std"]

    anomalies = df[df["z_score"].abs() > z_threshold]
    return [
        {
            "date":          str(row[date_col]),
            "value":         round(row[metric_col], 2),
            "expected":      round(row["rolling_mean"], 2),
            "z_score":       round(row["z_score"], 2),
            "direction":     "spike" if row["z_score"] > 0 else "drop",
        }
        for _, row in anomalies.iterrows()
    ]


def compute_kpi_summary(df: pd.DataFrame) -> dict:
    """Compute high-level KPI summary stats."""
    numeric = df.select_dtypes(include=[np.number])
    latest  = df.sort_values(df.columns[0]).iloc[-1]
    prev    = df.sort_values(df.columns[0]).iloc[-2] if len(df) > 1 else latest

    summary = {}
    for col in numeric.columns:
        curr_val = latest[col]
        prev_val = prev[col]
        pct_chg  = ((curr_val - prev_val) / prev_val * 100) if prev_val != 0 else 0
        summary[col] = {
            "latest":     round(curr_val, 2),
            "previous":   round(prev_val, 2),
            "pct_change": round(pct_chg, 2),
            "trend":      "up" if pct_chg > 0 else ("down" if pct_chg < 0 else "flat"),
            "period_avg": round(numeric[col].mean(), 2),
            "period_min": round(numeric[col].min(), 2),
            "period_max": round(numeric[col].max(), 2),
        }
    return summary


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def build_prompt(kpi_summary: dict, anomalies: list[dict],
                 context: str = "healthcare revenue operations") -> str:
    summary_text = json.dumps(kpi_summary, indent=2)
    anomaly_text = json.dumps(anomalies, indent=2) if anomalies else "None detected."

    return f"""
Business context: {context}
Reporting period: {datetime.utcnow().strftime('%B %Y')}

KPI SUMMARY:
{summary_text}

ANOMALIES DETECTED:
{anomaly_text}

Write the executive narrative now.
""".strip()


# ── Main Narrator ──────────────────────────────────────────────────────────────

class AnalyticsNarrator:
    """Orchestrates anomaly detection + LLM narrative generation."""

    def __init__(self, model: str = "gpt-4", temperature: float = 0.3):
        self.model       = model
        self.temperature = temperature

    def narrate(self, df: pd.DataFrame, date_col: str, metric_cols: list[str],
                context: str = "business operations") -> dict:
        """
        Full pipeline:
        1. Compute KPI summary
        2. Detect anomalies per metric
        3. Build structured prompt
        4. Call GPT-4
        5. Return narrative + metadata
        """
        print(f"📊 Computing KPIs for {len(df)} rows...")
        kpi_summary = compute_kpi_summary(df[[date_col] + metric_cols])

        print("🔍 Running anomaly detection...")
        all_anomalies = []
        for col in metric_cols:
            anomalies = detect_anomalies(df, col, date_col)
            for a in anomalies:
                a["metric"] = col
            all_anomalies.extend(anomalies)

        print(f"   Found {len(all_anomalies)} anomaly(ies)")

        prompt = build_prompt(kpi_summary, all_anomalies, context=context)

        print("🤖 Calling GPT-4 for narrative generation...")
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]
        )

        narrative = response.choices[0].message.content.strip()
        tokens    = response.usage.total_tokens

        return {
            "narrative":     narrative,
            "kpi_summary":   kpi_summary,
            "anomalies":     all_anomalies,
            "model":         self.model,
            "tokens_used":   tokens,
            "generated_at":  datetime.utcnow().isoformat(),
        }

    def to_html_report(self, result: dict, title: str = "Analytics Report") -> str:
        """Render the narrative result as a styled HTML report."""
        anomaly_html = ""
        if result["anomalies"]:
            rows = "".join(
                f"<tr><td>{a['metric']}</td><td>{a['date']}</td>"
                f"<td style='color:{'#c0392b' if a['direction']=='drop' else '#27ae60'}'>"
                f"{a['direction'].upper()}</td><td>{a['value']:,}</td>"
                f"<td>{a['pct_change'] if 'pct_change' in a else a['z_score']}x σ</td></tr>"
                for a in result["anomalies"]
            )
            anomaly_html = f"""
            <h3 style="color:#c0392b; margin-top:2rem">⚠️ Anomalies Detected</h3>
            <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;width:100%">
              <thead style="background:#f8f9fa">
                <tr><th>Metric</th><th>Date</th><th>Type</th><th>Value</th><th>Magnitude</th></tr>
              </thead>
              <tbody>{rows}</tbody>
            </table>"""

        return f"""<!DOCTYPE html><html><head>
        <meta charset="UTF-8"/>
        <style>
          body {{ font-family: Georgia, serif; max-width: 860px; margin: 2rem auto; padding: 2rem;
                  color: #1a1a1a; line-height: 1.7; }}
          h1   {{ font-size: 1.8rem; border-bottom: 2px solid #2c3e50; padding-bottom: 0.5rem; }}
          h2   {{ font-size: 1.2rem; color: #2c3e50; margin-top: 2rem; }}
          .narrative {{ background: #f0f4f8; padding: 1.5rem; border-left: 4px solid #2c3e50;
                        border-radius: 4px; font-size: 1.05rem; }}
          .meta {{ font-size: 0.78rem; color: #888; margin-top: 1rem; }}
        </style>
        </head><body>
        <h1>{title}</h1>
        <p class="meta">Generated: {result['generated_at']} · Model: {result['model']} · Tokens: {result['tokens_used']}</p>
        <h2>Executive Summary</h2>
        <div class="narrative">{result['narrative'].replace(chr(10), '<br/>')}</div>
        {anomaly_html}
        </body></html>"""


# ── Demo Runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate 12 months of revenue KPI data
    np.random.seed(42)
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    df = pd.DataFrame({
        "month":             months.strftime("%Y-%m"),
        "total_revenue":     np.random.normal(2_500_000, 150_000, 12).clip(2_000_000),
        "claims_processed":  np.random.normal(18_000, 1_200, 12).clip(12_000).astype(int),
        "denial_rate_pct":   np.random.normal(12.5, 2.0, 12).clip(5, 30).round(2),
        "avg_payment_ratio": np.random.normal(0.78, 0.04, 12).clip(0.5, 1.0).round(3),
    })

    # Inject anomaly in month 9
    df.loc[8, "total_revenue"]   = 1_800_000   # revenue drop
    df.loc[10, "denial_rate_pct"] = 28.5        # denial spike

    narrator = AnalyticsNarrator(model="gpt-4", temperature=0.3)
    result   = narrator.narrate(
        df=df,
        date_col="month",
        metric_cols=["total_revenue", "claims_processed", "denial_rate_pct", "avg_payment_ratio"],
        context="healthcare revenue cycle management"
    )

    print("\n" + "="*60)
    print("EXECUTIVE NARRATIVE")
    print("="*60)
    print(result["narrative"])
    print(f"\nTokens used: {result['tokens_used']}")

    html = narrator.to_html_report(result, title="Healthcare Revenue — Monthly KPI Report")
    with open("/tmp/analytics_report.html", "w") as f:
        f.write(html)
    print("\n✅ HTML report saved to /tmp/analytics_report.html")
