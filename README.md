# 🤖 GenAI Analytics Narrator — LLM-Powered Automated Insights

> **Built by:** Anuprita Dhamale | [LinkedIn](https://linkedin.com/in/dhamaleanuprita) | [Portfolio](https://dhamaleanuprita.github.io)

Connects OpenAI's GPT-4 to a live analytics pipeline to **automatically generate executive-ready business narratives** from KPI data — with statistical anomaly detection baked in. This is the kind of AI integration that saved 50% of manual reporting time in production.

> Not a chatbot. Not a demo. A real analytics automation tool.

---

## 🎯 Problem Solved

Every month, analysts spend hours writing the same narrative reports:
*"Revenue was up 8% MoM. Claims processing hit a new high. Denial rate spiked in Q3..."*

This tool does it in seconds — automatically detecting what matters, quantifying it, and generating crisp prose a CFO can read immediately.

---

## 🏗️ Architecture

```
KPI DataFrame (any source: SQL, CSV, API)
        │
        ▼
┌──────────────────────────────┐
│   Anomaly Detection Engine   │  ← Z-score rolling window analysis
│   compute_kpi_summary()      │  ← Period stats: avg, min, max, % change
└──────────────────────────────┘
        │
        ▼ Structured JSON context
┌──────────────────────────────┐
│   Prompt Builder             │  ← Business context + KPIs + anomalies
│   System prompt engineering  │  ← Role-instructed for executive tone
└──────────────────────────────┘
        │
        ▼ GPT-4 API call
┌──────────────────────────────┐
│   OpenAI GPT-4               │  ← Narrative generation
│   temp=0.3 for consistency   │
└──────────────────────────────┘
        │
        ▼
  Narrative + HTML Report
```

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| LLM | OpenAI GPT-4 (via API) |
| Anomaly Detection | Python, NumPy, rolling Z-score |
| Data Layer | Pandas |
| API Wrapper | FastAPI |
| Output | HTML report + JSON |

---

## 📁 Folder Structure

```
genai-analytics-narrator/
├── narrator.py          # Core: anomaly detection + GPT-4 narrative
├── api.py               # FastAPI endpoint — POST /generate-narrative
├── prompts/
│   └── system_prompt.txt
├── examples/
│   ├── sample_kpi_data.csv
│   └── sample_output.html   # Example generated report
├── tests/
│   └── test_narrator.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/dhamaleanuprita/genai-analytics-narrator.git
cd genai-analytics-narrator
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
python narrator.py   # Run demo with synthetic healthcare data
```

### FastAPI Endpoint

```bash
uvicorn api:app --reload
# POST to http://localhost:8000/generate-narrative
```

```json
POST /generate-narrative
{
  "data": [...],
  "date_col": "month",
  "metric_cols": ["revenue", "claims", "denial_rate"],
  "context": "healthcare revenue cycle"
}
```

---

## 💡 Sample Output

> *"Provider revenue reached $2.74M in November, a 9.6% increase from October and the highest point in the trailing 12 months. Claims volume grew to 19,847 — up 6.2% month-over-month — while the average payment ratio held steady at 0.81. However, September's revenue of $1.8M represents a statistically significant anomaly (2.4σ below trend), driven by a concurrent spike in denial rates to 28.5% in November that warrants immediate review. The claims denial rate has exceeded the 20% threshold twice in the last quarter, suggesting a systematic issue in clinical documentation or payer policy changes. Recommend an urgent audit of denial reasons for cardiology and orthopedic claims, which historically drive the highest billed amounts."*

---

## 💡

- **Real GenAI integration** — not a basic chatbot, but a structured data → LLM pipeline
- **Prompt engineering** — system prompt crafted for consistent executive tone
- **Anomaly detection** — statistical analysis feeds context to the LLM
- **Production API** — FastAPI wrapper makes this deployable as a service
- **Business context** — directly mirrors what was built at American Choice Healthcare


