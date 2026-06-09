# Elytics — AI-Powered Natural Language Analytics Platform

> **Phase 1 — Backend complete. Frontend (React + TypeScript) coming in Phase 2.**

An AI-powered analytics platform that lets business users ask questions in plain English and receive SQL queries, Python analysis, interactive charts, and business insights — without writing a single line of code.

---

## Architecture (Phase 1)

```
User Question (NL)
  → React + TypeScript Frontend
    → FastAPI Backend
      → LangGraph 8-Agent Pipeline:
          1. Planner Agent      — identifies intent, builds analytical plan
          2. Schema Agent       — discovers relevant Redshift tables/columns
          3. SQL Agent          — generates Amazon Redshift SQL
          4. Validation Agent   — two-stage SQL safety + semantic check
          5. Redshift Node      — executes SQL, retrieves rows
          6. Python Agent       — LLM writes Pandas/NumPy/Plotly analysis code
          7. Executor Agent     — runs code safely in restricted sandbox
          8. Insights Agent     — generates business narrative
      → Response: charts[] + insights[] + SQL + Python code
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + TypeScript + Plotly.js |
| Backend API | FastAPI + Uvicorn |
| Orchestration | LangGraph |
| LLM | AWS Bedrock (Claude 3.5 Sonnet / Claude 3 Haiku) |
| Database | Amazon Redshift |
| Analytics | Pandas, NumPy, Plotly |

---

## Project Structure

```
Elytics/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── planner.py        # Agent 1: intent + plan
│   │   │   ├── schema.py         # Agent 2: schema discovery
│   │   │   ├── sql.py            # Agent 3: SQL generation
│   │   │   ├── validator.py      # Agent 4: SQL validation
│   │   │   ├── python_agent.py   # Agent 5: Python code generation (LLM)
│   │   │   ├── executor.py       # Agent 6: sandboxed code execution
│   │   │   └── insights.py       # Agent 7: business narrative
│   │   ├── utils/
│   │   │   ├── bedrock.py        # AWS Bedrock / Claude client
│   │   │   └── redshift.py       # Amazon Redshift client
│   │   ├── config.py             # Environment config (Pydantic Settings)
│   │   ├── state.py              # LangGraph shared state schema
│   │   ├── graph.py              # LangGraph workflow orchestrator
│   │   └── main.py               # FastAPI entry point
│   ├── .env.example              # Environment variable template
│   └── requirements.txt          # Python dependencies
└── frontend/                     # React + TypeScript (Phase 2)
```

---

## Setup & Run (Mac)

### 1. Clone the repo
```bash
git clone https://github.com/akkkshat07/Elytics.git
cd Elytics
```

### 2. Create virtual environment & install dependencies
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure credentials
```bash
cp .env.example .env
# Edit .env with your AWS and Redshift credentials
```

### 4. Start the backend
```bash
uvicorn app.main:app --reload
```

### 5. Test the API
Open **http://127.0.0.1:8000/docs** → use the `/api/query` Swagger panel.

---

## Setup & Run (Windows)

```cmd
cd backend
python -m venv venv
backend\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
REM Edit .env with credentials
uvicorn app.main:app --reload
```

---

## Environment Variables

Copy `backend/.env.example` to `backend/.env` and fill in:

```env
# AWS Bedrock (for Claude LLM)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1

# Amazon Redshift
REDSHIFT_HOST=your-cluster.redshift.amazonaws.com
REDSHIFT_PORT=5439
REDSHIFT_DB=your_database
REDSHIFT_USER=your_user
REDSHIFT_PASSWORD=your_password
```

---

## API Response Shape

```json
{
  "status": "success",
  "query": "Top 10 products by revenue last quarter?",
  "insights": ["Revenue totaled $4.2M across 847 products.", "..."],
  "generated_sql": "SELECT product, SUM(revenue) AS total...",
  "generated_python": "df = pd.DataFrame(query_results)\n...",
  "charts": [{ "data": [...], "layout": {...} }],
  "execution_results": { "statistics": { "total_revenue": 4200000 } },
  "step_log": ["✅ Planner: intent=ranking", "✅ Schema: tables=[sales]", "..."]
}
```
