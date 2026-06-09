# Implementation Plan - Elytics Phase 1

This plan details the folder structure, development environment setup, and the foundational files for Phase 1 of the **AI-Powered Natural Language Data Analytics Platform (Elytics)**.

We will focus on building the system modularly, prioritizing the Python/FastAPI/LangGraph backend environment and key schema structures first, before moving to the agent implementations and frontend.

---

## User Review Required

> [!IMPORTANT]
> The setup instructions cover both your current macOS environment and a guide to installing Node.js/npm on Windows without administrative privileges.

---

## Open Questions

> [!WARNING]
> 1. **Redshift Schema**: Do you have a specific database schema/tables (e.g., Sales, Customers, Transactions) already set up in Amazon Redshift that we should query, or should we create a mock local database schema for testing purposes first?
> 2. **AWS Bedrock / LLM Access**: Will you be using a real AWS Bedrock environment for testing on your Mac (which requires configured AWS credentials/region in `~/.aws/credentials` or `.env`), or should we include a local mock LLM provider option for testing the LangGraph nodes offline?

---

## Folder Structure

We propose the following clean, modular structure to separate the FastAPI backend, LangGraph agents, and React/TypeScript frontend.

```
/Users/aksha/Desktop/Elytics/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI entry point & API endpoints
│   │   ├── config.py           # Configuration (Bedrock, Redshift credentials)
│   │   ├── state.py            # LangGraph workflow state schema
│   │   ├── graph.py            # LangGraph workflow compilation
│   │   ├── agents/             # Python packages for LangGraph agents
│   │   │   ├── __init__.py
│   │   │   ├── planner.py      # Planner Agent (Intent analysis)
│   │   │   ├── schema.py       # Schema Agent (Metadata & column mapping)
│   │   │   ├── sql.py          # SQL Agent (Redshift SQL generator)
│   │   │   ├── validator.py    # Validation Agent (SQL query syntax validator)
│   │   │   ├── analyst.py      # Python Analysis Agent (Pandas/NumPy processing + Plotly)
│   │   │   └── insights.py     # Insights Agent (Natural language interpretation)
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── redshift.py     # Amazon Redshift database query executor
│   │       └── bedrock.py      # Bedrock LLM client wrapper (Claude 3.5 Sonnet / Haiku)
│   ├── requirements.txt        # Backend python dependencies
│   └── .env.example            # Environment variables example template
├── frontend/
│   ├── package.json            # Node project configuration
│   ├── tsconfig.json           # TypeScript configuration
│   ├── vite.config.ts          # Vite configuration
│   ├── index.html              # HTML entry point
│   ├── src/
│   │   ├── main.tsx            # React application root mount
│   │   ├── App.tsx             # Main layout and layout container
│   │   ├── types.ts            # Type definitions for states and API responses
│   │   └── components/
│   │       ├── Dashboard.tsx   # Dashboard layout containing components
│   │       ├── ChatInterface.tsx # Chat interface for NL queries
│   │       ├── ChartViewer.tsx # Chart rendering module (Plotly-based)
│   │       └── SQLViewer.tsx   # SQL rendering and validation feedback component
└── README.md                   # Documentation and project setup instructions
```

---

## Environment Setup Instructions

### 1. Mac Setup (Your local environment)

Run the following commands in your Mac terminal to set up the backend directory and environment:

```bash
# Navigate to the workspace
cd /Users/aksha/Desktop/Elytics

# Create the folder structure
mkdir -p backend/app/agents backend/app/utils frontend

# Create backend virtual environment
python3 -m venv backend/venv

# Activate the virtual environment
source backend/venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

We will define backend dependencies in `backend/requirements.txt`. Key dependencies include:
- `fastapi` & `uvicorn` (Backend web server)
- `pydantic` & `pydantic-settings` (Config management & validation)
- `langgraph` & `langchain` & `langchain-aws` (Bedrock client integration, Graph workflow orchestration)
- `pandas` & `numpy` & `plotly` (Python data analysis and plotting)
- `boto3` (AWS Redshift / Bedrock SDK interactions)
- `psycopg2-binary` (PostgreSQL driver compatible with Amazon Redshift)
- `python-dotenv` (Load local configuration variables)

### 2. Windows Corporate Laptop Setup (Without Admin Rights)

To replicate this environment on your corporate Windows machine, you can follow these steps:

#### Python Environment Setup
1. **Virtual Environment**: Use the pre-installed Python. Open Command Prompt (`cmd`) or PowerShell, navigate to the `backend/` directory, and run:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

#### Node.js / React Setup (Without Admin Rights)
If your corporate Windows machine does not have Node.js and you do not have administrator privileges, you can set it up manually:
1. **Download Node.js Windows Binary (.zip)**: Go to the official [Node.js Downloads page](https://nodejs.org/en/download/) and select **Windows Binary (.zip)** (x64 or x86 depending on your CPU).
2. **Extract to User Profile**: Create a directory in your user profile, for example: `C:\Users\<YourUsername>\AppData\Local\nodejs`, and extract the contents of the zip file into it.
3. **Update PATH Environment Variable (User Level)**:
   - Search for **"Edit environment variables for your account"** in the Windows Start Menu. (This does not require admin privileges).
   - Under **User variables**, select the `Path` variable and click **Edit**.
   - Click **New** and add the path to the extracted Node.js folder: `C:\Users\<YourUsername>\AppData\Local\nodejs`.
   - Click **OK** on all windows.
4. **Verify installation**: Open a *new* PowerShell or Prompt window and run:
   ```cmd
   node -v
   npm -v
   ```
   This will run Node.js and npm successfully without requiring any administrative rights!

---

## Proposed Changes

We will create the core configuration file, the state schema, and the FastAPI main entry point.

### [Component Name] Backend Foundation

#### [NEW] [requirements.txt](file:///Users/aksha/Desktop/Elytics/backend/requirements.txt)
Defines all backend libraries including FastAPI, LangGraph, Pandas, Plotly, AWS SDK (Boto3) and psycopg2.

#### [NEW] [.env.example](file:///Users/aksha/Desktop/Elytics/backend/.env.example)
A template configuration file demonstrating the required AWS Bedrock region/credentials and Redshift database connection settings.

#### [NEW] [config.py](file:///Users/aksha/Desktop/Elytics/backend/app/config.py)
Config class using Pydantic settings to load and validate configurations from the environment.

#### [NEW] [state.py](file:///Users/aksha/Desktop/Elytics/backend/app/state.py)
The LangGraph shared state object. This is the structural backbone of our agent graph. Every node (Agent) in the workflow reads from and writes updates to this schema.

#### [NEW] [main.py](file:///Users/aksha/Desktop/Elytics/backend/app/main.py)
The FastAPI entry point. It sets up CORS, endpoints for queries, and health-checks.

---

## Verification Plan

### Automated Verification
- Run syntax and lint checks on backend code.
- Run `uvicorn app.main:app --reload` to verify the web server boots successfully and schema types are correct.

### Manual Verification
- Verify the Swagger UI (`http://127.0.0.1:8000/docs`) is accessible.
