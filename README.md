# Deep Research Agent

This project implements a **Deep Research Agent** using LangChain DeepAgents with:

The system behaves like a research system that gives detailed answers for any question prompted by its user.

---

## 1. Environment Setup (venv + requirements)

### Create and activate a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Using `uv` (optional)

If you prefer using **uv** for faster dependency installation:

### Install uv

```bash
pip install uv
```

### Install dependencies

```bash
uv pip install -r requirements.txt
```

---

## 3. Environment Variables (`.env` file)

Create a file named **`.env`** in the project root directory.

Add the following environment variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

These are required for:

* GPT model access
* Web search via Tavily

---

## 4. Running the Agent

From the project root directory:

```bash
python main.py
```

You will be prompted with:

```text
Your Question >>>
```

Enter any research, planning, or learning query.

The agent will:

1. Research and write notes
2. Generate an answer
3. Run the critic
4. Revise the answer (up to the configured maximum limit)
5. Stop when the critic returns `ENOUGH` or the limit is reached

---

## 5. Outputs, Logs, and Artifacts

All outputs are stored in the **`artifacts/`** directory.

### Project Directory Structure

```text
DeepResearchAgent/
│
├── main.py
├── requirements.txt
├── README.md
├── .env
├── .venv/
│
└── artifacts/
    ├── notes.md
    ├── open_questions.md
    ├── output.txt
    ├── critic_thoughts.txt
    └── tool_log.jsonl
```

---

## 6. Artifact Descriptions for analysis

### `artifacts/notes.md`

* Append-only research notes
* Timestamped reasoning and sources
* Corrections and post-critic additions
* Full research history preserved across runs

### `artifacts/open_questions.md`

* Unresolved questions and assumptions
* “Resolved:” entries when questions are answered
* No deletions; full reasoning history is preserved

### `artifacts/output.txt`

* Latest finalized answer only
* Overwritten on each run
* Used as the critic’s evaluation target

### `artifacts/critic_thoughts.txt`

* Full critic feedback across all runs
* Each critique round is timestamped
* Useful for understanding why revisions occurred

### `artifacts/tool_log.jsonl`

* One JSON object per tool call
* Includes search query, execution time, and number of results
* Useful for debugging, auditing, and analysis

---

## 7. Behavior Summary

* Research memory **persists across runs** (so delete for new question)
* Critic feedback **accumulates** (so delete for new question)
* Tool logs **accumulate** (so delete for new question)
* Only `output.txt` is overwritten per run
* Maximum revision count is controlled in `main.py`

This design provides cumulative knowledge, transparent agent behavior, and reproducible reasoning.
