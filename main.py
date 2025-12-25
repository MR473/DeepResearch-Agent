import os
import json
import time
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

load_dotenv()

ARTIFACT_DIR = "artifacts"
NOTES_PATH = os.path.join(ARTIFACT_DIR, "notes.md")
OPEN_Q_PATH = os.path.join(ARTIFACT_DIR, "open_questions.md")
TOOL_LOG_PATH = os.path.join(ARTIFACT_DIR, "tool_log.jsonl")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """
    Search the internet using Tavily.

    Returns results containing titles, URLs, and snippets (and optionally raw content).
    """
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[TOOL internet_search START {ts}] query={query!r} max_results={max_results} topic={topic}")

    t0 = time.time()
    result = tavily_client.search(
        query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )
    dt = time.time() - t0

    n = len(result.get("results", [])) if isinstance(result, dict) else 0
    print(f"[TOOL internet_search END] took={dt:.2f}s results={n}")

    with open(TOOL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "tool": "internet_search",
                    "ts": ts,
                    "query": query,
                    "max_results": max_results,
                    "topic": topic,
                    "include_raw_content": include_raw_content,
                    "took_sec": round(dt, 3),
                    "n_results": n,
                }
            )
            + "\n"
        )

    return result


model = init_chat_model(model="gpt-5")

SYSTEM_PROMPT = f"""\
You are a general-purpose Deep Research Agent.

You must use the filesystem as a NOTEPAD and keep it updated during work:
- Notes file: {NOTES_PATH}
- Open questions file: {OPEN_Q_PATH}
- Save an Append details in these files 

Process (repeat only as needed):
1) Plan approach.
2) Research using internet_search whenever facts/events/schedules are needed.
3) Write/update notes in {NOTES_PATH} (bullets + URLs).
4) Write/update uncertainties/TODOs in {OPEN_Q_PATH}.
5) Draft the final answer.
6) Critique the draft briefly:
   - missing evidence/URLs?
   - missing logistics/timings?
   - contradictions?
7) If there are major gaps, do ONE more research+revision pass, then finalize.

Hard requirements:
- If the task involves events, schedules, “this year”, holidays, or New Year’s, you MUST call internet_search at least once.
- Major factual claims in the final answer must include at least one URL (put the URL right next to the claim).
- Before producing the final answer you MUST ensure:
  - {NOTES_PATH} exists and contains at least 5 bullets (or says "No external research needed.")
  - {OPEN_Q_PATH} exists and contains "None" if there are no open questions.
- Stop once the checklist is satisfied and you have a complete answer; do not keep refining forever.
"""

agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt=SYSTEM_PROMPT,
)

if __name__ == "__main__":
    user_query = input("Your Question >>> ").strip()
    if not user_query:
        raise SystemExit("Please enter a question.")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        config={"recursion_limit": 80},
    )

    print("\n=== FINAL ANSWER ===\n")
    print(result["messages"][-1].content)

    print("\n=== ARTIFACTS SAVED ===")
    print(f"- {NOTES_PATH}")
    print(f"- {OPEN_Q_PATH}")
    print(f"- {TOOL_LOG_PATH}")
