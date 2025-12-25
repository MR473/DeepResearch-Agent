import os
import json
import time
from datetime import datetime
from typing import Literal, Any, Dict

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

load_dotenv()

ARTIFACT_DIR = os.path.abspath("./artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

NOTES_PATH = "/artifacts/notes.md"
OPEN_Q_PATH = "/artifacts/open_questions.md"
TOOL_LOG_PATH = os.path.join(ARTIFACT_DIR, "tool_log.jsonl")

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> Dict[str, Any]:
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


class ToolCallLogger(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "unknown_tool")
        s = input_str if isinstance(input_str, str) else str(input_str)
        print(f"[TOOL START] {name} input={s[:500]}")

    def on_tool_end(self, output, **kwargs):
        preview = str(output)
        if len(preview) > 500:
            preview = preview[:500] + "..."
        print(f"[TOOL END] output={preview}")


logger = ToolCallLogger()
model = init_chat_model(model="gpt-5")
backend = FilesystemBackend(root_dir=ARTIFACT_DIR, virtual_mode=True)

SYSTEM_PROMPT = f"""\
You are a general-purpose Deep Research Agent.

You must use the filesystem tools to maintain a NOTEPAD on disk:
- Notes: {NOTES_PATH}
- Open questions: {OPEN_Q_PATH}

Rules:
- For any information needed ask by calling the internet_search tool. Avoid pretrained knowledge.
- If the task involves events/schedules/“this year”/holidays/New Year’s, you MUST call internet_search at least once.
- During work, frequently write/update {NOTES_PATH} with bullet notes + URLs (one bullet per claim).
- Track uncertainties/TODOs in {OPEN_Q_PATH}.
- BEFORE producing your final answer, you MUST ensure:
  - {NOTES_PATH} exists and has at least 5 bullets (or says "No external research needed.")
  - {OPEN_Q_PATH} exists and contains "None" if there are no open questions.
- Do at most ONE critique+revision cycle after drafting, then finalize.
- Points added after critic and revision need to be mentioned in the .md files with a ["Added after critique" tag].
- Note down number of critics and revisions that were done before finalizing. Note it down at the end of {NOTES_PATH}.

Major factual claims in the final answer must include at least one URL next to the claim.
"""

critic_prompt = f"""\
You are a critic for a Deep Research Agent. Your job is to review the agent's work and provide feedback.
Rules:
- Check {NOTES_PATH} for completeness and accuracy.
- Check {OPEN_Q_PATH} for any unresolved questions.
- Identify any missing information or errors.
- Provide constructive feedback and suggest specific improvements.
CRITIC INSTRUCTIONS:
- If everything is perfect, respond with "No issues found."
- Otherwise, list the issues found and suggest improvements.
- 
"""

agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt=SYSTEM_PROMPT,
    backend=backend,
)

if __name__ == "__main__":
    user_query = input("Your Question >>> ").strip()
    if not user_query:
        raise SystemExit("Please enter a question.")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        config={"recursion_limit": 150, "callbacks": [logger]},
    )

    print(result["messages"][-1].content)
    print(os.path.join(ARTIFACT_DIR, "notes.md"))
    print(os.path.join(ARTIFACT_DIR, "open_questions.md"))
    print(TOOL_LOG_PATH)
