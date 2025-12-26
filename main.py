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
OUTPUT_DISK_PATH = os.path.join(ARTIFACT_DIR, "output.txt")
TOOL_LOG_PATH = os.path.join(ARTIFACT_DIR, "tool_log.jsonl")
CRITIC_LOG_PATH = os.path.join(ARTIFACT_DIR, "critic_thoughts.txt")

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> Dict[str, Any]:
    """Search the internet using Tavily and return results with URLs/snippets."""
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[TOOL internet_search START {ts}] query={query!r}")

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
        print(f"[TOOL START] {name} input={s[:300]}")

    def on_tool_end(self, output, **kwargs):
        preview = str(output)
        if len(preview) > 300:
            preview = preview[:300] + "..."
        print(f"[TOOL END] output={preview}")


def append_critic_thoughts(text: str, round_num: int):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CRITIC_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n=== CRITIC ROUND {round_num} | {ts} ===\n")
        f.write(text.strip() + "\n")


logger = ToolCallLogger()
model = init_chat_model(model="gpt-5")
backend = FilesystemBackend(root_dir=ARTIFACT_DIR, virtual_mode=True)

system_prompt = f"""\
You are a Deep Research Agent.

You must maintain append-only research history:
- Notes: {NOTES_PATH}
- Open questions: {OPEN_Q_PATH}

Rules:
- Never overwrite these files after creation.
- Always read first, then append timestamped sections.
- Corrections must be appended, not deleted.
- Resolved questions must be marked with "Resolved:" lines.
- Use internet_search for factual or time-sensitive info or any information you need.
- At least one search required.
- After critic feedback, additions must include [Added after critique].
- At the end of notes, record:
  Critique rounds: <N>, Revision rounds: <M>

Final answers must cite URLs for factual claims.
"""

critic_prompt = f"""\
You are a critic agent.

Review:
- {NOTES_PATH}
- {OPEN_Q_PATH}
- /artifacts/output.txt

Decide sufficiency.

Output exactly:
- ENOUGH
or
- REVISE:
  - <specific fix>
  - <specific fix>
"""

agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt=system_prompt,
    backend=backend,
)

critic = create_deep_agent(
    model=model,
    system_prompt=critic_prompt,
    backend=backend,
)


def write_output(text: str):
    with open(OUTPUT_DISK_PATH, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    user_query = input("Your Question >>> ").strip()
    if not user_query:
        raise SystemExit("Please enter a question.")

    MAX_REVISIONS = 3
    critique_rounds = 0
    revision_rounds = 0

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        config={"recursion_limit": 150, "callbacks": [logger]},
    )

    current_text = result["messages"][-1].content
    print(current_text)
    write_output(current_text)

    while revision_rounds < MAX_REVISIONS:
        critique_rounds += 1

        critic_result = critic.invoke(
            {"messages": [{"role": "user", "content": "Review artifacts now."}]},
            config={"recursion_limit": 60, "callbacks": [logger]},
        )

        decision = critic_result["messages"][-1].content.strip()
        append_critic_thoughts(decision, critique_rounds)
        print("\nCRITIC DECISION:\n", decision)

        if decision.upper().startswith("ENOUGH"):
            break

        if not decision.upper().startswith("REVISE"):
            break

        revision_rounds += 1

        feedback = (
            decision
            + f"\nCritique rounds={critique_rounds}, Revision rounds={revision_rounds}."
        )

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_query + "\n\nCRITIC FEEDBACK:\n" + feedback}]},
            config={"recursion_limit": 150, "callbacks": [logger]},
        )

        current_text = result["messages"][-1].content
        print("\nREVISED OUTPUT:\n")
        print(current_text)
        write_output(current_text)

    print(OUTPUT_DISK_PATH)
    print(os.path.join(ARTIFACT_DIR, "notes.md"))
    print(os.path.join(ARTIFACT_DIR, "open_questions.md"))
    print(TOOL_LOG_PATH)
    print(CRITIC_LOG_PATH)
