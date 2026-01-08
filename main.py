import os
import json
import time
from datetime import datetime
from typing import Literal, Any, Dict, List

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

load_dotenv()

ARTIFACT_DIR = os.path.abspath("./artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

NOTES_PATH = "./artifacts/notes.md"
OPEN_Q_PATH = "./artifacts/open_questions.md"
OUTPUT_DISK_PATH = "./artifacts/output.txt"
TOOL_LOG_PATH = "./artifacts/tool_log.jsonl"
CRITIC_LOG_PATH = "./artifacts/critic_thoughts.txt"

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


def write_output(text: str):
    # output.txt is intended to be "current best answer" (overwrite is fine)
    with open(OUTPUT_DISK_PATH, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def warn_if_output_missing_sections(output_text: str) -> None:
    """Optional guard: warns if required sections are missing in output.txt."""
    required = ["Title:", "Overview:", "Main Discussion:", "Key Takeaways:", "Sources:"]
    missing = [s for s in required if s not in output_text]
    if missing:
        print("\n⚠️ FORMAT WARNING: output.txt is missing required sections:")
        for m in missing:
            print(f"  - {m}")
        print("The critic should request revision if this persists.\n")


logger = ToolCallLogger()
model = init_chat_model(model="gpt-5")
backend = FilesystemBackend(root_dir=ARTIFACT_DIR, virtual_mode=True)


system_prompt = f"""\
You are a Deep Research Agent that produces clear, well-structured, and reader-friendly research outputs.

Your primary goal is to explain findings in fluent, natural English that flows logically from one idea to the next.

You must maintain append-only research history:
- Notes: {NOTES_PATH}
- Open questions: {OPEN_Q_PATH}

Hard rules (do not violate):
- Never overwrite notes.md or open_questions.md after creation.
- Always read existing content first, then append new, timestamped sections.
- Corrections must be appended; never delete prior content.
- When a question is resolved, clearly mark it with a line starting with: [Resolved]
- Use internet_search for factual or time-sensitive information.
- At least one internet_search call is required per user question.
- Final factual claims must include source URLs.

Writing quality rules (VERY IMPORTANT):
- Write in complete paragraphs unless a list is clearly more readable.
- Use smooth transitions between ideas (e.g., “However”, “As a result”, “In contrast”).
- Prefer explanation and synthesis over listing raw facts.
- Avoid robotic or tool-like phrasing.
- Assume the reader is intelligent but unfamiliar with the topic.

--------------------------------------------------
ARTIFACT FORMAT CONTRACTS (MUST FOLLOW)
--------------------------------------------------

OUTPUT FILE: ./artifacts/output.txt
This file is the final, human-readable answer.

When you respond to the user, your returned message MUST follow EXACTLY this structure (in this order):

Title:
<One concise line>

Overview:
<One short paragraph explaining what this answer covers>

Main Discussion:
<Multiple paragraphs with smooth transitions and clear explanations>

Key Takeaways:
- <Bullet>
- <Bullet>

Sources:
- <URL>
- <URL>

Do NOT include timestamps, critique notes, or tool logs in output.txt.

--------------------------------------------------
NOTES FILE: {NOTES_PATH}
Append-only research notes.
Each append MUST follow this structure:

## Research Notes — <YYYY-MM-DD HH:MM>

Context:
<Why this research was done>

Findings:
<Paragraph-style explanation>

Interpretation:
<What these findings mean>

Sources Consulted:
- <URL>
- <URL>

At the end of notes.md, append a final line:
Critique rounds: <N>, Revision rounds: <M>
(These counts reset for each new user question.)

--------------------------------------------------
OPEN QUESTIONS FILE: {OPEN_Q_PATH}
Append-only list of unresolved or resolved questions.
Each entry MUST follow one of these formats:

Unresolved:
- Question: <text>
  Why it matters: <text>

Resolved:
- [Resolved] <question>
  Resolution summary: <1–2 lines>
  Resolved on: <YYYY-MM-DD>

--------------------------------------------------

Self-check (silent, before finalizing):
- Verify all required sections and headers are present and spelled exactly.
- If verification fails, fix before returning the final response.
"""


critic_prompt = f"""\
You are a critic agent evaluating the clarity, completeness, and readability of the research output.

Review the following files:
- {NOTES_PATH}
- {OPEN_Q_PATH}
- /artifacts/output.txt

Decide sufficiency AND format correctness.

The output is ONLY acceptable if:
- output.txt follows the required section structure EXACTLY:
  Title:, Overview:, Main Discussion:, Key Takeaways:, Sources:
- notes.md is append-only and each append follows the Research Notes structure
- open_questions.md is append-only and follows the Open/Resolved formats
- Writing is clear, fluent, and explanatory

Return EXACTLY one of the following two formats and nothing else:

1)
ENOUGH

2)
REVISE:
- <specific change needed, referencing sections or paragraphs>
- <specific clarity, structure, missing explanation, or synthesis improvement>

Rules:
- Be precise and actionable.
- Focus on clarity, flow, missing explanations, weak synthesis, or formatting violations.
- Do NOT suggest deleting timestamps or prior content from notes.md/open_questions.md.
- Do NOT rewrite the document yourself.
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

if __name__ == "__main__":
    print("Deep Research Agent started. Type \\exit to quit.\n")

    while True:
        user_query = input("Your Question >>> ").strip()

        if user_query.lower() == r"\exit":
            print("Exiting. Goodbye!")
            break

        if not user_query:
            print("Please enter a non-empty question.\n")
            continue

        MAX_REVISIONS = 3
        critique_rounds = 0
        revision_rounds = 0

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_query}]},
            config={"recursion_limit": 150, "callbacks": [logger]},
        )

        current_text = result["messages"][-1].content
        print("\nINITIAL OUTPUT:\n")
        print(current_text)
        warn_if_output_missing_sections(current_text)
        write_output(current_text)

        while revision_rounds < MAX_REVISIONS:
            critique_rounds += 1

            critic_result = critic.invoke(
                {"messages": [{"role": "user", "content": "Review artifacts now."}]},
                config={"recursion_limit": 60, "callbacks": [logger]},
            )

            decision = critic_result["messages"][-1].content.strip()
            append_critic_thoughts(decision, critique_rounds)

            print("\nCRITIC DECISION:\n")
            print(decision)

            if decision.upper().startswith("ENOUGH"):
                break

            if not decision.upper().startswith("REVISE"):
                print("Unexpected critic response. Stopping revisions.")
                break

            revision_rounds += 1

            revision_prompt = f"""\
                You are revising an existing answer to satisfy BOTH:
                1) The critic's requested fixes, and
                2) The artifact format contracts in the system prompt.

                Apply ONLY the changes requested in the critic feedback below.

                CRITIC FEEDBACK:
                {decision}

                Rules:
                - Prefer minimal edits, but you MAY restructure sections if needed to meet required format.
                - Ensure output.txt structure is EXACT: Title / Overview / Main Discussion / Key Takeaways / Sources.
                - notes.md and open_questions.md must remain append-only: only append new timestamped sections or [Resolved] lines.
                - If adding text, clearly mark it as [Added after critique] within notes.md/open_questions.md entries.
                - If you must remove content inside output.txt, replace it with: [Removed after critique]
                - Update the end of notes.md with:
                Critique rounds: {critique_rounds}, Revision rounds: {revision_rounds}
                Before finalizing, self-check that all required headers are present and spelled exactly.
            """

            result = agent.invoke(
                {
                    "messages": [
                        {"role": "assistant", "content": current_text},
                        {"role": "user", "content": revision_prompt},
                    ]
                },
                config={"recursion_limit": 150, "callbacks": [logger]},
            )

            current_text = result["messages"][-1].content
            print("\nREVISED OUTPUT:\n")
            print(current_text)
            warn_if_output_missing_sections(current_text)
            write_output(current_text)

        print("\nArtifacts written to:")
        print(OUTPUT_DISK_PATH)
        print(os.path.join(ARTIFACT_DIR, "notes.md"))
        print(os.path.join(ARTIFACT_DIR, "open_questions.md"))
        print(TOOL_LOG_PATH)
        print(CRITIC_LOG_PATH)
        print("\n--- Ready for next question ---\n")
