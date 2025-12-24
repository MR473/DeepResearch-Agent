"""
ROUGH SKETCH OF A DEEP RESEARCH AGENT
PARTS:
- Multi agent environemt
    - Agents:
        > Planner: Breaks down the question into sub-questions
        > Researcher: Use tool to search internet for information. 
        > Synthesizer: Formulate a detailed proper output for the user. 
        > Critic: Look into everything along with reference. Cross check everything. Identify missing parts (re-research if necessary)
"""

# Import libraries
import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from tavily import TavilyClient
from langchain_core.tools import tool 
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

import json
from pathlib import Path


# load environemt variables (API keys)
load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# retrieve schema from file
def load_schema(path: str) -> str:
    return json.dumps(json.loads(Path(path).read_text()), indent=2)

PLANNER_SCHEMA = load_schema("schemas/planner.json")
RESEARCHER_SCHEMA = load_schema("schemas/researcher.json")
CRITIC_SCHEMA = load_schema("schemas/critic.json")


# Tools
@tool
def internet_search(query: str, max_results: int = 5):
    """Run a web search"""
    return tavily_client.search(query, max_results=max_results)

# Set LLM Model
model = init_chat_model("openai:gpt-5")

# Agent prompts
planner_prompt = """ 
You are a Planner Agent. 
Purpose: Identify possible questions that need to be answered to provide a comprehensive response to the user's query.
Your duty: 
    - Give a JSON ouptut ONLY.
    - is to break down the question into sub-questions that relate to the query.
    - Each sub-question should be relevant to the main query.
    - The sub-questions should be specific and focused.
    - generate a list of 15 search queries that can be done to gather information to answer the sub-questions and query.
    - Use the schema provided below to structure your response.

    STRICT JSON Schema to follow: {PLANNER_SCHEMA}
"""
researcher_prompt = """ """
critic_prompt = """ """
synthesizer_prompt = """ """

# Create Agents

planner_agent = create_deep_agent(
    model=model,
    system_prompt=planner_prompt,
)

research_agent = create_deep_agent(
    model=model,
    system_prompt=researcher_prompt,
)

critic_agent = create_deep_agent(
    model=model,
    system_prompt=critic_prompt,
)

synthesizer_agent = create_deep_agent(
    model=model,
    system_prompt=synthesizer_prompt,
)


user_query = input("Your Question >>>  ")


planner_output = planner_agent.invoke({
        "messages": [{"role": "user", "content": "Help me research about recent AI developments."}],
    })

print("OUTPUT:\n")
print(result["messages"][-1].content)
