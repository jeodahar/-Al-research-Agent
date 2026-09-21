"""The single CrewAI research agent, its task, and the Groq LLM setup."""

from crewai import Agent, Task, Crew, Process, LLM

from search_tool import web_search

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Groq's model id is literally "openai/gpt-oss-120b".
# CrewAI strips ONE leading "openai/" when custom_openai=True, so we add an
# extra one. The second entry is a fallback in case your CrewAI version
# passes the name through untouched.
MODEL_CANDIDATES = [
    "openai/openai/gpt-oss-120b",
    "openai/gpt-oss-120b",
]


def build_llm(model_name: str, api_key: str) -> LLM:
    return LLM(
        model=model_name,
        custom_openai=True,          # use CrewAI's native OpenAI client (no LiteLLM)
        base_url=GROQ_BASE_URL,      # ...but point it at Groq
        api_key=api_key,
        temperature=0.3,
    )


def build_crew(llm: LLM) -> Crew:
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Research the topic '{topic}' on the web and write an accurate, well-structured report.",
        backstory=(
            "You are a careful research analyst. You search the web, compare "
            "several sources, avoid guessing, and always list your sources."
        ),
        tools=[web_search],
        llm=llm,
        allow_delegation=False,
        max_iter=8,        # limits how many think/search loops the agent may do
        verbose=False,
    )

    task = Task(
        description=(
            "Research the topic: {topic}\n\n"
            "Use the DuckDuckGo Search tool 3 to 5 times with different queries. "
            "Base your report only on what you found. If sources disagree or "
            "information is missing, say so."
        ),
        expected_output=(
            "A report in Markdown with these sections:\n"
            "# Title\n"
            "## Summary (3-5 sentences)\n"
            "## Key Findings (bullet points)\n"
            "## Detailed Analysis\n"
            "## Conclusion\n"
            "## Sources (list of URLs you actually used)"
        ),
        agent=researcher,
    )

    return Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )


def run_research(topic: str, api_key: str) -> str:
    """Runs the agent and returns the report as Markdown text."""
    last_error = None
    for i, model_name in enumerate(MODEL_CANDIDATES):
        try:
            crew = build_crew(build_llm(model_name, api_key))
            result = crew.kickoff(inputs={"topic": topic})
            return getattr(result, "raw", str(result))
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            is_model_problem = any(k in msg for k in ("model", "404", "not found"))
            is_last = i == len(MODEL_CANDIDATES) - 1
            if is_model_problem and not is_last:
                continue  # try the next model-name format
            raise
    raise last_error
