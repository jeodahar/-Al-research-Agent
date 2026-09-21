"""The single CrewAI research agent, its task, and the Groq LLM setup."""

from crewai import Agent, Task, Crew, Process, LLM

from search_tool import web_search, read_webpage

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Groq's model id is literally "openai/gpt-oss-120b".
# CrewAI strips ONE leading "openai/" when custom_openai=True, so we add an
# extra one. The second entry is a fallback in case your CrewAI version
# passes the name through untouched.
MODEL_CANDIDATES = [
    "openai/openai/gpt-oss-120b",
    "openai/gpt-oss-120b",
]

# gpt-oss sometimes tries to call a tool that doesn't exist (e.g. "open_file")
# and Groq rejects it. The mistake is random, so we simply retry.
MAX_TOOL_RETRIES = 3


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
            "several sources, avoid guessing, and always list your sources. "
            "You can ONLY use the tools you were given: 'DuckDuckGo Search' and "
            "'Read Webpage'. You never call any other tool."
        ),
        tools=[web_search, read_webpage],
        llm=llm,
        allow_delegation=False,
        max_iter=12,       # limits how many think/search loops the agent may do
        verbose=False,
    )

    task = Task(
        description=(
            "Research the topic: {topic}\n\n"
            "Step 1: Use the 'DuckDuckGo Search' tool 3 to 5 times with different queries.\n"
            "Step 2: Optionally use the 'Read Webpage' tool (with a URL from the search "
            "results) on 1 or 2 of the best pages. PDF links can't be read, so skip them.\n"
            "Only use these two tools. Never call any other tool such as open_file, "
            "browser or python.\n"
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
    for model_name in MODEL_CANDIDATES:
        for attempt in range(MAX_TOOL_RETRIES + 1):
            try:
                crew = build_crew(build_llm(model_name, api_key))
                result = crew.kickoff(inputs={"topic": topic})
                return getattr(result, "raw", str(result))
            except Exception as e:
                last_error = e
                msg = str(e).lower()

                # 1) Model called a tool that doesn't exist -> just try again.
                if ("tool_use_failed" in msg or "tool call validation failed" in msg) \
                        and attempt < MAX_TOOL_RETRIES:
                    continue

                # 2) Wrong model-name format -> try the next candidate name.
                if any(k in msg for k in ("model", "404", "not found")):
                    break

                # 3) Anything else (bad key, rate limit...) -> show the error.
                raise
    raise last_error
