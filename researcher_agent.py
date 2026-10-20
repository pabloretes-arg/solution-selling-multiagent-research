"""Researcher: searches the web and reports facts and dated news about a company, each with its source."""
from datetime import date, timedelta
from smolagents import CodeAgent
from model import budget_reminder, model
from tools import ResilientSearchTool, SafeVisitWebpageTool

TODAY = date.today()
CUTOFF = TODAY - timedelta(days=365)

RESEARCHER_STEPS = 8

RESEARCHER_INSTRUCTIONS = f"""
You are a company researcher. Today's date is {TODAY.isoformat()}.
Find facts and recent news about the company you are asked about, and report them with sources.

Rules:
- Only report facts you found in your sources. Never invent numbers, names or dates.
- News must be about this company, dated on or after {CUTOFF.isoformat()}. General market data
  (e.g. "e-commerce grew 60% in the country") is context: label it as market context.
- If a group has units in several countries, only report facts about the unit in the requested market.
  Don't mix in founding dates, store counts or figures from other countries or from the whole group.
- If several companies share the name, say which one you researched and which others exist.
- Copy URLs exactly as they appear. Link each news item to its own article.
- Use each article's own publication date. News pages list other stories with other dates
  (sidebars, "latest news"): ignore those. Never use today's date as a guess. If you can't find
  the article's date, write 'undated' instead of a date.
- Be efficient: run several searches or visits in one code block. Prefer news articles:
  online store homepages and PDFs are usually unreadable.

Report format (pass it to final_answer as one string):
COMPANY: which company you researched, and other companies with the same name
FACTS: one fact per line, each with its source URL
NEWS: one item per line, as 'YYYY-MM-DD | Headline | Why it matters for the company's business | URL'
MARKET CONTEXT: optional, one line each, with source URL
"""

researcher = CodeAgent(
    model=model,
    tools=[ResilientSearchTool(max_results=5), SafeVisitWebpageTool(max_output_length=8000)],
    instructions=RESEARCHER_INSTRUCTIONS,
    max_steps=RESEARCHER_STEPS,
    step_callbacks=[budget_reminder(RESEARCHER_STEPS, "Stop searching and pass your report to final_answer now.")],
    name="researcher",
    description=(
        "Researches a company on the web. Give it the company name, the market (country) and what to look for. "
        "It returns facts and dated news, each with its exact source URL."
    ),
    verbosity_level=1,
)


if __name__ == "__main__":
    # Test the researcher on its own
    print(researcher.run("Research Ualá. Market: Argentina. Facts and news from the last 12 months."))