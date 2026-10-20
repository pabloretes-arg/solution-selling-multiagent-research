"""Account Strategy Agent (manager): plans, delegates research to the Researcher Agent and writes the
discovery brief. Before delivering, the brief goes through two checks: it must be the real brief, and the
Sales Manager Agent must approve it."""
import re
from datetime import date
from smolagents import ActionStep, CodeAgent, FinalAnswerTool
from model import budget_reminder, model
from researcher_agent import researcher
from salesmanager_agent import review_brief
from tools import DiscoveryBriefTool

TODAY = date.today()

MANAGER_STEPS = 8

brief_tool = DiscoveryBriefTool()  # kept in a variable so app.py can read the brief

MANAGER_INSTRUCTIONS = f"""
You are an account strategist preparing a B2B seller for a first discovery call.
Today's date is {TODAY.isoformat()}. You don't search the web yourself: the researcher does.

Steps:
1. Ask the researcher for facts and recent news about the company (include the market if given).
2. If something important is missing, ask the researcher one follow-up question at most.
3. Build the brief and return it in the same code block:
   brief = build_discovery_brief(...)
   final_answer(brief)

Rules for the brief:
- Only use facts from the researcher's report. Keep its URLs exactly as written.
- Recent signals: only news about this company from the report, never market statistics.
  Don't repeat the same story twice.
- Likely priorities are the company's own business priorities (growth, products, markets,
  efficiency, risk). Never phrase a priority in terms of what we sell.
- Our angle goes only in where_we_fit: 1-3 hypotheses, each linked to a specific signal.
- Exactly 5 discovery questions, open-ended, from the seller to the buyer:
  the first 2 about the company's goals and initiatives, the last 3 about the area
  where our offering could help. Never pitch inside a question. Never ask for confidential figures.
- If your final answer is rejected by the review, fix exactly what it says and build the brief again.
"""


# ---------- FINAL ANSWER CHECKS ----------
# Run automatically before the final answer is accepted. If one fails, the agent sees why and retries.
MAX_REVIEWS = 2  # after this, the brief is accepted as is (avoids endless loops)


def brief_was_built(final_answer, memory, agent=None) -> bool:
    """The final answer must be the brief produced by the tool, not a summary of it."""
    if brief_tool.last_brief is None or "# Discovery brief" not in str(final_answer):
        raise ValueError("Pass the text returned by build_discovery_brief to final_answer, not a summary.")
    return True


# Generic words that don't reveal a pitch on their own
GENERIC_WORDS = {"software", "platform", "solution", "solutions", "service", "services", "finance", "financial",
                 "teams", "companies", "business", "management", "system", "systems", "tools", "automation"}


def _times_rejected(memory, check_name: str) -> int:
    return sum(1 for s in memory.steps if isinstance(s, ActionStep) and s.error and check_name in str(s.error))


def no_offering_words_in_priorities(final_answer, memory, agent=None) -> bool:
    """Checked in code: a 'priority' that uses the words of what we sell is our pitch, not the client's priority."""
    if _times_rejected(memory, "no_offering_words_in_priorities") >= MAX_REVIEWS:
        return True
    offering = re.search(r"We sell (.+?)\.?$", agent.task if agent else "")
    if not offering:
        return True
    keywords = {w for w in re.findall(r"[a-záéíóúñ]+", offering.group(1).lower()) if len(w) >= 6} - GENERIC_WORDS
    section = str(final_answer).split("## Likely priorities")[-1].split("## Where our offering")[0].split("## Discovery")[0]
    for line in section.splitlines():
        hits = [w for w in keywords if w[:7] in line.lower()]
        if line.startswith("- ") and hits:
            raise ValueError(
                f"Priority '{line[2:]}' uses our offering's words ({', '.join(hits)}). Rewrite it as the client's "
                "own business priority; our angle goes only in where_we_fit."
            )
    return True


def sales_manager_review(final_answer, memory, agent=None) -> bool:
    """The Sales Manager Agent rejects briefs that pitch instead of listening, or that have doubtful facts."""
    if _times_rejected(memory, "sales_manager_review") >= MAX_REVIEWS:
        return True
    try:
        verdict = review_brief(str(final_answer), agent.task if agent else "")
    except Exception as error:
        # If the reviewer itself fails (e.g. the model provider is down), deliver the brief rather than lose it
        print(f"Sales manager review skipped: {error}")
        return True
    if verdict.upper().startswith("FAIL"):
        raise ValueError(f"Sales manager review rejected the brief. {verdict}")
    return True


manager = CodeAgent(
    model=model,
    tools=[brief_tool, FinalAnswerTool()],
    managed_agents=[researcher],
    instructions=MANAGER_INSTRUCTIONS,
    max_steps=MANAGER_STEPS,
    planning_interval=4,  # writes a short plan at step 1 and reviews it at step 5
    final_answer_checks=[brief_was_built, no_offering_words_in_priorities, sales_manager_review],
    step_callbacks=[budget_reminder(MANAGER_STEPS, "Build the brief with what you have and pass it to final_answer now.")],
    # The researcher runs inside the strategist's code, and a web search can take minutes.
    # The default limit (30 s per code block) would kill the research halfway.
    executor_kwargs={"timeout_seconds": 600},
    verbosity_level=1,
)


agent = manager  # app.py runs the strategist; it calls the researcher when it needs it

if __name__ == "__main__":
    agent.run("Research the company Ualá. Our market is Argentina. We sell payment reconciliation software.")