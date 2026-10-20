"""Sales Manager Agent: reviews the discovery brief before it reaches the seller.
It is a ToolCallingAgent (actions as JSON tool calls, not code): it only needs to read
the brief, open a source when a fact looks doubtful, and give a verdict."""
from smolagents import ToolCallingAgent
from model import model
from tools import SafeVisitWebpageTool

SALES_MANAGER_STEPS = 4
MAX_SOURCE_CHECKS = 1  # enforced in code: one source per review, then a verdict


class OneSourceCheckTool(SafeVisitWebpageTool):
    """Opens one page per review. After that, it tells the agent to decide instead of browsing more."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checks_left = MAX_SOURCE_CHECKS

    def forward(self, url: str) -> str:
        if self.checks_left <= 0:
            return "You already checked a source. Don't open more pages: give your verdict now with final_answer."
        self.checks_left -= 1
        return super().forward(url)


SALES_MANAGER_INSTRUCTIONS = """
You are a demanding B2B sales manager reviewing a discovery brief written by a junior seller.
Check these rules:
1. "Likely priorities" must be the client's own business priorities. FAIL if any priority is really
   the seller's product or pitch in disguise (e.g. "improve payment reconciliation" when we sell
   reconciliation software).
2. "Discovery questions": the first 2 must be about the client's goals, not the seller's product area,
   and no question may be leading or a pitch (e.g. "What would faster reconciliation mean for you?").
3. Facts: if one number, claim or date in the snapshot or signals looks doubtful (e.g. a figure for the
   whole group attributed to one brand, or news dated today), open its source with visit_webpage and check
   it, including the article's own publication date. Check at most one.
Be strict: approving a weak brief is worse than sending it back.
Write the reason in the same language as the brief.

Give your verdict with final_answer, in exactly one line:
PASS
or
FAIL: <which item breaks which rule, and how to rewrite it>
"""

sales_manager = ToolCallingAgent(
    model=model,
    tools=[OneSourceCheckTool(max_output_length=6000)],
    instructions=SALES_MANAGER_INSTRUCTIONS,
    max_steps=SALES_MANAGER_STEPS,
    name="sales_manager",
    description="Reviews a discovery brief and answers PASS or FAIL with the reason.",
    verbosity_level=1,
)


def review_brief(brief: str, request: str) -> str:
    """Returns the sales manager's verdict: 'PASS' or 'FAIL: ...'."""
    sales_manager.tools["visit_webpage"].checks_left = MAX_SOURCE_CHECKS  # a fresh allowance for each review
    verdict = sales_manager.run(f"The seller's request: {request}\n\nBRIEF TO REVIEW:\n{brief}")
    return str(verdict).strip()


if __name__ == "__main__":
    # Test the sales manager on its own with a brief that pitches in its priorities
    print(review_brief(
        "## Likely priorities\n- Improve payment reconciliation\n## Discovery questions\n"
        "1. What would faster reconciliation mean for you?",
        "We sell payment reconciliation software.",
    ))