"""Sales Manager Agent: reviews the discovery brief before it reaches the seller.
It is a ToolCallingAgent (actions as JSON tool calls, not code): it only needs to read
the brief, open a source when a fact looks doubtful, and give a verdict."""
import contextvars
import threading
from smolagents import ToolCallingAgent
from model import review_model
from tools import SafeVisitWebpageTool

SALES_MANAGER_STEPS = 3
MAX_SOURCE_CHECKS = 1  # enforced in code: one source per review, then a verdict
REVIEW_TIMEOUT = 180  # seconds; enforced in code: a stuck review is skipped and the brief is delivered


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
Think briefly: this is a quick review, not a rewrite. Keep your reasoning under 150 words, then answer.

Give your verdict with final_answer, in exactly one line:
PASS
or
FAIL: <which item breaks which rule, and how to rewrite it>
"""

sales_manager = ToolCallingAgent(
    model=review_model,
    tools=[OneSourceCheckTool(max_output_length=3000)],  # the start of an article is enough to verify a fact
    instructions=SALES_MANAGER_INSTRUCTIONS,
    max_steps=SALES_MANAGER_STEPS,
    name="sales_manager",
    description="Reviews a discovery brief and answers PASS or FAIL with the reason.",
    verbosity_level=1,
)


_last_review = None  # the thread of the last review, to never run two reviews on the same agent at once


def review_brief(brief: str, request: str) -> str:
    """Returns the sales manager's verdict: 'PASS' or 'FAIL: ...'."""
    global _last_review
    if _last_review is not None and _last_review.is_alive():
        # A review that timed out may still be finishing its last call; sharing the agent would mix both reviews
        raise RuntimeError("the previous review is still finishing")
    sales_manager.tools["visit_webpage"].checks_left = MAX_SOURCE_CHECKS  # a fresh allowance for each review
    result = {}

    def work():
        try:
            result["verdict"] = sales_manager.run(f"The seller's request: {request}\n\nBRIEF TO REVIEW:\n{brief}")
        except Exception as error:
            result["error"] = error

    # The review runs in its own thread with a deadline, so a slow or stuck model can't freeze the team.
    # copy_context() carries the current trace into that thread: in Langfuse the review stays inside the run.
    worker = threading.Thread(target=contextvars.copy_context().run, args=(work,), daemon=True)
    _last_review = worker
    worker.start()
    worker.join(REVIEW_TIMEOUT)
    if worker.is_alive():
        sales_manager.interrupt()  # it stops at the end of its current step instead of going on
        raise TimeoutError(f"the review took more than {REVIEW_TIMEOUT} seconds")
    if "error" in result:
        raise result["error"]
    return str(result["verdict"]).strip()


if __name__ == "__main__":
    # Test the sales manager on its own with a brief that pitches in its priorities
    print(review_brief(
        "## Likely priorities\n- Improve payment reconciliation\n## Discovery questions\n"
        "1. What would faster reconciliation mean for you?",
        "We sell payment reconciliation software.",
    ))
