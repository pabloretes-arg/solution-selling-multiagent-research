"""Optional observability with Langfuse.

Every step of every agent (what it thought, which tool it used, tokens, time, errors) is recorded
through OpenTelemetry and sent to Langfuse, where you can inspect a whole run as a timeline.
If there are no Langfuse keys in .env, the app works exactly the same, just without tracing."""
import os
from contextlib import contextmanager

langfuse = None  # the Langfuse client, once setup() succeeds


def setup() -> bool:
    """Turns tracing on if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are in .env."""
    global langfuse
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        print("Langfuse: off (no keys in .env)")
        return False
    try:
        from langfuse import get_client
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor

        client = get_client()
        if not client.auth_check():
            print("Langfuse: off (keys or host rejected, check .env)")
            return False
        SmolagentsInstrumentor().instrument()  # records every agent step from now on
        langfuse = client
        print("Langfuse: on")
        return True
    except Exception as error:  # tracing must never break the app
        print(f"Langfuse: off ({error})")
        return False


class _Trace:
    def __init__(self, span=None, url=None):
        self._span, self.url = span, url

    def set_output(self, output) -> None:
        if self._span is not None:
            self._span.update(output=output)


@contextmanager
def trace_run(name: str, task: str):
    """Groups a whole team run under one trace, so the three agents appear together in Langfuse."""
    if langfuse is None:
        yield _Trace()
        return
    with langfuse.start_as_current_observation(name=name, as_type="agent", input=task) as span:
        trace = _Trace(span)
        try:
            trace.url = langfuse.get_trace_url(trace_id=langfuse.get_current_trace_id())
        except Exception:
            trace.url = None
        yield trace
    langfuse.flush()  # send everything now, don't wait for the background export