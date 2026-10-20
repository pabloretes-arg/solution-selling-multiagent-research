import os
from dotenv import load_dotenv
from smolagents import ActionStep, LiteLLMModel, LiteLLMRouterModel

load_dotenv()

# ---------- MODEL ----------
# The model is read from MODEL_ID in the .env file (see .env.example).
# To switch providers, change the .env, not the code.
MODEL_ID = os.getenv("MODEL_ID")
if not MODEL_ID:
    raise SystemExit("Missing MODEL_ID in .env. Copy .env.example to .env and fill it in.")

# Optional backups, comma-separated (e.g. FALLBACK_MODEL_IDS=gemini/gemini-flash-lite-latest).
# If the main provider fails or is overloaded, the request goes to the next model automatically.
FALLBACK_MODEL_IDS = [m.strip() for m in os.getenv("FALLBACK_MODEL_IDS", "").split(",") if m.strip()]


def build_model(max_tokens: int = 4096):
    if not FALLBACK_MODEL_IDS:
        return LiteLLMModel(model_id=MODEL_ID, max_tokens=max_tokens)
    backups = [f"backup-{i}" for i in range(len(FALLBACK_MODEL_IDS))]
    model_list = [{"model_name": "main", "litellm_params": {"model": MODEL_ID}}] + [
        {"model_name": name, "litellm_params": {"model": model_id}}
        for name, model_id in zip(backups, FALLBACK_MODEL_IDS)
    ]
    return LiteLLMRouterModel(
        model_id="main",
        model_list=model_list,
        client_kwargs={"fallbacks": [{"main": backups}], "num_retries": 1},
        max_tokens=max_tokens,
    )


model = build_model()

# Same model for the Sales Manager, with more room to answer: a reasoning model like Nemotron can spend
# its whole budget thinking and return nothing, which wastes a full step (it happened with 4096 tokens).
review_model = build_model(max_tokens=8192)

# ---------- STEP BUDGET ----------
# Enforced in code: near its step limit, an agent is told to stop and deliver what it has.
def budget_reminder(max_steps: int, message: str):
    def reminder(step: ActionStep, agent) -> None:
        left = max_steps - step.step_number
        if left <= 3:
            step.observations = (step.observations or "") + f"\n\nBUDGET: only {left} step(s) left. {message}"

    return reminder