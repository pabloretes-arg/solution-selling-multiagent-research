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

# A call to the provider that takes longer than this is cut and retried (or sent to the backup).
# Without it, a stuck request can hang the whole team forever.
REQUEST_TIMEOUT = 120  # seconds


def build_model(max_tokens: int = 8192, main_id: str = MODEL_ID, fallback_ids: list[str] = FALLBACK_MODEL_IDS):
    if not fallback_ids:
        return LiteLLMModel(model_id=main_id, max_tokens=max_tokens, timeout=REQUEST_TIMEOUT)
    backups = [f"backup-{i}" for i in range(len(fallback_ids))]
    model_list = [{"model_name": "main", "litellm_params": {"model": main_id, "timeout": REQUEST_TIMEOUT}}] + [
        {"model_name": name, "litellm_params": {"model": model_id, "timeout": REQUEST_TIMEOUT}}
        for name, model_id in zip(backups, fallback_ids)
    ]
    return LiteLLMRouterModel(
        model_id="main",
        model_list=model_list,
        client_kwargs={"fallbacks": [{"main": backups}], "num_retries": 1, "timeout": REQUEST_TIMEOUT},
        max_tokens=max_tokens,
    )


# Room to answer: a reasoning model like Nemotron thinks before it writes, and the thinking counts against
# max_tokens. With 4096 it sometimes spent the whole budget thinking and returned nothing: an empty plan,
# an "Error in code parsing" step, or an empty review. It is a cap, not a cost: you pay only what is used.
model = build_model()

# The Sales Manager gets its own instance (same model, same settings), so the app can tell its calls apart
review_model = build_model()

# ---------- TRANSLATION MODEL ----------
# Translating doesn't need a reasoning model: a fast one answers in seconds instead of minutes.
# The app uses two LLMs, each for what it does best: Nemotron for the agents, a fast model for translating.
# TRANSLATION_MODEL_ID in .env chooses it; by default it is the first backup model (e.g. Gemini Flash Lite).
# If it fails, the translation goes to the main model.
TRANSLATION_MODEL_ID = os.getenv("TRANSLATION_MODEL_ID") or (FALLBACK_MODEL_IDS[0] if FALLBACK_MODEL_IDS else MODEL_ID)
translation_model = build_model(
    main_id=TRANSLATION_MODEL_ID,
    fallback_ids=[MODEL_ID] if TRANSLATION_MODEL_ID != MODEL_ID else [],
)

# ---------- STEP BUDGET ----------
# Enforced in code: near its step limit, an agent is told to stop and deliver what it has.
def budget_reminder(max_steps: int, message: str):
    def reminder(step: ActionStep, agent) -> None:
        left = max_steps - step.step_number
        if left <= 3:
            step.observations = (step.observations or "") + f"\n\nBUDGET: only {left} step(s) left. {message}"

    return reminder