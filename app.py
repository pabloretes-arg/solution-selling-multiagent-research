import functools
import html
import os
import re
import tempfile
import threading
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
from smolagents.memory import ActionStep, PlanningStep

from accountstrategy_agent import agent, brief_tool
from researcher_agent import researcher
from salesmanager_agent import sales_manager
from translator import translate_brief
from model import (FALLBACK_MODEL_IDS, MODEL_ID, TRANSLATION_MODEL_ID, model, review_model,
                   translation_model)
import tracing

tracing.setup()  # Langfuse, only if its keys are in .env

# ---------- LANGUAGE: every visible text, in English and Spanish ----------
UI_TEXT = {
    "en": {
        "eyebrow": "Solution Selling Intelligence · Multi-Agent",
        "title": "Account research for your first discovery call",
        "intro": "Three agents work as a solution-selling team: the Researcher gathers the facts, the Account "
                 "Strategist writes the brief, and the Sales Manager approves it before it reaches you.",
        "company": "Company", "company_info": "e.g. Ualá, Jumbo, Despegar",
        "market": "Market", "market_info": "For ambiguous names",
        "offering": "What do you sell? (optional)", "offering_info": "e.g. payment reconciliation software for finance teams",
        "button": "Generate brief", "download": "Download brief (.md)", "steps": "How the team worked",
        "need_company": "Enter a company name.",
        "stages": ["Plan", "Research", "Write brief", "Review", "Done"],
        "roles": {"strategist": "Manager · CodeAgent · Plans, delegates and writes the brief",
                  "researcher": "CodeAgent · Searches the web and reads sources",
                  "manager": "ToolCallingAgent · Approves or rejects the brief"},
        "chips": {"working": "Working", "idle": "Not started", "waiting": "Waiting", "done": "Done"},
        "head": {"progress": "In progress", "ready": "Brief ready", "stopped": "Stopped"},
        "no_activity": "No activity yet",
        "thinking": "Thinking… {s}s",
        "trace_link": "Open this run in Langfuse ↗",
        "source_limit": "Source limit reached: deciding with what it has",
        "planning": "Planning the research",
        "delegating": "Delegating the research to the Researcher Agent",
        "starting": "Starting the investigation",
        "report_sent": "Report delivered to the Account Strategy Agent",
        "reading_report": "Reading the research report",
        "searching": "Searching “{q}”",
        "reading": "Reading {site}",
        "verifying": "Verifying a source on {site}",
        "writing": "Writing the brief",
        "self_check": "Self-check: are there pitch words in the priorities?",
        "rejected": "✗ Rejected: {reason}",
        "fixing": "Fixing the brief",
        "reviewing": "Reviewing the brief",
        "approved": "✓ Approved",
        "delivered": "Brief delivered",
        "provider_failed": "✗ The model provider failed (it may be overloaded)",
        "provider_message": "⚠️ The model provider didn't respond (it may be overloaded). Wait a minute and try again.",
        "fallback_tip": " Tip: set FALLBACK_MODEL_IDS in .env so the team switches to a backup model automatically.",
        "no_brief": "⚠️ The team finished without building a brief. Please try again.",
        "translating": "Translating the content of the brief… (the headings are already in English)",
        "models": "Models (LLMs)",
        "llm_roles": {"agents": "Brain of the 3 agents", "backup": "Backup if the main model fails",
                      "translation": "Translates the brief"},
        "llm_now": "Now: {who}", "llm_last": "Last: {who} · {s}s", "llm_calls": "{n} calls", "llm_call": "1 call", "llm_translating": "translation",
        "llm_idle": "Idle", "llm_working": "Working",
        "translate_failed": "The content couldn't be translated, so it stays in the language it was written in. "
                            "To get it in English, generate the brief with English selected.",
    },
    "es": {
        "eyebrow": "Solution Selling Intelligence · Multi-Agente",
        "title": "Investigación de cuentas para tu primera reunión de discovery",
        "intro": "Tres agentes trabajan como un equipo de solution selling: el Researcher reúne los hechos, el "
                 "Account Strategist escribe el brief y el Sales Manager lo aprueba antes de que llegue a vos.",
        "company": "Empresa", "company_info": "ej. Ualá, Jumbo, Despegar",
        "market": "Mercado", "market_info": "Para nombres ambiguos",
        "offering": "¿Qué vendés? (opcional)", "offering_info": "ej. software de conciliación de pagos para equipos de finanzas",
        "button": "Generar brief", "download": "Descargar brief (.md)", "steps": "Cómo trabajó el equipo",
        "need_company": "Ingresá el nombre de una empresa.",
        "stages": ["Plan", "Investigación", "Redacción", "Revisión", "Listo"],
        "roles": {"strategist": "Manager · CodeAgent · Planifica, delega y escribe el brief",
                  "researcher": "CodeAgent · Busca en la web y lee las fuentes",
                  "manager": "ToolCallingAgent · Aprueba o rechaza el brief"},
        "chips": {"working": "Trabajando", "idle": "Sin iniciar", "waiting": "En espera", "done": "Listo"},
        "head": {"progress": "En curso", "ready": "Brief listo", "stopped": "Detenido"},
        "no_activity": "Sin actividad todavía",
        "thinking": "Pensando… {s}s",
        "trace_link": "Ver esta corrida en Langfuse ↗",
        "source_limit": "Límite de fuentes alcanzado: decide con lo que tiene",
        "planning": "Planificando la investigación",
        "delegating": "Delegando la investigación al Researcher Agent",
        "starting": "Iniciando la investigación",
        "report_sent": "Informe entregado al Account Strategy Agent",
        "reading_report": "Leyendo el informe de investigación",
        "searching": "Buscando “{q}”",
        "reading": "Leyendo {site}",
        "verifying": "Verificando una fuente en {site}",
        "writing": "Escribiendo el brief",
        "self_check": "Autocontrol: ¿hay palabras de nuestra oferta en las prioridades?",
        "rejected": "✗ Rechazado: {reason}",
        "fixing": "Corrigiendo el brief",
        "reviewing": "Revisando el brief",
        "approved": "✓ Aprobado",
        "delivered": "Brief entregado",
        "provider_failed": "✗ Falló el proveedor del modelo (puede estar saturado)",
        "provider_message": "⚠️ El proveedor del modelo no respondió (puede estar saturado). Esperá un minuto y probá de nuevo.",
        "fallback_tip": " Consejo: configurá FALLBACK_MODEL_IDS en .env para que el equipo pase a un modelo de respaldo.",
        "no_brief": "⚠️ El equipo terminó sin armar el brief. Probá de nuevo.",
        "translating": "Traduciendo el contenido del brief… (los títulos ya están en español)",
        "models": "Modelos (LLMs)",
        "llm_roles": {"agents": "Cerebro de los 3 agentes", "backup": "Respaldo si falla el modelo principal",
                      "translation": "Traduce el brief"},
        "llm_now": "Ahora: {who}", "llm_last": "Último: {who} · {s}s", "llm_calls": "{n} llamadas", "llm_call": "1 llamada", "llm_translating": "traducción",
        "llm_idle": "En espera", "llm_working": "Trabajando",
        "translate_failed": "No se pudo traducir el contenido, así que queda en el idioma en que se escribió. "
                            "Para tenerlo en español, generá el brief con Español elegido.",
    },
}


def tr(key: str, lang: str | None = None, **values):
    """Text in the given language (by default, the language the run started with)."""
    text = UI_TEXT[lang or STATUS["lang"]][key]
    return text.format(**values) if values else text


def build_task(company: str, market: str, offering: str, lang: str = "en") -> str:
    task = f"Research the company {company.strip()}."
    if market.strip():
        task += (
            f" Our market is {market.strip()}: if several companies share that name, "
            "prefer the one operating there."
        )
    if offering.strip():
        task += f" We sell {offering.strip()}."
    if lang == "es":
        task += (" Write all the content of the brief (snapshot, signals, priorities, fit and questions) "
                 "in Spanish. Keep URLs and dates as they are.")
    return task


def describe_step(step: ActionStep) -> str:
    """One readable block per agent step: code it ran, time, and errors."""
    status = "❌ error" if step.error else "✅"
    duration = getattr(step.timing, "duration", None)
    header = f"**Step {step.step_number}** · {status}" + (f" · {duration:.1f}s" if duration else "")
    code = (step.code_action or "").strip()
    if not code and step.tool_calls:  # ToolCallingAgent steps are JSON tool calls, not code
        code = "\n".join(f"{call.name}({call.arguments})" for call in step.tool_calls)
    block = f"{header}\n```python\n{code}\n```" if code else header
    if step.error:
        first_line = str(step.error).splitlines()[0][:200]
        block += f"\n> {first_line}"
    return block


# The Researcher and Sales Manager agents run inside the strategist's steps:
# collect their steps to show them nested under it.
researcher_steps: list[str] = []


def _track(label: str):
    def track(step: ActionStep, agent=None) -> None:
        researcher_steps.append(describe_step(step).replace("**Step", f"↳ **{label} step", 1))

    return track


researcher.step_callbacks.register(ActionStep, _track("Researcher Agent"))
sales_manager.step_callbacks.register(ActionStep, _track("Sales Manager Agent"))


def save_brief(company: str, brief: str) -> str:
    plain = unicodedata.normalize("NFKD", company).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-") or "company"
    path = Path(tempfile.gettempdir()) / f"discovery-brief-{slug}.md"
    path.write_text(brief.replace("\\$", "$"), encoding="utf-8")
    return str(path)


# ---------- LIVE PROGRESS: which agent is working, and what each one is doing ----------
# The agents run in a background thread. Small wrappers around their tools write to STATUS and LOGS,
# and the interface reads them every second to draw the stage bar and one card per agent.
STAGES = [("plan", "Plan"), ("research", "Research"), ("write", "Write brief"), ("review", "Review"), ("done", "Done")]
AGENTS = [
    ("strategist", "Account Strategy Agent", "Manager · CodeAgent", "Plans, delegates and writes the brief"),
    ("researcher", "Researcher Agent", "CodeAgent", "Searches the web and reads sources"),
    ("manager", "Sales Manager Agent", "ToolCallingAgent", "Approves or rejects the brief"),
]
STATUS = {"stage": "plan", "active": "strategist", "start": time.time(), "lang": "en"}
LOGS = {key: [] for key, *_ in AGENTS}


def log(agent_key: str, key: str, stage: str | None = None, **values) -> None:
    """Records what an agent is doing and marks it as the active one.
    It keeps the text key, not the translated text: the page translates it when it draws it,
    so switching the language also changes what is already on screen."""
    STATUS["active"] = agent_key
    if stage:
        STATUS["stage"] = stage
    LOGS[agent_key].append((time.time() - STATUS["start"], key, values))


def log_text(entry, lang: str) -> str:
    _, key, values = entry
    return tr(key, lang, **values) if key else values["text"]  # no key: an agent's thought, shown as written


def _wrap_tool(tool, agent_key: str, stage: str, describe) -> None:
    original = tool.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        first = next(iter(kwargs.values()), args[0] if args else "")
        key, values = describe(str(first))
        log(agent_key, key, stage, **values)
        return original(*args, **kwargs)

    tool.forward = forward


def _wrap_check(check):
    is_review = check.__name__ == "sales_manager_review"

    @functools.wraps(check)  # keeps the name: each check counts its own rejections by name
    def wrapped(*args, **kwargs):
        if is_review:
            log("manager", "reviewing", "review")
            STEP_MARK["manager"] = len(LOGS["manager"])
        elif check.__name__ == "no_offering_words_in_priorities":
            log("strategist", "self_check", "review")
        try:
            result = check(*args, **kwargs)
        except Exception as error:
            reason = str(error).split("rejected the brief.")[-1].strip().removeprefix("FAIL:").strip()
            reason = reason.split(". Rewrite")[0][:110]
            log("manager" if is_review else "strategist", "rejected", reason=reason)
            log("strategist", "fixing", "write")
            raise
        if is_review:
            log("manager", "approved")
        return result

    return wrapped


def _wrap_researcher_run() -> None:
    original = researcher.run

    @functools.wraps(original)
    def run(*args, **kwargs):
        log("strategist", "delegating")
        log("researcher", "starting", "research")
        STEP_MARK["researcher"] = len(LOGS["researcher"])
        result = original(*args, **kwargs)
        log("researcher", "report_sent")
        log("strategist", "reading_report", "write")
        return result

    researcher.run = run


_wrap_tool(researcher.tools["web_search"], "researcher", "research", lambda q: ("searching", {"q": q[:60]}))
_wrap_tool(researcher.tools["visit_webpage"], "researcher", "research",
           lambda url: ("reading", {"site": urlparse(url).netloc or url[:50]}))
_wrap_tool(sales_manager.tools["visit_webpage"], "manager", "review",
           lambda url: ("verifying", {"site": urlparse(url).netloc or url[:50]})
           if sales_manager.tools["visit_webpage"].checks_left > 0 else ("source_limit", {}))
_wrap_tool(brief_tool, "strategist", "write", lambda _: ("writing", {}))
agent.final_answer_checks = [_wrap_check(check) for check in agent.final_answer_checks]
_wrap_researcher_run()


# ---------- THOUGHTS: what each agent is thinking, step by step ----------
# A step = the model thinks, then acts. The thought only arrives when the step ends, so it is inserted
# where the step began: the log reads in the right order (thought first, then the actions it led to).
STEP_MARK = {key: 0 for key, *_ in AGENTS}


def _first_sentence_of_thought(step: ActionStep) -> str:
    text = str(step.model_output or "")
    text = text.split("<code>")[0].split("```")[0].strip()
    text = re.sub(r"^[\s*#_]*(Thoughts?|Thought process)[\s*_]*:[\s*_]*", "", text, flags=re.IGNORECASE)
    text = text.replace("**", "").strip()
    if text.startswith(("{", "[", "<")):  # a raw tool call (JSON or <tool_call> tags), not a thought
        return ""
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0] if text else ""
    return first[:160]


def _thought_logger(agent_key: str):
    def on_step(step: ActionStep, agent=None) -> None:
        thought = _first_sentence_of_thought(step)
        if thought:
            started = getattr(step.timing, "start_time", None) or time.time()
            LOGS[agent_key].insert(STEP_MARK[agent_key], (max(started - STATUS["start"], 0), None, {"text": f"💭 {thought}"}))
        STEP_MARK[agent_key] = len(LOGS[agent_key])

    return on_step


agent.step_callbacks.register(ActionStep, _thought_logger("strategist"))
researcher.step_callbacks.register(ActionStep, _thought_logger("researcher"))
sales_manager.step_callbacks.register(ActionStep, _thought_logger("manager"))


def reset_progress(lang: str) -> None:
    STATUS.update(stage="plan", active="strategist", start=time.time(), end=None, lang=lang)
    LLM["calls"] = {model_id: 0 for model_id in LLM_IDS}
    LLM["last"] = {}
    for entries in LOGS.values():
        entries.clear()
    log("strategist", "planning", "plan")
    STEP_MARK.update({key: len(entries) for key, entries in LOGS.items()})


# ---------- MODELS: which LLM is answering right now ----------
# Each model call goes through a small wrapper that records which model it is for, and which model
# really answered (the router can switch to the backup). The page shows one box per model.
PROVIDERS = {"nvidia_nim": "NVIDIA", "gemini": "Google", "openai": "OpenAI", "anthropic": "Anthropic",
             "groq": "Groq", "ollama": "Ollama (local)", "mistral": "Mistral"}
LLM_IDS = list(dict.fromkeys([MODEL_ID, *FALLBACK_MODEL_IDS, TRANSLATION_MODEL_ID]))  # unique, in order
LLM = {"active": {}, "calls": {model_id: 0 for model_id in LLM_IDS}, "last": {}}
# active: call id -> (model id, purpose, agent) · last: model id -> (purpose, agent, seconds) of its last call


def _which_llm(answer, expected: str) -> str:
    """The model id that really answered, read from the provider's response."""
    answered = str(getattr(getattr(answer, "raw", None), "model", "") or "")
    for model_id in LLM_IDS:
        name = model_id.split("/")[-1]
        if answered and (name in answered or answered in model_id):
            return model_id
    return expected


def _watch_llm(llm, expected: str, purpose: str) -> None:
    original = llm.generate

    @functools.wraps(original)
    def generate(*args, **kwargs):
        call, started = object(), time.time()
        LLM["active"][call] = (expected, purpose, STATUS["active"])
        try:
            answer = original(*args, **kwargs)
            used = _which_llm(answer, expected)
            LLM["calls"][used] = LLM["calls"].get(used, 0) + 1
            # A fast call can end between two refreshes of the page: the box keeps its last job and how long it took
            LLM["last"][used] = (purpose, STATUS["active"], time.time() - started)
            return answer
        finally:
            LLM["active"].pop(call, None)

    llm.generate = generate


_watch_llm(model, MODEL_ID, "agents")
_watch_llm(review_model, MODEL_ID, "manager")
_watch_llm(translation_model, TRANSLATION_MODEL_ID, "translation")


def render_models(lang: str) -> str:
    agent_names = {key: name for key, name, *_ in AGENTS}
    boxes = []
    for model_id in LLM_IDS:
        provider = PROVIDERS.get(model_id.split("/")[0], model_id.split("/")[0])
        roles = tr("llm_roles", lang)
        role = " · ".join(
            [roles["agents"]] * (model_id == MODEL_ID)
            + [roles["backup"]] * (model_id in FALLBACK_MODEL_IDS)
            + [roles["translation"]] * (model_id == TRANSLATION_MODEL_ID)
        )
        def who(purpose, agent_key):
            if purpose == "translation":
                return tr("llm_translating", lang)
            return agent_names["manager" if purpose == "manager" else agent_key]

        jobs = [who(p, a) for used, p, a in list(LLM["active"].values()) if used == model_id]
        working = bool(jobs)
        if working:
            now = tr("llm_now", lang, who=", ".join(dict.fromkeys(jobs)))
        elif model_id in LLM["last"]:
            purpose, agent_key, seconds = LLM["last"][model_id]
            now = tr("llm_last", lang, who=who(purpose, agent_key), s=round(seconds))
        else:
            now = ""
        chip = tr("llm_working", lang) if working else tr("llm_idle", lang)
        n = LLM["calls"].get(model_id, 0)
        calls = tr("llm_call", lang) if n == 1 else tr("llm_calls", lang, n=n)
        boxes.append(
            f'<div class="llm {"working" if working else ""}"><div class="top"><div>'
            f'<div class="nm">{html.escape(model_id.split("/")[-1])}</div>'
            f'<div class="kd">{html.escape(provider)} · {html.escape(role)}</div></div>'
            f'<span class="chip">{chip}</span></div>'
            f'<div class="meta"><span>{calls}</span>'
            f'<span class="now">{html.escape(now)}</span></div></div>'
        )
    return f'<div class="llms-title">{tr("models", lang)}</div><div class="llms">{"".join(boxes)}</div>'


def render_status(lang: str, failed: bool = False) -> str:
    elapsed = (STATUS.get("end") or time.time()) - STATUS["start"]  # the clock stops when the run ends
    finished = STATUS["stage"] == "done" or failed
    keys = [key for key, _ in STAGES]
    current = keys.index(STATUS["stage"])
    cells = "".join(
        f'<div class="stg {"done" if i < current or STATUS["stage"] == "done" else ("active" if i == current else "")}">'
        f'<span class="n">{i + 1:02d}</span><span class="l">{label}</span></div>'
        for i, label in enumerate(tr("stages", lang))
    )
    cards = []
    for key, name, *_ in AGENTS:
        working = key == STATUS["active"] and not finished
        state = "working" if working else ("idle" if not LOGS[key] else "waiting")
        chips = tr("chips", lang)
        chip = {"working": chips["working"], "idle": chips["idle"],
                "waiting": chips["done"] if finished else chips["waiting"]}[state]
        lines = "".join(
            f'<li><span class="ts">{int(entry[0]) // 60}:{int(entry[0]) % 60:02d}</span>{html.escape(log_text(entry, lang))}</li>'
            for entry in LOGS[key][-6:]
        ) or f'<li class="empty">{tr("no_activity", lang)}</li>'
        idle_for = int(elapsed - max((entry[0] for entry in LOGS[key]), default=elapsed))
        if working and idle_for >= 4:  # the model is generating its next thought
            lines += f'<li class="thinking">{tr("thinking", lang, s=idle_for)}</li>'
        cards.append(
            f'<div class="card {state}"><div class="top"><div><div class="nm">{name}</div>'
            f'<div class="kd">{tr("roles", lang)[key]}</div></div><span class="chip">{chip}</span></div><ul>{lines}</ul></div>'
        )
    minutes, seconds = divmod(int(elapsed), 60)
    heads = tr("head", lang)
    note = heads["ready"] if STATUS["stage"] == "done" else (heads["stopped"] if failed else heads["progress"])
    return (
        f'<div id="prog"><div class="head"><span>{note}</span><span class="t">{minutes}:{seconds:02d}</span></div>'
        f'<div class="stgs">{cells}</div><div class="cards">{"".join(cards)}</div>{render_models(lang)}</div>'
    )


# ---------- THE RUN: the team works in the background, the page asks how it is going every second ----------
# The run doesn't depend on the browser: if the page stops updating (another tab, the laptop sleeps,
# a dropped connection), the next tick or a page reload shows where the team is, and the brief when ready.
# Everything is drawn in the language on screen when it is shown, so the language switch works at any time.
# The content the agents write (snapshot, priorities, questions...) is in the language of the run;
# if the user switches language afterwards, it is translated once in the background and kept.
RUN = {"running": False, "started": False, "steps": [], "company": "", "brief_args": None,
       "message": None, "trace_url": None, "failed": False,
       "content_lang": "en", "translations": {}, "translating": set(), "translate_failed": set()}


def _translate(lang: str, args: dict, translations: dict, translating: set, failed: set) -> None:
    # It receives this run's containers: if a new run starts meanwhile, this result can't land in it
    try:
        translations[lang] = translate_brief(args, lang)
    except Exception as error:  # the original content stays on screen, with a note
        print(f"Translation failed: {error}")
        failed.add(lang)
    finally:
        translating.discard(lang)


def brief_content(lang: str) -> tuple[dict, str]:
    """The brief's content for the language on screen, and a note if it is still being translated."""
    args = RUN["brief_args"]
    if lang == RUN["content_lang"]:
        return args, ""
    if lang in RUN["translations"]:
        return RUN["translations"][lang], ""
    if lang in RUN["translate_failed"]:
        return args, tr("translate_failed", lang)
    if lang not in RUN["translating"]:  # first time in this language: translate it in the background
        RUN["translating"].add(lang)
        threading.Thread(target=_translate, daemon=True, args=(
            lang, args, RUN["translations"], RUN["translating"], RUN["translate_failed"])).start()
    return args, tr("translating", lang)


def _team_work(company: str, task: str) -> None:
    """Runs the whole team in a background thread and leaves the result in RUN."""
    delivered = False
    try:
        with tracing.trace_run("discovery-brief", task) as trace:
            RUN["trace_url"] = trace.url
            for event in agent.run(task, stream=True):
                if isinstance(event, PlanningStep):
                    plan = event.plan.strip()
                    if re.search(r"```\s*(None)?\s*```", plan):
                        # A reasoning model sometimes spends its whole answer thinking and returns no plan text
                        plan = "_The model returned an empty plan this time. The agent goes on without it._"
                    RUN["steps"].append(f"**Plan**\n\n{plan}")
                elif isinstance(event, ActionStep):
                    RUN["steps"].extend(researcher_steps)
                    researcher_steps.clear()
                    RUN["steps"].append(describe_step(event).replace("**Step", "**Account Strategy Agent step", 1))
            trace.set_output(brief_tool.last_brief)
            if brief_tool.last_brief:
                RUN["brief_args"] = dict(brief_tool.last_args)
                log("strategist", "delivered", "done")
                # The other language is prepared right away in the background (one extra model call),
                # so switching the language later shows it at once
                other = "es" if RUN["content_lang"] == "en" else "en"
                RUN["translating"].add(other)
                threading.Thread(target=_translate, daemon=True, args=(
                    other, RUN["brief_args"], RUN["translations"], RUN["translating"], RUN["translate_failed"])).start()
            else:
                RUN.update(message="no_brief", failed=True)
            # Delivered before Langfuse sends its data (on leaving this block): tracing can never delay the brief
            delivered = True
            STATUS["end"] = time.time()
            RUN["running"] = False
    except Exception as error:  # usually the model provider is down or rate-limited
        print(f"Agent run failed: {error}")
        if not delivered:
            log(STATUS["active"], "provider_failed")
            RUN.update(message="provider_message", failed=True)
    finally:
        STATUS["end"] = STATUS.get("end") or time.time()
        RUN["running"] = False


def show_run(lang: str):
    """What the page shows right now, in the language on screen.
    Called every second while the team works, when the page loads and when the language changes."""
    hidden = gr.update(visible=False)
    download_label = tr("download", lang)
    if not RUN["started"]:
        return hidden, "", gr.update(visible=False, label=download_label), hidden, gr.Timer(active=False)
    running = RUN["running"]
    report, download = hidden, gr.update(visible=False, label=download_label)
    if RUN["brief_args"]:
        args, note = brief_content(lang)
        brief = brief_tool.render(lang, args)
        report = gr.update(value=f"> _{note}_\n\n{brief}" if note else brief, visible=True)
        download = gr.update(value=save_brief(RUN["company"], brief), visible=True, label=download_label)
    elif RUN["message"]:
        message = tr(RUN["message"], lang)
        if RUN["message"] == "provider_message" and not os.getenv("FALLBACK_MODEL_IDS"):
            message += tr("fallback_tip", lang)
        report = gr.update(value=message, visible=True)
    steps = list(RUN["steps"])
    if RUN["trace_url"] and not running:
        steps.insert(0, f"[{tr('trace_link', lang)}]({RUN['trace_url']})")
    progress = gr.update(value=render_status(lang, failed=RUN["failed"]), visible=True)
    busy = running or bool(RUN["translating"])  # keep asking every second until the brief is final
    return report, "\n\n".join(steps), download, progress, gr.Timer(active=busy)


def start_run(company: str, market: str, offering: str, lang: str):
    """Starts the team in the background and returns at once; the timer takes care of the updates."""
    if RUN["running"]:  # a second click while the team works: just keep showing the current run
        return show_run(lang)
    if not company.strip():
        raise gr.Error(tr("need_company", lang))

    brief_tool.last_brief = None
    brief_tool.language = lang  # the agents write the brief in the language chosen when the run starts
    researcher_steps.clear()
    reset_progress(lang)
    RUN.update(running=True, started=True, steps=[], company=company, brief_args=None, message=None,
               trace_url=None, failed=False, content_lang=lang, translations={}, translating=set(),
               translate_failed=set())
    task = build_task(company, market, offering, lang)
    threading.Thread(target=_team_work, args=(company, task), daemon=True).start()
    return show_run(lang)


def header_html(lang: str) -> str:
    return (
        f'<div id="hdr"><div class="eyebrow">{tr("eyebrow", lang)}</div>'
        f'<h1>{tr("title", lang)}</h1><p>{tr("intro", lang)}</p><hr></div>'
    )


def switch_language(lang: str):
    """Updates every visible text when the language switch changes, including the progress and the brief."""
    return (
        header_html(lang),
        gr.update(label=tr("company", lang), info=tr("company_info", lang)),
        gr.update(label=tr("market", lang), info=tr("market_info", lang)),
        gr.update(label=tr("offering", lang), info=tr("offering_info", lang)),
        gr.update(value=tr("button", lang)),
        gr.update(label=tr("steps", lang)),
        *show_run(lang),
    )


# ---------- LOOK & FEEL: editorial, "strategy consulting report" style ----------
INK, PAPER, RULE, ACCENT, MUTED = "#0B1F3A", "#F7F5F0", "#DDD8CC", "#9A7B4F", "#6B6F76"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
html, body, gradio-app, .gradio-container, .main, .wrap {{background: {PAPER} !important;}}
.gradio-container {{max-width: 920px !important; margin: 0 auto !important; padding-top: 32px !important;}}
footer {{display: none !important;}}

/* Language switch */
#langrow {{background: transparent !important; border: 0 !important; box-shadow: none !important; gap: 0 !important;}}
#langrow > div, #langrow .block, #langrow fieldset {{background: transparent !important; border: 0 !important; box-shadow: none !important;}}
#lang {{margin-left: auto; background: transparent !important; border: 0 !important; box-shadow: none !important;}}
#lang .wrap {{gap: 0 !important; justify-content: flex-end;}}
#lang label {{border: 1px solid {RULE} !important; border-radius: 0 !important; padding: 6px 14px !important;
          background: #fff !important; font-size: 11px !important; letter-spacing: .12em; text-transform: uppercase;
          font-weight: 600 !important; color: {MUTED} !important; margin: 0 !important;}}
#lang label.selected, #lang label:has(input:checked) {{background: {INK} !important; color: #fff !important; border-color: {INK} !important;}}
#lang input {{display: none !important;}}

/* Header */
#hdr .eyebrow {{font-size: 11px; letter-spacing: .18em; text-transform: uppercase; color: {ACCENT}; font-weight: 600;}}
#hdr h1 {{font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 38px; line-height: 1.15;
          color: {INK}; margin: 10px 0 12px;}}
#hdr p {{color: {MUTED}; font-size: 15px; max-width: 640px; margin: 0;}}
#hdr hr {{border: 0; border-top: 1px solid {INK}; margin: 28px 0 8px; opacity: .9;}}

/* Form */
#form {{background: transparent !important; border: 0 !important; box-shadow: none !important; gap: 24px !important;}}
#form textarea, #form input {{background: #fff !important; border: 1px solid {RULE} !important; border-radius: 0 !important;
          box-shadow: none !important; font-size: 15px !important;}}
#form label span {{font-size: 11px !important; letter-spacing: .14em; text-transform: uppercase; color: {INK} !important;
          font-weight: 600 !important;}}
#run-btn {{background: {INK} !important; color: #fff !important; border: 0 !important; border-radius: 0 !important;
          letter-spacing: .14em; text-transform: uppercase; font-size: 12px !important; font-weight: 600 !important;
          padding: 14px 28px !important;}}
#run-btn:hover {{background: #1C3A66 !important;}}

/* Brief: a printed report page */
#brief {{background: #fff; border: 1px solid {RULE}; padding: 48px 56px !important; margin-top: 16px;}}
#brief h1 {{font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 30px; color: {INK};
          border-bottom: 1px solid {INK}; padding-bottom: 14px; margin-bottom: 6px;}}
#brief h2 {{font-size: 11px !important; letter-spacing: .18em; text-transform: uppercase; color: {ACCENT} !important;
          font-weight: 700; border-top: 1px solid {RULE}; padding-top: 22px; margin-top: 30px;}}
#brief p, #brief li {{font-family: 'Source Serif 4', Georgia, serif; font-size: 16.5px; line-height: 1.6; color: #22262B;}}
#brief em {{color: {MUTED};}}
#brief blockquote {{border-left: 2px solid {ACCENT}; background: transparent; padding: 2px 0 2px 16px; margin: 8px 0 14px;}}
#brief blockquote p {{font-family: inherit; font-size: 13.5px; color: {MUTED}; font-style: italic;}}
#brief a {{color: {INK}; text-decoration: underline; text-underline-offset: 3px;}}
#brief strong {{color: {INK};}}

#dl {{background: transparent !important; border: 1px solid {INK} !important; color: {INK} !important; border-radius: 0 !important;
       font-size: 11px !important; letter-spacing: .14em; text-transform: uppercase; font-weight: 600 !important; padding: 10px 20px !important;}}

/* Progress: stage bar + one card per agent */
#prog {{margin-top: 18px; background: #fff; border: 1px solid {RULE}; padding: 18px 22px 22px;}}
#prog .head {{display: flex; justify-content: space-between; font-size: 11px; letter-spacing: .16em;
          text-transform: uppercase; color: {MUTED}; font-weight: 600; margin-bottom: 12px;}}
#prog .t {{font-variant-numeric: tabular-nums;}}
#prog .stgs {{display: flex; gap: 6px;}}
#prog .stg {{flex: 1; border-top: 3px solid {RULE}; padding-top: 8px; color: #A39E93;}}
#prog .stg.done {{border-color: {INK}; color: {INK};}}
#prog .stg.active {{border-color: {ACCENT}; color: {ACCENT};}}
#prog .n {{font-size: 11px; font-weight: 700; margin-right: 6px;}}
#prog .l {{font-size: 11px; letter-spacing: .14em; text-transform: uppercase; font-weight: 600;}}
#prog .cards {{display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 20px;}}
@media (max-width: 760px) {{ #prog .cards {{grid-template-columns: 1fr;}} }}
#prog .card {{border: 1px solid {RULE}; padding: 14px 14px 10px; min-height: 190px; transition: all .3s;
          opacity: .55; background: {PAPER};}}
#prog .card.working {{opacity: 1; background: #fff; border-color: {ACCENT};
          box-shadow: 0 0 0 1px {ACCENT}, 0 8px 24px rgba(154, 123, 79, .22);}}
#prog .card.waiting {{opacity: .8;}}
#prog .top {{display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;}}
#prog .nm {{font-family: 'Source Serif 4', Georgia, serif; font-size: 17px; font-weight: 600; color: {INK};}}
#prog .kd {{font-size: 10.5px; color: {MUTED}; margin-top: 3px; line-height: 1.35;}}
#prog .chip {{font-size: 9.5px; letter-spacing: .12em; text-transform: uppercase; font-weight: 700; white-space: nowrap;
          padding: 3px 7px; border: 1px solid {RULE}; color: {MUTED};}}
#prog .card.working .chip {{border-color: {ACCENT}; color: #fff; background: {ACCENT}; animation: pulse 1.4s infinite;}}
#prog ul {{list-style: none; padding: 0; margin: 12px 0 0; font-size: 12.5px; color: #22262B;}}
#prog li {{padding: 4px 0; border-top: 1px dotted {RULE}; line-height: 1.35; overflow-wrap: anywhere;}}
#prog li .ts {{color: #A39E93; font-variant-numeric: tabular-nums; margin-right: 8px; font-size: 11px;}}
#prog li.empty {{color: #A39E93; font-style: italic;}}
#prog li.thinking {{color: {ACCENT}; font-style: italic; animation: pulse 1.4s infinite;}}
#prog .card.working li:last-child:not(.thinking) {{font-weight: 600; color: {INK};}}
@keyframes pulse {{0%, 100% {{opacity: .55;}} 50% {{opacity: 1;}}}}

/* Models (LLMs): one small box per model, the one answering is highlighted */
#prog .llms-title {{margin-top: 18px; font-size: 10.5px; letter-spacing: .16em; text-transform: uppercase;
          color: {MUTED}; font-weight: 600;}}
#prog .llms {{display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px;}}
#prog .llm {{flex: 1 1 200px; border: 1px solid {RULE}; padding: 10px 12px; background: {PAPER}; opacity: .7;
          transition: all .3s;}}
#prog .llm.working {{opacity: 1; background: #fff; border-color: {INK}; box-shadow: 0 0 0 1px {INK};}}
#prog .llm .nm {{font-size: 13px;}}
#prog .llm.working .chip {{border-color: {INK}; color: #fff; background: {INK}; animation: pulse 1.4s infinite;}}
#prog .llm .meta {{display: flex; justify-content: space-between; gap: 8px; margin-top: 8px; font-size: 11px; color: {MUTED};}}
#prog .llm .now {{color: {INK}; font-weight: 600; text-align: right;}}

/* Steps panel and footer */
#steps {{border: 1px solid {RULE} !important; border-radius: 0 !important; background: #fff !important;}}
#author, #author * {{text-align: center; color: {MUTED} !important; font-size: 12px !important; letter-spacing: .08em;}}
#author {{margin-top: 32px;}}
"""

THEME = gr.themes.Base(
    primary_hue="slate",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    radius_size="none",
)

# Starting language: DEFAULT_LANGUAGE=es in .env opens the app in Spanish
START_LANG = os.getenv("DEFAULT_LANGUAGE", "en").lower()
START_LANG = START_LANG if START_LANG in UI_TEXT else "en"

with gr.Blocks(title="Solution Selling Multi-Agent Research") as demo:
    with gr.Row(elem_id="langrow"):
        gr.HTML("")
        lang = gr.Radio(
            choices=[("English", "en"), ("Español", "es")], value=START_LANG,
            label="Language", show_label=False, elem_id="lang", scale=0, min_width=220,
        )
    header = gr.HTML(header_html(START_LANG))

    with gr.Row(elem_id="form"):
        company = gr.Textbox(
            label=tr("company", START_LANG),
            info=tr("company_info", START_LANG),
            scale=2,
        )
        market = gr.Textbox(
            label=tr("market", START_LANG),
            value=os.getenv("DEFAULT_MARKET", ""),  # set your default in .env
            info=tr("market_info", START_LANG),
            scale=1,
        )
        offering = gr.Textbox(
            label=tr("offering", START_LANG),
            info=tr("offering_info", START_LANG),
            scale=3,
        )
    with gr.Row():
        run_btn = gr.Button(tr("button", START_LANG), elem_id="run-btn", size="md", scale=0, min_width=200)

    progress = gr.HTML(visible=False)
    brief_md = gr.Markdown(latex_delimiters=[], elem_id="brief", visible=False)  # "$" = money, not math
    with gr.Row():
        download = gr.DownloadButton(tr("download", START_LANG), visible=False, elem_id="dl", scale=0, min_width=220)

    with gr.Accordion(tr("steps", START_LANG), open=False, elem_id="steps") as steps_box:
        steps_md = gr.Markdown(latex_delimiters=[])

    gr.Markdown("Pablo Retes · Buenos Aires", elem_id="author")

    timer = gr.Timer(1.0, active=False)
    run_outputs = [brief_md, steps_md, download, progress, timer]
    for trigger in (run_btn.click, company.submit, market.submit, offering.submit):
        trigger(start_run, inputs=[company, market, offering, lang], outputs=run_outputs, show_progress="hidden")
    timer.tick(show_run, inputs=lang, outputs=run_outputs, show_progress="hidden")
    demo.load(show_run, inputs=lang, outputs=run_outputs, show_progress="hidden")  # after a reload, pick up the run

    lang.change(switch_language, inputs=lang,
                outputs=[header, company, market, offering, run_btn, steps_box, *run_outputs], show_progress="hidden")

if __name__ == "__main__":
    demo.launch(share=False, theme=THEME, css=CSS)