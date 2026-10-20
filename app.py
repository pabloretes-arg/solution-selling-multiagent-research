import functools
import html
import os
import queue
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
    },
}


def tr(key: str, lang: str | None = None, **values):
    """Text in the current language (the one chosen when the run started)."""
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
# and the interface reads them every half second to draw the stage bar and one card per agent.
STAGES = [("plan", "Plan"), ("research", "Research"), ("write", "Write brief"), ("review", "Review"), ("done", "Done")]
AGENTS = [
    ("strategist", "Account Strategy Agent", "Manager · CodeAgent", "Plans, delegates and writes the brief"),
    ("researcher", "Researcher Agent", "CodeAgent", "Searches the web and reads sources"),
    ("manager", "Sales Manager Agent", "ToolCallingAgent", "Approves or rejects the brief"),
]
STATUS = {"stage": "plan", "active": "strategist", "start": time.time(), "lang": "en"}
LOGS = {key: [] for key, *_ in AGENTS}


def log(agent_key: str, text: str, stage: str | None = None) -> None:
    """Records what an agent is doing and marks it as the active one."""
    STATUS["active"] = agent_key
    if stage:
        STATUS["stage"] = stage
    LOGS[agent_key].append((time.time() - STATUS["start"], text))


def _wrap_tool(tool, agent_key: str, stage: str, describe) -> None:
    original = tool.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        first = next(iter(kwargs.values()), args[0] if args else "")
        log(agent_key, describe(str(first)), stage)
        return original(*args, **kwargs)

    tool.forward = forward


def _wrap_check(check):
    is_review = check.__name__ == "sales_manager_review"

    @functools.wraps(check)  # keeps the name: each check counts its own rejections by name
    def wrapped(*args, **kwargs):
        if is_review:
            log("manager", tr("reviewing"), "review")
            STEP_MARK["manager"] = len(LOGS["manager"])
        elif check.__name__ == "no_offering_words_in_priorities":
            log("strategist", tr("self_check"), "review")
        try:
            result = check(*args, **kwargs)
        except Exception as error:
            reason = str(error).split("rejected the brief.")[-1].strip().removeprefix("FAIL:").strip()
            reason = reason.split(". Rewrite")[0][:110]
            log("manager" if is_review else "strategist", tr("rejected", reason=reason))
            log("strategist", tr("fixing"), "write")
            raise
        if is_review:
            log("manager", tr("approved"))
        return result

    return wrapped


def _wrap_researcher_run() -> None:
    original = researcher.run

    @functools.wraps(original)
    def run(*args, **kwargs):
        log("strategist", tr("delegating"))
        log("researcher", tr("starting"), "research")
        STEP_MARK["researcher"] = len(LOGS["researcher"])
        result = original(*args, **kwargs)
        log("researcher", tr("report_sent"))
        log("strategist", tr("reading_report"), "write")
        return result

    researcher.run = run


_wrap_tool(researcher.tools["web_search"], "researcher", "research", lambda q: tr("searching", q=q[:60]))
_wrap_tool(researcher.tools["visit_webpage"], "researcher", "research",
           lambda url: tr("reading", site=urlparse(url).netloc or url[:50]))
_wrap_tool(sales_manager.tools["visit_webpage"], "manager", "review",
           lambda url: tr("verifying", site=urlparse(url).netloc or url[:50])
           if sales_manager.tools["visit_webpage"].checks_left > 0 else tr("source_limit"))
_wrap_tool(brief_tool, "strategist", "write", lambda _: tr("writing"))
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
    if text.startswith(("{", "[")):  # a raw tool call, not a thought
        return ""
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0] if text else ""
    return first[:160]


def _thought_logger(agent_key: str):
    def on_step(step: ActionStep, agent=None) -> None:
        thought = _first_sentence_of_thought(step)
        if thought:
            started = getattr(step.timing, "start_time", None) or time.time()
            LOGS[agent_key].insert(STEP_MARK[agent_key], (max(started - STATUS["start"], 0), f"💭 {thought}"))
        STEP_MARK[agent_key] = len(LOGS[agent_key])

    return on_step


agent.step_callbacks.register(ActionStep, _thought_logger("strategist"))
researcher.step_callbacks.register(ActionStep, _thought_logger("researcher"))
sales_manager.step_callbacks.register(ActionStep, _thought_logger("manager"))


def reset_progress(lang: str) -> None:
    STATUS.update(stage="plan", active="strategist", start=time.time(), lang=lang)
    for entries in LOGS.values():
        entries.clear()
    log("strategist", tr("planning"), "plan")
    STEP_MARK.update({key: len(entries) for key, entries in LOGS.items()})


def render_status(failed: bool = False) -> str:
    elapsed = time.time() - STATUS["start"]
    finished = STATUS["stage"] == "done" or failed
    keys = [key for key, _ in STAGES]
    current = keys.index(STATUS["stage"])
    cells = "".join(
        f'<div class="stg {"done" if i < current or STATUS["stage"] == "done" else ("active" if i == current else "")}">'
        f'<span class="n">{i + 1:02d}</span><span class="l">{label}</span></div>'
        for i, label in enumerate(tr("stages"))
    )
    cards = []
    for key, name, *_ in AGENTS:
        working = key == STATUS["active"] and not finished
        state = "working" if working else ("idle" if not LOGS[key] else "waiting")
        chips = tr("chips")
        chip = {"working": chips["working"], "idle": chips["idle"],
                "waiting": chips["done"] if finished else chips["waiting"]}[state]
        lines = "".join(
            f'<li><span class="ts">{int(t) // 60}:{int(t) % 60:02d}</span>{html.escape(text)}</li>'
            for t, text in LOGS[key][-6:]
        ) or f'<li class="empty">{tr("no_activity")}</li>'
        idle_for = int(elapsed - max((t for t, _ in LOGS[key]), default=elapsed))
        if working and idle_for >= 4:  # the model is generating its next thought
            lines += f'<li class="thinking">{tr("thinking", s=idle_for)}</li>'
        cards.append(
            f'<div class="card {state}"><div class="top"><div><div class="nm">{name}</div>'
            f'<div class="kd">{tr("roles")[key]}</div></div><span class="chip">{chip}</span></div><ul>{lines}</ul></div>'
        )
    minutes, seconds = divmod(int(elapsed), 60)
    heads = tr("head")
    note = heads["ready"] if STATUS["stage"] == "done" else (heads["stopped"] if failed else heads["progress"])
    return (
        f'<div id="prog"><div class="head"><span>{note}</span><span class="t">{minutes}:{seconds:02d}</span></div>'
        f'<div class="stgs">{cells}</div><div class="cards">{"".join(cards)}</div></div>'
    )


def _run_agent(company: str, market: str, offering: str, lang: str):
    """Yields (brief or None, steps, download update, progress html)."""
    if not company.strip():
        raise gr.Error(tr("need_company", lang))

    brief_tool.last_brief = None
    researcher_steps.clear()
    reset_progress(lang)
    brief_tool.language = lang
    steps = []
    hidden = gr.update(visible=False)
    events = queue.Queue()
    task = build_task(company, market, offering, lang)

    def work():
        try:
            for event in agent.run(task, stream=True):
                events.put(("event", event))
            events.put(("done", None))
        except Exception as error:  # usually the model provider is down or rate-limited
            events.put(("error", error))

    threading.Thread(target=work, daemon=True).start()

    while True:
        try:
            kind, payload = events.get(timeout=0.5)
        except queue.Empty:
            kind, payload = None, None

        if kind == "event" and isinstance(payload, PlanningStep):
            steps.append(f"**Plan**\n\n{payload.plan.strip()}")
        elif kind == "event" and isinstance(payload, ActionStep):
            steps.extend(researcher_steps)
            researcher_steps.clear()
            steps.append(describe_step(payload).replace("**Step", "**Account Strategy Agent step", 1))
        elif kind == "error":
            print(f"Agent run failed: {payload}")
            log(STATUS["active"], tr("provider_failed"))
            message = tr("provider_message")
            if not os.getenv("FALLBACK_MODEL_IDS"):
                message += tr("fallback_tip")
            yield message, "\n\n".join(steps), hidden, render_status(failed=True)
            return
        elif kind == "done":
            break

        yield None, "\n\n".join(steps), hidden, render_status()

    brief = brief_tool.last_brief
    if not brief:
        message = tr("no_brief")
        yield message, "\n\n".join(steps), hidden, render_status(failed=True)
        return

    log("strategist", tr("delivered"), "done")
    download = gr.update(value=save_brief(company, brief), visible=True)
    yield brief, "\n\n".join(steps), download, render_status()


def run_agent(company: str, market: str, offering: str, lang: str):
    """Shows the progress while the team works, and the report box once there is something to show.
    The language switch is locked during the run: the run keeps the language it started with."""
    locked = gr.update(interactive=False)
    for brief, steps, download, progress in _run_agent(company, market, offering, lang):
        report = gr.update() if brief is None else gr.update(value=brief, visible=True)
        yield report, steps, download, gr.update(value=progress, visible=True), locked
    yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=True)


def header_html(lang: str) -> str:
    return (
        f'<div id="hdr"><div class="eyebrow">{tr("eyebrow", lang)}</div>'
        f'<h1>{tr("title", lang)}</h1><p>{tr("intro", lang)}</p><hr></div>'
    )


def switch_language(lang: str):
    """Updates every visible text when the language switch changes."""
    return (
        header_html(lang),
        gr.update(label=tr("company", lang), info=tr("company_info", lang)),
        gr.update(label=tr("market", lang), info=tr("market_info", lang)),
        gr.update(label=tr("offering", lang), info=tr("offering_info", lang)),
        gr.update(value=tr("button", lang)),
        gr.update(label=tr("download", lang)),
        gr.update(label=tr("steps", lang)),
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
            show_label=False, elem_id="lang", scale=0, min_width=220,
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

    for trigger in (run_btn.click, company.submit, market.submit, offering.submit):
        trigger(run_agent, inputs=[company, market, offering, lang],
                outputs=[brief_md, steps_md, download, progress, lang])

    lang.change(switch_language, inputs=lang,
                outputs=[header, company, market, offering, run_btn, download, steps_box])

if __name__ == "__main__":
    demo.launch(share=False, theme=THEME, css=CSS)