from datetime import date, timedelta
import time

from smolagents import DuckDuckGoSearchTool, Tool, VisitWebpageTool


def _parse_date(signal: str):
    """Reads the 'YYYY-MM-DD' at the start of a signal. Returns None if it can't."""
    try:
        return date.fromisoformat(signal.split("|")[0].strip())
    except ValueError:
        return None


# Fixed texts of the brief, in each language the app offers
BRIEF_TEXT = {
    "en": {
        "title": "Discovery brief",
        "generated": "Generated on {day} from public sources. Verify before using with a client.",
        "snapshot": "Company snapshot",
        "signals": "Recent signals (last 12 months)",
        "no_signals": "No signals found in the last 12 months.",
        "excluded": "Excluded (older than 12 months or undated):",
        "priorities": "Likely priorities",
        "priorities_note": "These are hypotheses inferred from public signals, to be validated in the call.",
        "fit": "Where our offering could fit",
        "fit_note": "Our angle, to test in the call only after the client describes their priorities.",
        "questions": "Discovery questions",
        "sources": "Sources",
        "source": "source",
    },
    "es": {
        "title": "Brief de discovery",
        "generated": "Generado el {day} a partir de fuentes públicas. Verificar antes de usarlo con un cliente.",
        "snapshot": "Resumen de la empresa",
        "signals": "Señales recientes (últimos 12 meses)",
        "no_signals": "No se encontraron señales en los últimos 12 meses.",
        "excluded": "Excluidas (más de 12 meses o sin fecha):",
        "priorities": "Prioridades probables",
        "priorities_note": "Son hipótesis inferidas de señales públicas, para validar en la reunión.",
        "fit": "Dónde podría encajar nuestra oferta",
        "fit_note": "Nuestro ángulo: probarlo en la reunión solo después de que el cliente cuente sus prioridades.",
        "questions": "Preguntas de discovery",
        "sources": "Fuentes",
        "source": "fuente",
    },
}


def _format_signal(signal: str, source_word: str = "source") -> str:
    """Turns 'date | headline | why | url' into a readable markdown line."""
    parts = [p.strip() for p in signal.split("|")]
    if len(parts) == 4:
        day, headline, why, url = parts
        return f"- **{day}** · **{headline}**: {why} ([{source_word}]({url}))"
    return f"- {signal}"


def _headline(signal: str) -> str:
    """Date and headline of a signal, without the rest."""
    parts = [p.strip() for p in signal.split("|")]
    return " · ".join(parts[:2]) if len(parts) >= 2 else signal


class DiscoveryBriefTool(Tool):
    name = "build_discovery_brief"
    description = (
        "Formats the final discovery brief for a B2B sales call. "
        "Always call this tool once you have finished researching, "
        "then pass its output directly to final_answer."
    )
    inputs = {
        "company": {"type": "string", "description": "Company name."},
        "snapshot": {
            "type": "string",
            "description": "2-4 sentences: what the company does, its market and footprint. Facts only.",
        },
        "recent_signals": {
            "type": "array",
            "description": (
                "Recent news. Each item is a string with this exact format: "
                "'YYYY-MM-DD | Headline | Why it matters for the company\'s business | Source URL'. "
                "Only news about this company, not general market statistics."
            ),
        },
        "likely_priorities": {
            "type": "array",
            "description": (
                "3 to 5 of the company's own business priorities (growth, new products, markets, "
                "efficiency, risk), inferred from the signals. Hypotheses, not facts. "
                "Do not phrase them in terms of what the seller sells."
            ),
        },
        "discovery_questions": {
            "type": "array",
            "description": (
                "5 open-ended questions a seller would ask the buyer in a first call, from broad to "
                "specific: first about the company's goals and initiatives, then about the area where "
                "the seller's offering could help. Not investor or press questions. Never pitch."
            ),
        },
        "where_we_fit": {
            "type": "array",
            "description": (
                "1 to 3 hypotheses on where the seller's offering could connect to those priorities, "
                "each one linked to a specific signal. Empty list if the user didn't say what they sell."
            ),
        },
        "sources": {"type": "array", "description": "Every source URL used, copied exactly as found."},
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_brief = None  # the app reads the brief from here, so it never gets lost
        self.last_args = None  # what the agent sent, to write the brief again in another language
        self.language = "en"  # set by the app: "en" or "es"

    @property
    def text(self) -> dict:
        return BRIEF_TEXT.get(self.language, BRIEF_TEXT["en"])

    def forward(self, company, snapshot, recent_signals, likely_priorities, where_we_fit, discovery_questions, sources):
        # Structure enforced in code: if the model breaks it, it gets the error and must fix it
        problems = []
        if len(discovery_questions) != 5:
            problems.append(
                f"discovery_questions must have exactly 5 items (you sent {len(discovery_questions)}): "
                "2 broad questions about the company's goals, then 3 about the area of the offering."
            )
        if not 3 <= len(likely_priorities) <= 5:
            problems.append(f"likely_priorities must have 3 to 5 items (you sent {len(likely_priorities)}).")
        if len(where_we_fit) > 3:
            problems.append(f"where_we_fit must have at most 3 items (you sent {len(where_we_fit)}).")
        if problems:
            raise ValueError("Fix this and call build_discovery_brief again: " + " ".join(problems))

        self.last_args = {
            "company": company, "snapshot": snapshot, "recent_signals": recent_signals,
            "likely_priorities": likely_priorities, "where_we_fit": where_we_fit,
            "discovery_questions": discovery_questions, "sources": sources,
        }
        self.last_brief = self.render(self.language)
        return self.last_brief

    def render(self, language: str, args: dict | None = None) -> str:
        """Writes the brief with its headings in the given language. The app calls it again when the
        language switch changes, so the headings follow the switch (the content stays as the agents wrote it)."""
        a = args or self.last_args
        t = BRIEF_TEXT.get(language, BRIEF_TEXT["en"])

        # Date filter enforced in code: only the last 12 months, newest first
        cutoff = date.today() - timedelta(days=365)
        dated = [(_parse_date(s), s) for s in a["recent_signals"]]
        kept = sorted((d, s) for d, s in dated if d and d >= cutoff)[::-1]
        excluded = [s for d, s in dated if not (d and d >= cutoff)]

        signals_md = "\n".join(_format_signal(s, t["source"]) for _, s in kept) or f"- {t['no_signals']}"
        if excluded:
            # Show what was dropped, so the reader can tell if a priority relies on old news
            dropped = "\n".join(f"- {_headline(s)}" for s in excluded)
            signals_md += f"\n\n_{t['excluded']}_\n{dropped}"

        bullets = lambda items: "\n".join(f"- {item}" for item in items)
        fit_md = (
            f"\n## {t['fit']}\n> {t['fit_note']}\n\n{bullets(a['where_we_fit'])}\n"
        ) if a["where_we_fit"] else ""
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(a["discovery_questions"], start=1))

        brief = f"""# {t['title']}: {a['company']}
_{t['generated'].format(day=date.today().isoformat())}_

## {t['snapshot']}
{a['snapshot']}

## {t['signals']}
{signals_md}

## {t['priorities']}
> {t['priorities_note']}

{bullets(a['likely_priorities'])}
{fit_md}
## {t['questions']}
{numbered}

## {t['sources']}
{bullets(a['sources'])}
"""
        # "$" makes markdown viewers render text as math formulas. Escape it.
        return brief.replace("$", "\\$")


# Sites that block bots or sit behind a paywall: visiting them only wastes a step
BLOCKED_DOMAINS = ("bloomberg.com", "wsj.com", "ft.com", "reuters.com", "linkedin.com")


class SafeVisitWebpageTool(VisitWebpageTool):
    """Same as VisitWebpageTool, but skips pages it can't read instead of wasting a step on them."""

    def forward(self, url: str) -> str:
        if ".pdf" in url.lower():
            return "Skipped: this tool can't read PDF files. Look for a news article or web page instead."
        if any(domain in url for domain in BLOCKED_DOMAINS):
            return "Skipped: this site blocks automated access. Use another source."
        page = super().forward(url)
        if not page.strip():
            return "Empty page (probably built with JavaScript). Use news articles or another source instead."
        return page


class ResilientSearchTool(DuckDuckGoSearchTool):
    """Same as DuckDuckGoSearchTool, but faster and sturdier:
    - if a search engine throttles us, it tries two others;
    - if all of them fail, it stops retrying for a minute (no point waiting on every query);
    - repeated queries come from a cache instead of the internet."""

    BACKENDS = ("auto", "bing", "brave")
    COOLDOWN_SECONDS = 60
    UNAVAILABLE = (
        "Search is temporarily unavailable. Don't search again for now: "
        "work with the sources you already have, or visit known pages directly."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: dict[str, str] = {}
        self._down_until = 0.0

    def forward(self, query: str) -> str:
        key = query.strip().lower()
        if key in self._cache:
            return self._cache[key]
        if time.time() < self._down_until:
            return self.UNAVAILABLE
        for attempt, backend in enumerate(self.BACKENDS):
            if attempt:
                time.sleep(1)
            try:
                self._enforce_rate_limit()
                results = self.ddgs.text(query, max_results=self.max_results, backend=backend)
            except Exception:
                continue
            if results:
                lines = [f"[{r['title']}]({r['href']})\n{r['body']}" for r in results]
                self._cache[key] = "## Search Results\n\n" + "\n\n".join(lines)
                return self._cache[key]
        self._down_until = time.time() + self.COOLDOWN_SECONDS
        return self.UNAVAILABLE
