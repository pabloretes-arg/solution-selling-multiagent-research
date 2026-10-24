"""Translates the content of a finished brief when the user switches the language afterwards.
The headings are already translated by the brief tool; this translates what the agents wrote
(snapshot, signals, priorities, fit and questions), keeping company names, dates and URLs as they are."""
import json

from smolagents.models import ChatMessage, MessageRole

from model import translation_model  # a fast model (not the agents' reasoning model): see model.py

LANGUAGE_NAMES = {"en": "English", "es": "Spanish"}
TEXT_FIELDS = ("snapshot", "recent_signals", "likely_priorities", "where_we_fit", "discovery_questions")

PROMPT = """Translate the values of this JSON into {language}. Rules:
- Keep the same keys, the same number of items in each list and the same order.
- Keep company and product names, numbers, dates, URLs and the " | " separators exactly as they are.
- In Spanish, use a neutral, professional tone, and open questions with "¿".
Answer only with the translated JSON, nothing else.

{payload}"""


def _json_in(text: str) -> dict:
    """The JSON object inside the answer (the model may add text or a code fence around it)."""
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def translate_brief(args: dict, language: str) -> dict:
    """Returns a copy of the brief's arguments with the content translated. Raises if the answer is unusable."""
    source = {field: args[field] for field in TEXT_FIELDS}
    prompt = PROMPT.format(language=LANGUAGE_NAMES[language], payload=json.dumps(source, ensure_ascii=False, indent=1))
    answer = translation_model.generate([ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])])
    translated = _json_in(str(answer.content or ""))

    # Checked in code: same fields and the same number of items, or the original stays on screen
    for field in TEXT_FIELDS:
        same_shape = (
            isinstance(translated.get(field), str) if isinstance(source[field], str)
            else isinstance(translated.get(field), list) and len(translated[field]) == len(source[field])
        )
        if not same_shape:
            raise ValueError(f"the translation changed the field '{field}'")
    return {**args, **{field: translated[field] for field in TEXT_FIELDS}}
