"""Mind Map service: turn a PDF's content or a topic into a concept map.

Calls Groq once for a single JSON object and builds a bounded tree the SPA can
always render. The model is asked for a deliberately *flat* shape
(``central`` + ``branches`` of ``{title, points[]}``) rather than a recursive
node tree — small, fast models follow the flat schema far more reliably, while
the result is still a clean 3-level mind map. Uses a lighter model by default
(the mindmap override / DEFAULT_MODEL) and truncates content so a single request
stays within limits.
"""
import json

from groq import APIError, Groq

from backend.config import model_for, require_groq_key
from backend.documents import clip_to_tokens

# Mind Map runs on a lighter model by default (MINDMAP_MODEL — typically the 6000
# TPM free-tier 8B). Its per-minute budget counts input + reserved output, so the
# content slice is bounded and the (small) output is capped below.
_MAX_CONTENT_TOKENS: int = 3000
_MAX_OUTPUT_TOKENS: int = 1024   # a flat branches/points map is small; cap the completion
_MAX_BRANCHES: int = 8       # cap main branches
_MAX_POINTS: int = 6         # cap points per branch
_MAX_LABEL_CHARS: int = 80

_EXAMPLE: str = (
    '{"central": "Water Cycle", "branches": ['
    '{"title": "Evaporation", "points": ["Sun heats water", "Vapour rises"]}, '
    '{"title": "Condensation", "points": ["Clouds form"]}]}'
)


class MindmapError(Exception):
    """Raised when a usable mind map cannot be generated from the input."""


def _build_prompt(topic: str, content: str | None) -> str:
    spec = (
        "Return ONLY a single JSON object of exactly this form:\n"
        '{"central": "<main concept, a few words>", "branches": '
        '[{"title": "<branch name>", "points": ["<short point>", "..."]}]}\n'
        "Use 4-7 branches; each branch has 2-4 short points (max 6 words each). "
        'Every branch MUST be an object with a "title" string and a "points" array '
        "of strings. No markdown, no extra keys, no commentary.\n"
        f"Example: {_EXAMPLE}"
    )
    if content:
        return (
            "Summarise the key concepts of the material below as a mind map.\n\n"
            f"{spec}\n\nMATERIAL:\n{clip_to_tokens(content, _MAX_CONTENT_TOKENS)}"
        )
    return f"Create a mind map about: {topic}.\n\n{spec}"


def _tree_from_payload(data: object, fallback_central: str) -> dict:
    """Build a bounded {label, children} tree from the flat model payload."""
    central = ""
    branches_raw: object = None
    if isinstance(data, dict):
        central = str(data.get("central", "")).strip()[:_MAX_LABEL_CHARS]
        branches_raw = data.get("branches")

    branches: list[dict] = []
    if isinstance(branches_raw, list):
        for branch in branches_raw[:_MAX_BRANCHES]:
            if not isinstance(branch, dict):
                continue
            title = str(branch.get("title", "")).strip()[:_MAX_LABEL_CHARS]
            if not title:
                continue
            leaves: list[dict] = []
            points = branch.get("points")
            if isinstance(points, list):
                for point in points[:_MAX_POINTS]:
                    label = str(point).strip()[:_MAX_LABEL_CHARS]
                    if label:
                        leaves.append({"label": label, "children": []})
            branches.append({"label": title, "children": leaves})

    return {"label": central or fallback_central, "children": branches}


def generate_mindmap(*, topic: str | None = None, content: str | None = None, model: str | None = None) -> dict:
    """Generate a bounded mind-map tree (returns the root {label, children} dict)."""
    central = (topic or "").strip()
    body = (content or "").strip()
    if not central and not body:
        raise MindmapError("Provide a topic or a document to build a mind map from.")

    require_groq_key()
    client = Groq()
    try:
        response = client.chat.completions.create(
            model=model_for("mindmap", model),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You distil material into clear, hierarchical mind maps. You always "
                        "respond with a single valid JSON object and nothing else."
                    ),
                },
                {"role": "user", "content": _build_prompt(central, body or None)},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=_MAX_OUTPUT_TOKENS,
        )
        data = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError as exc:
        raise MindmapError(f"The model returned invalid JSON: {exc}") from exc
    except APIError:
        raise  # Groq errors (rate limit, …) are formatted by the global handlers
    except Exception as exc:  # noqa: BLE001 — wrap any other transport/SDK failure
        raise MindmapError(f"Mind map generation failed: {exc}") from exc

    root = _tree_from_payload(data, fallback_central=central or "Mind Map")
    if central:                                   # topic mode: keep the user's wording
        root["label"] = central[:_MAX_LABEL_CHARS]
    if not root["children"]:
        raise MindmapError("The model did not return a usable mind map — please try again.")
    return root
