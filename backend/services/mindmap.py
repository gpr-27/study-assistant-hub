"""Mind Map service: turn a PDF's content or a topic into a concept map.

One chain call (JSON mode) builds a bounded tree the SPA can always render. The model is asked
for a deliberately *flat* shape (``central`` + ``branches`` of ``{title, points[]}``) rather than a
recursive node tree — models follow the flat schema far more reliably, while the result is still a
clean 3-level mind map. Content is truncated so a single request stays within limits.
"""
from backend.services import _gen

_MAX_OUTPUT_TOKENS: int = 2500   # a flat branches/points map is small; headroom for reasoning models
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
            f"{spec}\n\nMATERIAL:\n{content}"
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


_SYSTEM = (
    "You distil material into clear, hierarchical mind maps. You always "
    "respond with a single valid JSON object and nothing else."
)


def generate_mindmap_ex(*, topic: str | None = None, content: str | None = None,
                        model: str | None = None) -> tuple[dict, dict | None]:
    """Generate a bounded mind-map tree -> ``(root {label, children}, llm_public)``."""
    central = _gen.clean_topic(topic)
    body = (content or "").strip()
    if not central and not body:
        raise MindmapError("Provide a topic or a document to build a mind map from.")

    def shape(data: object) -> dict | None:
        root = _tree_from_payload(data, fallback_central=central or "Mind Map")
        return root if root["children"] else None

    root, res = _gen.generate_json(_SYSTEM, _build_prompt(central, body or None), shape=shape,
                                   max_tokens=_MAX_OUTPUT_TOKENS, error_cls=MindmapError,
                                   what="mind map", temperature=0.3)
    if central:                                   # topic mode: keep the user's wording
        root["label"] = central[:_MAX_LABEL_CHARS]
    return root, _gen.public(res)


def generate_mindmap(*, topic: str | None = None, content: str | None = None, model: str | None = None) -> dict:
    """Back-compat: the root {label, children} dict only (see :func:`generate_mindmap_ex`)."""
    return generate_mindmap_ex(topic=topic, content=content, model=model)[0]
