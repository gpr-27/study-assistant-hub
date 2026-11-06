"""Pydantic request/response models — the API contract shared by all routers.

Routers and services MUST import their models from here so the frontend and
backend agree on shapes.
"""
from pydantic import BaseModel

# Optional model override sent by the UI's dynamic model selector. When present
# it must be one of the configured AVAILABLE_MODELS (enforced server-side by
# backend.config.model_for); otherwise the tool's configured default is used.
# None means "use the default" — no model id is ever hardcoded on client or server.
ModelOverride = str | None


# ---- documents (implemented in main.py) ----
class SubjectsResponse(BaseModel):
    subjects: list[str]          # bundled + own (unchanged, back-compat)
    own: list[str] = []         # the caller's uploaded subjects (deletable)
    bundled: list[str] = []     # built-in, read-only subjects
    hidden: list[str] = []      # standard subjects THIS account hid (still in ``bundled``, not in ``subjects``)
    limits: dict | None = None  # upload limits the SPA copy is generated from


class PdfInfo(BaseModel):
    name: str
    pages: int


class PdfsResponse(BaseModel):
    pdfs: list[PdfInfo]


class AddSubjectResponse(BaseModel):
    subject: str
    saved_pdfs: list[str]           # filenames stored under documents/<subject>/
    chunk_count: int                # number of text chunks indexed (all are BM25-searchable)
    indexing: str = "complete"      # "complete" | "partial" — partial: POST /api/subjects/{name}/continue-index to finish embedding
    n_embedded: int = 0             # chunks with a dense vector so far
    skipped_duplicates: list[str] = []   # uploaded files identical to ones already in the subject
    warnings: list[str] = []        # user-facing notes, e.g. "This PDF looks scanned; OCR it via Scan Notes."
    reading: dict = {}              # {filename: {done, total}} while pages of a big PDF are still being read


# ---- chat ----
from pydantic import Field as _ChatField  # noqa: E402


class ChatStartRequest(BaseModel):
    mode: str = _ChatField(max_length=16)                          # "subject" | "topic"
    subject: str | None = _ChatField(default=None, max_length=255)
    topic: str | None = _ChatField(default=None, max_length=400)
    model: ModelOverride = None     # accepted for back-compat, ignored (the provider chain picks the model)


class ChatSessionResponse(BaseModel):
    session_id: str                 # signed stateless token: valid on any instance, only for its owner
    mode: str
    label: str                      # human-readable context label for the UI
    chat_id: str | None = None      # persisted history id (None when persistence is off)


class ChatMessageRequest(BaseModel):
    session_id: str = _ChatField(min_length=1, max_length=4000)
    message: str = _ChatField(min_length=1, max_length=8000)
    tool: str | None = _ChatField(default=None, max_length=32)   # manual tool pick (slash command)
    # Optional prior turns [{role, content}]; only used when server-side persistence is off
    # (otherwise history is rebuilt from the database). Bounded and validated by the service.
    history: list[dict] | None = _ChatField(default=None, max_length=40)
    # Turn control (ChatGPT-style). "send" = a new message; "edit" = replace the user message ``target_id`` (and drop
    # everything after it); "regenerate" = re-answer the latest answer ``target_id`` (the old one is kept as a
    # version); "continue" = extend the latest (stopped / cut-off) answer. With persistence off the client sends
    # the already-truncated ``history`` (and ``prefix`` for "continue"); with it on the server edits the stored chat.
    action: str = _ChatField(default="send", pattern="^(send|edit|regenerate|continue)$")
    target_id: str | None = _ChatField(default=None, max_length=64)
    prefix: str | None = _ChatField(default=None, max_length=60000)
    # Client-minted message ids (so a Stopped answer can be continued / rated before the server replies).
    user_id: str | None = _ChatField(default=None, max_length=40, pattern="^[A-Za-z0-9_-]{6,40}$")
    assistant_id: str | None = _ChatField(default=None, max_length=40, pattern="^[A-Za-z0-9_-]{6,40}$")


class ChatMessageResponse(BaseModel):
    answer: str
    steps: list[dict] = []   # structured agentic-trace events (kind/title/queries/passages/…)
    sources: list[str] = []
    artifact: dict | None = None   # first artifact (kept for back-compat with older clients)
    artifacts: list[dict] = []     # every artifact built this turn (one message can create several)
    llm: dict | None = None        # ChatResult.to_public() of the model that answered
    followups: list[str] = []      # <= 3 suggested next questions (parsed from the model's trailing line)
    citations: list[dict] = []     # [{source, page, text}]
    finish: str | None = None      # "length" / "interrupted" -> the UI offers "Continue generating"
    usage: dict | None = None
    message_ids: dict | None = None


class ChatOverviewResponse(BaseModel):
    label: str                      # subject or topic name
    sources: int                    # count of source PDFs (0 for topic sessions)
    overview: str                   # 2-3 sentence markdown summary; "" on failure
    questions: list[str]            # up to 3 starter questions; [] on failure
    llm: dict | None = None         # model that generated it (None for pre-baked / empty cards)


class ChatMessage(BaseModel):
    role: str                       # "user" | "assistant"
    content: str
    llm: dict | None = None         # model label of a saved assistant message
    id: str | None = None           # stable message id (edit / regenerate / feedback target)
    ts: str | None = None
    stopped: bool = False           # the user pressed Stop; ``content`` is the partial answer
    finish: str | None = None       # "length" = cut off by the token limit
    versions: list[dict] = []       # earlier answers replaced by "Regenerate" ([{content, llm, ts}])
    feedback: str | None = None     # "up" | "down"
    citations: list[dict] = []      # [{source, page, text}] retrieved passages behind the answer
    usage: dict | None = None


class ChatFeedbackRequest(BaseModel):
    message_id: str = _ChatField(min_length=1, max_length=64)
    value: str | None = _ChatField(default=None, pattern="^(up|down)$")     # null clears the rating


class ChatPatchRequest(BaseModel):
    title: str | None = _ChatField(default=None, max_length=120)
    pinned: bool | None = None


class ChatResumeRequest(BaseModel):
    chat_id: str = _ChatField(min_length=1, max_length=64)


class ChatResumeResponse(BaseModel):
    session_id: str
    mode: str
    label: str
    chat_id: str
    messages: list[ChatMessage]


# ---- quiz / flashcards / planner / mind map (study tools) ----
# Request bounds keep prompts and costs bounded; `model` is accepted for back-compat but ignored
# (the provider chain picks the model). Every response carries `llm` = ChatResult.to_public().
from pydantic import Field  # noqa: E402

_Name = Field(min_length=1, max_length=255)
_Topic = Field(min_length=1, max_length=400)


class QuizRequest(BaseModel):
    subject: str = _Name
    pdf: str = _Name
    num_questions: int = Field(5, ge=1, le=20)
    difficulty: str = Field("medium", max_length=20)
    model: ModelOverride = None


class QuizTopicRequest(BaseModel):
    topic: str = _Topic
    num_questions: int = Field(5, ge=1, le=20)
    difficulty: str = Field("medium", max_length=20)
    model: ModelOverride = None


class QuizQuestion(BaseModel):
    question: str
    options: dict[str, str]         # keys "A".."D"
    correct: str                    # one of "A".."D"
    explanation: str
    sig: str | None = None          # HMAC over question+key (services/grading.py): lets /api/grade verify the attempt


class QuizResponse(BaseModel):
    questions: list[QuizQuestion]
    id: str | None = None           # saved history id (None when persistence is off)
    title: str | None = None        # saved history title
    llm: dict | None = None         # ChatResult.to_public() of the generating call


# ---- flashcards ----
class FlashcardsRequest(BaseModel):
    subject: str = _Name
    pdf: str = _Name
    num_cards: int = Field(10, ge=1, le=30)
    difficulty: str = Field("medium", max_length=20)
    model: ModelOverride = None


class FlashcardsTopicRequest(BaseModel):
    topic: str = _Topic
    num_cards: int = Field(10, ge=1, le=30)
    difficulty: str = Field("medium", max_length=20)
    model: ModelOverride = None


class Flashcard(BaseModel):
    front: str
    back: str
    topic: str


class FlashcardsResponse(BaseModel):
    cards: list[Flashcard]
    id: str | None = None           # saved history id (None when persistence is off)
    title: str | None = None
    llm: dict | None = None


# ---- planner ----
class SummariesRequest(BaseModel):
    subject: str = _Name
    pdfs: list[str] = Field(min_length=1, max_length=12)
    model: ModelOverride = None


class SummariesResponse(BaseModel):
    summaries: dict[str, str]       # pdf name -> summary text
    id: str | None = None           # saved history id (None when persistence is off)
    title: str | None = None
    llm: dict | None = None


class PlanRequest(BaseModel):
    subject: str = _Name
    pdfs: list[str] = Field(min_length=1, max_length=12)
    target_date: str = Field(max_length=40)    # ISO date "YYYY-MM-DD"
    model: ModelOverride = None


class TopicPlanRequest(BaseModel):
    topic: str = _Topic
    target_date: str = Field(max_length=40)    # ISO date "YYYY-MM-DD"
    model: ModelOverride = None


class PlanResponse(BaseModel):
    plan: str                       # markdown
    id: str | None = None           # saved history id (None when persistence is off)
    title: str | None = None
    llm: dict | None = None


# ---- mind map ----
class MindmapRequest(BaseModel):
    mode: str = Field(max_length=20)    # "subject" | "topic"
    subject: str | None = Field(None, max_length=255)
    pdf: str | None = Field(None, max_length=255)      # subject mode: exactly one PDF
    topic: str | None = Field(None, max_length=400)    # topic mode
    model: ModelOverride = None


class MindmapNode(BaseModel):
    label: str
    children: list["MindmapNode"] = []


class MindmapResponse(BaseModel):
    root: MindmapNode               # root.label is the central concept
    id: str | None = None           # saved history id (None when persistence is off)
    title: str | None = None
    llm: dict | None = None


MindmapNode.model_rebuild()         # resolve the self-referential `children`


# ---- history ----
class RenameRequest(BaseModel):
    title: str


# ---- guest (anonymous, MongoDB-backed accounts) ----
class GuestResponse(BaseModel):
    guestId: str
    isGuest: bool = True
    displayName: str
    createdAt: str | None = None        # ISO timestamp (None only if absent)
    lastActiveAt: str | None = None     # ISO timestamp


class GuestMigrateResponse(BaseModel):
    migrated: bool                      # True if a guest was found and re-owned
    total: int                         # total artifacts moved to the Clerk account
    moved: dict[str, int]              # per-kind counts (chats/quizzes/flashcards/plans/mindmaps)
