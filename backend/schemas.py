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
    subjects: list[str]


class PdfInfo(BaseModel):
    name: str
    pages: int


class PdfsResponse(BaseModel):
    pdfs: list[PdfInfo]


class AddSubjectResponse(BaseModel):
    subject: str
    saved_pdfs: list[str]           # filenames stored under documents/<subject>/
    chunk_count: int                # number of embedded text chunks indexed


# ---- chat ----
class ChatStartRequest(BaseModel):
    mode: str                       # "subject" | "topic"
    subject: str | None = None
    topic: str | None = None
    model: ModelOverride = None     # UI model selector (validated against AVAILABLE_MODELS)


class ChatSessionResponse(BaseModel):
    session_id: str
    mode: str
    label: str                      # human-readable context label for the UI
    chat_id: str | None = None      # persisted history id (None when persistence is off)


class ChatMessageRequest(BaseModel):
    session_id: str
    message: str


class ChatMessageResponse(BaseModel):
    answer: str
    steps: list[dict] = []   # structured agentic-trace events (kind/title/queries/passages/…)
    sources: list[str] = []
    artifact: dict | None = None   # first artifact (kept for back-compat with older clients)
    artifacts: list[dict] = []     # every artifact built this turn (one message can create several)


class ChatOverviewResponse(BaseModel):
    label: str                      # subject or topic name
    sources: int                    # count of source PDFs (0 for topic sessions)
    overview: str                   # 2-3 sentence markdown summary; "" on failure
    questions: list[str]            # up to 3 starter questions; [] on failure


class ChatMessage(BaseModel):
    role: str                       # "user" | "assistant"
    content: str


class ChatResumeRequest(BaseModel):
    chat_id: str


class ChatResumeResponse(BaseModel):
    session_id: str
    mode: str
    label: str
    chat_id: str
    messages: list[ChatMessage]


# ---- quiz ----
class QuizRequest(BaseModel):
    subject: str
    pdf: str
    num_questions: int = 5
    difficulty: str = "medium"
    model: ModelOverride = None


class QuizTopicRequest(BaseModel):
    topic: str
    num_questions: int = 5
    difficulty: str = "medium"
    model: ModelOverride = None


class QuizQuestion(BaseModel):
    question: str
    options: dict[str, str]         # keys "A".."D"
    correct: str                    # one of "A".."D"
    explanation: str


class QuizResponse(BaseModel):
    questions: list[QuizQuestion]
    id: str | None = None           # saved history id (None when persistence is off)
    title: str | None = None        # saved history title


# ---- flashcards ----
class FlashcardsRequest(BaseModel):
    subject: str
    pdf: str
    num_cards: int = 10
    difficulty: str = "medium"
    model: ModelOverride = None


class FlashcardsTopicRequest(BaseModel):
    topic: str
    num_cards: int = 10
    difficulty: str = "medium"
    model: ModelOverride = None


class Flashcard(BaseModel):
    front: str
    back: str
    topic: str


class FlashcardsResponse(BaseModel):
    cards: list[Flashcard]
    id: str | None = None           # saved history id (None when persistence is off)
    title: str | None = None


# ---- planner ----
class SummariesRequest(BaseModel):
    subject: str
    pdfs: list[str]
    model: ModelOverride = None


class SummariesResponse(BaseModel):
    summaries: dict[str, str]       # pdf name -> summary text
    id: str | None = None           # saved history id (None when persistence is off)
    title: str | None = None


class PlanRequest(BaseModel):
    subject: str
    pdfs: list[str]
    target_date: str                # ISO date "YYYY-MM-DD"
    model: ModelOverride = None


class TopicPlanRequest(BaseModel):
    topic: str
    target_date: str                # ISO date "YYYY-MM-DD"
    model: ModelOverride = None


class PlanResponse(BaseModel):
    plan: str                       # markdown
    id: str | None = None           # saved history id (None when persistence is off)
    title: str | None = None


# ---- mind map ----
class MindmapRequest(BaseModel):
    mode: str                       # "subject" | "topic"
    subject: str | None = None
    pdf: str | None = None          # subject mode: exactly one PDF
    topic: str | None = None        # topic mode
    model: ModelOverride = None


class MindmapNode(BaseModel):
    label: str
    children: list["MindmapNode"] = []


class MindmapResponse(BaseModel):
    root: MindmapNode               # root.label is the central concept
    id: str | None = None           # saved history id (None when persistence is off)
    title: str | None = None


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
