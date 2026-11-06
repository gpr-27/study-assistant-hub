"""Flashcard decks: card types, cloze/reversed per-unit SM-2, import/export, duplicates (mocked embeddings),
isolation, limits, undo, typed-answer checking and the daily new-card budget. No network, no live provider."""
from __future__ import annotations

from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import db
from backend.auth import current_owner
from backend.routers import decks as decks_router
from backend.routers import flashcards as fc_router
from backend.services import attempts, deckio, grading, progress, srs
from tests._fakemongo import FakeMongo
from tests._study_helpers import chain_error, result as chat_result


@pytest.fixture()
def mongo(monkeypatch):
    fm = FakeMongo()
    monkeypatch.setattr(db, "_get_db", lambda: fm)
    monkeypatch.setattr(db, "is_enabled", lambda: True)
    attempts.clear_memory(); db._epochs.clear(); progress.clear_cache()
    yield fm
    attempts.clear_memory()


def client_for(owner="alice"):
    app = FastAPI()
    app.include_router(decks_router.router)
    app.include_router(fc_router.router)
    who = {"o": owner}
    app.dependency_overrides[current_owner] = lambda: who["o"]
    return TestClient(app), who


def mkdeck(c, cards=None, title="Bio"):
    r = c.post("/api/flashcards/decks", json={"title": title, "cards": cards or []})
    assert r.status_code == 200, r.text
    return r.json()["id"]


BASIC = {"front": "Powerhouse of the cell?", "back": "Mitochondria", "tags": ["Cells", "cells"]}
CLOZE = {"type": "cloze", "front": "The {{c1::nucleus}} stores {{c2::DNA::molecule}}."}
MCQ = {"type": "mcq", "front": "2+2?", "options": ["3", "4", "5"], "answer_index": 1}


# ---- card types ----
def test_card_validation_per_type():
    assert deckio.normalise_card(BASIC)["tags"] == ["cells"]
    assert deckio.normalise_card({"front": "x"}) is None                         # basic needs a back
    assert deckio.normalise_card({"type": "cloze", "front": "no deletion"}) is None
    assert deckio.normalise_card({**MCQ, "answer_index": 9}) is None
    assert deckio.normalise_card({"type": "bogus", "front": "a", "back": "b"}) is None
    assert deckio.normalise_card(MCQ)["back"] == "4"


def test_units_cloze_per_deletion_and_reversed_two_way():
    cards = [dict(deckio.normalise_card(CLOZE), id="cl1"), dict(deckio.normalise_card({"type": "reversed", "front": "cat", "back": "gato"}), id="rv1")]
    u = srs.units(cards)
    assert [x["cid"] for x in u] == ["cl1:c1", "cl1:c2", "rv1:a", "rv1:b"]
    assert u[0]["front"] == "The [...] stores DNA." and "**nucleus**" in u[0]["back"]
    assert u[1]["front"] == "The nucleus stores [molecule]."
    assert (u[3]["front"], u[3]["back"]) == ("gato", "cat")


def test_cloze_deletions_are_scheduled_independently(mongo):
    c, _ = client_for()
    deck = mkdeck(c, [CLOZE])
    cid = mongo["flashcards"].docs[0]["cards"][0]["id"]
    q = c.get(f"/api/flashcards/due?deck_id={deck}").json()
    assert [x["cid"] for x in q["cards"]] == [f"{cid}:c1", f"{cid}:c2"] and q["counts"]["total"] == 2
    c.post("/api/flashcards/review", json={"deck_id": deck, "batch_id": "batch-cloze1",
                                           "reviews": [{"card_id": f"{cid}:c1", "rating": "good"}, {"card_id": f"{cid}:c2", "rating": "again"}]})
    st = mongo["flashcards"].docs[0]["srs"]
    assert st[f"{cid}:c1"]["i"] == 1 and st[f"{cid}:c2"]["l"] == 1


# ---- deck CRUD, isolation, limits ----
def test_deck_lifecycle_rename_duplicate_delete_and_isolation(mongo):
    c, who = client_for("alice")
    d = mkdeck(c, [BASIC])
    assert c.patch(f"/api/flashcards/decks/{d}", json={"title": "Biology"}).json()["title"] == "Biology"
    dup = c.post(f"/api/flashcards/decks/{d}/duplicate").json()["id"]
    listing = c.get("/api/flashcards/decks").json()["decks"]
    assert {x["title"] for x in listing} == {"Biology", "Biology (copy)"} and all(x["cards"] == 1 and x["new"] == 1 for x in listing)
    who["o"] = "bob"
    assert c.get("/api/flashcards/decks").json()["decks"] == []
    for call in (lambda: c.get(f"/api/flashcards/decks/{d}"), lambda: c.delete(f"/api/flashcards/decks/{d}"),
                 lambda: c.patch(f"/api/flashcards/decks/{d}", json={"title": "x"}),
                 lambda: c.post(f"/api/flashcards/decks/{d}/cards", json={"cards": [BASIC]}),
                 lambda: c.get(f"/api/flashcards/decks/{d}/export")):
        assert call().status_code == 404
    who["o"] = "alice"
    assert c.delete(f"/api/flashcards/decks/{dup}").json() == {"deleted": True}
    assert len(c.get("/api/flashcards/decks").json()["decks"]) == 1


def test_limits_return_clear_4xx(mongo):
    c, _ = client_for()
    with mock.patch.object(decks_router, "MAX_DECKS_PER_OWNER", 2):
        mkdeck(c); mkdeck(c)
        r = c.post("/api/flashcards/decks", json={"title": "third"})
        assert r.status_code == 409 and "at most 2 decks" in r.json()["detail"]
    d = mkdeck(c)
    with mock.patch.object(decks_router, "MAX_CARDS_PER_DECK", 2):
        assert c.post(f"/api/flashcards/decks/{d}/cards", json={"cards": [BASIC, {"front": "a", "back": "b"}]}).status_code == 200
        r = c.post(f"/api/flashcards/decks/{d}/cards", json={"cards": [{"front": "c", "back": "d"}]})
        assert r.status_code == 409 and "at most 2 cards" in r.json()["detail"]
    assert c.post(f"/api/flashcards/decks/{d}/cards", json={"cards": [{"front": "only front"}]}).status_code == 422
    assert c.post(f"/api/flashcards/decks/{d}/import", json={"text": "no separators here"}).status_code == 422


# ---- editing, undo, optimistic concurrency ----
def test_edit_delete_and_undo_keep_the_schedule(mongo):
    c, _ = client_for()
    d = mkdeck(c, [BASIC])
    cid = mongo["flashcards"].docs[0]["cards"][0]["id"]
    c.post("/api/flashcards/review", json={"deck_id": d, "batch_id": "batch-edit01", "reviews": [{"card_id": cid, "rating": "good"}]})
    assert c.patch(f"/api/flashcards/decks/{d}/cards/{cid}", json={"back": "The mitochondrion", "starred": True, "hint": "M"}).json()["card"]["starred"]
    assert cid in mongo["flashcards"].docs[0]["srs"]                              # editing text keeps the schedule
    removed = c.delete(f"/api/flashcards/decks/{d}/cards/{cid}").json()["removed"]
    assert mongo["flashcards"].docs[0]["cards"] == [] and mongo["flashcards"].docs[0]["srs"] == {}
    assert c.post(f"/api/flashcards/decks/{d}/cards", json={"cards": [removed]}).json()["added"] == [cid]      # Undo = re-add by id
    assert c.patch(f"/api/flashcards/decks/{d}/cards/nope", json={"back": "x"}).status_code == 404
    assert c.patch(f"/api/flashcards/decks/{d}/cards/{cid}", json={"back": ""}).status_code == 422


def test_concurrent_edit_is_a_409_not_a_lost_update(mongo):
    c, _ = client_for()
    d = mkdeck(c, [BASIC])
    real = decks_router._coll

    class Racy:
        def __init__(self, coll): self.c = coll
        def find_one(self, *a, **k):
            doc = self.c.find_one(*a, **k)
            self.c.update_one({"_id": doc["_id"]}, {"$set": {"ver": (doc.get("ver") or 0) + 1}})   # someone else saved meanwhile
            return doc
        def __getattr__(self, n): return getattr(self.c, n)

    with mock.patch.object(decks_router, "_coll", lambda: Racy(real())):
        assert c.patch(f"/api/flashcards/decks/{d}", json={"title": "x"}).status_code == 409


# ---- import / export ----
def test_bulk_import_formats_and_round_trip(mongo):
    c, _ = client_for()
    d = mkdeck(c)
    text = "Capital of France :: Paris\nQ: 2+2?\nA: 4\nThe {{c1::sun}} is a star\nbad line without separator"
    r = c.post(f"/api/flashcards/decks/{d}/import", json={"text": text}).json()
    assert (r["parsed"], len(r["added"]), r["total"]) == (3, 3, 3)
    csv_out = c.get(f"/api/flashcards/decks/{d}/export?format=csv").text
    d2 = mkdeck(c, title="again")
    r2 = c.post(f"/api/flashcards/decks/{d2}/import", json={"text": csv_out, "format": "csv"}).json()
    assert len(r2["added"]) == 3
    a = [(x["front"], x["back"], x["type"]) for x in mongo["flashcards"].docs[0]["cards"]]
    b = [(x["front"], x["back"], x["type"]) for x in mongo["flashcards"].docs[1]["cards"]]
    assert a == b and ("The {{c1::sun}} is a star", "", "cloze") in a


def test_anki_tsv_and_json_export(mongo):
    c, _ = client_for()
    d = mkdeck(c, [{**BASIC, "back": "Line1\nLine2"}, MCQ])
    tsv = c.get(f"/api/flashcards/decks/{d}/export?format=tsv").text.splitlines()
    assert tsv[0].split("\t") == ["Powerhouse of the cell?", "Line1<br>Line2", "cells"]
    assert "A. 3<br>B. 4" in tsv[1]
    assert len(c.get(f"/api/flashcards/decks/{d}/export?format=json").json()["cards"]) == 2
    assert deckio.parse_csv("a\tb\ttag1 tag2\n")[0]["tags"] == ["tag1", "tag2"]          # Anki TSV re-imports
    assert c.get(f"/api/flashcards/decks/{d}/export?format=pdf").status_code == 422


# ---- duplicates / related ----
def test_duplicates_lexical_and_skip_on_import(mongo):
    c, _ = client_for()
    d = mkdeck(c, [BASIC])
    r = c.post(f"/api/flashcards/decks/{d}/cards", json={"cards": [{"front": "powerhouse of the CELL", "back": "x"},
                                                                  {"front": "What is the powerhouse of the cell", "back": "y"},
                                                                  {"front": "Unrelated question", "back": "z"}], "skip_duplicates": True}).json()
    assert [x["how"] for x in r["duplicates"]] == ["exact", "similar"] and len(r["added"]) == 1 and r["total"] == 2
    r = c.post(f"/api/flashcards/decks/{d}/cards", json={"cards": [{"front": "Unrelated question", "back": "again"}]}).json()
    assert len(r["duplicates"]) == 1 and len(r["added"]) == 1                          # reported, but kept unless skip_duplicates


def test_semantic_duplicates_use_one_batched_embedding_call_and_respect_privacy(monkeypatch):
    import numpy as np
    calls = []

    def fake_embed(texts, **kw):
        calls.append(list(texts))
        return np.array([[1.0, 0.0] if "ATP" in t or "energy" in t else [0.0, 1.0] for t in texts])

    deckio._emb_cache.clear()
    from backend import gemini_embeddings as ge
    monkeypatch.setattr(ge, "have_keys", lambda: True)
    monkeypatch.setattr(ge, "embed_documents", fake_embed)
    new = [{"id": "n1", "front": "Cellular energy currency", "back": ""}, {"id": "n2", "front": "Something else", "back": ""}]
    old = [{"id": "o1", "front": "ATP is the energy carrier", "back": ""}]
    with mock.patch.object(deckio.privacy, "policy_active", return_value=False):
        d = deckio.find_duplicates(new, old)
        deckio.find_duplicates(new, old)
    assert [(x["index"], x["how"]) for x in d] == [(0, "semantic")] and len(calls) == 1      # batched + hash-cached
    with mock.patch.object(deckio.privacy, "policy_active", return_value=True):              # user content: never to Gemini
        deckio._emb_cache.clear(); calls.clear()
        assert deckio.find_duplicates(new, old) == [] and calls == []


def test_related_cards_lexical(mongo):
    c, _ = client_for()
    d = mkdeck(c, [{"front": "Mitochondria produce ATP", "back": "energy"}, {"front": "Ribosomes make protein", "back": "translation"},
                   {"front": "ATP is produced by mitochondria", "back": "energy currency"}])
    cards = mongo["flashcards"].docs[0]["cards"]
    r = c.get(f"/api/flashcards/decks/{d}/related?card_id={cards[0]['id']}").json()
    assert r["method"] == "lexical" and r["related"][0]["id"] == cards[2]["id"]


# ---- search / tags ----
def test_search_and_tag_filter(mongo):
    c, _ = client_for()
    d = mkdeck(c, [{"front": "Alpha question", "back": "a", "tags": ["greek"]}, {"front": "Beta question", "back": "b", "tags": ["greek", "x"]},
                   {"front": "Other", "back": "c"}])
    assert len(c.get(f"/api/flashcards/decks/{d}?q=question").json()["cards"]) == 2
    out = c.get(f"/api/flashcards/decks/{d}?tag=x").json()
    assert len(out["cards"]) == 1 and out["tags"] == ["greek", "x"] and out["total_cards"] == 3


# ---- study modes + daily new limit ----
def test_modes_starred_tag_cram_and_daily_new_budget(mongo):
    c, _ = client_for()
    cards = [{"front": f"f{i}", "back": f"b{i}", "starred": i == 1, "tags": ["t"] if i < 2 else []} for i in range(30)]
    d = mkdeck(c, cards)
    ids = [x["id"] for x in mongo["flashcards"].docs[0]["cards"]]
    assert len(c.get(f"/api/flashcards/due?deck_id={d}&mode=starred").json()["cards"]) == 1
    assert len(c.get(f"/api/flashcards/due?deck_id={d}&tag=t&mode=cram").json()["cards"]) == 2
    assert len(c.get(f"/api/flashcards/due?deck_id={d}&mode=cram&limit=200").json()["cards"]) == 30
    assert len(c.get(f"/api/flashcards/due?deck_id={d}&new_limit=50").json()["cards"]) == 20          # daily budget caps new cards
    c.post("/api/flashcards/review", json={"deck_id": d, "batch_id": "batch-new0001",
                                           "reviews": [{"card_id": i, "rating": "good"} for i in ids[:15]]})
    left = c.get(f"/api/flashcards/due?deck_id={d}&new_limit=50").json()["cards"]
    assert len(left) == 5 and all(x["state"] == "new" for x in left)


# ---- typed answers / MCQ ----
def test_typed_check_normalised_first_then_llm_labelled_then_degrades(mongo):
    c, _ = client_for()
    d = mkdeck(c, [{"type": "typed", "front": "Powerhouse?", "back": "Mitochondria / mitochondrion"}, MCQ])
    cards = mongo["flashcards"].docs[0]["cards"]
    with mock.patch("backend.llm_chain.chat") as chat:
        ok = c.post(f"/api/flashcards/decks/{d}/check", json={"card_id": cards[0]["id"], "answer": "  MITOCHONDRION! "}).json()
        mc = c.post(f"/api/flashcards/decks/{d}/check", json={"card_id": cards[1]["id"], "answer": "B"}).json()
        assert chat.call_count == 0
    assert ok["correct"] and ok["llm"] is None and mc["correct"]
    reply = {"grades": [{"id": "c1", "covered": [1], "score": 1.0, "verdict": "correct", "feedback": "Yes, same idea."}]}
    with mock.patch("backend.llm_chain.chat", return_value=chat_result(reply)):
        llm = c.post(f"/api/flashcards/decks/{d}/check", json={"card_id": cards[0]["id"], "answer": "the organelle making ATP: mitochondria"}).json()
    assert llm["correct"] and llm["llm"]["label"].startswith("Model:")
    with mock.patch("backend.llm_chain.chat", side_effect=chain_error()):
        down = c.post(f"/api/flashcards/decks/{d}/check", json={"card_id": cards[0]["id"], "answer": "it is the mitochondria organelle"}).json()
    assert down["provisional"] and down["llm"]["label"].startswith("Model: none")


# ---- AI generation of typed cards from pasted text ----
def test_generate_cloze_deck_from_pasted_text_is_labelled_and_saved(mongo):
    c, _ = client_for()
    reply = {"cards": [{"front": "The {{c1::mitochondrion}} makes ATP", "topic": "Cells"}, {"front": "no deletion", "topic": "x"}]}
    with mock.patch("backend.llm_chain.chat", return_value=chat_result(reply)) as chat:
        r = c.post("/api/flashcards/text", json={"text": "notes " * 5000, "card_type": "cloze", "num_cards": 5}).json()
    assert r["llm"]["label"].startswith("Model:") and len(r["cards"]) == 1 and r["id"]
    assert "notes " * 100 in chat.call_args.args[1][0]["content"] or "notes" in chat.call_args.kwargs.get("messages", [{}])[0].get("content", "notes")
    deck = mongo["flashcards"].docs[0]
    assert deck["cards"][0]["type"] == "cloze" and deck["num_units"] == 1 and deck["source"] == "text"


def test_similar_cards_are_grounded_deduped_labelled(mongo):
    c, _ = client_for()
    d = mkdeck(c, [BASIC, {"front": "Site of protein synthesis?", "back": "Ribosome"}])
    reply = {"cards": [{"front": "powerhouse of the CELL", "back": "dup"}, {"front": "Where is DNA replicated?", "back": "Nucleus"}]}
    with mock.patch("backend.llm_chain.chat", return_value=chat_result(reply)) as chat:
        r = c.post(f"/api/flashcards/decks/{d}/similar", json={"num_cards": 2}).json()
    assert len(r["added"]) == 1 and r["duplicates_skipped"] == 1 and r["total"] == 3 and r["llm"]["label"].startswith("Model:")
    assert "Mitochondria" in chat.call_args.args[1][0]["content"]                 # seeds are in the prompt


# ---- flashcard reviews feed Analytics (progress summary, weak topics) ----
def _summary_reads_fake_mongo(mongo):
    """db.multi_recent over the in-memory FakeMongo (its aggregate pipeline is not implemented): newest-first
    per kind, owner-scoped, projected like the real branch. Lets build_summary run exactly as in production."""
    def fake(owner, specs):
        out = {}
        for kind, limit, fields in specs:
            rows = sorted((d for d in mongo[kind].docs if d.get("owner") == owner),
                          key=lambda d: str(d.get("created_at")), reverse=True)[:limit]
            out[kind] = {"n": len(rows), "rows": [] if fields is None else
                         [{k: d[k] for k in fields if k in d} for d in rows]}
        return out
    return mock.patch.object(db, "multi_recent", side_effect=fake)


def _review(c, deck, batch, ratings):
    return c.post("/api/flashcards/review", json={"deck_id": deck, "batch_id": batch,
                                                  "reviews": [{"card_id": cid, "rating": r} for cid, r in ratings]})


def test_review_batch_counts_in_analytics_with_good_easy_as_correct(mongo):
    c, _ = client_for("alice")
    deck = mkdeck(c, [{"front": "a", "back": "b", "topic": "Cells"}, {"front": "c", "back": "d", "topic": "Cells"},
                      {"front": "e", "back": "f"}], title="Biology")
    ids = [x["id"] for x in mongo["flashcards"].docs[0]["cards"]]
    with _summary_reads_fake_mongo(mongo):
        assert progress.build_summary("alice")["overall"]["total_attempts"] == 0
        r = _review(c, deck, "batch-anal001", [(ids[0], "good"), (ids[1], "easy"), (ids[2], "again")])
        assert r.status_code == 200 and r.json()["saved"] is True
        docs = [d for d in mongo["attempts"].docs if d.get("owner") == "alice"]
        assert len(docs) == 1
        assert (docs[0]["kind"], docs[0]["subject"], docs[0]["correct"], docs[0]["total"]) == ("flashcards", "Biology", 2, 3)
        summary = progress.build_summary("alice")          # epoch bump on the write: the cached summary is not stale
    assert summary["overall"]["total_attempts"] == 1
    assert summary["overall"]["overall_accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    bio = summary["subjects"][0]
    assert bio["subject"] == "Biology"
    cells = {t["topic"]: t for t in bio["topics"]}
    assert cells["Cells"]["accuracy"] == 1.0 and cells["Biology"]["accuracy"] == 0.0   # card topic, else the deck title
    assert [w["topic"] for w in summary["weakest"]] == []                             # one attempt per cell: not yet "weak"
    flash = next(a for a in summary["activity"] if a["tool"] == "flashcards")
    assert flash["count"] == 1


def test_review_replay_is_idempotent_for_analytics_and_schedule(mongo):
    c, _ = client_for("alice")
    deck = mkdeck(c, [BASIC], title="Bio")
    cid = mongo["flashcards"].docs[0]["cards"][0]["id"]
    with _summary_reads_fake_mongo(mongo):
        first = _review(c, deck, "batch-idem001", [(cid, "good")]).json()
        srs_before = dict(mongo["flashcards"].docs[0]["srs"])
        again = _review(c, deck, "batch-idem001", [(cid, "again")])     # retried / double-clicked batch
        assert again.status_code == 200 and again.json()["replayed"] is True
        assert again.json()["id"] == first["id"]
        assert mongo["flashcards"].docs[0]["srs"] == srs_before          # not rescheduled twice
        assert len([d for d in mongo["attempts"].docs if d.get("owner") == "alice"]) == 1
        summary = progress.build_summary("alice")
    assert summary["overall"]["total_attempts"] == 1 and summary["overall"]["overall_accuracy"] == 1.0


def test_review_analytics_are_isolated_per_owner(mongo):
    alice, _ = client_for("alice")
    bob, _ = client_for("bob")
    deck = mkdeck(alice, [BASIC], title="Bio")
    cid = mongo["flashcards"].docs[0]["cards"][0]["id"]
    with _summary_reads_fake_mongo(mongo):
        assert _review(alice, deck, "batch-iso00001", [(cid, "good")]).status_code == 200
        assert progress.build_summary("alice")["overall"]["total_attempts"] == 1
        assert progress.build_summary("bob")["overall"]["total_attempts"] == 0      # bob never sees alice's study
        # bob cannot review alice's deck, so nothing can leak into his analytics through it
        assert _review(bob, deck, "batch-iso00002", [(cid, "good")]).status_code == 404
        assert progress.build_summary("bob")["overall"]["total_attempts"] == 0
