"""Shared generator plumbing: tolerant JSON parsing, repair/re-sample flow, TTL cache."""
import time
import unittest

from backend.services import _gen
from tests._study_helpers import chain_error, patch_chat, result


class ParseJsonTests(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(_gen.parse_json('{"a": 1}'), {"a": 1})

    def test_fenced_and_think_blocks(self):
        self.assertEqual(_gen.parse_json('<think>hmm</think>\n```json\n{"a": [1, 2]}\n```'), {"a": [1, 2]})

    def test_prose_around_object_with_braces_in_strings(self):
        txt = 'Sure! Here you go: {"q": "what is {x}?", "n": 2} Hope it helps.'
        self.assertEqual(_gen.parse_json(txt), {"q": "what is {x}?", "n": 2})

    def test_trailing_commas(self):
        self.assertEqual(_gen.parse_json('{"a": [1, 2,], "b": 3,}'), {"a": [1, 2], "b": 3})

    def test_truncated_output_keeps_complete_items(self):
        txt = '{"questions": [{"question": "a", "n": 1}, {"question": "b", "n": 2}, {"question": "cut off'
        out = _gen.parse_json(txt)
        self.assertEqual([q["question"] for q in out["questions"]], ["a", "b"])

    def test_hopeless_returns_none(self):
        self.assertIsNone(_gen.parse_json("I cannot do that."))
        self.assertIsNone(_gen.parse_json(""))
        self.assertIsNone(_gen.parse_json("```\n```"))


class _Err(Exception):
    pass


def _shape(d):
    return d.get("items") if isinstance(d, dict) else None


class GenerateJsonTests(unittest.TestCase):
    def _run(self, *replies, **kw):
        ctx, fake = patch_chat(*replies)
        with ctx:
            try:
                return _gen.generate_json("sys", "prompt", shape=_shape, max_tokens=2048,
                                          error_cls=_Err, what="thing", **kw), fake
            except Exception as exc:       # noqa: BLE001
                return exc, fake

    def test_success_uses_json_mode_and_returns_result(self):
        (items, res), fake = self._run('{"items": [1]}')
        self.assertEqual(items, [1])
        self.assertEqual(res.provider, "groq")
        self.assertTrue(fake.calls[0]["json_mode"])
        self.assertGreaterEqual(fake.calls[0]["max_tokens"], 2048)
        self.assertEqual(len(fake.calls), 1)

    def test_local_repair_costs_no_extra_call(self):
        (items, _), fake = self._run('```json\n{"items": [1, 2,],}\n```')
        self.assertEqual(items, [1, 2])
        self.assertEqual(len(fake.calls), 1)

    def test_llm_repair_when_unparseable(self):
        (items, res), fake = self._run("totally not json", '{"items": [7]}')
        self.assertEqual(items, [7])
        self.assertEqual(len(fake.calls), 2)
        self.assertIn("repair", fake.calls[1]["system"].lower())

    def test_resample_once_when_nothing_usable(self):
        (items, _), fake = self._run('{"items": []}', '{"items": [3]}')
        self.assertEqual(items, [3])
        self.assertEqual(len(fake.calls), 2)

    def test_gives_up_with_error_cls(self):
        exc, fake = self._run('{"items": []}')
        self.assertIsInstance(exc, _Err)
        self.assertLessEqual(len(fake.calls), 4)

    def test_chain_error_propagates_untouched(self):
        exc, _ = self._run(chain_error())
        self.assertEqual(type(exc).__name__, "LLMChainError")

    def test_skipped_labels_merged(self):
        (_, res), _ = self._run(result('{"items": [1]}', skipped=["anthropic 401"]))
        self.assertIn("anthropic 401", res.to_public()["skipped"])


class TTLCacheTests(unittest.TestCase):
    def test_bounded_lru(self):
        c = _gen.TTLCache(maxsize=2, ttl_s=60)
        c.set("a", 1); c.set("b", 2); c.get("a"); c.set("c", 3)
        self.assertIsNone(c.get("b"))
        self.assertEqual((c.get("a"), c.get("c")), (1, 3))

    def test_ttl_expiry(self):
        c = _gen.TTLCache(maxsize=2, ttl_s=0.01)
        c.set("a", 1)
        time.sleep(0.03)
        self.assertIsNone(c.get("a"))


class HelperTests(unittest.TestCase):
    def test_clean_topic_and_dedupe(self):
        self.assertEqual(_gen.clean_topic("  a \n  b "), "a b")
        self.assertEqual(len(_gen.clean_topic("x" * 1000)), _gen.MAX_TOPIC_CHARS)
        self.assertEqual(_gen.dedupe_names([" a ", "a", "", "b", 3], 5), ["a", "b"])
        self.assertEqual(_gen.dedupe_names(list("abcdef"), 3), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
