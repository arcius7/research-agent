"""
Unit tests for the pure logic — the parts where a silent bug costs a debugging
session (the _chunk infinite loop was exactly this).

Run:  .venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent                      # noqa: E402
import timer_state                # noqa: E402
from references import _coerce_json_list   # noqa: E402


class TestChunk(unittest.TestCase):
    def test_terminates_and_covers_text(self):
        text = ("word " * 5000).strip()          # 25k chars, only spaces
        chunks = agent._chunk(text, {"source": "x"}, 1000, 100)
        self.assertTrue(chunks)
        joined = "".join(c for c, _ in chunks)
        # Every word must appear; overlap means duplicates are fine
        self.assertIn(text[:900], joined)
        self.assertIn(text[-900:].strip(), joined)

    def test_pathological_overlap_still_terminates(self):
        # overlap == chunk_size used to stall the cursor forever
        text = "a" * 5000                          # no separators at all
        chunks = agent._chunk(text, {}, 100, 100)
        self.assertTrue(chunks)
        self.assertLess(len(chunks), 6000)         # forward progress each step

    def test_empty_and_tiny(self):
        self.assertEqual(agent._chunk("", {}, 1000, 100), [])
        chunks = agent._chunk("short.", {}, 1000, 100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], "short.")

    def test_chunk_meta_positions(self):
        chunks = agent._chunk("para one.\n\npara two.\n\npara three.", {"source": "p"}, 12, 2)
        for i, (_, meta) in enumerate(chunks):
            self.assertEqual(meta["chunk"], i)
            self.assertEqual(meta["source"], "p")


class TestPaperProfile(unittest.TestCase):
    def test_pomodoro_clamps(self):
        self.assertEqual(agent.paper_profile("a.pdf", 1000, 1)["work_minutes"], 15)
        self.assertEqual(agent.paper_profile("a.pdf", 900_000, 400)["work_minutes"], 50)
        self.assertEqual(agent.paper_profile("a.pdf", 30_000, 12)["work_minutes"], 24)

    def test_chunk_size_tiers(self):
        self.assertEqual(agent.paper_profile("a.pdf", 5_000, 3)["chunk_size"], 1800)
        self.assertEqual(agent.paper_profile("a.pdf", 200_000, 90)["chunk_size"], 850)

    def test_speaker_id_deterministic_and_bounded(self):
        a = agent.paper_profile("same.pdf", 10_000, 5)["speaker_id"]
        b = agent.paper_profile("same.pdf", 99_000, 50)["speaker_id"]
        self.assertEqual(a, b)                     # depends only on filename
        self.assertTrue(0 <= a < agent._N_VOICES)


class TestJoinCapped(unittest.TestCase):
    def test_caps_but_keeps_first(self):
        big = "x" * (agent.CONTEXT_CHAR_CAP + 500)
        self.assertEqual(agent._join_capped([big]), big)   # first chunk always kept
        out = agent._join_capped([big, "second"])
        self.assertNotIn("second", out)

    def test_joins_within_cap(self):
        out = agent._join_capped(["aa", "bb"])
        self.assertEqual(out, "aa\n\n---\n\nbb")


class TestCoerceJsonList(unittest.TestCase):
    def test_plain_array(self):
        self.assertEqual(_coerce_json_list('[{"title": "T"}]'), [{"title": "T"}])

    def test_wrapped_dict(self):
        self.assertEqual(_coerce_json_list('{"references": [{"title": "T"}]}'),
                         [{"title": "T"}])

    def test_prose_wrapped_array(self):
        txt = 'Sure! Here is the JSON:\n[{"title": "T"}]\nHope that helps.'
        self.assertEqual(_coerce_json_list(txt), [{"title": "T"}])

    def test_garbage(self):
        self.assertEqual(_coerce_json_list("no json here"), [])

    def test_truncated_array_salvages_complete_objects(self):
        # Token cap cuts output mid-array: no closing ] — salvage what parsed.
        txt = '[{"title": "First", "year": "2020"}, {"title": "Second", "ye'
        out = _coerce_json_list(txt)
        self.assertEqual(out, [{"title": "First", "year": "2020"}])

    def test_json_object_per_line(self):
        txt = '{"title": "A"}\n{"title": "B"}'
        out = _coerce_json_list(txt)
        self.assertEqual([o["title"] for o in out], ["A", "B"])


class TestSinglePaperConfig(unittest.TestCase):
    def test_single_paper_default_on(self):
        # Default behaviour is single-paper so questions don't bleed across papers
        self.assertTrue(agent.SINGLE_PAPER)

    def test_retrieve_drops_filter_in_single_paper(self):
        # In single-paper mode a stale `paper` arg must NOT filter retrieval.
        captured = {}
        real = agent._get_store
        class FakeStore:
            def similarity_search_with_score(self, q, k, filter):
                captured["filter"] = filter
                return []
        agent._get_store = lambda: FakeStore()
        try:
            agent._retrieve("q", paper="whatever.pdf")
            self.assertIsNone(captured["filter"])
        finally:
            agent._get_store = real


class TestTimerState(unittest.TestCase):
    def test_shared_singleton_with_agent(self):
        # The regression this repo shipped with: agent must mutate the SAME dict
        self.assertIs(agent._timer, timer_state.state)

    def test_advance_mode_cycle(self):
        with timer_state.state_lock:
            timer_state.state.update(mode="work", session_count=0, elapsed=7, running=True)
            timer_state.advance_mode()
            self.assertEqual(timer_state.state["mode"], "short_break")
            self.assertEqual(timer_state.state["elapsed"], 0)
            self.assertFalse(timer_state.state["running"])
            timer_state.advance_mode()
            self.assertEqual(timer_state.state["mode"], "work")
            # 4th work session ends in a long break
            timer_state.state["session_count"] = 3
            timer_state.advance_mode()
            self.assertEqual(timer_state.state["mode"], "long_break")


if __name__ == "__main__":
    unittest.main()
