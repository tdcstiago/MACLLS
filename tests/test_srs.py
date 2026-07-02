import unittest
from datetime import date, timedelta

from database.db_manager import DatabaseManager
from database.srs_engine import review, DEFAULT_EASE_FACTOR, MIN_EASE_FACTOR


class TestSm2Engine(unittest.TestCase):
    def test_good_progression_intervals(self):
        # First Good: interval 1; second: 6; third: round(6 * ease).
        s1 = review(2, repetitions=0, ease_factor=DEFAULT_EASE_FACTOR, interval=0)
        self.assertEqual((s1.repetitions, s1.interval), (1, 1))
        s2 = review(2, s1.repetitions, s1.ease_factor, s1.interval)
        self.assertEqual((s2.repetitions, s2.interval), (2, 6))
        s3 = review(2, s2.repetitions, s2.ease_factor, s2.interval)
        self.assertEqual(s3.repetitions, 3)
        self.assertEqual(s3.interval, round(6 * s2.ease_factor))

    def test_good_keeps_ease_factor(self):
        s = review(2, repetitions=1, ease_factor=2.5, interval=1)
        self.assertAlmostEqual(s.ease_factor, 2.5, places=4)

    def test_again_resets_schedule(self):
        s = review(0, repetitions=5, ease_factor=2.5, interval=40)
        self.assertEqual(s.repetitions, 0)
        self.assertEqual(s.interval, 1)
        self.assertLess(s.ease_factor, 2.5)  # ease factor still drops on a lapse

    def test_easy_raises_and_hard_lowers_ease(self):
        easy = review(3, repetitions=2, ease_factor=2.5, interval=6)
        hard = review(1, repetitions=2, ease_factor=2.5, interval=6)
        self.assertGreater(easy.ease_factor, 2.5)
        self.assertLess(hard.ease_factor, 2.5)

    def test_ease_factor_floor(self):
        ease = MIN_EASE_FACTOR
        for _ in range(10):
            ease = review(0, repetitions=0, ease_factor=ease, interval=1).ease_factor
        self.assertGreaterEqual(ease, MIN_EASE_FACTOR)

    def test_next_review_is_today_plus_interval(self):
        today = date(2026, 1, 1)
        s = review(2, repetitions=1, ease_factor=2.5, interval=1, today=today)  # interval -> 6
        self.assertEqual(s.next_review, today + timedelta(days=6))

    def test_invalid_grade_raises(self):
        with self.assertRaises(ValueError):
            review(9, 0, 2.5, 0)


class TestFlashcardDb(unittest.TestCase):
    def setUp(self):
        self.db = DatabaseManager(":memory:")

    def tearDown(self):
        self.db.close()

    def test_flashcards_table_schema(self):
        cols = {row[1] for row in self.db.conn.execute("PRAGMA table_info(flashcards)").fetchall()}
        self.assertEqual(
            cols,
            {"id", "cache_key", "front_text", "back_text",
             "next_review", "interval", "ease_factor", "repetitions"},
        )

    def test_add_flashcard_is_due_immediately(self):
        card_id = self.db.add_flashcard("k1", "front", "back")
        self.assertIsInstance(card_id, int)
        due = self.db.get_due_flashcards()
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["front_text"], "front")
        self.assertEqual(due[0]["ease_factor"], DEFAULT_EASE_FACTOR)

    def test_future_card_is_not_due(self):
        future = date.today() + timedelta(days=5)
        self.db.add_flashcard("k", "f", "b", next_review=future)
        self.assertEqual(self.db.get_due_flashcards(), [])
        self.assertEqual(len(self.db.get_due_flashcards(today=future)), 1)

    def test_update_progress_reschedules_card(self):
        today = date.today()
        card_id = self.db.add_flashcard("k", "f", "b")
        state = self.db.update_flashcard_progress(card_id, 2, today=today)  # Good, first rep
        self.assertEqual(state.interval, 1)
        self.assertEqual(state.repetitions, 1)
        # Rescheduled to tomorrow → no longer due today, due tomorrow.
        self.assertEqual(self.db.get_due_flashcards(today=today), [])
        self.assertEqual(len(self.db.get_due_flashcards(today=today + timedelta(days=1))), 1)

    def test_update_missing_card_returns_none(self):
        self.assertIsNone(self.db.update_flashcard_progress(999, 2))


if __name__ == "__main__":
    unittest.main()
