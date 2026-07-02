"""SM-2 spaced-repetition scheduling math (pure, no I/O).

The UI exposes 4 grades; they map onto SM-2 quality values:
    0 = Again -> q0 (fail, reset)
    1 = Hard  -> q3
    2 = Good  -> q4
    3 = Easy  -> q5
"""

from dataclasses import dataclass
from datetime import date, timedelta

DEFAULT_EASE_FACTOR = 2.5
MIN_EASE_FACTOR = 1.3

GRADE_AGAIN, GRADE_HARD, GRADE_GOOD, GRADE_EASY = 0, 1, 2, 3
_GRADE_TO_QUALITY = {GRADE_AGAIN: 0, GRADE_HARD: 3, GRADE_GOOD: 4, GRADE_EASY: 5}


@dataclass
class SrsState:
    """The scheduling fields produced by a review, ready to persist."""

    repetitions: int
    ease_factor: float
    interval: int
    next_review: date


def review(
    grade: int,
    repetitions: int,
    ease_factor: float,
    interval: int,
    today: date | None = None,
) -> SrsState:
    """Apply one SM-2 review and return the updated scheduling state.

    Args:
        grade: 0=Again, 1=Hard, 2=Good, 3=Easy.
        repetitions: consecutive successful reviews so far.
        ease_factor: current ease factor (>= 1.3).
        interval: current interval in days.
        today: base date for next_review (defaults to date.today()).
    """
    if grade not in _GRADE_TO_QUALITY:
        raise ValueError(f"grade must be 0..3, got {grade!r}")

    today = today or date.today()
    quality = _GRADE_TO_QUALITY[grade]

    if grade == GRADE_AGAIN:
        # Failed recall: reset the schedule but keep the (now lower) ease factor.
        repetitions = 0
        interval = 1
    else:
        if repetitions == 0:
            interval = 1
        elif repetitions == 1:
            interval = 6
        else:
            interval = round(interval * ease_factor)
        repetitions += 1

    # SM-2 ease-factor update (applied on every review, floored at MIN_EASE_FACTOR).
    ease_factor = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ease_factor = max(MIN_EASE_FACTOR, round(ease_factor, 4))

    return SrsState(
        repetitions=repetitions,
        ease_factor=ease_factor,
        interval=interval,
        next_review=today + timedelta(days=interval),
    )
