"""Kern von Hand: Punktzahlen, erwartete Punktzahl, beste Meldung, Verlust, Erkennung, Mischung und Übertreiben, mit den Handrechnungen aus dem Plan (p = 3/10, p = 1/2)."""

import math
from fractions import Fraction

import numpy as np
import pytest

import zeh_algorithm as A
import zeh_constants as C

F = Fraction


def test_scores_by_hand():
    assert float(A.score("Brier", 0.4, 1)) == pytest.approx(-0.36) and float(A.score("Brier", 0.4, 0)) == pytest.approx(-0.16)
    assert float(A.score("Log", 0.5, 1)) == pytest.approx(math.log(0.5)) and float(A.score("Log", 0.8, 0)) == pytest.approx(math.log(0.2))
    assert float(A.score("sphärisch", 0.6, 1)) == pytest.approx(0.6 / math.sqrt(0.36 + 0.16)) and float(A.score("sphärisch", 0.6, 0)) == pytest.approx(0.4 / math.sqrt(0.52))
    assert float(A.score("Trefferquote", 0.3, 1)) == 0.0 and float(A.score("Trefferquote", 0.3, 0)) == 1.0 and float(A.score("Trefferquote", 0.7, 1)) == 1.0
    assert float(A.score("Trefferquote", 0.5, 1)) == 0.5 and float(A.score("Trefferquote", 0.5, 0)) == 0.5
    assert float(A.score("Absolutfehler", 0.3, 1)) == pytest.approx(-0.7) and float(A.score("Absolutfehler", 0.3, 0)) == pytest.approx(-0.3)


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError):
        A.score("Unsinn", 0.5, 1)
    with pytest.raises(ValueError):
        A.exact_expected("Log", F(1, 2), F(1, 2))


def test_brier_by_hand_p_three_tenths():
    p = F(3, 10)
    assert A.exact_expected("Brier", p, F(3, 10)) == F(-21, 100)               # -(q - p)^2 - p (1 - p) mit q = p
    assert A.exact_expected("Brier", p, F(4, 10)) == F(-22, 100)
    assert A.exact_expected("Brier", p, F(1, 2)) == F(-25, 100)
    assert A.exact_expected("Brier", p, F(3, 10)) - A.exact_expected("Brier", p, F(4, 10)) == F(1, 100)      # Verlust (q - p)^2
    assert A.exact_expected("Brier", p, F(3, 10)) - A.exact_expected("Brier", p, F(1, 2)) == F(4, 100)


def test_absolute_error_by_hand_the_lie_wins():
    p = F(3, 10)
    assert A.exact_expected("Absolutfehler", p, F(3, 10)) == F(-42, 100)       # -(p (1 - q) + (1 - p) q) = -(21/100 + 21/100)
    assert A.exact_expected("Absolutfehler", p, F(0)) == F(-30, 100)
    assert A.exact_expected("Absolutfehler", p, F(0)) - A.exact_expected("Absolutfehler", p, F(3, 10)) == F(12, 100)   # die Lüge gewinnt 12/100


def test_hit_rate_by_hand_honesty_is_one_of_many():
    p = F(3, 10)
    assert A.exact_expected("Trefferquote", p, F(1, 10)) == F(7, 10) and A.exact_expected("Trefferquote", p, F(3, 10)) == F(7, 10)
    assert A.exact_expected("Trefferquote", p, F(1, 2)) == F(1, 2) and A.exact_expected("Trefferquote", p, F(7, 10)) == F(3, 10)
    assert A.best_set("Trefferquote", 30) == list(range(1, 50))                 # eine von 49 gleich guten Meldungen


def test_log_by_hand_p_one_half():
    assert float(A.expected("Log", 0.5, 0.5)) == pytest.approx(math.log(0.5))
    assert float(A.expected("Log", 0.5, 0.6)) == pytest.approx(0.5 * math.log(0.6) + 0.5 * math.log(0.4)) and float(A.expected("Log", 0.5, 0.6)) == pytest.approx(-0.7136, abs=1e-4)
    assert A.loss("Log", 50, 60) == pytest.approx(0.0204, abs=1e-4)


def test_best_set_cases():
    assert A.best_set("Brier", 30) == [30] and A.best_set("Log", 30) == [30] and A.best_set("sphärisch", 30) == [30]
    assert A.best_set("Absolutfehler", 30) == [1] and A.best_set("Absolutfehler", 70) == [99]
    assert A.best_set("Absolutfehler", 50) == list(C.GRID)                      # alle Meldungen gleich gut
    assert A.best_set("Trefferquote", 70) == list(range(51, 100)) and A.best_set("Trefferquote", 50) == list(C.GRID)


def test_loss_and_n95_by_hand_p_one_half():
    assert A.loss("Brier", 50, 60) == pytest.approx(0.01)
    assert round(A.n95("Brier", 50, 60), 1) == 270.6                              # Verlust 0,01, Varianz 0,01: 1,645^2 * 0,01 / 0,0001
    assert round(A.n95("Log", 50, 60), 1) == 267.0 and round(A.n95("sphärisch", 50, 60), 1) == 276.0
    assert A.n95("Trefferquote", 30, 40) is None and A.n95("Absolutfehler", 50, 60) is None


def test_mix_by_hand():
    assert A.mix_optimum(80, 90) == F(5, 6) and A.best_set_mix(80, 90) == [83]    # 0,8 + 0,1 * 0,6 / 1,8
    assert A.best_set_mix(80, 100) == [80] and A.best_set_mix(80, 0) == [99] and A.mix_optimum(80, 0) is None
    assert A.best_set_mix(50, 30) == [50] and A.mix_optimum(50, 30) == F(1, 2)
    assert A.mix_optimum(99, 10) == F(99, 100)                                      # auf 0,99 begrenzt
    assert A.mix_optimum(30, 50) == F(1, 10) and A.best_set_mix(30, 50) == [10]


def test_exaggerate_moves_away_from_one_half_and_clips():
    p = np.array([0.3, 0.7, 0.5, 0.95, 0.02])
    assert list(np.round(A.exaggerate(p, 0.1), 2)) == [0.2, 0.8, 0.6, 0.99, 0.01]
    assert list(np.round(A.exaggerate(p, -0.1), 2)) == [0.4, 0.6, 0.4, 0.85, 0.12]


def test_no_shift_gives_a_tie_in_every_trial():
    res = A.finite(10, 0, 200, 7)
    for rule in C.RULES:
        assert res[rule]["over"]["tie"] == 100.0 and res[rule]["under"]["tie"] == 100.0


def test_logit_sigmoid_roundtrip():
    x = np.array([0.1, 0.5, 0.9])
    assert np.allclose(A.sigmoid(A.logit(x)), x)
