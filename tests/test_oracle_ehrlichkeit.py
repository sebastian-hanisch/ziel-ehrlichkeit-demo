"""Orakel (ORACLE-PLAYBOOK §5): reines Python (Schleifen, `Fraction`, `math`), keine numpy-Zeile im Referenzweg.

Beste Meldungen und Verluste werden aus der Definition gerechnet (erwartete Punktzahl jeder Meldung des Rasters); Zufallsversuche und Wettbewerb gegen eine Schleife über dieselben gezogenen Zahlen."""

import math
from fractions import Fraction

import numpy as np
import pytest

import zeh_algorithm as A
import zeh_constants as C
import zeh_scenario as S

F = Fraction


# --- Punktzahlen in reinem Python -------------------------------------------------------------------------------------------------------------

def py_score(rule, q, y):
    if rule == "Brier":
        return -((q - y) ** 2)
    if rule == "Log":
        return math.log(q) if y == 1 else math.log(1 - q)
    if rule == "sphärisch":
        return (q if y == 1 else 1 - q) / math.sqrt(q * q + (1 - q) * (1 - q))
    if rule == "Trefferquote":
        side = 1 if q > 0.5 else 0 if q < 0.5 else 0.5
        return side if y == 1 else 1 - side
    if rule == "Absolutfehler":
        return -abs(q - y)
    raise ValueError(rule)


def py_expected(rule, p, q):
    return p * py_score(rule, q, 1) + (1 - p) * py_score(rule, q, 0)


EXACT = ("Brier", "Trefferquote", "Absolutfehler")


@pytest.mark.parametrize("rule", EXACT)
def test_exact_best_sets_match_a_loop_over_all_reports(rule):
    for pi in range(1, 100):
        p = F(pi, 100)
        vals = [py_expected_exact(rule, p, F(qi, 100)) for qi in range(1, 100)]
        top = max(vals)
        assert A.best_set(rule, pi) == [qi for qi, v in zip(range(1, 100), vals) if v == top], (rule, pi)


def py_expected_exact(rule, p, q):
    if rule == "Brier":
        return p * (-((q - 1) ** 2)) + (1 - p) * (-(q ** 2))
    if rule == "Absolutfehler":
        return p * (-(1 - q)) + (1 - p) * (-q)
    side = F(1) if q > F(1, 2) else F(0) if q < F(1, 2) else F(1, 2)
    return p * side + (1 - p) * (1 - side)


@pytest.mark.parametrize("rule", ("Log", "sphärisch"))
def test_float_best_sets_match_a_loop(rule):
    for pi in range(1, 100):
        vals = [py_expected(rule, pi / 100, qi / 100) for qi in range(1, 100)]
        top = max(vals)
        assert A.best_set(rule, pi) == [qi for qi, v in zip(range(1, 100), vals) if v >= top - 1e-12], (rule, pi)


@pytest.mark.parametrize("rule", C.RULES)
def test_expected_curve_matches_a_loop(rule):
    for pi in (1, 17, 30, 50, 83, 99):
        curve = A.expected_curve(rule, pi)
        for qi in range(1, 100):
            assert curve[qi - 1] == pytest.approx(py_expected(rule, pi / 100, qi / 100), abs=1e-12)


@pytest.mark.parametrize("rule", EXACT)
def test_exact_expected_matches_the_loop_definition(rule):
    for pi in (1, 30, 50, 77, 99):
        for qi in (1, 10, 49, 50, 51, 99):
            assert A.exact_expected(rule, F(pi, 100), F(qi, 100)) == py_expected_exact(rule, F(pi, 100), F(qi, 100))


def test_mix_matches_a_loop_and_the_closed_form():
    for mixpct in (10, 30, 50, 70, 90, 100):
        mix = F(mixpct, 100)
        for pi in (1, 20, 40, 50, 60, 80, 99):
            p = F(pi, 100)
            vals = [mix * py_expected_exact("Brier", p, F(qi, 100)) + (1 - mix) * py_expected_exact("Absolutfehler", p, F(qi, 100)) for qi in range(1, 100)]
            top = max(vals)
            best = [qi for qi, v in zip(range(1, 100), vals) if v == top]
            assert A.best_set_mix(pi, mixpct) == best
            qs = A.mix_optimum(pi, mixpct)
            assert abs(best[0] - qs * 100) <= F(1, 2)                      # die Nullstelle der Ableitung trifft das Raster


def test_loss_matches_the_loop():
    for rule in C.PROPER:
        for pi, qi in ((30, 40), (50, 60), (80, 90), (20, 5)):
            want = py_expected(C.RULES[rule], pi / 100, pi / 100) - py_expected(C.RULES[rule], pi / 100, qi / 100)
            assert A.loss(C.RULES[rule], pi, qi) == pytest.approx(want, abs=1e-12)


def test_n95_matches_a_loop():
    for rule in ("Brier", "Log", "sphärisch"):
        for pi, qi in ((30, 40), (50, 60), (80, 90)):
            p, q = pi / 100, qi / 100
            d1, d0 = py_score(rule, p, 1) - py_score(rule, q, 1), py_score(rule, p, 0) - py_score(rule, q, 0)
            mean, var = p * d1 + (1 - p) * d0, p * (1 - p) * (d1 - d0) ** 2
            assert A.n95(rule, pi, qi) == pytest.approx(1.645 ** 2 * var / mean ** 2, rel=1e-9)


# --- Zufallsversuche und Wettbewerb gegen eine Schleife über dieselben Zahlen -------------------------------------------------------------

def test_finite_matches_a_loop_over_the_same_draws():
    trials, n, seed, d = 60, 7, 4242, 0.10
    p, y = S.draw_events(trials, n, seed)
    got = A.finite(n, 10, trials, seed)
    for rule in C.RULES:
        counts = {"over": [0, 0, 0], "under": [0, 0, 0]}
        for t in range(trials):
            honest = sum(py_score(rule, float(p[t, j]), float(y[t, j])) for j in range(n))
            for label, dd in (("over", d), ("under", -d)):
                other = 0.0
                for j in range(n):
                    pj = float(p[t, j])
                    dist = abs(pj - 0.5)                                                  # Abstand der Meldung von 1/2
                    new_dist = dist + dd if dd > 0 else max(dist + dd, 0.0)               # Untertreiben (dd < 0) stoppt bei 1/2, geht nicht darüber hinaus
                    q = min(max(0.5 + (1 if pj >= 0.5 else -1) * new_dist, 0.01), 0.99)
                    other += py_score(rule, q, float(y[t, j]))
                diff = honest - other
                counts[label][0 if diff > 1e-12 else 1 if abs(diff) <= 1e-12 else 2] += 1
        for label in ("over", "under"):
            assert [got[rule][label][k] for k in ("win", "tie", "lose")] == pytest.approx([c / trials * 100 for c in counts[label]]), (rule, label)


def test_contest_matches_a_loop_over_the_same_draws():
    trials, m, n, seed = 40, 3, 6, 777
    belief, y = S.draw_contest(trials, m, n, seed)
    lams = [0.5, 1.0, 2.0]
    got = A.contest("Brier", m, lams, trials, seed, n)
    for lam in lams:
        wins = 0.0
        score_sum = 0.0
        for t in range(trials):
            totals = []
            for i in range(m):
                tot = 0.0
                for j in range(n):
                    b = float(belief[t, i, j])
                    q = min(max(b, 0.01), 0.99)
                    if i == 0:
                        z = lam * math.log(b / (1 - b))
                        q = min(max(1 / (1 + math.exp(-z)), 0.01), 0.99)
                    tot += py_score("Brier", q, float(y[t, j]))
                totals.append(tot)
            top = max(totals)
            leaders = [i for i, v in enumerate(totals) if v >= top - 1e-12]
            wins += (1 / len(leaders)) if 0 in leaders else 0
            score_sum += totals[0] / n
        assert got[lam]["win"] == pytest.approx(wins / trials * 100) and got[lam]["mean"] == pytest.approx(score_sum / trials)


def test_draws_are_deterministic_and_inside_their_ranges():
    p1, y1 = S.draw_events(30, 10, 5)
    p2, y2 = S.draw_events(30, 10, 5)
    assert (p1 == p2).all() and (y1 == y2).all() and p1.min() >= 0.05 and p1.max() <= 0.95 and set(np.unique(y1)) <= {0.0, 1.0}
    b1, _ = S.draw_contest(20, 4, 6, 9)
    b2, _ = S.draw_contest(20, 4, 6, 9)
    assert (b1 == b2).all() and b1.shape == (20, 4, 6) and 0 < b1.min() and b1.max() < 1
