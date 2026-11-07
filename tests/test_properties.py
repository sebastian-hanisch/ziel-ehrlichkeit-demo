"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Aussagen, die für jeden Glauben, jede Meldung und jede Regel gelten müssen."""

import numpy as np
import pytest

import zeh_algorithm as A
import zeh_constants as C
import zeh_evaluation as ev

PROPER = [C.RULES[i] for i in C.PROPER]
QS = np.array(C.GRID) / 100.0


@pytest.mark.parametrize("rule", PROPER)
def test_proper_rules_make_honesty_the_unique_best_for_every_belief(rule):
    for p in range(1, 100):
        assert A.best_set(rule, p) == [p]


@pytest.mark.parametrize("rule", PROPER)
def test_proper_rules_never_reward_a_lie(rule):
    for p in range(1, 100, 7):
        curve = A.expected_curve(rule, p)
        assert (curve <= curve[p - 1] + 1e-12).all() and ((curve < curve[p - 1] - 1e-15).sum() == 98)


@pytest.mark.parametrize("rule", PROPER)
def test_expected_score_rises_up_to_the_belief_and_falls_after_it_for_proper_rules(rule):
    for p in (5, 30, 50, 80):
        d = np.diff(A.expected_curve(rule, p))
        assert (d[: p - 1] > 0).all() and (d[p - 1:] < 0).all()


@pytest.mark.parametrize("rule", ("Brier", "Log"))
def test_expected_score_is_concave_in_the_report_for_brier_and_log(rule):
    for p in (5, 30, 50, 80):
        c = A.expected_curve(rule, p)
        assert (c[:-2] + c[2:] - 2 * c[1:-1] < 1e-12).all()


def test_the_spherical_rule_is_proper_but_not_concave_in_the_report():
    """Messung: die erwartete sphärische Punktzahl hat Wendepunkte (strikt eigentlich heißt nicht konkav)."""
    c = A.expected_curve("sphärisch", 30)
    assert (c[:-2] + c[2:] - 2 * c[1:-1] > 1e-9).any()


def test_brier_loss_is_exactly_the_squared_distance():
    for p in range(1, 100, 6):
        for q in range(1, 100, 9):
            assert A.loss("Brier", p, q) == pytest.approx(((q - p) / 100) ** 2, abs=1e-12)


@pytest.mark.parametrize("rule", C.RULES)
def test_mirror_symmetry(rule):
    for p in (10, 30, 50, 77):
        for q in (5, 25, 50, 60, 95):
            assert float(A.expected(rule, p / 100, q / 100)) == pytest.approx(float(A.expected(rule, 1 - p / 100, 1 - q / 100)), abs=1e-12)


def test_absolute_error_is_linear_in_the_report_with_slope_two_p_minus_one():
    for p in (10, 30, 70, 90):
        c = A.expected_curve("Absolutfehler", p)
        assert np.allclose(np.diff(c), (2 * p / 100 - 1) / 100) and np.allclose(c[:-2] + c[2:] - 2 * c[1:-1], 0, atol=1e-12)
        best = A.best_set("Absolutfehler", p)
        assert best == ([99] if p > 50 else [1])


def test_hit_rate_only_sees_the_side_of_one_half():
    for p in (10, 30, 70, 90):
        c = A.expected_curve("Trefferquote", p)
        below, above = c[:49], c[50:]
        assert np.allclose(below, below[0]) and np.allclose(above, above[0]) and c[49] == pytest.approx(0.5)
        assert A.loss("Trefferquote", p, 20 if p < 50 else 80) == pytest.approx(0.0)


def test_hit_rate_punishes_exactly_the_lies_that_leave_the_side_of_one_half():
    """Lügen kosten bei der Trefferquote nur, wenn die Meldung auf die andere Seite von 1/2 oder genau auf 1/2 rückt (und p ≠ 1/2); auf derselben Seite nie."""
    sgn = lambda x: (x > 50) - (x < 50)
    costly = lies = 0
    for p in range(1, 100):
        for q in range(1, 100):
            if q == p:
                continue
            lost = A.exact_expected("Trefferquote", A.Fraction(p, 100), A.Fraction(p, 100)) - A.exact_expected("Trefferquote", A.Fraction(p, 100), A.Fraction(q, 100))
            assert lost >= 0
            assert (lost > 0) == (p != 50 and sgn(q) != sgn(p)), (p, q)
            lies += 1
            costly += lost > 0
    assert (costly, lies) == (4900, 9702)
    assert not any(A.best_set("Trefferquote", p) == [p] for p in range(1, 100))             # ehrlich ist nie die einzige beste


@pytest.mark.parametrize("rule", ("Trefferquote", "Absolutfehler"))
def test_improper_rules_never_make_honesty_the_unique_best_for_inner_beliefs(rule):
    for p in range(2, 99):
        assert A.best_set(rule, p) != [p]


def test_incentive_is_never_negative_for_proper_rules_and_n95_is_positive():
    for rule in PROPER:
        for p in range(1, 100, 11):
            for q in range(1, 100, 13):
                if q != p:
                    assert A.loss(rule, p, q) > 0 and A.n95(rule, p, q) > 0


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_finite_shares_add_to_one_hundred_and_are_deterministic(seed):
    a = A.finite(20, 10, 300, seed)
    b = A.finite(20, 10, 300, seed)
    assert a == b
    for rule in C.RULES:
        for key in ("over", "under"):
            assert sum(a[rule][key].values()) == pytest.approx(100.0)


def test_hit_rate_is_blind_to_exaggeration_and_absolute_error_rewards_it():
    r = A.finite(50, 10, 400, 11)
    assert r["Trefferquote"]["over"] == {"win": 0.0, "tie": 100.0, "lose": 0.0}
    assert r["Absolutfehler"]["over"]["win"] < 5 and r["Absolutfehler"]["under"]["win"] > 95


def test_honesty_wins_more_often_with_more_events_for_proper_rules():
    shares = {n: A.finite(n, 10, 1000, 3) for n in (10, 100, 500)}
    for rule in PROPER:
        w = [shares[n][C.RULES[C.RULES.index(rule)]]["over"]["win"] for n in (10, 100, 500)]
        assert w == sorted(w)


def test_hit_rate_contest_does_not_depend_on_the_scaling():
    res = A.contest("Trefferquote", 5, [0.5, 1.0, 1.5, 3.0], 500, 9)
    assert len({round(v["win"], 9) for v in res.values()}) == 1 and len({round(v["mean"], 9) for v in res.values()}) == 1


def test_scaling_one_is_the_honest_baseline_and_extreme_scaling_costs_points():
    res = A.contest("Brier", 5, [0.5, 1.0, 1.5, 3.0], 1500, 9)
    assert res[1.0]["mean"] > res[0.5]["mean"] and res[1.0]["mean"] > res[1.5]["mean"] > res[3.0]["mean"]


@pytest.mark.parametrize("p", (10, 30, 60, 90))
def test_mix_shift_has_the_sign_of_the_belief_and_shrinks_with_more_brier(p):
    shifts = [A.best_set_mix(p, mx)[0] - p for mx in range(10, 101, 10)]
    if p > 50:
        assert all(s >= 0 for s in shifts) and shifts == sorted(shifts, reverse=True)
    else:
        assert all(s <= 0 for s in shifts) and shifts == sorted(shifts)
    assert shifts[-1] == 0


def test_mix_is_honest_only_at_pure_brier_or_belief_one_half():
    res = ev.mix_over_all_p()
    assert [mx for mx, v in res.items() if v["unique"] == 99] == [100]
    assert all(v["unique"] >= 1 for mx, v in res.items() if mx > 0)            # bei p = 50 % hat die Verzerrung keinen Ansatzpunkt
    means = [res[mx]["mean_shift"] for mx in range(10, 101, 10)]
    assert means == sorted(means, reverse=True) and means[-1] == 0
