"""Randfälle: extreme Glauben, keine Verschiebung, Verschiebung bis an den Rand, kleinste und größte Ereigniszahl, zwei Prognostiker, Mischung 0 und 100, Seeds am Rand."""

from fractions import Fraction

import pytest

import zeh_algorithm as A
import zeh_constants as C
import zeh_evaluation as ev


@pytest.mark.parametrize("rule", C.RULES)
def test_every_rule_works_at_the_extreme_beliefs(rule):
    for p in (1, 99):
        s = ev.Settings(p=p, rule=C.RULES.index(rule))
        info = ev.best_info(s)
        assert len(info["curve"]) == 99 and all(v == v for v in info["curve"]) and info["gain"] >= -1e-12
        assert ev.verdict_best(info)[0] in ("success", "warning", "error")


def test_absolute_error_at_the_extreme_beliefs_is_honest_at_the_edge_of_the_grid():
    assert A.best_set("Absolutfehler", 1) == [1] and A.best_set("Absolutfehler", 99) == [99]


def test_one_half_makes_absolute_error_and_hit_rate_flat():
    for rule in ("Absolutfehler", "Trefferquote"):
        info = ev.best_info(ev.Settings(p=50, rule=C.RULES.index(rule)))
        assert info["best"] == list(C.GRID) and not info["unique"] and info["honest_best"] and info["gain"] == 0
        assert ev.verdict_best(info)[0] == "warning" and "eine von 99 gleich guten" in ev.verdict_best(info)[1]


def test_no_shift_means_nothing_to_detect():
    inc = ev.incentive_info(ev.Settings(d=0))
    assert inc["q"] == inc["p"] and all(abs(r["loss"]) < 1e-12 for r in inc["rows"].values()) and all(r["n95"] is None for r in inc["rows"].values())
    kind, text = ev.verdict_incentive(inc)
    assert kind == "info" and "Verschiebung 0" in text


def test_shift_is_clipped_at_the_edge_of_the_grid():
    s = ev.Settings(p=95, d=30)
    assert s.q == 99
    inc = ev.incentive_info(s)
    assert inc["q"] == 99 and inc["brier_exact"] == (Fraction(99, 100) - Fraction(95, 100)) ** 2


def test_smallest_and_largest_event_counts():
    for n in (C.N_MIN, C.N_MAX):
        info = ev.finite_info(ev.Settings(n=n))
        for rule in C.RULES:
            for key in ("over", "under"):
                assert sum(info["rules"][rule][key].values()) == pytest.approx(100.0)


def test_two_forecasters_never_reward_extreme_scaling_with_brier():
    info = ev.contest_info(ev.Settings(m=2, lam=30))
    assert info["win"] < info["win_honest"] and info["base"] == 50.0 and ev.verdict_contest(info)[0] == "info"


def test_scaling_one_in_the_contest_equals_the_honest_baseline():
    info = ev.contest_info(ev.Settings(lam=10))
    assert info["win"] == info["win_honest"] and info["mean"] == info["mean_honest"]
    assert ev.verdict_contest(info)[0] == "info" and "ehrlich" in ev.verdict_contest(info)[1]


def test_mix_zero_is_pure_absolute_error_and_mix_hundred_is_pure_brier():
    zero = ev.mix_info(ev.Settings(p=30, mix=0))
    assert zero["best"] == A.best_set("Absolutfehler", 30) and zero["q_star"] is None and zero["shift"] == -29
    hundred = ev.mix_info(ev.Settings(p=30, mix=100))
    assert hundred["unique"] and hundred["shift"] == 0 and ev.verdict_mix(hundred)[0] == "success"


def test_mix_at_one_half_is_honest_for_every_brier_share():
    for mx in range(0, 101, 10):
        info = ev.mix_info(ev.Settings(p=50, mix=mx))
        assert info["honest_best"]
        if 0 < mx < 100:
            assert info["unique"] and ev.verdict_mix(info)[0] == "info"


@pytest.mark.parametrize("seed", (0, 999999))
def test_seeds_at_the_edge(seed):
    s = ev.Settings(seed=seed, n=10, m=3)
    assert 0 <= ev.finite_info(s)["rules"]["Brier"]["over"]["win"] <= 100 and 0 <= ev.contest_info(s)["win"] <= 100


def test_settings_defaults_match_the_constants():
    s = ev.Settings()
    assert (s.p, s.rule, s.d, s.n, s.m, s.lam, s.mix, s.seed) == (C.DEFAULT_P, C.DEFAULT_RULE, C.DEFAULT_D, C.DEFAULT_N, C.DEFAULT_M, C.DEFAULT_LAM, C.DEFAULT_MIX, C.DEFAULT_SEED)
    assert s.rule_name == "Brier" and s.q == 40


def test_formatting_helpers():
    assert ev.de(0.04, 1) == "0,0" and ev.de(-0.0) == "0,0" and ev.pct(0.1234) == "12,3 %" and ev.de(0.116, 4) == "0,1160"
