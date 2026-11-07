"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, und was jedes Preset zeigen soll."""

import zeh_algorithm as A
import zeh_constants as C
import zeh_evaluation as ev
import zeh_presets as P

(P_BRIER, P_ABS, P_HIT, P_LOG, P_TEN, P_CONTEST, P_MIX, P_TWO) = list(C.PRESETS)


def _s(name):
    return ev.Settings(**C.PRESETS[name])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert C.P_MIN <= preset["p"] <= C.P_MAX and 0 <= preset["rule"] < len(C.RULES) and C.D_MIN <= preset["d"] <= C.D_MAX and C.N_MIN <= preset["n"] <= C.N_MAX
        assert C.M_MIN <= preset["m"] <= C.M_MAX and C.LAM_MIN <= preset["lam"] <= C.LAM_MAX and preset["lam"] % C.LAM_STEP == 0
        assert C.MIX_MIN <= preset["mix"] <= C.MIX_MAX and preset["mix"] % C.MIX_STEP == 0 and 0 <= preset["seed"] <= C.SEED_MAX and preset["n"] % C.N_STEP == 0


def test_default_preset_equals_the_default_settings():
    assert _s(P_BRIER) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_no_non_default_preset_equals_the_default_settings():
    """Lehre aus Stück 4: ein Preset, das den Voreinstellungen gleicht, wäre nicht von „nichts geklickt“ zu unterscheiden."""
    for name, preset in C.PRESETS.items():
        if name != P_BRIER:
            assert ev.Settings(**preset) != ev.Settings(), name
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)


def test_brier_preset_shows_the_unique_honest_report_and_the_weak_incentive():
    s = _s(P_BRIER)
    b, inc, fin = ev.best_info(s), ev.incentive_info(s), ev.finite_info(s)
    assert b["unique"] and str(inc["brier_exact"]) == "1/100" and round(inc["rows"]["Brier"]["n95"]) == 227 and round(fin["rules"]["Brier"]["over"]["win"], 1) == 68.8


def test_absolute_error_preset_rewards_the_lie():
    s = _s(P_ABS)
    b, fin = ev.best_info(s), ev.finite_info(s)
    assert b["best"] == [1] and round(b["gain"], 3) == 0.116 and round(fin["rules"]["Absolutfehler"]["over"]["win"], 1) == 0.1 and fin["n"] == 50


def test_hit_rate_preset_is_indifferent():
    s = _s(P_HIT)
    b, fin = ev.best_info(s), ev.finite_info(s)
    assert b["best"] == list(range(1, 50)) and b["honest_best"] and not b["unique"] and fin["rules"]["Trefferquote"]["over"]["tie"] == 100.0
    assert round(A.loss("Trefferquote", 30, 60), 12) == 0.4 and round(A.expected("Trefferquote", 0.3, 0.6), 12) == 0.3 and round(A.expected("Trefferquote", 0.3, 0.3), 12) == 0.7
    assert "kostet 0,4 Punkte je Ereignis (erwartet 0,3 statt 0,7)" in C.PRESET_HELP[P_HIT]                  # die Lüge über die 50-%-Grenze kostet, die Hilfe sagt es


def test_log_preset_costs_0_0204_and_is_detectable_after_267_events():
    s = _s(P_LOG)
    inc, fin = ev.incentive_info(s), ev.finite_info(s)
    assert round(inc["rows"]["Log"]["loss"], 4) == 0.0204 and round(inc["rows"]["Log"]["n95"]) == 267 and round(fin["rules"]["Log"]["over"]["win"], 1) == 70.4


def test_ten_events_preset_loses_in_39_5_percent():
    fin = ev.finite_info(_s(P_TEN))
    assert fin["n"] == 10 and (round(fin["rules"]["Brier"]["over"]["win"], 1), round(fin["rules"]["Brier"]["over"]["lose"], 1)) == (60.5, 39.5)


def test_contest_preset_lets_the_exaggerator_win_more_with_fewer_points():
    c = ev.contest_info(_s(P_CONTEST))
    assert (c["m"], c["rule"], round(c["win"], 1), round(c["win_honest"], 1), round(c["mean_honest"], 4), round(c["mean"], 4)) == (20, "Log", 7.6, 5.0, -0.6026, -0.6421)
    assert c["win"] > c["win_honest"] and c["mean"] < c["mean_honest"]


def test_mix_preset_shifts_the_best_report_by_three_points():
    mi = ev.mix_info(_s(P_MIX))
    assert mi["best"] == [83] and mi["shift"] == 3 and not mi["honest_best"] and round(mi["q_star"], 1) == 83.3


def test_two_player_preset_punishes_the_exaggerator():
    c = ev.contest_info(_s(P_TWO))
    assert (c["m"], round(c["win"], 1), round(c["win_honest"], 1)) == (2, 40.5, 50.6) and ev.finite_info(_s(P_TWO))["n"] == 10
