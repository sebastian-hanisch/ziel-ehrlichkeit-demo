"""Zählungen über feste Zufallsversuche (2 000 Versuche je Ereigniszahl; 20 000 Versuche je Wettbewerbszelle): die Zahlen aus der App und dem README, hier festgehalten."""

import zeh_constants as C
import zeh_evaluation as ev

NS = C.SWEEP_NS
LAMS = [x / 10.0 for x in C.SWEEP_LAMS]


def _over(res, rule, key="win"):
    return [round(res[n][rule]["over"][key], 1) for n in NS]


def _under(res, rule, key="win"):
    return [round(res[n][rule]["under"][key], 1) for n in NS]


def test_finite_sweep_honest_rises_with_the_number_of_events_for_proper_rules(sweeps):
    f = sweeps["finite"]
    for rule in ("Brier", "Log", "sphärisch"):
        w = _over(f, rule)
        assert w == sorted(w) and w[0] < 62 and w[-1] > 98
        assert abs(sum(f[NS[0]][rule]["over"].values()) - 100.0) < 1e-9


def test_finite_sweep_numbers_over(sweeps):
    f = sweeps["finite"]
    assert _over(f, "Brier") == [60.5, 69.0, 79.3, 86.4, 99.1]
    assert _over(f, "Log") == [59.4, 69.5, 83.0, 91.1, 100.0]
    assert _over(f, "sphärisch") == [58.5, 67.1, 75.4, 83.5, 98.2]
    assert _over(f, "Absolutfehler") == [7.0, 2.2, 0.0, 0.0, 0.0] and _over(f, "Absolutfehler", "lose") == [90.7, 97.6, 100.0, 100.0, 100.0]
    assert _over(f, "Trefferquote", "tie") == [100.0] * 5 and _over(f, "Trefferquote") == [0.0] * 5


def test_finite_sweep_numbers_under(sweeps):
    f = sweeps["finite"]
    assert _under(f, "Brier") == [66.3, 71.7, 79.7, 87.6, 99.6]
    assert _under(f, "Log") == [65.8, 71.5, 79.8, 88.1, 99.5]
    assert _under(f, "sphärisch") == [65.1, 70.5, 78.0, 86.5, 99.2]
    assert _under(f, "Absolutfehler") == [89.6, 97.2, 100.0, 100.0, 100.0]
    assert _under(f, "Trefferquote") == [42.5, 49.1, 56.8, 66.1, 82.5] and _under(f, "Trefferquote", "tie") == [27.8, 17.8, 11.1, 7.3, 2.3]


def test_finite_sweep_loss_share_for_brier(sweeps):
    assert _over(sweeps["finite"], "Brier", "lose") == [39.5, 31.0, 20.6, 13.6, 0.9]


def _wins(res, rule, m):
    return [round(res[rule][m][lam]["win"], 1) for lam in LAMS]


def _means(res, rule, m):
    return [round(res[rule][m][lam]["mean"], 4) for lam in LAMS]


def test_contest_sweep_sample_numbers(sweeps):
    c = sweeps["contest"]
    assert _wins(c, "Brier", 20) == [3.4, 5.1, 7.0, 7.1, 6.0] and _wins(c, "Brier", 10) == [6.8, 9.9, 10.9, 10.1, 8.2]
    assert _wins(c, "Brier", 5) == [14.7, 20.2, 18.4, 15.5, 11.6] and _wins(c, "Brier", 2) == [42.6, 49.9, 41.0, 33.0, 23.9]
    assert _wins(c, "Log", 20) == [3.8, 5.0, 8.2, 7.8, 5.1] and _wins(c, "Log", 10) == [7.7, 10.1, 12.0, 10.2, 6.3]
    assert _wins(c, "Log", 5) == [15.6, 20.0, 17.8, 13.6, 7.8] and _wins(c, "Log", 2) == [42.5, 49.9, 36.7, 25.1, 13.0]
    assert [_wins(c, "Trefferquote", m)[0] for m in C.SWEEP_MS] == [50.1, 20.0, 9.8, 5.2]
    assert all(len(set(_wins(c, "Trefferquote", m))) == 1 for m in C.SWEEP_MS)


def test_contest_sweep_mean_scores_fall_with_extreme_scaling(sweeps):
    c = sweeps["contest"]
    assert _means(c, "Brier", 20) == [-0.2107, -0.2067, -0.2156, -0.2266, -0.2449] and _means(c, "Log", 20)[1:] == [-0.603, -0.6428, -0.7081, -0.8445]
    for rule in ("Brier", "Log"):
        for m in C.SWEEP_MS:
            means = _means(c, rule, m)
            assert means[1] == max(means)                                          # ehrlich hat die höchste mittlere Punktzahl


def test_contest_sweep_extreme_scaling_pays_only_with_many_players(sweeps):
    for key in ("contest", "holdout"):
        c = sweeps[key]
        for rule in ("Brier", "Log"):
            hon = {m: c[rule][m][1.0]["win"] for m in C.SWEEP_MS}
            assert c[rule][2][1.5]["win"] < hon[2] and c[rule][5][1.5]["win"] < hon[5]            # bei 2 und 5 Prognostikern lohnt Übertreiben nicht
            assert c[rule][20][1.5]["win"] > hon[20] and c[rule][20][2.0]["win"] > hon[20]        # bei 20 schon


def test_holdout_numbers(sweeps):
    h = sweeps["holdout"]
    assert _wins(h, "Brier", 20) == [3.0, 5.0, 7.0, 7.1, 6.2] and _wins(h, "Log", 20) == [3.6, 5.1, 8.2, 8.0, 5.4]
    assert _wins(h, "Brier", 10) == [6.7, 9.7, 10.9, 10.0, 8.0] and _wins(h, "Log", 10) == [7.7, 9.9, 11.9, 10.1, 6.3]
    assert _wins(h, "Brier", 2) == [43.2, 50.0, 41.0, 33.0, 23.7] and _wins(h, "Log", 2) == [43.5, 49.9, 36.5, 24.5, 12.7]
    assert [_wins(h, "Trefferquote", m)[0] for m in C.SWEEP_MS] == [50.3, 20.2, 9.8, 5.1]


def test_holdout_agrees_with_the_sample_within_one_percentage_point(sweeps):
    for rule in ("Brier", "Log"):
        for m in (10, 20):
            for lam in LAMS:
                assert abs(sweeps["contest"][rule][m][lam]["win"] - sweeps["holdout"][rule][m][lam]["win"]) < 1.0


def test_mix_over_all_beliefs():
    res = ev.mix_over_all_p()
    assert [res[mx]["unique"] for mx in range(0, 101, 10)] == [2, 3, 3, 3, 3, 3, 3, 5, 5, 11, 99]
    assert [round(res[mx]["mean_shift"], 1) for mx in range(0, 101, 10)] == [24.3, 21.8, 19.4, 17.0, 14.5, 12.1, 9.7, 7.3, 4.8, 2.4, 0.0]
    assert [res[mx]["max_shift"] for mx in range(0, 101, 10)] == [49, 44, 39, 34, 29, 24, 19, 15, 10, 5, 0]
