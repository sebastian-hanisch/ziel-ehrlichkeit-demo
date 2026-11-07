"""Jede Zahl aus dem README wird hier nachgerechnet (DEMO-PLAYBOOK §4). Exakte Rechnungen und feste Seeds sind deterministisch."""

from pathlib import Path

import numpy as np

import zeh_algorithm as A
import zeh_constants as C
import zeh_evaluation as ev

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
(P_BRIER, P_ABS, P_HIT, P_LOG, P_TEN, P_CONTEST, P_MIX, P_TWO) = list(C.PRESETS)
NS = C.SWEEP_NS
LAMS = [x / 10.0 for x in C.SWEEP_LAMS]
d1 = lambda x: ev.de(x, 1)


def _claim(text):
    assert text in README, text


def _slash(values, fmt=d1):
    return " / ".join(fmt(v) for v in values)


def _s(name):
    return ev.Settings(**C.PRESETS[name])


# --- Exakte Rechnung über alle 99 Glauben ----------------------------------------------------------------------------------------------------

def test_unique_among_lose_and_gain_over_all_beliefs():
    unique, among, lose = [], [], []
    for rule in C.RULES:
        u = a = lo = 0
        for p in range(1, 100):
            best = A.best_set(rule, p)
            u += best == [p]
            a += p in best
            lo += p not in best
        unique.append(u)
        among.append(a)
        lose.append(lo)
    _claim(f"**{' / '.join(str(x) for x in unique[:3])} / {unique[3]} / {unique[4]}**")
    _claim(" / ".join(str(x) for x in among))
    _claim(f"0 / 0 / 0 / 0 / **{lose[4]} von 99**")
    gains = [ev.best_info(ev.Settings(p=p, rule=4))["gain"] for p in range(1, 100)]
    _claim(f"**{ev.de(float(np.mean(gains)), 4)} / {ev.de(max(gains), 4)}**")
    for rule in range(4):
        assert max(ev.best_info(ev.Settings(p=p, rule=rule))["gain"] for p in range(1, 100)) < 1e-12


def test_hit_rate_punishes_only_lies_across_one_half():
    costly = lies = same_side = 0
    for p in range(1, 100):
        for q in range(1, 100):
            if q != p:
                lies += 1
                lost = A.exact_expected("Trefferquote", A.Fraction(p, 100), A.Fraction(p, 100)) - A.exact_expected("Trefferquote", A.Fraction(p, 100), A.Fraction(q, 100))
                costly += lost > 0
                same_side += lost > 0 and (q > 50) == (p > 50) and q != 50 and p != 50
    assert (costly, lies, same_side) == (4900, 9702, 0)
    _claim("(von 9 702 Paaren) | **4 900**;")
    _claim("Beispiel `p` = 30 %, Meldung 60 %: erwartet 0,3 statt 0,7")
    _claim("kostet 0,4 (erwartet 0,3 statt 0,7)")
    _claim("4 900 von 9 702 Paaren `p`, `q` mit `q` ≠ `p`")
    assert abs(A.expected("Trefferquote", 0.3, 0.6) - 0.3) < 1e-12 and abs(A.expected("Trefferquote", 0.3, 0.3) - 0.7) < 1e-12


def test_losses_and_detection_counts():
    for p, q in ((30, 40), (50, 60), (80, 90)):
        assert round(A.loss("Brier", p, q), 4) == 0.01
    _claim("Brier **0,0100** bei allen (exakt 1/100)")
    _claim("Log " + " / ".join(ev.de(A.loss("Log", p, p + 10), 4) for p in (30, 50, 80)))
    _claim("sphärisch " + " / ".join(ev.de(A.loss("sphärisch", p, p + 10), 4) for p in (30, 50, 80)))
    for rule in ("Brier", "Log", "sphärisch"):
        _claim(f"{rule} " + " / ".join(ev.de(A.n95(rule, p, p + 10), 0) for p in (30, 50, 80)))


def test_mix_over_all_beliefs_and_the_formula():
    res = ev.mix_over_all_p()
    _claim(" / ".join(str(res[mx]["unique"]) for mx in range(0, 100, 10)) + f" / **{res[100]['unique']}**")
    cells = [f"{d1(res[mx]['mean_shift'])}; {res[mx]['max_shift']}" for mx in range(0, 101, 10)]
    cells[9] = f"**{cells[9]}**"
    _claim(" / ".join(cells))
    hits = 0
    for mx in range(10, 101, 10):
        for p in range(1, 100):
            hits += abs(A.best_set_mix(p, mx)[0] - A.mix_optimum(p, mx) * 100) <= A.Fraction(1, 2)
    assert hits == 99 * 10
    _claim("in 99 von 99 Glauben bei jedem `μ` von 10 bis 100 %")
    assert max(res[mx]["unique"] for mx in range(0, 100, 10)) == 11
    _claim("bei keinem kleineren Anteil in mehr als 11")


# --- Zählungen --------------------------------------------------------------------------------------------------------------------------------

def _over(f, rule, key="win"):
    return [f[n][rule]["over"][key] for n in NS]


def _under(f, rule, key="win"):
    return [f[n][rule]["under"][key] for n in NS]


def test_finite_sweep_claims(sweeps):
    f = sweeps["finite"]
    _claim(f"Brier **{_slash(_over(f, 'Brier'))} %**")
    _claim(f"Log {_slash(_over(f, 'Log'))} %")
    _claim(f"sphärisch {_slash(_over(f, 'sphärisch'))} %")
    _claim(_slash(_over(f, "Brier", "lose")) + " %")
    _claim(f"Brier {_slash(_under(f, 'Brier'))} %")
    _claim(f"Log {_slash(_under(f, 'Log'))} %")
    _claim(f"sphärisch {_slash(_under(f, 'sphärisch'))} %")
    _claim(f"{_slash(_over(f, 'Absolutfehler'))} % und {_slash(_under(f, 'Absolutfehler'), lambda x: ev.de(x, 1) if x not in (100.0,) else '100')} %")
    _claim(f"{_slash(_over(f, 'Trefferquote', 'tie'), lambda x: ev.de(x, 0))} %** und {_slash(_under(f, 'Trefferquote'))} %")
    _claim(f"(Gleichstand {_slash(_under(f, 'Trefferquote', 'tie'))} %)")
    _claim(f"(−10; höchstens bis 50 %) | Brier {_slash(_under(f, 'Brier'))} %")
    _claim(f"Brier bei 10 Ereignissen {d1(f[10]['Brier']['under']['win'])} statt 66,3 %, Absolutfehler {d1(f[10]['Absolutfehler']['under']['win'])} statt 89,6 %")

def _wins(c, rule, m):
    return [c[rule][m][lam]["win"] for lam in LAMS]


def test_contest_claims(sweeps):
    c, h = sweeps["contest"], sweeps["holdout"]
    _claim(f"{d1(_wins(c, 'Brier', 20)[0])} / {d1(_wins(c, 'Brier', 20)[1])} / **{d1(_wins(c, 'Brier', 20)[2])}** / **{d1(_wins(c, 'Brier', 20)[3])}** / {d1(_wins(c, 'Brier', 20)[4])} %")
    _claim(f"(Hold-out {_slash(_wins(h, 'Brier', 20))} %)")
    lg = _wins(c, "Log", 20)
    _claim(f"{d1(lg[0])} / {d1(lg[1])} / **{d1(lg[2])}** / {d1(lg[3])} / {d1(lg[4])} %")
    _claim(f"(Hold-out {_slash(_wins(h, 'Log', 20))} %)")
    _claim(f"{_slash(_wins(c, 'Brier', 10))} % und {_slash(_wins(c, 'Log', 10))} %")
    _claim(f"(Hold-out Brier {_slash(_wins(h, 'Brier', 10))} %)")
    _claim(f"{_slash(_wins(c, 'Brier', 5))} % und {_slash(_wins(c, 'Log', 5))} %")
    _claim(f"{_slash(_wins(c, 'Brier', 2))} % und {_slash(_wins(c, 'Log', 2))} %")
    means = [c["Brier"][20][lam]["mean"] for lam in LAMS]
    _claim(" / ".join(ev.de(x, 4) if i != 1 else f"**{ev.de(x, 4)}**" for i, x in enumerate(means)).replace("-", "−"))
    _claim(" / ".join(d1(c["Trefferquote"][m][0.5]["win"]) for m in C.SWEEP_MS) + " %")
    assert all(len({round(c["Trefferquote"][m][lam]["win"], 9) for lam in LAMS}) == 1 for m in C.SWEEP_MS)
    _claim("(Brier 10,9 gegen 9,9 %, im Hold-out 10,9 gegen 9,7 %)")
    assert (d1(c["Brier"][10][1.5]["win"]), d1(c["Brier"][10][1.0]["win"]), d1(h["Brier"][10][1.5]["win"]), d1(h["Brier"][10][1.0]["win"])) == ("10,9", "9,9", "10,9", "9,7")


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------

def test_brier_preset():
    s = _s(P_BRIER)
    inc, fin = ev.incentive_info(s), ev.finite_info(s)
    _claim(f"erkennbar nach {ev.de(inc['rows']['Brier']['n95'], 0)} Ereignissen; nach 20 Ereignissen ehrlich vorn in **{d1(fin['rules']['Brier']['over']['win'])} %**, vor untertreibend {d1(fin['rules']['Brier']['under']['win'])} %")


def test_absolute_error_preset():
    s = _s(P_ABS)
    b, fin = ev.best_info(s), ev.finite_info(s)
    assert (ev.de(b["truth"], 4), ev.de(b["top"], 4)) == ("-0,4200", "-0,3040")
    _claim(f"beste Meldung **{b['best'][0]} %**, Gewinn {ev.de(b['gain'], 4)} (−0,3040 gegen −0,4200)")
    _claim(f"ehrlich vor übertreibend in **{d1(fin['rules']['Absolutfehler']['over']['win'])} %** der Versuche")


def test_hit_rate_preset():
    s = _s(P_HIT)
    b, fin = ev.best_info(s), ev.finite_info(s)
    _claim(f"ehrlich eine von **{len(b['best'])}** gleich guten Meldungen ({b['best'][0]} % bis {b['best'][-1]} %); die Meldung 60 % auf der anderen Seite kostet 0,4 (erwartet 0,3 statt 0,7); Gleichstand gegen Übertreiben in {d1(fin['rules']['Trefferquote']['over']['tie'])} %, ehrlich vor untertreibend in {d1(fin['rules']['Trefferquote']['under']['win'])} %")


def test_log_preset():
    s = _s(P_LOG)
    inc, fin = ev.incentive_info(s), ev.finite_info(s)
    _claim(f"Meldung 60 % kostet **{ev.de(inc['rows']['Log']['loss'], 4)}**, erkennbar nach {ev.de(inc['rows']['Log']['n95'], 0)} Ereignissen; ehrlich vor übertreibend in {d1(fin['rules']['Log']['over']['win'])} %")


def test_ten_events_preset():
    fin = ev.finite_info(_s(P_TEN))
    r = fin["rules"]["Brier"]
    _claim(f"ehrlich vor übertreibend in **{d1(r['over']['win'])} %**, verliert in {d1(r['over']['lose'])} %; vor untertreibend {d1(r['under']['win'])} %")


def test_contest_preset():
    c = ev.contest_info(_s(P_CONTEST))
    _claim(f"Abweichler gewinnt **{d1(c['win'])} %**, ehrlich {d1(c['win_honest'])} %")
    _claim("mittlere Punktzahl −0,6421 gegen −0,6026")
    assert (ev.de(c["mean"], 4), ev.de(c["mean_honest"], 4)) == ("-0,6421", "-0,6026")


def test_mix_preset():
    mi = ev.mix_info(_s(P_MIX))
    _claim(f"beste Meldung **{mi['best'][0]} %** (+{mi['shift']} Prozentpunkte), Formel {d1(mi['q_star'])} %")


def test_two_player_preset():
    c = ev.contest_info(_s(P_TWO))
    _claim(f"Abweichler gewinnt {d1(c['win'])} %, ehrlich {d1(c['win_honest'])} %")


# --- Korrektur der Planungshypothese -----------------------------------------------------------------------------------------------------------

def test_correction_numbers(sweeps):
    f = sweeps["finite"]
    _claim("bei `μ` = 90 % um 2,4 Prozentpunkte im Mittel (höchstens 5)")
    _claim(f"bei 10 Ereignissen (Brier) {d1(f[10]['Brier']['over']['win'])} gegen 62,1 %")
    _claim(f"20 Prognostikern bei Skalierung 1,5 (Brier) {d1(sweeps['contest']['Brier'][20][1.5]['win'])} gegen 6,9 %")
    spher = A.expected_curve("sphärisch", 30)
    assert (spher[:-2] + spher[2:] - 2 * spher[1:-1] > 1e-9).any()


def test_readme_names_every_preset():
    for name in C.PRESETS:
        _claim(f"Preset „{name}“")


def test_readme_test_count_is_filled_in():
    assert "TESTZAHL" not in README
