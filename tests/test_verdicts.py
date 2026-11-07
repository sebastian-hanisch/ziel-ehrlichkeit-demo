"""Verdikte: Art und Wortlaut in den Fällen, die sie unterscheiden (DEMO-PLAYBOOK §6): jede Zahl im Text kommt aus der Rechnung."""

import zeh_constants as C
import zeh_evaluation as ev

(P_BRIER, P_ABS, P_HIT, P_LOG, P_TEN, P_CONTEST, P_MIX, P_TWO) = list(C.PRESETS)


def _s(name):
    return ev.Settings(**C.PRESETS[name])


def test_best_verdict_unique_tie_and_loss():
    assert ev.verdict_best(ev.best_info(_s(P_BRIER))) == ("success", "Bei p = 30 % ist die ehrliche Meldung bei der Regel Brier die **einzige beste**: jede Lüge kostet erwartete Punkte.")
    assert ev.verdict_best(ev.best_info(_s(P_HIT))) == ("warning", "Bei p = 30 % ist die ehrliche Meldung bei der Regel Trefferquote **eine von 49 gleich guten** Meldungen (1 % bis 49 %): "
                                                                  "Lügen innerhalb dieser Meldungen kostet nichts, eine Meldung außerhalb kostet erwartete Punkte; Ehrlichkeit wird hier nicht belohnt.")
    for rule in (3, 4):                                                                                                  # bei p = 50 % sind alle 99 Meldungen gleich gut
        kind, text = ev.verdict_best(ev.best_info(ev.Settings(p=50, rule=rule)))
        assert kind == "warning" and "eine von 99 gleich guten" in text and "(1 % bis 99 %): Lügen kostet nichts, Ehrlichkeit wird hier nicht belohnt." in text
    assert ev.verdict_best(ev.best_info(_s(P_ABS))) == ("error", "Bei p = 30 % ist die ehrliche Meldung bei der Regel Absolutfehler **nicht die beste**: die beste Meldung ist 1 % und bringt erwartet 0,1160 Punkte mehr "
                                                                 "als die ehrliche (-0,3040 gegen -0,4200).")
    assert "Regel Log die **einzige beste**" in ev.verdict_best(ev.best_info(_s(P_LOG)))[1]


def test_incentive_verdict_cases():
    kind, text = ev.verdict_incentive(ev.incentive_info(_s(P_BRIER)))
    assert kind == "info" and text == ("Die Meldung 40 % statt 30 % kostet erwartet pro Ereignis: Brier 0,0100 (erkennbar nach 227 Ereignissen), Log 0,0216 (erkennbar nach 238 Ereignissen), "
                                       "sphärisch 0,0127 (erkennbar nach 215 Ereignissen). Beim Brier-Maß ist der Verlust genau (q − p)² = 1/100 = 0,0100.")
    text = ev.verdict_incentive(ev.incentive_info(_s(P_LOG)))[1]
    assert "Die Meldung 60 % statt 50 % kostet" in text and "Log 0,0204 (erkennbar nach 267 Ereignissen)" in text and text.endswith("Beim Absolutfehler und bei der Trefferquote ist die Meldung gleich gut.")
    text = ev.verdict_incentive(ev.incentive_info(ev.Settings(p=80, d=10)))[1]
    assert "Log 0,0444 (erkennbar nach 144 Ereignissen)" in text and text.endswith("Beim Absolutfehler **gewinnt** die Meldung 90 % erwartet 0,0600 Punkte.")


def test_finite_verdict_cases():
    kind, text = ev.verdict_finite(ev.finite_info(_s(P_BRIER)), "Brier")
    assert kind == "warning" and text == ("Bei der Regel Brier liegt die ehrliche Meldung nach 20 Ereignissen in **68,8 %** der 2000 Versuche strikt vor einer um 10 Prozentpunkte übertriebenen und in "
                                          "**69,5 %** vor einer untertriebenen; in 31,2 % beziehungsweise 30,6 % verliert sie.")
    text = ev.verdict_finite(ev.finite_info(_s(P_HIT)), "Trefferquote")[1]
    assert "**0,0 %**" in text and "(gleich viele Punkte: 100,0 % beziehungsweise 18,7 %)" in text
    kind, text = ev.verdict_finite(ev.finite_info(ev.Settings(n=500)), "Log")
    assert kind == "success" and "nach 500 Ereignissen" in text


def test_contest_verdict_cases():
    kind, text = ev.verdict_contest(ev.contest_info(_s(P_CONTEST)))
    assert kind == "warning" and text == ("Bei 20 Prognostikern und der Regel Log gewinnt der Prognostiker mit der Skalierung 1,5 in **7,6 %** der 4000 Versuche, ehrlich in 5,0 % (mehr als ehrlich); "
                                          "seine mittlere Punktzahl je Ereignis ist -0,6421 gegen -0,6026 bei ehrlicher Meldung.")
    kind, text = ev.verdict_contest(ev.contest_info(ev.Settings()))
    assert kind == "info" and "(weniger als ehrlich)" in text and "18,6 %" in text
    kind, text = ev.verdict_contest(ev.contest_info(_s(P_TWO)))
    assert kind == "info" and "Bei 2 Prognostikern" in text and "in **40,5 %**" in text and "ehrlich in 50,6 %" in text


def test_mix_verdict_cases():
    kind, text = ev.verdict_mix(ev.mix_info(_s(P_MIX)))
    assert kind == "warning" and text == ("Schon mit 10 % Absolutfehler in der Regel ist die beste Meldung bei p = 80 % nicht mehr die ehrliche: die beste Meldung ist 83 % (+3 Prozentpunkte); "
                                          "die Formel q* = p + (1 − m)(2p − 1) / (2m) mit m = 90 % gibt 83,3 %.")
    assert "(-20 Prozentpunkte)" in ev.verdict_mix(ev.mix_info(ev.Settings()))[1] and "gibt 10,0 %" in ev.verdict_mix(ev.mix_info(ev.Settings()))[1]
    assert ev.verdict_mix(ev.mix_info(ev.Settings(mix=100))) == ("success", "Mit 100 % Brier ist die ehrliche Meldung die einzige beste (p = 30 %).")
    assert ev.verdict_mix(ev.mix_info(ev.Settings(p=50)))[0] == "info"
    assert "gibt -" in ev.verdict_mix(ev.mix_info(ev.Settings(mix=0)))[1]


def test_sweep_verdicts(sweeps):
    kind, text = ev.verdict_finite_sweep(sweeps["finite"])
    assert kind == "info" and text.startswith("Ehrlich liegt strikt vor dem Übertreiben um 10 Prozentpunkte bei n = 10 / 20 / 50 / 100 / 500 Ereignissen in (je 2000 Versuche): Brier: 60,5 / 69,0 / 79,3 / 86,4 / 99,1 %")
    assert "Trefferquote: Gleichstand in 100 / 100 / 100 / 100 / 100 %" in text
    text = ev.verdict_contest_sweep(sweeps["contest"], "Stichprobe")[1]
    assert text.startswith("Stichprobe, Skalierung 0,5 / 1,0 / 1,5 / 2,0 / 3,0: Brier: m = 2 (Faustwert 50,0 %): 42,6 / 49,9 / 41,0 / 33,0 / 23,9 %; m = 20 (Faustwert 5,0 %): 3,4 / 5,1 / 7,0 / 7,1 / 6,0 %")
