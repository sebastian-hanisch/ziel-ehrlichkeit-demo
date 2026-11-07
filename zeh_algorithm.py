"""Kern: Scoring-Regeln (Brier, Log, sphärisch, Trefferquote, Absolutfehler), erwartete Punktzahl und beste Meldung, Anreizstärke, Zufallsversuche „ehrlich gegen übertreibend“ und der Wettbewerb.

Ein Prognostiker glaubt, ein Ereignis trete mit Wahrscheinlichkeit p ein, und meldet q; die Regel bewertet die Meldung gegen den Ausgang y in {0, 1}, größer ist besser. Ehrlich ist q = p.
Brier, Trefferquote und Absolutfehler sind für Brüche exakt (`exact_expected`); Log und sphärisch nur in Gleitkommazahlen (Gleichstände mit Toleranz).
Meldungen und Glauben auf dem Raster der ganzen Prozent 1 bis 99."""

from fractions import Fraction

import numpy as np

import zeh_constants as C
import zeh_scenario as S

TOL = 1e-12


def score(rule: str, q, y):
    """Punktzahl der Meldung q bei Ausgang y (Arrays oder Skalare); größer ist besser."""
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    if rule == "Brier":
        return -(q - y) ** 2
    if rule == "Log":
        return y * np.log(q) + (1 - y) * np.log(1 - q)
    if rule == "sphärisch":
        return np.where(y == 1, q, 1 - q) / np.sqrt(q ** 2 + (1 - q) ** 2)
    if rule == "Trefferquote":
        side = np.where(q > 0.5, 1.0, np.where(q < 0.5, 0.0, 0.5))
        return np.where(y == 1, side, 1.0 - side)
    if rule == "Absolutfehler":
        return -np.abs(q - y)
    raise ValueError(rule)


def expected(rule: str, p, q):
    """Erwartete Punktzahl der Meldung q, wenn das Ereignis mit Wahrscheinlichkeit p eintritt."""
    return p * score(rule, q, 1.0) + (1 - p) * score(rule, q, 0.0)


def exact_expected(rule: str, p: Fraction, q: Fraction) -> Fraction:
    """Wie `expected`, aber exakt (nur Brier, Trefferquote, Absolutfehler)."""
    if rule == "Brier":
        return -(p * (q - 1) ** 2 + (1 - p) * q ** 2)
    if rule == "Absolutfehler":
        return -(p * (1 - q) + (1 - p) * q)
    if rule == "Trefferquote":
        side = Fraction(1) if q > Fraction(1, 2) else Fraction(0) if q < Fraction(1, 2) else Fraction(1, 2)
        return p * side + (1 - p) * (1 - side)
    raise ValueError(f"{rule} hat keine exakte Rechnung")


def exact_expected_mix(p: Fraction, q: Fraction, mix: Fraction) -> Fraction:
    """Mischung mix * Brier + (1 - mix) * Absolutfehler."""
    return mix * exact_expected("Brier", p, q) + (1 - mix) * exact_expected("Absolutfehler", p, q)


def expected_curve(rule: str, p_pct: int) -> np.ndarray:
    """Erwartete Punktzahl über alle Meldungen 1 bis 99 % bei dem Glauben p."""
    q = np.array(C.GRID) / 100.0
    return expected(rule, p_pct / 100.0, q)


def best_set(rule: str, p_pct: int) -> list[int]:
    """Alle Meldungen in ganzen Prozent, die die erwartete Punktzahl maximieren (exakt für Brier, Trefferquote, Absolutfehler, sonst mit Toleranz 1e-12)."""
    p = Fraction(p_pct, 100)
    if rule in ("Brier", "Trefferquote", "Absolutfehler"):
        vals = [exact_expected(rule, p, Fraction(g, 100)) for g in C.GRID]
        top = max(vals)
        return [g for g, v in zip(C.GRID, vals) if v == top]
    vals = expected_curve(rule, p_pct)
    top = vals.max()
    return [g for g, v in zip(C.GRID, vals) if v >= top - TOL]


def best_set_mix(p_pct: int, mix_pct: int) -> list[int]:
    """Wie `best_set` für die Mischung mix * Brier + (1 - mix) * Absolutfehler (exakt)."""
    p, mix = Fraction(p_pct, 100), Fraction(mix_pct, 100)
    vals = [exact_expected_mix(p, Fraction(g, 100), mix) for g in C.GRID]
    top = max(vals)
    return [g for g, v in zip(C.GRID, vals) if v == top]


def mix_optimum(p_pct: int, mix_pct: int) -> Fraction | None:
    """Nullstelle der Ableitung der erwarteten Mischung: q* = p + (1 - mix)(2p - 1) / (2 mix), auf [0,01; 0,99] begrenzt; bei mix = 0 gibt es keine."""
    if mix_pct == 0:
        return None
    p, mix = Fraction(p_pct, 100), Fraction(mix_pct, 100)
    q = p + (1 - mix) * (2 * p - 1) / (2 * mix)
    return min(max(q, Fraction(1, 100)), Fraction(99, 100))


def loss(rule: str, p_pct: int, q_pct: int) -> float:
    """Erwarteter Verlust gegenüber der ehrlichen Meldung (nie negativ bei strikt eigentlichen Regeln)."""
    p = p_pct / 100.0
    return float(expected(rule, p, p) - expected(rule, p, q_pct / 100.0))


def n95(rule: str, p_pct: int, q_pct: int) -> float | None:
    """Zahl der Ereignisse (gleiches p), nach der die ehrliche Meldung im Mittel 1,645 Standardabweichungen vor der Meldung q liegt (Normalnäherung); None, wenn q nicht schlechter ist."""
    p, q = p_pct / 100.0, q_pct / 100.0
    d1 = float(score(rule, p, 1.0) - score(rule, q, 1.0))
    d0 = float(score(rule, p, 0.0) - score(rule, q, 0.0))
    de = p * d1 + (1 - p) * d0
    if de <= TOL:
        return None
    var = p * (1 - p) * (d1 - d0) ** 2
    return 1.645 ** 2 * var / de ** 2


def exaggerate(p: np.ndarray, d: float) -> np.ndarray:
    """Meldung um d (Wahrscheinlichkeit) vom Wert 1/2 weg (d > 0) oder auf ihn zu (d < 0), in [0,01; 0,99]."""
    sign = np.where(p >= 0.5, 1.0, -1.0)
    return np.clip(p + sign * d, C.Q_LOW, C.Q_HIGH)


def finite(n: int, d_pct: int, trials: int, seed: int) -> dict:
    """Ehrlich gegen übertreibend (+d) und untertreibend (-d) auf n Ereignissen: je Regel Anteil der Versuche, in denen die ehrliche Meldung strikt mehr / gleich viele / weniger Punkte hat."""
    p, y = S.draw_events(trials, n, seed)
    out = {}
    for rule in C.RULES:
        honest = score(rule, p, y).sum(axis=1)
        cells = {}
        for label, d in (("over", d_pct / 100.0), ("under", -d_pct / 100.0)):
            other = score(rule, exaggerate(p, d), y).sum(axis=1)
            diff = honest - other
            cells[label] = {"win": float((diff > TOL).mean() * 100), "tie": float((np.abs(diff) <= TOL).mean() * 100), "lose": float((diff < -TOL).mean() * 100)}
        out[rule] = cells
    return out


def logit(x):
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def contest(rule: str, m: int, lams: list[float], trials: int, seed: int, n: int = C.N_CONTEST) -> dict:
    """Wettbewerb: m Prognostiker mit eigenem Glauben auf n Ereignissen, Sieger = höchste Gesamtpunktzahl (Gleichstand geteilt).
    Prognostiker 0 meldet sigmoid(lambda * Log-Odds seines Glaubens), die anderen ehrlich. Rückgabe je lambda: Gewinnchance (%) und mittlere Punktzahl je Ereignis des Prognostikers 0; gleiche Zufallszahlen für alle lambda."""
    wins = {lam: 0.0 for lam in lams}
    means = {lam: 0.0 for lam in lams}
    done = 0
    block = 0
    while done < trials:
        t = min(C.CHUNK, trials - done)
        belief, y = S.draw_contest(t, m, n, seed + 1000 * block)
        rep = np.clip(belief, C.Q_LOW, C.Q_HIGH)
        others = score(rule, rep[:, 1:, :], y[:, None, :]).sum(axis=2)
        best_other = others.max(axis=1) if m > 1 else np.full(t, -np.inf)
        for lam in lams:
            r0 = np.clip(sigmoid(lam * logit(belief[:, 0, :])), C.Q_LOW, C.Q_HIGH)
            tot0 = score(rule, r0, y).sum(axis=1)
            top = np.maximum(tot0, best_other)
            n_top_others = (others >= top[:, None] - TOL).sum(axis=1)
            is0 = tot0 >= top - TOL
            share = np.where(is0, 1.0 / (1 + n_top_others), 0.0)
            wins[lam] += float(share.sum())
            means[lam] += float(tot0.sum() / n)
        done += t
        block += 1
    return {lam: {"win": wins[lam] / trials * 100, "mean": means[lam] / trials} for lam in lams}
