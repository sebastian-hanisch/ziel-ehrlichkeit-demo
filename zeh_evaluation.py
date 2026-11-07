"""Experimente, Zählungen über feste Zufallsversuche und Verdikte (reine Funktionen: Art, Text; jede Zahl im Text kommt aus der Rechnung).

Gefragt wird: Welches Ziel (Scoring-Regel) belohnt die wahre Angabe? Ein Prognostiker glaubt p, meldet q und wird nach einer Regel bewertet; ehrlich ist q = p.
Fünf Regeln, fünf Experimente: die beste Meldung, die Stärke des Anreizes, ehrlich gegen übertreibend auf endlich vielen Ereignissen, der Wettbewerb, die Mischung aus Brier und Absolutfehler."""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np

import zeh_algorithm as A
import zeh_constants as C


def de(x: float, digits: int = 1) -> str:
    text = f"{x:.{digits}f}"
    if float(text) == 0.0:
        text = f"{0.0:.{digits}f}"
    return text.replace(".", ",")


def pct(x: float, digits: int = 1) -> str:
    return de(x * 100.0, digits) + " %"


@dataclass(frozen=True)
class Settings:
    p: int = C.DEFAULT_P
    rule: int = C.DEFAULT_RULE
    d: int = C.DEFAULT_D
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    lam: int = C.DEFAULT_LAM
    mix: int = C.DEFAULT_MIX
    seed: int = C.DEFAULT_SEED

    @property
    def rule_name(self) -> str:
        return C.RULES[self.rule]

    @property
    def q(self) -> int:
        """Die um d nach oben verschobene Meldung, auf 1 bis 99 begrenzt."""
        return min(max(self.p + self.d, 1), 99)


# --- 1. Die beste Meldung ------------------------------------------------------------------------------------------------------------------

def best_info(s: Settings) -> dict:
    name = s.rule_name
    curve = A.expected_curve(name, s.p)
    best = A.best_set(name, s.p)
    truth = float(curve[s.p - 1])
    top = float(curve.max())
    return {"rule": name, "p": s.p, "curve": [float(v) for v in curve], "best": best, "honest_best": s.p in best, "unique": best == [s.p], "gain": top - truth, "truth": truth, "top": top}


# --- 2. Die Stärke des Anreizes ------------------------------------------------------------------------------------------------------------

def incentive_info(s: Settings) -> dict:
    """Verlust einer Verschiebung der Meldung von p auf q = p + d je Regel; Ereignisse bis zur Erkennung (Normalnäherung)."""
    q = s.q
    rows = {}
    for name in C.RULES:
        rows[name] = {"loss": A.loss(name, s.p, q), "n95": A.n95(name, s.p, q)}
    brier_exact = (Fraction(q, 100) - Fraction(s.p, 100)) ** 2
    return {"p": s.p, "q": q, "rows": rows, "brier_exact": brier_exact}


# --- 3. Ehrlich gegen übertreibend auf endlich vielen Ereignissen -----------------------------------------------------------------------

@lru_cache(maxsize=64)
def _finite_cached(n: int, d: int, seed: int) -> dict:
    return A.finite(n, d, C.T_FINITE, seed)


def finite_info(s: Settings) -> dict:
    return {"n": s.n, "d": s.d, "trials": C.T_FINITE, "rules": _finite_cached(s.n, s.d, s.seed)}


# --- 4. Der Wettbewerb ----------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _contest_cached(rule: str, m: int, lam10: int, seed: int) -> dict:
    return A.contest(rule, m, [lam10 / 10.0, 1.0], C.T_CONTEST, seed)


def contest_info(s: Settings) -> dict:
    res = _contest_cached(s.rule_name, s.m, s.lam, s.seed)
    lam = s.lam / 10.0
    dev, hon = res[lam], res[1.0]
    return {"rule": s.rule_name, "m": s.m, "lam": lam, "trials": C.T_CONTEST, "win": dev["win"], "win_honest": hon["win"], "mean": dev["mean"], "mean_honest": hon["mean"], "base": 100.0 / s.m}


# --- 5. Die Mischung aus Brier und Absolutfehler ------------------------------------------------------------------------------------------

def mix_info(s: Settings) -> dict:
    best = A.best_set_mix(s.p, s.mix)
    qs = A.mix_optimum(s.p, s.mix)
    return {"p": s.p, "mix": s.mix, "best": best, "honest_best": s.p in best, "unique": best == [s.p], "shift": best[0] - s.p, "q_star": None if qs is None else float(qs) * 100}


@lru_cache(maxsize=None)
def mix_curve(p: int) -> tuple:
    """Verschiebung der besten Meldung je Brier-Anteil 0, 10, ..., 100 % bei dem Glauben p (Prozentpunkte, mit Vorzeichen)."""
    return tuple(A.best_set_mix(p, mx)[0] - p for mx in range(0, 101, 10))


@lru_cache(maxsize=None)
def mix_over_all_p() -> dict:
    """Je Brier-Anteil 0, 10, ..., 100 %: ehrlich die einzige beste in wie vielen der 99 Glauben, mittlere und größte Verschiebung (Prozentpunkte)."""
    out = {}
    for mx in range(0, 101, 10):
        uniq, shifts = 0, []
        for p in range(1, 100):
            best = A.best_set_mix(p, mx)
            uniq += best == [p]
            shifts.append(abs(best[0] - p))
        out[mx] = {"unique": uniq, "mean_shift": float(np.mean(shifts)), "max_shift": max(shifts)}
    return out


# --- Zählungen ----------------------------------------------------------------------------------------------------------------------------------

def sweep_finite(seed: int = C.DEFAULT_SEED) -> dict:
    """Für n = 10, 20, 50, 100, 500 und die Verschiebung 10 Prozentpunkte: ehrlich gegen übertreibend und untertreibend, 2 000 Versuche je Zeile."""
    return {n: A.finite(n, C.SWEEP_D, C.T_FINITE, seed + i) for i, n in enumerate(C.SWEEP_NS)}


def sweep_contest(seed: int = C.DEFAULT_SEED) -> dict:
    """Wettbewerb über Regeln, Spielerzahl und Skalierung: Gewinnchance und mittlere Punktzahl des Abweichlers (20 000 Versuche, gleiche Zufallszahlen für alle Skalierungen)."""
    lams = [x / 10.0 for x in C.SWEEP_LAMS]
    return {C.RULES[r]: {m: A.contest(C.RULES[r], m, lams, C.SWEEP_T, seed + 17 * m) for m in C.SWEEP_MS} for r in C.SWEEP_CONTEST_RULES}


def summary_contest(res: dict) -> dict:
    return res


# --- Verdikte -----------------------------------------------------------------------------------------------------------------------------------

def verdict_best(info: dict) -> tuple[str, str]:
    name, p, best = info["rule"], info["p"], info["best"]
    if info["unique"]:
        return "success", f"Bei p = {p} % ist die ehrliche Meldung bei der Regel {name} die **einzige beste**: jede Lüge kostet erwartete Punkte."
    if info["honest_best"]:
        if len(best) == len(C.GRID):
            return "warning", (f"Bei p = {p} % ist die ehrliche Meldung bei der Regel {name} **eine von {len(best)} gleich guten** Meldungen ({best[0]} % bis {best[-1]} %): Lügen kostet nichts, "
                               f"Ehrlichkeit wird hier nicht belohnt.")
        return "warning", (f"Bei p = {p} % ist die ehrliche Meldung bei der Regel {name} **eine von {len(best)} gleich guten** Meldungen ({best[0]} % bis {best[-1]} %): Lügen innerhalb dieser Meldungen kostet nichts, "
                           f"eine Meldung außerhalb kostet erwartete Punkte; Ehrlichkeit wird hier nicht belohnt.")
    return "error", (f"Bei p = {p} % ist die ehrliche Meldung bei der Regel {name} **nicht die beste**: die beste Meldung ist {best[0]} % und bringt erwartet {de(info['gain'], 4)} Punkte mehr "
                     f"als die ehrliche ({de(info['top'], 4)} gegen {de(info['truth'], 4)}).")


def verdict_incentive(info: dict) -> tuple[str, str]:
    p, q, rows = info["p"], info["q"], info["rows"]
    if q == p:
        return "info", f"Verschiebung 0: die Meldung ist ehrlich, es gibt nichts zu verlieren und nichts zu erkennen (p = {p} %)."
    b = rows["Brier"]
    parts = []
    for name in ("Brier", "Log", "sphärisch"):
        n95 = rows[name]["n95"]
        parts.append(f"{name} {de(rows[name]['loss'], 4)}" + (f" (erkennbar nach {de(n95, 0)} Ereignissen)" if n95 is not None else ""))
    abs_loss = rows["Absolutfehler"]["loss"]
    tail = ""
    if abs_loss < -A.TOL:
        tail = f" Beim Absolutfehler **gewinnt** die Meldung {q} % erwartet {de(-abs_loss, 4)} Punkte."
    elif abs(abs_loss) <= A.TOL:
        tail = " Beim Absolutfehler und bei der Trefferquote ist die Meldung gleich gut."
    return "info", (f"Die Meldung {q} % statt {p} % kostet erwartet pro Ereignis: {', '.join(parts)}. Beim Brier-Maß ist der Verlust genau (q − p)² = {info['brier_exact']} = {de(float(info['brier_exact']), 4)}." + tail)


def verdict_finite(info: dict, rule: str) -> tuple[str, str]:
    r = info["rules"][rule]
    o, u = r["over"], r["under"]
    n, d = info["n"], info["d"]
    kind = "warning" if o["win"] < 90 or u["win"] < 90 else "success"
    ties = ""
    if o["tie"] > 0 or u["tie"] > 0:
        ties = f" (gleich viele Punkte: {de(o['tie'])} % beziehungsweise {de(u['tie'])} %)"
    return kind, (f"Bei der Regel {rule} liegt die ehrliche Meldung nach {n} Ereignissen in **{de(o['win'])} %** der {info['trials']} Versuche strikt vor einer um {d} Prozentpunkte übertriebenen und in "
                  f"**{de(u['win'])} %** vor einer untertriebenen{ties}; in {de(o['lose'])} % beziehungsweise {de(u['lose'])} % verliert sie.")


def verdict_contest(info: dict) -> tuple[str, str]:
    lam, m, rule = info["lam"], info["m"], info["rule"]
    if lam == 1.0:
        return "info", (f"Skalierung 1,0 ist ehrlich: bei {m} Prognostikern gewinnt der Prognostiker in {de(info['win'])} % der {info['trials']} Versuche (Faustwert {de(info['base'])} %); "
                        f"wähle eine andere Skalierung, um die Abweichung zu sehen.")
    more = info["win"] > info["win_honest"]
    sign = "mehr" if more else "weniger"
    kind = "warning" if more else "info"
    return kind, (f"Bei {m} Prognostikern und der Regel {rule} gewinnt der Prognostiker mit der Skalierung {de(lam)} in **{de(info['win'])} %** der {info['trials']} Versuche, ehrlich in {de(info['win_honest'])} % "
                  f"({sign} als ehrlich); seine mittlere Punktzahl je Ereignis ist {de(info['mean'], 4)} gegen {de(info['mean_honest'], 4)} bei ehrlicher Meldung.")


def verdict_mix(info: dict) -> tuple[str, str]:
    mx, p = info["mix"], info["p"]
    if mx == 100:
        return "success", f"Mit 100 % Brier ist die ehrliche Meldung die einzige beste (p = {p} %)."
    if info["unique"]:
        return "info", f"Bei p = {p} % und {mx} % Brier-Anteil ist die ehrliche Meldung noch die einzige beste (bei p = 50 % hat die Verzerrung keinen Ansatzpunkt)."
    q = "-" if info["q_star"] is None else f"{de(info['q_star'], 1)} %"
    return "warning", (f"Schon mit {100 - mx} % Absolutfehler in der Regel ist die beste Meldung bei p = {p} % nicht mehr die ehrliche: die beste Meldung ist {info['best'][0]} % "
                       f"({'+' if info['shift'] > 0 else ''}{info['shift']} Prozentpunkte); die Formel q* = p + (1 − m)(2p − 1) / (2m) mit m = {mx} % gibt {q}.")


def verdict_finite_sweep(res: dict) -> tuple[str, str]:
    ns = list(res)
    cells = []
    for rule in ("Brier", "Log", "sphärisch"):
        cells.append(f"{rule}: " + " / ".join(de(res[n][rule]["over"]["win"]) for n in ns) + " %")
    abs_ = " / ".join(de(res[n]["Absolutfehler"]["over"]["win"]) for n in ns)
    hit = " / ".join(de(res[n]["Trefferquote"]["over"]["tie"], 0) for n in ns)
    return "info", (f"Ehrlich liegt strikt vor dem Übertreiben um {C.SWEEP_D} Prozentpunkte bei n = {' / '.join(str(n) for n in ns)} Ereignissen in (je {C.T_FINITE} Versuche): " + "; ".join(cells)
                    + f". Absolutfehler: {abs_} %. Trefferquote: Gleichstand in {hit} % der Versuche (das Übertreiben ist nie zu erkennen).")


def verdict_contest_sweep(res: dict, label: str) -> tuple[str, str]:
    out = []
    for rule, per_m in res.items():
        parts = []
        for m in (min(per_m), max(per_m)):
            lams = sorted(per_m[m])
            parts.append(f"m = {m} (Faustwert {de(100 / m)} %): " + " / ".join(de(per_m[m][lam]["win"]) for lam in lams) + " %")
        out.append(f"{rule}: " + "; ".join(parts))
    lams = sorted(next(iter(next(iter(res.values())).values())))
    return "info", f"{label}, Skalierung {' / '.join(de(l) for l in lams)}: " + " | ".join(out) + "."
