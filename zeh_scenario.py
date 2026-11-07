"""Szenario: ein Prognostiker glaubt, ein Ereignis tritt mit Wahrscheinlichkeit p ein, meldet q; der Ausgang y wird mit p gezogen. Alles simuliert.

Zufallsversuche: p je Ereignis gleichverteilt 0,05 bis 0,95, der Ausgang mit p gezogen. Alle Zahlen eines Versuchs kommen aus einem Strom (`default_rng(seed)`), ein Versuch ist eine Zeile."""

import numpy as np

import zeh_constants as C


def draw_events(trials: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """p (trials, n) und der Ausgang y (trials, n) in {0, 1}."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(C.P_EVENT_LOW, C.P_EVENT_HIGH, (trials, n))
    y = (rng.uniform(size=(trials, n)) < p).astype(float)
    return p, y


def draw_contest(trials: int, m: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Der Glaube der m Prognostiker (trials, m, n): wahre Wahrscheinlichkeit je Ereignis (gleichverteilt 0,1 bis 0,9) plus Rauschen auf den Log-Odds; und der Ausgang (trials, n)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, (trials, n))
    y = (rng.uniform(size=(trials, n)) < p).astype(float)
    logit = np.log(p / (1 - p))[:, None, :] + rng.normal(0.0, C.BELIEF_NOISE, (trials, m, n))
    return 1.0 / (1.0 + np.exp(-logit)), y
