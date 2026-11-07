"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets (Presets stehen unten, nachdem ihre Zahlen gemessen sind)."""

RULES = ("Brier", "Log", "sphärisch", "Trefferquote", "Absolutfehler")
PROPER = (0, 1, 2)                                      # strikt eigentliche Regeln: Brier, Log, sphärisch
GRID = tuple(range(1, 100))                             # Meldungen in ganzen Prozent, 0 und 100 sind nicht erlaubt (Log)
Q_LOW, Q_HIGH = 0.01, 0.99                              # Grenzen der Meldung bei den Zufallsversuchen

P_MIN, P_MAX, P_STEP, DEFAULT_P = 1, 99, 1, 30          # Glaube des Prognostikers in Prozent
DEFAULT_RULE = 0
D_MIN, D_MAX, D_STEP, DEFAULT_D = 0, 30, 1, 10          # Verschiebung der Meldung in Prozentpunkten
N_MIN, N_MAX, N_STEP, DEFAULT_N = 10, 500, 10, 20       # Zahl der Ereignisse
M_MIN, M_MAX, DEFAULT_M = 2, 20, 5                      # Zahl der Prognostiker im Wettbewerb
LAM_MIN, LAM_MAX, LAM_STEP, DEFAULT_LAM = 5, 30, 5, 15  # Skalierung der Log-Odds in Zehnteln (15 = 1,5)
MIX_MIN, MIX_MAX, MIX_STEP, DEFAULT_MIX = 0, 100, 10, 50    # Brier-Anteil der Mischung in Prozent
SEED_MAX, DEFAULT_SEED = 999999, 100000

P_EVENT_LOW, P_EVENT_HIGH = 0.05, 0.95                  # p je Ereignis bei den Zufallsversuchen gleichverteilt
T_FINITE = 2000                                         # Versuche für „ehrlich gegen übertreibend“
T_CONTEST = 4000                                        # Versuche für den Wettbewerb (live)
N_CONTEST = 20                                          # Ereignisse im Wettbewerb
BELIEF_NOISE = 0.5                                      # Standardabweichung des Rauschens auf den Log-Odds des Glaubens
CHUNK = 4000                                            # Versuche je Block (Speicher)

SWEEP_NS = (10, 20, 50, 100, 500)
SWEEP_D = 10
SWEEP_MS = (2, 5, 10, 20)
SWEEP_LAMS = (5, 10, 15, 20, 30)                        # Zehntel
SWEEP_CONTEST_RULES = (0, 1, 3)                         # Brier, Log, Trefferquote
SWEEP_T = 20000
HOLDOUT_SEED = 200000

COLOR_BAR = "#4c78a8"
COLOR_BAD = "#c0504d"
COLOR_OK = "#2F6B65"
COLOR_TRUE = "#d8a13a"
COLOR_FIT = "#7b5ea7"

PRESETS = {
    "Brier: Ehrlichkeit lohnt sich": {"p": 30, "rule": 0, "d": 10, "n": 20, "m": 5, "lam": 15, "mix": 50, "seed": 100000},
    "Absolutfehler belohnt die Lüge": {"p": 30, "rule": 4, "d": 10, "n": 50, "m": 5, "lam": 15, "mix": 50, "seed": 100000},
    "Trefferquote ist gleichgültig": {"p": 30, "rule": 3, "d": 10, "n": 20, "m": 5, "lam": 15, "mix": 50, "seed": 100000},
    "Log-Verlust bei 50 %": {"p": 50, "rule": 1, "d": 10, "n": 20, "m": 5, "lam": 15, "mix": 50, "seed": 100000},
    "Zehn Ereignisse verraten kaum etwas": {"p": 30, "rule": 0, "d": 10, "n": 10, "m": 5, "lam": 15, "mix": 50, "seed": 100000},
    "Der Wettbewerb verleitet": {"p": 30, "rule": 1, "d": 10, "n": 20, "m": 20, "lam": 15, "mix": 50, "seed": 100000},
    "Ein bisschen uneigentlich, ein bisschen verzerrt": {"p": 80, "rule": 0, "d": 10, "n": 20, "m": 5, "lam": 15, "mix": 90, "seed": 100000},
    "Kleinster Wettbewerb (2 Prognostiker)": {"p": 30, "rule": 0, "d": 10, "n": 10, "m": 2, "lam": 15, "mix": 50, "seed": 100000},
}
PRESET_HELP = {
    "Brier: Ehrlichkeit lohnt sich": "Brier, Glaube 30 %: die ehrliche Meldung ist die einzige beste. Eine Meldung von 40 % kostet 0,01 je Ereignis (genau 1/100), erkennbar nach etwa 227 Ereignissen; nach 20 Ereignissen liegt die Ehrlichkeit vor dem Übertreiben in 68,8 % der Versuche vorn.",
    "Absolutfehler belohnt die Lüge": "Absolutfehler, Glaube 30 %: die beste Meldung ist 1 % und bringt erwartet 0,116 Punkte mehr als die ehrliche; nach 50 Ereignissen liegt die Ehrlichkeit vor dem Übertreiben nur in 0,1 % der Versuche vorn.",
    "Trefferquote ist gleichgültig": "Trefferquote, Glaube 30 %: die ehrliche Meldung ist eine von 49 gleich guten (1 % bis 49 %): Lügen auf derselben Seite von 50 % kosten nichts, die Meldung 60 % auf der anderen Seite kostet 0,4 Punkte je Ereignis (erwartet 0,3 statt 0,7). Das Übertreiben um 10 Prozentpunkte bleibt auf derselben Seite und ist in 100 % der Versuche nicht zu erkennen (gleich viele Punkte).",
    "Log-Verlust bei 50 %": "Log-Regel, Glaube 50 %: die Meldung 60 % kostet 0,0204 je Ereignis, erkennbar nach etwa 267 Ereignissen; ehrlich liegt nach 20 Ereignissen in 70,4 % der Versuche vor dem Übertreiben.",
    "Zehn Ereignisse verraten kaum etwas": "Zehn Ereignisse, Brier: die ehrliche Meldung liegt vor dem Übertreiben um 10 Prozentpunkte nur in 60,5 % der Versuche vorn, in 39,5 % verliert sie.",
    "Der Wettbewerb verleitet": "20 Prognostiker, Log-Regel: der Prognostiker mit Skalierung 1,5 gewinnt in 7,6 % der Versuche, ehrlich in 5,0 %; seine mittlere Punktzahl je Ereignis sinkt von −0,6026 auf −0,6421.",
    "Ein bisschen uneigentlich, ein bisschen verzerrt": "Glaube 80 %, 90 % Brier und 10 % Absolutfehler: die beste Meldung ist 83 %, drei Prozentpunkte über dem Glauben; ehrlich ist sie nur bei 100 % Brier.",
    "Kleinster Wettbewerb (2 Prognostiker)": "Zwei Prognostiker, zehn Ereignisse, Brier: mit Skalierung 1,5 gewinnt der Übertreiber in 40,5 % der Versuche, ehrlich in 50,6 %; bei zwei Spielern lohnt Übertreiben nicht.",
}
