"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import zeh_constants as C

RULE_COLORS = {"Brier": "#4c78a8", "Log": "#2F6B65", "sphärisch": "#7b5ea7", "Trefferquote": "#d8a13a", "Absolutfehler": "#c0504d"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_expected(info: dict):
    """Erwartete Punktzahl über die Meldung q: die ehrliche Meldung (goldene Linie) und die beste(n) Meldung(en) (Punkte)."""
    q = list(C.GRID)
    fig = go.Figure(go.Scatter(x=q, y=info["curve"], mode="lines", line=dict(color=RULE_COLORS[info["rule"]], width=2.5), name="erwartete Punktzahl",
                               hovertemplate="Meldung %{x} %: erwartet %{y:.4f}<extra></extra>"))
    best = info["best"]
    fig.add_trace(go.Scatter(x=best, y=[info["curve"][b - 1] for b in best], mode="markers", marker=dict(size=7 if len(best) > 5 else 11, color=C.COLOR_BAD, symbol="diamond"),
                             name="beste Meldung(en)", hovertemplate="beste Meldung %{x} %<extra></extra>"))
    fig.add_vline(x=info["p"], line_color=C.COLOR_TRUE, line_width=2, annotation_text=f"ehrlich: {info['p']} %", annotation_position="top")
    fig.update_xaxes(title_text="Meldung q (%)", range=[0, 100])
    fig.update_yaxes(title_text="Erwartete Punktzahl (größer ist besser)")
    return _base(fig, height=340)


def build_incentive(inc: dict):
    """Verlust der verschobenen Meldung gegen die ehrliche je Regel (negativ: die Verschiebung gewinnt)."""
    rows = inc["rows"]
    names = list(rows)
    fig = go.Figure(go.Bar(x=names, y=[rows[n]["loss"] for n in names], marker_color=[C.COLOR_OK if rows[n]["loss"] > 1e-12 else C.COLOR_BAD for n in names],
                           hovertemplate="%{x}: Verlust %{y:.4f}<extra></extra>"))
    fig.update_xaxes(type="category", title_text="Regel")
    fig.update_yaxes(title_text="Verlust der verschobenen Meldung", zeroline=True)
    return _base(fig, height=300)


def build_finite(info: dict):
    """Ehrlich gegen übertreibend und untertreibend: Anteil der Versuche, in denen die Ehrlichkeit strikt vorn liegt, gleich ist oder verliert."""
    labels, win, tie, lose = [], [], [], []
    for rule, cells in info["rules"].items():
        for key, text in (("over", "übertreibt"), ("under", "untertreibt")):
            labels.append(f"{rule}, {text}")
            win.append(cells[key]["win"])
            tie.append(cells[key]["tie"])
            lose.append(cells[key]["lose"])
    fig = go.Figure([go.Bar(y=labels, x=win, orientation="h", name="ehrlich vorn", marker_color=C.COLOR_OK, hovertemplate="%{y}: ehrlich vorn in %{x:.1f} %<extra></extra>"),
                     go.Bar(y=labels, x=tie, orientation="h", name="gleich viele Punkte", marker_color="#9aa0a6", hovertemplate="%{y}: gleich in %{x:.1f} %<extra></extra>"),
                     go.Bar(y=labels, x=lose, orientation="h", name="ehrlich verliert", marker_color=C.COLOR_BAD, hovertemplate="%{y}: ehrlich verliert in %{x:.1f} %<extra></extra>")])
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Anteil der Versuche (%)", range=[0, 100])
    fig.update_yaxes(autorange="reversed")
    return _base(fig, height=470)


def build_contest(info: dict):
    """Gewinnchance des Abweichlers und des ehrlichen Prognostikers gegen den Faustwert 1/m."""
    fig = go.Figure(go.Bar(x=["ehrlich (Skalierung 1,0)", f"Skalierung {info['lam']:.1f}".replace(".", ",")], y=[info["win_honest"], info["win"]], marker_color=[C.COLOR_OK, C.COLOR_FIT],
                           hovertemplate="%{x}: gewinnt in %{y:.1f} % der Versuche<extra></extra>"))
    fig.add_hline(y=info["base"], line_dash="dot", line_color="rgba(150,150,150,0.9)", annotation_text=f"1/m = {info['base']:.1f} %".replace(".", ","), annotation_position="top left")
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="Gewinnchance (%)", rangemode="tozero")
    return _base(fig, height=300)


def build_mix(p: int, curve: tuple, mix: int):
    """Verschiebung der besten Meldung je Brier-Anteil bei dem Glauben p."""
    xs = list(range(0, 101, 10))
    fig = go.Figure(go.Scatter(x=xs, y=list(curve), mode="lines+markers", line=dict(color=C.COLOR_BAD), name="Verschiebung der besten Meldung",
                               hovertemplate="Brier-Anteil %{x} %: beste Meldung %{y:+d} Prozentpunkte vom Glauben<extra></extra>"))
    fig.add_vline(x=mix, line_color=C.COLOR_TRUE, line_width=2, annotation_text=f"gewählt: {mix} %", annotation_position="top")
    fig.update_xaxes(title_text="Brier-Anteil der Regel (%); der Rest ist Absolutfehler", range=[-2, 102])
    fig.update_yaxes(title_text="Beste Meldung − Glaube (Prozentpunkte)", zeroline=True)
    return _base(fig, height=300)


def build_mix_all(res: dict):
    """Mittlere und größte Verschiebung der besten Meldung über alle Glauben 1 bis 99 % je Brier-Anteil."""
    xs = list(res)
    fig = go.Figure([go.Bar(x=[f"{x} %" for x in xs], y=[res[x]["mean_shift"] for x in xs], name="Mittel", marker_color=C.COLOR_BAR, hovertemplate="Brier-Anteil %{x}: Mittel %{y:.1f} Prozentpunkte<extra></extra>"),
                     go.Scatter(x=[f"{x} %" for x in xs], y=[res[x]["max_shift"] for x in xs], mode="lines+markers", name="Maximum", line=dict(color=C.COLOR_BAD), hovertemplate="Brier-Anteil %{x}: Maximum %{y} Prozentpunkte<extra></extra>")])
    fig.update_xaxes(type="category", title_text="Brier-Anteil der Regel")
    fig.update_yaxes(title_text="Abstand der besten Meldung vom Glauben (Prozentpunkte)", rangemode="tozero")
    return _base(fig, height=320)


def build_finite_sweep(res: dict):
    """Anteil der Versuche, in denen die Ehrlichkeit strikt vor dem Übertreiben liegt, je Zahl der Ereignisse und Regel."""
    ns = list(res)
    fig = go.Figure([go.Scatter(x=[str(n) for n in ns], y=[res[n][rule]["over"]["win"] for n in ns], mode="lines+markers", name=rule, line=dict(color=RULE_COLORS[rule]),
                                hovertemplate=f"{rule}, %{{x}} Ereignisse: ehrlich vorn in %{{y:.1f}} %<extra></extra>") for rule in C.RULES])
    fig.update_xaxes(type="category", title_text="Zahl der Ereignisse")
    fig.update_yaxes(title_text="Ehrlich strikt vor dem Übertreiben (% der Versuche)", range=[0, 100])
    return _base(fig, height=340)


def build_contest_sweep(per_m: dict, rule: str):
    """Gewinnchance des Abweichlers geteilt durch den Faustwert 1/m: über 1 heißt, die Abweichung lohnt."""
    ms = sorted(per_m)
    lams = sorted(per_m[ms[0]])
    bars = []
    for lam in lams:
        label = f"Skalierung {lam:.1f}".replace(".", ",")
        bars.append(go.Bar(x=[f"m = {m}" for m in ms], y=[per_m[m][lam]["win"] / (100.0 / m) for m in ms], name=label, hovertemplate=label + ", %{x}: %{y:.2f}-fache Faustchance<extra></extra>"))
    fig = go.Figure(bars)
    fig.update_layout(barmode="group")
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(150,150,150,0.9)", annotation_text="Faustchance 1/m", annotation_position="top left")
    fig.update_xaxes(type="category", title_text="Zahl der Prognostiker")
    fig.update_yaxes(title_text=f"Gewinnchance geteilt durch 1/m ({rule})", rangemode="tozero")
    return _base(fig, height=340)
