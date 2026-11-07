"""Ehrlichkeit belohnen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 7 der Reihe „Zielfunktionsdesign“ (gelb, das Beispiel wechselt von der Zuordnung zur Prognose): Welches Ziel belohnt die wahre Angabe?
Die Demo zeigt, bei welcher Regel die ehrliche Meldung die beste ist (Brier, Log, sphärisch gegen Trefferquote und Absolutfehler), wie schwach der Anreiz ist,
wie oft die Ehrlichkeit auf endlich vielen Ereignissen vorn liegt, warum ein Wettbewerb, in dem nur der Sieger zählt, sie bei vielen Prognostikern trotzdem verdirbt und was schon ein Anteil Absolutfehler anrichtet.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import zeh_constants as C
from zeh_evaluation import (
    Settings,
    best_info,
    contest_info,
    de,
    finite_info,
    incentive_info,
    mix_curve,
    mix_info,
    mix_over_all_p,
    sweep_contest,
    sweep_finite,
    verdict_best,
    verdict_contest,
    verdict_contest_sweep,
    verdict_finite,
    verdict_finite_sweep,
    verdict_incentive,
    verdict_mix,
)
from zeh_layout import dunkles_design, kopf_css
from zeh_pdf_export import build_pdf
from zeh_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from zeh_visualization import (
    build_contest,
    build_contest_sweep,
    build_expected,
    build_finite,
    build_finite_sweep,
    build_incentive,
    build_mix,
    build_mix_all,
)

st.set_page_config(page_title="Ehrlichkeit belohnen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _best(settings):
    return best_info(settings)


@st.cache_data(show_spinner="Die Zufallsversuche werden gerechnet ...")
def _finite(settings):
    return finite_info(settings)


@st.cache_data(show_spinner="Der Wettbewerb wird gerechnet ...")
def _contest(settings):
    return contest_info(settings)


@st.cache_data(show_spinner="Je 2 000 Versuche für fünf Größen werden gerechnet ...")
def _sweep_finite(seed):
    return sweep_finite(seed)


@st.cache_data(show_spinner="Je 20 000 Versuche für drei Regeln, vier Spielerzahlen und fünf Skalierungen werden gerechnet ...")
def _sweep_contest(seed):
    return sweep_contest(seed)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in zeh_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🤥 Ehrlichkeit belohnen")
st.markdown(
    """
**Welches Ziel belohnt die wahre Angabe?** Das Beispiel wechselt: kein Zuordnungsplan, sondern eine **Prognose**. Ein Prognostiker glaubt, ein Ereignis trete mit Wahrscheinlichkeit **p** ein, und **meldet** eine Wahrscheinlichkeit **q**;
eine **Regel** (das Ziel) vergibt Punkte gegen den Ausgang. **Ehrlich** ist q = p. Bei **Brier**, **Log** und **sphärisch** ist die ehrliche Meldung im Erwartungswert die beste, bei der **Trefferquote** zählt nur die Seite von 50 % (die ehrliche Meldung ist eine von vielen gleich guten; eine Lüge auf die andere Seite kostet Punkte, eine auf derselben Seite nichts),
beim **Absolutfehler** lohnt die Lüge (außer bei 1, 50 und 99 %). Aber auch das richtige Ziel hat Grenzen: der Anreiz ist schwach, auf wenigen Ereignissen liegt die Ehrlichkeit oft nicht vorn, und im **Wettbewerb** mit vielen Prognostikern, in dem nur der Sieger zählt, lohnt Übertreiben trotzdem (bei 2 und 5 Prognostikern nicht).

Weiter unten: **🔬 Die Stärke des Anreizes**, **🔬 Ehrlich gegen übertreibend**, **🔬 Der Wettbewerb**, **🔬 Ein bisschen uneigentlich**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Vergleich", expanded=True):
    st.markdown(
        """
1. **Meldung und Punkte:** der Prognostiker meldet q (ganze Prozent 1 bis 99), das Ereignis tritt ein (y = 1) oder nicht (y = 0). **Brier** −(q−y)², **Log** y·ln q + (1−y)·ln(1−q), **sphärisch** (Wahrscheinlichkeit des Ausgangs geteilt durch die Länge von (q, 1−q)), **Trefferquote** (1 Punkt, wenn q auf der Seite des Ausgangs von 50 % liegt), **Absolutfehler** −|q−y|.
2. **Die beste Meldung:** die Meldung mit der höchsten **erwarteten** Punktzahl bei dem Glauben p. Ist sie immer q = p, heißt die Regel **strikt eigentlich**.
3. **Der Anreiz:** wie viele Punkte kostet eine um d Prozentpunkte verschobene Meldung, und nach wie vielen Ereignissen erkennt man sie im Mittel (1,645 Standardabweichungen, Normalnäherung)?
4. **Zufallsversuche:** die Ereignisse haben p gleichverteilt 0,05 bis 0,95 (nicht der Regler p), der Ausgang wird mit p gezogen; verglichen wird die ehrliche Meldung mit einer um d Prozentpunkte **übertriebenen** (vom Wert 50 % weg) oder **untertriebenen** (auf 50 % zu, höchstens bis 50 %, nie darüber hinaus).
5. **Wettbewerb:** m Prognostiker mit eigenem verrauschten Glauben, Sieger ist die höchste Gesamtpunktzahl auf 20 Ereignissen; einer skaliert die Log-Odds seines Glaubens mit λ (1,0 ist ehrlich).
6. **Messungen, kein Satz:** Alle Zahlen sind Zählungen und exakte Rechnungen an diesem Modell; sie sagen nichts über echte Prognostiker.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Der Prognostiker**")
    p = st.slider("Glaube p in %", *bounds("p_slider"), key="p_slider", help="Die Wahrscheinlichkeit, die der Prognostiker dem Ereignis zutraut. Ehrlich ist, genau das zu melden.")
    rule = st.selectbox("Regel (das Ziel)", list(range(len(C.RULES))), key="rule_select", format_func=lambda i: C.RULES[i], help="Brier, Log und sphärisch sind strikt eigentlich; Trefferquote und Absolutfehler nicht.")
    d = st.slider("Verschiebung der Meldung in Prozentpunkten", *bounds("d_slider"), key="d_slider", help="Wie weit die Meldung vom Glauben abweicht: im Abschnitt Anreiz nach oben (p + d), bei den Zufallsversuchen vom Wert 50 % weg oder auf ihn zu.")
    st.markdown("**Viele Ereignisse**")
    n = st.slider("Zahl der Ereignisse", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Je Versuch: wie viele Ereignisse die Punkte aufsummiert werden.")
    st.markdown("**Der Wettbewerb**")
    m = st.slider("Zahl der Prognostiker m", *bounds("m_slider"), key="m_slider", help="Wie viele im Wettbewerb um den ersten Platz mitspielen; einer davon weicht ab.")
    lam = st.slider("Skalierung der Log-Odds λ (in Zehnteln)", *bounds("lam_slider"), key="lam_slider", step=C.LAM_STEP, help="10 ist ehrlich, 15 meint 1,5: der Abweichler meldet extremere Wahrscheinlichkeiten als er glaubt; 5 meint 0,5: vorsichtigere.")
    st.markdown("**Die Mischung**")
    mix = st.slider("Brier-Anteil der Regel in %", *bounds("mix_slider"), key="mix_slider", step=C.MIX_STEP, help="Die Regel ist Anteil · Brier + (100 % − Anteil) · Absolutfehler; 100 ist reiner Brier.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Alle Zufallsversuche (Ereignisse und Wettbewerb) kommen aus diesem Seed.")
    st.button("🎲 Neue Versuche generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"p_slider": int(p), "rule_select": int(rule), "d_slider": int(d), "n_slider": int(n), "m_slider": int(m), "lam_slider": int(lam), "mix_slider": int(mix), "seed_input": int(seed)})

settings = Settings(int(p), int(rule), int(d), int(n), int(m), int(lam), int(mix), int(seed))
info = _best(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: ist Ehrlichkeit die beste Meldung?")
c1, c2, c3 = st.columns(3)
c1.metric("Glaube p", f"{settings.p} %", delta=f"Regel {settings.rule_name}", delta_color="off")
c2.metric("Beste Meldung", f"{info['best'][0]} %" if len(info["best"]) == 1 else f"{info['best'][0]}–{info['best'][-1]} %", delta="ehrlich" if info["unique"] else ("gleich gut" if info["honest_best"] else "nicht ehrlich"), delta_color="off")
c3.metric("Gewinn durch Lügen", de(info["gain"], 4), delta="erwartete Punkte je Ereignis", delta_color="off")
st.plotly_chart(build_expected(info), width="stretch", key="expected_chart")
st.caption("Die erwartete Punktzahl jeder Meldung bei dem Glauben p: die goldene Linie ist die ehrliche Meldung, die roten Punkte sind die besten Meldungen (bei vielen gleich guten sind es viele Punkte).")
st.download_button("📄 Ergebnis als PDF", data=build_pdf(settings), file_name="ehrlichkeit-belohnen.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Belohnt dieses Ziel die wahre Angabe?")
kind, text = verdict_best(info)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die Stärke des Anreizes")
st.caption("Selbst ein strikt eigentliches Ziel setzt nur einen schwachen Anreiz: eine um d Prozentpunkte verschobene Meldung kostet beim Brier-Maß genau d² (bei 10 Prozentpunkten 0,01 Punkte je Ereignis). Rot: die Verschiebung gewinnt oder ist gleich gut.")
inc = incentive_info(settings)
st.plotly_chart(build_incentive(inc), width="stretch", key="incentive_chart")
kind, text = verdict_incentive(inc)
getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Ehrlich gegen übertreibend")
st.caption(f"Auf {settings.n} Ereignissen (je {C.T_FINITE} Versuche, p je Ereignis gleichverteilt 0,05 bis 0,95): wie oft hat die ehrliche Meldung strikt mehr Punkte als eine um {settings.d} Prozentpunkte übertriebene oder untertriebene?")
fin = _finite(settings)
st.plotly_chart(build_finite(fin), width="stretch", key="finite_chart")
kind, text = verdict_finite(fin, settings.rule_name)
getattr(st, kind)(text)
if st.toggle("Die Ehrlichkeit über feste Ereigniszahlen nachrechnen (10 bis 500, Verschiebung 10 Prozentpunkte)", value=False, key="sweep_finite"):
    res = _sweep_finite(settings.seed)
    kind, text = verdict_finite_sweep(res)
    getattr(st, kind)(text)
    st.plotly_chart(build_finite_sweep(res), width="stretch", key="finite_sweep_chart")

st.markdown("---")

st.subheader("🔬 Der Wettbewerb")
st.caption(f"Wer nicht nach Punkten, sondern nur als Sieger bezahlt wird, hat bei vielen Prognostikern auch beim richtigen Ziel einen Grund zu übertreiben (bei 2 und 5 Prognostikern nicht): {settings.m} Prognostiker mit verrauschtem Glauben, Sieger ist die höchste Gesamtpunktzahl auf {C.N_CONTEST} Ereignissen; "
           f"einer meldet extremere (λ über 1) oder vorsichtigere Wahrscheinlichkeiten (λ unter 1) als er glaubt. {C.T_CONTEST} Versuche.")
con = _contest(settings)
st.plotly_chart(build_contest(con), width="stretch", key="contest_chart")
kind, text = verdict_contest(con)
getattr(st, kind)(text)
if st.toggle("Den Wettbewerb über feste Versuche nachrechnen (Brier, Log, Trefferquote; 2 bis 20 Prognostiker)", value=False, key="sweep_contest"):
    res = _sweep_contest(settings.seed)
    kind, text = verdict_contest_sweep(res, f"Stichprobe (Seed {settings.seed}, je {C.SWEEP_T} Versuche), Gewinnchance des Abweichlers")
    getattr(st, kind)(text)
    for rule_name, per_m in res.items():
        st.caption(rule_name)
        st.plotly_chart(build_contest_sweep(per_m, rule_name), width="stretch", key=f"contest_sweep_{rule_name}")
if st.toggle("Den Wettbewerb an einer zweiten Menge fester Versuche gegenprüfen (Hold-out)", value=False, key="sweep_holdout"):
    res = _sweep_contest(C.HOLDOUT_SEED)
    kind, text = verdict_contest_sweep(res, f"Hold-out (Seed {C.HOLDOUT_SEED}, je {C.SWEEP_T} Versuche), Gewinnchance des Abweichlers")
    getattr(st, kind)(text)

st.markdown("---")

st.subheader("🔬 Ein bisschen uneigentlich")
st.caption("Was geschieht, wenn die Regel nur zum Teil Absolutfehler ist? Die Regel ist Anteil · Brier + (100 % − Anteil) · Absolutfehler; die beste Meldung liegt exakt bei q* = p + (1 − Anteil)(2p − 1) / (2 · Anteil), auf 1 bis 99 % begrenzt.")
mi = mix_info(settings)
st.plotly_chart(build_mix(settings.p, mix_curve(settings.p), settings.mix), width="stretch", key="mix_chart")
kind, text = verdict_mix(mi)
getattr(st, kind)(text)
allp = mix_over_all_p()
st.caption("Über alle Glauben von 1 bis 99 %: Abstand der besten Meldung vom Glauben je Brier-Anteil.")
st.plotly_chart(build_mix_all(allp), width="stretch", key="mix_all_chart")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Prognostiker maximiert die erwartete Punktzahl** | Wer Punkte nicht linear bewertet (Risikoscheu, Auszahlung nach Rang, Preise für den Ersten), hat auch bei einer eigentlichen Regel Gründe zu übertreiben; der Wettbewerb zeigt das | – |
| **Der Glaube ist ehrlich gebildet** | Die Regel belohnt die Übereinstimmung von Meldung und **Glaube**, nicht die Wahrheit: wer sich irrt, wird nicht bestraft, solange er ehrlich meldet | Stück 13 (Ziel testen) |
| **Binäres Ereignis, ganze Prozent** | Mehrklassen-, Intervall- und Quantilregeln sind nicht Teil; Meldungen 0 und 100 % sind ausgeschlossen (Log) | `forecast-interval-demo` |
| **Anreiz nur im Erwartungswert** | Auf endlich vielen Ereignissen liegt die Ehrlichkeit auch bei Brier nach 20 Ereignissen gegen eine um 10 Prozentpunkte übertriebene Meldung nur in etwa 69 % der Versuche vorn | – |
| **Ein Prognostiker, kein Markt** | Händler gegen einen Marktmacher (LMSR) sind ein anderes Modell; keine Anreizverträglichkeit mehrerer Spieler, kein Mechanism Design | `prognosemarkt-demo` |
| **Ersatzziele und Ausnutzen** | Dass ein Ziel Ehrlichkeit belohnt, heißt nicht, dass es nichts anderes belohnt; die Überoptimierung eines Ersatzziels ist ein eigenes Thema | Stück 8, 9 |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Setting.** Der Prognostiker glaubt $P(y = 1) = p$ und meldet $q \in (0, 1)$; die Regel $S(q, y)$ vergibt Punkte, größer ist besser. Die erwartete Punktzahl ist $\bar S(q; p) = p\, S(q, 1) + (1 - p)\, S(q, 0)$.
Die Regel heißt **strikt eigentlich**, wenn $\bar S(q; p)$ für jedes $p$ nur bei $q = p$ maximal ist.

**Regeln.** Brier: $S = -(q - y)^2$. Log: $S = y \ln q + (1 - y) \ln (1 - q)$. Sphärisch: $S = \dfrac{q^{y} (1 - q)^{1 - y}}{\sqrt{q^2 + (1 - q)^2}}$. Trefferquote: $S = 1$ für $q$ auf der Seite des Ausgangs von $\tfrac12$ (bei $q = \tfrac12$ ein halber Punkt). Absolutfehler: $S = -|q - y|$.

**Brier.** $\bar S(q; p) = -(q - p)^2 - p(1 - p)$; die ehrliche Meldung ist die einzige beste, der Verlust einer Verschiebung ist genau $(q - p)^2$. **Absolutfehler.** $\bar S(q; p) = -\bigl(p(1 - q) + (1 - p) q\bigr)$ ist linear in $q$ mit Steigung $2p - 1$: für $p > \tfrac12$ ist $q = 1$ die beste Meldung, für $p < \tfrac12$ ist es $q = 0$.

**Erkennung.** Mit $D_1 = S(p, 1) - S(q, 1)$ und $D_0 = S(p, 0) - S(q, 0)$ ist der erwartete Vorsprung der Ehrlichkeit je Ereignis $\Delta = p D_1 + (1 - p) D_0$ und die Varianz $p(1 - p)(D_1 - D_0)^2$;
$n_{95} = 1{,}645^2 \cdot p(1 - p)(D_1 - D_0)^2 / \Delta^2$ (Normalnäherung).

**Mischung.** Für $S_\mu = \mu\, S_{\text{Brier}} + (1 - \mu)\, S_{\text{abs}}$ ist $\bar S_\mu'(q) = -2\mu (q - p) - (1 - \mu)(1 - 2p) = 0$ bei $q^* = p + \dfrac{(1 - \mu)(2p - 1)}{2\mu}$ (auf $[0{,}01;\, 0{,}99]$ begrenzt): im stetigen Fall nur bei $\mu = 1$ oder $p = \tfrac12$ ist die ehrliche Meldung die beste (auf dem Prozentraster mit Begrenzung auch bei $p$ = 1 % und 99 % und, bei kleinem Anteil Absolutfehler, nahe 50 %).

**Wettbewerb.** Der Glaube von Prognostiker $i$ auf Ereignis $j$ ist $\sigma\bigl(\operatorname{logit}(p_j) + \varepsilon_{ij}\bigr)$ mit $\varepsilon_{ij} \sim \mathcal N(0, 0{,}5^2)$; der Abweichler meldet $\sigma(\lambda \cdot \operatorname{logit}(\text{Glaube}))$.

**Literatur.** Keine Aussage dieser Demo stützt sich auf gelesene Literatur; alle Zahlen sind exakte Rechnungen oder Messungen an dem Modell der Demo.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Zielfunktionsdesign: das Ziel ist das Modell](https://sebastianhanisch.net/konzepte-zielfunktionsdesign.html)."
)
