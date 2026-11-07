"""PDF-Export des Ergebnisses (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from zeh_evaluation import (
    Settings,
    de,
    best_info,
    contest_info,
    finite_info,
    incentive_info,
    mix_info,
    verdict_best,
    verdict_contest,
    verdict_finite,
    verdict_incentive,
    verdict_mix,
)

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "„": '"', "“": '"', "²": "2", "λ": "lambda", "∞": "unendlich"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: Settings) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Ehrlichkeit belohnen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Glaube {s.p} %, Regel {s.rule_name}, Verschiebung {s.d} Prozentpunkte, {s.n} Ereignisse, {s.m} Prognostiker, Skalierung {de(s.lam / 10)}, Brier-Anteil {s.mix} %, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    sections = (
        ("Die beste Meldung", verdict_best(best_info(s))[1]),
        ("Die Stärke des Anreizes", verdict_incentive(incentive_info(s))[1]),
        ("Ehrlich gegen übertreibend", verdict_finite(finite_info(s), s.rule_name)[1]),
        ("Der Wettbewerb", verdict_contest(contest_info(s))[1]),
        ("Die Mischung aus Brier und Absolutfehler", verdict_mix(mix_info(s))[1]),
    )
    for title, text in sections:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, _s(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, _s(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Meldungen in ganzen Prozent 1 bis 99. Quelle: ziel-ehrlichkeit-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
