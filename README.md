# Ehrlichkeit belohnen – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-ziel-ehrlichkeit-demo.streamlit.app/)** -->

Stück 7 der Linie **Zielfunktionsdesign** („Das Ziel ist das Modell“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Arbeitsname; Name, Bereich und Themenfeld werden erst bei der Einbindung in die Website festgelegt). Es ist ein **gelbes** (optionales) Stück und das letzte der Linie (Stück 1 bis 6 und 8 bis 14 stehen davor: `zielvergleich-demo`, `preis-der-fairness-demo`, `zwei-kriterien-demo`, `lexikographisch-demo`, `gleichstand-demo`, `hart-gegen-weich-demo`, `goodhart-kurve-demo`, `modellluecken-demo`, `verlagerung-demo`, `planstabilitaet-demo`, `ziel-lernen-demo`, `zielkarte-demo`, `ziel-verfahrenswahl-demo`). **Das Beispiel wechselt**: keine Zuordnung, sondern eine **Prognose**, weil Ehrlichkeit nur bei Angaben Sinn hat. Ein Prognostiker glaubt, ein Ereignis trete mit Wahrscheinlichkeit `p` ein, und **meldet** `q`; eine **Regel** (das Ziel) vergibt Punkte gegen den Ausgang. **Ehrlich** ist `q = p`. Die Demo misst, **bei welcher Regel die ehrliche Meldung die beste ist, wie schwach der Anreiz selbst dann ist, wie oft die Ehrlichkeit auf endlich vielen Ereignissen vorn liegt, warum ein Wettbewerb, in dem nur der Sieger zählt, sie bei vielen Prognostikern trotzdem verdirbt und was schon ein Anteil Absolutfehler in der Regel anrichtet**. Alles ist **simuliert**; keine Aussage über echte Prognostiker.

## Warum dieses Problem

Wer eine Prognose einkauft oder Prognostiker vergleicht, wählt mit der Regel, nach der er sie bewertet, auch, **was sie melden werden**. Eine Regel, die Übertreibung belohnt, bekommt Übertreibung; eine, der Ehrlichkeit gleichgültig ist, bekommt irgendetwas. Die Fragen: Welche Regeln belohnen die wahre Angabe, welche nicht? Wie stark ist der Anreiz? Reicht er auf wenigen Ereignissen, um Übertreiben zu erkennen? Und was geschieht, wenn nicht nach Punkten, sondern nach Rang bezahlt wird?

## Modell

Der Prognostiker glaubt `p` (ganze Prozent 1 bis 99) und meldet `q` (ganze Prozent 1 bis 99; 0 und 100 sind ausgeschlossen, weil die Log-Regel dort unendlich wird); der Ausgang `y` ist 1 oder 0. Fünf **Regeln**, größer ist besser: **Brier** `−(q − y)²`, **Log** `y·ln q + (1 − y)·ln(1 − q)`, **sphärisch** (Wahrscheinlichkeit des Ausgangs geteilt durch die Länge von `(q, 1 − q)`), **Trefferquote** (1 Punkt, wenn `q` auf der Seite des Ausgangs von 1/2 liegt, bei `q` = 1/2 ein halber) und **Absolutfehler** `−|q − y|`. **Beste Meldung:** die mit der höchsten **erwarteten** Punktzahl bei dem Glauben `p`; sie ist exakt in Brüchen gerechnet (Brier, Trefferquote, Absolutfehler) oder mit Toleranz 1e-12 (Log, sphärisch). **Anreiz:** der Verlust einer um `d` Prozentpunkte verschobenen Meldung und die Zahl der Ereignisse, nach der man sie im Mittel mit 1,645 Standardabweichungen erkennt (Normalnäherung, gleiches `p`). **Zufallsversuche:** je Ereignis `p` gleichverteilt 0,05 bis 0,95, der Ausgang mit `p` gezogen; verglichen wird die ehrliche Meldung mit einer um `d` Prozentpunkte **übertriebenen** (vom Wert 1/2 weg) oder **untertriebenen** (um `d` auf 1/2 zu, höchstens bis 1/2 und nie darüber hinaus), 2 000 Versuche. **Wettbewerb:** `m` Prognostiker mit eigenem Glauben (Log-Odds der wahren Wahrscheinlichkeit plus Rauschen mit Standardabweichung 0,5), Sieger ist die höchste Gesamtpunktzahl auf 20 Ereignissen (Gleichstand geteilt); einer meldet `sigmoid(λ · Log-Odds seines Glaubens)`, die anderen ehrlich. **Mischung:** die Regel `μ·Brier + (1 − μ)·Absolutfehler`; die beste Meldung liegt exakt bei `q* = p + (1 − μ)(2p − 1) / (2μ)`. Formeln stehen im 📐-Expander der App.

## Methodik

Die App zeigt: die erwartete Punktzahl jeder Meldung (mit ehrlicher und bester Meldung), den Verlust einer Verschiebung je Regel, ehrlich gegen übertreibend und untertreibend je Regel auf `n` Ereignissen (gestapelte Balken), den Wettbewerb (Gewinnchance gegen den Faustwert `1/m`) und die Verschiebung der besten Meldung bei einer Mischung. Zählungen über feste Ereigniszahlen (10, 20, 50, 100, 500; 2 000 Versuche je Zeile) und über den Wettbewerb (Brier, Log, Trefferquote; 2, 5, 10, 20 Prognostiker; fünf Skalierungen; 20 000 Versuche je Zelle, dieselben Zufallszahlen für alle Skalierungen) liegen hinter Schaltern, dazu der Wettbewerb an einer zweiten Menge fester Versuche (Hold-out, Seed 200000). Verdikte entstehen live aus der Rechnung, jede Zahl im Text kommt aus dem Ergebnis.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus den Auswertungsfunktionen der App und sind in `tests/test_claims.py` nachgerechnet. „Zählung“ heißt: feste Seeds (Stichprobe: Seed 100000; Hold-out: Seed 200000), nie ein einzelner Versuch; die Zahlen sind **exakte Rechnungen oder Zählungen an diesem Modell**, kein Satz und keine Aussage über echte Prognostiker. Reihenfolge der Regeln: Brier / Log / sphärisch / Trefferquote / Absolutfehler. Die Zählung über Ereigniszahlen nimmt je Zahl den Seed 100000 plus ihren Index (10: 100000, 20: 100001, …), die Presets den Seed 100000; deshalb weicht zum Beispiel „68,8 %“ (Preset, 20 Ereignisse) von „69,0 %“ (Zählung) ab.

| Frage | Befund |
|---|---|
| ehrliche Meldung ist die **einzige** beste (von 99 Glauben `p`, exakt) | **99 / 99 / 99 / 0 / 2** |
| ehrliche Meldung ist **eine** der besten | 99 / 99 / 99 / 99 / 3 |
| ehrliche Meldung **verliert** | 0 / 0 / 0 / 0 / **96 von 99** (nur bei `p` = 1 % und 99 % ist sie beste, bei 50 % sind alle Meldungen gleich gut) |
| Gewinn durch Lügen (Mittel / Maximum, erwartete Punkte) | 0 / 0 / 0 / 0 (die Trefferquote belohnt Lügen nie, bestraft aber Lügen über die 50-%-Grenze, siehe nächste Zeile) / **0,0792 / 0,1200** (Absolutfehler) |
| Trefferquote: Meldungen `q` ≠ `p`, die erwartete Punkte kosten (von 9 702 Paaren) | **4 900**; alle bringen die Meldung auf die andere Seite von 50 % oder genau auf 50 % (bei `p` ≠ 50 %), keine bleibt auf derselben Seite; Beispiel `p` = 30 %, Meldung 60 %: erwartet 0,3 statt 0,7 |
| Verlust der Meldung um 10 Prozentpunkte nach oben bei `p` = 30 / 50 / 80 % | Brier **0,0100** bei allen (exakt 1/100); Log 0,0216 / 0,0204 / 0,0444; sphärisch 0,0127 / 0,0137 / 0,0074 |
| Ereignisse bis zur Erkennung (1,645 Standardabweichungen; Normalnäherung) bei `p` = 30 / 50 / 80 % | Brier 227 / 271 / 173; Log 238 / 267 / 144; sphärisch 215 / 276 / 191 |
| ehrlich strikt vor **übertreibend** (+10), `n` = 10 / 20 / 50 / 100 / 500 | Brier **60,5 / 69,0 / 79,3 / 86,4 / 99,1 %**; Log 59,4 / 69,5 / 83,0 / 91,1 / 100,0 %; sphärisch 58,5 / 67,1 / 75,4 / 83,5 / 98,2 % |
| dabei: ehrlich verliert gegen Übertreiben (Brier) | 39,5 / 31,0 / 20,6 / 13,6 / 0,9 % |
| ehrlich strikt vor **untertreibend** (−10; höchstens bis 50 %) | Brier 65,5 / 70,2 / 79,3 / 86,1 / 99,5 %; Log 65,6 / 70,2 / 79,3 / 86,2 / 99,4 %; sphärisch 65,3 / 69,8 / 77,4 / 85,4 / 99,2 % |
| Absolutfehler: ehrlich vor übertreibend / vor untertreibend | 7,0 / 2,2 / 0,0 / 0,0 / 0,0 % und 95,1 / 99,2 / 100 / 100 / 100 % (bei 50 Ereignissen liegt die ehrliche Meldung in diesen 2 000 Versuchen nicht vorn; mit sehr vielen Versuchen kommt das in seltenen Fällen vor) |
| Trefferquote: Gleichstand gegen Übertreiben / ehrlich vor untertreibend | **100 / 100 / 100 / 100 / 100 %** und 42,5 / 49,1 / 56,8 / 66,1 / 82,5 % (Gleichstand 27,8 / 17,8 / 11,1 / 7,3 / 2,3 %) |
| Wettbewerb, **20** Prognostiker (Faustwert 5 %): Gewinnchance bei Skalierung 0,5 / 1 / 1,5 / 2 / 3, Brier | 3,4 / 5,1 / **7,0** / **7,1** / 6,0 % (Hold-out 3,0 / 5,0 / 7,0 / 7,1 / 6,2 %) |
| dasselbe, Log | 3,8 / 5,0 / **8,2** / 7,8 / 5,1 % (Hold-out 3,6 / 5,1 / 8,2 / 8,0 / 5,4 %) |
| Wettbewerb, **10** Prognostiker (Faustwert 10 %), Brier / Log | 6,8 / 9,9 / 10,9 / 10,1 / 8,2 % und 7,7 / 10,1 / 12,0 / 10,2 / 6,3 % (Hold-out Brier 6,7 / 9,7 / 10,9 / 10,0 / 8,0 %) |
| Wettbewerb, **5** Prognostiker, Brier / Log | 14,7 / 20,2 / 18,4 / 15,5 / 11,6 % und 15,6 / 20,0 / 17,8 / 13,6 / 7,8 % (Übertreiben lohnt nicht) |
| Wettbewerb, **2** Prognostiker, Brier / Log | 42,6 / 49,9 / 41,0 / 33,0 / 23,9 % und 42,5 / 49,9 / 36,7 / 25,1 / 13,0 % |
| dabei: mittlere Punktzahl je Ereignis des Abweichlers, `m` = 20, Brier | −0,2107 / **−0,2067** / −0,2156 / −0,2266 / −0,2449 (ehrlich ist die beste) |
| Trefferquote im Wettbewerb | die Gewinnchance hängt nicht von der Skalierung ab: 50,1 / 20,0 / 9,8 / 5,2 % bei 2 / 5 / 10 / 20 Prognostikern |
| Mischung `μ·Brier + (1 − μ)·Absolutfehler`: ehrlich die **einzige** beste (von 99 Glauben) bei `μ` = 0 / 10 / … / 90 / 100 % | 2 / 3 / 3 / 3 / 3 / 3 / 3 / 5 / 5 / 11 / **99** |
| dabei: Abstand der besten Meldung vom Glauben (Mittel; Maximum, Prozentpunkte) | 24,3; 49 / 21,8; 44 / 19,4; 39 / 17,0; 34 / 14,5; 29 / 12,1; 24 / 9,7; 19 / 7,3; 15 / 4,8; 10 / **2,4; 5** / 0,0; 0 |
| dabei: Formel `q* = p + (1 − μ)(2p − 1) / (2μ)` trifft die beste Meldung (±0,5 Prozentpunkte) | in 99 von 99 Glauben bei jedem `μ` von 10 bis 100 % |
| Preset „Brier: Ehrlichkeit lohnt sich“ (Glaube 30 %) | ehrlich die einzige beste; Meldung 40 % kostet **1/100**, erkennbar nach 227 Ereignissen; nach 20 Ereignissen ehrlich vorn in **68,8 %**, vor untertreibend 69,5 % |
| Preset „Absolutfehler belohnt die Lüge“ (Glaube 30 %, 50 Ereignisse) | beste Meldung **1 %**, Gewinn 0,1160 (−0,3040 gegen −0,4200); ehrlich vor übertreibend in **0,1 %** der Versuche |
| Preset „Trefferquote ist gleichgültig“ (Glaube 30 %) | ehrlich eine von **49** gleich guten Meldungen (1 % bis 49 %); die Meldung 60 % auf der anderen Seite kostet 0,4 (erwartet 0,3 statt 0,7); Gleichstand gegen Übertreiben in 100,0 %, ehrlich vor untertreibend in 48,9 % |
| Preset „Log-Verlust bei 50 %“ | Meldung 60 % kostet **0,0204**, erkennbar nach 267 Ereignissen; ehrlich vor übertreibend in 70,4 % |
| Preset „Zehn Ereignisse verraten kaum etwas“ (Brier) | ehrlich vor übertreibend in **60,5 %**, verliert in 39,5 %; vor untertreibend 65,5 % |
| Preset „Der Wettbewerb verleitet“ (Log, 20 Prognostiker, Skalierung 1,5) | Abweichler gewinnt **7,6 %**, ehrlich 5,0 %; mittlere Punktzahl −0,6421 gegen −0,6026 |
| Preset „Ein bisschen uneigentlich, ein bisschen verzerrt“ (Glaube 80 %, 90 % Brier) | beste Meldung **83 %** (+3 Prozentpunkte), Formel 83,3 % |
| Preset „Kleinster Wettbewerb (2 Prognostiker)“ (Brier, zehn Ereignisse) | Abweichler gewinnt 40,5 %, ehrlich 50,6 % |

**Korrektur der Planungshypothese und Abweichungen von der Messreihe:** (1) „Genauigkeit belohnt Übertreibung“ stimmt im Erwartungswert **nur für den Absolutfehler** (unter den fünf Regeln; im Wettbewerb lohnt Übertreiben auch bei Brier und Log, siehe (3)); die **Trefferquote** sieht nur die Seite von 50 %: Lügen bringt nie Punkte (Gewinn 0), aber die ehrliche Meldung ist nie die einzige beste (0 von 99); Lügen auf derselben Seite von 50 % kosten nichts, Lügen auf die andere Seite oder genau auf 50 % kosten Punkte (4 900 von 9 702 Paaren `p`, `q` mit `q` ≠ `p`, bei `p` = 50 % keine), und das Übertreiben um 10 Prozentpunkte bleibt auf derselben Seite und ist in 100 % der Versuche nicht zu erkennen. Die frühere Fassung dieser Demo ließ das Untertreiben über 1/2 hinausschießen (`p` = 45 %, `d` = 10 gab 55 %); seit der Korrektur nimmt es die Meldung höchstens bis 50 % zurück, und die Spalte „untertreibend“ der Zählung ist neu gemessen (Brier bei 10 Ereignissen 65,5 statt 66,3 %, Absolutfehler 95,1 statt 89,6 %; die Trefferquote und alle Zahlen zum Übertreiben ändern sich nicht). (2) Die **Mischung** (in der Planung als „vor dem Bau zu messen“ offen) zeigt: schon ein Anteil Absolutfehler verschiebt die beste Meldung, bei `μ` = 90 % um 2,4 Prozentpunkte im Mittel (höchstens 5); **ehrlich ist nur reiner Brier**, und zwar bei `μ` = 100 % in 99 von 99 Fällen, bei keinem kleineren Anteil in mehr als 11. (3) Der **Wettbewerb** zeigt den Vorteil des Übertreibens nicht nur bei 20, sondern in kleinerem Maß auch bei 10 Prognostikern (Brier 10,9 gegen 9,9 %, im Hold-out 10,9 gegen 9,7 %); in der Messreihe mit 4 000 Versuchen war das bei 10 Prognostikern innerhalb der Streuung. Bei 2 und 5 Prognostikern lohnt es sich nicht. (4) Die **sphärische Regel** ist strikt eigentlich, ihre erwartete Punktzahl aber **nicht konkav** in der Meldung (nur einhöckrig); die Tests prüfen für sie deshalb Einhöckrigkeit statt Konkavität. (5) Die Zählungen der Demo weichen leicht von der Messreihe der Planung ab (zum Beispiel ehrlich vor übertreibend bei 10 Ereignissen (Brier) 60,5 gegen 62,1 %, Wettbewerb mit 20 Prognostikern bei Skalierung 1,5 (Brier) 7,0 gegen 6,9 %): die Demo zieht die Zahlen eines Versuchs aus einem gemeinsamen Zufallsstrom je Ereigniszahl (die Messreihe aus einem Strom je Versuch), und der Wettbewerb nutzt 20 000 statt 4 000 Versuche. Von der Literatur ist nichts gelesen und nichts zitiert (siehe „Ehrliche Grenzen“).

## Ehrliche Grenzen

- **Ein Modell:** binäre Ereignisse, `p` je Ereignis gleichverteilt 0,05 bis 0,95, Meldungen in ganzen Prozent; Mehrklassen-, Intervall- und Quantilregeln kommen nicht vor.
- **Erwartete Punktzahl:** der Prognostiker maximiert die Erwartung; wer Punkte nicht linear bewertet, verhält sich anders (der Wettbewerb zeigt das für Rangauszahlungen).
- **Verzerrung auf eine Art:** der Wettbewerb kennt nur die Skalierung `λ` der Log-Odds als Abweichung, die ehrlichen Mitspieler haben verrauschte Überzeugungen; andere Abweichungen oder Strategien kommen nicht vor.
- **Erkennung als Normalnäherung:** `n95` rechnet mit gleichem `p` an allen Ereignissen und der Normalverteilung der Summe; die Zufallsversuche zeigen die Streuung selbst.
- **Ehrlich heißt hier: Meldung gleich Glaube.** Die Regeln belohnen nicht, dass der Glaube richtig ist.
- **Kein Mechanism Design:** keine Anreizverträglichkeit mehrerer Spieler, keine Auktionen; der Marktmacher mit Log-Verlust (LMSR) ist Gegenstand der `prognosemarkt-demo`.
- **Keine Literatur:** strikt eigentliche Scoring-Regeln (Gneiting und Raftery 2007) sind nur als Suchtreffer und über andere Demos bekannt; Primärtexte sind nicht gelesen, die Demo übernimmt daraus nichts.
- **Kein Satz:** die Zahlen sind Zählungen und Rechnungen an diesem Modell, nicht allgemein.

## Verifikation

Insgesamt 158 Tests (`pytest tests/`), alle grün.

- **Kern:** Handrechnungen (Brier `(q − p)²` mit `p` = 3/10: Verluste 1/100 und 4/100; Absolutfehler 12/100; Trefferquote 7/10, 1/2, 3/10 und 49 gleich gute Meldungen; Log 0,0204; `n95` 270,6 / 267,0 / 276,0; Mischung `q*` = 5/6); jede Regel, jede beste Meldung und jede erwartete Punktzahl gegen eine Schleife in reinem Python (`Fraction`, `math`), die Mischung gegen die Schleife **und** die Formel, Zufallsversuche und Wettbewerb gegen eine Schleife über dieselben gezogenen Zahlen.
- **Eigenschaften:** bei Brier, Log und sphärisch ist ehrlich für jeden Glauben die einzige beste Meldung und die erwartete Punktzahl steigt bis `p` und fällt danach; Brier und Log sind konkav, die sphärische Regel nicht; der Brier-Verlust ist genau `(q − p)²`; Spiegelsymmetrie `p ↔ 1 − p`; der Absolutfehler ist linear in `q` mit Steigung `2p − 1`; die Trefferquote sieht nur die Seite von 1/2 und kostet genau die Lügen, die die Meldung auf die andere Seite oder auf 1/2 rücken (4 900 von 9 702 Paaren), nie eine auf derselben Seite; Untertreiben geht nie über 1/2 hinaus; die Anteile der Versuche summieren sich auf 100 %; Ehrlichkeit gewinnt bei eigentlichen Regeln mit mehr Ereignissen öfter; die Trefferquote im Wettbewerb hängt nicht von `λ` ab; die Verschiebung der Mischung hat das Vorzeichen von `p − 1/2` und schrumpft mit dem Brier-Anteil.
- **Randfälle:** Glauben 1, 50 und 99 %, keine Verschiebung, Verschiebung bis an den Rand, 10 und 500 Ereignisse, zwei Prognostiker, Skalierung 1, Mischung 0 und 100, Seeds 0 und 999999.
- **Verdikte:** reine Funktionen, jede Verzweigung und jede Zahl im Text an Presetwerten (`tests/test_verdicts.py`).
- **Presets:** Vollständigkeit, Werte innerhalb der Reglergrenzen, Standard-Preset gleich Standardeinstellung, kein anderes Preset gleich den Voreinstellungen, und dass jedes Preset zeigt, was sein Name sagt.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), jedes Preset, Footer-Test, Layout.
- **Zählungen:** `tests/test_experiments.py` hält alle Zahlen der Zählungen und des Hold-outs fest; `tests/test_claims.py` jede Zahl dieser README.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `zeh_algorithm.py` | Kern: Regeln, erwartete Punktzahl, beste Meldung, Verlust, Erkennung, Zufallsversuche, Wettbewerb |
| `zeh_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `zeh_evaluation.py` | Experimente, Zählungen, Verdikte |
| `zeh_layout.py` | Fester Kopf (Stylesheet) |
| `zeh_pdf_export.py` | PDF-Export des Ergebnisses |
| `zeh_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste), Zufalls-Button |
| `zeh_scenario.py` | Ziehung der Ereignisse und der Überzeugungen im Wettbewerb |
| `zeh_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Orakel, Handrechnungen, Eigenschaften, Randfälle, Verdikte, Presets, Zählungen, Claims, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Kein Mechanism Design und keine Märkte:** keine Auktionen, keine Mehr-Spieler-Anreizverträglichkeit; der Prognosemarkt (LMSR) ist die `prognosemarkt-demo`.
- **Keine Mehrklassen-, Intervall- oder Quantilregeln:** die Messlatte für Intervalle ist die `forecast-interval-demo`.
- **Keine Kalibrierungsdiagramme und kein Güte-Vergleich:** hier zählt der Anreiz, nicht die Güte der Prognose (siehe `rating-vergleich-demo`).
- **Keine Überoptimierung eines Ersatzziels:** das sind Stück 8 und 9 der Linie.
- **Keine Nachbauten der Literatur.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Zielfunktionsdesign: das Ziel ist das Modell](https://sebastianhanisch.net/konzepte-zielfunktionsdesign.html).
