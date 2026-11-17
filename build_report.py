"""Build the two-page milestone draft from the default benchmark outputs."""

import argparse
import csv
import json
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build(results, out, fonts, authors):
    """Read experiment artifacts and typeset the report in the course format."""
    def read_csv(name):
        """Read one results CSV as a list of row dicts."""
        with (results / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    baselines = read_csv("baselines.csv")
    search = {int(r["l_init"]): r for r in read_csv("mlp_search.csv")}
    settings = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    if set(search) != set(range(15)) or len(baselines) != 30 or settings["step_cap"] != 5:
        raise ValueError("Build the report from the default full benchmark, including plots.")

    for name, file in (("TNR", "times.ttf"), ("TNRBold", "timesbd.ttf"),
                       ("TNRItalic", "timesi.ttf"), ("TNRBoldItalic", "timesbi.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNRBold", italic="TNRItalic", boldItalic="TNRBoldItalic")
    tnr = ParagraphStyle("Text", fontName="TNR", fontSize=12, leading=14,
                         textColor=colors.HexColor("#191919"), spaceAfter=7)
    heading = ParagraphStyle("Heading", parent=tnr, fontName="TNRBold", spaceBefore=4, spaceAfter=4, keepWithNext=True)
    centered = ParagraphStyle("Center", parent=tnr, alignment=TA_CENTER, spaceAfter=4)
    number = ParagraphStyle("Numbers", parent=tnr, alignment=TA_RIGHT, spaceAfter=0)
    label = ParagraphStyle("Column", parent=number, fontName="TNRBold")

    def p(s):
        """Set a body paragraph."""
        return Paragraph(s, tnr)

    def h(s):
        """Set a section heading."""
        return Paragraph(s, heading)

    summary_counts = (sum(float(row["saved_ms"]) > 0 for row in search.values()), len(search))
    best = search[14]
    attn = next(row for row in baselines if row["block"] == "Attention")
    min_low = min(float(search[level]["saved_percent"]) for level in range(14))
    max_low = max(float(search[level]["saved_percent"]) for level in range(14))
    move_count = [int(row["accepted_moves"]) for row in search.values()]
    min_runtime = min(float(row["solver_seconds"]) for row in search.values())
    max_runtime = max(float(row["solver_seconds"]) for row in search.values())

    story = [
        Paragraph("<b>Automated Bootstrapping Placement for Encrypted Inference</b>", centered),
        Paragraph("CS5491 Artificial Intelligence · Project Milestone", centered),
        *([Paragraph(authors, centered)] if authors else []),
        Spacer(1, 10),
        h("1. Problem and motivation"),
        p("In CKKS-style encrypted inference, multiplication consumes ciphertext levels. A bootstrap refreshes the ciphertext but adds substantial latency. Given a directed acyclic graph (DAG), an initial level from 0 to 14, and the course latency model, our goal is to place Bootstrap1, Bootstrap2 and Modswitch operations so every node has enough levels and the block's estimated latency is as low as possible. The reference placement refreshes early even when the input already has a high level. We ask which refreshes or level reductions are needed at each input level."),
        h("2. Placement and search"),
        p("Each original edge chooses one chain: empty, Bootstrap1, Bootstrap2, Modswitch, or either bootstrap followed by Modswitch. Bootstrap1 refreshes to 14; Bootstrap2 refreshes to 13 and is permitted only on edges carrying 8 or 16 ciphertexts. Modswitch lowers the level without a latency charge in the provided model. A node with several parents receives the minimum incoming level. We compile matching insertion prefixes on edges from the same source into a single node, so shared operations are charged once."),
        p("Starting from the official MLP placement, each sweep replaces the chain on one original edge at a time. We consider all allowed bootstrap types and candidate switch targets, reject moves that violate level or ciphertext constraints, and score each surviving placement with the reference evaluator. The search accepts the best strict improvement per sweep. It stops after five accepted moves or when no feasible single-edge move improves the score. The resulting placements can seed the branch-and-bound solver planned for the final project."),
        h("3. Evaluation setup"),
        p(f"We use the supplied MLP and Attention graphs. Before optimizing, we rebuilt each supplied placement and checked its original node levels, estimated latency, depth, budget, and bootstrap counts against the reference evaluator in <i>dag_builder.py</i>. All {len(baselines)} block/level combinations matched. The MLP reference ranged from {float(search[0]['baseline_ms']):,.2f} to {float(best['baseline_ms']):,.2f} ms. The Attention reference was {float(attn['latency_ms']):,.2f} ms at every level. We optimized the MLP for this milestone."),
        p("Estimated latency is the sum of fitted operation costs. We count Bootstrap1 per ciphertext and Bootstrap2 per pair. Depth is the largest level cost along a path, excluding insertions. Budget is the input level plus 14 per counted Bootstrap1 and 13 per counted Bootstrap2. Utilization is depth divided by budget. Solver time measures the search itself on the CPU."),
        PageBreak(),
        h("4. Preliminary MLP results"),
        p(f"The search found a feasible improvement at all {summary_counts[0]} of {summary_counts[1]} input levels. Table 1 shows three examples. The savings at levels 0 through 13 were {min_low:.2f}–{max_low:.2f}%, with the same total bootstrap count as the reference. At level 14, we could remove the first Bootstrap1, which processed four ciphertexts. This move alone saved 4,160.60 ms at level 14 but left insufficient levels at level 6."),
    ]

    table_rows = [[Paragraph(s, label) for s in ("Input level", "Baseline (ms)", "Search (ms)", "Saved", "Bootstraps")]]
    for level in (0, 6, 14):
        row = search[level]
        values = (str(level), f"{float(row['baseline_ms']):,.2f}", f"{float(row['candidate_ms']):,.2f}",
                  f"{float(row['saved_percent']):.2f}%", f"{row['baseline_bootstraps']} → {row['candidate_bootstraps']}")
        table_rows.append([Paragraph(value, number) for value in values])
    table = Table(table_rows, colWidths=[76, 104, 99, 67, 97], hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eff5")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#7992a8")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.45, colors.HexColor("#b8c6d2")),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([
        table, Spacer(1, 5),
        p("<b>Table 1.</b> Representative MLP results. Bootstraps are counted per ciphertext or pair, as applicable."),
        p("A Modswitch before the inverse square root in LayerNorm lowered its input from 14 to 7 and saved 1,019.13 ms on that operator alone, while leaving enough levels downstream. At input level 14, removing the first bootstrap also reduced the budget from 136 to 80. Depth stayed at 26, so utilization rose from 19.1% to 32.5%."),
        Image(str(results / "mlp_comparison.png"), width=14.6 * cm, height=14.6 * cm * 684 / 1416),
        p("<b>Figure 1.</b> Estimated MLP latency saved relative to the official placement."),
        h("5. Limitations and next steps"),
        p(f"Each search accepted {min(move_count)}–{max(move_count)} moves ({min_runtime:.2f}–{max_runtime:.2f} s per input on the test laptop) before exhausting improving single-edge moves. Some improvements may need coordinated changes to multiple edges. We have no global optimality certificate. Next we will use these placements to initialize a bounded branch-and-bound search for the MLP and port the search to Attention, where fan-out and shared costs make isolated moves more restrictive."),
        p("<b>Reproduction.</b> Run <i>python benchmark.py --output-dir results</i> from the code directory. The command writes the full benchmark tables and annotated MLP graphs at input levels 0 and 14."),
    ])

    def footer(canvas, doc):
        """Draw the centered page number."""
        canvas.saveState()
        canvas.setFont("TNR", 12)
        canvas.setFillColor(colors.HexColor("#4b4b4b"))
        canvas.drawCentredString(A4[0] / 2, 1.65 * cm, str(doc.page))
        canvas.restoreState()

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                            topMargin=2.5 * cm, bottomMargin=2.5 * cm, title="CS5491 Project Milestone",
                            author=authors, pageCompression=1)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    """Read report paths and font location from command-line flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fonts", type=Path, default=Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")
    parser.add_argument("--authors", default="", help="Optional group author line for the draft")
    args = parser.parse_args()
    build(args.results, args.out, args.fonts, args.authors)


if __name__ == "__main__":
    main()
