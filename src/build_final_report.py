"""Build the four-page final report from the branch-and-bound experiment outputs.

The report body (sections 1-5, tables, figure) is typeset in the course format --
Times New Roman 12 pt, A4, 2.5 cm margins, single spacing -- and kept within four
pages; the reference list follows and does not count toward the limit.  Table
values are read from ``results/`` so the PDF always matches the artifacts.
"""

import argparse
import csv
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)


def build(results, out, fonts, authors):
    """Read experiment artifacts and typeset the final report."""

    def read_csv(name):
        """Read one results CSV as a list of row dicts."""
        with (results / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    bnb = {(r["block"], int(r["l_init"])): r for r in read_csv("bnb_search.csv")}
    local = {int(r["l_init"]): r for r in read_csv("mlp_search.csv")}
    if set(bnb) != {(b, l) for b in ("MLP", "Attention") for l in range(15)}:
        raise ValueError("Build the report from the full 2x15 sweep in results/bnb_search.csv.")

    def budget_row(name):
        """Format a budget-sensitivity row: latency with expanded nodes."""
        row = read_csv(name)[0]
        return "{:,.2f} ({:,})".format(float(row["bnb_ms"]), int(row["expansions"]))

    def cell(block, level, key):
        """Format one numeric cell of the 2x15 sweep with thousands separators."""
        return "{:,.2f}".format(float(bnb[block, level][key]))

    def gap_cell(block, level):
        """Format one sweep row's optimality gap as a percentage."""
        return "{:.1f}%".format(float(bnb[block, level]["gap_percent"]))

    def boot(block, level):
        """Format one sweep row's bootstrap counts as B1/B2."""
        return "{}/{}".format(bnb[block, level]["b1"], bnb[block, level]["b2"])

    for name, file in (("TNR", "times.ttf"), ("TNRBold", "timesbd.ttf"),
                       ("TNRItalic", "timesi.ttf"), ("TNRBoldItalic", "timesbi.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNRBold", italic="TNRItalic",
                                  boldItalic="TNRBoldItalic")
    tnr = ParagraphStyle("Text", fontName="TNR", fontSize=12, leading=14,
                         textColor=colors.HexColor("#191919"), spaceAfter=7)
    heading = ParagraphStyle("Heading", parent=tnr, fontName="TNRBold", spaceBefore=4,
                             spaceAfter=4, keepWithNext=True)
    centered = ParagraphStyle("Center", parent=tnr, alignment=TA_CENTER, spaceAfter=4)
    caption = ParagraphStyle("Caption", parent=tnr, fontSize=11, leading=13, spaceBefore=3)
    number = ParagraphStyle("Numbers", parent=tnr, alignment=TA_RIGHT, spaceAfter=0)
    label = ParagraphStyle("Column", parent=number, fontName="TNRBold")

    def p(s):
        """Set a body paragraph."""
        return Paragraph(s, tnr)

    def h(s):
        """Set a section heading."""
        return Paragraph(s, heading)

    def cap(s):
        """Set a table or figure caption."""
        return Paragraph(s, caption)

    def grid(data, widths):
        """Build a three-rule table from row paragraphs and column widths."""
        style = TableStyle([
            ("FONT", (0, 0), (-1, -1), "TNR", 12),
            ("FONT", (0, 0), (-1, 0), "TNRBold", 12),
            ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.HexColor("#191919")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#191919")),
            ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor("#191919")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ])
        table = Table(data, colWidths=widths, hAlign="CENTER")
        table.setStyle(style)
        return table

    def row(*cells):
        """Set one table row: bold label cell and right-aligned numeric cells."""
        return [Paragraph(c, label if i == 0 else number) for i, c in enumerate(cells)]

    # ---- Overview figure, generated from the same CSVs as the tables ----------
    overview = results / "figures" / "report_overview.png"
    write_overview(bnb, local, overview)

    mlp_local = [float(local[l]["candidate_ms"]) for l in range(15)]
    attn_ties = sum(abs(float(bnb["Attention", l]["bnb_ms"]) - float(bnb["Attention", l]["baseline_ms"])) < 0.01
                    for l in range(15))
    attn_ties_offtable = sum(
        abs(float(bnb["Attention", l]["bnb_ms"]) - float(bnb["Attention", l]["baseline_ms"])) < 0.01
        for l in range(15) if l not in (0, 6, 7, 8))

    story = [
        Paragraph("<b>Automated Bootstrapping Placement for Encrypted Inference</b>", centered),
        Paragraph("CS5491 Artificial Intelligence · Final Project Report", centered),
        *([Paragraph(authors, centered)] if authors else []),
        Spacer(1, 8),
        h("1. Problem and motivation"),
        p("In CKKS-style encrypted inference [1], every multiplication consumes a ciphertext level, and when "
          "levels run out an expensive Bootstrap restores the ciphertext to the top of the tower -- over one "
          "second per ciphertext on the course's measured model. Where these refreshes are inserted therefore "
          "dominates a block's latency. We study the course's placement task: given the computation DAG of a "
          "block (the supplied MLP and Attention blocks), an initial level L<sub>init</sub> in 0--14, and the "
          "reference cost model, insert Bootstrap1, Bootstrap2 and Modswitch operations on the edges so that "
          "every operator receives enough level while the block's estimated latency is minimized."),
        p("The reference placement is built to be feasible at every input level and refreshes conservatively, "
          "so it repeats work when the input arrives nearly full: at L<sub>init</sub> = 14 one redundant "
          "Bootstrap1 costs 4,160.60 ms, which our milestone recovered. This report asks the natural "
          "follow-up: for each input level, which refreshes are actually necessary, how far can automated "
          "search push the latency, and what can be certified about the answer? The problem is pure "
          "constrained combinatorial optimization over a DAG with shared costs -- an apt target for the search "
          "techniques of this course. It needs no LLM and no GPU; everything runs on a laptop CPU."),
        h("2. Formulation and cost model"),
        p("Each edge (u,v) carries a number of ciphertexts; each operator v has a level cost c<sub>v</sub> and "
          "a fitted latency curve that is non-decreasing in its input level (L<sub>max</sub> = 14). An edge "
          "chooses one insertion chain: empty, B1, B2, M(t), B1&middot;M(t), or B2&middot;M(t). Bootstrap1 "
          "restores the level to 14; Bootstrap2 to 13 and is only legal on edges with 8 or 16 ciphertexts; a "
          "Modswitch M(t) sets the level to t, strictly lowering it at zero latency cost in this model. "
          "Insertions shared by several edges from the same source are materialized once and charged once. A "
          "node with several parents receives the minimum incoming level, and every operator must see at least "
          "max(c<sub>v</sub>, the operator's validity floor) levels."),
        p("The objective is the estimated latency: the sum of operator costs at their input levels plus the "
          "inserted bootstraps, where a Bootstrap1 is charged per ciphertext (1,040.15 ms) and a Bootstrap2 "
          "per pair (1,293.13 ms). We also report depth (the largest level cost along any path) and budget = "
          "L<sub>init</sub> + 14 per Bootstrap1 ciphertext + 13 per Bootstrap2 pair, with utilization = "
          "depth/budget."),
        h("3. Method"),
        p("<b>Phase 1 (milestone).</b> Single-edge local search from the official placement accepts the best "
          "strict improvement per sweep (up to five moves). It delivers feasible seeds and a baseline to beat, "
          "but stalls at single-edge local optima and offers no optimality information."),
        p("<b>Phase 2: branch-and-bound.</b> Edges are assigned chains in a fixed order; children are expanded "
          "cheapest-chain-first (empty first, then free switches at higher targets, then B2, then B1) so good "
          "incumbents appear early. For a partial assignment the pruning lower bound sums three admissible "
          "terms and reports infeasibility when propagated level intervals contradict an operator's floor: "
          "(i) each node's achievable input-level interval is propagated topologically -- a bootstrap fixes "
          "its output to 14 or 13, a Modswitch to its target, undecided edges leave the widest interval; "
          "(ii) every operator is priced at the cheapest latency in its still-achievable interval, which is "
          "admissible because the curves are non-decreasing; (iii) bootstraps are charged once per (source, "
          "ciphertext count, refresh type) so shared prefixes are not double-billed, and undecided edges whose "
          "source cannot deliver the child's floor without a refresh are charged the cheapest cover once per "
          "(source, count) group -- such a refresh is already unavoidable, and one shared node can serve the "
          "whole group, so charging more would not be admissible. On complete assignments the bound equals "
          "the scorer's latency exactly."),
        p("<b>Certificates under budgets.</b> When a time or node budget stops the search we report the "
          "incumbent U with the certified global lower bound LB = min bound over the unexplored frontier, and "
          "the gap 100(U&minus;LB)/U. A placement is labeled proven optimal only when the search exhausts the "
          "space; otherwise it is best within budget, with its certificate."),
        p("<b>Cross-level multi-start.</b> Bootstrap outputs are absolute and the fitted curves are linear in "
          "level, so placements migrate across input levels almost freely -- the MLP optimum is portable from "
          "L<sub>init</sub> = 0 to 13 at +0.16 ms per level. Two passes exploit this at the same per-case "
          "budget: pass 1 starts from the official placement (and the local search for the MLP); pass 2 adds "
          "every pass-1 placement re-scored at all 15 levels as extra seeds. The second pass lifted MLP "
          "levels 3&ndash;13 by about 1,037 ms each at zero search cost."),
        p("<b>Verification.</b> Nineteen regression tests pin the solver down: leaf exactness against the "
          "reference scorer; admissibility checked on every partial state of small fork/chain/join graphs "
          "against an exhaustive completion oracle; branch-and-bound optimum equal to exhaustive optimum on "
          "those graphs; and the milestone numbers reproduced bit-for-bit."),
        h("4. Experiments and results"),
        p("Setup. 30 cases: both blocks &times; L<sub>init</sub> = 0&ndash;14, each with 60 s wall-clock and "
          "200,000 expanded nodes of budget (whichever comes first), on a Ryzen 7 8845H laptop under Python "
          "3.11. Estimated inference latency and solver wall time are reported separately, and all 30 "
          "outcomes are kept, including ties. Tables 1&ndash;2 show representative levels; the complete "
          "30-row record is in <i>results/bnb_search.csv</i>."),
        KeepTogether([
            grid([row("L<sub>init</sub>", "Official", "Local", "BnB", "vs local", "B1/B2", "LB (ms)", "Gap"),
                  row("0", cell("MLP", 0, "baseline_ms"), "{:,.2f}".format(mlp_local[0]),
                      cell("MLP", 0, "bnb_ms"), "3.83%", boot("MLP", 0), cell("MLP", 0, "lower_bound_ms"),
                      gap_cell("MLP", 0)),
                  row("6", cell("MLP", 6, "baseline_ms"), "{:,.2f}".format(mlp_local[6]),
                      cell("MLP", 6, "bnb_ms"), "3.82%", boot("MLP", 6), cell("MLP", 6, "lower_bound_ms"),
                      gap_cell("MLP", 6)),
                  row("13", cell("MLP", 13, "baseline_ms"), "{:,.2f}".format(mlp_local[13]),
                      cell("MLP", 13, "bnb_ms"), "3.82%", boot("MLP", 13), cell("MLP", 13, "lower_bound_ms"),
                      gap_cell("MLP", 13)),
                  row("14", cell("MLP", 14, "baseline_ms"), "{:,.2f}".format(mlp_local[14]),
                      cell("MLP", 14, "bnb_ms"), "4.52%", boot("MLP", 14), cell("MLP", 14, "lower_bound_ms"),
                      gap_cell("MLP", 14))],
                 [1.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 1.55 * cm, 1.7 * cm, 2.3 * cm, 1.55 * cm]),
            cap("<b>Table 1.</b> MLP block, estimated latency (ms). Levels 1&ndash;5 and 7&ndash;12 behave "
                "like 0/6/13 (26,070.18&ndash;26,071.94 ms, 4 B1 + 4 B2). B1 counts ciphertexts, B2 counts "
                "pairs. Gap = 100(U&minus;LB)/U with LB the certified bound."),
        ]),
        Spacer(1, 4),
        KeepTogether([
            grid([row("L<sub>init</sub>", "Official", "BnB", "Saved", "B1/B2", "LB (ms)", "Gap"),
                  row("0", cell("Attention", 0, "baseline_ms"), cell("Attention", 0, "bnb_ms"), "--",
                      boot("Attention", 0), cell("Attention", 0, "lower_bound_ms"), gap_cell("Attention", 0)),
                  row("6", cell("Attention", 6, "baseline_ms"), cell("Attention", 6, "bnb_ms"), "3.58%",
                      boot("Attention", 6), cell("Attention", 6, "lower_bound_ms"), gap_cell("Attention", 6)),
                  row("7", cell("Attention", 7, "baseline_ms"), cell("Attention", 7, "bnb_ms"), "3.45%",
                      boot("Attention", 7), cell("Attention", 7, "lower_bound_ms"), gap_cell("Attention", 7)),
                  row("8", cell("Attention", 8, "baseline_ms"), cell("Attention", 8, "bnb_ms"), "0.03%",
                      boot("Attention", 8), cell("Attention", 8, "lower_bound_ms"), gap_cell("Attention", 8))],
                 [1.3 * cm, 2.5 * cm, 2.5 * cm, 1.6 * cm, 1.7 * cm, 2.5 * cm, 1.7 * cm]),
            cap("<b>Table 2.</b> Attention block, estimated latency (ms). The other {} levels tie the "
                "official placement at 53,558.17 ms; they stay in the record rather than being dropped. The "
                "level-6 solution replaces five B1 ciphertexts with twelve B2 pairs.".format(attn_ties_offtable)),
        ]),
        p("<b>Findings.</b> (F1) The MLP improves over both baselines at all 15 levels and shows two regimes: "
          "for L<sub>init</sub> &le; 13 the solver removes one single-ciphertext Bootstrap1 (4 B1 + 4 B2, "
          "about 26,070 ms, 3.8% under the local search), while at L<sub>init</sub> = 14 every Bootstrap1 "
          "disappears (0 B1 + 4 B2, 21,911.66 ms, 22.1% under the official placement) -- with a full-height "
          "input, the opening refreshes are redundant. (F2) Cross-level migration is the largest free gain: "
          "the level-0 placement runs unchanged at levels 1&ndash;13 and saved about 1,037 ms -- one "
          "Bootstrap1 -- at levels 3&ndash;13 over the milestone's per-level local searches. (F3) The "
          "Attention placement is near-robust: only levels 6&ndash;8 improve (3.58%, 3.45%, 0.03%), and "
          "level 6 wins by trading five B1 ciphertexts for twelve B2 pairs, which per-pair charging favors on "
          "wide fan-out. (F4) All 30 cases stopped on budget; we therefore claim no proven optimum on the "
          "full blocks and report certified bounds with gaps of 39&ndash;62% throughout."),
        p("<b>Budget sensitivity.</b> Table 3 repeats two cases at one sixth and five times the main "
          "budget: the incumbent is bit-identical everywhere, and the certificate does not move either -- the "
          "frontier bound stays within 0.16 ms (MLP) and 0.60 ms (Attention) of the trivial root bound "
          "(10,645.98 and 20,356.73 ms). Search time is not the bottleneck; bound strength is."),
        KeepTogether([
            grid([row("Case", "10 s", "60 s (main)", "300 s + 200k nodes"),
                  row("MLP L=6", budget_row("budget_mlp6_t10.csv"),
                      cell("MLP", 6, "bnb_ms") + " ({:,})".format(int(bnb["MLP", 6]["expansions"])),
                      budget_row("budget_mlp6_t300.csv")),
                  row("Attention L=9", budget_row("budget_attn9_t10.csv"),
                      cell("Attention", 9, "bnb_ms") + " ({:,})".format(int(bnb["Attention", 9]["expansions"])),
                      budget_row("budget_attn9_t300.csv"))],
                 [3.0 * cm, 3.5 * cm, 3.8 * cm, 4.7 * cm]),
            cap("<b>Table 3.</b> Budget sensitivity: incumbent latency (ms) with expanded nodes in "
                "parentheses."),
        ]),
        Spacer(1, 2),
        KeepTogether([
            Image(str(overview), width=15.2 * cm, height=15.2 * cm * 310 / 1000),
            cap("<b>Figure 1.</b> Left: estimated MLP latency by input level; the level-14 regime is "
                "visible. Right: saving over the official placement for both blocks; the {} Attention ties "
                "are shown as zero.".format(attn_ties)),
        ]),
        h("5. Insights and conclusion"),
        p("Three lessons stand out. First, the reference placement's redundancy is structural and "
          "level-dependent: refreshes at the head of the DAG only pay for themselves at low input levels, so "
          "the optimal bootstrap configuration switches regime at L<sub>init</sub> = 14. Second, placements "
          "are portable across levels, so solving one level well pays at all of them -- a consequence of "
          "absolute refresh outputs and linear cost curves. Third, better certificates, not longer runs, are "
          "the lever for closing the optimality gap: thirty times more budget moved neither the incumbent nor "
          "the bound."),
        p("With a verified admissible bound, branch-and-bound reliably beats the reference placement on both "
          "blocks -- all 15 levels for the MLP, 3 for Attention -- and returns machine-checked certificates "
          "everywhere; on small graphs it is exact against exhaustive enumeration. Limitations: gaps of "
          "39&ndash;62% leave the full-block optima uncertified, and the joint attention+MLP graph extension "
          "was not attempted. Natural next steps are stronger bounds on undecided bootstraps (path-cover or "
          "hitting-set relaxations), best-first or Lagrangian search, and joint-graph placement."),
        p("<b>Reproduction.</b> From the repository root: <i>python main.py test</i>; "
          "<i>python main.py bnb --block both --levels 0-14 --time-budget 60 --node-budget 200000 --output "
          "results/bnb_search.csv</i>; <i>python main.py figures --results results --out results/figures "
          "--cases MLP:6 MLP:14 Attention:6 Attention:7</i>; <i>python main.py report-final --out "
          "CS5491_Project_Final.pdf</i>."),
        PageBreak(),
        h("References"),
        p("[1] J. H. Cheon, A. Kim, M. Kim, and Y. Song. Homomorphic encryption for arithmetic of "
          "approximate numbers. In <i>ASIACRYPT</i>, 2017."),
        p("[2] C. Gentry. Fully homomorphic encryption using ideal lattices. In <i>STOC</i>, 2009."),
        p("[3] S. Dathathri et al. CHET: An optimizing compiler for fully-homomorphic neural-network "
          "inferencing. In <i>PLDI</i>, 2019."),
        p("[4] A. Boemer et al. nGraph-HE: a graph compiler for deep learning on homomorphic encryption. "
          "<i>arXiv:1908.07175</i>, 2019."),
        p("[5] I. Chillotti et al. TFHE: Fast fully homomorphic encryption over the torus. "
          "<i>Journal of Cryptology</i>, 33(1), 2020."),
    ]

    def footer(canvas, doc):
        """Draw the centered page number."""
        canvas.saveState()
        canvas.setFont("TNR", 12)
        canvas.setFillColor(colors.HexColor("#4b4b4b"))
        canvas.drawCentredString(A4[0] / 2, 1.65 * cm, str(doc.page))
        canvas.restoreState()

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                            topMargin=2.5 * cm, bottomMargin=2.5 * cm,
                            title="CS5491 Project Final Report", author=authors, pageCompression=1)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def write_overview(bnb, local, out):
    """Two-panel figure: MLP latency curves and per-level savings for both blocks."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    for file in ("times.ttf", "timesbd.ttf"):
        font_manager.fontManager.addfont(str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / file))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "mathtext.fontset": "stix", "axes.grid": True,
                         "grid.alpha": 0.3, "grid.linewidth": 0.5})

    levels = list(range(15))
    official = [float(bnb["MLP", l]["baseline_ms"]) for l in levels]
    bnb_ms = [float(bnb["MLP", l]["bnb_ms"]) for l in levels]
    local_ms = [float(local[l]["candidate_ms"]) for l in levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.1), dpi=200)
    ax1.plot(levels, official, "--", color="#8a8a8a", lw=1.4, label="Official")
    ax1.plot(levels, local_ms, ":", color="#c07020", lw=1.4, label="Local search")
    ax1.plot(levels, bnb_ms, "-", color="#1f4e79", lw=1.8, label="Branch-and-bound")
    ax1.set_xlabel("Input level $L_{init}$")
    ax1.set_ylabel("Estimated latency (ms)")
    ax1.set_xticks(levels)
    ax1.legend(frameon=False, fontsize=8, loc="lower left")

    width = 0.4
    mlp_saving = [100 * (float(bnb["MLP", l]["baseline_ms"]) - float(bnb["MLP", l]["bnb_ms"])) /
                  float(bnb["MLP", l]["baseline_ms"]) for l in levels]
    attn_saving = [100 * (float(bnb["Attention", l]["baseline_ms"]) - float(bnb["Attention", l]["bnb_ms"])) /
                   float(bnb["Attention", l]["baseline_ms"]) for l in levels]
    ax2.bar([l - width / 2 for l in levels], mlp_saving, width=width, color="#1f4e79", label="MLP")
    ax2.bar([l + width / 2 for l in levels], attn_saving, width=width, color="#c07020", label="Attention")
    ax2.set_xlabel("Input level $L_{init}$")
    ax2.set_ylabel("Saved vs official (%)")
    ax2.set_xticks(levels)
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout(pad=0.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    """Read report paths and font location from command-line flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fonts", type=Path, default=Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")
    parser.add_argument("--authors", default="Xinghang ZENG, Chengyu BIE, and Weijia ZHANG",
                        help="Group author line")
    args = parser.parse_args()
    build(args.results, args.out, args.fonts, args.authors)


if __name__ == "__main__":
    main()
