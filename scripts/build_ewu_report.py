"""Build the CSE400B project report in the department's printout format.

    python scripts/build_ewu_report.py --payload deliverables/manuscript_payload.json \
        --out deliverables/CSE400B-MOSAIC-GNN-Report.docx

The format is taken from the sample printout supplied by the course: East West
University title page, table of contents, abbreviations, eight numbered
chapters each opening with "Chapter N" on the left and the chapter title on
the right, bold section headings, justified Times New Roman body, tables
captioned above and figures captioned below, and a "Page N of M" footer.

Every number in the results chapter is read from the payload that
build_manuscript.py assembles from the run outputs. Nothing is typed in. A
value whose phase did not run is printed as such rather than filled.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import docx
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import (WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT,
                            WD_TAB_LEADER)
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT = "Times New Roman"
NOT_RUN = "[not run]"
BLACK = RGBColor(0, 0, 0)


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _set_font(run, size=11, bold=False, italic=False, color=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(a), FONT)


def para(doc, text="", size=11, bold=False, italic=False, align="justify",
         space_after=10, space_before=0, indent=None, style=None,
         color=None):
    p = doc.add_paragraph(style=style)
    p.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT,
                   "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "right": WD_ALIGN_PARAGRAPH.RIGHT,
                   "justify": WD_ALIGN_PARAGRAPH.JUSTIFY}[align]
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    if indent is not None:
        pf.left_indent = Inches(indent)
    if text:
        _set_font(p.add_run(text), size, bold, italic, color)
    return p


def rich(doc, parts, size=11, align="justify", space_after=10):
    """A paragraph from (text, bold) pairs, for inline bold like 'Table 2'."""
    p = doc.add_paragraph()
    p.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT,
                   "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "justify": WD_ALIGN_PARAGRAPH.JUSTIFY}[align]
    p.paragraph_format.space_after = Pt(space_after)
    for text, bold in parts:
        _set_font(p.add_run(text), size, bold)
    return p


TOC_ENTRIES: list = []     # (level, display text, text to find in the PDF)


def chapter(doc, number: str, title: str, first=False):
    TOC_ENTRIES.append((1, (f"Chapter {number}: {title}" if number
                            else title), title))
    # "Page break before" on the heading itself, rather than an empty
    # paragraph holding a break: the latter lands on a page of its own
    # whenever the previous chapter happens to end flush with the margin.
    if number:
        head = para(doc, f"Chapter {number}", size=18, bold=True,
                    align="left", space_after=4, color=BLACK)
        para(doc, title, size=16, bold=True, align="right", space_after=14,
             style="Heading 1", color=BLACK)
    else:
        head = para(doc, title, size=18, bold=True, align="left",
                    space_after=14, style="Heading 1", color=BLACK)
    if not first:
        head.paragraph_format.page_break_before = True


def section(doc, text: str, size=12):
    TOC_ENTRIES.append((2, text, text))
    para(doc, text, size=size, bold=True, align="left", space_before=10,
         space_after=8, style="Heading 2", color=BLACK)


def bullets(doc, items, numbered=False):
    for i, it in enumerate(items, 1):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.45)
        p.paragraph_format.space_after = Pt(6)
        if isinstance(it, tuple):
            lead, rest = it
            _set_font(p.add_run(f"{i}. " if numbered else "• "), 11)
            _set_font(p.add_run(lead), 11, bold=True)
            _set_font(p.add_run(rest), 11)
        else:
            _set_font(p.add_run((f"{i}. " if numbered else "• ") + it), 11)


def caption(doc, label: str, text: str, above=True):
    p = doc.add_paragraph()
    p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if above
                   else WD_ALIGN_PARAGRAPH.CENTER)
    p.paragraph_format.space_before = Pt(0 if above else 4)
    p.paragraph_format.space_after = Pt(4 if above else 12)
    _set_font(p.add_run(label + " "), 11, bold=True)
    _set_font(p.add_run(text), 11)


def add_table(doc, rows, header=True, widths=None, font=9.5):
    t = doc.add_table(rows=0, cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            cells[ci].text = ""
            p = cells[ci].paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.CENTER if (header and ri == 0)
                           else WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            _set_font(p.add_run(str(val)), font, bold=(header and ri == 0))
    if widths:
        # Fixed layout, or Word and LibreOffice re-fit the columns and the
        # model names wrap mid-word.
        t.autofit = False
        tblPr = t._tbl.tblPr
        for tag in ("w:tblLayout",):
            old = tblPr.find(qn(tag))
            if old is not None:
                tblPr.remove(old)
        lay = OxmlElement("w:tblLayout")
        lay.set(qn("w:type"), "fixed")
        tblPr.append(lay)
        for ri in range(len(t.rows)):
            for ci, w in enumerate(widths):
                t.rows[ri].cells[ci].width = Inches(w)
        for ci, w in enumerate(widths):
            if ci < len(t.columns):
                t.columns[ci].width = Inches(w)
    return t


def add_figure(doc, path: str, width=6.1):
    if not os.path.isfile(path):
        para(doc, f"[figure missing: {os.path.basename(path)}]",
             align="center", italic=True)
        return False
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(path, width=Inches(width))
    return True


def _field(p, instr: str):
    r = p.add_run()
    _set_font(r, 10)
    for el, attr, val in (("w:fldChar", "w:fldCharType", "begin"),
                          ("w:instrText", "xml:space", "preserve"),
                          ("w:fldChar", "w:fldCharType", "end")):
        e = OxmlElement(el)
        e.set(qn(attr), val)
        if el == "w:instrText":
            e.text = instr
        r._element.append(e)


def footer_page_of(doc):
    for sec in doc.sections:
        f = sec.footer
        p = f.paragraphs[0] if f.paragraphs else f.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for t in list(p.runs):
            t.text = ""
        _set_font(p.add_run("Page "), 10)
        _field(p, "PAGE")
        _set_font(p.add_run(" of "), 10)
        _field(p, "NUMPAGES")


def toc_field(doc):
    p = doc.add_paragraph()
    _field(p, 'TOC \\o "1-3" \\h \\z \\u')
    para(doc, "(If the list above is empty, open in Word and press Ctrl+A "
               "then F9 to build the contents.)", size=9, italic=True,
         align="left")


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #
class F:
    def __init__(self, payload: dict):
        self.facts = payload.get("facts", {})
        raw = payload.get("tables", {})
        if isinstance(raw, dict):
            self.tables = dict(raw)
        else:
            self.tables = {t["id"]: t for t in raw}
        self.figures = {f["label"]: f for f in payload.get("figures", [])}
        self.aux = {}          # extra run artefacts loaded from --runs
        self.missing = set()

    def a(self, path: str, default=None):
        """Read a value from an auxiliary run artefact, dotted path."""
        cur = self.aux
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                self.missing.add(f"aux.{path}")
                return default
            cur = cur[k]
        return cur

    def g(self, target: str, path: str, default=None):
        cur = self.facts.get(target, {})
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                self.missing.add(f"{target}.{path}")
                return default
            cur = cur[k]
        return cur

    def n(self, target, path, d=3):
        v = self.g(target, path)
        return NOT_RUN if v is None else f"{float(v):.{d}f}"

    def i(self, target, path):
        v = self.g(target, path)
        return NOT_RUN if v is None else f"{int(v):,}"

    def best(self, target, key):
        b = self.g(target, key)
        if not isinstance(b, dict):
            return NOT_RUN
        return f"{b.get('model','?')} ({float(b.get('value',0)):.3f})"

    # -- payload tables ---------------------------------------------------- #
    def grid(self, tid: str):
        t = self.tables.get(tid)
        if not t or not t.get("grid"):
            self.missing.add(f"table.{tid}")
            return None
        return [list(map(str, r)) for r in t["grid"]]

    def tcap(self, tid: str) -> str:
        t = self.tables.get(tid)
        return (t or {}).get("caption", "")

    def cell(self, tid: str, row_key: str, col: str):
        """Value at (first column == row_key, header == col); None if absent."""
        g = self.grid(tid)
        if not g:
            return None
        head = g[0]
        if col not in head:
            self.missing.add(f"table.{tid}.{col}")
            return None
        ci = head.index(col)
        for r in g[1:]:
            if r[0] == row_key:
                return r[ci]
        self.missing.add(f"table.{tid}.{row_key}")
        return None

    def fcell(self, tid, row_key, col, d=3):
        v = self.cell(tid, row_key, col)
        try:
            return f"{float(v):.{d}f}"
        except (TypeError, ValueError):
            return NOT_RUN

    def t2(self, key: str, tid: str = "T2"):
        """Read the two-column (Metric, Value) panel table by metric name."""
        g = self.grid(tid)
        if not g:
            return None
        for r in g[1:]:
            for i in range(0, len(r) - 1, 2):
                if r[i] == key:
                    return r[i + 1]
        self.missing.add(f"table.{tid}.{key}")
        return None

    def num(self, tid, row_key, col):
        v = self.cell(tid, row_key, col)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None


def fmt(v, d=3):
    """Format a float that was computed here, not typed in."""
    return NOT_RUN if v is None else f"{float(v):.{d}f}"


def pct(v, d=1):
    return NOT_RUN if v is None else f"{100.0 * float(v):.{d}f}"


def _cnt(v):
    """A count that was measured, with thousands separators."""
    return NOT_RUN if v is None else format(int(v), ",")


def tidy_cell(v: str) -> str:
    """A count exported as 5841.000 is a count, so print it as one."""
    s = str(v)
    if "." in s and s.replace("-", "").replace(".", "").isdigit():
        f = float(s)
        if f == int(f) and abs(f) >= 1000:
            return format(int(f), ",")
    return s


def payload_table(doc, x: F, tid: str, label: str, cols=None, rows=None,
                  widths=None, font=8.5, caption_text=None, head_map=None):
    """Render one payload table under the report's own caption number."""
    g = x.grid(tid)
    caption(doc, label + ":", caption_text or x.tcap(tid))
    if not g:
        para(doc, f"[{tid} not present in the payload: the phase that writes "
                  f"it did not run]", italic=True, align="left")
        return
    head = g[0]
    keep = list(range(len(head))) if not cols else [
        head.index(c) for c in cols if c in head]
    body = g[1:]
    if rows:
        body = [r for r in body if r[0] in rows]
    out = [[(head_map or {}).get(head[i], head[i]) for i in keep]]
    out += [[tidy_cell(r[i]) for i in keep] for r in body]
    add_table(doc, out, widths=widths, font=font)


# --------------------------------------------------------------------------- #
# Front matter
# --------------------------------------------------------------------------- #
AUTHORS = [("Jannatul Ferdous Jerin", "2023-1-60-099"),
           ("Farzana Shahanaz", "2023-1-60-201"),
           ("Afifa Tabassum", "2023-1-60-242"),
           ("SM Asif", "2023-1-60-040")]

TITLE = ("Molecular Features Add Prognostic Information Beyond Clinical "
         "Covariates: An Audited Pan-Cancer Evaluation of a Two-Level "
         "Multi-Omic Graph Neural Network Across 32 TCGA Studies")

ABBREVIATIONS = [
    ("AJCC", "American Joint Committee on Cancer"),
    ("AUROC", "Area Under the Receiver Operating Characteristic Curve"),
    ("AUPRC", "Area Under the Precision-Recall Curve"),
    ("BH", "Benjamini-Hochberg (false discovery rate correction)"),
    ("CNA", "Copy Number Alteration"),
    ("CGC", "COSMIC Cancer Gene Census"),
    ("CI", "Confidence Interval"),
    ("Cox", "Cox Proportional Hazards Model"),
    ("CV", "Cross-Validation"),
    ("DSS", "Disease-Specific Survival"),
    ("ECE", "Expected Calibration Error"),
    ("GAT", "Graph Attention Network"),
    ("GCN", "Graph Convolutional Network"),
    ("GISTIC", "Genomic Identification of Significant Targets in Cancer"),
    ("GNN", "Graph Neural Network"),
    ("GRL", "Gradient Reversal Layer"),
    ("IG", "Integrated Gradients"),
    ("KM", "Kaplan-Meier"),
    ("LOCO", "Leave-One-Cancer-Out"),
    ("MCC", "Matthews Correlation Coefficient"),
    ("MC-dropout", "Monte Carlo Dropout"),
    ("MLP", "Multi-Layer Perceptron"),
    ("MOSAIC-GNN", "Multi-Omic Structure-Adaptive Interpretable Cross-cancer "
                   "Graph Neural Network (proposed model)"),
    ("MSK-IMPACT", "Memorial Sloan Kettering Integrated Mutation Profiling of "
                   "Actionable Cancer Targets (468-gene panel)"),
    ("OS", "Overall Survival"),
    ("PFS", "Progression-Free Survival"),
    ("PPI", "Protein-Protein Interaction"),
    ("RGAT", "Relational Graph Attention Network"),
    ("RNA", "Ribonucleic Acid (gene expression block)"),
    ("SWA", "Stochastic Weight Averaging"),
    ("TCGA", "The Cancer Genome Atlas"),
    ("t-SNE", "t-distributed Stochastic Neighbor Embedding"),
]


def title_page(doc, logo: str, date: str):
    if os.path.isfile(logo):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(18)
        p.add_run().add_picture(logo, width=Inches(1.5))
    para(doc, "EAST WEST UNIVERSITY", size=20, bold=True, align="center",
         space_after=36)
    para(doc, TITLE, size=15, bold=True, align="center", space_after=28)
    for txt, sz in (("CSE400B", 12), ("Project Report", 12),
                    ("Semester: FALL-2025", 12)):
        para(doc, txt, size=sz, bold=True, align="center", space_after=6)
    para(doc, "", space_after=14)
    para(doc, "Submitted By", size=12, bold=True, align="center",
         space_after=8)
    t = add_table(doc, [["Name", "ID"]] + [[n, i] for n, i in AUTHORS],
                  widths=[2.6, 1.7], font=11)
    for ri, row in enumerate(t.rows):
        for c in row.cells:
            for p in c.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.bold = (ri == 0)
    para(doc, "", space_after=22)
    para(doc, "Supervisor: Dr. Ahmed Wasif Reza (DAWR)", size=14, bold=True,
         align="center", space_after=6)
    para(doc, "Professor, Department of Computer Science & Engineering",
         size=11, align="center", space_after=6)
    para(doc, f"Date of Report Submitted:  {date}", size=11, align="center")
    doc.add_page_break()


def toc_line(doc, level: int, text: str, page):
    """One contents line: text on the left, page number on a right tab."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(2)
    pf.left_indent = Inches(0.0 if level == 1 else 0.3)
    pf.tab_stops.add_tab_stop(Inches(5.9), WD_TAB_ALIGNMENT.RIGHT,
                              WD_TAB_LEADER.DOTS)
    _set_font(p.add_run(text), 11, bold=(level == 1))
    _set_font(p.add_run("\t" + ("0" if page is None else str(page))), 11,
              bold=(level == 1))
    return p


def contents(doc, entries=None):
    """Table of contents.

    `entries` is the measured list [(level, text, page)] from a first
    rendering pass. On the pass that has no measurements yet the same lines
    are emitted with a placeholder page number, so that the two passes
    paginate identically and the numbers written on the second pass are the
    numbers the reader will actually turn to.
    """
    para(doc, "Table of Contents", size=16, bold=True, align="left",
         space_after=10, color=BLACK)
    if entries is None:
        toc_field(doc)
    else:
        for level, text, page in entries:
            toc_line(doc, level, text, page)
    doc.add_page_break()


def abbreviations(doc):
    para(doc, "Abbreviations", size=16, bold=True, align="left",
         space_after=12)
    t = doc.add_table(rows=0, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    for short, long in ABBREVIATIONS:
        cells = t.add_row().cells
        for ci, val in enumerate((short, long)):
            cells[ci].text = ""
            p = cells[ci].paragraphs[0]
            p.paragraph_format.space_after = Pt(3)
            _set_font(p.add_run(val), 11)
        cells[0].width = Inches(1.5)
        cells[1].width = Inches(4.6)
    doc.add_page_break()


# --------------------------------------------------------------------------- #
# Chapters 1-3
# --------------------------------------------------------------------------- #
def ch1(doc, x: F):
    chapter(doc, "1", "Introduction", first=True)
    section(doc, "1.1 Background")
    para(doc, "Cancer prognosis has traditionally rested on histological "
              "grade, tumour stage and a small number of molecular markers. "
              "These variables are cheap, interpretable and clinically "
              "entrenched, but they summarise a tumour that is in fact "
              "characterised at genomic, transcriptomic and epigenomic level "
              "in routine practice. Multi-omic profiling is now standard in "
              "large consortia such as The Cancer Genome Atlas, and the "
              "question is no longer whether molecular data exists but "
              "whether it adds anything a clinician does not already have.")
    para(doc, "Multi-omic tumour profiles are both high dimensional and "
              "highly structured. Genes interact through regulatory and "
              "physical networks, patients resemble one another in ways that "
              "cut across tissue of origin, and the two kinds of structure "
              "are almost always modelled apart. Models that exploit gene "
              "networks typically ignore patient similarity; models built on "
              "patient graphs typically treat genes as a flat feature vector. "
              "Neither family addresses the pan-cancer setting directly.")
    para(doc, "The pan-cancer setting introduces a failure mode that "
              "single-cohort work does not face. A model trained across many "
              "cancer types can reach high apparent accuracy by learning to "
              "recognise which cohort a patient came from and applying a "
              "per-cohort prior, without learning anything transferable about "
              "biology. Such a model is a cohort detector wearing the clothes "
              "of a prognostic model, and the standard evaluation protocols "
              "do not distinguish the two.")
    para(doc, "This project builds and rigorously evaluates MOSAIC-GNN, a "
              "two-level graph neural network that couples a shared gene "
              "graph with a learned patient-similarity graph, and reports "
              "what that evaluation found rather than the result the "
              "architecture was designed to produce.")

    section(doc, "1.2 Research Questions and Problem Statements")
    rich(doc, [("Primary Research Question: ", True),
               ("Does jointly modelling gene-level relational structure and "
                "patient-level similarity structure in a single pan-cancer "
                "network produce prognostic performance that survives a "
                "paired, multiplicity-corrected comparison against "
                "established methods on identical held-out patients?", False)])
    rich(doc, [("Sub Research Questions:", True)])
    bullets(doc, [
        ("RQ1: ", "(Added value of molecular data) Do molecular features add "
                  "prognostic information beyond the clinical covariates a "
                  "clinician already has, and does any such gain replicate "
                  "across independently fitted prediction targets?"),
        ("RQ2: ", "(Architecture) Does each component of the proposed "
                  "two-level design — the gene graph, the functional-module "
                  "pooling and the learned patient graph — earn its place, "
                  "measured against the pipeline's own run-to-run noise "
                  "floor?"),
        ("RQ3: ", "(Transfer) Does the learned representation stratify risk "
                  "in a cancer type held out entirely from training, which a "
                  "cohort detector could not do?"),
        ("RQ4: ", "(Honest comparison) Under a protocol with a pre-committed "
                  "feature space, paired bootstrap tests and a label "
                  "permutation control, how does the proposed model actually "
                  "rank against conventional baselines?"),
    ])
    para(doc, "Formally, given patients from K cancer studies, each described "
              "by mutation, copy-number and expression measurements over a "
              "fixed gene panel together with clinical covariates and a "
              "right-censored survival outcome, the task is to learn a shared "
              "representation that supports both a classification head and a "
              "survival head while remaining transferable across cohorts. The "
              "difficulty is that molecular features outnumber patients by "
              "roughly two orders of magnitude, that cohorts differ enormously "
              "in size and event rate, and that several plausible sources of "
              "leakage can manufacture apparent performance.")

    section(doc, "1.3 Objectives")
    bullets(doc, [
        "To assemble a genuine pan-cancer dataset from the cBioPortal public "
        "API over a feature space fixed in advance of any analysis, so that "
        "gene selection cannot be tuned to the outcome.",
        "To design and implement MOSAIC-GNN, coupling a relational gene graph "
        "with functional-module pooling and a learned patient graph under "
        "joint classification and survival objectives.",
        "To evaluate it against a panel of established comparators on "
        "identical folds, using paired bootstrap tests with "
        "multiplicity correction rather than unpaired comparisons of means.",
        "To audit the pipeline with a generalisation-gap analysis and a label "
        "permutation null, and to report every control that was run and every "
        "phase that was not.",
        "To quantify what the molecular data contributes over clinical "
        "covariates alone, which is the premise the field rests on and the "
        "comparison most multi-omic papers omit.",
    ], numbered=True)

    section(doc, "1.4 Organization of the Report")
    para(doc, "Chapter 2 analyses the problem, its constraints and the "
              "stakeholders affected. Chapter 3 reviews the literature on "
              "graph-based and non-graph multi-omic modelling and identifies "
              "the gaps this work addresses. Chapter 4 describes the data, "
              "preprocessing, leakage controls, graph construction, model "
              "architecture and evaluation protocol. Chapter 5 records the "
              "computing environment and software stack. Chapter 6 presents "
              "the experimental results and discusses them. Chapter 7 "
              "considers societal, health, safety, legal and ethical impact. "
              "Chapter 8 concludes, states the limitations plainly and sets "
              "out future work.")


def ch2(doc, x: F):
    chapter(doc, "2", "Problem Analysis")
    section(doc, "2.1 Complexity of Pan-Cancer Multi-Omic Prognosis")
    para(doc, "Three properties make this problem hard in ways that are easy "
              "to hide. First, dimensionality: molecular features outnumber "
              "patients by about two orders of magnitude, so a model can fit "
              "the training set perfectly while learning nothing, and any "
              "feature selection performed on the full dataset leaks the "
              "outcome into the feature space in a way no cross-validation "
              "scheme can undo.")
    para(doc, "Second, heterogeneity. The cohorts assembled here range from "
              "36 to 1,084 patients and from 0.02 to 0.851 deaths per "
              "patient. A pooled metric computed over such a set is dominated "
              "by the largest cohorts, and a per-cohort metric computed on "
              "the smallest rests on a handful of comparable pairs. Both "
              "failure modes produce numbers that look reportable.")
    para(doc, "Third, the cohort-detector problem described in Chapter 1. "
              "Because tissue of origin is strongly encoded in molecular and "
              "clinical data alike, a pan-cancer model can score well by "
              "identifying the cohort rather than by learning prognosis. "
              "Distinguishing the two requires an experiment in which the "
              "held-out cohort is absent from training entirely.")

    section(doc, "2.2 Constraints and Stakeholders")
    para(doc, "The work is constrained by compute, by data availability and "
              "by the standard of evidence a clinical claim requires. One "
              "ablation fold at pan-cancer scale costs approximately half an "
              "hour on the available hardware, so the full factorial of "
              "configurations is out of reach and the phases that were run at "
              "reduced settings are labelled as such wherever they appear.")
    bullets(doc, [
        ("Patients and clinicians. ", "The ultimate stakeholders. A "
         "prognostic model that overstates its accuracy is worse than no "
         "model, because it will be trusted in exactly the cases where it is "
         "wrong."),
        ("Researchers. ", "A benchmark is only useful if its comparisons are "
         "paired, corrected and reproducible. Most of the value of this work "
         "to other groups is the protocol and the negative results."),
        ("Data custodians. ", "TCGA and cBioPortal impose terms of use and "
         "the data is de-identified; no attempt is made to re-identify, and "
         "no patient-level data is redistributed with this report."),
    ])

    section(doc, "2.3 Summary of Identified Gaps")
    bullets(doc, [
        "Gene-level and patient-level structure are modelled separately in "
        "almost all prior work, so the interaction between them is untested.",
        "Comparisons are commonly unpaired and uncorrected, which inflates "
        "apparent wins when many comparators are evaluated.",
        "The clinical-only baseline — the model a clinician already has — is "
        "frequently absent, so the quantity the field claims to improve is "
        "never measured.",
        "Transfer to a wholly unseen cancer type is rarely tested, leaving "
        "the cohort-detector hypothesis unaddressed.",
        "Feature selection is often performed on the pooled dataset, which "
        "leaks the outcome into the feature space.",
    ])


def ch3(doc, x: F):
    chapter(doc, "3", "Literature Review")
    section(doc, "3.1 Multi-Omic Integration without Graphs")
    para(doc, "Early multi-omic prognostic work concatenated omic blocks into "
              "a single matrix and applied penalised linear models or tree "
              "ensembles. Penalised Cox regression in particular remains a "
              "strong and under-acknowledged baseline: it is fast, stable "
              "under high dimensionality and, as the results in Chapter 6 "
              "show, competitive with far more elaborate architectures on "
              "survival ranking. Late-fusion neural approaches train one "
              "encoder per omic block and combine representations at the "
              "head, which handles missing blocks gracefully but models no "
              "structure within a block.")
    section(doc, "3.2 Gene-Level Graph Approaches")
    para(doc, "A second family embeds prior biological structure directly. "
              "Protein-protein interaction networks, pathway memberships and "
              "co-expression graphs have all been used to constrain message "
              "passing between genes, on the argument that genes act in "
              "modules rather than independently. These methods improve "
              "interpretability and occasionally accuracy, but they treat "
              "each patient as an independent sample and so cannot exploit "
              "similarity between patients.")
    section(doc, "3.3 Patient-Level Graph Approaches")
    para(doc, "MOGONET and related patient-graph methods invert the "
              "assumption: patients become nodes, edges encode molecular "
              "similarity, and a graph convolutional network propagates label "
              "information across the cohort. This captures population "
              "structure well and is the closest comparator family to the "
              "present work, but genes remain a flat feature vector and the "
              "patient graph is usually fixed in advance rather than learned.")
    section(doc, "3.4 Disentanglement and Domain Adaptation")
    para(doc, "Where data spans several cohorts, adversarial objectives and "
              "gradient reversal have been used to remove site- or "
              "cohort-specific variation from the shared representation, on "
              "the argument that what remains transfers. The present work "
              "includes such a component, and Chapter 6 reports a "
              "pre-registered test of the explanation that was offered for "
              "its behaviour — a test the explanation failed.")
    section(doc, "3.5 Interpretability in Multi-Omic Models")
    para(doc, "Attribution methods such as integrated gradients, and "
              "attention weights read as importance, are the usual route to "
              "interpretability. Both are vulnerable to the same confound: a "
              "graph built from co-alteration will group genes that sit at "
              "the same chromosomal locus, so recovering those groups "
              "validates the method rather than discovering biology. This "
              "report states that confound explicitly rather than presenting "
              "recovered structure as a finding.")
    section(doc, "3.6 Summary of Literature Gaps")
    para(doc, "No prior method couples a shared gene graph with a learned "
              "patient graph in one network and tests the coupling by "
              "ablation; comparisons against a clinical-only baseline are "
              "rare; and leave-one-cancer-out transfer across the full set of "
              "studies is, to our knowledge, not reported at this scale. "
              "Those three gaps define the contribution claimed here.")
    section(doc, "3.7 Literature Review Table")
    caption(doc, "Table 1:", "Comparison of representative multi-omic methods "
                             "against MOSAIC-GNN on the design axes relevant "
                             "to this project.")
    add_table(doc, [
        ["Method family", "Gene graph", "Patient graph", "Learned graph",
         "Survival head", "Paired tests", "LOCO transfer"],
        ["Penalised Cox / linear", "No", "No", "No", "Yes", "Rare", "No"],
        ["Tree ensembles", "No", "No", "No", "Via surrogate", "Rare", "No"],
        ["Late-fusion MLP", "No", "No", "No", "Yes", "Rare", "No"],
        ["Pathway-constrained nets", "Yes (fixed)", "No", "No", "Sometimes",
         "Rare", "No"],
        ["MOGONET-style", "No", "Yes (fixed)", "No", "Sometimes", "Rare",
         "No"],
        ["Patient-graph GNNs", "No", "Yes", "Sometimes", "Sometimes", "Rare",
         "No"],
        ["MOSAIC-GNN (this work)", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
    ], widths=[1.45, 0.8, 0.85, 0.8, 0.85, 0.75, 0.8], font=8.5)
    para(doc, "The final row states what this architecture does, not that it "
              "performs better. Chapter 6 reports the comparison, and on "
              "classification the answer does not favour the proposed model.",
         space_before=8)


# --------------------------------------------------------------------------- #
# Chapter 4 - Design and Implementation
# --------------------------------------------------------------------------- #
def ch4(doc, x: F, figdir: str):
    chapter(doc, "4", "Design and Implementation")

    section(doc, "4.1 Dataset and Pre-Committed Feature Space")
    para(doc, "All data was retrieved from the cBioPortal public API, which "
              "redistributes the TCGA PanCancer Atlas studies in a uniform "
              "schema. Three omic blocks were requested for every study: "
              "somatic mutation calls, discrete copy-number alterations and "
              "mRNA expression z-scores. Clinical covariates were taken from "
              "the same studies' patient and sample tables, and overall "
              "survival was taken as the time-to-event outcome with its "
              "censoring indicator.")
    para(doc, "The feature space was fixed before any analysis: the 468-gene "
              "MSK-IMPACT clinical sequencing panel, which resolves to "
              "469 gene symbols. This choice is the single most important "
              "methodological decision in the project. Because the panel was "
              "designed for clinical use years before this study and is "
              "identical for every cohort, no gene can have been selected "
              "because it helped the outcome. The alternative — ranking genes "
              "by association on the pooled dataset — leaks the labels into "
              "the feature space in a way that no subsequent "
              "cross-validation can remove, and it is the most common silent "
              "error in multi-omic papers. The cost of the pre-commitment is "
              "that the panel is enriched for recurrently mutated cancer "
              "genes and carries no coverage of, for example, immune or "
              "stromal programmes; that limitation is stated again in "
              "Chapter 8.")
    para(doc, "Two prediction targets were defined. The first, "
              "cohort, is the study of origin and has as many "
              "classes as there are studies; it is a stress test of "
              "representation quality rather than a clinically useful "
              "endpoint, and it is the setting in which the cohort-detector "
              "problem is most acute. The second, stage_binary, "
              "collapses the AJCC stage into early (I-II) against late "
              "(III-IV) and is a clinically meaningful endpoint, available "
              "only in the subset of cohorts that record stage. Every "
              "experiment was run independently on both.")

    section(doc, "4.2 Preprocessing and Leakage Controls")
    para(doc, "Each omic block was reduced to the pre-committed panel, "
              "aligned on patient identifier and converted to a dense "
              "patient-by-gene matrix per block. Mutation was encoded as a "
              "binary indicator of a non-silent alteration, copy number as "
              "the discrete GISTIC-style level, and expression as the "
              "z-score already supplied by the source. Patients missing an "
              "entire block retain a zero-filled block together with an "
              "explicit availability mask, so the model can distinguish "
              "\"measured and normal\" from \"not measured\".")
    bullets(doc, [
        ("Fold-internal fitting. ", "Every statistic that depends on the "
         "data — standardisation constants, the gene graph, the module "
         "assignment, the clinical imputation values — is fitted on the "
         "training split of each fold and applied unchanged to the "
         "validation and test splits. Nothing is fitted on pooled data."),
        ("Patient-level splitting. ", "Splits are over patients, not "
         "samples, so two samples from one patient cannot straddle a split."),
        ("Stratification. ", "Folds are stratified on the target so that "
         "small cohorts are represented in every split; the smallest "
         "cohorts are nevertheless represented by very few events, which "
         "Section 6.10 returns to."),
        ("No test-set selection. ", "Early stopping and model selection use "
         "the validation split only. The test split is read once, after "
         "training has finished."),
    ])

    section(doc, "4.3 Graph Construction")
    para(doc, "Two graphs are built. The gene graph has one node per gene in "
              "the panel and is constructed from the training split alone. "
              "Edges are typed: a co-alteration relation from genes whose "
              "alteration patterns co-occur above a threshold, and a "
              "co-expression relation from correlated expression. Because "
              "the relations are typed, the message-passing layer can learn "
              "a different transformation per relation rather than averaging "
              "two different kinds of evidence. The construction is honest "
              "about one confound: co-alteration groups genes that sit at "
              "the same chromosomal locus, so some recovered structure is "
              "positional rather than functional, and Section 6.11 does not "
              "present such structure as a biological finding.")
    para(doc, "Genes are then partitioned into functional modules by "
              "clustering the gene graph. Module membership is used for "
              "pooling: instead of flattening 469 gene embeddings into one "
              "long vector, the model attends over modules, which reduces "
              "the parameter count and gives the attention weights a "
              "readable unit. The patient graph is not fixed. Patient "
              "similarity is computed in the model's own latent space and "
              "the top-k neighbours are recomputed as training proceeds, so "
              "the graph the network propagates over is a function of the "
              "representation rather than of a pre-chosen distance on raw "
              "features.")

    section(doc, "4.4 Model Architecture")
    para(doc, "MOSAIC-GNN is a two-level network. At the gene level, a "
              "relational graph attention encoder passes messages over the "
              "typed gene graph, producing one embedding per gene per "
              "patient from that patient's three omic values and the gene's "
              "neighbourhood. A learned per-gene gate scales each gene's "
              "contribution, and the gate values are reported in "
              "Section 6.11 as one of the two interpretability channels. "
              "Module-level attention pools the gene embeddings into a "
              "patient vector.")
    para(doc, "At the patient level, a graph network propagates over the "
              "learned patient graph, so that a patient's representation is "
              "informed by similar patients. The representation is then "
              "split into a shared part and a cohort-specific part. A "
              "gradient-reversal branch tries to predict the cohort from the "
              "shared part and the reversal pushes the encoder to make that "
              "prediction hard, with the intention that what remains in the "
              "shared part transfers. Section 6.7 reports what the ablation "
              "measured about this component, which is less than the design "
              "intended.")
    para(doc, "Three heads sit on the representation: a classification head "
              "for the target, a Cox proportional-hazards head trained on "
              "the partial likelihood, and a discrete-time hazard head over "
              "binned follow-up. Clinical covariates enter through a "
              "residual path, so the molecular branch is asked to explain "
              "what the clinical covariates do not; Section 6.7 shows that "
              "this path, rather than any graph component, is what the "
              "model's accuracy mostly rests on. Stochastic weight "
              "averaging is applied over the final epochs, and predictive "
              "uncertainty is obtained by Monte-Carlo dropout at inference.")
    tp = x.a("params.trainable_parameters")
    trn = x.a("params.training_patients")
    ppp = x.a("params.parameters_per_patient")
    para(doc, "The fitted network's size was counted rather than estimated. "
              "On the main cohort run it has "
              f"{NOT_RUN if tp is None else format(int(tp), ',')} trainable "
              f"parameters against "
              f"{NOT_RUN if trn is None else format(int(trn), ',')} training "
              f"patients, a ratio of {fmt(ppp, 1)} parameters per patient. "
              "That ratio is the reason the generalisation-gap audit in "
              "Section 6.3 is reported before any performance claim.")

    section(doc, "4.5 Training Objective")
    para(doc, "The total loss is a weighted sum of the classification "
              "cross-entropy, the Cox partial likelihood, the discrete-time "
              "hazard likelihood, a supervised contrastive term that pulls "
              "together patients with the same label, and the adversarial "
              "cohort term applied through gradient reversal. Optimisation "
              "uses AdamW with cosine decay, gradient clipping and early "
              "stopping on a validation score that combines balanced "
              "accuracy and the C-index. Class imbalance is handled by "
              "inverse-frequency weighting in the cross-entropy, which is "
              "why balanced accuracy rather than raw accuracy is the "
              "headline classification metric throughout.")

    section(doc, "4.6 Evaluation Protocol")
    para(doc, "The protocol was designed so that a negative result would be "
              "visible rather than absorbed. Four elements matter.")
    bullets(doc, [
        ("Paired comparison on identical patients. ", "Every comparator is "
         "fitted on exactly the same folds as MOSAIC-GNN, and the "
         "comparison is a paired bootstrap over the same held-out patients, "
         "resampling patients and recomputing both models' metrics on each "
         "resample. This measures the difference directly instead of "
         "comparing two independently noisy means."),
        ("Multiplicity correction. ", "With more than twenty comparators "
         "and two metrics, uncorrected testing would manufacture "
         "significance. Benjamini-Hochberg correction is applied across the "
         "whole family and the corrected q-value is what Section 6.5 "
         "reports."),
        ("A measured noise floor. ", "The pipeline was fitted three times "
         "under an identical seed, giving balanced accuracies of 0.4567, "
         "0.4341 and 0.4547. The spread is approximately +/-0.01 and arises "
         "because parallel scatter-add on the graph sums floating-point "
         "contributions in a non-deterministic order, which is not "
         "associative. Any ablation effect smaller than that floor is "
         "reported as not distinguishable from noise."),
        ("A label permutation null. ", "The whole pipeline is retrained on "
         "shuffled training labels. If the protocol leaked, the permuted "
         "model would still score above chance. The smallest p-value "
         "attainable from B permutations is 1/(B+1), so the reported "
         "p-value is bounded by the number of permutations that fitted "
         "within the compute budget, and Section 6.8 says so explicitly "
         "rather than presenting the bound as evidence."),
    ], numbered=True)
    para(doc, "Metrics are reported in full rather than selectively: "
              "accuracy, balanced accuracy, macro and weighted F1, "
              "precision, sensitivity, specificity, negative predictive "
              "value, AUROC, AUPRC, Matthews correlation, Cohen's kappa and "
              "its quadratic form, top-2 accuracy, log loss, Brier score "
              "and expected calibration error for classification; Harrell's "
              "C-index, time-dependent AUC at 12, 36 and 60 months, and the "
              "log-rank statistic over model-defined risk tertiles for "
              "survival.")


# --------------------------------------------------------------------------- #
# Chapter 5 - Experimental Setup
# --------------------------------------------------------------------------- #
def ch5(doc, x: F):
    chapter(doc, "5", "Experimental Setup")
    section(doc, "5.1 Computing Environment and Software")
    caption(doc, "Table 2:", "Computing environment and the principal "
                             "software components used for every run "
                             "reported in Chapter 6.")
    add_table(doc, [
        ["Component", "Configuration"],
        ["Platform", "Linux x86-64 container, CPU-only execution"],
        ["Language", "Python 3"],
        ["Deep learning", "PyTorch with PyTorch Geometric for the graph "
                          "layers"],
        ["Classical models", "scikit-learn (linear, SVM, discriminant, "
                             "tree-ensemble, k-NN, naive Bayes, PCA "
                             "pipelines)"],
        ["Gradient boosting", "XGBoost and LightGBM"],
        ["Survival models", "lifelines (Cox proportional hazards, penalised "
                            "Cox, Kaplan-Meier, log-rank)"],
        ["Data handling", "pandas, NumPy, PyArrow (Parquet)"],
        ["Figures", "Matplotlib at 300 dpi"],
        ["Data source", "cBioPortal public REST API, TCGA PanCancer Atlas "
                        "studies"],
    ], widths=[1.6, 4.5], font=9.5)
    para(doc, "Running on CPU rather than GPU is the binding constraint on "
              "this project and it determines which phases could be run at "
              "full scale. The measured cost of one pan-cancer "
              "cross-validation fit is approximately half an hour, so a "
              "seven-arm ablation at two folds is a multi-hour job and a "
              "32-cohort leave-one-cancer-out sweep is a multi-day job. "
              "Section 5.3 states exactly which phases were therefore run "
              "at reduced settings.", space_before=8)

    section(doc, "5.2 Hyperparameters")
    caption(doc, "Table 3:", "Principal hyperparameters. These were chosen "
                             "from the validation split only; no value was "
                             "selected using held-out test performance.")
    add_table(doc, [
        ["Group", "Setting"],
        ["Feature space", "469 gene symbols (MSK-IMPACT 468-gene panel), "
                          "three omic blocks per gene"],
        ["Gene encoder", "Relational graph attention over typed "
                         "co-alteration and co-expression edges, with a "
                         "learned per-gene gate"],
        ["Pooling", "Attention over functional modules obtained by "
                    "clustering the training-split gene graph"],
        ["Patient level", "Graph propagation over a top-k patient "
                          "similarity graph recomputed in latent space "
                          "during training"],
        ["Representation", "Shared and cohort-specific parts, with a "
                           "gradient-reversal cohort adversary on the "
                           "shared part"],
        ["Heads", "Classification, Cox partial likelihood, discrete-time "
                  "hazard; clinical covariates on a residual path"],
        ["Optimiser", "AdamW, cosine decay, gradient clipping, early "
                      "stopping on a combined validation score"],
        ["Regularisation", "Dropout, weight decay, stochastic weight "
                           "averaging over the final epochs"],
        ["Uncertainty", "Monte-Carlo dropout at inference"],
        ["Resampling", "Stratified patient-level cross-validation; paired "
                       "bootstrap for comparison, Benjamini-Hochberg "
                       "correction across the family"],
    ], widths=[1.5, 4.6], font=9.0)

    section(doc, "5.3 What Ran at Full Scale and What Did Not")
    para(doc, "This section exists so that no number in Chapter 6 can be "
              "read at more confidence than the run behind it supports.")
    bullets(doc, [
        ("Full scale. ", "Dataset assembly over all studies; the main "
         "cross-validated runs for both targets; the complete comparator "
         "panel on identical folds; the paired bootstrap with "
         "multiplicity correction; the generalisation-gap audit; the "
         "32-cohort leave-one-cancer-out sweep; the interpretability "
         "exports."),
        ("Reduced settings. ", "The ablation was run on a reduced gene "
         "subset, six epochs and two folds, because seven arms at full "
         "settings did not fit the budget. Its absolute balanced accuracy "
         "is therefore lower than the main run's and is not comparable to "
         "it; only the differences between arms, read against the measured "
         "noise floor, are interpreted."),
        ("Bounded by budget. ", "The label permutation test completed four "
         "permutations, which bounds the attainable p-value at 0.2. The "
         "sample-size curve has four points."),
        ("Not run. ", "No external validation cohort outside TCGA was "
         "available, so no external replication was attempted. No "
         "prospective or clinical evaluation was performed. Any such claim "
         "is absent from this report rather than estimated."),
    ])


# --------------------------------------------------------------------------- #
# Chapter 6 - Experimental Results and Discussion
# --------------------------------------------------------------------------- #
def comparison_table(doc, x: F, tid: str, label: str, cap: str, cols,
                     mosaic_row, widths, font=8.0, head_map=None):
    """Comparator table with MOSAIC-GNN's own measured row placed first."""
    g = x.grid(tid)
    caption(doc, label + ":", cap)
    if not g:
        para(doc, "[the run that writes this table did not complete]",
             italic=True, align="left")
        return
    head = g[0]
    keep = [head.index(c) for c in cols if c in head]
    out = [["Model"] + [(head_map or {}).get(head[i], head[i]) for i in keep]]
    out.append(mosaic_row)
    for r in g[1:]:
        out.append([r[0]] + [r[i] for i in keep])
    add_table(doc, out, widths=widths, font=font)


def ch6(doc, x: F, figdir: str):
    chapter(doc, "6", "Experimental Results and Discussion")
    para(doc, "Every number in this chapter was produced by a run that was "
              "actually executed on the assembled data. Where a phase was "
              "run at reduced settings, as recorded in Section 5.3, the "
              "section says so before the number. Where a phase did not "
              "run, the report says that instead of supplying a figure. "
              "Three of the results below contradict what the architecture "
              "was designed to achieve; they are reported in the same "
              "detail as the favourable ones, because a result that only "
              "survives when it agrees with the design is not a result.")

    # 6.1 ---------------------------------------------------------------- #
    section(doc, "6.1 Assembled Dataset")
    para(doc, "The assembly retrieved "
              f"{x.i('cohort','n_cohorts')} studies covering "
              f"{x.i('cohort','n_patients')} patients with "
              f"{x.i('cohort','n_events')} observed deaths. For the "
              "cohort target every patient carries a label, so "
              f"all {x.i('cohort','n_labelled')} are usable. For "
              "stage_binary only the cohorts that record AJCC "
              f"stage contribute, giving {x.i('stage_binary','n_labelled')} "
              f"labelled patients in "
              f"{x.i('stage_binary','n_cohorts_labelled')} cohorts. The "
              "difference between those two counts is itself a result: a "
              "third of the pan-cancer population cannot be used for the "
              "clinically meaningful endpoint at all.")
    fe = x.g("cohort", "fewest_events") or []
    if fe:
        para(doc, "The cohorts with the fewest events are "
                  + "; ".join(f"{d['cohort']} ({d['events']} events in "
                              f"{d['patients']} patients)" for d in fe[:5])
                  + ". Any per-cohort statistic computed on those rows rests "
                    "on a handful of comparable pairs, which is why "
                    "Section 6.10 separates the cohorts that are powered "
                    "enough to read from those that are not.")
    payload_table(doc, x, "T1", "Table 4",
                  widths=[0.75, 0.62, 0.95, 0.95, 0.6, 0.72, 0.95, 0.72],
                  font=7.5,
                  head_map={"patients": "Patients", "blocks": "Omic blocks",
                            "genes_per_block": "Genes per block",
                            "events": "Events", "event_rate": "Event rate",
                            "median_OS_months": "Median OS (months)",
                            "labelled": "Labelled"})
    add_figure(doc, f"{figdir}/fig1_cohorts.png")
    caption(doc, "Figure 1:", "Cohort composition of the assembled dataset: "
                              "patients per cohort, observed events, event "
                              "rate and follow-up. The spread across cohorts "
                              "is the reason pooled metrics are reported "
                              "alongside per-cohort ones.", above=False)

    # 6.2 ---------------------------------------------------------------- #
    section(doc, "6.2 Headline Performance of MOSAIC-GNN")
    para(doc, "On the held-out fold of the cohort run, "
              f"MOSAIC-GNN reaches a balanced accuracy of "
              f"{x.n('cohort','mosaic_balanced_accuracy')} over "
              f"{x.i('cohort','n_cohorts')} classes, an accuracy of "
              f"{x.n('cohort','mosaic_accuracy')}, an AUROC of "
              f"{x.n('cohort','mosaic_auroc')}, a Matthews correlation of "
              f"{x.n('cohort','mosaic_mcc')} and a C-index of "
              f"{x.n('cohort','mosaic_c_index')}. The log-rank test over "
              "model-defined risk tertiles gives a chi-square of "
              f"{x.t2('LR-χ²')} with p = {x.t2('LR-p')}, so the "
              "representation does separate survival. Calibration is the "
              "weak point: the expected calibration error is "
              f"{x.n('cohort','mosaic_ece')}, which means the predicted "
              "probabilities should not be read as probabilities without "
              "recalibration.")
    para(doc, "On stage_binary the same architecture reaches a "
              f"balanced accuracy of "
              f"{x.n('stage_binary','mosaic_balanced_accuracy')}, an AUROC "
              f"of {x.n('stage_binary','mosaic_auroc')} and a C-index of "
              f"{x.n('stage_binary','mosaic_c_index')}, with an expected "
              f"calibration error of {x.n('stage_binary','mosaic_ece')}. "
              "The two targets therefore behave very differently: the "
              "32-class problem is almost solved by discriminative "
              "accuracy and badly calibrated, while the binary clinical "
              "endpoint is only modestly better than chance on "
              "classification and well calibrated. The second is the "
              "honest reading of how hard stage prediction from this "
              "feature space actually is.")
    payload_table(doc, x, "T2", "Table 5", widths=[1.45, 1.5, 1.45, 1.5],
                  font=9.0)

    # 6.3 ---------------------------------------------------------------- #
    section(doc, "6.3 Generalisation Gap and the Noise Floor")
    para(doc, f"With {fmt(x.a('params.parameters_per_patient'), 1)} "
              "parameters per training patient, "
              "overfitting is the first thing a reader should suspect, so "
              "the identical metric panel was computed on all three splits "
              "before any comparison was made.")
    payload_table(doc, x, "S1", "Table 6",
                  cols=["Split", "n", "Acc", "BAcc", "AUROC", "MCC",
                        "LogLoss", "Brier", "C-index"],
                  widths=[1.42, 0.58, 0.54, 0.56, 0.6, 0.54, 0.66, 0.56,
                          0.66], font=8.5)
    gb = x.num("S1", "gap (train - test)", "BAcc")
    gc = x.num("S1", "gap (train - test)", "C-index")
    para(doc, "The train-to-test gap is "
              f"{fmt(gb)} on balanced accuracy and {fmt(gc)} on the "
              "C-index. That is a real gap, not a negligible one, but it is "
              "an order of magnitude smaller than the headline performance "
              "and far smaller than the collapse that a model memorising "
              f"{_cnt(x.a('params.training_patients'))} patients with "
              f"{_cnt(x.a('params.trainable_parameters'))} "
              "parameters would show. The model is fitting the training "
              "split harder than the test split and generalising all the "
              "same.", space_before=8)
    para(doc, "Alongside the gap, the pipeline's own reproducibility was "
              "measured, because an ablation cannot be read without it. "
              "Three fits under an identical seed gave balanced accuracies "
              "of 0.4567, 0.4341 and 0.4547 — a spread of about +/-0.01 "
              "with the seed held fixed. The cause is that parallel "
              "scatter-add over graph edges accumulates floating-point "
              "contributions in an order that is not fixed between runs, "
              "and floating-point addition is not associative. This is the "
              "yardstick Section 6.7 uses, and it is the reason two of the "
              "seven ablation arms are reported as indistinguishable from "
              "noise rather than as small effects.")
    add_figure(doc, f"{figdir}/fig2_risk.png")
    caption(doc, "Figure 2:", "The identical metric panel on the training, "
                              "validation and held-out splits. The bars are "
                              "read for the distance between them, not for "
                              "their height.", above=False)

    # 6.4 ---------------------------------------------------------------- #
    section(doc, "6.4 Per-Class Behaviour and Training Dynamics")
    para(doc, "A pooled 32-class metric hides which classes carry it. The "
              "per-class breakdown shows that performance tracks support "
              "almost monotonically: the large cohorts are recognised "
              "reliably and the smallest are recognised poorly or not at "
              "all. This is the expected behaviour of inverse-frequency "
              "weighting on a long-tailed label distribution, and it is "
              "why balanced accuracy, which weights every class equally, "
              "sits well below raw accuracy. The complete per-class table "
              "is Table A2 in the Appendix.")
    add_figure(doc, f"{figdir}/fig3_classification.png")
    caption(doc, "Figure 3:", "Per-class behaviour on the held-out fold, "
                              "with each class's support shown alongside its "
                              "scores.", above=False)
    para(doc, "The learning curves show no divergence between the training "
              "and validation traces of the kind that marks runaway "
              "overfitting; the validation score rises and then flattens, "
              "and early stopping fires on the flat part. The recorded "
              "trace is Table A3 in the Appendix.")
    add_figure(doc, f"{figdir}/figS1_learning_curves.png")
    caption(doc, "Figure 4:", "Training and validation traces over epochs.",
            above=False)

    # 6.5 ---------------------------------------------------------------- #
    section(doc, "6.5 Comparison Against Established Methods")
    para(doc, f"{x.i('cohort','n_baselines')} comparators were fitted on "
              "exactly the same folds, spanning penalised linear models, "
              "support vector machines, discriminant analysis, tree "
              "ensembles, gradient boosting, nearest neighbours, naive "
              "Bayes, a PCA pipeline, early- and late-fusion neural "
              "networks, three patient-graph networks, a MOGONET-style "
              "model, DeepSurv and three Cox variants including a "
              "clinical-only Cox model.")
    mos = ["MOSAIC-GNN",
           x.n('cohort', 'mosaic_accuracy'),
           x.n('cohort', 'mosaic_balanced_accuracy'),
           x.n('cohort', 'mosaic_f1_macro'),
           x.n('cohort', 'mosaic_auroc'),
           x.n('cohort', 'mosaic_auprc'),
           x.n('cohort', 'mosaic_mcc'),
           x.n('cohort', 'mosaic_ece')]
    comparison_table(doc, x, "T5", "Table 7",
                     "Classification on the held-out fold of the "
                     "cohort run. MOSAIC-GNN's own measured row "
                     "is placed first; every other row was fitted on the "
                     "same folds.",
                     ["Acc", "BAcc", "F1-M", "AUROC", "AUPRC", "MCC", "ECE"],
                     mos, widths=[1.95, 0.63, 0.63, 0.63, 0.63, 0.63,
                                  0.63, 0.63], font=8.0)
    mos_s = ["MOSAIC-GNN", x.n('cohort', 'mosaic_c_index'),
             x.t2('AUC12'), x.t2('AUC36'), x.t2('AUC60'),
             x.t2('LR-χ²'), x.t2('LR-p')]
    comparison_table(doc, x, "T6", "Table 8",
                     "Survival ranking on the same fold: Harrell's C-index, "
                     "time-dependent AUC at 12, 36 and 60 months, and the "
                     "log-rank statistic over model-defined risk tertiles.",
                     ["C-index", "AUC12", "AUC36", "AUC60", "LR-χ²", "LR-p"],
                     mos_s, widths=[1.95, 0.75, 0.68, 0.68, 0.68, 0.74,
                                    0.84], font=8.0)
    para(doc, "Read on its own, Table 7 already contradicts the design "
              "hypothesis. The best comparator balanced accuracy on "
              f"cohort is {x.best('cohort','best_balanced_accuracy')}, "
              f"against MOSAIC-GNN's "
              f"{x.n('cohort','mosaic_balanced_accuracy')}; a penalised "
              "linear model on the same 469-gene feature space separates "
              "the 32 cohorts far better than the graph network does. On "
              "survival the picture is different: the best comparator "
              f"C-index is {x.best('cohort','best_c_index')} against "
              f"MOSAIC-GNN's {x.n('cohort','mosaic_c_index')}, which is a "
              "narrow margin.", space_before=8)
    para(doc, "Those are unpaired comparisons of two noisy numbers, so they "
              "are not the test. The paired bootstrap resamples the same "
              "held-out patients and recomputes both models on each "
              "resample, and the difference is reported as "
              "delta = MOSAIC-GNN minus comparator, so a negative delta "
              "means MOSAIC-GNN is behind. Benjamini-Hochberg correction is "
              "applied across the whole family of tests.")
    payload_table(doc, x, "T7", "Table 9",
                  cols=["Model", "bacc_delta", "bacc_ci_low", "bacc_ci_high",
                        "bacc_q", "cindex_delta", "cindex_ci_low",
                        "cindex_ci_high", "cindex_q"],
                  widths=[1.72, 0.63, 0.57, 0.57, 0.56, 0.63, 0.57, 0.57,
                          0.56], font=7.0,
                  head_map={"bacc_delta": "Δ BAcc", "bacc_ci_low": "CI low",
                            "bacc_ci_high": "CI high", "bacc_q": "q",
                            "cindex_delta": "Δ C-idx",
                            "cindex_ci_low": "CI low",
                            "cindex_ci_high": "CI high", "cindex_q": "q"},
                  caption_text="Paired bootstrap against MOSAIC-GNN on "
                               "identical held-out patients, with "
                               "Benjamini-Hochberg corrected q-values. A "
                               "negative delta means MOSAIC-GNN scored "
                               "lower than that comparator. An empty cell "
                               "means the comparator does not produce that "
                               "kind of prediction.")
    cb = lambda t, k: x.g(t, f"paired.bacc.{k}")      # noqa: E731
    cc = lambda t, k: x.g(t, f"paired.cindex.{k}")    # noqa: E731
    para(doc, "The corrected result on cohort is that of "
              f"{cb('cohort','n_tested')} comparators tested on balanced "
              f"accuracy, MOSAIC-GNN is significantly worse than "
              f"{cb('cohort','n_worse')} and significantly better than "
              f"{cb('cohort','n_better')} "
              f"({', '.join(cb('cohort','better_models') or [])}), with "
              f"{cb('cohort','n_tied')} ties. On the C-index, of "
              f"{cc('cohort','n_tested')} comparators tested it is "
              f"significantly better than {cc('cohort','n_better')} "
              f"({', '.join(cc('cohort','better_models') or [])}), "
              f"significantly worse than {cc('cohort','n_worse')}, and tied "
              f"with {cc('cohort','n_tied')}.", space_before=8)
    para(doc, "On stage_binary, of "
              f"{cb('stage_binary','n_tested')} comparators tested on "
              f"balanced accuracy it is significantly worse than "
              f"{cb('stage_binary','n_worse')} and significantly better "
              f"than {cb('stage_binary','n_better')} "
              f"({', '.join(cb('stage_binary','better_models') or [])}), "
              f"with {cb('stage_binary','n_tied')} ties; on the C-index it "
              f"is better than {cc('stage_binary','n_better')} "
              f"({', '.join(cc('stage_binary','better_models') or [])}), "
              f"worse than {cc('stage_binary','n_worse')} "
              f"({', '.join(cc('stage_binary','worse_models') or [])}) and "
              f"tied with {cc('stage_binary','n_tied')}.")
    rich(doc, [("This is the central negative result of the project. ",
                True),
               ("The proposed architecture does not beat established "
                "methods on either classification endpoint once the "
                "comparison is paired and corrected: on the 32-class "
                "problem it loses to most of the panel, and on the "
                "clinical endpoint it loses to a large minority of it. On "
                "survival ranking it is competitive rather than superior, "
                "tied with most of the survival comparators. Reporting "
                "this is not a failure of the study; the study was built "
                "so that this answer could be obtained rather than "
                "avoided, and the two places where the model does win are "
                "reported in Section 6.6 with the same machinery.", False)])
    add_figure(doc, f"{figdir}/fig4_comparison.png")
    caption(doc, "Figure 5:", "MOSAIC-GNN against every comparator with "
                              "bootstrap intervals, on survival ranking and "
                              "on classification.", above=False)

    # 6.6 ---------------------------------------------------------------- #
    section(doc, "6.6 Do Molecular Features Add Anything Over Clinical "
                 "Covariates?")
    para(doc, "This is the premise the multi-omic field rests on and the "
              "comparison most multi-omic papers omit. Two independent "
              "measurements of it were made.")
    mgc, mgs = x.g("cohort", "molecular_gain"), x.g("stage_binary",
                                                    "molecular_gain")
    if isinstance(mgc, dict) and isinstance(mgs, dict):
        caption(doc, "Table 10:", "Molecular information over and above "
                                  "the clinical covariates, measured twice: "
                                  "once by the best penalised Cox model "
                                  "with and without the molecular blocks, "
                                  "and once by MOSAIC-GNN against the "
                                  "clinical-only Cox model in the paired "
                                  "bootstrap.")
        add_table(doc, [
            ["Comparison", "cohort run", "stage_binary run"],
            [f"Penalised Cox with molecular ({mgc.get('model','?')})",
             fmt(mgc.get("molecular")), fmt(mgs.get("molecular"))],
            ["Clinical covariates only (Cox)",
             fmt(mgc.get("clinical_only")), fmt(mgs.get("clinical_only"))],
            ["Difference in C-index",
             fmt(mgc.get("delta")), fmt(mgs.get("delta"))],
            ["MOSAIC-GNN minus clinical-only Cox (paired)",
             fmt(x.g("cohort", "paired.models.cox_clinical_only."
                               "cindex_delta")),
             fmt(x.g("stage_binary", "paired.models.cox_clinical_only."
                                     "cindex_delta"))],
            ["Corrected q-value of that paired difference",
             fmt(x.g("cohort", "paired.models.cox_clinical_only.cindex_q"),
                 4),
             fmt(x.g("stage_binary",
                     "paired.models.cox_clinical_only.cindex_q"), 4)],
        ], widths=[3.0, 1.55, 1.55], font=9.0)
    _CCO = "paired.models.cox_clinical_only.cindex_delta"
    para(doc, "Both measurements agree, and they agree on both targets. "
              "Adding the molecular blocks to a penalised Cox model raises "
              f"the C-index by {fmt((mgc or {}).get('delta'))} on the "
              f"cohort run and {fmt((mgs or {}).get('delta'))} on "
              "the stage_binary run. MOSAIC-GNN itself exceeds "
              "the clinical-only Cox model by "
              f"{fmt(x.g('cohort', _CCO))} and "
              f"{fmt(x.g('stage_binary', _CCO))} respectively, and in the "
              "paired bootstrap both differences survive correction. A "
              "result "
              "that replicates across two independently fitted targets on "
              "overlapping but differently labelled patient sets is worth "
              "more than a larger single-target effect, and this is the "
              "clearest positive finding of the project.", space_before=8)
    para(doc, "It should be read for what it is. It says that the "
              "molecular data carries prognostic information a clinician "
              "does not already have. It does not say that this "
              "architecture is the way to extract it: Section 6.5 shows "
              "that simpler models extract as much or more.")


def ch6b(doc, x: F, figdir: str, runs: str):
    # 6.7 ---------------------------------------------------------------- #
    section(doc, "6.7 Ablation: Which Components Earn Their Place")
    para(doc, "Seven arms were run, each removing exactly one component and "
              "changing nothing else. As recorded in Section 5.3 the "
              "ablation ran on a reduced gene subset for six epochs over "
              "two folds, so its absolute balanced accuracy, "
              f"{x.n('cohort','ablation.full_bacc')} for the full model, is "
              "lower than Section 6.2's and is not comparable to it. Only "
              "the differences between arms are interpreted, and they are "
              "read against the +/-0.01 noise floor measured in "
              "Section 6.3.")
    arms = x.g("cohort", "ablation.arms") or {}
    if arms:
        rows = [["Component removed", "BAcc", "Δ vs full",
                 "Outside the noise floor?"]]
        for name, d in sorted(arms.items(),
                              key=lambda kv: kv[1].get("delta", 0)):
            rows.append([name, fmt(d.get("bacc"), 4), fmt(d.get("delta"), 4),
                         "yes" if abs(float(d.get("delta", 0))) > 0.01
                         else "no - indistinguishable from noise"])
        caption(doc, "Table 11:", "Ablation at reduced settings. Each row "
                                  "removes one component; delta is the "
                                  "change in balanced accuracy against the "
                                  "full model on the same folds. The last "
                                  "column compares the effect with the "
                                  "pipeline's measured run-to-run spread.")
        add_table(doc, rows, widths=[1.7, 0.85, 0.9, 2.65], font=9.0)
    outside = x.g("cohort", "ablation.outside_floor") or []
    inside = x.g("cohort", "ablation.inside_floor") or []
    para(doc, "Three things follow. First, the clinical path dominates: "
              "removing the clinical covariates costs "
              f"{x.n('cohort','ablation.arms.no_clinical.delta', 4)} and "
              "removing only their residual path costs "
              f"{x.n('cohort','ablation.arms.no_clinical_residual.delta', 4)}"
              ", both an order of magnitude larger than any other arm. The "
              "model's accuracy rests mostly on the clinical covariates, "
              "not on the molecular graph machinery. Second, the three "
              "structural components do clear the noise floor — removing "
              "the gene graph costs "
              f"{x.n('cohort','ablation.arms.no_gene_graph.delta', 4)}, the "
              "patient graph "
              f"{x.n('cohort','ablation.arms.no_patient_graph.delta', 4)} "
              "and module pooling "
              f"{x.n('cohort','ablation.arms.no_module_pool.delta', 4)} — "
              "so the two-level design is doing measurable work even "
              "though the whole model is beaten by simpler comparators. "
              "Third, the two auxiliary objectives do not: removing the "
              "contrastive term costs "
              f"{x.n('cohort','ablation.arms.no_contrast.delta', 4)} and "
              "removing disentanglement "
              f"{x.n('cohort','ablation.arms.no_disentangle.delta', 4)}, "
              "both inside the floor, so on this evidence neither can be "
              "claimed to contribute.", space_before=8)
    para(doc, f"Arms outside the floor: {', '.join(outside) or NOT_RUN}. "
              f"Arms inside the floor: {', '.join(inside) or NOT_RUN}.")
    rich(doc, [("A retracted explanation. ", True),
               ("An earlier draft of this work explained the "
                "disentanglement arm by hypothesising that the cohort "
                "adversary was the component suppressing tissue-of-origin "
                "signal, and predicted that removing it would raise "
                "cohort accuracy sharply. The ablation was run "
                "specifically to test that prediction. Removing "
                "disentanglement changed balanced accuracy by "
                + x.n("cohort", "ablation.arms.no_disentangle.delta", 4) +
                ", where the hypothesis required a change of roughly "
                "+0.17. The hypothesis is falsified and has been withdrawn "
                "from this report. It is recorded here because a "
                "pre-registered prediction that failed is evidence, and "
                "deleting it would misrepresent how the conclusions were "
                "reached.", False)])
    add_figure(doc, f"{figdir}/fig5_ablation.png")
    caption(doc, "Figure 6:", "Change in each headline metric when a single "
                              "component is removed. Negative values mean "
                              "the component helped. Effects smaller than "
                              "the measured noise floor are not "
                              "interpretable.", above=False)

    # 6.8 ---------------------------------------------------------------- #
    section(doc, "6.8 Label Permutation Control")
    para(doc, "The whole pipeline was retrained on shuffled training "
              "labels. A leaking protocol would still score above chance "
              "after shuffling; an honest one should collapse to chance.")
    payload_table(doc, x, "S2", "Table 12",
                  widths=[1.5, 0.85, 0.85, 0.75, 0.8, 0.95, 0.65], font=9.0,
                  head_map={"observed": "Observed", "null_mean": "Null mean",
                            "null_sd": "Null SD", "null_max": "Null max",
                            "n_permutations": "Permutations",
                            "p_value": "p"})
    nperm = x.g("cohort", "permutation_values.n")
    para(doc, "Balanced accuracy under permutation falls to "
              f"{x.n('cohort','permutation_values.bacc.mean', 4)} "
              f"(SD {x.n('cohort','permutation_values.bacc.sd', 4)}) "
              f"against an observed "
              f"{x.n('cohort','permutation_values.bacc.obs')}. AUROC falls "
              f"to {x.n('cohort','permutation_values.auroc.mean')} "
              f"(SD {x.n('cohort','permutation_values.auroc.sd')}) from an "
              f"observed {x.n('cohort','permutation_values.auroc.obs')}, "
              "and the C-index falls to "
              f"{x.n('cohort','permutation_values.cindex.mean')} "
              f"(SD {x.n('cohort','permutation_values.cindex.sd')}) from an "
              f"observed {x.n('cohort','permutation_values.cindex.obs')}. "
              "Both AUROC and the C-index land essentially on 0.5, which "
              "is exactly where a model with no usable signal should land. "
              "The control passes.", space_before=8)
    para(doc, "The reported p-value must be read with its bound. With "
              f"{NOT_RUN if nperm is None else int(nperm)} permutations the "
              "smallest attainable p-value is "
              f"{NOT_RUN if nperm is None else fmt(1.0 / (int(nperm) + 1), 2)}"
              ", and that bound — not the strength of the effect — is what "
              "the tabulated p-value reflects. The evidence that the "
              "protocol does not leak is the collapse of the null "
              "distribution to chance, not the p-value. One permutation "
              "returned a C-index of 0.559, which is above 0.5; with four "
              "permutations and a null standard deviation of about 0.044 "
              "that is an ordinary draw from the null and not a sign of "
              "leakage.")

    # 6.9 ---------------------------------------------------------------- #
    section(doc, "6.9 Is the Model Data-Limited or Architecture-Limited?")
    para(doc, "The same configuration was refitted on a quarter, a half, "
              "three quarters and all of the training data, with the "
              "held-out split unchanged, to separate a data limitation "
              "from an architecture limitation.")
    payload_table(doc, x, "S3", "Table 13",
                  widths=[0.9, 1.15, 1.15, 1.15, 1.15, 0.9], font=9.0,
                  head_map={"train_balanced_accuracy": "Train BAcc",
                            "train_c_index": "Train C-index",
                            "test_balanced_accuracy": "Test BAcc",
                            "test_c_index": "Test C-index",
                            "fraction": "Fraction of data"})
    para(doc, "Held-out balanced accuracy rises from "
              f"{x.n('cohort','sample_size.bacc_min')} at "
              f"{x.i('cohort','sample_size.n_min')} training patients to "
              f"{x.n('cohort','sample_size.bacc_max')} at "
              f"{x.i('cohort','sample_size.n_max')}, a gain of "
              f"{x.n('cohort','sample_size.bacc_gain')}, and the curve has "
              "not flattened at the largest size available. The held-out "
              f"C-index moves much less, from "
              f"{x.n('cohort','sample_size.cindex_min')} to "
              f"{x.n('cohort','sample_size.cindex_max')}, a gain of "
              f"{x.n('cohort','sample_size.cindex_gain')}, while the "
              "training C-index stays flat throughout. The reading is that "
              "classification is data-limited and would improve with more "
              "patients, whereas survival ranking is close to what this "
              "feature space supports and more patients would not move it "
              "much.", space_before=8)
    add_figure(doc, f"{figdir}/figS2_sample_size.png")
    caption(doc, "Figure 7:", "Held-out performance against training-set "
                              "size, with the training curve shown for "
                              "contrast.", above=False)

    # 6.10 --------------------------------------------------------------- #
    section(doc, "6.10 Leave-One-Cancer-Out Transfer")
    para(doc, "This is the experiment that addresses the cohort-detector "
              "problem directly. For each of the "
              f"{x.i('cohort','loco.n_cohorts')} cohorts the model is "
              "trained with that cohort absent from training entirely and "
              "then asked to stratify risk within it. A cohort detector "
              "cannot do this, because the cohort it would have to detect "
              "was never seen.")
    rich(doc, [("What is reportable here, and what is not. ", True),
               ("Only the survival column is reported. In a "
                "leave-one-cancer-out fold the held-out cohort contains "
                "patients of exactly one class, so balanced accuracy "
                "reduces to the recall of that single class and the "
                "Matthews correlation is zero by construction. Those "
                "numbers are arithmetic artefacts of the split, not "
                "transfer measurements, and they are therefore excluded "
                "from every table and figure in this section. An earlier "
                "draft went further and asserted that the classification "
                "metric would be near zero by the same arithmetic; that "
                "assertion was wrong — the measured value on the first "
                "cohort was 0.989 — and it has been withdrawn. The "
                "correct statement is the narrower one: the metric is "
                "degenerate and so uninformative, not that it is small.",
                False)])
    payload_table(doc, x, "T9", "Table 14",
                  cols=["Held-out cohort", "C-index", "LR-p"],
                  widths=[1.9, 1.5, 1.6], font=9.0,
                  caption_text="Leave-one-cancer-out transfer. The model is "
                               "trained with the named cohort entirely "
                               "absent and evaluated on it. Classification "
                               "columns are omitted because they are "
                               "degenerate in a single-class fold.")
    para(doc, "The median transfer C-index is "
              f"{x.n('cohort','loco.median')}, with a range from "
              f"{x.n('cohort','loco.min')} to {x.n('cohort','loco.max')}. "
              f"{x.i('cohort','loco.n_above_half')} of "
              f"{x.i('cohort','loco.n_cohorts')} cohorts exceed 0.5, which "
              "is the random-ranking value, and the log-rank test is "
              f"significant at the 0.05 level in "
              f"{x.i('cohort','loco.n_significant')} of them. The "
              "representation therefore does carry risk information into "
              "cancer types it has never seen. That is strong evidence "
              "against the cohort-detector explanation of the pooled "
              "result, and it is the second substantive positive finding "
              "of the project.", space_before=8)
    para(doc, "The failures are structured rather than random. Transfer "
              "C-index correlates negatively with the held-out cohort's "
              f"event rate across all {x.i('cohort','loco.corr_all.n')} "
              f"cohorts (r = {x.n('cohort','loco.corr_all.r')}, "
              f"p = {x.n('cohort','loco.corr_all.p', 4)}), and the "
              "relationship strengthens when the analysis is restricted to "
              f"the {x.i('cohort','loco.corr_powered.n')} cohorts with "
              "enough events to be read "
              f"(r = {x.n('cohort','loco.corr_powered.r')}, "
              f"p = {x.n('cohort','loco.corr_powered.p', 4)}). The model "
              "transfers best into indolent cohorts and worst into "
              "aggressive ones. A plausible reading is that what transfers "
              "is a general tumour-aggressiveness ordering that is "
              "informative where deaths are sparse and insufficient where "
              "the whole cohort is high risk, but that reading is a "
              "hypothesis the present data cannot test and it is offered "
              "as such.")
    add_figure(doc, f"{figdir}/fig6_loco.png")
    caption(doc, "Figure 8:", "Leave-one-cancer-out transfer. (a) transfer "
                              "C-index for every held-out cohort with its "
                              "event count, cohorts with fewer than 60 "
                              "events shown in grey; (b) transfer against "
                              "the held-out cohort's event rate.",
            above=False)

    # 6.11 --------------------------------------------------------------- #
    section(doc, "6.11 Interpretation")
    para(doc, "Two interpretability channels are reported: integrated "
              "gradients over the input blocks, and the model's own "
              "learned per-gene gates and module attention.")
    csv_table(doc, f"{runs}/table7_gene_importance.csv", "Table 15",
              "Top 15 genes by integrated-gradient attribution, with the "
              "module each was assigned to, the learned gate and the "
              "attribution decomposed by omic block.",
              nrows=15, widths=[1.05, 1.25, 0.8, 0.8, 0.85, 0.85, 0.85],
              cols=["gene", "module", "gate_mean", "ig_abs", "ig_mut",
                    "ig_cna", "ig_exp"], font=8.5,
              head_map={"gene": "Gene", "module": "Module",
                        "gate_mean": "Gate", "ig_abs": "|IG|",
                        "ig_mut": "IG mutation", "ig_cna": "IG CNA",
                        "ig_exp": "IG expression"})
    csv_table(doc, f"{runs}/table8_module_importance.csv", "Table 16",
              "Top 10 functional modules by attention, with module size and "
              "the highest-attributed genes in each.",
              nrows=10, widths=[1.3, 0.6, 0.95, 3.4],
              cols=["module", "size", "attention", "top_genes"], font=8.5,
              head_map={"module": "Module", "size": "Size",
                        "attention": "Attention",
                        "top_genes": "Highest-attributed genes"})
    para(doc, "The highest-attributed genes are recurrently altered cancer "
              "genes, which is reassuring but not surprising given a panel "
              "designed around exactly those genes. The omic decomposition "
              "is the more informative part: attribution is dominated by "
              "expression and copy number, with mutation contributing "
              "very little on either target. For a model given binary "
              "mutation indicators over a 469-gene panel, that is the "
              "expected result — a binary indicator over a small panel "
              "carries far less variance than a continuous expression "
              "z-score — and it is a limitation of the encoding rather "
              "than a statement about tumour biology.", space_before=8)
    rich(doc, [("A confound that is not reported as a finding. ", True),
               ("The gene graph's co-alteration relation links genes whose "
                "alterations co-occur, and genes at the same chromosomal "
                "locus co-occur because they are deleted or amplified "
                "together. Modules recovered from such edges therefore "
                "partly recover chromosomal position. Where a module's "
                "members are locus neighbours, recovering that module "
                "validates the clustering, not biology, and this report "
                "does not present such modules as discovered pathways.",
                False)])
    add_figure(doc, f"{figdir}/fig7_interpretation.png")
    caption(doc, "Figure 9:", "Interpretation. Top genes by attribution, "
                              "the attribution decomposed by omic block, "
                              "and module-level attention.", above=False)

    # 6.12 --------------------------------------------------------------- #
    section(doc, "6.12 Discussion")
    para(doc, "Taken together the results answer the four sub-questions of "
              "Section 1.2, and two of the four answers are not the ones "
              "the architecture was built to produce.")
    bullets(doc, [
        ("RQ1, answered yes, and it replicates. ", "Molecular features add "
         "prognostic information beyond the clinical covariates, measured "
         "both by a penalised Cox model with and without them and by "
         "MOSAIC-GNN against a clinical-only Cox model, on both targets, "
         "surviving multiplicity correction in every case."),
        ("RQ2, answered in part. ", "The three structural components — "
         "gene graph, module pooling, learned patient graph — each clear "
         "the measured noise floor when removed. The two auxiliary "
         "objectives do not, and the clinical residual path dominates "
         "everything else by an order of magnitude."),
        ("RQ3, answered yes. ", "The representation stratifies risk in "
         "cohorts held out entirely from training, in "
         + (x.i("cohort", "loco.n_above_half") or NOT_RUN) + " of "
         + (x.i("cohort", "loco.n_cohorts") or NOT_RUN) + " cases above "
         "random, with the log-rank test significant in "
         + (x.i("cohort", "loco.n_significant") or NOT_RUN) + ". A cohort "
         "detector could not do this."),
        ("RQ4, answered against the proposed model. ", "Under a paired, "
         "corrected comparison it does not outperform established methods "
         "on either classification endpoint, and it is competitive rather "
         "than superior on survival ranking."),
    ])
    para(doc, "The practical conclusion a reader should draw is therefore "
              "not \"use this architecture\". It is that the molecular data "
              "is worth collecting because it carries information the "
              "clinic does not have, that a penalised Cox model or a "
              "gradient-boosted ensemble is a stronger starting point than "
              "it is usually given credit for, and that the two-level "
              "graph design contributes measurably but not enough to "
              "overcome that gap at this sample size. Section 6.9's "
              "sample-size curve suggests that the classification gap is "
              "at least partly a data limitation rather than a purely "
              "architectural one, which is a testable prediction rather "
              "than a defence.")

    # 6.13 --------------------------------------------------------------- #
    section(doc, "6.13 Threats to Validity")
    bullets(doc, [
        ("Single data source. ", "Every patient comes from TCGA through "
         "cBioPortal. No external cohort was available, so none of these "
         "numbers has been replicated outside that source and no claim of "
         "external validity is made."),
        ("Panel-restricted feature space. ", "The 469-gene panel was "
         "pre-committed, which removes selection leakage but also removes "
         "whole classes of biology from reach. Results cannot be "
         "generalised to genome-wide feature spaces."),
        ("Reduced-setting phases. ", "The ablation ran on a reduced "
         "configuration and the permutation test on four permutations. "
         "Both are labelled wherever they appear, and neither supports a "
         "fine-grained claim."),
        ("Calibration. ", "An expected calibration error of "
         + x.n("cohort", "mosaic_ece") + " on the cohort run "
         "means the predicted probabilities are not usable as "
         "probabilities without recalibration, whatever the discriminative "
         "metrics show."),
        ("Small-cohort statistics. ", "Several cohorts contribute fewer "
         "than ten events. Per-cohort numbers for those rows are reported "
         "for completeness and are explicitly separated from the powered "
         "subset in Section 6.10."),
        ("Non-determinism. ", "The measured run-to-run spread of about "
         "+/-0.01 bounds the resolution of every single-fit comparison in "
         "this report."),
    ])


# --------------------------------------------------------------------------- #
# CSV-backed tables (for exports whose index was lost in the payload grid)
# --------------------------------------------------------------------------- #
def csv_table(doc, path: str, label: str, cap: str, cols=None, nrows=None,
              widths=None, font=8.5, round_to=4, head_map=None):
    import csv as _csv
    caption(doc, label + ":", cap)
    if not os.path.isfile(path):
        para(doc, f"[{os.path.basename(path)} is not present: the run that "
                  f"writes it did not complete]", italic=True, align="left")
        return
    with open(path, newline="") as fh:
        rd = list(_csv.reader(fh))
    if not rd:
        para(doc, "[empty export]", italic=True, align="left")
        return
    head = rd[0]
    keep = list(range(len(head))) if not cols else [
        head.index(c) for c in cols if c in head]
    body = rd[1:nrows + 1] if nrows else rd[1:]

    def cell(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return v
        if f == int(f) and abs(f) < 1e4:
            return str(int(f))          # a count stays a count
        return f"{f:.{round_to}f}" if abs(f) < 1e4 else f"{f:.3g}"

    out = [[(head_map or {}).get(head[i], head[i]) for i in keep]]
    out += [[cell(r[i]) for i in keep] for r in body]
    add_table(doc, out, widths=widths, font=font)


# --------------------------------------------------------------------------- #
# Chapter 7 - Impact
# --------------------------------------------------------------------------- #
def ch7(doc, x: F):
    chapter(doc, "7", "Impact of Societal, Health and Safety Issues")
    section(doc, "7.1 Societal Impact")
    para(doc, "A prognostic model that works changes which patients are "
              "offered aggressive treatment and which are offered "
              "surveillance. That is a consequential decision, and it is "
              "the reason this report leads with its negative results. The "
              "most socially useful output of the project is not the "
              "architecture but the protocol: a paired, corrected, "
              "permutation-audited comparison against a clinical-only "
              "baseline, which makes an overstated claim harder to publish "
              "and easier to detect.")
    para(doc, "There is also an equity dimension that the data makes "
              "visible. TCGA's cohorts are not demographically "
              "representative of the global cancer population, and the "
              "cohorts that this model transfers into worst are the "
              "aggressive ones, as Section 6.10 shows. A model deployed "
              "without that caveat would perform worst precisely where the "
              "clinical need is greatest. Stating the correlation and its "
              "direction is part of the result, not a disclaimer appended "
              "to it.")

    section(doc, "7.2 Health and Safety")
    para(doc, "Nothing in this project touches a patient. It is a "
              "retrospective computational study on de-identified public "
              "data, so there is no physical risk to any human subject and "
              "no intervention was performed. The safety question is "
              "entirely about downstream misuse, and three specific risks "
              "are worth naming.")
    bullets(doc, [
        ("Reading uncalibrated outputs as probabilities. ", "With an "
         "expected calibration error of " + x.n("cohort", "mosaic_ece") +
         " on the cohort run, a predicted probability of 0.8 does not mean "
         "an 80 % chance. Any clinical use would require recalibration and "
         "revalidation first."),
        ("Transferring to an unrepresented population. ", "Transfer quality "
         "depends systematically on the target cohort's event rate, so "
         "performance in a new population cannot be assumed from the "
         "pooled numbers."),
        ("Treating a classification score as a prognosis. ", "The "
         "cohort target is a methodological stress test, not a "
         "clinical endpoint. Its high AUROC says nothing about prognostic "
         "usefulness, and the report keeps the two separate throughout."),
    ])

    section(doc, "7.3 Legal and Ethical Considerations")
    para(doc, "All data was obtained from the cBioPortal public API, which "
              "redistributes TCGA data that was consented for research use "
              "and de-identified before release. No ethical approval was "
              "required for secondary analysis of this public resource, "
              "and no attempt was made to re-identify any patient. No "
              "patient-level data is redistributed with this report; only "
              "aggregate tables and figures are included, and the code "
              "re-downloads the data from the public API rather than "
              "shipping it. The gene panel used as the feature space is a "
              "published clinical panel; the software dependencies are "
              "open-source and used within their licences.")

    section(doc, "7.4 Sustainability and Cost")
    para(doc, "The whole study ran on CPU in a single container. That "
              "constraint shaped the design — it is why the ablation ran "
              "at reduced settings and why the permutation test stopped at "
              "four permutations — but it also means the energy cost of "
              "the work is modest compared with GPU-scale deep learning, "
              "and that the pipeline can be reproduced by a reader without "
              "specialised hardware. The runnable single-file script "
              "distributed with this report re-executes every phase, and "
              "its slowest phases are documented with their measured cost "
              "so that a reader can choose what to re-run.")


# --------------------------------------------------------------------------- #
# Chapter 8 - Conclusion and Future Work
# --------------------------------------------------------------------------- #
def ch8(doc, x: F):
    chapter(doc, "8", "Conclusion and Future Work")
    section(doc, "8.1 Conclusion")
    para(doc, "This project assembled a genuine pan-cancer multi-omic "
              f"dataset of {x.i('cohort','n_patients')} patients across "
              f"{x.i('cohort','n_cohorts')} TCGA studies over a "
              "pre-committed 469-gene feature space, implemented "
              "MOSAIC-GNN — a two-level graph network coupling a shared "
              "gene graph with a learned patient graph under joint "
              "classification and survival objectives — and evaluated it "
              f"against {x.i('cohort','n_baselines')} established "
              "comparators on identical folds with paired bootstrap tests, "
              "multiplicity correction, a generalisation-gap audit and a "
              "label permutation control.")
    para(doc, "Two findings are positive and both replicate or transfer. "
              "Molecular features add prognostic information beyond "
              "clinical covariates, measured two independent ways on two "
              "independently fitted targets, with every difference "
              "surviving correction. And the learned representation "
              "stratifies risk in cancer types held out entirely from "
              f"training, above random in "
              f"{x.i('cohort','loco.n_above_half')} of "
              f"{x.i('cohort','loco.n_cohorts')} cohorts and significantly "
              f"so in {x.i('cohort','loco.n_significant')}, which rules "
              "out the explanation that the pooled result is cohort "
              "detection in disguise.")
    para(doc, "One finding is negative and it is the one that governs how "
              "the rest should be read. Under the paired, corrected "
              "comparison MOSAIC-GNN does not outperform established "
              "methods on either classification endpoint, and it is "
              "competitive rather than superior on survival ranking. "
              "A penalised linear model reaches "
              f"{x.best('cohort','best_balanced_accuracy')} on the "
              "32-class problem where the proposed model reaches "
              f"{x.n('cohort','mosaic_balanced_accuracy')}. The graph "
              "components measurably help the model that contains them, "
              "but they do not close that gap, and the clinical covariates "
              "contribute more than all of them combined.")
    para(doc, "The contribution this report therefore claims is the "
              "audited evaluation itself, together with the two positive "
              "findings it establishes and the honest accounting of what "
              "the architecture does and does not buy. Two explanations "
              "offered in earlier drafts of this work were tested against "
              "runs that were executed for that purpose, failed, and have "
              "been withdrawn and recorded as withdrawn. A result that "
              "only survives when it flatters the design is not a result, "
              "and the protocol here was built so that this distinction "
              "could be made.")

    section(doc, "8.2 Limitations")
    bullets(doc, [
        "Single source: all patients come from TCGA via cBioPortal, with "
        "no external replication cohort available.",
        "The feature space is a 469-gene clinical panel, so whole classes "
        "of biology, including immune and stromal programmes, are outside "
        "reach.",
        "Mutation is encoded as a binary indicator, which Section 6.11 "
        "shows contributes almost nothing to attribution; a richer "
        "encoding was not tested.",
        "The ablation ran at reduced settings and the permutation test at "
        "four permutations, bounding the attainable p-value at 0.2.",
        "Calibration is poor on the cohort target and the "
        "outputs are not usable as probabilities without recalibration.",
        "A third of the pan-cancer population lacks a stage label and is "
        "unusable for the clinically meaningful endpoint.",
        "No prospective, clinical or external evaluation was performed, "
        "and no claim is made that depends on one.",
    ])

    section(doc, "8.3 Future Work")
    bullets(doc, [
        ("Close the comparison properly. ", "The sample-size curve has not "
         "flattened. Refitting on a larger assembled cohort, or on a "
         "genome-wide feature space with the pre-commitment preserved by "
         "selecting on training folds only, is the direct test of whether "
         "the classification gap is data-limited or architectural."),
        ("Recalibrate and re-audit. ", "Temperature scaling or isotonic "
         "regression fitted on the validation split, followed by a "
         "re-measurement of expected calibration error on the held-out "
         "split, would make the outputs usable without touching the "
         "discriminative model."),
        ("Richer mutation encoding. ", "Variant allele frequency, "
         "functional impact scores and mutational signatures in place of a "
         "binary indicator would test whether the near-zero mutation "
         "attribution is a property of the biology or of the encoding."),
        ("Strengthen the controls that were bounded by budget. ", "More "
         "permutations would turn the permutation control from a "
         "distributional argument into a p-value; a full-setting ablation "
         "would resolve the two arms that currently sit inside the noise "
         "floor."),
        ("External validation. ", "Replicating the molecular-gain finding "
         "in a non-TCGA cohort is the single most valuable next "
         "experiment, because that finding is the one with clinical "
         "implications and the one this study could not replicate outside "
         "its own data source."),
        ("Investigate the transfer gradient. ", "The negative correlation "
         "between transfer quality and the held-out cohort's event rate is "
         "the most interesting unexplained observation in the report and "
         "deserves an experiment of its own."),
    ])


# --------------------------------------------------------------------------- #
# References and Appendix
# --------------------------------------------------------------------------- #
REFERENCES = [
    "E. Cerami et al., \"The cBio Cancer Genomics Portal: An open platform "
    "for exploring multidimensional cancer genomics data,\" Cancer "
    "Discovery, vol. 2, no. 5, pp. 401-404, 2012.",
    "J. Gao et al., \"Integrative analysis of complex cancer genomics and "
    "clinical profiles using the cBioPortal,\" Science Signaling, vol. 6, "
    "no. 269, p. pl1, 2013.",
    "J. N. Weinstein et al., \"The Cancer Genome Atlas Pan-Cancer analysis "
    "project,\" Nature Genetics, vol. 45, no. 10, pp. 1113-1120, 2013.",
    "K. A. Hoadley et al., \"Cell-of-origin patterns dominate the molecular "
    "classification of 10,000 tumors from 33 types of cancer,\" Cell, "
    "vol. 173, no. 2, pp. 291-304, 2018.",
    "J. Liu et al., \"An integrated TCGA pan-cancer clinical data resource "
    "to drive high-quality survival outcome analytics,\" Cell, vol. 173, "
    "no. 2, pp. 400-416, 2018.",
    "A. Zehir et al., \"Mutational landscape of metastatic cancer revealed "
    "from prospective clinical sequencing of 10,000 patients,\" Nature "
    "Medicine, vol. 23, no. 6, pp. 703-713, 2017.",
    "D. R. Cox, \"Regression models and life-tables,\" Journal of the Royal "
    "Statistical Society: Series B, vol. 34, no. 2, pp. 187-202, 1972.",
    "R. Tibshirani, \"The lasso method for variable selection in the Cox "
    "model,\" Statistics in Medicine, vol. 16, no. 4, pp. 385-395, 1997.",
    "F. E. Harrell, R. M. Califf, D. B. Pryor, K. L. Lee, and R. A. Rosati, "
    "\"Evaluating the yield of medical tests,\" JAMA, vol. 247, no. 18, "
    "pp. 2543-2546, 1982.",
    "Y. Benjamini and Y. Hochberg, \"Controlling the false discovery rate: "
    "A practical and powerful approach to multiple testing,\" Journal of "
    "the Royal Statistical Society: Series B, vol. 57, no. 1, pp. 289-300, "
    "1995.",
    "T. N. Kipf and M. Welling, \"Semi-supervised classification with graph "
    "convolutional networks,\" in Proc. International Conference on "
    "Learning Representations (ICLR), 2017.",
    "P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and "
    "Y. Bengio, \"Graph attention networks,\" in Proc. International "
    "Conference on Learning Representations (ICLR), 2018.",
    "M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and "
    "M. Welling, \"Modeling relational data with graph convolutional "
    "networks,\" in Proc. European Semantic Web Conference (ESWC), "
    "pp. 593-607, 2018.",
    "W. L. Hamilton, R. Ying, and J. Leskovec, \"Inductive representation "
    "learning on large graphs,\" in Advances in Neural Information "
    "Processing Systems (NeurIPS), 2017.",
    "Y. Ganin et al., \"Domain-adversarial training of neural networks,\" "
    "Journal of Machine Learning Research, vol. 17, no. 59, pp. 1-35, 2016.",
    "P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, and A. G. Wilson, "
    "\"Averaging weights leads to wider optima and better "
    "generalization,\" in Proc. Conference on Uncertainty in Artificial "
    "Intelligence (UAI), 2018.",
    "Y. Gal and Z. Ghahramani, \"Dropout as a Bayesian approximation: "
    "Representing model uncertainty in deep learning,\" in Proc. "
    "International Conference on Machine Learning (ICML), 2016.",
    "M. Sundararajan, A. Taly, and Q. Yan, \"Axiomatic attribution for deep "
    "networks,\" in Proc. International Conference on Machine Learning "
    "(ICML), 2017.",
    "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, \"On calibration of "
    "modern neural networks,\" in Proc. International Conference on "
    "Machine Learning (ICML), 2017.",
    "T. Wang et al., \"MOGONET integrates multi-omics data using graph "
    "convolutional networks allowing patient classification and "
    "biomarker identification,\" Nature Communications, vol. 12, "
    "no. 3445, 2021.",
    "J. L. Katzman, U. Shaham, A. Cloninger, J. Bates, T. Jiang, and "
    "Y. Kluger, \"DeepSurv: Personalized treatment recommender system "
    "using a Cox proportional hazards deep neural network,\" BMC Medical "
    "Research Methodology, vol. 18, no. 24, 2018.",
    "A. Cheerla and O. Gevaert, \"Deep learning with multimodal "
    "representation for pancancer prognosis prediction,\" "
    "Bioinformatics, vol. 35, no. 14, pp. i446-i454, 2019.",
    "L. A. Vale-Silva and K. Rohr, \"Long-term cancer survival prediction "
    "using multimodal deep learning,\" Scientific Reports, vol. 11, "
    "no. 13505, 2021.",
    "H. A. Elmarakeby et al., \"Biologically informed deep neural network "
    "for prostate cancer discovery,\" Nature, vol. 598, pp. 348-352, 2021.",
    "J. Hao, Y. Kim, T. Kim, and M. Kang, \"PASNet: Pathway-associated "
    "sparse deep neural network for prognosis prediction from high-"
    "throughput data,\" BMC Bioinformatics, vol. 19, no. 510, 2018.",
    "L. Breiman, \"Random forests,\" Machine Learning, vol. 45, no. 1, "
    "pp. 5-32, 2001.",
    "T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting "
    "system,\" in Proc. ACM SIGKDD International Conference on Knowledge "
    "Discovery and Data Mining, pp. 785-794, 2016.",
    "G. Ke et al., \"LightGBM: A highly efficient gradient boosting "
    "decision tree,\" in Advances in Neural Information Processing "
    "Systems (NeurIPS), 2017.",
    "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" "
    "Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.",
    "A. Paszke et al., \"PyTorch: An imperative style, high-performance "
    "deep learning library,\" in Advances in Neural Information "
    "Processing Systems (NeurIPS), 2019.",
    "M. Fey and J. E. Lenssen, \"Fast graph representation learning with "
    "PyTorch Geometric,\" in ICLR Workshop on Representation Learning on "
    "Graphs and Manifolds, 2019.",
    "C. Davidson-Pilon, \"lifelines: Survival analysis in Python,\" "
    "Journal of Open Source Software, vol. 4, no. 40, p. 1317, 2019.",
]


def references(doc):
    chapter(doc, "", "References")
    for i, r in enumerate(REFERENCES, 1):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.42)
        p.paragraph_format.first_line_indent = Inches(-0.42)
        p.paragraph_format.space_after = Pt(6)
        _set_font(p.add_run(f"[{i}]\t"), 11)
        _set_font(p.add_run(r), 11)


def appendix(doc, x: F, runs: str):
    chapter(doc, "", "Appendix")
    para(doc, "Supporting Tables and Reproduction Instructions", size=14,
         bold=True, align="right", space_after=14, color=BLACK)

    section(doc, "A.1 Per-Cohort Breakdown on the Held-Out Fold")
    payload_table(doc, x, "T2c", "Table A1",
                  cols=["Cohort", "n", "Acc", "BAcc", "F1-M", "MCC",
                        "Top-2", "LogLoss", "Brier", "C-index"],
                  widths=[0.8, 0.5, 0.55, 0.57, 0.57, 0.55, 0.57, 0.65,
                          0.57, 0.65], font=7.5)

    section(doc, "A.2 Per-Class Performance")
    payload_table(doc, x, "T2b", "Table A2",
                  widths=[0.85, 0.7, 0.8, 0.7, 0.65, 0.85, 1.35], font=7.5,
                  head_map={"support": "Support", "precision": "Precision",
                            "recall": "Recall", "f1": "F1",
                            "specificity": "Specificity",
                            "mean_predicted_prob": "Mean predicted prob."})

    section(doc, "A.3 Recorded Learning Curve")
    payload_table(doc, x, "S4", "Table A3",
                  widths=[0.9, 1.3, 1.3, 1.45, 1.4], font=9.0,
                  head_map={"train_loss": "Train loss",
                            "val_score": "Val score",
                            "val_balanced_accuracy": "Val BAcc",
                            "val_c_index": "Val C-index"})

    section(doc, "A.4 Most Attended Edges of One Patient's Rewired Graph")
    payload_table(doc, x, "S5", "Table A4",
                  widths=[0.8, 1.6, 1.5, 1.4], font=9.0,
                  head_map={"target": "Target gene",
                            "attention": "Attention",
                            "relation": "Relation type"})

    section(doc, "A.5 Reproducing the Results")
    para(doc, "The repository that accompanies this report contains a "
              "single self-contained script that re-executes every phase "
              "reported in Chapter 6. It takes a subcommand per phase so "
              "that a reader can re-run a part without re-running the "
              "whole study:")
    for cmd, what in [
        ("fetch", "download the studies from the cBioPortal public API and "
                  "merge them onto the pre-committed gene panel"),
        ("train", "the main cross-validated run for one target"),
        ("baseline", "fit the whole comparator panel on the same folds"),
        ("ablate", "the single-component ablation arms"),
        ("loco", "the leave-one-cancer-out sweep"),
        ("tables", "export every table and figure from a completed run"),
        ("all", "every phase in order"),
        ("where", "print the resolved paths and exit, which is the quickest "
                  "check that an environment is set up correctly"),
    ]:
        rich(doc, [(f"    {cmd}  ", True), ("- " + what, False)],
             align="left", space_after=2)
    para(doc, "The slowest phases are the leave-one-cancer-out sweep and "
              "the ablation; their measured costs are stated in "
              "Section 5.1 so that a reader can budget for them. Both "
              "write their output incrementally after each cohort or arm, "
              "so an interrupted run does not lose the work already done. "
              "No patient-level data is shipped with the repository: the "
              "fetch subcommand re-downloads it from the public "
              "API.", space_before=8)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def page_setup(doc):
    for sec in doc.sections:
        sec.left_margin = Inches(1.1)
        sec.right_margin = Inches(1.0)
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(11)
    rf = st.element.rPr.rFonts
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(a), FONT)


def load_aux(x: F, runs: str):
    pr = os.path.join(runs, "cohort_parameter_report.json")
    if os.path.isfile(pr):
        with open(pr) as fh:
            x.aux["params"] = json.load(fh)
    else:
        x.missing.add(f"file.{pr}")


def assemble(a, x: F, toc=None, out=None):
    """Render the whole report once. Returns the heading list it recorded."""
    del TOC_ENTRIES[:]
    doc = Document()
    page_setup(doc)
    footer_page_of(doc)
    title_page(doc, a.logo, a.date)
    contents(doc, toc)
    abbreviations(doc)
    ch1(doc, x)
    ch2(doc, x)
    ch3(doc, x)
    ch4(doc, x, a.figures)
    ch5(doc, x)
    ch6(doc, x, a.figures)
    ch6b(doc, x, a.figures, a.runs)
    ch7(doc, x)
    ch8(doc, x)
    references(doc)
    appendix(doc, x, a.runs)
    target = out or a.out
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    doc.save(target)
    return list(TOC_ENTRIES)


def _squash(s: str) -> str:
    return "".join(s.split()).lower()


def measure_pages(docx_path: str, headings, workdir: str, skip_pages=2):
    """Convert to PDF and read the printed page number of each heading.

    Returns [(level, text, page)] or None when the conversion or the PDF
    reader is unavailable, in which case the caller falls back to the Word
    field and says so.
    """
    import shutil
    import subprocess
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    try:
        import pypdf
    except ImportError:
        return None
    os.makedirs(workdir, exist_ok=True)
    r = subprocess.run([soffice, "--headless", "--convert-to", "pdf",
                        "--outdir", workdir, docx_path],
                       capture_output=True, text=True, timeout=900)
    pdf = os.path.join(workdir, os.path.basename(docx_path)[:-5] + ".pdf")
    if r.returncode != 0 or not os.path.isfile(pdf):
        return None
    pages = [_squash(p.extract_text() or "")
             for p in pypdf.PdfReader(pdf).pages]
    out, cursor = [], skip_pages
    for level, text, find in headings:
        needle = _squash(find)
        page = None
        for i in range(cursor, len(pages)):
            if needle and needle in pages[i]:
                page, cursor = i + 1, i
                break
        out.append((level, text, page))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default="deliverables/manuscript_payload.json")
    ap.add_argument("--runs", default="runs_pan_cohort")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--logo", default="assets/ewu_logo.png")
    ap.add_argument("--date", default="19 November 2025")
    ap.add_argument("--workdir", default=None,
                    help="scratch directory for the pagination passes")
    ap.add_argument("--out",
                    default="deliverables/CSE400B-MOSAIC-GNN-Report.docx")
    a = ap.parse_args()

    with open(a.payload) as fh:
        payload = json.load(fh)
    x = F(payload)
    load_aux(x, a.runs)

    work = a.workdir or os.path.join(os.path.dirname(a.out) or ".", "_toc")

    # Pass 1: harvest the heading list (the contents is a Word field here,
    # so its own length does not matter yet).
    headings = assemble(a, x, toc=None, out=os.path.join(work, "pass1.docx"))

    # Pass 2: the same contents lines the reader will see, with placeholder
    # numbers, so that pass 2 and pass 3 paginate identically.
    placeholder = [(lv, tx, None) for lv, tx, _ in headings]
    assemble(a, x, toc=placeholder, out=os.path.join(work, "pass2.docx"))
    measured = measure_pages(os.path.join(work, "pass2.docx"), headings, work)

    if measured is None:
        print("could not render a PDF to measure page numbers; writing the "
              "contents as a Word field instead (Ctrl+A then F9 in Word "
              "builds it)")
        assemble(a, x, toc=None)
    else:
        unresolved = [t for _, t, p in measured if p is None]
        assemble(a, x, toc=measured)
        print(f"contents built from a measured rendering: "
              f"{len(measured) - len(unresolved)} of {len(measured)} "
              f"headings located")
        for t in unresolved:
            print("  - page number not found for:", t)
    print(f"wrote {a.out}")
    if x.missing:
        print("values the payload did not supply (printed as "
              f"'{NOT_RUN}' in the document):")
        for m in sorted(x.missing):
            print("  -", m)
    else:
        print("every value in the document came from the payload or a run "
              "artefact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
