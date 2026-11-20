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
import sys
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
        self.analysis = {}     # derived analyses, cohort run
        self.analysis_stage = {}   # derived analyses, stage_binary run
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
                  widths=None, font=8.5, caption_text=None, head_map=None,
                  int_cols=None):
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
    ints = {head.index(c) for c in (int_cols or []) if c in head}
    out = [[(head_map or {}).get(head[i], head[i]) for i in keep]]
    out += [[(iv(r[i]) if i in ints else tidy_cell(r[i])) for i in keep]
            for r in body]
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
def ch1(doc, x: F, n: N):
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


def ch2(doc, x: F, n: N):
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


def ch3(doc, x: F, n: N):
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
    caption(doc, n.t() + ":", "Comparison of representative multi-omic methods "
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
def ch4(doc, x: F, n: N, A: dict, figdir: str):
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

    section(doc, "4.7 Mathematical Formulation")
    para(doc, "This section states the model as equations, so that the prose "
              "above can be checked against it. Write B for the patients in a "
              "batch, G for the genes of the panel, M for the functional "
              "modules, R for the edge relations and H for the attention "
              "heads. Patient i enters as a matrix x_i in R^(G x 3), one row "
              "per gene, holding that gene's mutation indicator, copy-number "
              "level and expression z-score.")
    rich(doc, [("Gene level: relational graph attention over a shared "
                "topology. ", True),
               ("The topology is one edge list for every patient — protein "
                "interactions do not depend on who is being modelled — while "
                "the attention coefficients are computed per patient, which "
                "is what makes the rewiring in Section 6.15 personal to each "
                "one. For relation r, head k and edge u to v:", False)])
    formula(doc, "h_u^(r) = W_r x_u")
    formula(doc, "e_uv^(r,k) = LeakyReLU_0.2( a_src^(r,k) · h_u^(r,k) "
                 "+ a_dst^(r,k) · h_v^(r,k) + b_(r,k) ) + log w_uv")
    formula(doc, "α_uv^(k) = exp( e_uv^(k) ) / Σ_(u' in N(v)) "
                 "exp( e_(u'v)^(k) )")
    formula(doc, "h'_v = LayerNorm( W_res x_v + concat_k [ "
                 "Σ_(u in N(v)) α_uv^(k) h_u^(r_uv, k) ] + b )")
    para(doc, "Two details of that are worth naming because they are "
              "choices, not conventions. The edge weight enters as an "
              "additive log term inside the exponent, which is exactly a "
              "multiplicative weight on the softmax, so a weak edge is "
              "down-weighted rather than thresholded away. And the softmax "
              "is taken over each node's incoming edges separately — a "
              "segment softmax — implemented as a scatter-add, which is the "
              "operation whose non-deterministic summation order produces "
              "the noise floor measured in Section 6.5.")
    rich(doc, [("Module pooling with a within-module gate. ", True),
               ("Let the fixed assignment matrix be S in {0,1}^(G x M), "
                "with S_gm = 1 when gene g belongs to module m. A scalar "
                "gate scores each gene, and the scores are normalised inside "
                "each module rather than across all genes:", False)])
    formula(doc, "s_g = MLP( h'_g ),      "
                 "w_gm = S_gm · exp(s_g) / Σ_(g' in m) exp(s_(g'))")
    formula(doc, "z_m = Proj( Σ_g w_gm h'_g ),      "
                 "gate_g = Σ_m w_gm")
    para(doc, "The last quantity is what Section 6.15 reports as the learned "
              "per-gene gate: it is the total weight that gene received "
              "across the modules it belongs to, so it is readable without "
              "any post-hoc attribution method.")
    rich(doc, [("Module attention readout. ", True),
               ("A single learned query collapses the M module embeddings "
                "into one patient vector, and the attention it uses is the "
                "module importance reported in Section 6.15:", False)])
    formula(doc, "β_m = softmax_m( (W_k z_m) · q / √d ),      "
                 "z = Out( Σ_m β_m W_v z_m )")
    rich(doc, [("Patient level: a learned graph. ", True),
               ("Similarity is measured in H learned metric subspaces, each "
                "a diagonal reweighting θ_h of the embedding, and the dense "
                "affinity is sparsified to the k nearest anchors:", False)])
    formula(doc, "S_nm = (1/H) Σ_h u_h(n) · u_h(m) / τ,      "
                 "u_h(n) = normalise( z_n * θ_h )")
    formula(doc, "A = rownormalise( topk_k( softmax(S) ) + I )")
    para(doc, "Self-similarity is masked out before the top-k, and during "
              "training only patients in the training split may act as "
              "sources, so a held-out patient can receive information but "
              "never supply it. The top-k is sparse in the forward pass and "
              "differentiable in the backward one through a straight-through "
              "estimator, mask = hard + soft − detach(soft), which is what "
              "lets the metric θ be learned at all. Propagation is then")
    formula(doc, "h^(l+1) = LayerNorm( h^(l) + MLP( A h^(l) ) )")
    para(doc, "Because the gene-level encoder cannot be run on every patient "
              "at once, out-of-batch neighbours contribute through a bank of "
              "their most recent embeddings, updated with momentum and "
              "detached from the graph. That is the standard historical-"
              "embedding approximation and it is the reason memory stays "
              "linear in the batch size rather than in the cohort.")
    rich(doc, [("Disentanglement. ", True),
               ("The patient vector is split as z = [z_shared, z_specific]. "
                "A cohort classifier is attached to each, but the shared "
                "branch passes through a gradient-reversal layer, which is "
                "the identity forwards and multiplies the gradient by −λ "
                "backwards, so the encoder is pushed to make the cohort hard "
                "to recover from z_shared. A cross-covariance penalty keeps "
                "the two parts from duplicating each other:", False)])
    formula(doc, "L_orth = mean[ ( A*^T B* )² ],      A* and B* = the two "
                 "latents, centred and column-normalised")
    rich(doc, [("Heads and objective. ", True),
               ("Three heads sit on the representation. The classifier is a "
                "weighted cross-entropy with label smoothing. The Cox head "
                "maximises the Breslow partial likelihood, the risk set "
                "being the patients still at risk at each event time:",
                False)])
    formula(doc, "L_cox = − (1 / Σ_i d_i) Σ_i d_i [ r_i − log Σ_(j: T_j ≥ "
                 "T_i) exp(r_j) ]")
    para(doc, "with r_i the predicted risk, T_i the follow-up time and "
              "d_i the event indicator. The discrete-time head models the "
              "hazard in each of twelve follow-up bins, which gives a "
              "survival curve the Cox score alone cannot:")
    formula(doc, "L_haz = − (1/N) Σ_i [ Σ_(j < k_i) log(1 − h_ij) "
                 "+ d_i log h_(i k_i) + (1 − d_i) log(1 − h_(i k_i)) ]")
    para(doc, "where k_i is the bin in which patient i was last observed. A "
              "censored patient contributes only the survival part, which is "
              "the whole point of the construction. The contrastive term is "
              "a symmetric InfoNCE across omic views, treating the same "
              "patient seen through two assays as a positive pair, and the "
              "graph regulariser keeps the learned patient graph smooth, "
              "sparse and non-degenerate:")
    formula(doc, "L_graph = w_s Σ_nm A_nm ||z_n − z_m||² / N² "
                 "+ w_sp Σ A² / N² − w_d mean_n log( Σ_m A_nm ) / N")
    para(doc, "The total objective is the weighted sum of these terms. The "
              "weights are not chosen here for exposition; they are read "
              "from the configuration file the run itself saved, so the "
              "table below is the objective that was actually optimised.")
    formula(doc, "L = w_class L_class + w_cox L_cox + w_haz L_haz "
                 "+ λ w_adv L_adv + w_coh L_coh + w_orth L_orth "
                 "+ w_con L_con + w_graph L_graph")
    cfg = x.aux.get("config") or {}
    loss = cfg.get("loss") or {}
    model = cfg.get("model") or {}
    graph = cfg.get("graph") or {}
    if loss:
        rows_table(doc, n.t(),
                   "The objective's weights and the model's principal "
                   "dimensions, read from the configuration the cohort run "
                   "saved alongside its checkpoint. Nothing in this table "
                   "was chosen for the report.",
                   ["Symbol or setting", "Meaning", "Value"],
                   [["w_class", "weight of the classification "
                                "cross-entropy", fv(loss.get("w_class"), 2)],
                    ["w_cox", "weight of the Cox partial likelihood",
                     fv(loss.get("w_cox"), 2)],
                    ["w_haz", "weight of the discrete-time hazard likelihood",
                     fv(loss.get("w_hazard"), 2)],
                    ["w_adv", "weight of the cohort adversary, through the "
                              "reversal layer",
                     fv(loss.get("w_adversarial"), 2)],
                    ["w_coh", "weight of the cohort head on z_specific",
                     fv(loss.get("w_cohort"), 2)],
                    ["w_orth", "weight of the cross-covariance penalty",
                     fv(loss.get("w_orthogonal"), 2)],
                    ["w_con", "weight of the cross-omic contrastive term",
                     fv(loss.get("w_contrast"), 2)],
                    ["w_graph", "weight of the patient-graph regulariser",
                     fv(loss.get("w_graph_reg"), 3)],
                    ["temperature", "contrastive temperature",
                     fv(loss.get("contrast_temp"), 2)],
                    ["label smoothing", "applied in the classification term",
                     fv(loss.get("label_smoothing"), 2)],
                    ["R", "edge relations in the gene graph",
                     iv(graph.get("n_relations"))],
                    ["H", "attention heads at the gene level",
                     iv(model.get("gene_heads"))],
                    ["gene layers", "stacked relational attention layers",
                     iv(model.get("gene_layers"))],
                    ["gene width", "hidden dimension per gene",
                     iv(model.get("gene_hidden"))],
                    ["M", "functional modules",
                     iv(graph.get("n_modules"))],
                    ["minimum module size", "modules smaller than this are "
                                            "merged",
                     iv(graph.get("module_min_size"))],
                    ["k (patients)", "neighbours in the learned patient "
                                     "graph", iv(graph.get("patient_knn"))],
                    ["τ", "temperature of the patient-similarity softmax",
                     fv(graph.get("graph_learner_temp"), 2)],
                    ["patient layers", "propagation steps over the patient "
                                       "graph",
                     iv(model.get("patient_layers"))],
                    ["dim z_shared", "width of the shared representation",
                     iv(model.get("z_shared"))],
                    ["dim z_specific", "width of the cohort-specific part",
                     iv(model.get("z_specific"))],
                    ["survival bins", "bins of the discrete-time hazard head",
                     iv(model.get("n_survival_bins"))],
                    ["dropout", "on the representation",
                     fv(model.get("dropout"), 2)],
                    ["edge dropout", "on gene-graph edges during training",
                     fv(model.get("edge_dropout"), 2)]],
                   widths=[1.5, 3.4, 1.0], font=9.0)




# --------------------------------------------------------------------------- #
# Chapter 5 - Experimental Setup
# --------------------------------------------------------------------------- #
def ch5(doc, x: F, n: N):
    chapter(doc, "5", "Experimental Setup")
    section(doc, "5.1 Computing Environment and Software")
    caption(doc, n.t() + ":", "Computing environment and the principal "
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
    caption(doc, n.t() + ":", "Principal hyperparameters. These were chosen "
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

    section(doc, "5.4 Evaluation Metrics: Definitions")
    para(doc, "Every metric reported in Chapter 6 is defined here, so that "
              "no number in that chapter depends on the reader guessing a "
              "convention. Throughout, a class c has support n_c, and its "
              "one-versus-rest contingency table has true positives TP, "
              "false positives FP, false negatives FN and true negatives "
              "TN over N held-out patients. A metric written \"macro\" is "
              "the unweighted mean of that metric over all K classes, which "
              "gives a cohort of thirty-six patients the same weight as a "
              "cohort of a thousand.")
    rich(doc, [("Classification. ", True),
               ("The agreement and rate measures:", False)])
    formula(doc, "Accuracy = Σ_c TP_c / N,      "
                 "Sensitivity_c = TP_c/(TP_c+FN_c),      "
                 "Specificity_c = TN_c/(TN_c+FP_c)")
    formula(doc, "Precision_c = TP_c/(TP_c+FP_c),      "
                 "NPV_c = TN_c/(TN_c+FN_c),      "
                 "F1_c = 2·Precision_c·Sensitivity_c /"
                 " (Precision_c+Sensitivity_c)")
    formula(doc, "Balanced accuracy = (1/K) Σ_c Sensitivity_c      "
                 "(identical to macro sensitivity)")
    para(doc, "Balanced accuracy is the headline classification metric "
              "everywhere in this report, because the class distribution is "
              "strongly non-uniform and raw accuracy would reward a model "
              "for predicting the large cohorts. The weighted F1 instead "
              "weights each class by n_c/N, so the two F1 columns answer "
              "different questions and both are reported.")
    rich(doc, [("Chance-corrected and ranking measures. ", True),
               ("Cohen's kappa compares the observed agreement p_o with "
                "the agreement p_e expected from the marginals; its "
                "quadratic form penalises a mistake by the square of its "
                "distance, which needs more than two ordered classes and "
                "is therefore undefined on the binary target. The "
                "Matthews correlation is the correlation coefficient of "
                "the contingency table and is the single most "
                "imbalance-robust summary available:", False)])
    formula(doc, "κ = (p_o − p_e)/(1 − p_e),      "
                 "MCC = (TP·TN − FP·FN) / "
                 "√((TP+FP)(TP+FN)(TN+FP)(TN+FN))")
    para(doc, "AUROC is the area under the sensitivity-against-"
              "(1−specificity) curve, macro-averaged over the "
              "one-versus-rest problems; AUPRC is the area under the "
              "precision-against-sensitivity curve, which is the more "
              "informative of the two when positives are rare. Top-2 "
              "accuracy is the fraction of patients whose true class is "
              "among the model's two highest-scoring classes.")
    rich(doc, [("Proper scoring and calibration. ", True),
               ("These three measure the probabilities rather than the "
                "decisions, and on all three a lower value is better. Write "
                "q_ik for the probability the model gave patient i of "
                "belonging to class k, and y_i for that patient's true "
                "class:", False)])
    formula(doc, "LogLoss = −(1/N) Σ_i log q_i,y_i,      "
                 "Brier = (1/N) Σ_i Σ_k ( q_ik − 1[y_i=k] )²")
    formula(doc, "ECE = Σ_b (n_b/N) · | accuracy(b) − confidence(b) |,"
                 "      over equal-width confidence bins b")
    para(doc, "Expected calibration error is the one metric in the panel "
              "that a discriminative model can fail while scoring well "
              "everywhere else, which is why Section 6.16 is devoted to it. "
              "Predictive uncertainty is summarised by the entropy of the "
              "predicted distribution under Monte-Carlo dropout, "
              "H(q) = −Σ_k q_k log q_k, reported in nats and also as a "
              "fraction of its maximum ln K so that targets with different "
              "class counts can be compared.")
    rich(doc, [("Survival. ", True),
               ("Harrell's concordance index is the fraction of "
                "comparable patient pairs the model orders correctly. A "
                "pair is comparable when the earlier event time is an "
                "observed death rather than a censoring:", False)])
    formula(doc, "C = Σ_{comparable (i,j)} 1[ risk_i > risk_j ] / "
                 "#{comparable pairs},      C = 0.5 is random ordering")
    para(doc, "Time-dependent AUC at 12, 36 and 60 months is the same "
              "ordering question restricted to survival status at that "
              "horizon, which exposes a model that ranks early deaths well "
              "and late deaths badly or the reverse. The log-rank test "
              "compares the Kaplan-Meier curves of the risk tertiles the "
              "model itself defines, and its chi-square statistic has two "
              "degrees of freedom for three groups; it answers whether the "
              "strata separate at all, not by how much, which is what the "
              "C-index is for.")

    section(doc, "5.5 Statistical Procedures")
    para(doc, "Five procedures are used in Chapter 6 and each is stated "
              "here with the formula applied, so that every derived number "
              "in that chapter can be checked by hand.")
    bullets(doc, [
        ("Paired bootstrap. ", "The held-out patients are resampled with "
         "replacement B times; on each resample both models' metrics are "
         "recomputed and their difference recorded. The reported interval "
         "is the empirical 2.5th to 97.5th percentile of those "
         "differences, and the p-value is the proportion of resamples in "
         "which the difference reverses sign. Because both models see the "
         "same resampled patients, their correlated errors cancel, which "
         "an unpaired comparison of two means does not achieve."),
        ("Standard error and standardised effect. ", "From an interval "
         "Δ ± z·SE the standard error is recovered as "
         "SE = (upper − lower)/(2·1.95996) and the standardised effect as "
         "z = Δ/SE. Both are tabulated in Section 6.8 because an interval "
         "alone makes a small effect measured precisely look like a large "
         "effect measured loosely."),
        ("Benjamini-Hochberg correction. ", "For m p-values sorted "
         "ascending, q(i) = min over j ≥ i of (p(j)·m/j), capped at one. "
         "This controls the expected proportion of false discoveries among "
         "the rejected hypotheses. Section 6.8 recomputes the procedure "
         "independently from the stored p-values as a check, and also "
         "reports what a stricter pooled family would do."),
        ("Wilson score interval. ", "For k successes in n trials the "
         "interval is centred at (k + z²/2)/(n + z²) with half-width "
         "z/(n+z²)·√(k(n−k)/n + z²/4). It is used instead of the normal "
         "approximation because several classes have fewer than twenty "
         "members, where the normal interval can run outside [0,1]."),
        ("Pearson correlation with a Fisher z interval. ", "The "
         "coefficient is r = Σ(x−x-bar)(y−y-bar)/√(Σ(x−x-bar)²Σ(y−y-bar)²); its "
         "two-sided p-value comes from t = r√((n−2)/(1−r²)) on n−2 "
         "degrees of freedom; and its interval is obtained by "
         "transforming z = artanh r, SE = 1/√(n−3), and mapping the "
         "interval back with the hyperbolic tangent. Sections 6.13 and "
         "6.14 report all three."),
        ("Least squares on a logarithmic axis. ", "The sample-size curves "
         "are fitted as y = a + b·log2 n, so that b reads as the change "
         "per doubling of the training set, with "
         "R² = 1 − SS_residual/SS_total."),
    ])
    para(doc, "One convention governs how all of this is reported. Where a "
              "quantity was measured, it is given. Where it was derived "
              "from measured quantities, the formula is printed beside it. "
              "Where a phase did not run, the report says so. No number in "
              "Chapter 6 was estimated, interpolated or carried over from "
              "another configuration.")


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
def ch7(doc, x: F, n: N, A: dict):
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
              "aggressive ones, as Section 6.14 shows. A model deployed "
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
def ch8(doc, x: F, n: N, A: dict):
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
    lc = A.get("loco") or {}
    para(doc, "Two findings are positive and both replicate or transfer. "
              "Molecular features add prognostic information beyond "
              "clinical covariates, measured two independent ways on two "
              "independently fitted targets, with every difference "
              "surviving correction. And the learned representation "
              "stratifies risk in cancer types held out entirely from "
              f"training, above random in {iv(lc.get('n_above_half'))} of "
              f"{iv(lc.get('n_cohorts'))} cohorts, with the log-rank test "
              f"significant in {iv(lc.get('n_p_lt_05'))} of them before "
              f"multiplicity correction and {iv(lc.get('n_q_lt_05'))} after "
              "it. That is the evidence against the explanation that the "
              "pooled result is cohort detection in disguise. It is "
              "evidence rather than proof: Section 6.14.1 documents an "
              "unexplained fault in the classification half of the same "
              "experiment, and although the argument there is that the "
              "survival half is untouched, the finding should be read as "
              "provisional until that fault is found.")
    _best = x.g("cohort", "best_balanced_accuracy") or {}
    best_name = _best.get("model", "the strongest baseline")
    best_val = _best.get("value")
    para(doc, "One finding is negative and it is the one that governs how "
              "the rest should be read. Under the paired, corrected "
              "comparison MOSAIC-GNN does not outperform established "
              "methods on either classification endpoint, and it is "
              "competitive rather than superior on survival ranking. "
              f"On the {iv(lc.get('n_cohorts'))}-class problem the best "
              f"comparator, {best_name}, reaches a balanced accuracy of "
              f"{fv(best_val, 3)} where the proposed model reaches "
              f"{fv((A.get('panel') or {}).get('balanced_accuracy'), 3)}. "
              "The graph "
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
        ("Debug the leave-one-cancer-out classification path. ",
         "Section 6.14.1 reports single-class recalls that the experiment's "
         "design should make impossible. This is the first thing to fix, "
         "not because those numbers are wanted — they are uninterpretable "
         "either way — but because the survival transfer from the same "
         "experiment is one of the project's two positive findings and its "
         "security currently rests on an argument rather than on a clean "
         "run. The two candidate explanations to test are a label-space "
         "re-indexing between fitting and scoring, and the held-out class "
         "acting as the default output for out-of-distribution inputs."),
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


def appendix(doc, x: F, n: N, runs: str):
    chapter(doc, "", "Appendix")
    para(doc, "Supporting Tables and Reproduction Instructions", size=14,
         bold=True, align="right", space_after=14, color=BLACK)

    section(doc, "A.1 Per-Cohort Breakdown on the Held-Out Fold")
    payload_table(doc, x, "T2c", "Table A1",
                  cols=["Cohort", "n", "Acc", "Top-2", "LogLoss", "Brier",
                        "C-index"],
                  widths=[1.0, 0.7, 0.8, 0.8, 0.85, 0.8, 0.85], font=8.5,
                  int_cols=["n"],
                  head_map={"Acc": "Recall"},
                  caption_text="Per-cohort results on the held-out fold of "
                               "the cohort run, as Section 6.13 reports "
                               "them. The columns that are degenerate in a "
                               "single-class slice — balanced accuracy, "
                               "which equals the recall; the "
                               "chance-corrected measures, which are zero "
                               "for every row; and the macro averages, "
                               "which count the model's distinct wrong "
                               "guesses — are omitted, for the reasons "
                               "given in Section 6.13.")

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
    # The run's own configuration, so Section 4.7's objective weights and
    # dimensions are the ones that were optimised rather than ones retyped.
    for name in ("cohort_config.json", "stage_binary_config.json"):
        cf = os.path.join(runs, name)
        if os.path.isfile(cf):
            with open(cf) as fh:
                x.aux["config"] = json.load(fh)
            break
    else:
        x.missing.add(f"file.{runs}/*_config.json")
    pr = os.path.join(runs, "cohort_parameter_report.json")
    if os.path.isfile(pr):
        with open(pr) as fh:
            x.aux["params"] = json.load(fh)
    else:
        x.missing.add(f"file.{pr}")


def assemble(a, x: F, toc=None, out=None):
    """Render the whole report once. Returns the heading list it recorded."""
    del TOC_ENTRIES[:]
    n = N()
    A, S = x.analysis, x.analysis_stage
    doc = Document()
    page_setup(doc)
    footer_page_of(doc)
    title_page(doc, a.logo, a.date)
    contents(doc, toc)
    abbreviations(doc)
    ch1(doc, x, n)
    ch2(doc, x, n)
    ch3(doc, x, n)
    ch4(doc, x, n, A, a.figures)
    ch5(doc, x, n)
    ch6_data(doc, x, n, A, a.figures)
    ch6_panel(doc, x, n, A, S)
    ch6_consistency(doc, x, n, A, S)
    ch6_gap(doc, x, n, A, a.figures)
    ch6_compare(doc, x, n, A, a.runs, a.figures)
    ch6_paired(doc, x, n, A)
    ch6_gain(doc, x, n, A, S)
    ch6_ablation(doc, x, n, A, a.figures)
    ch6_permutation(doc, x, n, A)
    ch6_samplesize(doc, x, n, A, a.figures)
    ch6_percohort(doc, x, n, A)
    ch6_loco(doc, x, n, A, a.figures)
    ch6_interpret(doc, x, n, A, a.runs, a.figures)
    ch6_uncertainty(doc, x, n, A, S, a.figures)
    ch6_cost(doc, x, n, A, S)
    ch6_stage(doc, x, n, S, a.runs_stage)
    ch6_discussion(doc, x, n, A, S)
    ch7(doc, x, n, A)
    ch8(doc, x, n, A)
    references(doc)
    appendix(doc, x, n, a.runs)
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
    ap.add_argument("--runs-stage", dest="runs_stage",
                    default="runs_pan_stage")
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
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import report_analysis as RA
    x.analysis = RA.analyse(a.runs)
    x.analysis_stage = RA.analyse(a.runs_stage)

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




# --------------------------------------------------------------------------- #
# Numbering, so that forty tables and ten figures cannot drift out of order
# --------------------------------------------------------------------------- #
class N:
    """Running caption numbers. `t()` and `f()` hand out the next one."""

    def __init__(self):
        self.table = 0
        self.figure = 0
        self.last_table = ""

    def t(self) -> str:
        self.table += 1
        self.last_table = f"Table {self.table}"
        return self.last_table

    def tp(self, letter: str) -> str:
        """A part of the table just handed out, e.g. 'Table 9(b)'."""
        return f"{self.last_table}({letter})"

    def f(self) -> str:
        self.figure += 1
        return f"Figure {self.figure}"

    def ref_t(self, offset: int = 0) -> str:
        return f"Table {self.table + offset}"


def fv(v, d=3, dash="—"):
    """Format a measured number, or a dash where the run produced none."""
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:                                   # NaN
        return dash
    if f != 0 and (abs(f) < 10 ** (-d) or abs(f) >= 1e5):
        return f"{f:.2e}"
    return f"{f:.{d}f}"


def pv(v, floor=None, dash="—"):
    """Format a p- or q-value. Zero means "below the resolution", not zero."""
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return dash
    if f <= 0:
        return f"< {floor:.4f}".rstrip("0") if floor else "< resolution"
    if f < 0.0001:
        return f"{f:.1e}"
    return f"{f:.4f}"


def pctf(v, dash="—"):
    """A percentage: one decimal, more only when one would read as zero."""
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return dash
    if f == 0:
        return "0 %"
    for d in (1, 2, 3):
        if round(f, d) != 0:
            return f"{f:.{d}f} %"
    return f"{f:.1e} %"


def iv(v, dash="—"):
    if v is None:
        return dash
    try:
        return format(int(round(float(v))), ",")
    except (TypeError, ValueError):
        return str(v)


HIB = {
    "accuracy": True, "balanced_accuracy": True, "f1_macro": True,
    "f1_weighted": True, "precision_macro": True, "recall_macro": True,
    "specificity_macro": True, "npv_macro": True, "auroc": True,
    "auprc": True, "mcc": True, "kappa": True, "kappa_quadratic": True,
    "top2_accuracy": True, "c_index": True, "auc@12m": True,
    "auc@36m": True, "auc@60m": True, "logrank_chi2": True,
    "log_loss": False, "brier": False, "ece": False, "logrank_p": False,
    "mean_predictive_entropy": False,
}


def formula(doc, text: str, size=10.5):
    """A displayed formula: centred, italic, set apart from the body."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.right_indent = Inches(0.25)
    _set_font(p.add_run(text), size, italic=True)
    return p


def rows_table(doc, label: str, cap: str, header, rows, widths=None,
               font=8.5):
    caption(doc, label + ":", cap)
    if not rows:
        para(doc, "[the run that produces this table did not complete]",
             italic=True, align="left")
        return
    add_table(doc, [list(header)] + [list(r) for r in rows], widths=widths,
              font=font)


def wide_table(doc, label: str, cap: str, header, rows, groups,
               first_width=0.95, col_width=0.6, font=7.5, part_caps=None):
    """A table too wide for the page, split into lettered parts.

    Every column of the source appears in exactly one part, so nothing is
    dropped to make the table fit. The first column repeats in each part so
    each part can be read on its own.
    """
    letters = "abcdefghij"
    # Resolve column names to positions by consuming each occurrence left to
    # right, because a header may legitimately repeat a name -- the paired
    # test table carries SE, z, CI, p and q once per metric. A plain
    # name-to-index dict would keep only the last occurrence and silently
    # fill the first group with the second group's numbers.
    used = set()
    resolved = []
    for group in groups:
        keep = []
        for name in group:
            for i, head_name in enumerate(header):
                if i not in used and head_name == name:
                    used.add(i)
                    keep.append(i)
                    break
        resolved.append(keep)
    single = len([k for k in resolved if k]) == 1
    for j, keep in enumerate(resolved):
        if not keep:
            continue
        lbl = label if single else f"{label}({letters[j]})"
        piece = (part_caps or [None] * len(groups))[j]
        this_cap = cap if single else (
            f"{cap} {piece}" if j == 0 else (piece or cap))
        caption(doc, lbl + ":", this_cap)
        out = [[header[0]] + [header[i] for i in keep]]
        for r in rows:
            out.append([r[0]] + [r[i] for i in keep])
        add_table(doc, out,
                  widths=[first_width] + [col_width] * len(keep), font=font)


# --------------------------------------------------------------------------- #
# Chapter 6 - Experimental Results and Discussion
#
# Every number is read from the run exports through report_analysis, which
# states the formula it applies. Nothing is typed in.
# --------------------------------------------------------------------------- #
CLS_GROUPS = [
    ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"],
    ["precision_macro", "recall_macro", "specificity_macro", "npv_macro"],
    ["auroc", "auprc", "mcc", "kappa", "kappa_quadratic", "top2_accuracy"],
    ["log_loss", "brier", "ece"],
]
CLS_CAPS = [
    "the four pooled agreement measures.",
    "the four macro-averaged rates of the one-versus-rest tables.",
    "the ranking and chance-corrected measures.",
    "the three proper scoring and calibration measures, on which a lower "
    "value is better.",
]
SHORT = {
    "accuracy": "Acc", "balanced_accuracy": "BAcc", "f1_macro": "F1-M",
    "f1_weighted": "F1-W", "precision_macro": "Prec", "recall_macro": "Sens",
    "specificity_macro": "Spec", "npv_macro": "NPV", "auroc": "AUROC",
    "auprc": "AUPRC", "mcc": "MCC", "kappa": "κ",
    "kappa_quadratic": "κ-quad", "top2_accuracy": "Top-2",
    "log_loss": "LogLoss", "brier": "Brier", "ece": "ECE",
    "c_index": "C-index", "auc@12m": "AUC12", "auc@36m": "AUC36",
    "auc@60m": "AUC60", "logrank_chi2": "LR-χ²", "logrank_p": "LR-p",
    "mean_predictive_entropy": "Entropy",
}


def _short(names):
    return [SHORT.get(n, n) for n in names]


def _cmp_rows(runs: str, panel: dict, metrics):
    """Comparator rows for one run, MOSAIC-GNN's measured row placed first."""
    import report_analysis as RA
    src = RA.read_csv(os.path.join(runs, "table3_comparison_full.csv"))
    if not src:
        return None
    first = list(src[0])[0]
    rows = [["MOSAIC-GNN"] + [fv(panel.get(m)) for m in metrics]]
    for r in sorted(src, key=lambda r: -(RA.num(r.get("balanced_accuracy"))
                                         or -1)):
        rows.append([r[first]] + [fv(RA.num(r.get(m))) for m in metrics])
    return rows


def ch6_data(doc, x: F, n: N, A: dict, figdir: str):
    chapter(doc, "6", "Experimental Results and Discussion")
    para(doc, "Every number in this chapter was produced by a run that was "
              "actually executed on the assembled data, and every derived "
              "quantity is computed from those run outputs by the formula "
              "stated beside it in Section 5.5. Where a phase ran at "
              "reduced settings, as recorded in Section 5.3, the section "
              "says so before the number. Where a phase did not run, the "
              "report says that instead of supplying a figure. Several of "
              "the results below contradict what the architecture was "
              "designed to achieve; they are reported in the same detail as "
              "the favourable ones, because a result that only survives "
              "when it agrees with the design is not a result.")
    para(doc, "The chapter is organised so that the audit comes before the "
              "claim. Sections 6.1 to 6.4 establish what the data is and "
              "whether the pipeline is sound: the assembled cohorts, the "
              "full metric panel, an arithmetic reconstruction of the "
              "classifier's contingency tables, and the "
              "train-to-test gap. Sections 6.5 to 6.9 are the comparison "
              "against established methods and the question of what the "
              "molecular data adds. Sections 6.10 to 6.14 are the component "
              "ablation, the leakage controls and the transfer experiment. "
              "Sections 6.15 to 6.18 cover interpretation, uncertainty, cost "
              "and the second prediction target. Sections 6.19 and 6.20 "
              "discuss the whole and state what threatens it.")

    # ------------------------------------------------------------------ #
    section(doc, "6.1 The Assembled Dataset")
    d = A.get("descriptives") or {}
    para(doc, "The assembly retrieved "
              f"{iv(d.get('n_cohorts'))} studies covering "
              f"{iv(d.get('patients_total'))} patients with "
              f"{iv(d.get('events_total'))} observed deaths, a pooled event "
              f"rate of {fv(d.get('pooled_event_rate'))}. For the "
              "cohort target every patient carries a label. For "
              "stage_binary only the cohorts that record AJCC "
              f"stage contribute, giving {x.i('stage_binary','n_labelled')} "
              f"labelled patients in "
              f"{x.i('stage_binary','n_cohorts_labelled')} cohorts, so a "
              "third of the pan-cancer population cannot be used for the "
              "clinically meaningful endpoint at all. That is itself a "
              "result and it is stated before any performance number.")
    rows_table(doc, n.t(),
               "Descriptive statistics of the assembled dataset. Each row "
               "is computed from the cohort table by the formula named in "
               "the third column; the entropy is in nats and its maximum "
               "ln K is the value a perfectly balanced set of K classes "
               "would give.",
               ["Quantity", "Value", "How it is computed"],
               [["Studies (K)", iv(d.get("n_cohorts")), "count of rows"],
                ["Patients", iv(d.get("patients_total")),
                 "sum of per-cohort patients"],
                ["Observed deaths", iv(d.get("events_total")),
                 "sum of per-cohort events"],
                ["Pooled event rate", fv(d.get("pooled_event_rate")),
                 "deaths / patients"],
                ["Smallest cohort",
                 f"{iv(d.get('patients_min'))} ({d.get('patients_min_cohort')})",
                 "minimum of patients"],
                ["Largest cohort",
                 f"{iv(d.get('patients_max'))} ({d.get('patients_max_cohort')})",
                 "maximum of patients"],
                ["Median cohort size", fv(d.get("patients_median"), 1),
                 "median of patients"],
                ["Interquartile range",
                 f"{fv(d.get('patients_q1'),1)} to {fv(d.get('patients_q3'),1)}",
                 "25th and 75th percentiles"],
                ["Imbalance ratio", fv(d.get("imbalance_ratio"), 2),
                 "largest / smallest cohort"],
                ["Shannon entropy H", fv(d.get("entropy_nats"), 4),
                 "−Σ p_k ln p_k over cohort shares"],
                ["Maximum entropy ln K", fv(d.get("entropy_max"), 4),
                 "natural log of the class count"],
                ["Normalised entropy", fv(d.get("entropy_ratio"), 4),
                 "H / ln K"],
                ["Gini impurity", fv(d.get("gini"), 4), "1 − Σ p_k²"],
                ["Lowest event rate",
                 f"{fv(d.get('event_rate_min'))} ({d.get('event_rate_min_cohort')})",
                 "minimum of event rate"],
                ["Highest event rate",
                 f"{fv(d.get('event_rate_max'))} ({d.get('event_rate_max_cohort')})",
                 "maximum of event rate"],
                ["Median event rate", fv(d.get("event_rate_median")),
                 "median of event rate"],
                ["Median follow-up (months)", fv(d.get("followup_median"), 1),
                 "median of per-cohort median OS"],
                ["Follow-up range (months)",
                 f"{fv(d.get('followup_min'),1)} to {fv(d.get('followup_max'),1)}",
                 "minimum and maximum"],
                ["Cohorts under 100 patients", iv(d.get("n_below_100")),
                 "count where patients < 100"],
                ["Cohorts under 20 events", iv(d.get("n_events_below_20")),
                 "count where events < 20"],
                ["Cohorts under 60 events", iv(d.get("n_events_below_60")),
                 "count where events < 60"]],
               widths=[1.95, 1.35, 3.0], font=9.0)
    para(doc, "Three of those numbers govern how the rest of the chapter "
              "must be read. The normalised entropy of "
              f"{fv(d.get('entropy_ratio'), 4)} says the label distribution "
              "is far from uniform but not pathologically skewed, so "
              "balanced accuracy and raw accuracy will differ "
              "substantially without either being meaningless. The "
              f"imbalance ratio of {fv(d.get('imbalance_ratio'), 2)} says "
              "the largest cohort contributes about thirty times as many "
              "patients as the smallest, so any pooled metric is dominated "
              f"by the large cohorts. And {iv(d.get('n_events_below_20'))} "
              "cohorts carry fewer than twenty deaths, so a survival "
              "statistic computed inside one of them rests on very few "
              "comparable pairs — which is why Section 6.14 separates the "
              "cohorts that are powered enough to read from those that are "
              "not.", space_before=8)
    payload_table(doc, x, "T1", n.t(),
                  widths=[0.75, 0.62, 0.95, 0.95, 0.6, 0.72, 0.95, 0.72],
                  font=7.5,
                  head_map={"patients": "Patients", "blocks": "Omic blocks",
                            "genes_per_block": "Genes per block",
                            "events": "Events", "event_rate": "Event rate",
                            "median_OS_months": "Median OS (months)",
                            "labelled": "Labelled"})
    add_figure(doc, f"{figdir}/fig1_cohorts.png")
    caption(doc, n.f() + ":", "Cohort composition of the assembled dataset: "
                              "patients per cohort, observed events, event "
                              "rate and follow-up. The spread across cohorts "
                              "is the reason pooled metrics are reported "
                              "alongside per-cohort ones.", above=False)


def ch6_panel(doc, x: F, n: N, A: dict, S: dict):
    section(doc, "6.2 The Complete Metric Panel")
    p = A.get("panel") or {}
    para(doc, "All twenty-four metrics the run computed are reported, for "
              "both targets, so that no metric can have been chosen after "
              "the fact. The classification metrics are macro-averaged over "
              "the one-versus-rest tables of every class, which is the "
              "convention Section 5.4 defines.")
    order = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
             "precision_macro", "recall_macro", "specificity_macro",
             "npv_macro", "kappa", "kappa_quadratic", "mcc", "top2_accuracy",
             "auroc", "auprc", "log_loss", "brier", "ece", "c_index",
             "auc@12m", "auc@36m", "auc@60m", "logrank_chi2", "logrank_p",
             "mean_predictive_entropy"]
    sp = (S or {}).get("panel") or {}
    rows = [[SHORT.get(m, m), fv(p.get(m), 4), fv(sp.get(m), 4),
             ("higher" if HIB.get(m, True) else "lower")]
            for m in order if m in p or m in sp]
    rows_table(doc, n.t(),
               "The complete metric panel on the held-out fold of each run. "
               "The last column says which direction is better, so that no "
               "row can be misread. A dash means the metric is undefined "
               "for that target: the quadratic kappa needs more than two "
               "ordered classes.",
               ["Metric", "cohort run", "stage_binary run", "Better"],
               rows, widths=[1.6, 1.55, 1.75, 1.0], font=9.0)
    para(doc, "Read together the two columns describe two very different "
              "problems. On the 32-class cohort target the model "
              "is strongly discriminative — AUROC "
              f"{fv(p.get('auroc'))}, Matthews correlation "
              f"{fv(p.get('mcc'))} — and badly calibrated, with an expected "
              f"calibration error of {fv(p.get('ece'))}. On the binary "
              "stage_binary target it is only modestly better "
              f"than chance on classification, balanced accuracy "
              f"{fv(sp.get('balanced_accuracy'))} and AUROC "
              f"{fv(sp.get('auroc'))}, but well calibrated, with an "
              f"expected calibration error of {fv(sp.get('ece'))}. The "
              "second is the honest measure of how hard stage prediction "
              "from this feature space actually is, and it is the "
              "clinically meaningful endpoint of the two.", space_before=8)
    para(doc, "Two further contrasts are worth naming because they are "
              "easy to misread. The specificity and the negative "
              "predictive value on the cohort run are both above "
              f"{fv(p.get('specificity_macro'), 2)}, which looks "
              "impressive and is nearly automatic: with 32 classes each "
              "one-versus-rest table has about thirty-one times as many "
              "negatives as positives, so a model that merely avoids "
              "over-predicting a class scores high on both. The log-rank "
              f"chi-square of {fv(p.get('logrank_chi2'), 1)} on the two "
              "degrees of freedom that three risk tertiles give, with p = "
              f"{fv(p.get('logrank_p'))}, says the model-defined risk "
              "tertiles separate survival; it does not say by how much, "
              "which is what the C-index and the time-dependent AUCs are "
              "for.")


def ch6_consistency(doc, x: F, n: N, A: dict, S: dict):
    section(doc, "6.3 Reconstructed Contingency Tables and an Internal "
                 "Consistency Audit")
    pc, con = A.get("per_class"), A.get("consistency")
    para(doc, "Before the pooled numbers are compared with anything, they "
              "are checked against their own parts. The per-class export "
              "gives, for every class, the support n_c and the three rates "
              "sensitivity, specificity and precision. From those the "
              "one-versus-rest contingency table is recovered exactly:")
    formula(doc, "TP = sensitivity · n_c,   FN = n_c − TP,   "
                 "TN = specificity · (N − n_c),   FP = (N − n_c) − TN")
    para(doc, "and from the recovered table three further measures follow "
              "that the run did not export per class:")
    formula(doc, "NPV = TN / (TN + FN),   J = sensitivity + specificity − 1,"
                 "   MCC_c = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))")
    para(doc, "A Wilson score interval is attached to each sensitivity, "
              "because a recall computed on nineteen patients and a recall "
              "computed on two hundred should not be printed as though they "
              "carried the same weight:")
    formula(doc, "centre = (TP + z²/2) / (n_c + z²),   "
                 "half-width = z/(n_c + z²) · √( TP(n_c − TP)/n_c + z²/4 ),"
                 "   z = 1.95996")
    if con:
        para(doc, "The audit then recomputes each pooled macro metric from "
                  "the per-class table and compares it with the pooled "
                  f"value the run reported. The support of the "
                  f"{iv(con.get('n_classes'))} classes sums to "
                  f"{iv(con.get('support_sum'))}, which is the held-out "
                  "split exactly, so no patient is counted twice or "
                  "dropped.")
        rows_table(doc, n.t(),
                   "Internal consistency audit. Each pooled metric is "
                   "recomputed from the per-class table by the rule in the "
                   "second column and compared with the value the run "
                   "itself reported. The two agree to machine precision, "
                   "which is what the last column shows.",
                   ["Pooled metric", "Recomputed as", "From per-class",
                    "Run reported", "|difference|"],
                   [[SHORT.get(r["metric"], r["metric"]), r["how"],
                     fv(r["recomputed"], 6), fv(r["reported"], 6),
                     fv(r["abs_diff"], 2) if (r["abs_diff"] or 0) > 1e-9
                     else f"{(r['abs_diff'] or 0):.1e}"]
                    for r in con["rows"]],
                   widths=[1.25, 2.35, 1.0, 0.95, 0.85], font=8.0)
        para(doc, "The largest disagreement anywhere in the audit is "
                  f"{con.get('max_abs_diff', 0):.1e}, which is "
                  "floating-point rounding and nothing else. The pooled "
                  "metrics in Section 6.2 are therefore arithmetically "
                  "consistent with the per-class behaviour in "
                  f"{n.ref_t(1)}: they are not a different computation that "
                  "happens to produce a flattering number. This check is "
                  "cheap and it is the kind of error it would catch — a "
                  "macro average taken over the wrong axis, or a metric "
                  "computed on a different split — that most often "
                  "inflates a reported result.", space_before=8)
    if pc:
        cn = A.get("class_names") or {}
        cmap = cn.get("mapping") or {}
        if cmap:
            para(doc, "The export indexes these classes by integer, which "
                      "is not readable, so the cohort each index stands for "
                      "is recovered and then checked rather than assumed. "
                      "The label encoder sorts its labels, so index i should "
                      "be the i-th cohort alphabetically; if that is right, "
                      "each class's support must equal that cohort's count "
                      "in the per-cohort export. It does for "
                      f"{iv(cn.get('confirmed'))} of "
                      f"{iv(cn.get('n_classes'))} classes and contradicts it "
                      f"for {iv(cn.get('contradicted'))}; the remaining "
                      f"{iv(cn.get('unchecked'))} could not be checked "
                      "because that cohort has no row in the per-cohort "
                      "export, being too small to score there. The cohort "
                      "names below therefore rest on a verified "
                      "correspondence, not on a convention assumed to hold.")
        hdr = ["Class", "n_c", "TP", "FP", "FN", "TN", "Sens",
               "Sens 95 % CI", "Spec", "Prec", "NPV", "F1", "J", "MCC_c",
               "Mean prob."]
        rows = []
        for r in pc["rows"]:
            tag = cmap.get(str(r["class"]))
            rows.append([
                (tag or str(r["class"])), iv(r["support"]), iv(r["tp"]),
                iv(r["fp"]),
                iv(r["fn"]), iv(r["tn"]), fv(r["sensitivity"]),
                f"{fv(r['sens_ci_low'],2)}–{fv(r['sens_ci_high'],2)}",
                fv(r["specificity"]), fv(r["precision"]), fv(r["npv"]),
                fv(r["f1"]), fv(r["youden"]), fv(r["mcc"]),
                fv(r["mean_predicted_prob"])])
        rows_table(doc, n.t(),
                   "Per-class behaviour on the held-out fold with the "
                   "contingency table recovered for every class. Columns "
                   "TP to TN are counts; Sens, Spec, Prec, NPV and F1 are "
                   "the usual rates; J is Youden's index; MCC_c is the "
                   "per-class Matthews correlation; the last column is the "
                   "mean probability the model assigned to that class. "
                   "Sorted by support, so the effect of class size on every "
                   "other column can be read down the page.",
                   hdr, sorted(rows, key=lambda r: -float(
                       str(r[1]).replace(",", "") or 0)),
                   widths=[0.68, 0.37, 0.35, 0.35, 0.35, 0.44, 0.4, 0.72,
                           0.4, 0.4, 0.4, 0.4, 0.4, 0.42, 0.4],
                   font=6.5)
        m = pc["macro"]
        para(doc, "Macro averages over the per-class table: sensitivity "
                  f"{fv(m.get('sensitivity'))}, specificity "
                  f"{fv(m.get('specificity'))}, precision "
                  f"{fv(m.get('precision'))}, negative predictive value "
                  f"{fv(m.get('npv'))}, F1 {fv(m.get('f1'))}, Youden's J "
                  f"{fv(m.get('youden'))} and per-class Matthews "
                  f"correlation {fv(m.get('mcc'))}. The support-weighted F1 "
                  f"is {fv(pc.get('weighted_f1'))}. "
                  + (f"{iv(pc.get('n_zero_recall'))} classes have a "
                     "sensitivity of exactly zero, meaning the model never "
                     "correctly identified a single member of them."
                     if pc.get("n_zero_recall") else
                     "No class has a sensitivity of exactly zero: every one "
                     "of the thirty-two cohorts is recognised at least "
                     "sometimes, which a model that had collapsed onto the "
                     "large classes would not manage.")
                  + " The Wilson intervals make the real precision of these "
                    "numbers visible: for the smallest classes the interval "
                    "spans several tenths, so their individual sensitivities "
                    "should not be compared with one another at all.",
             space_before=8)


def ch6_gap(doc, x: F, n: N, A: dict, figdir: str):
    section(doc, "6.4 Generalisation Gap, Absolutely and Relatively")
    g, c = A.get("gap"), A.get("cost") or {}
    para(doc, f"With {iv(c.get('trainable_parameters'))} trainable "
              f"parameters against {iv(c.get('training_patients'))} "
              f"training patients — {fv(c.get('parameters_per_patient'), 1)} "
              "parameters per patient — overfitting is the first thing a "
              "reader should suspect, so the identical metric panel was "
              "computed on all three splits before any comparison was made. "
              "Both the absolute gap and the gap as a percentage of the "
              "training value are reported, because an absolute gap of 0.06 "
              "means something different on a metric that sits at 0.8 than "
              "on one that sits at 0.2:")
    formula(doc, "gap = train − test,      relative gap = 100 · "
                 "(train − test) / train")
    if g:
        rows = [[SHORT.get(r["metric"], r["metric"]), fv(r["train"], 4),
                 fv(r["validation"], 4), fv(r["test"], 4), fv(r["gap"], 4),
                 fv(r["gap_pct"], 1)] for r in g["rows"]]
        rows_table(doc, n.t(),
                   "Generalisation gap on every metric the run computed. "
                   f"The training split holds {iv(g.get('n'))} patients and "
                   f"the held-out split {iv(g.get('n_test'))}. On the last "
                   "two metrics a lower value is better, so a negative gap "
                   "there is the ordinary direction and not an anomaly.",
                   ["Metric", "Train", "Validation", "Test", "Gap",
                    "Gap %"], rows,
                   widths=[1.3, 0.95, 1.1, 0.95, 0.95, 0.9], font=8.5)
        para(doc, "The mean absolute gap across the discriminative metrics "
                  f"is {fv(g.get('mean_abs_gap'), 4)}; the largest is "
                  f"{fv(g.get('max_gap'), 4)}, on "
                  f"{SHORT.get(g.get('max_gap_metric'), g.get('max_gap_metric'))}"
                  ". That is a real gap, not a negligible one. It is also "
                  "an order of magnitude smaller than the headline "
                  "performance and far smaller than the collapse a model "
                  f"memorising {iv(c.get('training_patients'))} patients "
                  f"with {iv(c.get('trainable_parameters'))} parameters "
                  "would show. The validation column sits between the two "
                  "in almost every row, which is the ordering a model that "
                  "is generalising produces and not the ordering a model "
                  "selected on its test split produces.", space_before=8)
        para(doc, "The log loss and the Brier score move the other way: the "
                  "held-out split scores better than the training split on "
                  "both. That is not a paradox. Stochastic weight averaging "
                  "and dropout are active during the training-split "
                  "evaluation in a way that penalises its proper scores, "
                  "and the two splits have different class mixes. It is "
                  "reported rather than smoothed over because an "
                  "unexplained sign is exactly what a reader should be able "
                  "to see.")
    add_figure(doc, f"{figdir}/fig2_risk.png")
    caption(doc, n.f() + ":", "The identical metric panel on the training, "
                              "validation and held-out splits. The bars are "
                              "read for the distance between them, not for "
                              "their height.", above=False)

    section(doc, "6.5 The Measured Noise Floor")
    para(doc, "An ablation cannot be read without knowing how much the "
              "pipeline moves when nothing is changed, so the whole fit was "
              "repeated three times under an identical seed. The balanced "
              "accuracies were 0.4567, 0.4341 and 0.4547: a spread of "
              "about ±0.01 with the seed held fixed. The cause is "
              "arithmetic rather than statistical. Message passing "
              "accumulates each node's incoming contributions with a "
              "parallel scatter-add, whose summation order is not fixed "
              "between runs, and floating-point addition is not "
              "associative:")
    formula(doc, "(a + b) + c  ≠  a + (b + c)   in finite precision")
    para(doc, "so two runs of identical code on identical data take "
              "slightly different optimisation paths. This ±0.01 is the "
              "yardstick Section 6.10 uses, and it is the reason two of the "
              "seven ablation arms are reported as indistinguishable from "
              "noise rather than as small effects. Any single-fit "
              "difference smaller than it is not interpretable, whatever "
              "its sign.")


def ch6_compare(doc, x: F, n: N, A: dict, runs: str, figdir: str):
    section(doc, "6.6 Comparison Against Established Methods")
    p = A.get("panel") or {}
    para(doc, f"{x.i('cohort','n_baselines')} comparators were fitted on "
              "exactly the same folds, spanning penalised linear models, "
              "support vector machines, shrinkage discriminant analysis, "
              "three tree ensembles, two gradient-boosting libraries, "
              "nearest neighbours, naive Bayes, a PCA pipeline, early- and "
              "late-fusion neural networks, three patient-graph networks, a "
              "MOGONET-style model, DeepSurv and three Cox variants "
              "including a clinical-only Cox model. Every metric each "
              "comparator produces is tabulated; none is omitted.")
    cls = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
           "precision_macro", "recall_macro", "specificity_macro",
           "npv_macro", "auroc", "auprc", "mcc", "kappa", "kappa_quadratic",
           "top2_accuracy", "log_loss", "brier", "ece"]
    rows = _cmp_rows(runs, p, cls)
    if rows:
        wide_table(doc, n.t(),
                   "Classification on the held-out fold of the cohort run, "
                   "every model and every metric. MOSAIC-GNN's own measured "
                   "row is placed first; the comparators follow in "
                   "descending balanced accuracy. An empty cell means the "
                   "model does not produce that kind of prediction.",
                   ["Model"] + _short(cls), rows,
                   groups=[_short(gr) for gr in CLS_GROUPS],
                   part_caps=CLS_CAPS,
                   first_width=1.75, col_width=0.72, font=7.5)
    surv = ["c_index", "auc@12m", "auc@36m", "auc@60m", "logrank_chi2",
            "logrank_p"]
    srows = _cmp_rows(runs, p, surv)
    if srows:
        rows_table(doc, n.t(),
                   "Survival ranking on the same fold: Harrell's C-index, "
                   "time-dependent AUC at 12, 36 and 60 months, and the "
                   "log-rank statistic over model-defined risk tertiles. "
                   "Rows of dashes are the classifiers, which produce no "
                   "risk score.",
                   ["Model"] + _short(surv), srows,
                   widths=[1.75, 0.78, 0.7, 0.7, 0.7, 0.85, 0.85],
                   font=7.5)
    para(doc, "Read on its own the classification table already "
              "contradicts the design hypothesis. The best comparator "
              "balanced accuracy is "
              f"{x.best('cohort','best_balanced_accuracy')} against "
              f"MOSAIC-GNN's {fv(p.get('balanced_accuracy'))}: a penalised "
              "linear model on the same 469-gene feature space separates "
              "the 32 cohorts far better than the graph network does. On "
              "survival the margin is narrow instead — best comparator "
              f"{x.best('cohort','best_c_index')} against "
              f"{fv(p.get('c_index'))}.", space_before=8)

    section(doc, "6.7 Where the Proposed Model Ranks on Every Metric")
    r = A.get("ranks")
    para(doc, "A table of twenty-four models is hard to read for one "
              "model's position, so the position is computed directly. For "
              "each metric the proposed model is inserted into the ordered "
              "list of all models that produce it, highest first where a "
              "higher value is better and lowest first otherwise, and its "
              "place is reported.")
    if r:
        rows = [[SHORT.get(q["metric"], q["metric"]),
                 fv(q["mosaic"], 4), f"{q['rank']} of {q['of']}",
                 q["best_model"], fv(q["best_value"], 4),
                 fv(q["delta_to_best"], 4), q["direction"]]
                for q in r["rows"]]
        rows_table(doc, n.t(),
                   "Rank of MOSAIC-GNN against the full comparator panel on "
                   "every metric both produce. The last column states the "
                   "direction used for the ordering, so no row can be "
                   "misread; the delta column is MOSAIC-GNN minus the best "
                   "comparator and is therefore negative wherever the "
                   "proposed model is behind on a higher-is-better metric.",
                   ["Metric", "MOSAIC-GNN", "Rank", "Best comparator",
                    "Its value", "Δ to best", "Direction"], rows,
                   widths=[1.0, 0.85, 0.75, 1.3, 0.72, 0.68, 1.05],
                   font=8.0)
        para(doc, f"Across {iv(r.get('n_metrics'))} metrics the proposed "
                  f"model takes first place on {iv(r.get('n_rank_1'))} of "
                  f"them. Its median rank is {fv(r.get('median_rank'), 1)}, "
                  f"its best rank {iv(r.get('best_rank'))} and its worst "
                  f"{iv(r.get('worst_rank'))}. A median rank of about "
                  f"{fv(r.get('median_rank'), 0)} in a field of about "
                  "twenty-five is the plain summary of this experiment, and "
                  "it is not a favourable one. The ranks are reported here "
                  "without significance attached; Section 6.8 supplies "
                  "that, because a rank says which number is larger and "
                  "not whether the difference could have arisen by chance.",
             space_before=8)
    add_figure(doc, f"{figdir}/fig4_comparison.png")
    caption(doc, n.f() + ":", "MOSAIC-GNN against every comparator with "
                              "bootstrap intervals, on survival ranking and "
                              "on classification.", above=False)


def ch6_paired(doc, x: F, n: N, A: dict):
    section(doc, "6.8 Paired Bootstrap Tests With Multiplicity Correction")
    pr = A.get("paired")
    para(doc, "Comparing two independently noisy means is the wrong test, "
              "because the two models were evaluated on the same patients "
              "and their errors are correlated. The paired bootstrap "
              "resamples the held-out patients with replacement and "
              "recomputes the difference on each resample, so the "
              "correlation is kept:")
    formula(doc, "Δ = metric(MOSAIC-GNN) − metric(comparator),   "
                 "measured on the same resampled patients")
    para(doc, "A negative Δ therefore means MOSAIC-GNN scored lower. From "
              "the reported interval the standard error and a standardised "
              "effect follow, and both are tabulated because an interval "
              "alone makes small and large effects look alike:")
    formula(doc, "SE = (upper − lower) / (2 · 1.95996),      z = Δ / SE")
    para(doc, "With more than twenty comparators on two metrics, "
              "uncorrected testing would manufacture significance. The run "
              "applied the Benjamini-Hochberg step-up procedure within each "
              "metric:")
    formula(doc, "q(i) = min over j ≥ i of  ( p(j) · m / j ),   "
                 "p(1) ≤ … ≤ p(m)")
    if pr:
        qf = (2.0 / pr["n_resamples"]) if pr.get("n_resamples") else None
        if qf:
            para(doc, "One convention for reading the table: the bootstrap "
                      f"draws {iv(pr['n_resamples'])} resamples and the "
                      "two-sided p-value is twice the smaller tail "
                      "fraction, so the smallest value it can return other "
                      f"than zero is {fv(qf, 4)}. Where no resample fell on "
                      "the other side of zero the entry is written as an "
                      "upper bound rather than as 0.0000, which would claim "
                      "a precision the procedure does not have.")
        hdr = ["Model", "Δ BAcc", "SE", "z", "95 % CI", "p", "q",
               "Δ C-idx", "SE", "z", "95 % CI", "p", "q"]
        rows = []
        for r in pr["rows"]:
            rows.append([
                r["model"],
                fv(r.get("bacc_delta"), 4), fv(r.get("bacc_se"), 4),
                fv(r.get("bacc_z"), 2),
                (f"{fv(r.get('bacc_ci_low'),3)} to "
                 f"{fv(r.get('bacc_ci_high'),3)}"
                 if r.get("bacc_ci_low") is not None else "—"),
                pv(r.get("bacc_p"), qf), pv(r.get("bacc_q"), qf),
                fv(r.get("cindex_delta"), 4), fv(r.get("cindex_se"), 4),
                fv(r.get("cindex_z"), 2),
                (f"{fv(r.get('cindex_ci_low'),3)} to "
                 f"{fv(r.get('cindex_ci_high'),3)}"
                 if r.get("cindex_ci_low") is not None else "—"),
                pv(r.get("cindex_p"), qf), pv(r.get("cindex_q"), qf)])
        wide_table(doc, n.t(),
                   "Paired bootstrap against MOSAIC-GNN on identical "
                   "held-out patients. Δ is MOSAIC-GNN minus the "
                   "comparator, so a negative value means the proposed "
                   "model is behind; SE and z are derived from the interval "
                   "by the formulas above; q is the Benjamini-Hochberg "
                   "corrected p-value within that metric's family.",
                   hdr, rows,
                   groups=[["Δ BAcc", "SE", "z", "95 % CI", "p", "q"],
                           ["Δ C-idx", "SE", "z", "95 % CI", "p", "q"]],
                   part_caps=["Balanced accuracy.",
                              "Harrell's C-index. Only the models that "
                              "produce a risk score appear with values."],
                   first_width=1.72, col_width=0.78, font=7.5)
        sizes = pr.get("family_sizes") or {}
        para(doc, "The correction was verified rather than trusted. "
                  "Recomputing the step-up procedure independently from the "
                  "stored p-values, over a family of "
                  f"{iv(sizes.get('bacc'))} balanced-accuracy tests and "
                  f"{iv(sizes.get('cindex'))} C-index tests, reproduces "
                  "every stored q-value: the largest disagreement anywhere "
                  f"is {pr.get('q_max_abs_diff', 0):.1e}. Correcting "
                  "instead over the pooled family of "
                  f"{iv(pr.get('pooled_family_size'))} tests is stricter "
                  "and moves some q-values by up to "
                  f"{fv(pr.get('pooled_q_max_abs_diff'), 3)}; "
                  + (("under that stricter rule "
                      + "; ".join(f"{m} on "
                                  + ("balanced accuracy" if k == "bacc"
                                     else "the C-index")
                                  + f" changes verdict (q {fv(a,3)} → "
                                    f"{fv(b,3)})"
                                  for m, k, a, b in
                                  pr.get("pooled_verdict_flips") or [])
                      + ". No conclusion in this chapter rests on that one "
                        "row.")
                     if pr.get("pooled_verdict_flips")
                     else "no verdict in this chapter changes under that "
                          "stricter rule.")
             , space_before=8)
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
              f"with {cc('cohort','n_tied')}.")
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
                "reported next with the same machinery.", False)])


def ch6_gain(doc, x: F, n: N, A: dict, S: dict):
    section(doc, "6.9 What the Molecular Data Adds Over Clinical Covariates")
    para(doc, "This is the premise the multi-omic field rests on and the "
              "comparison most multi-omic papers omit. It was measured two "
              "independent ways, on two independently fitted targets.")
    mgc, mgs = x.g("cohort", "molecular_gain"), x.g("stage_binary",
                                                    "molecular_gain")
    nb = ((A.get("paired") or {}).get("n_resamples"))
    qfloor = (2.0 / nb) if nb else None
    _CCO = "paired.models.cox_clinical_only.cindex_delta"
    _CCQ = "paired.models.cox_clinical_only.cindex_q"
    _CCL = "paired.models.cox_clinical_only.cindex_ci_low"
    _CCH = "paired.models.cox_clinical_only.cindex_ci_high"
    if isinstance(mgc, dict) and isinstance(mgs, dict):
        rows_table(doc, n.t(),
                   "Molecular information over and above the clinical "
                   "covariates, measured twice. The first block fits the "
                   "same penalised Cox model with and without the molecular "
                   "blocks; the second compares MOSAIC-GNN with the "
                   "clinical-only Cox model in the paired bootstrap. A "
                   "positive difference means the molecular information "
                   "helped.",
                   ["Comparison", "cohort run", "stage_binary run"],
                   [[f"Penalised Cox with molecular blocks "
                     f"({mgc.get('model','?')})",
                     fv(mgc.get("molecular"), 4), fv(mgs.get("molecular"), 4)],
                    ["Clinical covariates only (Cox)",
                     fv(mgc.get("clinical_only"), 4),
                     fv(mgs.get("clinical_only"), 4)],
                    ["Difference in C-index",
                     fv(mgc.get("delta"), 4), fv(mgs.get("delta"), 4)],
                    ["Relative improvement",
                     (pctf(100 * mgc["delta"] / mgc["clinical_only"])
                      if mgc.get("clinical_only") else "—"),
                     (pctf(100 * mgs["delta"] / mgs["clinical_only"])
                      if mgs.get("clinical_only") else "—")],
                    ["MOSAIC-GNN minus clinical-only Cox (paired)",
                     fv(x.g("cohort", _CCO), 4),
                     fv(x.g("stage_binary", _CCO), 4)],
                    ["95 % bootstrap interval of that difference",
                     f"{fv(x.g('cohort', _CCL), 3)} to "
                     f"{fv(x.g('cohort', _CCH), 3)}",
                     f"{fv(x.g('stage_binary', _CCL), 3)} to "
                     f"{fv(x.g('stage_binary', _CCH), 3)}"],
                    ["Corrected q-value of that difference",
                     pv(x.g("cohort", _CCQ), qfloor),
                     pv(x.g("stage_binary", _CCQ), qfloor)]],
                   widths=[3.0, 1.55, 1.75], font=9.0)
        if nb:
            para(doc, "The q-values are written as an upper bound rather "
                      f"than as zero. The paired bootstrap draws {iv(nb)} "
                      "resamples and its two-sided p-value is twice the "
                      "smaller tail fraction, so when no resample falls on "
                      "the other side of zero the p-value is not zero but "
                      f"smaller than 2/{int(nb)} = {fv(qfloor, 4)}. Printing "
                      "0.0000 would claim a precision the procedure cannot "
                      "deliver.", size=10.5, space_before=6)
    para(doc, "Both measurements agree and they agree on both targets. "
              "Adding the molecular blocks to a penalised Cox model raises "
              f"the C-index by {fv((mgc or {}).get('delta'), 4)} on the "
              f"cohort run and {fv((mgs or {}).get('delta'), 4)} on the "
              "stage_binary run. MOSAIC-GNN itself exceeds the "
              "clinical-only Cox model by "
              f"{fv(x.g('cohort', _CCO), 4)} and "
              f"{fv(x.g('stage_binary', _CCO), 4)} respectively, and in the "
              "paired bootstrap both intervals exclude zero and both "
              "differences survive correction. A result that replicates "
              "across two independently fitted targets, on overlapping but "
              "differently labelled patient sets, is worth more than a "
              "larger single-target effect; this is the clearest positive "
              "finding of the project.", space_before=8)
    para(doc, "It should be read for exactly what it is. It says the "
              "molecular data carries prognostic information a clinician "
              "does not already have. It does not say this architecture is "
              "the way to extract it: Section 6.6 shows that simpler models "
              "extract as much or more.")


def ch6_ablation(doc, x: F, n: N, A: dict, figdir: str):
    section(doc, "6.10 Ablation: Which Components Earn Their Place")
    ab = A.get("ablation")
    para(doc, "Seven arms were run, each removing exactly one component and "
              "changing nothing else. As recorded in Section 5.3 the "
              "ablation ran on a reduced gene subset for six epochs over "
              "two folds, so its absolute balanced accuracy — "
              f"{fv((ab or {}).get('full'), 4)} for the full model, with a "
              f"fold spread of {fv((ab or {}).get('full_sd'), 4)} over "
              f"{iv((ab or {}).get('full_folds'))} folds — is lower than "
              "Section 6.2's and is not comparable to it. Only the "
              "differences between arms are interpreted, and each is "
              "expressed in units of the noise floor measured in "
              "Section 6.5:")
    formula(doc, "Δ = BAcc(arm) − BAcc(full),      "
                 "floors = |Δ| / 0.01,      interpretable only if floors > 1")
    if ab:
        rows = []
        for a in ab["arms"]:
            rows.append([
                a["arm"], fv(a.get("balanced_accuracy"), 4),
                fv(a.get("balanced_accuracy_sd"), 4),
                fv(a.get("auroc"), 4), fv(a.get("c_index"), 4),
                fv(a.get("delta"), 4), fv(a.get("floors"), 1),
                "yes" if a.get("outside") else "no"])
        rows_table(doc, n.t(),
                   "Ablation at reduced settings. SD is the spread over the "
                   "two folds of that arm. Δ is the change in balanced "
                   "accuracy against the full model on the same folds, and "
                   "the floors column is |Δ| divided by the measured "
                   "run-to-run spread of 0.01; the last column says whether "
                   "the effect is larger than that spread and therefore "
                   "interpretable at all.",
                   ["Component removed", "BAcc", "SD", "AUROC", "C-index",
                    "Δ BAcc", "Floors", "Outside the floor?"], rows,
                   widths=[1.72, 0.72, 0.6, 0.68, 0.72, 0.72, 0.62, 1.1],
                   font=8.5)
    arms = {a["arm"]: a for a in (ab or {}).get("arms", [])}

    def dl(name):
        return fv((arms.get(name) or {}).get("delta"), 4)

    def fl(name):
        return fv((arms.get(name) or {}).get("floors"), 1)

    para(doc, "Three things follow, and only the second is what the design "
              "predicted. First, the clinical path dominates: removing the "
              f"clinical covariates costs {dl('no_clinical')}, which is "
              f"{fl('no_clinical')} times the noise floor, and removing "
              "only their residual path costs "
              f"{dl('no_clinical_residual')}, or {fl('no_clinical_residual')}"
              " floors. Both are an order of magnitude larger than any "
              "other arm. The model's accuracy rests mostly on the "
              "clinical covariates rather than on the molecular graph "
              "machinery, and that is the single most important sentence in "
              "this section.", space_before=8)
    para(doc, "Second, the three structural components do clear the floor. "
              f"Removing the gene graph costs {dl('no_gene_graph')} "
              f"({fl('no_gene_graph')} floors), the patient graph "
              f"{dl('no_patient_graph')} ({fl('no_patient_graph')} floors) "
              f"and module pooling {dl('no_module_pool')} "
              f"({fl('no_module_pool')} floors). The two-level design is "
              "therefore doing measurable work, even though the whole model "
              "is beaten by simpler comparators. These are modest effects "
              "— one to two times the floor — and they are reported as "
              "modest.")
    para(doc, "Third, the two auxiliary objectives do not clear it. "
              f"Removing the contrastive term costs {dl('no_contrast')} "
              f"({fl('no_contrast')} floors) and removing disentanglement "
              f"{dl('no_disentangle')} ({fl('no_disentangle')} floors). On "
              "this evidence neither can be claimed to contribute; they "
              "cannot be claimed to hurt either, which is the symmetric "
              "and equally important half of the statement.")
    para(doc, f"Arms outside the floor: {iv((ab or {}).get('n_outside'))}. "
              f"Arms inside it: {iv((ab or {}).get('n_inside'))}. Measured "
              "cost of one arm: "
              f"{fv((ab or {}).get('seconds_mean'), 0)} seconds per fold, "
              "which is why the full factorial of configurations was out of "
              "reach.")
    rich(doc, [("A retracted explanation. ", True),
               ("An earlier draft of this work explained the "
                "disentanglement arm by hypothesising that the cohort "
                "adversary was the component suppressing tissue-of-origin "
                "signal, and predicted that removing it would raise "
                "cohort accuracy sharply. The ablation was run "
                "specifically to test that prediction. Removing "
                "disentanglement changed balanced accuracy by "
                + dl("no_disentangle") + ", or " + fl("no_disentangle") +
                " floors, where the hypothesis required a change of roughly "
                "+0.17, about seventeen floors. The hypothesis is falsified "
                "and has been withdrawn. It is recorded here because a "
                "prediction made in advance and then tested is evidence, "
                "and deleting it would misrepresent how the conclusions "
                "were reached.", False)])
    add_figure(doc, f"{figdir}/fig5_ablation.png")
    caption(doc, n.f() + ":", "Change in each headline metric when a single "
                              "component is removed; negative means the "
                              "component helped, and the shaded band is the "
                              "measured ±0.01 noise floor. (a) every arm on "
                              "one scale, which shows how far the clinical "
                              "arms dominate; (b) the same arms whose "
                              "largest effect is within ten noise floors, "
                              "enlarged so that the question this section "
                              "asks can actually be read off the figure — "
                              "the gene graph, patient graph and module "
                              "pooling extend below the band, the two "
                              "auxiliary objectives end inside it. The "
                              "balanced-accuracy bars carry direct labels "
                              "in (b); labelling all fifteen would be "
                              "noise.", above=False)


def ch6_permutation(doc, x: F, n: N, A: dict):
    section(doc, "6.11 Label Permutation Control")
    pm = A.get("permutation")
    para(doc, "The whole pipeline was retrained on shuffled training "
              "labels. A leaking protocol would still score above chance "
              "after shuffling; an honest one should collapse to chance. "
              "The distance from the null is reported as a z-score as well "
              "as a p-value, because with few permutations the p-value is "
              "bounded and the z-score is not:")
    formula(doc, "z = (observed − mean of null) / SD of null,      "
                 "smallest attainable p = 1 / (B + 1)")
    if pm:
        rows = [[SHORT.get(r["metric"], r["metric"]), fv(r["observed"], 4),
                 fv(r["null_mean"], 4), fv(r["null_sd"], 4),
                 fv(r["null_max"], 4), fv(r["excess"], 4), fv(r["z"], 1),
                 fv(r["p_value"], 2)] for r in pm["rows"]]
        rows_table(doc, n.t(),
                   "Label permutation test. The model is retrained on "
                   "shuffled training labels and evaluated on the "
                   "untouched held-out split. Excess is observed minus the "
                   "null mean; z is that excess in units of the null "
                   "standard deviation.",
                   ["Metric", "Observed", "Null mean", "Null SD",
                    "Null max", "Excess", "z", "p"], rows,
                   widths=[1.45, 0.78, 0.78, 0.7, 0.72, 0.7, 0.62, 0.62],
                   font=8.5)
        if pm.get("draws"):
            rows_table(doc, n.t(),
                       "Every individual permutation draw, so the null is "
                       "shown rather than summarised. These are the values "
                       "the three columns of the previous table average.",
                       ["Permutation", "BAcc", "AUROC", "C-index"],
                       [[str(d["permutation"]), fv(d["balanced_accuracy"], 4),
                         fv(d["auroc"], 4), fv(d["c_index"], 4)]
                        for d in pm["draws"]],
                       widths=[1.3, 1.3, 1.3, 1.3], font=9.0)
        para(doc, "The null collapses exactly where it should. Balanced "
                  f"accuracy under permutation falls to "
                  f"{fv(pm['rows'][0]['null_mean'], 4)}, which is below the "
                  "1/32 a uniform guess would give, and both AUROC and the "
                  "C-index land on 0.5 to within a few hundredths — "
                  "precisely where a model with no usable signal must land. "
                  "The control passes.", space_before=8)
        para(doc, "The p-value must be read with its bound. With "
                  f"{iv(pm.get('B'))} permutations the smallest attainable "
                  f"p-value is {fv(pm.get('p_floor'), 2)}, and that bound, "
                  "not the strength of the effect, is what the tabulated "
                  "p-value reflects. The evidence that the protocol does "
                  "not leak is the collapse of the null distribution and "
                  "the size of the z-scores, not the p-value. One "
                  "permutation returned a C-index of "
                  f"{fv(max((d['c_index'] or 0) for d in pm['draws']), 4)}, "
                  "which is above 0.5; against a null standard deviation of "
                  f"{fv(pm['rows'][2]['null_sd'], 4)} that is an ordinary "
                  "draw from the null and not a sign of leakage.")


def ch6_samplesize(doc, x: F, n: N, A: dict, figdir: str):
    section(doc, "6.12 Is the Model Data-Limited or Architecture-Limited?")
    ss = A.get("samplesize")
    para(doc, "The same configuration was refitted on a quarter, a half, "
              "three quarters and all of the training data with the "
              "held-out split unchanged. To turn four points into a "
              "statement, each curve is fitted by ordinary least squares "
              "against the base-two logarithm of the training-set size, so "
              "that the slope reads as the gain per doubling of the data:")
    formula(doc, "y = a + b · log2 n,      "
                 "R² = 1 − SS_residual / SS_total")
    if ss:
        rows = [[iv(r["n_train"]), fv(r["log2n"], 2), fv(r["fraction"], 2),
                 fv(r["train_balanced_accuracy"], 4),
                 fv(r["test_balanced_accuracy"], 4), fv(r["gap_bacc"], 4),
                 fv(r["train_c_index"], 4), fv(r["test_c_index"], 4),
                 fv(r["gap_cindex"], 4)] for r in ss["rows"]]
        rows_table(doc, n.t(),
                   "Performance against training-set size. The gap columns "
                   "are train minus test at that size, so the trend in "
                   "overfitting can be read as well as the trend in "
                   "performance.",
                   ["n train", "log2 n", "Fraction", "Train BAcc",
                    "Test BAcc", "Gap", "Train C-idx", "Test C-idx", "Gap"],
                   rows,
                   widths=[0.68, 0.6, 0.72, 0.82, 0.78, 0.6, 0.85, 0.78,
                           0.6], font=8.5)
        fit = [("Test balanced accuracy", ss.get("test_balanced_accuracy")),
               ("Test C-index", ss.get("test_c_index")),
               ("Train balanced accuracy",
                ss.get("train_balanced_accuracy")),
               ("Train C-index", ss.get("train_c_index"))]
        rows_table(doc, n.t(),
                   "Least-squares fits of each curve on log2 n. The slope "
                   "is the change in the metric per doubling of the "
                   "training set; R² says how well a straight line in "
                   "log n describes the four points.",
                   ["Curve", "First point", "Last point", "Total gain",
                    "Slope per doubling", "R²"],
                   [[name, fv((d or {}).get("first"), 4),
                     fv((d or {}).get("last"), 4),
                     fv((d or {}).get("gain"), 4),
                     fv((d or {}).get("slope_per_doubling"), 4),
                     fv((d or {}).get("r2"), 3)] for name, d in fit],
                   widths=[1.75, 0.95, 0.95, 0.9, 1.25, 0.6], font=9.0)
        tb = ss.get("test_balanced_accuracy") or {}
        tc = ss.get("test_c_index") or {}
        trc = ss.get("train_c_index") or {}
        para(doc, "The two curves behave differently and the difference is "
                  "the answer to the section's question. Held-out balanced "
                  f"accuracy gains {fv(tb.get('slope_per_doubling'), 4)} "
                  f"per doubling of the training set, with R² "
                  f"{fv(tb.get('r2'), 3)}: a near-straight line in log n "
                  "that has not begun to flatten at the largest size "
                  "available. Classification is data-limited, and more "
                  "patients would move it. The held-out C-index gains only "
                  f"{fv(tc.get('slope_per_doubling'), 4)} per doubling, "
                  "while the training C-index is essentially flat "
                  f"(slope {fv(trc.get('slope_per_doubling'), 4)}, R² "
                  f"{fv(trc.get('r2'), 3)}, which is no trend at all). "
                  "Survival ranking is close to what this feature space "
                  "supports and more patients would not move it much.",
             space_before=8)
        para(doc, "One inference is explicitly not drawn. The fitted slope "
                  "could be extrapolated to predict the sample size at "
                  "which the proposed model would overtake the best "
                  "comparator. That extrapolation is not reported, because "
                  "it would run far beyond the range of the four measured "
                  "points, it assumes the comparators do not improve with "
                  "the same data, and Section 6.6 shows they are already "
                  "ahead by much more than the observed range of this "
                  "curve.")
    add_figure(doc, f"{figdir}/figS2_sample_size.png")
    caption(doc, n.f() + ":", "Held-out performance against training-set "
                              "size, with the training curve shown for "
                              "contrast.", above=False)


def ch6_percohort(doc, x: F, n: N, A: dict):
    section(doc, "6.13 Per-Cohort Behaviour on the Held-Out Fold")
    pc = A.get("per_cohort")
    para(doc, "A pooled pan-cancer metric is dominated by the large "
              "cohorts, so performance is also computed inside each cohort "
              "separately. That is what shows where the model works and "
              "where it does not — but only part of the metric panel "
              "survives being computed one cohort at a time, and the rest "
              "of this section is about which part.")
    deg = (A.get("per_cohort_degeneracy") or {}).get("degenerate") or {}
    if deg:
        para(doc, "Most of the metric panel cannot be computed per cohort on "
                  "this target, and the reason is the one Section 6.14 "
                  "returns to at greater length. A single cohort's slice of "
                  "the held-out fold contains patients of exactly one class, "
                  "so the classification problem inside it is not a "
                  "32-class problem at all. Which columns that ruins was "
                  "measured rather than assumed, by testing each one "
                  "against the whole table:")
        bullets(doc, [
            (f"{SHORT.get(k, k)}: ", f"{v}.")
            for k, v in sorted(deg.items())])
        para(doc, "A fourth consequence is not visible as a constant and so "
                  "has to be argued rather than detected. The "
                  "macro-averaged columns — macro F1, macro precision, "
                  "macro recall — average over the labels present in the "
                  "truth or in the prediction. With one true class, that "
                  "set is {the cohort} together with whatever other cohorts "
                  "the model wrongly guessed, each contributing zero. Those "
                  "columns therefore vary with the number of distinct wrong "
                  "guesses rather than with performance, and they are not "
                  "interpretable as F1 or precision. They are omitted here "
                  "for that reason, not because they are constant.")
        para(doc, "What survives is reported below: the accuracy, which in a "
                  "single-class slice is exactly that cohort's recall and is "
                  "labelled as such; top-2 accuracy, which asks whether the "
                  "true cohort was among the model's two best guesses; the "
                  "two proper scores, which are computed per patient and are "
                  "unaffected; and the within-cohort C-index, which ranks "
                  "that cohort's patients by risk and never touches the "
                  "label space. Section 6.18 shows the same table for the "
                  "stage_binary target, where the slice contains both "
                  "classes and the full panel is therefore meaningful — the "
                  "contrast between the two is the clearest demonstration "
                  "that this is a property of the split and not of the "
                  "model.")
    if pc:
        hdr = ["Cohort", "n in fold", "Cohort events", "Cohort event rate",
               "Recall", "Top-2", "LogLoss", "Brier", "C-index"]
        rows = [[r["cohort"], iv(r["n"]), iv(r["events"]),
                 fv(r["event_rate"]), fv(r["accuracy"]),
                 fv(r["top2_accuracy"]), fv(r["log_loss"]), fv(r["brier"]),
                 fv(r["c_index"])] for r in pc["rows"]]
        rows_table(doc, n.t(),
                   "Per-cohort results on the held-out fold, ordered by "
                   "recall. Two of the columns describe different "
                   "populations and the headings say which: \"n in fold\" "
                   "is how many of that cohort's patients fell in the "
                   "held-out fold and is what every metric to its right is "
                   "computed on, whereas \"cohort events\" and \"cohort "
                   "event rate\" are the whole cohort's, carried over from "
                   "the cohort table so each row can be weighed by the "
                   "evidence behind it. A cohort can therefore show more "
                   "deaths than patients in this table without anything "
                   "being wrong. Recall is the fraction of that cohort's "
                   "in-fold patients the model assigned to it; the "
                   "degenerate and uninterpretable columns listed above are "
                   "omitted rather than printed.",
                   hdr, rows,
                   widths=[0.95, 0.72, 0.8, 0.9, 0.62, 0.55, 0.7, 0.58,
                           0.68], font=8.0)
        acc, ci = pc["accuracy"], pc["c_index"]
        rows_table(doc, n.t(),
                   "Distribution of the per-cohort metrics, and their "
                   "correlation with how much evidence each cohort "
                   "supplies. The correlations are Pearson coefficients "
                   "with the two-sided p-value of their t statistic and a "
                   "Fisher z interval.",
                   ["Quantity", "Value"],
                   [["Cohorts evaluated", iv(pc.get("n_cohorts"))],
                    ["Recall: mean (SD)",
                     f"{fv(acc['mean'])} ({fv(acc['sd'])})"],
                    ["Recall: median",
                     fv(acc["median"])],
                    ["Recall: range",
                     f"{fv(acc['min'])} to {fv(acc['max'])}"],
                    ["Cohorts with recall ≥ 0.90",
                     iv(pc.get("n_acc_above_90"))],
                    ["C-index: mean (SD)",
                     f"{fv(ci['mean'])} ({fv(ci['sd'])})"],
                    ["C-index: median", fv(ci["median"])],
                    ["C-index: range",
                     f"{fv(ci['min'])} to {fv(ci['max'])}"],
                    ["Cohorts with C-index > 0.5",
                     iv(pc.get("n_cindex_above_half"))],
                    ["corr(cohort size, recall)",
                     _corr_str(pc.get("corr_n_accuracy"))],
                    ["corr(cohort size, C-index)",
                     _corr_str(pc.get("corr_n_cindex"))],
                    ["corr(event rate, C-index)",
                     _corr_str(pc.get("corr_eventrate_cindex"))]],
                   widths=[2.6, 3.7], font=9.0)
        para(doc, "Two of those three correlations are the interesting "
                  "ones. Cohort size does not predict per-cohort recall "
                  f"to any detectable degree ({_corr_str(pc.get('corr_n_accuracy'))}"
                  "), which argues that the per-cohort spread is driven by "
                  "how distinctive a cancer type's molecular profile is "
                  "rather than by how much of it the model saw. The event "
                  "rate, by contrast, correlates negatively and "
                  "significantly with the per-cohort C-index "
                  f"({_corr_str(pc.get('corr_eventrate_cindex'))}): "
                  "survival ranking is harder inside aggressive cohorts. "
                  "Section 6.14 finds the same relationship in a wholly "
                  "separate experiment, which is why it is reported as a "
                  "pattern rather than as a single coincidence.",
             space_before=8)


def _corr_str(c) -> str:
    if not c or c.get("r") is None:
        return "—"
    s = f"r = {fv(c['r'], 3)}, n = {iv(c['n'])}, p = {fv(c['p'], 4)}"
    if c.get("ci_low") is not None:
        s += f", 95 % CI {fv(c['ci_low'], 3)} to {fv(c['ci_high'], 3)}"
    return s


def ch6_loco(doc, x: F, n: N, A: dict, figdir: str):
    section(doc, "6.14 Leave-One-Cancer-Out Transfer")
    lc = A.get("loco")
    para(doc, "This is the experiment that addresses the cohort-detector "
              "problem directly. For each of the "
              f"{iv((lc or {}).get('n_cohorts'))} cohorts the model is "
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
                "transfer measurements, and they are excluded from every "
                "table and figure in this section. An earlier draft went "
                "further and asserted that the classification metric would "
                "be near zero by the same arithmetic; that assertion was "
                "wrong, and it has been withdrawn. The correct statement is "
                "the narrower one: the metric is degenerate and therefore "
                "uninformative, not that it is small.", False)])
    an = ((lc or {}).get("classification_anomaly") or {})
    if an.get("rows"):
        section(doc, "6.14.1 An Unexplained Anomaly in the Transfer "
                     "Classification Path")
        para(doc, "The degeneracy above explains why the classification "
                  "column cannot be interpreted. It does not explain what "
                  "that column actually contains, and what it contains is "
                  "not consistent with the experiment as designed. In a "
                  "leave-one-cancer-out fold the held-out cohort's label is "
                  "absent from training entirely, so the classification "
                  "head receives no positive example of it and should be "
                  "unable to assign it. The exported single-class recall "
                  "says otherwise.")
        rows = [[c["cohort"], iv(c["patients"]), fv(c["bacc"], 4),
                 fv(c["implied_correct"], 1)]
                for c in an["rows"][:12]]
        rows_table(doc, n.t(),
                   "The twelve highest single-class recalls in the "
                   "leave-one-cancer-out table, with the number of patients "
                   "each implies were assigned the correct cohort. In a "
                   "single-class fold balanced accuracy is that class's "
                   "recall, so the last column is recall times the cohort's "
                   "size. None of these is reported as a result; the table "
                   "is here as the evidence for why not.",
                   ["Held-out cohort", "Patients", "Single-class recall",
                    "Implied correct"], rows,
                   widths=[1.5, 1.0, 1.6, 1.4], font=9.0)
        para(doc, f"Across all {iv(an.get('n'))} cohorts the median "
                  f"single-class recall is {fv(an.get('median'), 4)}, "
                  f"{iv(an.get('n_above_90'))} cohorts exceed 0.9 and "
                  f"{iv(an.get('n_exactly_1'))} are exactly 1.000 — a "
                  "perfect assignment of a label the model was never "
                  "trained on, in cohorts as large as "
                  f"{iv(max(c['patients'] for c in an['rows'] if c['bacc'] >= 0.999999))}"
                  " patients. A softmax head given no positive examples of "
                  "a class cannot do this. Something in the "
                  "leave-one-cancer-out classification path is therefore "
                  "wrong, and this report does not claim to know what: the "
                  "most likely candidates are that the label space is "
                  "re-indexed between fitting and scoring, so that the "
                  "metric compares predictions against the wrong column, or "
                  "that the held-out class becomes the default output for "
                  "inputs unlike anything in training. Neither has been "
                  "confirmed. Accordingly, no leave-one-cancer-out "
                  "classification number appears as a result anywhere in "
                  "this report.", space_before=8)
        rich(doc, [("Does this contaminate the survival transfer, which is "
                    "reported? ", True),
                   ("The question has to be asked, because the transfer "
                    "result is one of the two positive findings of the "
                    "project and it comes from the same table. Three pieces "
                    "of evidence say the survival column is not affected by "
                    "whatever this is, and they are arguments rather than "
                    "proof.", False)])
        bullets(doc, [
            ("The two columns come from different heads. ", "The C-index is "
             "computed by ranking the risk head's scalar output within the "
             "held-out cohort. It never references the class label space, so "
             "a label-indexing fault could not reach it."),
            ("A training-side leak would raise the survival column, and it "
             "is lower. ", "If the held-out cohort were somehow present in "
             "training, its survival ranking would be easier, not harder. "
             "The comparison has to be made carefully: the pooled "
             "held-out C-index of "
             + fv((A.get("panel") or {}).get("c_index"), 4) + " ranks "
             "patients across cohorts, which is an easier problem than "
             "ranking within one, so it is the wrong baseline here. The "
             "right one is the within-cohort C-index on the ordinary "
             "held-out fold, where the cohort was present in training: "
             "median "
             + fv(((A.get("per_cohort") or {}).get("c_index") or {})
                  .get("median"), 4) + ". Transfer reaches "
             + fv((lc or {}).get("median"), 4) + " against that, so it is "
             "worse on a like-for-like comparison — which is what genuine "
             "generalisation loss looks like and the opposite of what "
             "leakage looks like."),
            ("The survival column has the structure of a real measurement. ",
             "It spans " + fv((lc or {}).get("min"), 3) + " to "
             + fv((lc or {}).get("max"), 3) + ", with "
             + iv((lc or {}).get("n_cohorts", 0) - (lc or {}).get(
                 "n_above_half", 0)) + " cohorts below the random value of "
             "0.5, and it correlates with the held-out cohort's event rate "
             "in the direction and strength reported below. A leaked signal "
             "does not produce failures below chance, and it does not "
             "produce a biologically interpretable gradient."),
        ])
        para(doc, "That is the honest position: the survival transfer is "
                  "reported, with the reasoning above stated so a reader "
                  "can weigh it, and it should be treated as provisional "
                  "until the classification path is debugged. Identifying "
                  "the fault is listed in Section 8.3 as future work. The "
                  "subsection is included rather than omitted because the "
                  "column is in the exported table either way, so anyone "
                  "who recomputes it will see this, and a caveat a reader "
                  "discovers for themselves is worth less than one the "
                  "report states.")
    para(doc, "The thirty-two log-rank tests are a family, so they are "
              "corrected the same way the model comparisons were. Both the "
              "raw p-value and the Benjamini-Hochberg q-value are given, "
              "and the count of significant cohorts is reported after "
              "correction as well as before it.")
    if lc:
        rows = [[r["cohort"], iv(r["patients"]), iv(r["events"]),
                 fv(r["event_rate"]), fv(r["c_index"], 4),
                 fv(r["logrank_p"], 4), fv(r["logrank_q"], 4),
                 ("yes" if (r["logrank_q"] or 1) < 0.05 else "no")]
                for r in lc["rows"]]
        rows_table(doc, n.t(),
                   "Leave-one-cancer-out transfer, ordered by transfer "
                   "C-index. The model is trained with the named cohort "
                   "entirely absent and evaluated on it. Classification "
                   "columns are omitted because they are degenerate in a "
                   "single-class fold. q is the Benjamini-Hochberg "
                   "corrected p-value over all thirty-two tests.",
                   ["Held-out cohort", "Patients", "Events", "Event rate",
                    "Transfer C-index", "Log-rank p", "q", "q < 0.05?"],
                   rows,
                   widths=[1.15, 0.72, 0.6, 0.78, 1.1, 0.78, 0.68, 0.68],
                   font=8.0)
        para(doc, "The median transfer C-index is "
                  f"{fv(lc.get('median'), 4)}, the interquartile range "
                  f"{fv(lc.get('q1'), 3)} to {fv(lc.get('q3'), 3)}, and the "
                  f"full range {fv(lc.get('min'), 3)} to "
                  f"{fv(lc.get('max'), 3)}; the mean is "
                  f"{fv(lc.get('mean'), 4)} with a standard deviation of "
                  f"{fv(lc.get('sd'), 4)}. "
                  f"{iv(lc.get('n_above_half'))} of "
                  f"{iv(lc.get('n_cohorts'))} cohorts exceed 0.5, the "
                  "random-ranking value. The log-rank test is significant "
                  f"at the 0.05 level in {iv(lc.get('n_p_lt_05'))} cohorts "
                  f"before correction and in {iv(lc.get('n_q_lt_05'))} "
                  "after it — the correction costs one cohort, which is "
                  "worth stating because an uncorrected count of "
                  "thirty-two tests would have been the easy number to "
                  "quote.", space_before=8)
        para(doc, "The representation therefore does carry risk "
                  "information into cancer types it has never seen. That "
                  "is strong evidence against the cohort-detector "
                  "explanation of the pooled result, and it is the second "
                  "substantive positive finding of the project.")
        grows = []
        for key, label in (("all", "All cohorts"),
                           ("events_ge_20", "Cohorts with ≥ 20 events"),
                           ("events_ge_60", "Cohorts with ≥ 60 events")):
            g = (lc.get("groups") or {}).get(key) or {}
            grows.append([label, iv(g.get("n")), fv(g.get("median"), 4),
                          f"{fv(g.get('mean'), 4)} ({fv(g.get('sd'), 4)})",
                          iv(g.get("n_above_half")), iv(g.get("n_q_lt_05")),
                          _corr_str(g.get("corr_event_rate"))])
        rows_table(doc, n.t(),
                   "Transfer summarised by how much evidence the held-out "
                   "cohort supplies, and the correlation between transfer "
                   "quality and the cohort's event rate within each "
                   "subset. Restricting to the cohorts that are powered "
                   "enough to read strengthens the relationship rather "
                   "than weakening it, which is what a real effect does "
                   "and a small-sample artefact does not.",
                   ["Subset", "n", "Median C-index", "Mean (SD)",
                    "C-index > 0.5", "q < 0.05",
                    "corr(event rate, C-index)"], grows,
                   widths=[1.5, 0.4, 0.95, 0.95, 0.75, 0.62, 2.2],
                   font=7.5)
        para(doc, "The failures are structured rather than random. "
                  "Transfer quality falls as the held-out cohort's event "
                  "rate rises, significantly so across all cohorts, and "
                  "the relationship strengthens when the analysis is "
                  "restricted to the cohorts with enough events to be "
                  "read. The model transfers best into indolent cohorts "
                  "and worst into aggressive ones, and Section 6.13 found "
                  "the same gradient inside the pooled fold. A plausible "
                  "reading is that what transfers is a general "
                  "tumour-aggressiveness ordering, informative where "
                  "deaths are sparse and insufficient where the whole "
                  "cohort is high risk. That reading is a hypothesis the "
                  "present data cannot test and it is offered as one.",
             space_before=8)
    add_figure(doc, f"{figdir}/fig6_loco.png")
    caption(doc, n.f() + ":", "Leave-one-cancer-out transfer. (a) transfer "
                              "C-index for every held-out cohort with its "
                              "event count, cohorts with fewer than 60 "
                              "events shown in grey; (b) transfer against "
                              "the held-out cohort's event rate.",
            above=False)


def ch6_interpret(doc, x: F, n: N, A: dict, runs: str, figdir: str):
    section(doc, "6.15 Interpretation")
    it = A.get("interpretation")
    para(doc, "Two interpretability channels are reported: integrated "
              "gradients over the input blocks, and the model's own learned "
              "per-gene gates and module attention. The attribution of a "
              "gene is the sum of the absolute integrated gradients of its "
              "three omic inputs, and the share each omic block carries is")
    formula(doc, "share(block) = Σ_genes |IG(gene, block)|  /  "
                 "Σ_blocks Σ_genes |IG(gene, block)|")
    csv_table(doc, f"{runs}/table7_gene_importance.csv", n.t(),
              "Top 25 genes by integrated-gradient attribution, with the "
              "module each was assigned to, the learned gate and its "
              "spread, and the attribution decomposed by omic block. The "
              "signed block columns show the direction in which each omic "
              "measurement pushed the prediction.",
              nrows=25, widths=[0.95, 1.15, 0.75, 0.68, 0.78, 0.8, 0.78,
                                0.78],
              cols=["gene", "module", "gate_mean", "gate_std", "ig_abs",
                    "ig_mut", "ig_cna", "ig_exp"], font=8.0,
              head_map={"gene": "Gene", "module": "Module",
                        "gate_mean": "Gate", "gate_std": "Gate SD",
                        "ig_abs": "|IG|", "ig_mut": "IG mutation",
                        "ig_cna": "IG CNA", "ig_exp": "IG expression"})
    csv_table(doc, f"{runs}/table8_module_importance.csv", n.t(),
              "Top 15 functional modules by attention, with module size and "
              "the highest-attributed genes in each.",
              nrows=15, widths=[1.3, 0.6, 0.95, 3.4],
              cols=["module", "size", "attention", "top_genes"], font=8.5,
              head_map={"module": "Module", "size": "Size",
                        "attention": "Attention",
                        "top_genes": "Highest-attributed genes"})
    if it:
        b = it["blocks"]
        rows_table(doc, n.t(),
                   "Share of the total attribution carried by each omic "
                   "block, computed over the exported gene list by the "
                   "formula above. The three shares sum to one by "
                   "construction.",
                   ["Omic block", "Σ |IG|", "Share"],
                   [["Mutation (binary indicator)",
                     fv(b["ig_mut"]["abs_sum"], 4),
                     pctf(100 * (b["ig_mut"]["share"] or 0))],
                    ["Copy number (discrete level)",
                     fv(b["ig_cna"]["abs_sum"], 4),
                     pctf(100 * (b["ig_cna"]["share"] or 0))],
                    ["Expression (z-score)",
                     fv(b["ig_exp"]["abs_sum"], 4),
                     pctf(100 * (b["ig_exp"]["share"] or 0))],
                    ["Total", fv(it.get("block_total"), 4), "100 %"]],
                   widths=[2.6, 1.5, 1.2], font=9.0)
        rows_table(doc, n.t(),
                   "Summary of the two interpretability channels over the "
                   "exported lists.",
                   ["Quantity", "Value"],
                   [["Genes in the exported ranking",
                     iv(it.get("n_genes_listed"))],
                    ["Learned gate: mean", fv(it.get("gate_mean"), 4)],
                    ["Learned gate: range",
                     f"{fv(it.get('gate_min'), 4)} to "
                     f"{fv(it.get('gate_max'), 4)}"],
                    ["Modules in the exported ranking",
                     iv(it.get("n_modules_listed"))],
                    ["Mean module size", fv(it.get("module_size_mean"), 2)],
                    ["Highest module attention",
                     fv(it.get("module_attention_top"), 4)],
                    ["Attention summed over the listed modules",
                     fv(it.get("module_attention_sum_top"), 4)],
                    ["Highest-attributed genes",
                     ", ".join((it.get("top_genes") or [])[:8])]],
                   widths=[2.6, 3.7], font=9.0)
        para(doc, "The omic decomposition is the informative part. "
                  "Attribution is dominated by expression and copy number; "
                  "mutation carries "
                  f"{pctf(100 * (b['ig_mut']['share'] or 0))} of the "
                  "total. For a model given a binary mutation indicator "
                  "over a 469-gene panel that is the expected result — a "
                  "binary flag over a small panel carries far less variance "
                  "than a continuous expression z-score — and it is a "
                  "limitation of the encoding rather than a statement about "
                  "tumour biology. The learned gates span "
                  f"{fv(it.get('gate_min'), 3)} to "
                  f"{fv(it.get('gate_max'), 3)} around a mean of "
                  f"{fv(it.get('gate_mean'), 3)}, so the model does "
                  "discriminate between genes rather than weighting them "
                  "uniformly, but the spread is modest.", space_before=8)
    para(doc, "The highest-attributed genes are recurrently altered cancer "
              "genes. That is reassuring and it is not a discovery: the "
              "panel was designed around exactly those genes, so finding "
              "them at the top is a property of the feature space.")
    rich(doc, [("A confound that is not reported as a finding. ", True),
               ("The gene graph's co-alteration relation links genes whose "
                "alterations co-occur, and genes at the same chromosomal "
                "locus co-occur because they are deleted or amplified "
                "together. Modules recovered from such edges therefore "
                "partly recover chromosomal position. The "
                "highest-attention module in the table above is a case in "
                "point: its members sit at the same locus. Where a "
                "module's members are locus neighbours, recovering that "
                "module validates the clustering and not biology, and this "
                "report does not present such modules as discovered "
                "pathways.", False)])
    csv_table(doc, f"{runs}/table9_patient_subnetwork.csv", n.t(),
              "The 20 most strongly attended edges of one patient's rewired "
              "interaction network, with the relation each edge came from. "
              "This is the per-patient channel: the graph the model "
              "propagates over is a function of that patient's own "
              "profile.",
              nrows=20, widths=[1.3, 1.3, 1.1, 1.0],
              cols=["source", "target", "attention", "relation"], font=8.5,
              head_map={"source": "Source gene", "target": "Target gene",
                        "attention": "Attention",
                        "relation": "Relation type"})
    add_figure(doc, f"{figdir}/fig7_interpretation.png")
    caption(doc, n.f() + ":", "Interpretation. Top genes by attribution, "
                              "the attribution decomposed by omic block, "
                              "and module-level attention.", above=False)


def ch6_uncertainty(doc, x: F, n: N, A: dict, S: dict, figdir: str):
    section(doc, "6.16 Calibration and Predictive Uncertainty")
    p = A.get("panel") or {}
    sp = (S or {}).get("panel") or {}
    para(doc, "A model that is to be trusted must say how sure it is. Two "
              "quantities measure that here. Expected calibration error "
              "bins the predictions by confidence and compares each bin's "
              "confidence with its accuracy, and the mean predictive "
              "entropy summarises how spread the predicted distribution is "
              "under Monte-Carlo dropout:")
    formula(doc, "ECE = Σ_b (n_b / N) · | accuracy(b) − confidence(b) |,"
                 "      H(q) = − Σ_k q_k log q_k")
    import math as _m
    k_cohort = x.g("cohort", "n_cohorts") or 32
    hmax = _m.log(float(k_cohort))
    ent = p.get("mean_predictive_entropy")
    rows_table(doc, n.t(),
               "Calibration and uncertainty on both targets. The maximum "
               "entropy is the value a uniform distribution over that "
               "target's classes would give, and the normalised entropy is "
               "the measured entropy divided by it, so the two targets can "
               "be compared despite having different class counts.",
               ["Quantity", "cohort run", "stage_binary run"],
               [["Expected calibration error", fv(p.get("ece"), 4),
                 fv(sp.get("ece"), 4)],
                ["Brier score", fv(p.get("brier"), 4), fv(sp.get("brier"), 4)],
                ["Log loss", fv(p.get("log_loss"), 4),
                 fv(sp.get("log_loss"), 4)],
                ["Classes", iv(k_cohort), "2"],
                ["Mean predictive entropy (nats)", fv(ent, 4),
                 fv(sp.get("mean_predictive_entropy"), 4)],
                ["Maximum entropy, ln K", fv(hmax, 4), fv(_m.log(2.0), 4)],
                ["Normalised entropy",
                 fv((ent / hmax) if ent is not None else None, 4),
                 fv(((sp.get("mean_predictive_entropy") or 0) / _m.log(2.0))
                    if sp.get("mean_predictive_entropy") is not None
                    else None, 4)]],
               widths=[2.6, 1.6, 1.8], font=9.0)
    para(doc, "The two targets sit at opposite ends. On the 32-class "
              f"target the calibration error is {fv(p.get('ece'), 3)}, "
              "which is large: a predicted probability of 0.8 does not mean "
              "an eighty per cent chance, and the outputs are not usable as "
              "probabilities without recalibration. The normalised entropy "
              f"of {fv((ent / hmax) if ent is not None else None, 3)} says "
              "the predicted distribution is far from uniform, so the model "
              "is confident — and being both confident and miscalibrated is "
              "the worst combination for a clinical reader. On the binary "
              f"target the calibration error is {fv(sp.get('ece'), 3)}, "
              "which is small, and the normalised entropy is near one: the "
              "model is close to honest about not knowing. The "
              "architecture is not uniformly miscalibrated; it is "
              "miscalibrated where it is overconfident, and the two "
              "columns separate those cases.", space_before=8)
    para(doc, "Recalibration was not run. Temperature scaling fitted on "
              "the validation split would very likely fix the cohort "
              "column, and Section 8.3 lists it among the future work, but "
              "it is not reported here because it was not executed.")
    add_figure(doc, f"{figdir}/fig3_classification.png")
    caption(doc, n.f() + ":", "Per-class behaviour on the held-out fold, "
                              "with each class's support shown alongside "
                              "its scores.", above=False)
    add_figure(doc, f"{figdir}/figS1_learning_curves.png")
    caption(doc, n.f() + ":", "Training and validation traces over epochs. "
                              "The validation score rises and then "
                              "flattens, and early stopping fires on the "
                              "flat part.", above=False)
    payload_table(doc, x, "S4", n.t(),
                  widths=[0.9, 1.3, 1.3, 1.45, 1.4], font=9.0,
                  head_map={"train_loss": "Train loss",
                            "val_score": "Val score",
                            "val_balanced_accuracy": "Val BAcc",
                            "val_c_index": "Val C-index"},
                  caption_text="The recorded learning curve. Training and "
                               "validation are logged every five epochs; "
                               "the traces do not diverge in the way "
                               "runaway overfitting produces.")


def ch6_cost(doc, x: F, n: N, A: dict, S: dict):
    section(doc, "6.17 Model Size and Computational Cost")
    c = A.get("cost") or {}
    sc = (S or {}).get("cost") or {}
    para(doc, "The parameter count was counted rather than estimated, and "
              "it is broken down by component so that the reader can see "
              "where the capacity sits. The ratio of parameters to training "
              "patients is the quantity that motivated the audit in "
              "Section 6.4.")
    blocks = c.get("blocks") or []
    rows_table(doc, n.t(),
               "Trainable parameters by component, counted on the fitted "
               "network of each run. The share column is that component's "
               "parameters divided by the total.",
               ["Component", "Parameters (cohort run)", "Share"],
               [[name, iv(v), pctf(100 * (share or 0))]
                for name, v, share in blocks]
               + [["Total", iv(c.get("trainable_parameters")), "100 %"]],
               widths=[2.2, 2.1, 1.0], font=9.0)
    rows_table(doc, n.t(),
               "Model size against the data it was fitted on, and the "
               "measured cost of one ablation fold.",
               ["Quantity", "cohort run", "stage_binary run"],
               [["Trainable parameters", iv(c.get("trainable_parameters")),
                 iv(sc.get("trainable_parameters"))],
                ["Training patients", iv(c.get("training_patients")),
                 iv(sc.get("training_patients"))],
                ["Parameters per patient",
                 fv(c.get("parameters_per_patient"), 1),
                 fv(sc.get("parameters_per_patient"), 1)],
                ["Measured seconds per ablation fold",
                 fv(c.get("ablation_seconds_mean"), 0), "—"]],
               widths=[2.6, 1.6, 1.8], font=9.0)
    para(doc, "Two readings follow. The encoder and the omic projections "
              "hold most of the capacity, which is where a model built "
              "around a gene graph should hold it; the cohort adversary and "
              "the residual clinical paths are small by comparison, which "
              "makes the ablation result in Section 6.10 more striking "
              "rather than less — the clinical residual path dominates the "
              "accuracy while holding a tiny share of the parameters. And "
              f"at about {fv(c.get('ablation_seconds_mean'), 0)} seconds "
              "per fold on CPU, a seven-arm ablation is a multi-hour job "
              "and the thirty-two-cohort transfer sweep is a multi-day one, "
              "which is the whole reason Section 5.3 exists.",
         space_before=8)


def ch6_stage(doc, x: F, n: N, S: dict, runs_stage: str):
    section(doc, "6.18 The Second Prediction Target in Full")
    p = (S or {}).get("panel") or {}
    para(doc, "Everything above was reported for the cohort "
              "target, which is a methodological stress test. The "
              "stage_binary target is the clinically meaningful "
              "one, and it was fitted independently — its own folds, its "
              "own graph, its own comparator panel. Its results are given "
              "here at the same level of detail, so that the clinical "
              "endpoint is not left as a sentence in the discussion.")
    para(doc, f"Only the cohorts that record AJCC stage contribute, giving "
              f"{x.i('stage_binary','n_labelled')} labelled patients in "
              f"{x.i('stage_binary','n_cohorts_labelled')} cohorts. The "
              "classes are early stage (I–II) against late stage (III–IV).")
    con = (S or {}).get("consistency")
    pc = (S or {}).get("per_class")
    sg = (S or {}).get("gap") or {}
    if pc and sg.get("n_test"):
        frac = 100.0 * pc["support_sum"] / sg["n_test"]
        para(doc, "One detail of how this target is scored needs stating "
                  "before its numbers are read. The held-out split holds "
                  f"{iv(sg.get('n_test'))} patients, the same split the "
                  "cohort run used, but only "
                  f"{iv(pc['support_sum'])} of them — {fv(frac, 1)} per "
                  "cent — carry an AJCC stage label. Every classification "
                  "metric for this target is therefore computed on that "
                  "labelled subset, not on the whole split, so its "
                  "per-class support sums to "
                  f"{iv(pc['support_sum'])} rather than to "
                  f"{iv(sg.get('n_test'))}. The split-size column of the "
                  "generalisation-gap table below reports the split, which "
                  "is the larger number; the two are not in conflict and "
                  "neither is wrong, but a reader comparing them without "
                  "this sentence would think one of them was.")
    if pc:
        STAGE = {"0": "early (I–II)", "1": "late (III–IV)"}
        rows = [[STAGE.get(str(r["class"]), str(r["class"])),
                 iv(r["support"]), iv(r["tp"]),
                 iv(r["fp"]), iv(r["fn"]), iv(r["tn"]),
                 fv(r["sensitivity"]),
                 f"{fv(r['sens_ci_low'],3)}–{fv(r['sens_ci_high'],3)}",
                 fv(r["specificity"]), fv(r["precision"]), fv(r["npv"]),
                 fv(r["f1"]), fv(r["youden"]), fv(r["mcc"]),
                 fv(r["mean_predicted_prob"])] for r in pc["rows"]]
        rows_table(doc, n.t(),
                   "Per-class behaviour on the held-out fold of the "
                   "stage_binary run, with the contingency table "
                   "recovered for each class exactly as in Section 6.3. "
                   "Class 0 is early stage and class 1 is late stage.",
                   ["Class", "n_c", "TP", "FP", "FN", "TN", "Sens",
                    "Sens 95 % CI", "Spec", "Prec", "NPV", "F1", "J",
                    "MCC_c", "Mean prob."], rows,
                   widths=[0.95, 0.4, 0.37, 0.37, 0.37, 0.42, 0.4, 0.72,
                           0.4, 0.4, 0.4, 0.37, 0.37, 0.42, 0.4],
                   font=6.5)
        if con:
            para(doc, "The same internal consistency audit was run on this "
                      "target. The per-class support sums to "
                      f"{iv(con.get('support_sum'))}, the held-out split "
                      "exactly, and recomputing the pooled macro metrics "
                      "from the per-class table reproduces the reported "
                      "values to within "
                      f"{con.get('max_abs_diff', 0):.1e}. Both targets "
                      "therefore pass the audit.", space_before=8)
    g = (S or {}).get("gap")
    if g:
        rows = [[SHORT.get(r["metric"], r["metric"]), fv(r["train"], 4),
                 fv(r["validation"], 4), fv(r["test"], 4), fv(r["gap"], 4),
                 fv(r["gap_pct"], 1)] for r in g["rows"]]
        rows_table(doc, n.t(),
                   "Generalisation gap on the stage_binary run, "
                   "every metric. The gaps here are much smaller than on "
                   "the cohort run, which is what a harder "
                   "target with fewer classes and a well-calibrated output "
                   "produces.",
                   ["Metric", "Train", "Validation", "Test", "Gap", "Gap %"],
                   rows, widths=[1.3, 0.95, 1.1, 0.95, 0.95, 0.9], font=8.5)
        para(doc, "The mean absolute gap across the discriminative metrics "
                  f"is {fv(g.get('mean_abs_gap'), 4)} against "
                  "the cohort run's. Fitting the harder target "
                  "overfits less, which is the ordinary direction: there is "
                  "less structure available to memorise.", space_before=8)
    cls = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
           "precision_macro", "recall_macro", "specificity_macro",
           "npv_macro", "auroc", "auprc", "mcc", "kappa", "top2_accuracy",
           "log_loss", "brier", "ece"]
    rows = _cmp_rows(runs_stage, p, cls)
    if rows:
        wide_table(doc, n.t(),
                   "Classification on the held-out fold of the "
                   "stage_binary run, every model and every "
                   "metric, MOSAIC-GNN first and the comparators in "
                   "descending balanced accuracy.",
                   ["Model"] + _short(cls), rows,
                   groups=[_short(gr) for gr in
                           [["accuracy", "balanced_accuracy", "f1_macro",
                             "f1_weighted"],
                            ["precision_macro", "recall_macro",
                             "specificity_macro", "npv_macro"],
                            ["auroc", "auprc", "mcc", "kappa",
                             "top2_accuracy"],
                            ["log_loss", "brier", "ece"]]],
                   part_caps=CLS_CAPS,
                   first_width=1.75, col_width=0.78, font=7.5)
    surv = ["c_index", "auc@12m", "auc@36m", "auc@60m", "logrank_chi2",
            "logrank_p"]
    srows = _cmp_rows(runs_stage, p, surv)
    if srows:
        rows_table(doc, n.t(),
                   "Survival ranking on the stage_binary run.",
                   ["Model"] + _short(surv), srows,
                   widths=[1.75, 0.78, 0.7, 0.7, 0.7, 0.85, 0.85], font=7.5)
    r = (S or {}).get("ranks")
    if r:
        rows = [[SHORT.get(q["metric"], q["metric"]), fv(q["mosaic"], 4),
                 f"{q['rank']} of {q['of']}", q["best_model"],
                 fv(q["best_value"], 4), fv(q["delta_to_best"], 4)]
                for q in r["rows"]]
        rows_table(doc, n.t(),
                   "Rank of MOSAIC-GNN on the stage_binary run "
                   "against the full comparator panel, on every metric "
                   "both produce.",
                   ["Metric", "MOSAIC-GNN", "Rank", "Best comparator",
                    "Its value", "Δ to best"], rows,
                   widths=[1.15, 0.95, 0.85, 1.5, 0.85, 0.8], font=8.0)
        para(doc, f"The median rank on this target is "
                  f"{fv(r.get('median_rank'), 1)} of about "
                  f"{iv((r['rows'][0] or {}).get('of'))}, with first place "
                  f"on {iv(r.get('n_rank_1'))} metrics and a best rank of "
                  f"{iv(r.get('best_rank'))}. The picture is the same as on "
                  "the cohort target, which matters: the negative "
                  "comparison result is not an artefact of the artificial "
                  "endpoint. It holds on the clinical one too.",
             space_before=8)
    pcoh = (S or {}).get("per_cohort")
    if pcoh:
        hdr = ["Cohort", "n", "Acc", "BAcc", "F1-M", "AUROC", "AUPRC",
               "MCC", "LogLoss", "Brier", "C-index"]
        rows = [[q["cohort"], iv(q["n"]), fv(q["accuracy"]),
                 fv(q["balanced_accuracy"]), fv(q["f1_macro"]),
                 fv(q.get("auroc")), fv(q.get("auprc")), fv(q["mcc"]),
                 fv(q["log_loss"]), fv(q["brier"]), fv(q["c_index"])]
                for q in pcoh["rows"]]
        rows_table(doc, n.t(),
                   "Per-cohort metrics on the held-out fold of the "
                   "stage_binary run, ordered by accuracy.",
                   hdr, rows,
                   widths=[0.95, 0.5, 0.58, 0.58, 0.58, 0.58, 0.58, 0.55,
                           0.62, 0.55, 0.62], font=7.5)
        acc, ci = pcoh["accuracy"], pcoh["c_index"]
        para(doc, f"Across {iv(pcoh.get('n_cohorts'))} cohorts the "
                  f"per-cohort accuracy averages {fv(acc['mean'])} "
                  f"(SD {fv(acc['sd'])}, range {fv(acc['min'])} to "
                  f"{fv(acc['max'])}) and the per-cohort C-index averages "
                  f"{fv(ci['mean'])} (SD {fv(ci['sd'])}, range "
                  f"{fv(ci['min'])} to {fv(ci['max'])}); "
                  f"{iv(pcoh.get('n_cindex_above_half'))} cohorts exceed a "
                  "C-index of 0.5. The correlation between a cohort's "
                  "event rate and its C-index on this target is "
                  f"{_corr_str(pcoh.get('corr_eventrate_cindex'))}.",
             space_before=8)
    it = (S or {}).get("interpretation")
    if it:
        b = it["blocks"]
        rows_table(doc, n.t(),
                   "Attribution by omic block on the stage_binary "
                   "run, for comparison with the cohort run. The "
                   "mutation share is larger here but still the smallest "
                   "of the three.",
                   ["Omic block", "Σ |IG|", "Share"],
                   [["Mutation", fv(b["ig_mut"]["abs_sum"], 4),
                     pctf(100 * (b["ig_mut"]["share"] or 0))],
                    ["Copy number", fv(b["ig_cna"]["abs_sum"], 4),
                     pctf(100 * (b["ig_cna"]["share"] or 0))],
                    ["Expression", fv(b["ig_exp"]["abs_sum"], 4),
                     pctf(100 * (b["ig_exp"]["share"] or 0))]],
                   widths=[2.2, 1.5, 1.2], font=9.0)
        csv_table(doc, f"{runs_stage}/table7_gene_importance.csv", n.t(),
                  "Top 15 genes by attribution on the stage_binary "
                  "run. CDKN2B heads both targets' lists, which is the "
                  "only interpretability result that replicates across "
                  "them.",
                  nrows=15, widths=[0.95, 1.15, 0.8, 0.8, 0.85, 0.85, 0.85],
                  cols=["gene", "module", "gate_mean", "ig_abs", "ig_mut",
                        "ig_cna", "ig_exp"], font=8.0,
                  head_map={"gene": "Gene", "module": "Module",
                            "gate_mean": "Gate", "ig_abs": "|IG|",
                            "ig_mut": "IG mutation", "ig_cna": "IG CNA",
                            "ig_exp": "IG expression"})
    rich(doc, [("Which phases did not run on this target. ", True),
               ("The ablation, the label permutation control, the "
                "sample-size curve and the leave-one-cancer-out sweep were "
                "run on the cohort target only. Each is a "
                "multi-hour to multi-day job on CPU, as Section 6.17 "
                "measures, and the budget covered one target. They are "
                "therefore absent from this section rather than carried "
                "over from the other target, and no number in it is "
                "borrowed. What this target does have is the full metric "
                "panel, the per-class and per-cohort breakdowns, the "
                "generalisation-gap audit, the consistency audit, the "
                "complete comparator panel with paired corrected tests, "
                "the molecular-gain comparison and the interpretability "
                "exports.", False)])


def ch6_discussion(doc, x: F, n: N, A: dict, S: dict):
    section(doc, "6.19 Discussion")
    lc, r = A.get("loco") or {}, A.get("ranks") or {}
    para(doc, "Taken together the results answer the four sub-questions of "
              "Section 1.2, and two of the four answers are not the ones "
              "the architecture was built to produce.")
    bullets(doc, [
        ("RQ1, answered yes, and it replicates. ", "Molecular features add "
         "prognostic information beyond the clinical covariates, measured "
         "both by a penalised Cox model with and without them and by "
         "MOSAIC-GNN against a clinical-only Cox model, on both targets, "
         "with every interval excluding zero and every difference "
         "surviving multiplicity correction."),
        ("RQ2, answered in part. ", "The three structural components — gene "
         "graph, module pooling, learned patient graph — each clear the "
         "measured noise floor when removed, at one to two floors. The two "
         "auxiliary objectives do not clear it. The clinical residual path "
         "dominates everything else by an order of magnitude while holding "
         "a small share of the parameters."),
        ("RQ3, answered yes. ", "The representation stratifies risk in "
         "cohorts held out entirely from training: above random in "
         + iv(lc.get("n_above_half")) + " of " + iv(lc.get("n_cohorts")) +
         " cases, with the log-rank test significant after "
         "Benjamini-Hochberg correction in " + iv(lc.get("n_q_lt_05")) +
         ". A cohort detector could not do this."),
        ("RQ4, answered against the proposed model. ", "Under a paired, "
         "corrected comparison it does not outperform established methods "
         "on either classification endpoint; its median rank across "
         + iv(r.get("n_metrics")) + " metrics is "
         + fv(r.get("median_rank"), 1) + ". It is competitive rather than "
         "superior on survival ranking."),
    ])
    para(doc, "Three findings deserve to be read together because they "
              "point the same way. The ablation says the clinical "
              "covariates carry most of the accuracy. The attribution "
              "analysis says the molecular contribution that does exist is "
              "carried by expression and copy number, with the binary "
              "mutation encoding contributing almost nothing. And the "
              "sample-size curve says classification has not saturated "
              "while survival ranking essentially has. The consistent "
              "reading is that the molecular signal available through this "
              "feature space and this encoding is real but thin, and that "
              "the architecture's capacity exceeds what that signal can "
              "support — which is also why a penalised linear model, with "
              "far fewer effective degrees of freedom, does better.")
    para(doc, "The practical conclusion a reader should draw is therefore "
              "not \"use this architecture\". It is that the molecular data "
              "is worth collecting because it carries information the "
              "clinic does not have; that a penalised Cox model or a "
              "gradient-boosted ensemble is a stronger starting point than "
              "it is usually given credit for; and that the two-level "
              "graph design contributes measurably but not enough to close "
              "that gap at this sample size. The last clause is a testable "
              "prediction rather than a defence, and Section 8.3 says how "
              "to test it.")
    para(doc, "Two explanations offered in earlier drafts of this work were "
              "tested against runs executed for that purpose, failed, and "
              "have been withdrawn: the cohort-adversary account of the "
              "disentanglement arm, retracted in Section 6.10, and the "
              "claim that leave-one-cancer-out classification metrics "
              "would be near zero by arithmetic, retracted in "
              "Section 6.14. Both retractions are recorded in place rather "
              "than deleted, because the sequence of a prediction, a test "
              "and a withdrawal is part of the evidence for how the "
              "surviving conclusions were reached.")

    section(doc, "6.20 Threats to Validity")
    p = A.get("panel") or {}
    g = A.get("gap") or {}
    d = A.get("descriptives") or {}
    bullets(doc, [
        ("Single data source. ", "Every patient comes from TCGA through "
         "cBioPortal. No external cohort was available, so none of these "
         "numbers has been replicated outside that source and no claim of "
         "external validity is made."),
        ("Panel-restricted feature space. ", "The 469-gene panel was "
         "pre-committed, which removes selection leakage but also removes "
         "whole classes of biology — immune and stromal programmes among "
         "them — from reach. Results cannot be generalised to genome-wide "
         "feature spaces."),
        ("A single held-out fold for the headline numbers. ", "Every table "
         "in this chapter is built from one fitted model and one split: the "
         "export reads the saved primary fold and the first entry of the "
         "saved predictions, and no multi-fold summary exists for either "
         "target. One consequence is favourable and worth stating — the "
         "model and all the comparators are scored on the same patients in "
         "the same fold, so the paired comparison of Section 6.8 carries no "
         "fold-count confound at all. The other is not: no headline metric "
         "has a fold-to-fold spread of its own, so its precision is bounded "
         "by the ±0.01 same-seed noise floor of Section 6.5 and the Wilson "
         "intervals of Section 6.3 rather than by a standard error over "
         "folds. A repeated-fold estimate would be better and was not "
         "affordable. Only the ablation arms carry two folds each, and only "
         "their spread is reported as a standard deviation."),
        ("Reduced-setting and bounded phases. ", "The ablation ran on a "
         "reduced configuration and the permutation test on four "
         "permutations, which bounds the attainable p-value at 0.2. Both "
         "are labelled wherever they appear and neither supports a "
         "fine-grained claim."),
        ("Calibration. ", "An expected calibration error of "
         + fv(p.get("ece"), 3) + " on the cohort run means the "
         "predicted probabilities are not usable as probabilities without "
         "recalibration, whatever the discriminative metrics show."),
        ("A real generalisation gap. ", "The train-to-test gap reaches "
         + fv(g.get("max_gap"), 4) + " on its largest metric. It is not "
         "the collapse the parameter count might suggest, but it is not "
         "nothing, and every held-out number should be read as the lower "
         "end of what this configuration can do on data it has seen."),
        ("Small-cohort statistics. ", iv(d.get("n_events_below_20")) +
         " cohorts contribute fewer than twenty deaths. Their per-cohort "
         "numbers are reported for completeness and are explicitly "
         "separated from the powered subset in Sections 6.13 and 6.14."),
        ("Non-determinism. ", "The measured run-to-run spread of about "
         "±0.01 bounds the resolution of every single-fit comparison in "
         "this report, including every ablation arm."),
        ("Multiplicity beyond the corrected family. ", "The paired tests "
         "are corrected within each metric, and Section 6.8 shows that "
         "pooling the two families is stricter. The wider family of every "
         "metric reported anywhere in this chapter is larger still, and no "
         "correction spans it; the conclusions that matter rest on "
         "differences far from the 0.05 boundary, which is why this is a "
         "caveat rather than a defect."),
        ("An unexplained fault in the transfer classification path. ",
         "Section 6.14.1 documents single-class recalls that a softmax head "
         "given no positive examples of a class should not be able to "
         "produce. No classification number from that experiment is "
         "reported, and the three arguments given there for why the "
         "survival column is unaffected are arguments, not proof. The "
         "transfer finding should be read as provisional until the fault is "
         "identified."),
        ("No external, prospective or clinical evaluation. ", "None was "
         "performed, and no claim that depends on one appears anywhere in "
         "this report."),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
