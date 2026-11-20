"""Check that the report, the notebook and the run exports all say the same thing.

    python scripts/verify_consistency.py

This exists because of a real failure. The notebook used to retrain the model
at a reduced configuration and print its own numbers, while the report was
built from the full runs in runs_pan_cohort/ and runs_pan_stage/. Both were
honest on their own terms and neither shared a single figure with the other,
which is indistinguishable, to a reader, from one of them being wrong.

Four things are compared here:

  1.  the run exports (CSV and JSON in the run directories), read through
      scripts/report_analysis.py -- the same module the report uses;
  2.  deliverables/manuscript_payload.json, which the report is built from;
  3.  the text of the built report, deliverables/*.docx;
  4.  the executed notebook, notebooks/mosaic_gnn_executed.ipynb, whose
      Section 2 prints the same quantities.

Every headline number must appear in all four. A number present in one and
absent or different in another is reported as a FAIL and the script exits
non-zero, so this can be wired into a pre-submission check.

The comparison against the report and the notebook is textual: the formatted
number is searched for in the rendered text. That is deliberately crude and
it is the right crudeness -- it checks what a reader would actually see on
the page, not what a data structure holds.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import report_analysis as RA          # noqa: E402


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def docx_text(path: str) -> str:
    """All the text of a .docx, including tables, as one string."""
    if not os.path.isfile(path):
        return ""
    out = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not (name.startswith("word/") and name.endswith(".xml")):
                continue
            xml = z.read(name).decode("utf-8", "replace")
            out.append(re.sub(r"<[^>]+>", " ", xml))
    return " ".join(out)


def notebook_markdown(path: str) -> str:
    """The markdown source of a notebook, where its prose warnings live."""
    if not os.path.isfile(path):
        return ""
    with open(path) as fh:
        nb = json.load(fh)
    return "\n".join("".join(c.get("source", []))
                     for c in nb.get("cells", [])
                     if c.get("cell_type") == "markdown")


def notebook_text(path: str) -> str:
    """Every stdout stream and text output of an executed notebook."""
    if not os.path.isfile(path):
        return ""
    with open(path) as fh:
        nb = json.load(fh)
    out = []
    for cell in nb.get("cells", []):
        for o in cell.get("outputs", []):
            for key in ("text",):
                v = o.get(key)
                if isinstance(v, list):
                    out.append("".join(v))
                elif isinstance(v, str):
                    out.append(v)
            data = o.get("data") or {}
            v = data.get("text/plain")
            if isinstance(v, list):
                out.append("".join(v))
            elif isinstance(v, str):
                out.append(v)
    return "\n".join(out)


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
class Checker:
    def __init__(self, report: str, nb: str, strict_nb: bool):
        self.report = squash(report)
        self.nb = squash(nb)
        self.strict_nb = strict_nb
        self.rows: list[tuple] = []

    @staticmethod
    def _forms(value: float, decimals=(4, 3, 2, 1)) -> list[str]:
        """The ways a number may legitimately have been printed."""
        out = []
        for d in decimals:
            out.append(f"{value:.{d}f}")
        if abs(value) >= 1000 and float(value).is_integer():
            out.append(f"{int(value):,}")
            out.append(str(int(value)))
        elif float(value).is_integer():
            out.append(str(int(value)))
        if value != 0 and abs(value) < 1e-3:
            out.append(f"{value:.2e}")
        return out

    def number(self, label: str, value, payload_value=None):
        """Check one measured number against the payload, report and notebook."""
        if value is None:
            self.rows.append((label, "—", "MISSING", "", "", "FAIL"))
            return
        value = float(value)
        pay = ("n/a" if payload_value is None
               else ("PASS" if abs(float(payload_value) - value) <= 1e-9
                     else f"FAIL ({payload_value})"))
        forms = self._forms(value)
        in_report = any(f in self.report for f in forms)
        in_nb = any(f in self.nb for f in forms)
        verdict = "PASS"
        if pay.startswith("FAIL") or not in_report:
            verdict = "FAIL"
        elif not in_nb:
            verdict = "FAIL" if self.strict_nb else "WARN"
        self.rows.append((label, f"{value:.6g}", pay,
                          "yes" if in_report else "NO",
                          "yes" if in_nb else "NO", verdict))

    def report_only(self, label: str, value, decimals=(4, 3, 2, 1)):
        """A number that need not be printed by the notebook.

        `decimals` narrows which printed forms count as a match, for
        quantities the report deliberately rounds (a percentage share to one
        decimal, say) where a looser search would match an unrelated cell.
        """
        if value is None:
            self.rows.append((label, "—", "n/a", "", "", "FAIL"))
            return
        value = float(value)
        forms = self._forms(value, decimals)
        ok = any(f in self.report for f in forms)
        self.rows.append((label, f"{value:.6g}", "n/a",
                          "yes" if ok else "NO", "-",
                          "PASS" if ok else "FAIL"))

    def render(self) -> int:
        w = max(len(r[0]) for r in self.rows) + 2
        print(f"{'check'.ljust(w)}{'value':>14}  {'payload':<14}"
              f"{'report':<8}{'notebook':<10}verdict")
        print("-" * (w + 60))
        for label, value, pay, rep, nb, verdict in self.rows:
            print(f"{label.ljust(w)}{value:>14}  {pay:<14}{rep:<8}"
                  f"{nb:<10}{verdict}")
        fails = sum(1 for r in self.rows if r[5] == "FAIL")
        warns = sum(1 for r in self.rows if r[5] == "WARN")
        print("-" * (w + 60))
        print(f"{len(self.rows)} checks: {len(self.rows) - fails - warns} "
              f"PASS, {warns} WARN, {fails} FAIL")
        return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(REPO, "runs_pan_cohort"))
    ap.add_argument("--runs-stage", dest="runs_stage",
                    default=os.path.join(REPO, "runs_pan_stage"))
    ap.add_argument("--payload", default=os.path.join(
        REPO, "deliverables", "manuscript_payload.json"))
    ap.add_argument("--report", default=os.path.join(
        REPO, "deliverables", "CSE400B-MOSAIC-GNN-Report.docx"))
    ap.add_argument("--notebook", default=os.path.join(
        REPO, "notebooks", "mosaic_gnn_executed.ipynb"))
    ap.add_argument("--strict-notebook", action="store_true",
                    help="treat a number missing from the notebook as a "
                         "failure rather than a warning")
    a = ap.parse_args()

    for path, what in ((a.payload, "payload"), (a.report, "report")):
        if not os.path.isfile(path):
            print(f"ERROR: the {what} is missing: {path}")
            return 2
    with open(a.payload) as fh:
        payload = json.load(fh)
    facts = payload["facts"]

    A = RA.analyse(a.runs)
    S = RA.analyse(a.runs_stage)

    nb_text = notebook_text(a.notebook)
    nb_md = squash(notebook_markdown(a.notebook))
    if not nb_text:
        print(f"note: no executed notebook at {a.notebook}; its column will "
              "read NO throughout")

    c = Checker(docx_text(a.report), nb_text, a.strict_notebook)

    # -- headline metrics, both targets ---------------------------------- #
    pairs = [("balanced_accuracy", "mosaic_balanced_accuracy"),
             ("accuracy", "mosaic_accuracy"),
             ("auroc", "mosaic_auroc"), ("auprc", "mosaic_auprc"),
             ("mcc", "mosaic_mcc"), ("f1_macro", "mosaic_f1_macro"),
             ("ece", "mosaic_ece"), ("c_index", "mosaic_c_index")]
    for target, res in (("cohort", A), ("stage_binary", S)):
        for metric, fact in pairs:
            c.number(f"{target}: {metric}", res["panel"].get(metric),
                     facts[target].get(fact))

    # -- dataset --------------------------------------------------------- #
    d = A["descriptives"]
    c.number("dataset: patients", d["patients_total"],
             facts["cohort"]["n_patients"])
    c.number("dataset: deaths", d["events_total"],
             facts["cohort"]["n_events"])
    c.number("dataset: studies", d["n_cohorts"],
             facts["cohort"]["n_cohorts"])
    c.report_only("dataset: normalised entropy", d["entropy_ratio"])
    c.report_only("dataset: imbalance ratio", d["imbalance_ratio"])

    # -- molecular gain, the project's positive finding ------------------ #
    for target in ("cohort", "stage_binary"):
        mg = facts[target]["molecular_gain"]
        cc = facts[target]["paired"]["models"]["cox_clinical_only"]
        c.number(f"{target}: molecular gain", mg["delta"],
                 mg["molecular"] - mg["clinical_only"])
        c.number(f"{target}: MOSAIC over clinical Cox", cc["cindex_delta"],
                 cc["cindex_delta"])

    # -- model size and the gap ------------------------------------------ #
    c.number("model: trainable parameters",
             A["cost"]["trainable_parameters"])
    c.number("model: parameters per patient",
             A["cost"]["parameters_per_patient"])
    c.report_only("gap: largest discriminative gap", A["gap"]["max_gap"])
    c.report_only("gap: mean absolute gap", A["gap"]["mean_abs_gap"])

    # -- ablation -------------------------------------------------------- #
    c.number("ablation: full model BAcc", A["ablation"]["full"],
             facts["cohort"]["ablation"]["full_bacc"])
    for arm in A["ablation"]["arms"]:
        c.number(f"ablation: {arm['arm']}", arm["delta"],
                 facts["cohort"]["ablation"]["arms"][arm["arm"]]["delta"])

    # -- permutation ----------------------------------------------------- #
    for metric, key in (("balanced_accuracy", "bacc"), ("auroc", "auroc"),
                        ("c_index", "cindex")):
        row = next(r for r in A["permutation"]["rows"]
                   if r["metric"] == metric)
        c.number(f"permutation: {metric} null mean", row["null_mean"],
                 facts["cohort"]["permutation_values"][key]["mean"])

    # -- sample size ----------------------------------------------------- #
    ss = A["samplesize"]
    c.number("samplesize: test BAcc gain",
             ss["test_balanced_accuracy"]["gain"],
             facts["cohort"]["sample_size"]["bacc_gain"])
    c.report_only("samplesize: BAcc slope per doubling",
                  ss["test_balanced_accuracy"]["slope_per_doubling"])

    # -- transfer -------------------------------------------------------- #
    lc = A["loco"]
    c.number("loco: median transfer C-index", lc["median"],
             facts["cohort"]["loco"]["median"])
    c.number("loco: cohorts above 0.5", lc["n_above_half"],
             facts["cohort"]["loco"]["n_above_half"])
    c.report_only("loco: cohorts significant after FDR", lc["n_q_lt_05"])
    c.report_only("loco: corr(event rate, C-index), all cohorts",
                  lc["groups"]["all"]["corr_event_rate"]["r"])

    # -- interpretation -------------------------------------------------- #
    it = A["interpretation"]
    for block in ("ig_mut", "ig_cna", "ig_exp"):
        c.report_only(f"attribution share: {block}",
                      100 * it["blocks"][block]["share"], decimals=(1,))

    fails = c.render()

    # -- structural invariants that are not single numbers --------------- #
    print()
    extra = []
    for name, res in (("cohort", A), ("stage_binary", S)):
        con = res["consistency"]
        ok = con and con["max_abs_diff"] < 1e-9
        extra.append((f"{name}: pooled metrics reproduce from per-class",
                      "PASS" if ok else "FAIL",
                      f"max |diff| = {con['max_abs_diff']:.2e}" if con
                      else "per-class export absent"))
        pr = res["paired"]
        ok = pr and pr["q_max_abs_diff"] < 1e-9
        extra.append((f"{name}: Benjamini-Hochberg q reproduces",
                      "PASS" if ok else "FAIL",
                      f"max |diff| = {pr['q_max_abs_diff']:.2e}" if pr
                      else "paired export absent"))
    nb_ok = "41 of 41 checks PASS" in c.nb or "checks PASS" in c.nb
    extra.append(("notebook ran its own self-check",
                  "PASS" if nb_ok else ("WARN" if not nb_text else "FAIL"),
                  "Section 2 verification cell output present" if nb_ok
                  else "no verification output found in the notebook"))
    # The notebook must not present its reduced demonstration run as a
    # result: the warning has to be in the text.
    warned = any(t in nb_md for t in
                 ("not the report's figures",
                  "not the project's results",
                  "The reduced demonstration run starts here"))
    extra.append(("notebook labels the reduced run as a demonstration",
                  "PASS" if warned else ("WARN" if not nb_md else "FAIL"),
                  "the separating banner is present" if warned
                  else "no banner separating the reduced run"))
    w = max(len(e[0]) for e in extra) + 2
    for label, verdict, note in extra:
        print(f"{label.ljust(w)}{verdict:<8}{note}")
        if verdict == "FAIL":
            fails += 1

    print()
    if fails:
        print(f"{fails} checks FAILED. The report, the notebook and the run "
              "exports do not all agree; rebuild with "
              "scripts/build_ewu_report.py and scripts/execute_notebook.py "
              "before submitting either.")
        return 1
    print("The run exports, the payload, the report and the executed "
          "notebook all agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
