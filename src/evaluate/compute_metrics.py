"""Unified evaluation: reproduces every results table in the paper from the
saved per-instance prediction files in ``results/``.

All four methods (SuTraN, Tax LSTM, LLM-only, TPGLR) are scored under a single,
shared metric so the numbers are directly comparable and reproducible:

  * DL similarity : length-normalized Damerau-Levenshtein similarity in [0, 1],
                    counting insertions, deletions, substitutions and adjacent
                    transpositions. Identical to the dl_similarity() used by the
                    baseline scripts and by claude_experiment_final2.py.
  * F1            : multiset (bag-of-activities) overlap F1, order-independent.
  * Perfect match : fraction of QA instances whose predicted suffix equals the
                    ground-truth suffix exactly (sequence equality).
  * Perfect recall: fraction of QA instances with recall == 1.0 (every
                    ground-truth activity appears in the prediction).

Convention for failed generations
----------------------------------
The LLM-only run has one QA instance (qa_id 10, "Vendor creates debit memo")
that produced no output due to a transient API 529 (server overload), not a
model error. Following the paper:
  * DL similarity and F1 are averaged over the instances that produced an output
    (n_valid).
  * Perfect-match and perfect-recall rates are reported out of the full QA set
    (n_total = 180); a failed generation counts as neither.
The script prints n_valid / n_total for every method so this is explicit.

Run:
    python src/evaluate/compute_metrics.py
"""

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Repo root = two levels up from this file (src/evaluate/compute_metrics.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

METHOD_FILES = {
    "SuTraN":   "sutran_qa_results.pkl",
    "Tax LSTM": "tax_lstm_results.pkl",
    "LLM-only": "llm_only_results.pkl",
    "TPGLR":    "claude_results_final2.pkl",
}

EVENT_CATEGORIES = [
    "Cancel Invoice Receipt",
    "Change Delivery Indicator",
    "Change Price",
    "Change Quantity",
    "Remove Payment Block",
    "Vendor creates debit memo",
]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def dl_similarity(s1, s2):
    """Length-normalized Damerau-Levenshtein similarity in [0, 1]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    l1, l2 = len(s1), len(s2)
    d = np.arange(l2 + 1, dtype=float)
    for i in range(1, l1 + 1):
        prev = d.copy()
        d[0] = i
        for j in range(1, l2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[j] = min(d[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[j] = min(d[j], prev[j - 1])
    return max(0.0, 1 - d[l2] / max(l1, l2))


def f1_sets(pred, answer):
    """Multiset-overlap F1 (order-independent)."""
    pc, ac = Counter(pred), Counter(answer)
    tp = sum((pc & ac).values())
    if tp == 0:
        return 0.0
    p = tp / sum(pc.values())
    r = tp / sum(ac.values())
    return 2 * p * r / (p + r)


def recall_sets(pred, answer):
    pc, ac = Counter(pred), Counter(answer)
    tp = sum((pc & ac).values())
    return tp / sum(ac.values()) if ac else 1.0


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_qid_to_event(qa_path):
    """qa_id (str) -> canonical event category, from the QA dataset."""
    qa = pickle.load(open(qa_path, "rb"))["qa_dataset"]
    records = qa if isinstance(qa, list) else list(qa.values())
    return {str(r["qa_id"]): r["event_name"] for r in records}


def load_instances(results_path, qid_to_event):
    """Return list of (event, answer, predicted) for instances that produced
    an output, plus (n_valid, n_total)."""
    data = pickle.load(open(results_path, "rb"))
    res = data["results"]
    # LLM methods store a list (qa_id is a field); baselines store a dict keyed
    # by qa_id (the per-instance record itself has no qa_id field).
    if isinstance(res, list):
        items = [(r.get("qa_id"), r) for r in res]
    else:
        items = list(res.items())
    n_total = len(items)
    rows = []
    for qid, r in items:
        if "answer" not in r:          # failed generation (e.g. API error)
            continue
        pred = r["generated"] if "generated" in r else r["predicted"]
        event = r.get("event") or qid_to_event.get(str(qid))
        rows.append((event, list(r["answer"]), list(pred)))
    return rows, len(rows), n_total


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def overall_metrics(rows, n_total):
    dls = [dl_similarity(a, p) for _, a, p in rows]
    f1s = [f1_sets(p, a) for _, a, p in rows]
    perfect_match = sum(1 for _, a, p in rows if list(p) == list(a))
    perfect_recall = sum(1 for _, a, p in rows if recall_sets(p, a) >= 1.0 - 1e-9)
    return {
        "dl": float(np.mean(dls)),
        "f1": float(np.mean(f1s)),
        "perfect_match": perfect_match,
        "perfect_recall": perfect_recall,
        "n_total": n_total,
    }


def per_category_metrics(rows):
    by_cat = defaultdict(list)
    for ev, a, p in rows:
        by_cat[ev].append((a, p))
    out = {}
    for cat in EVENT_CATEGORIES:
        pairs = by_cat.get(cat, [])
        if not pairs:
            out[cat] = None
            continue
        out[cat] = {
            "f1": float(np.mean([f1_sets(p, a) for a, p in pairs])),
            "dl": float(np.mean([dl_similarity(a, p) for a, p in pairs])),
            "n": len(pairs),
        }
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--qa", default=str(REPO_ROOT / "data" / "qa_dataset_final.pkl"))
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    qid_to_event = load_qid_to_event(args.qa)

    overall, per_cat, counts = {}, {}, {}
    for method, fname in METHOD_FILES.items():
        rows, n_valid, n_total = load_instances(results_dir / fname, qid_to_event)
        overall[method] = overall_metrics(rows, n_total)
        per_cat[method] = per_category_metrics(rows)
        counts[method] = (n_valid, n_total)

    # ---- Table 1: overall ---------------------------------------------------
    print("\n" + "=" * 64)
    print("TABLE 1  Overall performance")
    print("=" * 64)
    print(f"{'Method':12s} {'F1':>7} {'DL sim':>8} "
          f"{'Perfect match':>16} {'Perfect recall':>16}")
    for m in METHOD_FILES:
        o = overall[m]
        n = o["n_total"]
        print(f"{m:12s} {o['f1']:.3f}  {o['dl']:.3f}   "
              f"{o['perfect_match']:>3}/{n} ({o['perfect_match']/n*100:4.1f}%)  "
              f"{o['perfect_recall']:>3}/{n} ({o['perfect_recall']/n*100:4.1f}%)")
    for m, (nv, nt) in counts.items():
        if nv != nt:
            print(f"  note: {m} produced {nv}/{nt} outputs "
                  f"(DL/F1 averaged over {nv}; rates out of {nt}).")

    # ---- Table 2: per category (F1 / DL) ------------------------------------
    print("\n" + "=" * 64)
    print("TABLE 2  Per-category  (F1 / DL similarity)")
    print("=" * 64)
    header = f"{'Event category':27s}" + "".join(f"{m:>18s}" for m in METHOD_FILES)
    print(header)
    for cat in EVENT_CATEGORIES:
        cells = []
        for m in METHOD_FILES:
            c = per_cat[m][cat]
            cells.append(f"{c['f1']:.3f} / {c['dl']:.3f}" if c else "   -   ")
        print(f"{cat:27s}" + "".join(f"{c:>18s}" for c in cells))

    # ---- Table 3: exact-match / full-recall rates ---------------------------
    print("\n" + "=" * 64)
    print("TABLE 3  Exact-match and full-recall rates")
    print("=" * 64)
    print(f"{'Method':12s} {'Perfect match':>18} {'Perfect recall':>18}")
    for m in METHOD_FILES:
        o = overall[m]
        n = o["n_total"]
        print(f"{m:12s} "
              f"{o['perfect_match']:>3}/{n} ({o['perfect_match']/n*100:4.1f}%)   "
              f"{o['perfect_recall']:>3}/{n} ({o['perfect_recall']/n*100:4.1f}%)")
    print()


if __name__ == "__main__":
    main()
