"""Multi-label metrics with confidence intervals on recall."""
import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def score(y_true: list[set], y_pred: list[set], allergens: list[str]) -> dict:
    per = {}
    tp_all = fp_all = fn_all = 0
    for a in allergens:
        tp = sum(1 for t, p in zip(y_true, y_pred) if a in t and a in p)
        fp = sum(1 for t, p in zip(y_true, y_pred) if a not in t and a in p)
        fn = sum(1 for t, p in zip(y_true, y_pred) if a in t and a not in p)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else (0.0 if precision is not None and recall is not None else None))
        per[a] = {
            "support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "recall_ci": wilson(tp, tp + fn),
        }
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn

    p = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    r = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    return {
        "per": per,
        "micro_precision": p,
        "micro_recall": r,
        "micro_f1": 2 * p * r / (p + r) if p + r else 0.0,
        "exact_match": sum(t == q for t, q in zip(y_true, y_pred)) / len(y_true),
        "n": len(y_true),
    }


def _fmt(value) -> str:
    return "  -  " if value is None else f"{value:5.2f}"


def print_report(title: str, result: dict) -> None:
    print(f"\n{title}  (n={result['n']})")
    print(f"{'allergen':11s} {'support':>7s} {'prec':>5s} {'recall':>6s} {'recall 95% CI':>15s} {'f1':>5s}  fn  fp")
    for name, m in result["per"].items():
        low, high = m["recall_ci"]
        ci = f"[{low:.2f}, {high:.2f}]" if m["support"] else "      -"
        print(f"{name:11s} {m['support']:7d} {_fmt(m['precision'])} {_fmt(m['recall'])}  {ci:>15s} "
              f"{_fmt(m['f1'])} {m['fn']:3d} {m['fp']:3d}")
    print(f"micro precision {result['micro_precision']:.2f}   recall {result['micro_recall']:.2f}   "
          f"f1 {result['micro_f1']:.2f}   exact match {result['exact_match']:.2f}")