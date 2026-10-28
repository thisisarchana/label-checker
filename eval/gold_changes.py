"""Compares the blind labels with the adjudicated labels and writes a change log."""
from collections import Counter

import pandas as pd

from eval.gold import ALLERGENS, GOLD_DIR, load_gold

OUT = GOLD_DIR / "adjudication_log.csv"


def main() -> int:
    blind, problems_a = load_gold("blind")
    adjudicated, problems_b = load_gold("adjudicated")
    for name, problems in (("blind", problems_a), ("adjudicated", problems_b)):
        if problems:
            print(f"Fix unrecognized labels in the {name} file first:", problems)
            return 1
    if set(blind.id) != set(adjudicated.id):
        print("The two files do not contain the same product ids.")
        return 1

    new_by_id = adjudicated.set_index("id")
    rows, added, removed, no_reason = [], Counter(), Counter(), []
    for row in blind.itertuples():
        new = new_by_id.loc[row.id, "labels"]
        if row.labels == new:
            continue
        note = new_by_id.loc[row.id, "notes"]
        if note.strip() == (row.notes or "").strip():
            no_reason.append(row.id)
        for a in new - row.labels:
            added[a] += 1
        for a in row.labels - new:
            removed[a] += 1
        rows.append({
            "id": row.id, "product_name": row.product_name, "source": row.source,
            "before": ";".join(sorted(row.labels)), "after": ";".join(sorted(new)),
            "added": ";".join(sorted(new - row.labels)),
            "removed": ";".join(sorted(row.labels - new)), "reason": note,
        })

    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    cells = len(blind) * len(ALLERGENS)
    n_added, n_removed = sum(added.values()), sum(removed.values())
    print(f"Products changed: {len(rows)} of {len(blind)}")
    print(f"Label cells changed: {n_added + n_removed} of {cells} "
          f"({(n_added + n_removed) / cells:.1%}): {n_added} added, {n_removed} removed")
    print("\nAdded by allergen:  ", dict(added) or "none")
    print("Removed by allergen:", dict(removed) or "none")
    if no_reason:
        print(f"\nChanged rows with no new note explaining why: {', '.join(no_reason)}")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())