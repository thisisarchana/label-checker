"""After the cleaner fix: which of your label removals might the old bug have caused?"""
from app.extraction.clean import strip_advisory
from eval.gold import load_gold
from eval.gold_baseline import keyword_predict
from eval.gold_review import context


def main() -> int:
    blind, problems_a = load_gold("blind")
    adjudicated, problems_b = load_gold("adjudicated")
    if problems_a or problems_b:
        print("Fix unrecognized labels first.")
        return 1

    new_by_id = adjudicated.set_index("id")
    total = consistent = 0
    second_look = []
    for row in blind.itertuples():
        new = new_by_id.loc[row.id, "labels"]
        pred = keyword_predict(row.ingredients_text)
        kept = strip_advisory(row.ingredients_text)
        for allergen in sorted(row.labels - new):
            total += 1
            if allergen in pred:
                second_look.append(
                    f"{row.id} {row.product_name[:26]:26s} {allergen:10s} keyword {context(kept, allergen)}"
                )
            else:
                consistent += 1

    print(f"Labels you removed during adjudication: {total}")
    print(f"  No keyword for them anywhere in the text (removal looks fine): {consistent}")
    print(f"  Keyword DOES find them in the text (second look needed): {len(second_look)}\n")
    for line in second_look:
        print(line)
    print("\nFor each line: restore the label unless your written policy excludes it "
          "(refined oil only, plant milk, cross-contact statement).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())