"""Prints the column names and a few real rows from Open Food Facts."""
import json

from datasets import load_dataset

WANTED = [
    "code", "product_name", "ingredients_text", "allergens_tags",
    "traces_tags", "labels_tags", "countries_tags", "serving_size",
    "nutriments", "lang",
]

ds = load_dataset("openfoodfacts/product-database", split="food", streaming=True)

for i, row in enumerate(ds):
    if i == 0:
        print("COLUMNS:", sorted(row.keys()))
    print(f"\n--- row {i} ---")
    for key in WANTED:
        if key in row:
            print(f"{key}: {json.dumps(row[key], default=str)[:300]}")
        else:
            print(f"{key}: (missing)")
    if i >= 2:
        break