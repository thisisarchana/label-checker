"""Downloads the ingredient-panel photo of each gold-set product from Open Food Facts."""
import time
from pathlib import Path

import pandas as pd
import requests

from eval.gold import GOLD_DIR

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "images" / "gold"
KEY = GOLD_DIR / "allergen_gold_key.parquet"

API = "https://world.openfoodfacts.org/api/v2/product/{code}.json"
# Open Food Facts asks API users to identify themselves. Put your own contact here.
HEADERS = {"User-Agent": "label-checker-portfolio/0.1 (student project; contact: https://github.com/thisisarchana)"}
DELAY_SECONDS = 4.5   # stays under about 15 product reads per minute


def main() -> None:
    key = pd.read_parquet(KEY)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = already = no_photo = failed = 0

    for row in key.itertuples():
        target = OUT_DIR / f"{row.id}.jpg"
        if target.exists():
            already += 1
            continue
        try:
            reply = requests.get(API.format(code=row.code), params={"fields": "image_ingredients_url"},
                                 headers=HEADERS, timeout=30)
            reply.raise_for_status()
            url = (reply.json().get("product") or {}).get("image_ingredients_url")
            if not url:
                no_photo += 1
                print(f"{row.id}: no ingredients photo")
            else:
                image = requests.get(url, headers=HEADERS, timeout=60)
                image.raise_for_status()
                target.write_bytes(image.content)
                saved += 1
                print(f"{row.id}: saved")
        except (requests.RequestException, ValueError) as error:
            failed += 1
            print(f"{row.id}: failed ({error})")
        time.sleep(DELAY_SECONDS)

    print(f"\nSaved {saved}, already had {already}, no photo {no_photo}, failed {failed} "
          f"(of {len(key)} gold products). Images are in {OUT_DIR}")


if __name__ == "__main__":
    main()