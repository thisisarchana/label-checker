"""Downloads 21 CFR Part 101 (food labeling) from the eCFR API."""
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "cfr"
BASE = "https://www.ecfr.gov/api/versioner/v1"
# Identify yourself, as with the Open Food Facts script. Put your own contact here.
HEADERS = {"User-Agent": "label-checker-portfolio/0.1 (student project; contact: https://github.com/thisisarchana)"}
TITLE, PART = 21, 101


def latest_date() -> str:
    """eCFR lags a day or two behind, so ask for the newest available date, never today's."""
    reply = requests.get(f"{BASE}/titles.json", headers=HEADERS, timeout=60)
    reply.raise_for_status()
    for title in reply.json()["titles"]:
        if title["number"] == TITLE:
            return title.get("up_to_date_as_of") or title["latest_issue_date"]
    raise RuntimeError(f"Title {TITLE} not found in titles.json")


def main() -> None:
    date = latest_date()
    reply = requests.get(f"{BASE}/full/{date}/title-{TITLE}.xml", params={"part": PART},
                         headers=HEADERS, timeout=180)
    reply.raise_for_status()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"title{TITLE}_part{PART}_{date}.xml"
    path.write_bytes(reply.content)

    text = reply.text
    print(f"Regulation date: {date}")
    print(f"Saved {len(reply.content) / 1e6:.1f} MB to {path}")
    print(f"Elements marked TYPE=\"SECTION\": {len(re.findall(r'TYPE=\"SECTION\"', text))}")
    print("\nFirst 1200 characters:\n" + text[:1200])


if __name__ == "__main__":
    main()