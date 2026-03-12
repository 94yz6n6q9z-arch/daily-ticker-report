#!/usr/bin/env python3
"""
update_msci_world_classification.py  v3.0.0
============================================
Downloads the official iShares XLS holdings files for:
  • MSCI World (251882) — ~1,300 developed-market stocks
  • MSCI EM standard (251858) — ~830 emerging-market stocks

Each XLS row contains: RawTicker, Name, Sector, AssetClass, MarketValue,
Weight%, NominalValue, Shares, Price, Country, Exchange, Currency.

The Exchange column is the source of truth for Yahoo Finance suffix mapping.
No regex, no country ETFs, no guessing — 54 exact exchange name → suffix entries.

Usage
-----
    python update_msci_world_classification.py --universe world  --out config/msci_world_classification.csv
    python update_msci_world_classification.py --universe em     --out-em config/msci_em_classification.csv
    python update_msci_world_classification.py --universe both   --out config/msci_world_classification.csv --out-em config/msci_em_classification.csv

VERSION HISTORY
---------------
1.x  Old multi-ETF approach (IWDA + 9 country ETFs, CSV format, regex suffix guessing)
2.x  Removed Korea/Taiwan country ETFs
3.0.0  Complete rewrite. Reads iShares XLS (SpreadsheetML) directly.
       Exchange column → exact suffix lookup (54 entries).
       Drops Russia, unlisted, fund-of-fund rows.
       No country ETFs. No msci_manual_tickers.json. No EIMI.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests is required: pip install requests")

# ---------------------------------------------------------------------------
# Source URLs — German iShares site, XLS download endpoint
# ---------------------------------------------------------------------------
SOURCES = {
    "world": {
        "url": (
            "https://www.ishares.com/de/privatanleger/de/produkte/251882/"
            "ishares-msci-world-ucits-etf-acc-fund/1535604580385.ajax"
            "?fileType=xls&fileName=iShares-Core-MSCI-World-UCITS-ETF_fund&dataType=fund"
        ),
        "fund_id": "IWDA",
        "label": "World",
    },
    "em": {
        "url": (
            "https://www.ishares.com/de/privatanleger/de/produkte/251858/"
            "ishares-msci-emerging-markets-ucits-etf-acc-fund/1535604580385.ajax"
            "?fileType=xls&fileName=iShares-MSCI-EM-UCITS-ETF-USD-Acc_fund&dataType=fund"
        ),
        "fund_id": "MSCI_EM",
        "label": "EM",
    },
}

# ---------------------------------------------------------------------------
# Exchange name → Yahoo Finance suffix
# Source: exhaustive list of all exchange names found in both XLS files.
# None = skip this row entirely (dead market or unlistable).
# ---------------------------------------------------------------------------
EXCHANGE_SUFFIX: dict[str, str | None] = {
    # ── United States ──────────────────────────────────────────────────────
    "NASDAQ":                                       "",
    "New York Stock Exchange Inc.":                 "",
    "Cboe BZX":                                     "",
    # ── Canada ─────────────────────────────────────────────────────────────
    "Toronto Stock Exchange":                       ".TO",
    # ── UK ─────────────────────────────────────────────────────────────────
    "London Stock Exchange":                        ".L",
    "Irish Stock Exchange - All Market":            ".IR",
    # ── Continental Europe ─────────────────────────────────────────────────
    "Xetra":                                        ".DE",
    "Nyse Euronext - Euronext Paris":               ".PA",
    "Euronext Amsterdam":                           ".AS",
    "Borsa Italiana":                               ".MI",
    "Bolsa De Madrid":                              ".MC",
    "SIX Swiss Exchange":                           ".SW",
    "Nasdaq Omx Nordic":                            ".ST",   # Stockholm
    "Nasdaq Omx Helsinki Ltd.":                     ".HE",
    "Omx Nordic Exchange Copenhagen A/S":           ".CO",
    "Oslo Bors Asa":                                ".OL",
    "Nyse Euronext - Euronext Brussels":            ".BR",
    "Nyse Euronext - Euronext Lisbon":              ".LS",
    "Wiener Boerse Ag":                             ".VI",
    "Warsaw Stock Exchange/Equities/Main Market":   ".WA",
    "Athens Exchange S.A. Cash Market":             ".AT",
    "Prague Stock Exchange":                        ".PR",
    "Budapest Stock Exchange":                      ".BD",
    # ── Asia-Pacific (Developed) ───────────────────────────────────────────
    "Tokyo Stock Exchange":                         ".T",
    "Hong Kong Exchanges And Clearing Ltd":         ".HK",
    "Asx - All Markets":                            ".AX",
    "Singapore Exchange":                           ".SI",
    "New Zealand Exchange Ltd":                     ".NZ",
    # ── Asia-Pacific (Emerging) ────────────────────────────────────────────
    "Taiwan Stock Exchange":                        ".TW",
    "Gretai Securities Market":                     ".TWO",  # Taiwan OTC
    "Korea Exchange (Stock Market)":                ".KS",
    "Korea Exchange (Kosdaq)":                      ".KQ",
    "National Stock Exchange Of India":             ".NS",
    "Bse Ltd":                                      ".BO",
    "Shanghai Stock Exchange":                      ".SS",
    "Shenzhen Stock Exchange":                      ".SZ",
    "Bursa Malaysia":                               ".KL",
    "Indonesia Stock Exchange":                     ".JK",
    "Stock Exchange Of Thailand":                   ".BK",
    "Philippine Stock Exchange Inc.":               ".PS",
    # ── Middle East / Africa ───────────────────────────────────────────────
    "Saudi Stock Exchange":                         ".SR",
    "Kuwait Stock Exchange":                        ".KW",
    "Qatar Exchange":                               ".QA",
    "Abu Dhabi Securities Exchange":                ".AD",
    "Dubai Financial Market":                       ".DU",
    "Tel Aviv Stock Exchange":                      ".TA",
    "Johannesburg Stock Exchange":                  ".JO",
    "Egyptian Exchange":                            ".CA",
    # ── LatAm ──────────────────────────────────────────────────────────────
    "XBSP":                                         ".SA",   # B3 Brazil
    "Bolsa Mexicana De Valores":                    ".MX",
    "Santiago Stock Exchange":                      ".SN",
    "Bolsa De Valores De Colombia":                 ".CL",
    # ── Eastern Europe ─────────────────────────────────────────────────────
    "Istanbul Stock Exchange":                      ".IS",
    # ── Dead / skip ────────────────────────────────────────────────────────
    "Standard-Classica-Forts":                      None,    # Russia — sanctioned
    "NO MARKET (E.G. UNLISTED)":                    None,    # unlisted
}

# ---------------------------------------------------------------------------
# Rows to skip regardless of exchange (ETF-in-ETF / fund-of-fund)
# ---------------------------------------------------------------------------
SKIP_TICKERS = {"CNYA"}   # iShares MSCI China A ETF — fund-of-fund embedded in EM

# ---------------------------------------------------------------------------
# Per-exchange ticker formatting rules
# ---------------------------------------------------------------------------
_NORDIC_EXCHANGES = {
    "Nasdaq Omx Nordic",
    "Omx Nordic Exchange Copenhagen A/S",
    "Nasdaq Omx Helsinki Ltd.",
    "Oslo Bors Asa",
}

_HK_PAD_LEN = 4    # Hong Kong: zero-pad to 4 digits (e.g. "700" → "0700")

_NUMERIC_EXCHANGES = {
    "Tokyo Stock Exchange",
    "Hong Kong Exchanges And Clearing Ltd",
    "Taiwan Stock Exchange",
    "Korea Exchange (Stock Market)",
    "Korea Exchange (Kosdaq)",
    "Gretai Securities Market",
    "Shanghai Stock Exchange",
    "Shenzhen Stock Exchange",
    "Kuwait Stock Exchange",
    "Qatar Exchange",
    "Saudi Stock Exchange",
    "Indonesia Stock Exchange",
}


def _format_ticker(raw: str, exchange: str, suffix: str) -> str | None:
    """Apply exchange-specific formatting and append Yahoo suffix.

    Returns None if the ticker is unusable.
    """
    t = raw.strip()
    if not t or t == "-":
        return None

    # Nordic: share-class space → dash  ("NOVO B" → "NOVO-B")
    if exchange in _NORDIC_EXCHANGES and " " in t:
        t = t.replace(" ", "-")

    # Hong Kong: zero-pad 4 digits
    if exchange == "Hong Kong Exchanges And Clearing Ltd" and t.isdigit():
        t = t.zfill(_HK_PAD_LEN)

    return t + suffix


# ---------------------------------------------------------------------------
# iShares XLS download
# ---------------------------------------------------------------------------
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.ishares.com/",
}

_MAX_RETRIES = 3
_RETRY_DELAY = 15   # seconds


def _download_xls(url: str, label: str) -> bytes:
    """Download an XLS file from iShares with retries."""
    session = requests.Session()
    # Warm-up: visit the product page so cookies are set
    product_page = url.split("/1535604580385.ajax")[0]
    try:
        session.get(product_page, headers=_HEADERS, timeout=30)
        time.sleep(2)
    except Exception:
        pass  # warm-up is best-effort

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=_HEADERS, timeout=60)
            resp.raise_for_status()
            if len(resp.content) < 1000:
                raise ValueError(f"Response too small ({len(resp.content)} bytes) — likely a redirect/error page")
            print(f"[msci-refresh:{label}] downloaded {len(resp.content):,} bytes (attempt {attempt})")
            return resp.content
        except Exception as exc:
            print(f"[msci-refresh:{label}] attempt {attempt}/{_MAX_RETRIES} failed: {exc}")
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * attempt)
    sys.exit(f"[msci-refresh:{label}] all {_MAX_RETRIES} download attempts failed — aborting")


# ---------------------------------------------------------------------------
# XLS (SpreadsheetML) parser
# ---------------------------------------------------------------------------
def _parse_spreadsheetml(raw_bytes: bytes) -> list[list[str]]:
    """Parse iShares SpreadsheetML XML → list of rows (list of cell strings)."""
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    # Strip all namespace declarations for simpler parsing
    text = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', "", text)
    text = re.sub(r"\bss:", "", text)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        sys.exit(f"[msci-refresh] XML parse error: {exc}")

    rows: list[list[str]] = []
    for table in root.findall(".//Table"):
        for row in table.findall("Row"):
            cells: list[str] = []
            for cell in row.findall("Cell"):
                data = cell.find("Data")
                cells.append(data.text if data is not None and data.text else "")
            rows.append(cells)
    return rows


# ---------------------------------------------------------------------------
# Holdings extraction
# ---------------------------------------------------------------------------
# Column indices in the iShares XLS data block:
_COL_TICKER   = 0
_COL_NAME     = 1
_COL_SECTOR   = 2
_COL_CLASS    = 3   # "Aktien" = equity
_COL_MKVAL    = 4   # market value of holding (not company market cap)
_COL_WEIGHT   = 5   # weight % in fund
_COL_PRICE    = 8   # share price in local currency
_COL_COUNTRY  = 9
_COL_EXCHANGE = 10
_COL_CURRENCY = 11

_EQUITY_CLASS = "Aktien"   # German for "equities"
_HEADER_MARKER = "Emittententicker"   # first column of the data header row


def _find_data_start(rows: list[list[str]]) -> tuple[int, str]:
    """Return (data_start_index, fund_name) from the XLS rows."""
    fund_name = ""
    for i, row in enumerate(rows):
        if len(row) >= 2 and not fund_name:
            # Row 1 (0-indexed) usually has the fund name
            if row[0] and row[0] != rows[0][0]:
                fund_name = row[0]
        if row and row[0] == _HEADER_MARKER:
            return i + 1, fund_name
    sys.exit("[msci-refresh] could not find data header row in XLS")


def _extract_holdings(rows: list[list[str]], fund_id: str) -> list[dict]:
    """Extract equity holdings from parsed XLS rows."""
    data_start, _fund_name_raw = _find_data_start(rows)

    # Grab fund date from metadata rows (row index 3 or similar)
    source_date = ""
    for row in rows[:8]:
        if len(row) >= 2 and "Holdings as of" in str(row[0]):
            source_date = row[1]
            break

    results: list[dict] = []
    skipped_exchange: list[str] = []
    skipped_rows: int = 0

    for row in rows[data_start:]:
        if len(row) <= _COL_CURRENCY:
            continue
        if row[_COL_CLASS] != _EQUITY_CLASS:
            continue

        raw_ticker = row[_COL_TICKER].strip()
        exchange   = row[_COL_EXCHANGE].strip()
        name       = row[_COL_NAME].strip()
        sector     = row[_COL_SECTOR].strip()
        country    = row[_COL_COUNTRY].strip()
        weight     = row[_COL_WEIGHT].strip()

        # Skip fund-of-fund rows
        if raw_ticker in SKIP_TICKERS:
            skipped_rows += 1
            continue

        # Resolve suffix
        if exchange not in EXCHANGE_SUFFIX:
            skipped_exchange.append(exchange)
            continue

        suffix = EXCHANGE_SUFFIX[exchange]
        if suffix is None:
            skipped_rows += 1
            continue

        ticker = _format_ticker(raw_ticker, exchange, suffix)
        if ticker is None:
            skipped_rows += 1
            continue

        results.append({
            "Ticker":            ticker,
            "Company":           name,
            "Country":           country,
            "Sector":            sector,
            "RawTicker":         raw_ticker,
            "Exchange":          exchange,
            "ISIN":              "",
            "WeightPct":         weight,
            "SourceFund":        fund_id,
            "SourceURL":         "",          # filled in by caller
            "SourceAsOf":        source_date,
            "MappingConfidence": "high",
        })

    if skipped_exchange:
        uniq = sorted(set(skipped_exchange))
        print(f"[msci-refresh:{fund_id}] WARNING — {len(skipped_exchange)} rows with unmapped exchanges: {uniq}")
    if skipped_rows:
        print(f"[msci-refresh:{fund_id}] skipped {skipped_rows} rows (Russia/unlisted/ETF-in-ETF)")

    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
_CSV_FIELDS = [
    "Ticker", "Company", "Country", "Sector",
    "RawTicker", "Exchange", "ISIN", "WeightPct",
    "SourceFund", "SourceURL", "SourceAsOf", "MappingConfidence",
]


def _write_csv(holdings: list[dict], out_path: Path, url: str) -> None:
    """Write holdings to CSV, filling SourceURL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for h in holdings:
        h["SourceURL"] = url
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(holdings)
    print(f"[msci-refresh] wrote {len(holdings)} rows → {out_path}")


def _write_meta(holdings: list[dict], meta_path: Path, label: str, url: str) -> None:
    """Write a small JSON metadata file."""
    if not meta_path:
        return
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    country_counts = dict(Counter(h["Country"] for h in holdings).most_common(30))
    meta = {
        "as_of":      date.today().isoformat(),
        "source_url": url,
        "n_tickers":  len(holdings),
        "label":      label,
        "countries":  country_counts,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[msci-refresh] meta → {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _process(key: str, out_path: Path | None, meta_path: Path | None) -> list[dict]:
    cfg = SOURCES[key]
    label = cfg["label"]
    url   = cfg["url"]

    print(f"[msci-refresh:{label}] downloading from iShares …")
    raw = _download_xls(url, label)

    print(f"[msci-refresh:{label}] parsing XLS …")
    rows = _parse_spreadsheetml(raw)

    holdings = _extract_holdings(rows, cfg["fund_id"])
    print(f"[msci-refresh:{label}] {len(holdings)} equity holdings extracted")

    if out_path:
        _write_csv(holdings, out_path, url)
    if meta_path:
        _write_meta(holdings, meta_path, label, url)

    return holdings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh MSCI World / EM classification CSVs from iShares XLS",
    )
    parser.add_argument(
        "--universe",
        choices=["world", "em", "both", "all"],
        default="both",
        help="Which universe(s) to refresh (default: both)",
    )
    parser.add_argument("--out",     type=Path, default=None,
                        help="Output CSV path for MSCI World")
    parser.add_argument("--out-em",  type=Path, default=None,
                        help="Output CSV path for MSCI EM")
    parser.add_argument("--meta",    type=Path, default=None,
                        help="Output JSON metadata path for MSCI World")
    parser.add_argument("--meta-em", type=Path, default=None,
                        help="Output JSON metadata path for MSCI EM")
    args = parser.parse_args()

    do_world = args.universe in ("world", "both", "all")
    do_em    = args.universe in ("em", "both", "all")

    if do_world:
        _process("world", args.out, args.meta)
    if do_em:
        _process("em", args.out_em, args.meta_em)


if __name__ == "__main__":
    main()
