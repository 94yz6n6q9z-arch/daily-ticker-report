#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refresh config/msci_world_classification.csv from a public iShares MSCI World ETF holdings CSV
and normalize sector labels to the S&P 500 / GICS 11 sectors used by scan.py.

Why this exists
---------------
The exact MSCI World constituent list is typically distributed through licensed data feeds.
For an automated public-source workflow, we use a broad iShares MSCI World ETF holdings file as
an operational proxy (typically ~1,200-1,400 positions, depending on fund replication and date).

Outputs
-------
- CSV (default): config/msci_world_classification.csv
- Metadata JSON (optional): docs/msci_world_classification_meta.json

CSV columns (scan.py only requires Ticker/Company/Sector; extras are kept for debugging):
- Ticker (best-effort Yahoo Finance style symbol)
- Company
- Sector (canonical S&P 500 11 sectors)
- RawTicker
- Exchange
- Country
- ISIN
- WeightPct
- SourceFund
- SourceURL
- SourceAsOf
- MappingConfidence
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

# Try UCITS + US listings; first success wins.
# These endpoints are the same style as the ones exposed by iShares pages' "download holdings" links.
SOURCE_CANDIDATES = [
    {
        "fund": "EUNL",
        "url": "https://www.ishares.com/de/privatanleger/de/produkte/251882/ishares-msci-world-ucits-etf-acc-fund/1506575576011.ajax?dataType=fund&fileName=EUNL_holdings&fileType=csv",
        "referer": "https://www.ishares.com/de/privatanleger/de/produkte/251882/ishares-msci-world-ucits-etf-acc-fund",
    },
    {
        "fund": "IWDA",
        "url": "https://www.ishares.com/uk/individual/en/products/251882/ishares-core-msci-world-ucits-etf-acc-fund/1506575576011.ajax?dataType=fund&fileName=IWDA_holdings&fileType=csv",
        "referer": "https://www.ishares.com/uk/individual/en/products/251882/ishares-core-msci-world-ucits-etf-acc-fund",
    },
    {
        "fund": "URTH",
        "url": "https://www.ishares.com/us/products/239696/ishares-msci-world-etf/1467271812596.ajax?dataType=fund&fileName=URTH_holdings&fileType=csv",
        "referer": "https://www.ishares.com/us/products/239696/ishares-msci-world-etf",
    },
]

SP500_11 = [
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
]

SECTOR_MAP = {
    # English
    "communication services": "Communication Services",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "energy": "Energy",
    "financials": "Financials",
    "health care": "Health Care",
    "healthcare": "Health Care",
    "industrials": "Industrials",
    "information technology": "Information Technology",
    "informationstechnologie": "Information Technology",
    "materials": "Materials",
    "real estate": "Real Estate",
    "utilities": "Utilities",
    # German (common iShares DE export labels)
    "kommunikationsdienste": "Communication Services",
    "zyklische konsumgüter": "Consumer Discretionary",
    "zyklische konsumgueter": "Consumer Discretionary",
    "nichtzyklische konsumgüter": "Consumer Staples",
    "nichtzyklische konsumgueter": "Consumer Staples",
    "basiskonsumgüter": "Consumer Staples",
    "basiskonsumgueter": "Consumer Staples",
    "energie": "Energy",
    "finanzwerte": "Financials",
    "finanzen": "Financials",
    "gesundheitswesen": "Health Care",
    "industrie": "Industrials",
    "industrieunternehmen": "Industrials",
    "grundstoffe": "Materials",
    "roh-, hilfs- & betriebsstoffe": "Materials",
    "immobilien": "Real Estate",
    "versorger": "Utilities",
    "versorgungsunternehmen": "Utilities",
    # French / other possible variants from EU pages
    "services de communication": "Communication Services",
    "consommation discrétionnaire": "Consumer Discretionary",
    "consommation discretionnaire": "Consumer Discretionary",
    "biens de consommation de base": "Consumer Staples",
    "santé": "Health Care",
    "sante": "Health Care",
    "technologies de l'information": "Information Technology",
    "technologies de linformation": "Information Technology",
    "matériaux": "Materials",
    "materiaux": "Materials",
    "services publics": "Utilities",
    # Italian / Spanish (defensive)
    "beni di consumo discrezionali": "Consumer Discretionary",
    "beni di consumo di base": "Consumer Staples",
    "sanità": "Health Care",
    "sanita": "Health Care",
    "materiali": "Materials",
    "servizi di pubblica utilità": "Utilities",
    "servizi di pubblica utilita": "Utilities",
    "servicios de comunicación": "Communication Services",
    "servicios de comunicacion": "Communication Services",
    "consumo discrecional": "Consumer Discretionary",
    "productos de consumo básico": "Consumer Staples",
    "productos de consumo basico": "Consumer Staples",
    "salud": "Health Care",
    "tecnología de la información": "Information Technology",
    "tecnologia de la informacion": "Information Technology",
    "materiales": "Materials",
    "servicios públicos": "Utilities",
    "servicios publicos": "Utilities",
}

HEADER_SYNONYMS = {
    "ticker": ["ticker", "issuer ticker", "emittententicker", "ticker/symbol", "symbol"],
    "company": ["name", "issuer name", "security", "bezeichnung", "name des emittenten"],
    "sector": ["sector", "sektor", "gics sector"],
    "asset_class": ["asset class", "anlageklasse"],
    "exchange": ["exchange", "börse", "boerse", "trading venue"],
    "country": ["location", "country", "land", "domicile", "standort"],
    "isin": ["isin"],
    "weight": ["weight (%)", "gewichtung (%)", "% of net assets", "weight"],
}

FOOTER_MARKERS = [
    "fund holdings as of",
    "holdings are subject to change",
    "the values of",
    "important information",
    "positionen per",
    "die bestände",
    "nettoinventarwert",
]

EXCHANGE_SUFFIX_RULES: List[Tuple[re.Pattern, str, str]] = [
    # regex, yahoo suffix, confidence label for appended suffix
    (re.compile(r"NASDAQ|NEW\s+YORK\s+STOCK\s+EXCHANGE|NYSE|CBOE|BATS|ARCA", re.I), "", "high"),
    (re.compile(r"TORONTO\s+STOCK\s+EXCHANGE|TSX", re.I), ".TO", "high"),
    (re.compile(r"LONDON\s+STOCK\s+EXCHANGE|LSE", re.I), ".L", "high"),
    (re.compile(r"XETRA|DEUTSCHE\s+BOERSE|FRANKFURT", re.I), ".DE", "high"),
    (re.compile(r"EURONEXT\s+PARIS", re.I), ".PA", "high"),
    (re.compile(r"EURONEXT\s+AMSTERDAM", re.I), ".AS", "high"),
    (re.compile(r"EURONEXT\s+BRUSSELS", re.I), ".BR", "high"),
    (re.compile(r"EURONEXT\s+MILAN|BORSA\s+ITALIANA|MILAN", re.I), ".MI", "high"),
    (re.compile(r"SIX\s+SWISS|SWISS\s+EXCHANGE|SIX", re.I), ".SW", "med"),
    (re.compile(r"MADRID|BME", re.I), ".MC", "high"),
    (re.compile(r"TOKYO\s+STOCK\s+EXCHANGE|TSE\b|JPX", re.I), ".T", "high"),
    (re.compile(r"HONG\s*KONG|HKEX", re.I), ".HK", "high"),
    (re.compile(r"AUSTRALIAN\s+SECURITIES\s+EXCHANGE|\bASX\b", re.I), ".AX", "high"),
    (re.compile(r"NASDAQ\s+OMX\s+COPENHAGEN|COPENHAGEN", re.I), ".CO", "med"),
    (re.compile(r"NASDAQ\s+OMX\s+STOCKHOLM|STOCKHOLM", re.I), ".ST", "med"),
    (re.compile(r"NASDAQ\s+OMX\s+HELSINKI|HELSINKI", re.I), ".HE", "med"),
    (re.compile(r"OSLO\s+BORS|OSLO", re.I), ".OL", "med"),
    (re.compile(r"TEL\s+AVIV|TASE", re.I), ".TA", "med"),
    (re.compile(r"WIENER\s+BOERSE|VIENNA", re.I), ".VI", "med"),
    (re.compile(r"SINGAPORE\s+EXCHANGE|\bSGX\b", re.I), ".SI", "med"),
    (re.compile(r"NEW\s+ZEALAND\s+EXCHANGE|\bNZX\b", re.I), ".NZ", "med"),
    (re.compile(r"EURONEXT\s+DUBLIN|IRISH\s+STOCK\s+EXCHANGE|DUBLIN", re.I), ".IR", "low"),
    (re.compile(r"EURONEXT\s+LISBON|LISBON", re.I), ".LS", "med"),
]


@dataclass
class FetchResult:
    fund: str
    url: str
    text: str
    content_type: str


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).strip()


def _norm_key(s: str) -> str:
    s = _norm(s).lower()
    # strip accents minimally for common cases if unidecode not installed
    repl = {
        "ä": "a", "ö": "o", "ü": "u", "ß": "ss",
        "é": "e", "è": "e", "ê": "e", "á": "a", "à": "a", "â": "a",
        "í": "i", "ì": "i", "î": "i", "ó": "o", "ò": "o", "ô": "o",
        "ú": "u", "ù": "u", "û": "u", "ç": "c", "ñ": "n",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def canonical_sector(label: str) -> str:
    k = _norm_key(label)
    return SECTOR_MAP.get(k, _norm(label))


def fetch_holdings_csv(timeout: int = 45) -> FetchResult:
    last_err: Optional[Exception] = None
    for src in SOURCE_CANDIDATES:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/csv,text/plain,application/octet-stream,*/*",
            "Accept-Language": "en-US,en;q=0.9,de;q=0.7",
            "Referer": src["referer"],
            "Origin": re.match(r"https?://[^/]+", src["referer"]).group(0),
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        try:
            r = requests.get(src["url"], headers=headers, timeout=timeout)
            ct = r.headers.get("content-type", "")
            if r.status_code >= 400:
                # Include response snippet for debugging.
                snippet = (r.text or "")[:250].replace("\n", " ")
                raise RuntimeError(f"HTTP {r.status_code} from {src['fund']} ({src['url']}) | {snippet}")
            text = r.text
            if not text or len(text) < 100:
                raise RuntimeError(f"Empty/short response from {src['fund']}")
            return FetchResult(fund=src["fund"], url=src["url"], text=text, content_type=ct)
        except Exception as e:  # noqa: BLE001 - we want resilient fallback across sources
            last_err = e
            print(f"[msci-refresh] source {src['fund']} failed: {e}")
            continue
    raise RuntimeError(f"All holdings sources failed. Last error: {last_err}")


def _detect_encoding_to_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def _detect_delimiter(lines: List[str]) -> str:
    # Heuristic: look for a header row containing ticker+sector/company synonyms.
    candidates = [",", ";", "\t", "|"]
    for delim in candidates:
        for ln in lines[:120]:
            if not ln.strip():
                continue
            row = next(csv.reader([ln], delimiter=delim))
            keys = {_norm_key(x) for x in row}
            has_ticker = any(k in keys for k in HEADER_SYNONYMS["ticker"])
            has_company = any(k in keys for k in HEADER_SYNONYMS["company"])
            has_sector = any(k in keys for k in HEADER_SYNONYMS["sector"])
            if has_ticker and (has_company or has_sector):
                return delim
    # Fallback: choose the delimiter that yields the highest average columns.
    best = ","
    best_score = -1.0
    for delim in candidates:
        widths = []
        for ln in lines[:50]:
            if not ln.strip():
                continue
            try:
                widths.append(len(next(csv.reader([ln], delimiter=delim))))
            except Exception:
                pass
        score = sum(widths) / max(1, len(widths))
        if score > best_score:
            best = delim
            best_score = score
    return best


def _find_header_and_rows(text: str) -> Tuple[List[str], List[List[str]]]:
    lines = text.splitlines()
    delim = _detect_delimiter(lines)
    parsed = [next(csv.reader([ln], delimiter=delim)) for ln in lines]

    header_idx = None
    for i, row in enumerate(parsed[:250]):
        keys = {_norm_key(x) for x in row}
        has_ticker = any(k in keys for k in HEADER_SYNONYMS["ticker"])
        has_company = any(k in keys for k in HEADER_SYNONYMS["company"])
        has_sector = any(k in keys for k in HEADER_SYNONYMS["sector"])
        if has_ticker and (has_company or has_sector):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not find holdings CSV header row")

    header = [_norm(h) for h in parsed[header_idx]]
    rows: List[List[str]] = []
    expected_cols = len(header)
    for row in parsed[header_idx + 1 :]:
        if not row:
            continue
        first = _norm_key(row[0]) if row else ""
        joined = _norm_key(" ".join(x for x in row[:3] if x))
        if any(m in first for m in FOOTER_MARKERS) or any(m in joined for m in FOOTER_MARKERS):
            break
        # Pad/truncate row length to header length for safer indexing
        if len(row) < expected_cols:
            row = row + [""] * (expected_cols - len(row))
        elif len(row) > expected_cols:
            row = row[:expected_cols]
        # Skip obvious empty lines / separators
        if not any(_norm(x) for x in row):
            continue
        rows.append([_norm(x) for x in row])
    return header, rows


def _pick_col(cols: List[str], key: str) -> Optional[int]:
    synonyms = {_norm_key(x) for x in HEADER_SYNONYMS[key]}
    for i, c in enumerate(cols):
        if _norm_key(c) in synonyms:
            return i
    return None


def parse_ishares_holdings(text: str) -> Tuple[pd.DataFrame, Optional[str]]:
    header, rows = _find_header_and_rows(text)
    if not rows:
        raise RuntimeError("No data rows found under holdings CSV header")

    ci = {k: _pick_col(header, k) for k in HEADER_SYNONYMS}
    if ci.get("ticker") is None or ci.get("company") is None or ci.get("sector") is None:
        raise RuntimeError(f"Required columns missing. Header columns: {header}")

    data = []
    for row in rows:
        rec = {
            "RawTicker": row[ci["ticker"]] if ci["ticker"] is not None else "",
            "Company": row[ci["company"]] if ci["company"] is not None else "",
            "SectorRaw": row[ci["sector"]] if ci["sector"] is not None else "",
            "AssetClass": row[ci["asset_class"]] if ci.get("asset_class") is not None else "",
            "Exchange": row[ci["exchange"]] if ci.get("exchange") is not None else "",
            "Country": row[ci["country"]] if ci.get("country") is not None else "",
            "ISIN": row[ci["isin"]] if ci.get("isin") is not None else "",
            "WeightRaw": row[ci["weight"]] if ci.get("weight") is not None else "",
        }
        data.append(rec)

    df = pd.DataFrame(data)

    # Extract as-of date if present in top metadata text.
    m = re.search(
        r"(?:Fund Holdings as of|Positionen per|As of|Daten per)\s*[:\-]?\s*([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})",
        text,
        flags=re.I,
    )
    as_of = m.group(1).strip() if m else None

    return df, as_of


def _clean_symbol_base(raw: str) -> str:
    s = _norm(raw).upper()
    s = s.replace(" ", "")
    s = s.replace("/", "-")
    # Common US share class notation to Yahoo style (BRK.B -> BRK-B) when no exchange suffix yet.
    if re.fullmatch(r"[A-Z]{1,5}\.[A-Z]", s):
        s = s.replace(".", "-")
    return s


def _append_yahoo_suffix(base: str, exchange: str) -> Tuple[str, str]:
    ex = _norm(exchange)
    for pat, suffix, conf in EXCHANGE_SUFFIX_RULES:
        if pat.search(ex):
            if suffix == "":
                return base, "high"
            # HK / Tokyo formatting tweaks
            if suffix == ".HK":
                # Yahoo usually uses 4-digit zero-padded numeric symbols in HK
                if re.fullmatch(r"\d{1,4}", base):
                    base = base.zfill(4)
            return (base + suffix if not base.endswith(suffix) else base), conf
    return base, "low"


def guess_yahoo_ticker(raw_ticker: str, exchange: str) -> Tuple[str, str]:
    base = _clean_symbol_base(raw_ticker)
    if not base:
        return "", "low"

    # If symbol already looks like a Yahoo ticker with suffix, keep as-is.
    if re.search(r"\.[A-Z]{1,3}$", base):
        return base, "high"

    # Numeric Japanese stocks often 4 digits; no need to pad.
    # Numeric HK names handled in suffix append.
    guessed, conf = _append_yahoo_suffix(base, exchange)
    return guessed, conf


def normalize_weight(x: str) -> Optional[float]:
    s = _norm(x)
    if not s:
        return None
    # Handle locales: 1,23 or 1.23 or 1,234.56
    s = s.replace("%", "")
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    elif s.count(",") > 1 and s.count(".") == 0:
        s = s.replace(",", "")
    elif s.count(",") >= 1 and s.count(".") >= 1:
        # Assume comma is thousands sep in 1,234.56 format
        s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def filter_to_equities(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    d = df.copy()
    for c in ["RawTicker", "Company", "SectorRaw", "AssetClass", "Exchange", "Country", "ISIN"]:
        if c in d.columns:
            d[c] = d[c].fillna("").astype(str).str.strip()

    # Drop obvious footer/informational lines that slipped through
    bad_company = d["Company"].str.lower().str.contains(
        r"cash and/or derivatives|cash und/oder derivate|total|summe|futures|forward|swap",
        regex=True,
        na=False,
    )
    d = d[~bad_company]

    # Keep equities if AssetClass exists; otherwise keep rows with a sector and ticker.
    if "AssetClass" in d.columns and d["AssetClass"].str.strip().ne("").any():
        eq_mask = d["AssetClass"].str.lower().str.contains(r"equity|aktie|stock", regex=True, na=False)
        d = d[eq_mask]
    else:
        d = d[d["SectorRaw"].astype(str).str.strip().ne("")]

    d = d[d["RawTicker"].astype(str).str.strip().ne("")]
    d = d[d["Company"].astype(str).str.strip().ne("")]
    return d.reset_index(drop=True)


def build_output_dataframe(raw_df: pd.DataFrame, source_fund: str, source_url: str, source_as_of: Optional[str]) -> pd.DataFrame:
    df = filter_to_equities(raw_df)
    if df.empty:
        raise RuntimeError("No equity holdings after filtering")

    out = pd.DataFrame()
    out["RawTicker"] = df["RawTicker"].astype(str).str.strip()
    out["Company"] = df["Company"].astype(str).str.strip()
    out["Sector"] = df["SectorRaw"].map(canonical_sector).astype(str).str.strip()
    out["Exchange"] = df.get("Exchange", "").astype(str).str.strip() if "Exchange" in df.columns else ""
    out["Country"] = df.get("Country", "").astype(str).str.strip() if "Country" in df.columns else ""
    out["ISIN"] = df.get("ISIN", "").astype(str).str.strip() if "ISIN" in df.columns else ""
    out["WeightPct"] = df.get("WeightRaw", "").map(normalize_weight) if "WeightRaw" in df.columns else None

    guessed = out.apply(lambda r: guess_yahoo_ticker(r["RawTicker"], r["Exchange"]), axis=1)
    out["Ticker"] = [g[0] for g in guessed]
    out["MappingConfidence"] = [g[1] for g in guessed]

    # Clean pathological symbols that yfinance will reject often.
    out["Ticker"] = (
        out["Ticker"].astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(r"[^A-Z0-9\-\.=]", "", regex=True)
    )

    # Source metadata per row (handy for debugging when file is opened standalone)
    out["SourceFund"] = source_fund
    out["SourceURL"] = source_url
    out["SourceAsOf"] = (source_as_of or "")

    # Drop rows with unmapped/unknown sector labels (keep them if you prefer, but scan.py expects 11 sectors for non-watchlist names)
    known_sector = out["Sector"].isin(SP500_11)
    dropped_unknown = int((~known_sector).sum())
    if dropped_unknown:
        examples = out.loc[~known_sector, ["Company", "Sector"]].head(5).to_dict("records")
        print(f"[msci-refresh] dropping {dropped_unknown} rows with non-canonical sectors. examples={examples}")
    out = out[known_sector].copy()

    # Deduplicate by Ticker (prefer higher weight / higher confidence / more complete row)
    conf_rank = {"high": 2, "med": 1, "low": 0}
    out["_conf_rank"] = out["MappingConfidence"].map(conf_rank).fillna(0)
    out["_w"] = out["WeightPct"].fillna(-1.0)
    out["_len_company"] = out["Company"].astype(str).str.len().fillna(0)
    out = out.sort_values(["Ticker", "_w", "_conf_rank", "_len_company"], ascending=[True, False, False, False])
    out = out.drop_duplicates(subset=["Ticker"], keep="first")

    # Final column order expected by scan.py (+ extras)
    out = out[
        [
            "Ticker",
            "Company",
            "Country",
            "Sector",
            "RawTicker",
            "Exchange",
            "ISIN",
            "WeightPct",
            "SourceFund",
            "SourceURL",
            "SourceAsOf",
            "MappingConfidence",
        ]
    ].copy()

    out = out.sort_values(["Sector", "Company", "Ticker"], ascending=[True, True, True]).reset_index(drop=True)
    return out


def load_existing_ticker_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        prev = pd.read_csv(path, dtype=str)
        col = None
        for c in prev.columns:
            if str(c).strip().lower() in {"ticker", "symbol"}:
                col = c
                break
        if col is None:
            return set()
        return {str(x).strip() for x in prev[col].fillna("").astype(str) if str(x).strip()}
    except Exception as e:
        print(f"[msci-refresh] warning: failed reading previous file for diff: {e}")
        return set()


def write_metadata(meta_path: Path, payload: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh MSCI World proxy constituent classification CSV")
    ap.add_argument("--out", default="config/msci_world_classification.csv", help="Output CSV path")
    ap.add_argument("--meta", default="docs/msci_world_classification_meta.json", help="Metadata JSON path (optional)")
    ap.add_argument("--min-rows", type=int, default=900, help="Fail-safe minimum row count to avoid overwriting with partial data")
    ap.add_argument("--allow-partial", action="store_true", help="Allow writing even if row count < --min-rows")
    args = ap.parse_args()

    out_path = Path(args.out)
    meta_path = Path(args.meta) if args.meta else None

    prev_tickers = load_existing_ticker_set(out_path)

    fetched = fetch_holdings_csv()
    raw_df, source_as_of = parse_ishares_holdings(fetched.text)
    out_df = build_output_dataframe(raw_df, fetched.fund, fetched.url, source_as_of)

    row_count = len(out_df)
    if row_count < args.min_rows and not args.allow_partial:
        raise RuntimeError(
            f"Refusing to write {out_path}: only {row_count} rows (< {args.min_rows}). "
            "This looks like a partial parse or source issue."
        )

    new_tickers = set(out_df["Ticker"].astype(str).tolist())
    added = sorted(new_tickers - prev_tickers)
    removed = sorted(prev_tickers - new_tickers)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    stats = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_fund": fetched.fund,
        "source_url": fetched.url,
        "source_as_of": source_as_of,
        "row_count": int(row_count),
        "unique_tickers": int(len(new_tickers)),
        "sector_counts": {k: int(v) for k, v in out_df["Sector"].value_counts().sort_index().to_dict().items()},
        "mapping_confidence_counts": {k: int(v) for k, v in out_df["MappingConfidence"].value_counts().sort_index().to_dict().items()},
        "added_count": int(len(added)),
        "removed_count": int(len(removed)),
        "added_sample": added[:25],
        "removed_sample": removed[:25],
        "notes": [
            "Public-source proxy (iShares MSCI World ETF holdings), not a licensed direct MSCI constituent feed.",
            "scan.py requires Ticker + Sector and uses Company/Country when present; extra columns are for audit/debug.",
        ],
    }

    if meta_path is not None:
        write_metadata(meta_path, stats)

    print(f"[msci-refresh] source fund: {fetched.fund}")
    print(f"[msci-refresh] source as-of: {source_as_of or 'n/a'}")
    print(f"[msci-refresh] rows written: {row_count}")
    print(f"[msci-refresh] unique tickers: {len(new_tickers)}")
    print(f"[msci-refresh] added: {len(added)} | removed: {len(removed)}")
    if added:
        print(f"[msci-refresh] added sample: {', '.join(added[:10])}")
    if removed:
        print(f"[msci-refresh] removed sample: {', '.join(removed[:10])}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:  # noqa: BLE001
        print(f"[msci-refresh] ERROR: {e}", file=sys.stderr)
        raise
