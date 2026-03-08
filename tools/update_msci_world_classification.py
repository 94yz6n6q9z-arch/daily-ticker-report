#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refresh MSCI World + EM classification CSVs from public iShares ETF holdings
and normalize sector labels to the S&P 500 / GICS 11 sectors used by scan.py.

Why this exists
---------------
The exact MSCI World + EM constituent lists are typically distributed through
licensed data feeds.  For an automated public-source workflow, we use broad
iShares ETF holdings files as operational proxies:
  - MSCI World:            iShares EUNL / IWDA / URTH  (~1,200-1,400 positions)
  - MSCI Emerging Markets: iShares EIMI / EEM           (~800-1,300 positions)

Outputs
-------
- CSV (World):  config/msci_world_classification.csv
- CSV (EM):     config/msci_em_classification.csv
- Meta (World): docs/msci_world_classification_meta.json
- Meta (EM):    docs/msci_em_classification_meta.json

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

# ────────────────────────────────────────────────────────────────
# Version tracker (mirrors scan.py / gc_engine.py pattern)
# ────────────────────────────────────────────────────────────────
MSCI_UPDATE_VERSION = "1.4.0"

_MSCI_UPDATE_VERSION_LOG: dict = {
    "1.0.0": (
        "Initial release. Refreshes config/msci_world_classification.csv from iShares "
        "MSCI World ETF (EUNL / IWDA / URTH). Normalises sector labels to S&P 500 11-sector "
        "taxonomy. Deduplicates, validates, and writes metadata JSON."
    ),
    "1.1.0": (
        "EXCHANGE_SUFFIX_RULES reordered so Euronext/Nordic rules fire before the US "
        "catch-all, fixing 114 French/Swedish/Belgian/Finnish/Portuguese tickers. "
        "NORDIC regex extended to match Stockholm (NASDAQ OMX NORDIC → .ST)."
    ),
    "1.2.0": (
        "MSCI EM universe added: SOURCE_CANDIDATES_EM (EIMI + EEM), --universe world|em|both "
        "CLI flag, _run_one_universe() helper. 30+ new EM exchange suffix rules "
        "(.KS .TW .NS .BO .SA .JO .MX .SS .SZ .JK .BK .KL .SR .IS .WA .AD .DU .AT .CA "
        ".PS .QA .PR .BD .SN .CL .LM .KA). KNOWN_TICKER_OVERRIDES applied post-guessing. "
        "GHOST_COUNTRIES set drops Kuwait/Russia. COUNTRY_SUFFIX_FALLBACK covers "
        "Brazil/Greece/Chile/etc. when Exchange column is empty. multi-dot ticker filter. "
        "min-rows-em CLI arg (default 700)."
    ),
    "1.3.0": (
        "v92 sync: module docstring updated to reflect 'MSCI World + EM' scope. "
        "MSCI_UPDATE_VERSION constant + _MSCI_UPDATE_VERSION_LOG added so this file "
        "tracks changes identically to scan.py and gc_engine.py. CLI description updated. "
        "No logic changes."
    ),
    "1.4.0": (
        "Fix: UCITS iShares exports (IWDA, EIMI) truncate long sector names — 'Communication' "
        "instead of 'Communication Services', etc. This was silently dropping 56 World + 65 EM "
        "real constituents (121 total, including Alphabet, Meta, Netflix). "
        "Added full set of truncated/abbreviated variants to SECTOR_MAP: communication, "
        "consumer disc/discr/cons discr, consumer stap/cons staples, hlth care/health, "
        "info technology/info tech/it/technology, real est, financial, industrial, "
        "material, utility. Recovers all 121 previously lost constituents."
    ),
}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

# Try UCITS + US listings; first success wins.

# These endpoints are the same style as the ones exposed by iShares pages' "download holdings" links.
SOURCE_CANDIDATES_WORLD = [
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

# MSCI Emerging Markets ETF sources (EIMI = UCITS, EEM = US-listed)
SOURCE_CANDIDATES_EM = [
    {
        "fund": "EIMI",
        "url": "https://www.ishares.com/uk/individual/en/products/264659/ishares-core-msci-emerging-markets-imi-ucits-etf-acc-fund/1506575576011.ajax?dataType=fund&fileName=EIMI_holdings&fileType=csv",
        "referer": "https://www.ishares.com/uk/individual/en/products/264659/ishares-core-msci-emerging-markets-imi-ucits-etf-acc-fund",
    },
    {
        "fund": "EEM",
        "url": "https://www.ishares.com/us/products/239626/ishares-msci-emerging-markets-etf/1467271812596.ajax?dataType=fund&fileName=EEM_holdings&fileType=csv",
        "referer": "https://www.ishares.com/us/products/239626/ishares-msci-emerging-markets-etf",
    },
]

# Backward-compat alias
SOURCE_CANDIDATES = SOURCE_CANDIDATES_WORLD

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
    # ── Canonical English (full names) ──────────────────────────────────────
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
    # ── Truncated / abbreviated labels from UCITS iShares exports ───────────
    # These appear when iShares UK/EU CSVs shorten the longer sector names.
    # v1.4.0: added after losing 121 constituents (56 World + 65 EM) in prod.
    "communication": "Communication Services",       # truncated "Communication Services"
    "consumer disc": "Consumer Discretionary",       # truncated
    "consumer discr": "Consumer Discretionary",
    "cons discr": "Consumer Discretionary",
    "consumer stap": "Consumer Staples",             # truncated
    "cons staples": "Consumer Staples",
    "hlth care": "Health Care",
    "health": "Health Care",
    "info technology": "Information Technology",     # truncated
    "info tech": "Information Technology",
    "it": "Information Technology",
    "technology": "Information Technology",          # common alias in some ETF exports
    "real est": "Real Estate",
    "financial": "Financials",
    "industrial": "Industrials",
    "material": "Materials",
    "utility": "Utilities",
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
    # ── Specific Euronext / Nasdaq OMX exchanges MUST come before the US catch-all ──
    # (The US rule matches "NYSE" inside "Nyse Euronext" and "NASDAQ" inside "Nasdaq Omx"
    #  if evaluated first -- so specific European rules are listed first.)
    (re.compile(r"EURONEXT\s+PARIS", re.I), ".PA", "high"),
    (re.compile(r"EURONEXT\s+AMSTERDAM", re.I), ".AS", "high"),
    (re.compile(r"EURONEXT\s+BRUSSELS", re.I), ".BR", "high"),
    (re.compile(r"EURONEXT\s+MILAN|BORSA\s+ITALIANA|MILAN", re.I), ".MI", "high"),
    (re.compile(r"EURONEXT\s+DUBLIN|IRISH\s+STOCK\s+EXCHANGE|DUBLIN", re.I), ".IR", "low"),
    (re.compile(r"EURONEXT\s+LISBON|LISBON", re.I), ".LS", "med"),
    (re.compile(r"NASDAQ\s+OMX\s+COPENHAGEN|COPENHAGEN", re.I), ".CO", "med"),
    (re.compile(r"NASDAQ\s+OMX\s+STOCKHOLM|NASDAQ\s+OMX\s+NORDIC|STOCKHOLM", re.I), ".ST", "med"),
    (re.compile(r"NASDAQ\s+OMX\s+HELSINKI|HELSINKI", re.I), ".HE", "med"),
    # ── US exchanges (only match pure US, after Euronext/OMX rules) ──
    (re.compile(r"NASDAQ|NEW\s+YORK\s+STOCK\s+EXCHANGE|NYSE|CBOE|BATS|ARCA", re.I), "", "high"),
    # ── Developed market exchanges ──
    (re.compile(r"TORONTO\s+STOCK\s+EXCHANGE|TSX", re.I), ".TO", "high"),
    (re.compile(r"LONDON\s+STOCK\s+EXCHANGE|LSE", re.I), ".L", "high"),
    (re.compile(r"XETRA|DEUTSCHE\s+BOERSE|FRANKFURT", re.I), ".DE", "high"),
    (re.compile(r"SIX\s+SWISS|SWISS\s+EXCHANGE|SIX", re.I), ".SW", "med"),
    (re.compile(r"MADRID|BME", re.I), ".MC", "high"),
    (re.compile(r"TOKYO\s+STOCK\s+EXCHANGE|TSE\b|JPX", re.I), ".T", "high"),
    (re.compile(r"HONG\s*KONG|HKEX", re.I), ".HK", "high"),
    (re.compile(r"ASX\s+-\s+ALL\s+MARKETS|AUSTRALIAN\s+SECURITIES\s+EXCHANGE|ASX", re.I), ".AX", "high"),
    (re.compile(r"SGX|SINGAPORE\s+EXCHANGE", re.I), ".SI", "high"),
    (re.compile(r"OSLO\s+STOCK\s+EXCHANGE|OSLO\s+BORS|EURONEXT\s+OSLO", re.I), ".OL", "high"),
    (re.compile(r"VIENNA\s+STOCK\s+EXCHANGE|WIENER\s+BOERSE", re.I), ".VI", "med"),
    (re.compile(r"TEL\s+AVIV\s+STOCK\s+EXCHANGE|TASE", re.I), ".TA", "high"),
    (re.compile(r"NEW\s+ZEALAND\s+STOCK\s+EXCHANGE|NZX", re.I), ".NZ", "med"),
    # ── Emerging market exchanges ──
    # Korea
    (re.compile(r"KOREA\s+EXCHANGE|KRX|KOREA\s+STOCK\s+EXCHANGE|KSE", re.I), ".KS", "high"),
    # Taiwan
    (re.compile(r"TAIWAN\s+STOCK\s+EXCHANGE|TWSE|TAIPEI\s+EXCHANGE|TPEx", re.I), ".TW", "high"),
    # India
    (re.compile(r"NATIONAL\s+STOCK\s+EXCHANGE.*INDIA|NSE\s+INDIA|NSE$", re.I), ".NS", "high"),
    (re.compile(r"BOMBAY\s+STOCK\s+EXCHANGE|BSE\s+INDIA|BSE$", re.I), ".BO", "med"),
    # Brazil
    (re.compile(r"B3|BOLSA\s+BRASIL|BM&F|BOVESPA", re.I), ".SA", "high"),
    # South Africa
    (re.compile(r"JOHANNESBURG\s+STOCK\s+EXCHANGE|JSE", re.I), ".JO", "high"),
    # Mexico
    (re.compile(r"BOLSA\s+MEXICANA|BMV|MEXICO\s+STOCK\s+EXCHANGE", re.I), ".MX", "high"),
    # China (Shanghai/Shenzhen — Yahoo uses .SS and .SZ)
    (re.compile(r"SHANGHAI\s+STOCK\s+EXCHANGE|SSE\b", re.I), ".SS", "high"),
    (re.compile(r"SHENZHEN\s+STOCK\s+EXCHANGE|SZSE", re.I), ".SZ", "high"),
    # Indonesia
    (re.compile(r"INDONESIA\s+STOCK\s+EXCHANGE|IDX|BURSA\s+EFEK\s+INDONESIA", re.I), ".JK", "high"),
    # Thailand
    (re.compile(r"STOCK\s+EXCHANGE\s+OF\s+THAILAND|SET\b|THAILAND", re.I), ".BK", "high"),
    # Malaysia
    (re.compile(r"BURSA\s+MALAYSIA|KUALA\s+LUMPUR\s+STOCK\s+EXCHANGE|KLSE", re.I), ".KL", "high"),
    # Saudi Arabia
    (re.compile(r"TADAWUL|SAUDI\s+EXCHANGE|SAUDI\s+STOCK", re.I), ".SR", "high"),
    # UAE
    (re.compile(r"ABU\s+DHABI\s+SECURITIES|ADX\b", re.I), ".AD", "med"),
    (re.compile(r"DUBAI\s+FINANCIAL\s+MARKET|DFM\b", re.I), ".DU", "med"),
    # Turkey
    (re.compile(r"BORSA\s+ISTANBUL|ISTANBUL\s+STOCK\s+EXCHANGE|BIST", re.I), ".IS", "high"),
    # Poland
    (re.compile(r"WARSAW\s+STOCK\s+EXCHANGE|GPW\b|WSE\b", re.I), ".WA", "high"),
    # Greece
    (re.compile(r"ATHENS\s+STOCK\s+EXCHANGE|ATHEX", re.I), ".AT", "med"),
    # Egypt
    (re.compile(r"EGYPTIAN\s+EXCHANGE|EGX\b|CAIRO\s+STOCK", re.I), ".CA", "med"),
    # Philippines
    (re.compile(r"PHILIPPINE\s+STOCK\s+EXCHANGE|PSE\b", re.I), ".PS", "med"),
    # Qatar
    (re.compile(r"QATAR\s+STOCK\s+EXCHANGE|QSE\b", re.I), ".QA", "med"),
    # Czech Republic
    (re.compile(r"PRAGUE\s+STOCK\s+EXCHANGE|PSE.*PRAGUE|BURZA.*PRAHA", re.I), ".PR", "med"),
    # Hungary
    (re.compile(r"BUDAPEST\s+STOCK\s+EXCHANGE|BSE.*BUDAPEST", re.I), ".BD", "med"),
    # Chile
    (re.compile(r"BOLSA\s+DE\s+COMERCIO\s+DE\s+SANTIAGO|BCS\b|SANTIAGO\s+EXCHANGE", re.I), ".SN", "med"),
    # Colombia
    (re.compile(r"BOLSA\s+DE\s+VALORES\s+DE\s+COLOMBIA|BVC\b", re.I), ".CL", "med"),
    # Peru
    (re.compile(r"BOLSA\s+DE\s+VALORES\s+DE\s+LIMA|BVL\b", re.I), ".LM", "low"),
    # Pakistan
    (re.compile(r"PAKISTAN\s+STOCK\s+EXCHANGE|PSX\b|KARACHI\s+STOCK", re.I), ".KA", "low"),
]

# ────────────────────────────────────────────────────────────────
# Hard-coded ticker corrections applied AFTER exchange-suffix guessing.
# Covers cases where raw_ticker + exchange logic produces the wrong Yahoo symbol.
# Key = generated (wrong) ticker, Value = correct Yahoo Finance symbol.
# ────────────────────────────────────────────────────────────────
KNOWN_TICKER_OVERRIDES: Dict[str, str] = {
    # ── Bloomberg placeholders → correct Yahoo symbol ──────────────
    "2299955D.TO": "CSU.TO",    # Constellation Software
    # ── Formatting quirks in iShares CSV exports ──────────────────
    "NDAFI.HE":    "NDA-FI.HE", # Nordea Bank Helsinki — space in raw symbol
    "HEIA":        "HEI-A",     # HEICO Corp Class A (US)
    "BMW3.DE":     "BMWG.DE",   # BMW preference shares
    "BRKB":        "BRK-B",     # Berkshire Hathaway B
    "CICT.SI":     "C38U.SI",   # CapitaLand Integrated Commercial Trust
    "STLAM.MI":    "STLAM.MI",  # Stellantis
    "CSG.AS":      "CS.AS",     # Credit Suisse / UBS post-merger
    # ── HK dual-listings: MSCI uses HK primary, raw ticker may be US symbol ──
    "AIA":         "1299.HK",   # AIA Group — numeric HK code
    "BABA":        "9988.HK",   # Alibaba — HK primary for MSCI
    "JD":          "9618.HK",   # JD.com — HK primary
    "NTES":        "9999.HK",   # NetEase — HK primary
    # ── BSE India: Bloomberg uses .R suffix, Yahoo uses .BO ───────
    "WIPRO.R":     "WIPRO.BO",
    "INFY.R":      "INFY.BO",
    "TCS.R":       "TCS.BO",
    # ── Individual country-code mismatches ────────────────────────
    # These appear when Exchange column is empty so suffix-guessing fires
    # country-fallback but the raw symbol already has a wrong suffix.
    "ADMIE":       "ADMIE.AT",  # Greece — MSCI exports without .AT
    "ALPHA":       "ALPHA.AT",  # Greece — MSCI exports without .AT
    "AGUAS-A":     "AGUAS-A.SN", # Chile — MSCI exports without .SN
}

# ────────────────────────────────────────────────────────────────
# Countries whose stocks are not supported on Yahoo Finance.
# Tickers from these countries are dropped from the CSV so no
# downstream consumer (gc_engine.py, scan.py etc.) wastes a
# fetch slot on a symbol that will never resolve.
# ────────────────────────────────────────────────────────────────
GHOST_COUNTRIES: set = {
    "Kuwait",     # Boursa Kuwait — not available on Yahoo Finance
    "Russia",     # Removed from MSCI universe March 2022 (sanctions)
}

# ────────────────────────────────────────────────────────────────
# Country → Yahoo Finance exchange suffix fallback.
# Used when the Exchange column is empty or unrecognised so
# EXCHANGE_SUFFIX_RULES can't fire. Covers the "226 US ghost"
# problem where Brazilian, Greek, GCC etc. stocks appear without
# any exchange suffix in iShares EM exports.
# Only countries we're 100% confident about are listed here.
# ────────────────────────────────────────────────────────────────
COUNTRY_SUFFIX_FALLBACK: Dict[str, str] = {
    "Brazil":          ".SA",
    "Greece":          ".AT",
    "Chile":           ".SN",
    "Czech Republic":  ".PR",
    "Hungary":         ".BD",
    "Egypt":           ".CA",
    "Pakistan":        ".KA",
    "Colombia":        ".CL",
    "Peru":            ".LM",
}


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


def fetch_holdings_csv(timeout: int = 45, sources: Optional[List[Dict]] = None) -> FetchResult:
    source_list = sources if sources is not None else SOURCE_CANDIDATES_WORLD
    last_err: Optional[Exception] = None
    for src in source_list:
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
    # Nordic share classes: "VOLV B", "NOVO B", "SEB A" → "VOLV-B", "NOVO-B", "SEB-A"
    # Pattern: letters/digits, one space, single letter at end.
    # Must convert BEFORE removing spaces so yfinance gets VOLV-B.ST not VOLVB.ST.
    s = re.sub(r"^([A-Z0-9]+)\s+([A-Z])$", r"\1-\2", s)
    s = s.replace(" ", "")
    s = s.replace("/", "-")
    # Common US share class notation to Yahoo style (BRK.B -> BRK-B) when no exchange suffix yet.
    if re.fullmatch(r"[A-Z]{1,5}\.[A-Z]", s):
        s = s.replace(".", "-")
    # Strip trailing dots (LSE raw symbols like "JD.", "BP." produce double-dots when suffix appended).
    s = s.rstrip(".")
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


_GHOST_RAW_PATTERN = re.compile(
    r"^-$"               # bare dash
    r"|^\d+D?$"          # pure numeric optionally ending in D (Bloomberg placeholders)
    r"|^[A-Z]{1,5}\d{6,}D$"  # alpha prefix + long numeric + D  e.g. 2299955D
    r"|^\.$"             # bare dot
)


def _is_ghost_raw_ticker(raw: str) -> bool:
    """Return True for Bloomberg placeholder symbols that will never resolve on Yahoo Finance."""
    t = str(raw).strip()
    if not t:
        return True
    if "." not in t and t.replace("-", "").isdigit():
        return True
    return bool(_GHOST_RAW_PATTERN.match(t))


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

    # Drop ghost countries (Yahoo Finance doesn't carry these exchanges)
    if "Country" in d.columns:
        ghost_mask = d["Country"].isin(GHOST_COUNTRIES)
        n = ghost_mask.sum()
        if n:
            print(f"[msci-refresh] dropping {n} rows from unsupported countries: {sorted(GHOST_COUNTRIES)}")
        d = d[~ghost_mask]

    # Drop Bloomberg placeholder raw tickers (D-codes, pure numerics etc.)
    ghost_ticker_mask = d["RawTicker"].map(_is_ghost_raw_ticker)
    n_ghost = ghost_ticker_mask.sum()
    if n_ghost:
        print(f"[msci-refresh] dropping {n_ghost} Bloomberg placeholder / ghost raw tickers")
    d = d[~ghost_ticker_mask]

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

    # ── Country-suffix fallback ───────────────────────────────────
    # When Exchange-based guessing returns a bare symbol with "low"
    # confidence (exchange column was empty or unrecognised), try the
    # Country column to assign the correct Yahoo suffix.
    # This fixes the "226 US ghost" problem: Brazilian, Greek, GCC etc.
    # stocks that appear as bare symbols in iShares EM exports.
    if "Country" in out.columns:
        for idx, row in out[out["MappingConfidence"] == "low"].iterrows():
            ticker = row["Ticker"]
            country = row["Country"]
            suffix = COUNTRY_SUFFIX_FALLBACK.get(country)
            if suffix and "." not in ticker:
                out.at[idx, "Ticker"] = ticker + suffix
                out.at[idx, "MappingConfidence"] = "med"

    # Clean pathological symbols that yfinance will reject often.
    out["Ticker"] = (
        out["Ticker"].astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(r"[^A-Z0-9\-\.=]", "", regex=True)
    )

    # Drop tickers with multiple dots — malformed symbols like BAJAJ.AUTO.NS
    # that come from MSCI EM CSV exports where the company name has a dot.
    multi_dot_mask = out["Ticker"].str.count(r"\.") > 1
    n_multi = multi_dot_mask.sum()
    if n_multi:
        print(f"[msci-refresh] dropping {n_multi} multi-dot malformed tickers "
              f"(e.g. {out.loc[multi_dot_mask, 'Ticker'].head(3).tolist()})")
    out = out[~multi_dot_mask].copy()

    # Apply known hard overrides after suffix-based guessing.
    # Corrects cases where raw_ticker + exchange logic produces the wrong Yahoo symbol.
    out["Ticker"] = out["Ticker"].replace(KNOWN_TICKER_OVERRIDES)

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


def _run_one_universe(
    label: str,
    sources: List[Dict],
    out_path: Path,
    meta_path: Optional[Path],
    min_rows: int,
    allow_partial: bool,
) -> int:
    """Fetch, parse, build and write one universe (World or EM). Returns row count."""
    prev_tickers = load_existing_ticker_set(out_path)
    fetched = fetch_holdings_csv(sources=sources)
    raw_df, source_as_of = parse_ishares_holdings(fetched.text)
    out_df = build_output_dataframe(raw_df, fetched.fund, fetched.url, source_as_of)

    row_count = len(out_df)
    if row_count < min_rows and not allow_partial:
        raise RuntimeError(
            f"[{label}] Refusing to write {out_path}: only {row_count} rows (< {min_rows}). "
            "This looks like a partial parse or source issue."
        )

    new_tickers = set(out_df["Ticker"].astype(str).tolist())
    added = sorted(new_tickers - prev_tickers)
    removed = sorted(prev_tickers - new_tickers)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    stats = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "universe": label,
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
            f"Public-source proxy (iShares {label} ETF holdings), not a licensed direct MSCI constituent feed.",
            "scan.py requires Ticker + Sector and uses Company/Country when present; extra columns are for audit/debug.",
        ],
    }

    if meta_path is not None:
        write_metadata(meta_path, stats)

    print(f"[msci-refresh:{label}] source fund: {fetched.fund}")
    print(f"[msci-refresh:{label}] source as-of: {source_as_of or 'n/a'}")
    print(f"[msci-refresh:{label}] rows written: {row_count}")
    print(f"[msci-refresh:{label}] unique tickers: {len(new_tickers)}")
    print(f"[msci-refresh:{label}] added: {len(added)} | removed: {len(removed)}")
    if added:
        print(f"[msci-refresh:{label}] added sample: {', '.join(added[:10])}")
    if removed:
        print(f"[msci-refresh:{label}] removed sample: {', '.join(removed[:10])}")

    return row_count


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh MSCI World + EM proxy constituent classification CSVs")
    ap.add_argument(
        "--universe", choices=["world", "em", "both"], default="both",
        help="Which universe to refresh: world=MSCI World only, em=MSCI EM only, both=both (default)"
    )
    ap.add_argument("--out", default="config/msci_world_classification.csv",
                    help="Output CSV path for MSCI World (default: config/msci_world_classification.csv)")
    ap.add_argument("--out-em", default="config/msci_em_classification.csv",
                    help="Output CSV path for MSCI EM (default: config/msci_em_classification.csv)")
    ap.add_argument("--meta", default="docs/msci_world_classification_meta.json",
                    help="Metadata JSON path for MSCI World")
    ap.add_argument("--meta-em", default="docs/msci_em_classification_meta.json",
                    help="Metadata JSON path for MSCI EM")
    ap.add_argument("--min-rows", type=int, default=900,
                    help="Fail-safe minimum row count for World (default: 900)")
    ap.add_argument("--min-rows-em", type=int, default=700,
                    help="Fail-safe minimum row count for EM (default: 700)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Allow writing even if row count < --min-rows")
    args = ap.parse_args()

    if args.universe in ("world", "both"):
        _run_one_universe(
            label="World",
            sources=SOURCE_CANDIDATES_WORLD,
            out_path=Path(args.out),
            meta_path=Path(args.meta) if args.meta else None,
            min_rows=args.min_rows,
            allow_partial=args.allow_partial,
        )

    if args.universe in ("em", "both"):
        _run_one_universe(
            label="EM",
            sources=SOURCE_CANDIDATES_EM,
            out_path=Path(args.out_em),
            meta_path=Path(args.meta_em) if args.meta_em else None,
            min_rows=args.min_rows_em,
            allow_partial=args.allow_partial,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:  # noqa: BLE001
        print(f"[msci-refresh] ERROR: {e}", file=sys.stderr)
        raise
