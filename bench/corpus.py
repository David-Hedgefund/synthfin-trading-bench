"""Corpus loader and point-in-time views.

The loader is format-tolerant: it reads the ``index.json`` navigation file produced by the
SynthFin export, and falls back to directory scanning when fields are missing. It is agnostic
to universe size — the same code loads the 50x10 validation corpus and a 1000x10 corpus.

Point-in-time discipline is the whole game here:
  * A price bar for date D is knowable at the close of D.
  * A document is knowable from its ``date`` field onward.
  * A fundamentals record (earnings) is knowable from its ``filing_date`` onward.
`ScenarioData` exposes helpers that never return anything dated after the decision day.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

from .schema import Doc, Fundamentals, PriceBar


# --------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------


@dataclass
class ScenarioMeta:
    id: str
    seed: Optional[int]
    name: str
    slug: str
    start_date: str
    end_date: str
    structured_path: str
    unstructured_path: str
    tickers: list[str]


def _to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class Corpus:
    """Top-level handle on a corpus directory in the standard SynthFin export layout."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"corpus root does not exist: {self.root}")
        index_path = self.root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"no index.json under {self.root}; is this a SynthFin corpus export?"
            )
        self.index = json.loads(index_path.read_text())
        self.tickers: list[str] = list(self.index.get("tickers", []))
        self.profile: str = self.index.get("profile", "unknown")

    # -- scenario enumeration ---------------------------------------------------------

    def scenarios(self) -> list[ScenarioMeta]:
        out: list[ScenarioMeta] = []
        for s in self.index.get("scenarios", []):
            out.append(
                ScenarioMeta(
                    id=s["id"],
                    seed=s.get("seed"),
                    name=s.get("name", s.get("slug", s["id"])),
                    slug=s.get("slug", s["id"]),
                    start_date=s.get("start_date", ""),
                    end_date=s.get("end_date", ""),
                    structured_path=s.get("structured_path", f"structured/scenarios/{s.get('slug')}"),
                    unstructured_path=s.get(
                        "unstructured_path", f"unstructured/scenarios/{s.get('slug')}"
                    ),
                    tickers=self.tickers,
                )
            )
        return out

    def load(self, meta: ScenarioMeta) -> "ScenarioData":
        return ScenarioData(self.root, meta)

    # -- reproducibility --------------------------------------------------------------

    def content_hash(self, limit_bytes: int = 0) -> str:
        """Stable SHA-256 over structured price files + doc dates.

        This is the value you cite as the frozen dataset version. It hashes file paths and
        contents of everything that can change a benchmark result. ``limit_bytes`` optionally
        truncates each file's contribution for a faster (still order-sensitive) fingerprint.
        """
        h = hashlib.sha256()
        files = sorted(
            p for p in self.root.rglob("*.json") if ".bak" not in p.name and p.name != ".DS_Store"
        )
        for p in files:
            rel = p.relative_to(self.root).as_posix()
            h.update(rel.encode())
            data = p.read_bytes()
            h.update(data if limit_bytes <= 0 else data[:limit_bytes])
        return h.hexdigest()


# --------------------------------------------------------------------------------------
# A single scenario
# --------------------------------------------------------------------------------------


class ScenarioData:
    def __init__(self, root: Path, meta: ScenarioMeta):
        self.root = root
        self.meta = meta
        self.structured = root / meta.structured_path
        self.unstructured = root / meta.unstructured_path
        self._prices: dict[str, dict[str, PriceBar]] = {}
        self._trading_days: list[str] = []
        self._doc_index: list[Doc] = []
        self._doc_index_built = False
        self._macro: list[dict[str, Any]] = []
        # Effective universe: the index's global ticker list, or — when that is empty (corpora
        # built from a ticker_limit profile don't populate it) — the ticker folders on disk.
        self._universe = list(meta.tickers) or self._discover_tickers()
        self._load_prices()
        self._load_macro()
        # Doc index is built lazily on first docs_between() call — indexing hundreds of
        # thousands of docs is wasted for agents (e.g. baselines) that never read them.

    # -- prices -----------------------------------------------------------------------

    def _discover_tickers(self) -> list[str]:
        tickers_dir = self.structured / "tickers"
        if not tickers_dir.exists():
            return []
        return sorted(p.name for p in tickers_dir.iterdir() if p.is_dir())

    def _load_prices(self) -> None:
        tickers_dir = self.structured / "tickers"
        all_days: set[str] = set()
        for tk in self._universe:
            pf = tickers_dir / tk / "prices.json"
            if not pf.exists():
                continue
            bars: dict[str, PriceBar] = {}
            for row in json.loads(pf.read_text()):
                d = row["time"]
                close = _to_float(row.get("close")) or 0.0
                adj = _to_float(row.get("adjusted_close"))
                meta = row.get("bar_metadata") or {}
                comps = {
                    k: (_to_float(v) or 0.0)
                    for k, v in (meta.get("return_components") or {}).items()
                }
                bars[d] = PriceBar(
                    date=d,
                    open=_to_float(row.get("open")) or close,
                    high=_to_float(row.get("high")) or close,
                    low=_to_float(row.get("low")) or close,
                    close=close,
                    adj_close=adj if adj is not None else close,
                    volume=_to_float(row.get("volume")) or 0.0,
                    market_return=_to_float(row.get("market_return")) or 0.0,
                    sector_return=_to_float(row.get("sector_return")) or 0.0,
                    idiosyncratic_return=_to_float(row.get("idiosyncratic_return")) or 0.0,
                    event_return=_to_float(row.get("event_return")) or 0.0,
                    components=comps,
                )
                all_days.add(d)
            self._prices[tk] = bars
        self._trading_days = sorted(all_days)

    @property
    def trading_days(self) -> list[str]:
        return self._trading_days

    @property
    def priced_tickers(self) -> list[str]:
        return [t for t in self._universe if t in self._prices]

    def bar(self, ticker: str, date: str) -> Optional[PriceBar]:
        return self._prices.get(ticker, {}).get(date)

    def last_bar_on_or_before(self, ticker: str, date: str) -> Optional[PriceBar]:
        bars = self._prices.get(ticker)
        if not bars:
            return None
        best: Optional[PriceBar] = None
        for d, b in bars.items():
            if d <= date and (best is None or d > best.date):
                best = b
        return best

    def next_trading_day(self, date: str) -> Optional[str]:
        for d in self._trading_days:
            if d > date:
                return d
        return None

    def sector_of(self, ticker: str) -> str:
        cf = self.structured / "tickers" / ticker / "company.json"
        if cf.exists():
            try:
                c = json.loads(cf.read_text())
                c = c[0] if isinstance(c, list) and c else c
                return c.get("sector") or c.get("gics_sector") or "Unknown"
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
        return "Unknown"

    # -- macro ------------------------------------------------------------------------

    def _load_macro(self) -> None:
        mf = self.structured / "macro.json"
        if mf.exists():
            data = json.loads(mf.read_text())
            self._macro = data if isinstance(data, list) else [data]

    def macro_as_of(self, date: str) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        for row in self._macro:
            d = row.get("time") or row.get("date")
            if d and d <= date:
                latest = row
        return latest

    # -- documents (point-in-time) ----------------------------------------------------

    def _build_doc_index(self) -> None:
        """Index every unstructured doc by date, without loading body text.

        Cached to ``.ptindex.json`` inside the scenario's unstructured dir so repeated runs
        over the same corpus (esp. 1000-ticker) don't rescan tens of thousands of files.
        """
        if self._doc_index_built:
            return
        self._doc_index_built = True
        cache = self.unstructured / ".ptindex.json"
        if cache.exists():
            try:
                rows = json.loads(cache.read_text())
                self._doc_index = [Doc(**r) for r in rows]
                return
            except (json.JSONDecodeError, TypeError):
                pass
        docs: list[Doc] = []
        if self.unstructured.exists():
            for jf in self.unstructured.rglob("*.json"):
                if jf.name.startswith(".") or ".bak" in str(jf):
                    continue
                try:
                    d = json.loads(jf.read_text())
                except json.JSONDecodeError:
                    continue
                if not isinstance(d, dict) or "date" not in d:
                    continue
                docs.append(
                    Doc(
                        date=d.get("date", ""),
                        ticker=d.get("ticker", "MARKET"),
                        doc_type=d.get("doc_type", "news"),
                        title=d.get("title", ""),
                        period=d.get("period", ""),
                        path=str(jf),
                        words=int(d.get("words", 0) or 0),
                    )
                )
        docs.sort(key=lambda x: x.date)
        self._doc_index = docs
        try:
            cache.write_text(
                json.dumps([{k: v for k, v in vars(x).items() if k != "_text"} for x in docs])
            )
        except OSError:
            pass  # read-only corpus mount is fine; we just skip caching

    def docs_between(
        self,
        start_exclusive: str,
        end_inclusive: str,
        tickers: Optional[Iterable[str]] = None,
        doc_types: Optional[Iterable[str]] = None,
    ) -> list[Doc]:
        self._build_doc_index()  # lazy: builds on first use, then cached
        tset = set(tickers) if tickers is not None else None
        dset = set(doc_types) if doc_types is not None else None
        out = []
        for doc in self._doc_index:
            if not (start_exclusive < doc.date <= end_inclusive):
                continue
            if tset is not None and doc.ticker not in tset and doc.ticker != "MARKET":
                continue
            if dset is not None and doc.doc_type not in dset:
                continue
            out.append(doc)
        return out

    @staticmethod
    def doc_text(doc: Doc, max_chars: int = 2000) -> str:
        if doc._text is None:
            try:
                d = json.loads(Path(doc.path).read_text())
                doc._text = d.get("text", "")
            except (json.JSONDecodeError, OSError):
                doc._text = ""
        return doc._text[:max_chars]

    # -- fundamentals (point-in-time) -------------------------------------------------

    @lru_cache(maxsize=4096)
    def _load_json(self, path_str: str) -> Any:
        p = Path(path_str)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def fundamentals_as_of(self, ticker: str, date: str) -> Fundamentals:
        base = self.structured / "tickers" / ticker
        company = self._load_json(str(base / "company.json")) or {}
        if isinstance(company, list):
            company = company[0] if company else {}

        earnings = self._load_json(str(base / "earnings.json")) or []
        latest_e: dict[str, Any] = {}
        for e in earnings if isinstance(earnings, list) else []:
            fd = e.get("filing_date") or e.get("report_period")
            if fd and fd <= date and (not latest_e or fd >= latest_e.get("_fd", "")):
                latest_e = {**e, "_fd": fd}
        latest_e.pop("_fd", None)

        # NOTE: metrics_ttm.json is a single end-of-scenario snapshot with no availability date,
        # so using it would leak the future. We instead compute TTM from the last four
        # filing-date-gated 10-Qs (see _ttm_from_statements) — a proper point-in-time figure.
        ttm = self._ttm_from_statements(base, date)

        guidance = self._load_json(str(base / "guidance.json")) or []
        latest_g: dict[str, Any] = {}
        for g in guidance if isinstance(guidance, list) else [guidance]:
            d = g.get("issued_date") or g.get("date") or ""
            if d <= date and (not latest_g or d >= latest_g.get("_d", "")):
                latest_g = {**g, "_d": d}
        latest_g.pop("_d", None)

        statements = self._latest_statement(base, date)
        analyst = self._analyst_consensus(base, date)
        institutional = self._latest_dated(base / "institutional_holdings.json", date)
        insider = self._recent_insider(base, date)

        return Fundamentals(
            ticker=ticker,
            as_of=date,
            company=_compact(company, ("sector", "industry", "name", "market_cap")),
            latest_earnings=_compact(
                latest_e,
                ("fiscal_period", "filing_date", "quarterly"),
            ),
            ttm=ttm,
            guidance=_compact(latest_g, ("metric", "low", "high", "period")),
            statements=statements,
            analyst=analyst,
            institutional=institutional,
            insider=insider,
        )

    def _latest_statement(self, base: Path, date: str) -> dict[str, Any]:
        """Key income-statement + balance-sheet lines from the latest 10-Q filed on/before date."""
        rows = self._load_json(str(base / "statements_quarterly.json")) or []
        latest, best = {}, ""
        for r in rows if isinstance(rows, list) else []:
            fd = r.get("filing_date") or r.get("report_period") or ""
            if fd and fd <= date and fd >= best:
                latest, best = r, fd
        if not latest:
            return {}
        inc = latest.get("income_statement", {})
        bal = latest.get("balance_sheet", {})
        return {
            "fiscal_period": latest.get("fiscal_period"),
            "report_period": latest.get("report_period"),
            "revenue": inc.get("revenue"),
            "revenue_growth": inc.get("revenue_growth"),
            "gross_margin": inc.get("gross_margin"),
            "operating_margin": inc.get("operating_margin"),
            "net_income": inc.get("net_income"),
            "eps_diluted": inc.get("eps_diluted"),
            "total_debt": bal.get("long_term_debt"),
            "cash": bal.get("cash_and_equivalents"),
        }

    def _ttm_from_statements(self, base: Path, date: str) -> dict[str, Any]:
        """Trailing-12-month figures from the last four 10-Qs filed on/before date.

        Point-in-time by construction (each quarter is gated by its filing_date). ``pe_ttm`` is
        left to the observation layer, which has the current price; here we expose TTM EPS/revenue
        and the latest margins so P/E can be computed as price / ttm_eps without lookahead.
        """
        rows = self._load_json(str(base / "statements_quarterly.json")) or []
        avail = []
        for r in rows if isinstance(rows, list) else []:
            fd = r.get("filing_date") or r.get("report_period") or ""
            if fd and fd <= date:
                avail.append((fd, r))
        if not avail:
            return {}
        avail.sort(key=lambda x: x[0], reverse=True)
        last4 = [r for _, r in avail[:4]]
        ttm_rev = sum((_to_float(r.get("income_statement", {}).get("revenue")) or 0.0) for r in last4)
        ttm_ni = sum(
            (_to_float(r.get("income_statement", {}).get("net_income")) or 0.0) for r in last4
        )
        ttm_eps = sum(
            (_to_float(r.get("income_statement", {}).get("eps_diluted")) or 0.0) for r in last4
        )
        latest_inc = last4[0].get("income_statement", {})
        return {
            "n_quarters": len(last4),
            "ttm_revenue": ttm_rev or None,
            "ttm_net_income": ttm_ni or None,
            "ttm_eps": round(ttm_eps, 4) if ttm_eps else None,
            "gross_margin": latest_inc.get("gross_margin"),
            "net_margin": (ttm_ni / ttm_rev) if ttm_rev else None,
        }

    def _analyst_consensus(self, base: Path, date: str) -> dict[str, Any]:
        """Consensus from the latest per-firm estimate on/before date: rating mix, mean target,
        and the most recent target revision."""
        rows = self._load_json(str(base / "analyst_estimates.json")) or []
        by_firm: dict[str, dict[str, Any]] = {}
        for r in rows if isinstance(rows, list) else []:
            ed = r.get("estimate_date") or ""
            if not ed or ed > date:
                continue
            firm = r.get("firm", r.get("analyst", "?"))
            if firm not in by_firm or ed >= by_firm[firm].get("estimate_date", ""):
                by_firm[firm] = r
        if not by_firm:
            return {}
        targets = [_to_float(r.get("target_price")) for r in by_firm.values()]
        targets = [t for t in targets if t is not None]
        ratings: dict[str, int] = {}
        for r in by_firm.values():
            rt = str(r.get("rating", "")).lower()
            if rt:
                ratings[rt] = ratings.get(rt, 0) + 1
        most_recent = max(by_firm.values(), key=lambda r: r.get("estimate_date", ""))
        return {
            "n_firms": len(by_firm),
            "mean_target_price": round(sum(targets) / len(targets), 2) if targets else None,
            "rating_counts": ratings,
            "latest_revision_pct": _to_float(most_recent.get("revision_pct")),
            "latest_estimate_date": most_recent.get("estimate_date"),
        }

    def _latest_dated(self, path: Path, date: str) -> dict[str, Any]:
        """Latest single record on/before date from a dated JSON list (e.g. 13F panels)."""
        rows = self._load_json(str(path)) or []
        latest, best = {}, ""
        for r in rows if isinstance(rows, list) else []:
            d = r.get("report_date") or r.get("period") or r.get("date") or r.get("as_of") or ""
            if d and d <= date and d >= best:
                latest, best = r, d
        return latest

    def _recent_insider(self, base: Path, date: str, window_days: int = 90) -> dict[str, Any]:
        """Net insider buy/sell summary over the trailing window (Form-4-style trades)."""
        rows = self._load_json(str(base / "insider_trades.json")) or []
        if not isinstance(rows, list) or not rows:
            return {}
        buys = sells = n = 0
        for r in rows:
            td = r.get("transaction_date") or r.get("date") or ""
            if not td or td > date:
                continue
            n += 1
            val = _to_float(r.get("value")) or _to_float(r.get("shares")) or 0.0
            code = str(r.get("transaction_type") or r.get("code") or "").lower()
            if "buy" in code or code == "p":
                buys += val
            elif "sell" in code or code == "s":
                sells += val
        if n == 0:
            return {}
        return {"n_trades": n, "buy_value": round(buys, 0), "sell_value": round(sells, 0)}


def _compact(d: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    out = {k: d[k] for k in keys if k in d}
    return out or d  # if none of the preferred keys are present, keep the raw record
