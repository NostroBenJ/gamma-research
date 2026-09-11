"""The base rate for the expected-move call, measured before any signal.

    python -m research.em_base_rate            # uses the cached pull
    python -m research.em_base_rate --refresh  # re-pull from DoltHub

WHY THIS EXISTS. `options/daily.py` grades a size call by comparing the
realised session move against one expected move. That threshold is only
meaningful next to its BASE RATE: if 70% of all sessions come in under one
expected move, then a "compression" call that is right 70% of the time has
found nothing at all. This is the same control that `FINDINGS.md` §2 built for
the intraday work -- the one that showed every exit structure was negative
before any signal was applied.

WHAT THIS IS NOT. It is not a gamma backtest, and it cannot become one.
The DoltHub `post-no-preference/options` database -- the only free daily option
history found -- carries date, symbol, expiry, strike, right, bid, ask, IV and
all five greeks, and **no open interest**. Neither does its `volatility_history`
table. GEX is gamma x OPEN INTEREST, so dealer positioning cannot be
reconstructed from it at any price. What can be measured is the expected-move
half of the read, which is what this does.

SOURCES. IV and prices both come from `Downloads/spy_vix_daily.csv`, the VRP
project's cached series: SPY closes and VIX, 2006-07-25 .. 2026-07-23. VIX is
SPX 30-day implied vol rather than SPY ATM implied vol -- a different index and
a variance-swap construction rather than a single strike -- so it is an
APPROXIMATION of the number `options/daily.py` actually uses. It is used anyway
because it is local, clean, twenty years long, and already trusted by
`vrp_data.py`, and because the alternative is not.

WHY NOT DOLTHUB, having gone and looked. `post-no-preference/options` is the
only free daily option history found, and its SQL API **silently returns
partial results**. Measured 2026-09-09: the identical query returned 28, 49,
55, 54 and 0 rows on five consecutive calls, each after exactly ~54.5 seconds
-- a fixed server-side time budget that returns whatever it managed to scan,
with no error, no flag and no cursor. `LIMIT 500 OFFSET 500` returned zero rows
for a table that has 1,265. An earlier run of this study got 1,000 rows and
produced a confident 75.4% from a sample whose completeness could not be
established. Non-deterministic silent truncation is the worst failure mode a
data source can have, and it is the same class as the Unusual Whales chain
truncation already recorded in this project. The `--dolthub` path is kept for
cross-checking a handful of dates and is NOT to be used for a bulk pull.

THE OVERLAP RULE DOES NOT BITE HERE, and it is worth saying why rather than
leaving it ambiguous: each observation is one session's realised move against
an IV known before that session. Consecutive observations share no data, so the
standard error is the ordinary one. That is different from the 21-day forward
windows in `vrp_study.py`, which do overlap and do need the correction.

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

DOLT = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
CACHE = Path("data/spy_iv_history.csv")
# Built by `python vrp_data.py` in the options-research repo.
PRICES = Path(os.environ.get("SPY_VIX_CSV", "../options-research/spy_vix_daily.csv"))
ONE_SESSION = 1.0 / 252.0
UA = "Mozilla/5.0 (compatible; cisd-bot research)"


def dolt_query(sql: str, timeout: float = 120.0) -> list[dict]:
    url = f"{DOLT}?{urllib.parse.urlencode({'q': sql})}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read()).get("rows", [])


#: DoltHub's SQL API returns at most this many rows and says nothing about it.
PAGE = 1000


def refresh(symbol: str = "SPY") -> int:
    """Pull the IV series and cache it. One row per session, ascending.

    PAGINATED, because the API caps a response at 1,000 rows and reports no
    error, no flag and no `next` cursor when it truncates. The first version of
    this function asked for 1,265 rows, got 1,000, and produced a study that
    silently ended in August 2025 with a year of data missing. Same failure
    class as the chain truncation in the Unusual Whales adapter: an API that
    answers a smaller question than the one asked, quietly.

    The loop stops on a short page, so a full page is always followed by one
    more request -- the only way to tell "exactly 1,000 rows" from "truncated".
    """
    rows: list[dict] = []
    while True:
        page = dolt_query(
            "SELECT date, iv_current, hv_current FROM volatility_history "
            f"WHERE act_symbol='{symbol}' ORDER BY date "
            f"LIMIT {PAGE} OFFSET {len(rows)}")
        rows.extend(page)
        if len(page) < PAGE:
            break
    print(f"pulled {len(rows)} rows in {1 + len(rows) // PAGE} request(s)")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "iv", "hv"])
        for r in rows:
            if r.get("iv_current") in (None, ""):
                continue
            w.writerow([r["date"], r["iv_current"], r.get("hv_current") or ""])
    print(f"cached {len(rows)} rows -> {CACHE}")
    return len(rows)


def load_iv() -> dict[str, float]:
    """VIX as the implied-vol input, from the local cached series.

    Stored as a percentage (18.7 means 18.7 vol), so it is divided by 100 to
    reach the decimal convention every other module in this project uses.
    """
    out: dict[str, float] = {}
    with PRICES.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                v = float(row["vix"]) / 100.0
            except (TypeError, ValueError, KeyError):
                continue
            if v > 0:
                out[row["date"]] = v
    return out


def load_prices() -> list[tuple[str, float]]:
    if not PRICES.exists():
        raise SystemExit(f"no price series at {PRICES}")
    out: list[tuple[str, float]] = []
    with PRICES.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out.append((row["date"], float(row["spy"])))
            except (TypeError, ValueError, KeyError):
                continue
    out.sort()
    return out


def observations(iv: dict[str, float], prices: list[tuple[str, float]]
                 ) -> list[dict]:
    """One record per session that has an IV and a NEXT close.

    The IV is dated on the session it was observed, and the move it is compared
    against is the one that happens AFTERWARDS. Using the same session's move
    would be reading the answer off the page -- the expected move is a forecast
    or it is nothing.
    """
    idx = {d: i for i, (d, _p) in enumerate(prices)}
    out = []
    skipped_no_session = 0
    for d, v in sorted(iv.items()):
        i = idx.get(d)
        if i is None:
            # The early rows of this table are dated on SATURDAYS -- 2019-02-09,
            # -16, -23 are all weekends -- so they match no trading session and
            # are dropped here rather than snapped to a neighbouring day.
            # Snapping would attach an IV to a session it was not observed on,
            # which is a lookahead in the cheapest possible disguise.
            skipped_no_session += 1
            continue
        if i + 1 >= len(prices) or v <= 0:
            continue
        close = prices[i][1]
        nxt = prices[i + 1][1]
        em = close * v * math.sqrt(ONE_SESSION)
        if em <= 0:
            continue
        out.append({
            "date": d, "iv": v, "close": close,
            "expected_move": em,
            "realized_move": abs(nxt - close),
            "ratio": abs(nxt - close) / em,
        })
    if skipped_no_session:
        print(f"  note: {skipped_no_session} IV rows fell on non-trading dates "
              f"(the 2019 rows are Saturday-dated) and were dropped, not snapped")
    return out


def rate(obs: list[dict], threshold: float = 1.0) -> tuple[int, float, float]:
    """Share of sessions realising LESS than `threshold` expected moves."""
    n = len(obs)
    if n == 0:
        return 0, float("nan"), float("nan")
    hits = sum(1 for o in obs if o["ratio"] < threshold)
    p = hits / n
    se = math.sqrt(p * (1 - p) / n)
    return n, p, se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--emit", type=Path, default=Path("data/em_base_rate.json"),
                    help="write the base-rate table the daily module reads")
    args = ap.parse_args()
    if args.refresh:
        refresh(args.symbol)

    obs = observations(load_iv(), load_prices())
    if not obs:
        raise SystemExit("no overlapping sessions between the IV and price series")

    rule = "=" * 74
    print(rule)
    print("EXPECTED-MOVE BASE RATE -- the control, before any gamma signal")
    print(rule)
    print(f"  {len(obs)} sessions, {obs[0]['date']} .. {obs[-1]['date']}")
    print("  IV: VIX from spy_vix_daily.csv (SPX 30-day, an approximation of")
    print("      SPY ATM IV). This is NOT a gamma backtest and cannot become one:")
    print("      no free daily option history carries OPEN INTEREST, and GEX is")
    print("      gamma x open interest. See the module docstring on DoltHub.")
    print()

    n, p, se = rate(obs)
    print(f"  P(realised < 1 expected move) = {p:.1%}  +/- {se:.1%}  (n={n})")
    print(f"  a lognormal one-sigma would give 68.3%")
    print(f"  -> a COMPRESSION call is right {p:.1%} of the time by doing nothing.")
    print(f"  -> an EXPANSION call is right {1-p:.1%} of the time by doing nothing.")
    print()
    print("  Any gamma-conditioned size call must beat ITS OWN side's base rate,")
    print("  not 50%. Reporting a 70% compression hit rate against a coin flip")
    print("  would be the same error as scoring a strategy against zero instead")
    print("  of against the market.")

    ratios = [o["ratio"] for o in obs]
    print(f"\n  realised / expected:  median {st.median(ratios):.3f}   "
          f"mean {st.fmean(ratios):.3f}   p90 {sorted(ratios)[int(0.9*len(ratios))]:.3f}")
    if st.fmean(ratios) < 1.0:
        print("  Mean below 1.0 is the variance risk premium showing up: implied")
        print("  exceeds realised on average, which vrp_data.py measured at 3.62")
        print("  vol points and 83.6% of days. Same fact, different instrument.")

    print("\n  by IV regime (DESCRIPTIVE -- the buckets were not pre-registered):")
    ivs = sorted(o["iv"] for o in obs)
    cuts = [ivs[int(len(ivs) * f)] for f in (0.25, 0.5, 0.75)]
    buckets: dict[str, list[dict]] = {}
    for o in obs:
        if o["iv"] <= cuts[0]:
            k = f"IV <= {cuts[0]:.1%}"
        elif o["iv"] <= cuts[1]:
            k = f"{cuts[0]:.1%} - {cuts[1]:.1%}"
        elif o["iv"] <= cuts[2]:
            k = f"{cuts[1]:.1%} - {cuts[2]:.1%}"
        else:
            k = f"IV > {cuts[2]:.1%}"
        buckets.setdefault(k, []).append(o)
    for k in sorted(buckets, key=lambda x: len(buckets[x]), reverse=True):
        bn, bp, bse = rate(buckets[k])
        print(f"    {k:<18} n={bn:>4}   P(compress) {bp:.1%} +/- {bse:.1%}")

    # EMITTED, not printed-and-retyped. A base rate that lives in a docstring
    # gets copied wrong exactly once and is then never questioned again; one
    # written by the measurement carries its own n, its own edges, and the
    # sample it came from, so a stale table is visible rather than invisible.
    table = {
        "source": f"VIX + SPY closes, {obs[0]['date']}..{obs[-1]['date']}",
        "generated": date.today().isoformat(),
        "threshold_em": 1.0,
        "n": len(obs),
        "overall_p_compress": round(p, 6),
        "iv_edges": [round(c, 6) for c in cuts],
        "buckets": [],
    }
    ordered: list[list[dict]] = [[] for _ in range(len(cuts) + 1)]
    for o in obs:
        ordered[sum(1 for c in cuts if o["iv"] > c)].append(o)
    for k, bucket in enumerate(ordered):
        bn, bp, bse = rate(bucket)
        table["buckets"].append({
            "iv_from": None if k == 0 else round(cuts[k - 1], 6),
            "iv_to": None if k == len(cuts) else round(cuts[k], 6),
            "n": bn, "p_compress": round(bp, 6), "se": round(bse, 6)})
    args.emit.parent.mkdir(parents=True, exist_ok=True)
    args.emit.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(f"\n  base-rate table -> {args.emit}")

    print("\n  by year (DESCRIPTIVE -- not a test, and not corrected):")
    years: dict[str, list[dict]] = {}
    for o in obs:
        years.setdefault(o["date"][:4], []).append(o)
    for y in sorted(years):
        yn, yp, yse = rate(years[y])
        print(f"    {y}   n={yn:>4}   P(compress) {yp:.1%} +/- {yse:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
