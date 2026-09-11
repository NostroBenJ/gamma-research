"""Does the dealer-gamma regime beat its own null on realised move size?

    python -m research.gex_regime_test

THE QUESTION, and it is the one the whole gamma build has been circling.
`options/daily.py` turns a gamma regime into a SIZE call: positive gamma ->
compression, negative -> expansion. `research/em_base_rate.py` measured what
that call is worth by doing nothing -- 80.8% of sessions come in under one
expected move, rising to 87.3% in the calmest vol bucket. So the question is
not "is the compression call often right" (it is, trivially) but **does
knowing the gamma regime change the odds at all.**

THE DATA. `quantdesk/data/spx_gex_daily.csv`, 251 sessions
2025-08-28 .. 2026-08-28, built by `quantdesk/scripts/build_gex_series.py`
from PURCHASED Databento OPRA data: SPX and SPXW books merged, spot recovered
from put-call parity, IV solved from quotes, net GEX and the zero-gamma flip
computed at three horizons. This is not a free proxy; it is the real book.

NO LOOKAHEAD, and the source file is explicit about why:

    "`session` is the session the OPEN INTEREST describes -- the day BEFORE
     the file's date, because OPRA publishes OI at 06:30 ET for the prior
     close. The signal on row D is therefore knowable before session D opens,
     which is what makes it tradeable rather than hindsight."

So the regime on a row is compared against the move that happens AFTER that
row -- spot to the next row's spot. Nothing in the outcome window is used to
form the signal.

PRE-REGISTERED HORIZON, and it was registered before any of this existed.
`build_gex_series.py`, written in August: **"7DTE is the primary horizon.
1DTE and 30DTE are secondaries and count against the multiple-testing
correction."** That is honoured here rather than re-chosen: 7DTE is the
headline, the other two are reported with a Bonferroni bar across three.

THE STATISTIC. A two-proportion z-test on P(realised < 1 expected move),
positive-gamma sessions against negative-gamma sessions. The null is that the
regime carries no information about size, which is exactly the claim at issue,
and it needs no external base rate -- the two arms come from the same sample
and the same period.

WHY VIX IS THE RIGHT IV HERE, where it was an approximation before. VIX *is*
SPX 30-day implied volatility. `em_base_rate.py` used it against SPY closes
and said so; this study uses it against the SPX forward, which is the index it
is actually derived from.

Standard library only.
"""

from __future__ import annotations

import csv
import math
import os
import statistics as st
from pathlib import Path

# The SPX series is built from licensed Databento data and is not in this repo.
GEX = Path(os.environ.get("SPX_GEX_CSV", "../quantdesk/data/spx_gex_daily.csv"))
VIX = Path(os.environ.get("SPY_VIX_CSV", "../options-research/spy_vix_daily.csv"))
HORIZONS = ("7", "1", "30")          # primary first, per the registration
PRIMARY = "7"
N_REGISTERED = 3
#: Bonferroni across the three registered horizons, two-sided alpha 0.05.
BAR = 2.39
#: Pairs further apart than this are holiday stretches, not sessions.
MAX_GAP_SESSIONS = 4


def load_vix() -> dict[str, float]:
    out: dict[str, float] = {}
    with VIX.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["date"]] = float(row["vix"]) / 100.0
            except (TypeError, ValueError, KeyError):
                continue
    return out


def trading_calendar() -> list[str]:
    with VIX.open(encoding="utf-8") as fh:
        return sorted(r["date"] for r in csv.DictReader(fh))


def load_gex() -> list[dict]:
    with GEX.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)]
    rows.sort(key=lambda r: r["session"])
    return rows


def observations(horizon: str) -> tuple[list[dict], dict[str, int]]:
    """One record per (signal row, next row) pair, with why any were dropped."""
    vix = load_vix()
    cal = trading_calendar()
    pos = {d: i for i, d in enumerate(cal)}
    rows = load_gex()
    dropped = {"no_contracts": 0, "no_vix": 0, "gap_too_wide": 0, "bad_spot": 0}
    out: list[dict] = []

    for a, b in zip(rows, rows[1:]):
        try:
            used = float(a[f"used_{horizon}"] or 0)
            gex = float(a[f"net_gex_{horizon}"] or 0)
            s0, s1 = float(a["spot"]), float(b["spot"])
        except (TypeError, ValueError):
            dropped["bad_spot"] += 1
            continue
        # A horizon with no contracts has no regime. The 1DTE column is 0 on
        # many rows for exactly this reason and those rows are not evidence.
        if used <= 0 or gex == 0:
            dropped["no_contracts"] += 1
            continue
        if s0 <= 0 or s1 <= 0:
            dropped["bad_spot"] += 1
            continue
        iv = vix.get(a["session"])
        if iv is None or iv <= 0:
            dropped["no_vix"] += 1
            continue
        i, j = pos.get(a["session"]), pos.get(b["session"])
        gap = (j - i) if (i is not None and j is not None) else 1
        if gap < 1 or gap > MAX_GAP_SESSIONS:
            dropped["gap_too_wide"] += 1
            continue
        # Expected move over the ACTUAL elapsed sessions, not assumed to be one.
        em = s0 * iv * math.sqrt(gap / 252.0)
        if em <= 0:
            dropped["bad_spot"] += 1
            continue
        out.append({
            "session": a["session"], "gap": gap, "iv": iv, "spot": s0,
            "net_gex": gex, "regime": "positive" if gex > 0 else "negative",
            "expected_move": em, "realized_move": abs(s1 - s0),
            "ratio": abs(s1 - s0) / em,
        })
    return out, dropped


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """Pooled two-proportion z. Positive means arm 1 compresses more often."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else float("nan")


def welch_t(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    se = math.sqrt(va + vb)
    return (st.fmean(a) - st.fmean(b)) / se if se > 0 else float("nan")


def run(horizon: str, label: str) -> dict:
    obs, dropped = observations(horizon)
    pos = [o for o in obs if o["regime"] == "positive"]
    neg = [o for o in obs if o["regime"] == "negative"]
    kp = sum(1 for o in pos if o["ratio"] < 1.0)
    kn = sum(1 for o in neg if o["ratio"] < 1.0)

    print("=" * 74)
    print(f"{label}  --  net_gex_{horizon}")
    print("=" * 74)
    print(f"  {len(obs)} usable pairs   dropped: " + ", ".join(
        f"{k} {v}" for k, v in dropped.items() if v))
    if not pos or not neg:
        print("  one arm is empty -- no comparison possible")
        return {"z": float("nan"), "n": len(obs)}

    print(f"\n  {'regime':<10}{'n':>5}{'P(compress)':>13}{'mean ratio':>12}"
          f"{'median':>9}")
    for name, arm, k in (("positive", pos, kp), ("negative", neg, kn)):
        r = [o["ratio"] for o in arm]
        print(f"  {name:<10}{len(arm):>5}{k/len(arm):>12.1%}"
              f"{st.fmean(r):>12.3f}{st.median(r):>9.3f}")

    z = two_proportion_z(kp, len(pos), kn, len(neg))
    t = welch_t([o["ratio"] for o in pos], [o["ratio"] for o in neg])
    gap = kp / len(pos) - kn / len(neg)
    print(f"\n  PRIMARY  P(compress | positive) - P(compress | negative) "
          f"= {gap:+.1%}")
    print(f"           two-proportion z = {z:+.2f}   "
          f"(bar |z| > {BAR:.2f}, Bonferroni across {N_REGISTERED} horizons)")
    print(f"  secondary  mean-ratio Welch t = {t:+.2f}  "
          f"(negative gamma should move MORE, so t < 0 is the prior)")

    # What gap would this test have needed to SEE? A null is only informative
    # next to the effect it could have detected, and "underpowered" is a claim
    # that should carry a number like any other.
    p_pool = (kp + kn) / (len(pos) + len(neg))
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / len(pos) + 1 / len(neg)))
    mde = BAR * se
    print(f"  power      this split could only resolve a gap of "
          f"{mde:.1%} or larger; observed {abs(gap):.1%}")

    verdict = ("BEATS its null" if z > BAR else
               "INVERTED -- negative gamma compressed MORE" if z < -BAR else
               "indistinguishable from its null")
    print(f"\n  VERDICT: {verdict}")
    return {"z": z, "t": t, "n": len(obs), "gap": gap, "mde": mde,
            "p_pos": kp / len(pos), "p_neg": kn / len(neg),
            "n_pos": len(pos), "n_neg": len(neg), "verdict": verdict}


def main() -> int:
    print("=" * 74)
    print("DEALER GAMMA REGIME vs REALISED MOVE SIZE")
    print("=" * 74)
    print("  data     purchased Databento OPRA, SPX+SPXW, via quantdesk")
    print("  signal   knowable before the outcome session opens (OI is")
    print("           published 06:30 ET for the prior close)")
    print(f"  primary  {PRIMARY}DTE, pre-registered in build_gex_series.py")
    print("           before any of this was run")
    print()

    results = {}
    results[PRIMARY] = run(PRIMARY, f"PRIMARY -- {PRIMARY}DTE")
    print()
    for h in HORIZONS:
        if h == PRIMARY:
            continue
        results[h] = run(h, f"secondary -- {h}DTE")
        print()

    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  {'horizon':<10}{'n':>6}{'P(pos)':>9}{'P(neg)':>9}{'gap':>9}"
          f"{'z':>8}   verdict")
    for h in HORIZONS:
        r = results.get(h) or {}
        if not r or r.get("z") != r.get("z"):
            print(f"  {h+'DTE':<10}{r.get('n', 0):>6}   -- no comparison")
            continue
        print(f"  {h+'DTE':<10}{r['n']:>6}{r['p_pos']:>9.1%}{r['p_neg']:>9.1%}"
              f"{r['gap']:>+9.1%}{r['z']:>+8.2f}   {r['verdict']}")
    print(f"\n  The bar is |z| > {BAR:.2f}. One year is 251 sessions and the two")
    print("  arms split it further, so this is a LOW-POWER test: a null here is")
    print("  'cannot tell', not 'does not work'. What it CAN rule out is a")
    print("  large effect, and a large effect is what would be needed to pay")
    print("  for a data subscription out of this account.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
