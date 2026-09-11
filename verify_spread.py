"""Verification for `options/spread.py`. Run: `python verify_spread.py`

This is the first module in the project whose output is meant to be ACTED ON
with real money, so the risk arithmetic is proved rather than asserted. The
load-bearing checks:

  [2] max_profit + max_loss == width * 100, EXACTLY, on random inputs
  [3] the credit is the fill you GET (sell the bid, buy the ask), never mid
  [4] a spread whose arithmetic does not close is REFUSED, not reported
  [6] breakeven is where P&L is actually zero, checked against a payoff model

[6] matters most: `breakeven` is a formula, and a formula nobody has evaluated
a payoff against is a guess. It is checked by pricing the position at expiry.
"""

from __future__ import annotations

import random
import sys

from options.gex import Chain, Contract, bs_delta
from options.spread import build, find

PASS, FAIL = 0, 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}  {detail}")


T = 30 / 365.0


def leg(strike, right, bid, ask, iv=0.20):
    return Contract(strike=strike, right=right, expiry="2026-10-09", t_years=T,
                    iv=iv, oi=100, bid=bid, ask=ask)


def chain_of(*legs, spot=100.0):
    return Chain("TEST", "2026-09-09T16:00:00", spot, tuple(legs), r=0.0)


def payoff_at_expiry(v, underlying: float) -> float:
    """P&L in dollars if the underlying finishes at `underlying`.

    Written independently of the Vertical properties, so it can contradict
    them. Short leg collected, long leg paid, both settled at intrinsic.
    """
    if v.kind == "put credit":
        short_int = max(0.0, v.short_strike - underlying)
        long_int = max(0.0, v.long_strike - underlying)
    else:
        short_int = max(0.0, underlying - v.short_strike)
        long_int = max(0.0, underlying - v.long_strike)
    return (v.credit - short_int + long_int) * 100.0


def main() -> int:
    print("[1] a textbook put credit spread prices as expected")
    ch = chain_of(leg(95, "p", 1.00, 1.10), leg(90, "p", 0.40, 0.50))
    v = build(ch, "put credit", ch.contracts[0], ch.contracts[1])
    check("credit is short.bid - long.ask", abs(v.credit - 0.50) < 1e-12,
          f"{v.credit}")
    check("width is 5", v.width == 5.0)
    check("max profit is the credit x 100", abs(v.max_profit - 50.0) < 1e-9)
    check("max loss is (width - credit) x 100", abs(v.max_loss - 450.0) < 1e-9)
    check("collateral equals max loss", v.collateral == v.max_loss)
    check("breakeven is short strike minus credit",
          abs(v.breakeven - 94.5) < 1e-12, f"{v.breakeven}")
    check("no structural complaints", v.check() == [], str(v.check()))

    print("\n[2] THE IDENTITY: max_profit + max_loss == width * 100, exactly")
    rng = random.Random(20260909)
    worst = 0.0
    for _ in range(2000):
        w = rng.choice([0.5, 1.0, 2.5, 5.0, 10.0, 25.0])
        ks = rng.uniform(20, 500)
        cr = rng.uniform(0.01, 0.95) * w
        c = chain_of(leg(ks, "p", cr + 1.0, cr + 1.05),
                     leg(ks - w, "p", 1.0, 1.0))
        vv = build(c, "put credit", c.contracts[0], c.contracts[1])
        worst = max(worst, abs(vv.max_profit + vv.max_loss - w * 100.0))
    check(f"holds on 2,000 random spreads (worst error {worst:.2e})",
          worst < 1e-6, f"{worst}")

    print("\n[3] the fill is assumed AGAINST you")
    check("credit uses bid/ask, not mid", abs(v.credit - 0.50) < 1e-12)
    check("mid credit is strictly better", v.credit_mid > v.credit,
          f"{v.credit_mid} vs {v.credit}")
    check("the gap is reported in dollars",
          abs(v.slippage - (v.credit_mid - v.credit) * 100) < 1e-9)
    check("a mid fill would have looked 20% better here",
          abs(v.slippage - 10.0) < 1e-9, f"{v.slippage}")

    print("\n[4] arithmetic that does not close is REFUSED, not reported")
    bad = build(chain_of(leg(95, "p", 9.0, 9.1), leg(90, "p", 0.10, 0.20)),
                "put credit", leg(95, "p", 9.0, 9.1), leg(90, "p", 0.10, 0.20))
    check("credit >= width is caught", any("risk-free" in p for p in bad.check()),
          str(bad.check()))
    debit = build(chain_of(leg(90, "p", 0.10, 0.20), leg(95, "p", 9.0, 9.1)),
                  "put credit", leg(90, "p", 0.10, 0.20), leg(95, "p", 9.0, 9.1))
    check("a net debit is caught", any("debit" in p for p in debit.check()),
          str(debit.check()))
    check("a put credit spread buying the HIGHER strike is caught",
          any("lower strike" in p for p in debit.check()), str(debit.check()))
    check("the refusal is rendered, not hidden", "REFUSED" in bad.render())
    check("find() drops refused spreads",
          all(x.check() == [] for x in find(ch, width=5.0, min_credit_ratio=0.0)))

    print("\n[5] the call side mirrors it")
    cc = chain_of(leg(105, "c", 1.00, 1.10), leg(110, "c", 0.40, 0.50))
    cv = build(cc, "call credit", cc.contracts[0], cc.contracts[1])
    check("breakeven is short strike PLUS credit",
          abs(cv.breakeven - 105.5) < 1e-12, f"{cv.breakeven}")
    check("same identity holds",
          abs(cv.max_profit + cv.max_loss - 500.0) < 1e-9)
    check("no complaints", cv.check() == [], str(cv.check()))
    wrong = build(cc, "call credit", cc.contracts[1], cc.contracts[0])
    check("a call credit spread buying the LOWER strike is caught",
          any("higher strike" in p for p in wrong.check()), str(wrong.check()))

    print("\n[6] breakeven and the extremes match an INDEPENDENT payoff model")
    for name, vv in (("put credit", v), ("call credit", cv)):
        check(f"{name}: P&L at breakeven is zero",
              abs(payoff_at_expiry(vv, vv.breakeven)) < 1e-9,
              f"{payoff_at_expiry(vv, vv.breakeven)}")
        far_good = 1e6 if vv.kind == "put credit" else 0.0
        far_bad = 0.0 if vv.kind == "put credit" else 1e6
        check(f"{name}: best case equals max_profit",
              abs(payoff_at_expiry(vv, far_good) - vv.max_profit) < 1e-9)
        check(f"{name}: worst case equals -max_loss",
              abs(payoff_at_expiry(vv, far_bad) + vv.max_loss) < 1e-9)
        mid_k = 0.5 * (vv.short_strike + vv.long_strike)
        pnl = payoff_at_expiry(vv, mid_k)
        check(f"{name}: between the strikes P&L is between the extremes",
              -vv.max_loss < pnl < vv.max_profit, f"{pnl}")

    print("\n[7] the breakeven WIN RATE is the decision number")
    check("breakeven win = loss / (loss + profit)",
          abs(v.breakeven_win_rate - 450.0 / 500.0) < 1e-12,
          f"{v.breakeven_win_rate}")
    check("a 9:1 risk/reward needs 90% wins",
          abs(build(chain_of(leg(95, "p", 1.0, 1.05), leg(85, "p", 0.0, 0.0)),
                    "put credit", leg(95, "p", 1.0, 1.05),
                    leg(85, "p", 0.0, 0.0)).breakeven_win_rate - 0.90) < 1e-9)
    check("edge needed is breakeven minus the delta estimate",
          abs(v.edge_needed - (v.breakeven_win_rate - v.prob_profit)) < 1e-12)
    check("prob_profit is 1 - |short delta|",
          abs(v.prob_profit - (1 - abs(v.short_delta))) < 1e-12)
    check("the short delta is computed, not read from the vendor",
          abs(v.short_delta - bs_delta(100.0, 95.0, T, 0.20, 0.0, "p")) < 1e-12)

    print("\n[8] find() respects its filters")
    got = find(ch, width=5.0, min_credit_ratio=0.0)
    check("a sound spread is found at all", len(got) >= 1, str(len(got)))
    check("the credit floor excludes it when raised",
          find(ch, width=5.0, min_credit_ratio=0.99) == [])
    check("a collateral cap excludes it when tight",
          find(ch, width=5.0, min_credit_ratio=0.0, max_collateral=10.0) == [])
    check("a width with no matching strike finds nothing",
          find(ch, width=7.0, min_credit_ratio=0.0) == [])
    check("unquoted legs are skipped",
          find(chain_of(leg(95, "p", 0.0, 0.0), leg(90, "p", 0.0, 0.0)),
               width=5.0, min_credit_ratio=0.0) == [])

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
