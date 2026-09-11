"""Vertical credit spreads: find them, price them, state the risk exactly.

    python -m options.spread SPY --width 5 --delta 0.25

WHY THIS EXISTS. Every edge this project has measured points at SELLING
premium -- the variance risk premium at t = -4.26, the gamma size effect, the
80.8% compression base rate -- and a Level 2 account can only BUY it. A defined
-risk vertical is the smallest structure that gets on the right side of that,
and Level 3 is what permits it.

WHAT YOUR OWN BACKTEST SAYS ABOUT IT, stated here so nobody rediscovers it
after the money is on: `put_write_backtest.py` measured the 25d/10d put credit
spread at **1.8% CAGR, Sharpe 0.45**, against the naked 25-delta put at 5.0%
and Sharpe 0.77 -- "the 10-delta insurance costs more than it saves." The
spread is the WORSE risk-adjusted trade. It is also the only one that fits in
a small account, because the naked put on SPY needs the strike in collateral.
That trade-off is the price of admission, not a discovery.

THE FILL IS ASSUMED AGAINST YOU, always. The credit is computed as
`short.bid - long.ask` -- sell at the bid, buy at the ask. Mid-price credit is
the number that makes every spread look good and that nobody receives. The
report prints both so the gap is visible, and the gap on a wide market is
frequently larger than the edge.

THE RISK IDENTITY, checked in `verify_spread.py` rather than trusted:

    max_profit + max_loss == width * 100      (always, exactly)
    breakeven strictly between the two strikes
    collateral == max_loss                     (defined risk; this is what a
                                                broker holds against it)

If any of those fails the spread is not priced, it is mis-parsed, and the
builder refuses it rather than reporting a number that will be acted on.

Standard library only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .gex import MULTIPLIER, Chain, Contract, bs_delta

#: Credit below this fraction of width is not worth the tail risk. The
#: one-third rule is folklore, not a measurement, and is labelled as such.
MIN_CREDIT_RATIO = 0.20
#: Premium selling lives in the 30-45 DTE window by convention: enough theta
#: to matter, far enough out that gamma has not gone convex. Also folklore.
DTE_MIN, DTE_MAX = 21, 60


@dataclass(frozen=True)
class Vertical:
    """One defined-risk credit spread, priced at the fill you would get."""

    symbol: str
    kind: str                 # "put credit" | "call credit"
    expiry: str
    dte: int
    short_strike: float
    long_strike: float
    short_delta: float
    credit: float             # per share, at bid/ask
    credit_mid: float         # per share, at mid -- for comparison only
    spot: float

    @property
    def width(self) -> float:
        return abs(self.short_strike - self.long_strike)

    @property
    def max_profit(self) -> float:
        return self.credit * MULTIPLIER

    @property
    def max_loss(self) -> float:
        return (self.width - self.credit) * MULTIPLIER

    @property
    def collateral(self) -> float:
        """What the broker holds. For a defined-risk vertical, the max loss."""
        return self.max_loss

    @property
    def credit_ratio(self) -> float:
        return self.credit / self.width if self.width > 0 else 0.0

    @property
    def breakeven(self) -> float:
        if self.kind == "put credit":
            return self.short_strike - self.credit
        return self.short_strike + self.credit

    @property
    def prob_profit(self) -> float:
        """P(expiring worthless), approximated by 1 - |delta| of the short leg.

        Delta is not a probability -- it approximates the risk-neutral chance
        of finishing in the money, which is not the real-world chance and does
        not price the variance premium that is the entire reason to be here.
        Named `prob_profit` because that is what every broker calls it, and
        flagged in the report as an approximation for exactly this reason.
        """
        return 1.0 - abs(self.short_delta)

    @property
    def breakeven_win_rate(self) -> float:
        """The win rate at which this trade breaks even: loss / (loss + profit).

        THE NUMBER THAT DECIDES A CREDIT SPREAD, and the one brokers do not
        show. Risking $84 to make $16 needs 84% of trades to win just to stay
        level, against a delta-implied 75%. That gap is not a flaw in the
        trade -- it is the definition of risk-neutral pricing, under which
        every option is a fair bet and EV is zero before costs.

        The whole thesis is that the REAL-WORLD win rate exceeds the
        risk-neutral one, because implied vol exceeds realised. This project
        measured that premium at t = -4.26 in `vrp_data.py`. So the trade is
        positive only to the extent the variance risk premium is real and
        survives the fill -- and this field is what makes the required edge
        explicit instead of implied.
        """
        total = self.max_profit + self.max_loss
        return self.max_loss / total if total > 0 else float("nan")

    @property
    def edge_needed(self) -> float:
        """Percentage points the real win rate must beat the delta estimate by."""
        return self.breakeven_win_rate - self.prob_profit

    @property
    def slippage(self) -> float:
        """Dollars given up versus a mid fill. Often bigger than the edge."""
        return (self.credit_mid - self.credit) * MULTIPLIER

    def check(self) -> list[str]:
        """Structural checks. A non-empty list means DO NOT TRADE THIS."""
        bad: list[str] = []
        if self.credit <= 0:
            bad.append("credit is not positive -- this is a debit, not a credit")
        if self.credit >= self.width:
            bad.append(f"credit {self.credit:.2f} >= width {self.width:.2f}: "
                       "risk-free money does not exist, the quote is stale")
        if self.width <= 0:
            bad.append("strikes are identical")
        if self.kind == "put credit" and self.long_strike >= self.short_strike:
            bad.append("put credit spread must BUY the lower strike")
        if self.kind == "call credit" and self.long_strike <= self.short_strike:
            bad.append("call credit spread must BUY the higher strike")
        lo, hi = sorted((self.short_strike, self.long_strike))
        if not (lo < self.breakeven < hi):
            bad.append(f"breakeven {self.breakeven:.2f} is outside "
                       f"[{lo:.2f}, {hi:.2f}]")
        total = self.max_profit + self.max_loss
        if abs(total - self.width * MULTIPLIER) > 1e-6:
            bad.append(f"max_profit + max_loss = {total:.4f}, width*100 = "
                       f"{self.width * MULTIPLIER:.4f}")
        return bad

    def render(self) -> str:
        out = [
            f"{self.symbol} {self.kind.upper()}  {self.expiry} ({self.dte}d)",
            f"  SELL {self.short_strike:>8.2f} {self.kind[0].upper()}"
            f"   delta {self.short_delta:+.3f}",
            f"  BUY  {self.long_strike:>8.2f} {self.kind[0].upper()}"
            f"   width {self.width:.2f}",
            "",
            f"  credit          ${self.credit * MULTIPLIER:>8.2f}   "
            f"(${self.credit:.2f}/share, at bid/ask)",
            f"  max profit      ${self.max_profit:>8.2f}",
            f"  max loss        ${self.max_loss:>8.2f}",
            f"  collateral      ${self.collateral:>8.2f}   held until expiry "
            f"or close",
            f"  breakeven       {self.breakeven:>9.2f}   "
            f"({(self.breakeven / self.spot - 1) * 100:+.2f}% from spot "
            f"{self.spot:.2f})",
            f"  credit/width    {self.credit_ratio:>9.1%}",
            f"  P(profit)~      {self.prob_profit:>9.1%}   approximation from "
            f"delta, NOT a real-world probability",
            f"  BREAKEVEN WIN   {self.breakeven_win_rate:>9.1%}   <- must win "
            f"this often just to stay level",
            f"  edge needed     {self.edge_needed:>+9.1%}   the real win rate "
            f"must beat delta's by this much",
            f"  mid-fill gap    ${self.slippage:>8.2f}   "
            f"({self.slippage / self.max_profit:.0%} of max profit) what a mid "
            f"fill would add, and will not",
        ]
        problems = self.check()
        if problems:
            out.append("")
            out += [f"  REFUSED: {p}" for p in problems]
        return "\n".join(out)


def _delta_of(c: Contract, chain: Chain) -> float:
    return bs_delta(chain.spot, c.strike, c.t_years, c.iv, chain.r, c.right)


def build(chain: Chain, kind: str, short: Contract, long_: Contract) -> Vertical:
    """Price a specific pair. Credit assumes you sell the bid and buy the ask."""
    credit = short.bid - long_.ask
    credit_mid = short.mid - long_.mid
    return Vertical(
        symbol=chain.symbol, kind=kind, expiry=short.expiry,
        dte=int(round(short.t_years * 365.0)),
        short_strike=short.strike, long_strike=long_.strike,
        short_delta=_delta_of(short, chain),
        credit=credit, credit_mid=credit_mid, spot=chain.spot)


def find(chain: Chain, *, kind: str = "put credit", width: float = 5.0,
         target_delta: float = 0.25, dte_min: int = DTE_MIN,
         dte_max: int = DTE_MAX, min_credit_ratio: float = MIN_CREDIT_RATIO,
         max_collateral: float | None = None) -> list[Vertical]:
    """Every valid spread at the target, best credit ratio first.

    Selection is by SHORT DELTA, the standard way to name a strike, then the
    long leg is the contract `width` away. Candidates that fail `check()` are
    dropped -- a spread whose arithmetic does not close is a parsing error, and
    reporting one alongside sound ones invites it to be traded.
    """
    right = "p" if kind == "put credit" else "c"
    by_expiry: dict[str, list[Contract]] = {}
    for c in chain.contracts:
        if c.right != right or c.bid <= 0 or c.ask <= 0 or c.iv <= 0:
            continue
        dte = int(round(c.t_years * 365.0))
        if dte_min <= dte <= dte_max:
            by_expiry.setdefault(c.expiry, []).append(c)

    out: list[Vertical] = []
    for expiry, rows in by_expiry.items():
        strikes = {c.strike: c for c in rows}
        # The short leg: whichever strike sits closest to the target delta.
        scored = [(abs(abs(_delta_of(c, chain)) - target_delta), c)
                  for c in rows]
        if not scored:
            continue
        scored.sort(key=lambda x: (x[0], x[1].strike))
        for _gap, short in scored[:3]:      # a few nearby strikes, not just one
            k_long = (short.strike - width if right == "p"
                      else short.strike + width)
            long_ = strikes.get(k_long)
            if long_ is None:
                continue
            v = build(chain, kind, short, long_)
            if v.check():
                continue
            if v.credit_ratio < min_credit_ratio:
                continue
            if max_collateral is not None and v.collateral > max_collateral:
                continue
            out.append(v)
    out.sort(key=lambda v: -v.credit_ratio)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os
    import sys
    from .sources import from_chain_store

    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--store",
                    default=os.environ.get("QUANTDESK_CHAINS",
                                           "../quantdesk/data/chains.sqlite"))
    ap.add_argument("--kind", default="put credit",
                    choices=("put credit", "call credit"))
    ap.add_argument("--width", type=float, default=5.0)
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--max-collateral", type=float, default=None)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    chain = from_chain_store(args.store, args.symbol.upper(), max_dte=DTE_MAX + 5)
    found = find(chain, kind=args.kind, width=args.width,
                 target_delta=args.delta, max_collateral=args.max_collateral)
    print(f"{chain.symbol} spot {chain.spot:,.2f}   session {chain.asof[:10]}")
    print(f"{len(found)} valid {args.kind} spread(s) at width {args.width:g}, "
          f"target delta {args.delta:.2f}\n")
    for v in found[:args.top]:
        print(v.render())
        print()

    if not found:
        # Showing NOTHING when the only failure is the credit floor teaches you
        # nothing and invites you to lower the floor until something appears.
        # Show the best structurally-sound spreads and label why they failed,
        # so the decision is "the market pays 17%, is that enough" rather than
        # "the tool found nothing".
        near = find(chain, kind=args.kind, width=args.width,
                    target_delta=args.delta, min_credit_ratio=0.0,
                    max_collateral=args.max_collateral)
        if not near:
            print("  no structurally sound spread at this width -- the strikes")
            print("  may not exist, or one leg is unquoted.")
            return 0
        print(f"  None cleared the {MIN_CREDIT_RATIO:.0%} credit floor. The best")
        print(f"  sound ones are below, BELOW YOUR FLOOR -- that is the market")
        print(f"  telling you what it pays, not a tool failure.\n")
        for v in near[:args.top]:
            print(v.render())
            print(f"  ^ credit/width {v.credit_ratio:.1%} is under the "
                  f"{MIN_CREDIT_RATIO:.0%} floor")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
