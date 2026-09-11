"""Dealer gamma exposure, walls, and the flip, from an option-chain snapshot.

    python -m options.gex data/chains/INTC.json

Stdlib only. `math.erf` is the one special function this needs, and keeping the
maths dependency-free is what lets `verify_gex.py` prove it against finite
differences without an environment.

WHAT THIS IS FOR. Net GEX says whether dealers are positioned to DAMPEN moves
or AMPLIFY them, and the walls say where their hedging concentrates. It is a
map of where price is likely to stick or accelerate. It is NOT a direction
signal, and this repo has already measured that distinction on 247 sessions:
`FINDINGS.md` §7 found gamma regime predicts SIZE (t = -6.57) and NOT DIRECTION
(t = -1.22). Anything built on top of this module should respect that result.

THE DEALER SIGN CONVENTION, which is an assumption and not a law: dealers are
LONG CALLS and SHORT PUTS, so calls contribute positive gamma and puts
negative. This is the common retail convention; vendors tweak it. Changing it
invalidates every level and every stored record computed under the old one, so
it lives in one constant, `PUT_SIGN`, and is never inlined.

FOUR BUGS THIS FILE IS SHAPED TO AVOID. Each was live somewhere before:

1. **Regime comes from the SIGN OF NET GEX AT SPOT**, never from comparing spot
   to the flip. Those agree only when the flip is right, and a -$3.68B (deeply
   short gamma) book once got labelled "positive gamma, expect mean-reversion"
   -- exactly backwards, on the top line of a dashboard.
2. **The flip re-prices the WHOLE CHAIN at each candidate spot.** Gamma depends
   on spot, so a cumulative sum across strikes is not the same function. The
   shortcut fires on float-noise sign flips among worthless deep-OTM strikes;
   it once returned 421 against a spot of 684. Candidates are scanned OUTWARD
   from spot so the nearest crossing wins, and `None` is returned when no
   crossing exists in range -- "no flip in range" is a real answer.
3. **Walls carry a side AND a sign constraint.** A call wall is resistance and
   must sit at or above spot with positive gamma; a put wall is support and
   must sit at or below with negative. A global max/min will cheerfully render
   a floor above the current price.
4. **Degenerate rows return 0 rather than raising**, so one bad contract in a
   chain of nine hundred cannot take down the pipeline.

ASSUMPTION (sticky IV): re-pricing the chain at a candidate spot holds each
contract's implied volatility fixed. A real move walks along the skew, so the
computed flip is approximate in the same direction for everyone who computes it
this way. Stated because it is invisible in the output.

ASSUMPTION (open interest is a day stale): OI is published overnight, so
intraday GEX is built on yesterday's positioning plus today's spot. Volume is
carried alongside so a strike whose OI and volume disagree wildly is visible.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

#: Contract multiplier: one option covers 100 shares.
MULTIPLIER = 100.0

#: Dealers are long calls, short puts. The single place this convention lives.
CALL_SIGN = 1.0
PUT_SIGN = -1.0

#: A chain is flagged when strikes quoted on only ONE side carry at least this
#: share of its total absolute gamma. Counting strikes instead fires on every
#: complete chain, because far wings are naturally one-sided and weightless.
ONE_SIDED_TOLERANCE = 0.05

#: Calendar days in a year. `T` is ALWAYS in years (1 day = 1/365).
DAYS_PER_YEAR = 365.0

SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def bs_gamma(spot: float, strike: float, t_years: float, iv: float,
             r: float = 0.0, q: float = 0.0) -> float:
    """Black-Scholes gamma. Calls and puts share it.

        gamma = exp(-qT) * N'(d1) / (S * sigma * sqrt(T))

    Returns 0.0 for degenerate inputs rather than raising: a chain snapshot
    routinely contains an expired row, a zero-IV row, or a strike of 0, and one
    of them must not take down a nine-hundred-contract calculation.
    """
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return 0.0
    sqrt_t = math.sqrt(t_years)
    d1 = ((math.log(spot / strike) + (r - q + 0.5 * iv * iv) * t_years)
          / (iv * sqrt_t))
    return math.exp(-q * t_years) * norm_pdf(d1) / (spot * iv * sqrt_t)


def norm_cdf(x: float) -> float:
    """Standard normal CDF. `math.erf` is the only special function needed."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(spot: float, strike: float, t_years: float, iv: float,
             r: float = 0.0, right: str = "c") -> float:
    """Black-Scholes delta. Calls in [0,1], puts in [-1,0].

    Used for STRIKE SELECTION, not for hedging: "sell the 25-delta put" is the
    standard way to name a strike, and computing it here rather than reading
    the vendor's keeps one source of truth. The feed's own delta is carried
    alongside in the chain store as a cross-check, and CBOE's is unreliable
    in-the-money for the same reason its IV is.

    Returns 0.0 on degenerate inputs, matching `bs_gamma`.
    """
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return 0.0
    d1 = ((math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years)
          / (iv * math.sqrt(t_years)))
    return norm_cdf(d1) if right == "c" else norm_cdf(d1) - 1.0


def dollar_gamma(gamma: float, oi: float, spot: float) -> float:
    """Dollars of dealer delta per 1% move in the underlying.

        $GEX = gamma * OI * 100 * S^2 * 0.01

    The 100 is the contract multiplier; `S^2 * 0.01` converts gamma-per-$1 into
    dollars of delta created by a 1% move.
    """
    return gamma * oi * MULTIPLIER * spot * spot * 0.01


@dataclass(frozen=True)
class Contract:
    strike: float
    right: str            # "c" | "p"
    expiry: str           # YYYY-MM-DD
    t_years: float
    iv: float
    oi: float
    volume: float = 0.0
    #: Quote fields. Optional because GEX needs none of them -- they exist for
    #: the spread builder, which must price a fill rather than a level.
    bid: float = 0.0
    ask: float = 0.0
    vendor_delta: float | None = None

    @property
    def mid(self) -> float:
        """Mid, or whichever side exists. 0.0 when the contract is unquoted."""
        if self.bid > 0 and self.ask > 0:
            return 0.5 * (self.bid + self.ask)
        return self.ask or self.bid or 0.0

    @property
    def sign(self) -> float:
        return CALL_SIGN if self.right == "c" else PUT_SIGN


@dataclass(frozen=True)
class Chain:
    symbol: str
    asof: str
    spot: float
    contracts: tuple[Contract, ...]
    r: float = 0.0

    @property
    def total_oi(self) -> float:
        return sum(c.oi for c in self.contracts)


def net_gex_at(chain: Chain, spot: float) -> float:
    """Net dealer gamma in dollars-per-1% if the underlying were at `spot`.

    Every contract is RE-PRICED at the candidate spot. This is the function
    that makes the flip correct, and it is deliberately not a sum over a
    pre-computed per-strike profile -- gamma is a function of spot, so those
    are two different quantities and only this one has a meaningful zero.
    """
    total = 0.0
    for c in chain.contracts:
        g = bs_gamma(spot, c.strike, c.t_years, c.iv, chain.r)
        total += c.sign * dollar_gamma(g, c.oi, spot)
    return total


@dataclass(frozen=True)
class StrikeLevel:
    strike: float
    gex: float
    call_gex: float
    put_gex: float
    oi: float


def gex_profile(chain: Chain, spot: float | None = None) -> list[StrikeLevel]:
    """Per-strike net GEX at the CURRENT spot, ascending by strike."""
    s = chain.spot if spot is None else spot
    calls: dict[float, float] = {}
    puts: dict[float, float] = {}
    ois: dict[float, float] = {}
    for c in chain.contracts:
        g = bs_gamma(s, c.strike, c.t_years, c.iv, chain.r)
        dg = dollar_gamma(g, c.oi, s)
        if c.right == "c":
            calls[c.strike] = calls.get(c.strike, 0.0) + CALL_SIGN * dg
        else:
            puts[c.strike] = puts.get(c.strike, 0.0) + PUT_SIGN * dg
        ois[c.strike] = ois.get(c.strike, 0.0) + c.oi
    out = []
    for k in sorted(set(calls) | set(puts)):
        cg, pg = calls.get(k, 0.0), puts.get(k, 0.0)
        out.append(StrikeLevel(strike=k, gex=cg + pg, call_gex=cg, put_gex=pg,
                               oi=ois.get(k, 0.0)))
    return out


def _bisect_zero(chain: Chain, lo: float, hi: float, iters: int = 40) -> float:
    f_lo = net_gex_at(chain, lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = net_gex_at(chain, mid)
        if (f_lo < 0) == (f_mid < 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gamma_flip(chain: Chain, span: float = 0.15, steps: int = 60
               ) -> float | None:
    """The spot at which net dealer gamma changes sign, or None.

    Candidates alternate outward from spot -- spot*(1+d), spot*(1-d), spot*(1+2d)
    ... -- so the NEAREST crossing is found first. That ordering matters: a
    chain can cross more than once and the one that governs today's regime is
    the closest.

    Because candidates alternate sides, a bracket between the last candidate
    and the current one is only a genuine crossing when both lie on the SAME
    side of spot. When the side flips, the anchor is reset to (spot, f0).

    Returns None when net gamma holds one sign across the whole window. That is
    a real answer and far better than a fabricated middle strike.
    """
    # A chain with no gamma anywhere has no flip. Without this guard an empty
    # or all-expired chain reports net GEX of exactly 0.0 at spot and the
    # "already at the root" branch below hands back SPOT ITSELF as the flip --
    # a fabricated level, dressed as a measured one, on the emptiest possible
    # input. Checked before the zero test, because the two are indistinguishable
    # from the value alone.
    if not any(bs_gamma(chain.spot, c.strike, c.t_years, c.iv, chain.r) > 0
               and c.oi > 0 for c in chain.contracts):
        return None

    f0 = net_gex_at(chain, chain.spot)
    if f0 == 0.0:
        return round(chain.spot, 2)

    prev_s, prev_f = chain.spot, f0
    for i in range(1, steps + 1):
        d = span * i / steps
        for s in (chain.spot * (1.0 + d), chain.spot * (1.0 - d)):
            f = net_gex_at(chain, s)
            # Only a bracket on one side of spot is a real crossing.
            if (s - chain.spot) * (prev_s - chain.spot) < 0:
                prev_s, prev_f = chain.spot, f0
            if (prev_f < 0 <= f) or (prev_f > 0 >= f):
                lo, hi = (prev_s, s) if prev_s < s else (s, prev_s)
                return round(_bisect_zero(chain, lo, hi), 2)
            prev_s, prev_f = s, f
    return None


def call_wall(profile: list[StrikeLevel], spot: float) -> StrikeLevel | None:
    """Largest POSITIVE gamma strike AT OR ABOVE spot. Resistance.

    Both constraints are load-bearing. Without the side constraint a global
    maximum renders resistance below the current price; without the sign
    constraint a chain with no positive strike above spot still reports a wall,
    which is a level that does not exist.
    """
    above = [p for p in profile if p.strike >= spot]
    if not above:
        return None
    best = max(above, key=lambda p: p.gex)
    return best if best.gex > 0 else None


def put_wall(profile: list[StrikeLevel], spot: float) -> StrikeLevel | None:
    """Most NEGATIVE gamma strike AT OR BELOW spot. Support."""
    below = [p for p in profile if p.strike <= spot]
    if not below:
        return None
    best = min(below, key=lambda p: p.gex)
    return best if best.gex < 0 else None


def control_node(profile: list[StrikeLevel]) -> float | None:
    """Strike with the largest ABSOLUTE gamma -- where hedging concentrates."""
    if not profile:
        return None
    return max(profile, key=lambda p: abs(p.gex)).strike


def coverage_warning(chain: Chain) -> str | None:
    """Flag a chain whose call and put strikes were not sampled alike.

    Net GEX is a DIFFERENCE between two sums. Sample thirteen call strikes and
    five put strikes and the total is positive by construction, whatever the
    market is actually doing -- and it will look exactly like a real reading.

    This is not hypothetical: it is precisely what a strike-banded pull from a
    chain with eighty strikes per right produces if the band is applied to one
    side and not the other. The report carries the warning rather than the
    caller being trusted to remember.
    """
    # A row counts as coverage only if it can CONTRIBUTE gamma. Open interest
    # is not enough: Robinhood returns a null implied volatility on deep-ITM
    # strikes, and `bs_gamma` correctly returns 0.0 for those -- so a strike
    # with 175 contracts of open interest and no IV is, arithmetically, not
    # there. Counting it as coverage lets a pull that looks symmetric in the
    # request produce a lopsided chain in the arithmetic. Both halves of this
    # were live in the first real CSCO pull: a null-IV call at 100 and a
    # zero-OI put at 120, one at each end.
    def contributing(right: str) -> set[float]:
        return {c.strike for c in chain.contracts
                if c.right == right and c.oi > 0 and c.iv > 0}

    call_ks, put_ks = contributing("c"), contributing("p")
    if not call_ks or not put_ks:
        missing = "puts" if not put_ks else "calls"
        return (f"NO {missing.upper()} in this chain -- net GEX is one-sided "
                f"and its sign is an artefact of the sample, not a reading")
    one_sided = (call_ks - put_ks) | (put_ks - call_ks)
    if not one_sided:
        return None

    # COUNTING STRIKES IS THE WRONG TEST, and it took a full chain to show it.
    # Every real chain has far-wing strikes quoted on one side only -- 56 of
    # SPY's 354 call strikes had no matching put open interest. Those strikes
    # are deep out of the money and contribute almost no gamma, so their
    # absence cannot move the reading. A count-based warning fires on every
    # complete chain, and since `daily.rank()` EXCLUDES a warned chain, that
    # would silently empty the shortlist forever.
    #
    # What matters is whether the one-sided strikes carry enough gamma to move
    # the number. So the test is a SHARE OF TOTAL ABSOLUTE GEX, which is the
    # quantity actually at risk of being biased.
    total = 0.0
    lopsided = 0.0
    for c in chain.contracts:
        g = bs_gamma(chain.spot, c.strike, c.t_years, c.iv, chain.r)
        weight = abs(dollar_gamma(g, c.oi, chain.spot))
        total += weight
        if c.strike in one_sided:
            lopsided += weight
    if total <= 0:
        return None
    share = lopsided / total
    if share < ONE_SIDED_TOLERANCE:
        return None
    return (f"ASYMMETRIC STRIKE COVERAGE: {len(call_ks)} call strikes vs "
            f"{len(put_ks)} put strikes, and the one-sided strikes carry "
            f"{share:.0%} of the chain's gamma. Net GEX is a difference of two "
            f"sums, so an uneven sample biases it toward the better-sampled "
            f"side. Pull both rights over the same strike band before reading "
            f"the level.")


@dataclass(frozen=True)
class GexReport:
    symbol: str
    asof: str
    spot: float
    net_gex: float
    regime: str
    flip: float | None
    call_wall: StrikeLevel | None
    put_wall: StrikeLevel | None
    control_node: float | None
    total_oi: float
    n_contracts: int
    profile: list[StrikeLevel]
    warning: str | None = None

    def render(self) -> str:
        def lvl(x: StrikeLevel | None) -> str:
            if x is None:
                return "none in range"
            pct = (x.strike / self.spot - 1.0) * 100.0
            return f"{x.strike:,.2f} ({pct:+.2f}%)  {x.gex/1e6:+,.1f}M/1%"

        flip = "none in range" if self.flip is None else (
            f"{self.flip:,.2f} ({(self.flip/self.spot - 1)*100:+.2f}%)")
        out = [
            f"{self.symbol}  spot {self.spot:,.2f}   snapshot {self.asof}",
            f"  net GEX     {self.net_gex/1e6:+,.1f}M per 1%   "
            f"regime {self.regime.upper()}",
            f"  gamma flip  {flip}",
            f"  call wall   {lvl(self.call_wall)}",
            f"  put wall    {lvl(self.put_wall)}",
            f"  control     {self.control_node if self.control_node else 'n/a'}",
            f"  chain       {self.n_contracts} contracts, "
            f"{self.total_oi:,.0f} total OI",
        ]
        if self.regime == "positive":
            out.append("  reading     dealers buy dips / sell rips -- moves "
                       "get DAMPENED, price pins toward the control node")
        else:
            out.append("  reading     dealers sell dips / buy rips -- moves "
                       "get AMPLIFIED, fading is dangerous")
        out.append("  NOTE        this is a SIZE signal, not a direction "
                   "signal (FINDINGS.md #7)")
        if self.warning:
            out.append(f"  WARNING     {self.warning}")
        return "\n".join(out)


def analyze(chain: Chain) -> GexReport:
    """Everything the chain supports, and nothing it does not."""
    profile = gex_profile(chain)
    net = net_gex_at(chain, chain.spot)
    return GexReport(
        symbol=chain.symbol, asof=chain.asof, spot=chain.spot,
        net_gex=net,
        # REGIME IS THE SIGN OF NET GEX AT SPOT. Never spot vs the flip.
        regime="positive" if net >= 0 else "negative",
        flip=gamma_flip(chain),
        call_wall=call_wall(profile, chain.spot),
        put_wall=put_wall(profile, chain.spot),
        control_node=control_node(profile),
        total_oi=chain.total_oi, n_contracts=len(chain.contracts),
        profile=profile, warning=coverage_warning(chain),
    )


# ---------------------------------------------------------------------------
# chain snapshots
# ---------------------------------------------------------------------------

def years_to(expiry: str, asof: str | None = None) -> float:
    """Calendar years from `asof` to `expiry`. 1 day = 1/365, per convention."""
    end = date.fromisoformat(expiry)
    start = (datetime.fromisoformat(asof).date() if asof else date.today())
    return max(0.0, (end - start).days / DAYS_PER_YEAR)


def load_chain(path: str | Path) -> Chain:
    """Read a chain snapshot, computing `t_years` when it is not stored.

    Rows with zero open interest are KEPT: a strike with no OI contributes
    exactly zero to GEX by construction, and dropping it here would silently
    change which strikes appear in the profile.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("symbol", "asof", "spot", "contracts"):
        if key not in data:
            raise ValueError(f"chain snapshot {path} is missing {key!r}")
    if not (float(data["spot"]) > 0):
        raise ValueError(f"chain snapshot {path} has a non-positive spot")

    rows = []
    for c in data["contracts"]:
        right = str(c["right"]).lower()[0]
        if right not in ("c", "p"):
            raise ValueError(f"bad right {c['right']!r} in {path}")
        t = c.get("t_years")
        if t is None:
            t = years_to(c["expiry"], data["asof"])
        rows.append(Contract(
            strike=float(c["strike"]), right=right, expiry=str(c["expiry"]),
            t_years=float(t), iv=float(c.get("iv") or 0.0),
            oi=float(c.get("oi") or 0.0), volume=float(c.get("volume") or 0.0)))
    return Chain(symbol=str(data["symbol"]), asof=str(data["asof"]),
                 spot=float(data["spot"]), contracts=tuple(rows),
                 r=float(data.get("r", 0.0)))


def main(argv: list[str] | None = None) -> int:
    import sys
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(__doc__.strip().splitlines()[2])
        return 2
    for path in args:
        print(analyze(load_chain(path)).render())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
