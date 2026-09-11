"""The daily read: what the gamma map says about TODAY, per symbol.

    python -m options.daily data/chains/*.json

WHAT THIS OUTPUTS, AND WHAT IT DELIBERATELY DOES NOT
----------------------------------------------------
`FINDINGS.md` §7 measured dealer gamma against 247 sessions and found it
predicts SIZE (t = -6.57) and NOT DIRECTION (t = -1.22). So every `DailyRead`
carries a size call and `direction is None`. That is not modesty, it is the
measurement; `verify_daily.py` pins it so a direction claim cannot be added
here without someone deliberately deleting a test that names the reason.

If a direction rule ever earns its place, it arrives as a SEPARATE input with
its own evidence, and this module records both rather than blending them.

THE UNIT THAT MAKES LEVELS COMPARABLE. A wall 4% away means something different
on a 20-vol name and an 80-vol one. Everything here is therefore expressed in
EXPECTED MOVES -- `spot * iv * sqrt(T)` for one session -- so "the call wall is
0.6 expected moves away" reads the same across the book. A wall inside one
expected move is reachable today; one three expected moves away is scenery, and
the pin story people tell about it is not a story about today.

THE SIZE CALL, in the two regimes:

  POSITIVE gamma  dealers buy dips and sell rips, so realised movement is
                  SUPPRESSED. Strongest when spot sits between two walls that
                  are close in expected-move terms -- the box is tight.
  NEGATIVE gamma  dealers sell dips and buy rips, so movement is AMPLIFIED.
                  Strongest when spot has room, i.e. no wall within reach.

`conviction` is the strength of that call in [0, 1]. It is a RANKING device for
choosing which two or three names to look at, not a probability, and it is
named to avoid suggesting otherwise.

Standard library only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .gex import Chain, GexReport, analyze

#: One trading session, in years. 1 trading day = 1/252 (the convention).
ONE_SESSION = 1.0 / 252.0

#: A wall further than this many expected moves is not in play today.
REACHABLE_EM = 1.5

#: Written by `research/em_base_rate.py`. Absent is a valid state, and every
#: consumer says so out loud rather than substituting 50%.
DEFAULT_BASE_RATES = Path("data/em_base_rate.json")


def expected_move(spot: float, iv: float, t_years: float = ONE_SESSION) -> float:
    """One-sigma move over `t_years`, in dollars: spot * iv * sqrt(T).

    The lognormal one-sigma, which is the number every options desk quotes as
    "the expected move". It is not a forecast of direction and not a bound.
    """
    if spot <= 0 or iv <= 0 or t_years <= 0:
        return 0.0
    return spot * iv * math.sqrt(t_years)


def atm_iv(chain: Chain) -> float:
    """Implied vol at the money: average of the call and put nearest spot.

    Averaging the two rights rather than taking whichever is closer avoids a
    reading that jumps when spot crosses a strike, and both are quoted off the
    same forward so they should agree to within the skew's slope.
    """
    best: dict[str, tuple[float, float]] = {}
    for c in chain.contracts:
        if c.iv <= 0:
            continue
        d = abs(c.strike - chain.spot)
        if c.right not in best or d < best[c.right][0]:
            best[c.right] = (d, c.iv)
    ivs = [iv for _d, iv in best.values()]
    return sum(ivs) / len(ivs) if ivs else 0.0


@dataclass(frozen=True)
class DailyRead:
    symbol: str
    asof: str
    spot: float
    regime: str
    net_gex: float
    atm_iv: float
    expected_move: float
    #: distance to each level in EXPECTED MOVES, signed (+ above spot)
    call_wall_em: float | None
    put_wall_em: float | None
    flip_em: float | None
    size_call: str                 # "compression" | "expansion"
    conviction: float              # [0, 1], a ranking device
    notes: list[str] = field(default_factory=list)
    #: Always None. See the module docstring and verify_daily.py [4].
    direction: None = None

    def render(self) -> str:
        def em(x: float | None) -> str:
            return "none in range" if x is None else f"{x:+.2f} EM"
        out = [
            f"{self.symbol:<6} {self.spot:>9,.2f}  {self.regime.upper():<8} "
            f"{self.size_call.upper():<12} conv {self.conviction:.2f}",
            f"       EM {self.expected_move:,.2f} ({self.expected_move/self.spot:.2%})"
            f"  ATM IV {self.atm_iv:.1%}   net GEX {self.net_gex/1e6:+,.1f}M",
            f"       call wall {em(self.call_wall_em)}   "
            f"put wall {em(self.put_wall_em)}   flip {em(self.flip_em)}",
        ]
        out += [f"       note: {n}" for n in self.notes]
        return "\n".join(out)


def read(chain: Chain, report: GexReport | None = None) -> DailyRead:
    """The day's read for one symbol."""
    rep = report or analyze(chain)
    iv = atm_iv(chain)
    em = expected_move(chain.spot, iv)

    def to_em(level: float | None) -> float | None:
        if level is None or em <= 0:
            return None
        return (level - chain.spot) / em

    cw = to_em(rep.call_wall.strike if rep.call_wall else None)
    pw = to_em(rep.put_wall.strike if rep.put_wall else None)
    flip = to_em(rep.flip)

    notes: list[str] = []
    if rep.warning:
        notes.append(rep.warning)
    if em <= 0:
        notes.append("no usable ATM implied vol -- every level below is "
                     "unscaled and the conviction is zero")

    # Compression needs a BOX: walls on both sides, close enough to matter.
    # Expansion needs ROOM: no wall within reach.
    if rep.regime == "positive":
        size_call = "compression"
        reach = [abs(x) for x in (cw, pw) if x is not None]
        if not reach:
            conviction = 0.0
            notes.append("positive gamma but no wall in range -- nothing to "
                         "pin against, so the compression call is unsupported")
        else:
            # Tightest when BOTH walls are close; a one-sided box is weaker.
            widest = max(reach) if len(reach) == 2 else REACHABLE_EM
            conviction = max(0.0, min(1.0, 1.0 - widest / (2.0 * REACHABLE_EM)))
            if len(reach) == 1:
                conviction *= 0.5
                notes.append("only one wall in range -- half conviction, the "
                             "other side is unbounded")
    else:
        size_call = "expansion"
        nearest = min((abs(x) for x in (cw, pw) if x is not None),
                      default=None)
        if nearest is None:
            conviction = 1.0
        else:
            conviction = max(0.0, min(1.0, nearest / REACHABLE_EM))
            if nearest < 0.5:
                notes.append("negative gamma but a wall is within half an "
                             "expected move -- the amplification has somewhere "
                             "to stop almost immediately")
    if em <= 0:
        conviction = 0.0

    return DailyRead(
        symbol=chain.symbol, asof=chain.asof, spot=chain.spot,
        regime=rep.regime, net_gex=rep.net_gex, atm_iv=iv, expected_move=em,
        call_wall_em=cw, put_wall_em=pw, flip_em=flip,
        size_call=size_call, conviction=round(conviction, 4), notes=notes)


@dataclass(frozen=True)
class BaseRates:
    """P(realised < 1 expected move), by implied-vol bucket.

    Measured, not assumed: `research/em_base_rate.py` over 5,029 SPY sessions
    2006-2026 puts the overall figure at **80.8%**, against a lognormal 68.3%,
    and it RISES as vol falls -- 87.3% under 13.9 vol, 74.7% above 22.6.

    That conditionality is the whole reason this class exists. Grading a
    compression call against a flat null credits the signal for something
    implied vol alone explains, and the effect is worth 12 points of hit rate
    between the extreme buckets.
    """

    edges: tuple[float, ...]
    p_compress: tuple[float, ...]
    counts: tuple[int, ...]
    overall: float
    source: str
    n: int

    @classmethod
    def load(cls, path: str | Path | None = None) -> "BaseRates | None":
        p = Path(path or DEFAULT_BASE_RATES)
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            edges=tuple(d["iv_edges"]),
            p_compress=tuple(b["p_compress"] for b in d["buckets"]),
            counts=tuple(b["n"] for b in d["buckets"]),
            overall=float(d["overall_p_compress"]),
            source=str(d.get("source", "?")), n=int(d.get("n", 0)))

    def bucket(self, iv: float) -> int:
        return sum(1 for e in self.edges if iv > e)

    def p_for(self, iv: float, size_call: str) -> float:
        """The null hit rate for THIS side at THIS vol.

        Compression and expansion are complements, so the expansion null is
        1 - P(compress). Reporting an expansion hit rate against 50% would
        flatter it by thirty points.
        """
        p = self.p_compress[self.bucket(iv)]
        return p if size_call == "compression" else 1.0 - p


def room(read_: DailyRead, rates: "BaseRates | None") -> float:
    """How much the null leaves unexplained for this call: 1 - base rate.

    A compression call in a 12-vol name is right 87% of the time by doing
    nothing, so there is almost nothing for a signal to add. An expansion call
    has a 19% null and therefore room. Ranking on conviction alone treats those
    as equivalent, which is how a shortlist fills up with calls that cannot
    teach you anything.
    """
    if rates is None:
        return 1.0
    return max(0.0, 1.0 - rates.p_for(read_.atm_iv, read_.size_call))


def rank(reads: Sequence[DailyRead], n: int = 3,
         rates: "BaseRates | None" = None) -> list[DailyRead]:
    """The `n` reads with the most to teach, not merely the tidiest geometry.

    Score is `conviction x room`: how sure the gamma map is, times how much of
    the outcome the base rate does not already explain. With `rates` absent it
    degrades to conviction alone and the caller is told, because silently
    ranking on half the criterion is worse than ranking on one.

    Ties break on symbol so two runs on the same snapshots agree.

    A read carrying a coverage warning is EXCLUDED, not merely demoted: its
    regime may be an artefact of how the chain was pulled, and a ranking that
    quietly includes one is exactly how a sampling bug reaches a trade.
    """
    clean = [r for r in reads
             if r.conviction > 0 and not any("COVERAGE" in x or "NO PUTS" in x
                                             or "NO CALLS" in x for x in r.notes)]
    return sorted(clean,
                  key=lambda r: (-(r.conviction * room(r, rates)), r.symbol))[:n]


def grade(prior: DailyRead, realized_move: float,
          rates: "BaseRates | None" = None) -> dict:
    """Score a read against what the session actually did.

    `realized_move` is the ABSOLUTE move in dollars over the session the read
    was made for. The size call is right when compression is followed by less
    than one expected move, or expansion by more than one.

    THE HIT ALONE MEANS NOTHING, so the record carries its own null. A
    compression call that lands is unremarkable at an 87% base rate and
    striking at 19%, and the only way that distinction survives into an
    aggregate months from now is if the base rate is stamped on the record at
    the time it is made -- along with the sample it came from, so a later
    re-measurement is visible as a change rather than blended in.
    """
    if prior.expected_move <= 0:
        return {"symbol": prior.symbol, "graded": False,
                "reason": "no expected move to grade against"}
    ratio = abs(realized_move) / prior.expected_move
    hit = (ratio < 1.0) if prior.size_call == "compression" else (ratio > 1.0)
    out = {
        "symbol": prior.symbol, "asof": prior.asof, "graded": True,
        "size_call": prior.size_call, "conviction": prior.conviction,
        "atm_iv": round(prior.atm_iv, 6),
        "expected_move": round(prior.expected_move, 4),
        "realized_move": round(abs(realized_move), 4),
        "ratio": round(ratio, 4), "hit": hit,
        "rule": "abs(realized) vs 1.0 x expected_move, 1 session",
    }
    if rates is None:
        out["base_rate"] = None
        out["base_rate_source"] = "NONE -- this record cannot be scored against "\
                                  "its null later without one"
    else:
        out["base_rate"] = round(rates.p_for(prior.atm_iv, prior.size_call), 6)
        out["base_rate_source"] = rates.source
    return out


def aggregate(records: Sequence[dict]) -> dict:
    """Hit rate against the null, over many graded records.

    This is where a size call is actually judged. One graded day says nothing;
    what matters is whether the hit rate beats the base rates those same days
    carried, and by enough to clear their combined standard error.

    Records without a base rate are counted and EXCLUDED rather than folded in
    at 50% -- a missing null is not a neutral null.
    """
    scored = [r for r in records
              if r.get("graded") and r.get("base_rate") is not None]
    skipped = len(records) - len(scored)
    n = len(scored)
    if n == 0:
        return {"n": 0, "skipped_no_base_rate": skipped}
    hits = sum(1 for r in scored if r["hit"])
    observed = hits / n
    expected = sum(r["base_rate"] for r in scored) / n
    # Variance of the sum of independent Bernoullis with their own p.
    var = sum(r["base_rate"] * (1 - r["base_rate"]) for r in scored)
    se = math.sqrt(var) / n if var > 0 else float("nan")
    z = (observed - expected) / se if se and se == se and se > 0 else float("nan")
    return {
        "n": n, "skipped_no_base_rate": skipped,
        "hits": hits, "observed": round(observed, 4),
        "expected_from_base_rates": round(expected, 4),
        "edge": round(observed - expected, 4),
        "se": round(se, 4) if se == se else None,
        "z": round(z, 2) if z == z else None,
        "verdict": ("too few" if n < 30 else
                    "beats its null" if z > 2.0 else
                    "below its null" if z < -2.0 else
                    "indistinguishable from its null"),
    }


def main(argv: list[str] | None = None) -> int:
    import sys
    from .gex import load_chain
    paths = argv if argv is not None else sys.argv[1:]
    if not paths:
        print("usage: python -m options.daily data/chains/*.json")
        return 2
    reads = [read(load_chain(p)) for p in paths]
    print(f"{'sym':<6} {'spot':>9}  {'regime':<8} {'size call':<12} conv")
    for r in reads:
        print(r.render())
    top = rank(reads)
    print(f"\nshortlist ({len(top)} of {len(reads)}): "
          + (", ".join(f"{r.symbol} {r.conviction:.2f}" for r in top) or "none"))
    if len(top) < len(reads):
        print("  names are dropped for zero conviction or a coverage warning; "
              "see the notes above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
