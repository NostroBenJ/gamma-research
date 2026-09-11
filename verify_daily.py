"""Verification for `options/daily.py`. Run: `python verify_daily.py`

The load-bearing check is [4]: **the daily read must not carry a direction.**
`FINDINGS.md` §7 measured gamma against 247 sessions and found it predicts size
(t = -6.57) and not direction (t = -1.22). A test that names that reason is the
only thing standing between the measured result and a plausible-sounding
`direction = "long"` appearing here one afternoon.

[6] is the other one that matters: a read built on a chain with asymmetric
strike coverage must be EXCLUDED from the shortlist, not merely ranked lower.
Its regime may be an artefact of the pull, and a shortlist is the last step
before someone acts.
"""

from __future__ import annotations

import math
import sys

from options.daily import (
    ONE_SESSION,
    REACHABLE_EM,
    BaseRates,
    DailyRead,
    aggregate,
    room,
    atm_iv,
    expected_move,
    grade,
    rank,
    read,
)
from options.gex import Chain, Contract, analyze

PASS, FAIL = 0, 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}  {detail}")


T30 = 30 / 365.0


def chain(spot=100.0, call_oi=None, put_oi=None, iv=0.30, symbol="TEST"):
    """A symmetric five-strike chain; OI per side controls where walls land."""
    call_oi = call_oi or {90: 100, 95: 100, 100: 500, 105: 5000, 110: 200}
    put_oi = put_oi or {90: 200, 95: 5000, 100: 500, 105: 100, 110: 100}
    rows = []
    for k in (90.0, 95.0, 100.0, 105.0, 110.0):
        rows.append(Contract(k, "c", "2026-10-09", T30, iv, call_oi[int(k)]))
        rows.append(Contract(k, "p", "2026-10-09", T30, iv, put_oi[int(k)]))
    return Chain(symbol, "2026-09-09T16:00:00", spot, tuple(rows), r=0.0)


def main() -> int:
    print("[1] expected move is spot * iv * sqrt(T), in the right units")
    em = expected_move(100.0, 0.20, ONE_SESSION)
    check("1 session uses 1/252, not 1/365",
          abs(em - 100.0 * 0.20 * math.sqrt(1 / 252)) < 1e-12, f"{em:.6f}")
    check("a 20-vol name moves ~1.26% in a session",
          abs(em / 100.0 - 0.0126) < 0.0002, f"{em/100:.4%}")
    check("scales with sqrt(T): 4 sessions is 2x one",
          abs(expected_move(100, 0.2, 4 * ONE_SESSION) - 2 * em) < 1e-12)
    check("scales linearly in iv",
          abs(expected_move(100, 0.4, ONE_SESSION) - 2 * em) < 1e-12)
    for label, args in (("zero iv", (100.0, 0.0)), ("zero spot", (0.0, 0.2)),
                        ("negative iv", (100.0, -0.2))):
        check(f"{label} -> 0.0", expected_move(*args) == 0.0)

    print("\n[2] ATM iv averages the two rights nearest spot")
    rows = (Contract(100.0, "c", "2026-10-09", T30, 0.40, 10),
            Contract(100.0, "p", "2026-10-09", T30, 0.20, 10),
            Contract(150.0, "c", "2026-10-09", T30, 0.90, 10))
    check("averages the call and put at the money",
          abs(atm_iv(Chain("X", "x", 100.0, rows)) - 0.30) < 1e-12,
          str(atm_iv(Chain("X", "x", 100.0, rows))))
    check("a far strike does not contaminate it",
          atm_iv(Chain("X", "x", 100.0, rows)) < 0.5)
    check("zero-iv rows are ignored, not averaged in",
          abs(atm_iv(Chain("X", "x", 100.0, rows + (
              Contract(101.0, "c", "2026-10-09", T30, 0.0, 10),))) - 0.30) < 1e-12)

    print("\n[3] levels are expressed in EXPECTED MOVES, signed from spot")
    r = read(chain(100.0))
    check("call wall is ABOVE spot -> positive EM", r.call_wall_em > 0,
          str(r.call_wall_em))
    check("put wall is BELOW spot -> negative EM", r.put_wall_em < 0,
          str(r.put_wall_em))
    # 105 strike on a 30-vol name: EM = 100*0.30*sqrt(1/252) = 1.89, so 5/1.89.
    em1 = expected_move(100.0, 0.30)
    check("the call wall distance is exactly (strike - spot)/EM",
          abs(r.call_wall_em - 5.0 / em1) < 1e-9,
          f"{r.call_wall_em:.4f} vs {5.0/em1:.4f}")
    check("a higher-vol name reports the SAME wall as fewer EM away",
          read(chain(100.0, iv=0.60)).call_wall_em < r.call_wall_em)

    print("\n[4] THE READ CARRIES NO DIRECTION -- FINDINGS.md #7")
    check("direction is None", r.direction is None, repr(r.direction))
    check("and the field cannot hold anything else",
          DailyRead.__annotations__["direction"] == "None",
          DailyRead.__annotations__["direction"])
    check("the size call is one of the two measured outcomes",
          r.size_call in ("compression", "expansion"), r.size_call)
    check("nothing in the rendered read says long or short",
          not any(w in r.render().lower() for w in (" long", " short", "bullish",
                                                    "bearish")),
          r.render())

    print("\n[5] the size call follows the regime, and conviction follows room")
    pos = read(chain(100.0))
    check("positive gamma -> compression", pos.regime == "positive"
          and pos.size_call == "compression", f"{pos.regime}/{pos.size_call}")
    # Put-heavy book -> net short gamma -> expansion.
    neg = read(chain(100.0, call_oi={90: 10, 95: 10, 100: 10, 105: 10, 110: 10},
                     put_oi={90: 9000, 95: 9000, 100: 9000, 105: 9000, 110: 9000}))
    check("put-heavy book -> negative gamma -> expansion",
          neg.regime == "negative" and neg.size_call == "expansion",
          f"{neg.regime}/{neg.size_call}")
    check("conviction is inside [0, 1] on both",
          0.0 <= pos.conviction <= 1.0 and 0.0 <= neg.conviction <= 1.0)
    # A tighter box (walls closer in EM terms) must read as more conviction.
    tight = read(chain(100.0, iv=0.90))     # high vol -> walls are fewer EM away
    wide = read(chain(100.0, iv=0.12))      # low vol -> walls are many EM away
    check("a tighter box in EM terms gives HIGHER compression conviction",
          tight.conviction > wide.conviction,
          f"tight {tight.conviction} vs wide {wide.conviction}")
    check("walls out of reach give zero conviction, with the reason",
          wide.conviction == 0.0 or any("unsupported" in n or "half conviction"
                                        in n for n in wide.notes),
          f"{wide.conviction} {wide.notes}")

    print("\n[6] a chain with lopsided coverage cannot reach the shortlist")
    lop = Chain("LOP", "x", 100.0, tuple(
        [Contract(float(k), "c", "2026-10-09", T30, 0.30, 5000)
         for k in (95, 100, 105, 110)]
        + [Contract(100.0, "p", "2026-10-09", T30, 0.30, 5000)]), r=0.0)
    lr = read(lop)
    check("the coverage warning reaches the read's notes",
          any("COVERAGE" in n for n in lr.notes), str(lr.notes))
    good = read(chain(100.0, symbol="GOOD"))
    top = rank([lr, good], n=3)
    check("it is EXCLUDED from the shortlist, not just demoted",
          all(x.symbol != "LOP" for x in top), [x.symbol for x in top])
    check("the clean name still ranks", any(x.symbol == "GOOD" for x in top))

    print("\n[7] ranking is deterministic")
    a = read(chain(100.0, symbol="AAA"))
    b = read(chain(100.0, symbol="BBB"))
    check("ties break on symbol, so two runs agree",
          [x.symbol for x in rank([a, b], 2)] == [x.symbol for x in rank([b, a], 2)],
          f"{[x.symbol for x in rank([a, b], 2)]}")
    check("n is respected", len(rank([a, b], 1)) == 1)
    check("zero-conviction reads are dropped",
          all(x.conviction > 0 for x in rank([a, b, wide], 5)))

    print("\n[8] grading is honest about which rule produced it")
    comp = read(chain(100.0))
    g_small = grade(comp, 0.5 * comp.expected_move)
    g_big = grade(comp, 2.0 * comp.expected_move)
    check("compression + small move = hit", g_small["hit"] is True)
    check("compression + big move = miss", g_big["hit"] is False)
    exp_read = neg
    check("expansion + big move = hit",
          grade(exp_read, 2.0 * exp_read.expected_move)["hit"] is True)
    check("expansion + small move = miss",
          grade(exp_read, 0.2 * exp_read.expected_move)["hit"] is False)
    check("the grading rule is stamped on every record",
          "expected_move" in g_small["rule"], g_small["rule"])
    check("sign of the realised move does not matter -- it is a SIZE call",
          grade(comp, 0.5 * comp.expected_move)["hit"]
          == grade(comp, -0.5 * comp.expected_move)["hit"])
    check("an unscalable read is not graded, and says why",
          grade(DailyRead("X", "x", 100.0, "positive", 0.0, 0.0, 0.0,
                          None, None, None, "compression", 0.0),
                1.0)["graded"] is False)

    print("\n[9] the null is CONDITIONAL on implied vol")
    br = BaseRates(edges=(0.14, 0.17, 0.23),
                   p_compress=(0.873, 0.826, 0.785, 0.747),
                   counts=(1262, 1255, 1256, 1256), overall=0.808,
                   source="fixture", n=5029)
    check("a low-vol compression call has a HIGHER null",
          br.p_for(0.10, "compression") > br.p_for(0.30, "compression"),
          f"{br.p_for(0.10,'compression')} vs {br.p_for(0.30,'compression')}")
    check("compression and expansion nulls are complements",
          abs(br.p_for(0.10, "compression") + br.p_for(0.10, "expansion") - 1.0)
          < 1e-12)
    check("the expansion null is the SMALL one -- 12.7% at 10 vol",
          abs(br.p_for(0.10, "expansion") - 0.127) < 1e-9,
          str(br.p_for(0.10, "expansion")))
    check("bucket edges are inclusive-below",
          br.bucket(0.139) == 0 and br.bucket(0.141) == 1)
    check("the real emitted table loads", BaseRates.load() is not None)

    print("\n[10] grade stamps the null on the record, or says it could not")
    comp = read(chain(100.0))
    g = grade(comp, 0.5 * comp.expected_move, br)
    check("the base rate travels with the record", g["base_rate"] is not None)
    check("it is the rate for THIS side at THIS vol",
          abs(g["base_rate"] - br.p_for(comp.atm_iv, "compression")) < 1e-12)
    check("the source is stamped too, so a re-measure is visible",
          g["base_rate_source"] == "fixture", g["base_rate_source"])
    g0 = grade(comp, 0.5 * comp.expected_move, None)
    check("without a table the record says so rather than assuming 50%",
          g0["base_rate"] is None and "NONE" in g0["base_rate_source"],
          str(g0["base_rate_source"]))

    print("\n[11] aggregate judges hits against their OWN nulls, not 50%")
    # 30 compression calls at an 87.3% null, 26 of which hit = 86.7%. That is
    # a coin-flip-beating 87% that has found NOTHING, and the aggregate must
    # say so rather than celebrating it.
    lowvol = read(chain(100.0, iv=0.10))
    recs = [grade(lowvol, (0.5 if i < 26 else 1.5) * lowvol.expected_move, br)
            for i in range(30)]
    agg = aggregate(recs)
    # The record rounds to 4dp, so the tolerance is the rounding, not 1e-9.
    check("observed is reported", abs(agg["observed"] - 26 / 30) < 1e-4,
          str(agg["observed"]))
    check("expected comes from the base rates, not 0.5",
          agg["expected_from_base_rates"] > 0.8,
          str(agg["expected_from_base_rates"]))
    check("an 87% hit rate against an 87% null is NOT an edge",
          agg["verdict"] == "indistinguishable from its null", agg["verdict"])
    all_hit = [grade(lowvol, 0.1 * lowvol.expected_move, br) for _ in range(200)]
    check("but a genuine edge is detected", aggregate(all_hit)["z"] > 2.0,
          str(aggregate(all_hit)["z"]))
    check("records with no base rate are excluded, not folded in at 50%",
          aggregate([g0])["n"] == 0
          and aggregate([g0])["skipped_no_base_rate"] == 1)
    check("under 30 records the verdict refuses to call it",
          aggregate(recs[:10])["verdict"] == "too few")

    print("\n[12] rank prefers the call with ROOM, not the tidiest geometry")
    # Same conviction, opposite sides. Expansion has a 19% null and therefore
    # something to learn; compression at low vol has 87% and almost nothing.
    comp_lo = read(chain(100.0, iv=0.10, symbol="COMPLO"))
    exp_hi = read(chain(100.0, symbol="EXPHI",
                        call_oi={90: 10, 95: 10, 100: 10, 105: 10, 110: 10},
                        put_oi={90: 9000, 95: 9000, 100: 9000, 105: 9000,
                                110: 9000}))
    check("the fixture really is one of each side",
          comp_lo.size_call == "compression" and exp_hi.size_call == "expansion",
          f"{comp_lo.size_call}/{exp_hi.size_call}")
    check("room is larger for the expansion call",
          room(exp_hi, br) > room(comp_lo, br),
          f"{room(exp_hi, br):.3f} vs {room(comp_lo, br):.3f}")
    check("with no table, room is neutral and ranking is conviction alone",
          room(exp_hi, None) == 1.0 and room(comp_lo, None) == 1.0)
    ordered = rank([comp_lo, exp_hi], 2, br)
    check("ranking is still deterministic with rates supplied",
          [x.symbol for x in ordered]
          == [x.symbol for x in rank([exp_hi, comp_lo], 2, br)])
    check("a coverage-warned read is STILL excluded once rates are in play",
          all(x.symbol != "LOP" for x in rank([lr, comp_lo, exp_hi], 3, br)))

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
