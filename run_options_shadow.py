"""The daily options loop: read the map, journal it, grade what came before.

    python run_options_shadow.py            # read today, grade yesterday
    python run_options_shadow.py --report   # the standing record, no writes
    python run_options_shadow.py --symbols SPY INTC

TRADES NOTHING, ON PURPOSE. `research/gex_regime_test.py` measured the gamma
regime on 216 pairs of purchased SPX data and found +3.4% at z = +0.76, in a
split that could only have resolved 10.7 points. The honest reading is "cannot
tell", and the only instrument that resolves it is more sessions. So this runs
daily, writes down what it would have said, and grades itself -- which is the
one thing that makes tomorrow's answer better than today's.

WHY A SEPARATE JOURNAL from `bot/journal.py`. That one's `Decision` is
trade-shaped: direction, shares, stop, target. A daily read has no direction
(see `options/daily.py`) and no position. Forcing it into that schema would
mean writing `direction: 0` five thousand times and inviting someone to
aggregate a size call as if it were a trade record.

THE GRADING IS ALWAYS RETROSPECTIVE. A read journalled for session D is graded
only once a LATER session exists in the chain store, using the spot the
recorder stored then. Nothing grades itself on the day it is made, which is the
structural reason this loop cannot leak the outcome into the signal.

EVERY GRADED RECORD CARRIES ITS OWN NULL, stamped at the time it is made
(`options/daily.py::grade`). A compression call that lands is unremarkable at
an 87% base rate and striking at 19%, and the aggregate is only interpretable
because the base rate travelled with each record rather than being applied to
the pile afterwards.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

from options.daily import BaseRates, aggregate, grade, rank, read
from options.gex import analyze
from options.sources import from_chain_store, sessions_in_store

# The chain store is written by a separate CBOE recorder, not by this repo.
STORE = Path(os.environ.get("QUANTDESK_CHAINS", "../quantdesk/data/chains.sqlite"))
JOURNAL = Path("data/options_shadow.jsonl")
DEFAULT_SYMBOLS = ("SPY", "INTC", "CSCO")
MAX_DTE = 45
#: Calendar days. A weekend is 3; anything wider is a recorder gap, and a
#: one-session expected move cannot be graded across it.
MAX_GAP_DAYS = 5


def load_journal() -> list[dict]:
    if not JOURNAL.exists():
        return []
    out = []
    for line in JOURNAL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append(record: dict) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def spot_on(symbol: str, session: str) -> float | None:
    try:
        return from_chain_store(STORE, symbol, session, max_dte=MAX_DTE).spot
    except Exception:                       # noqa: BLE001 - absent is an answer
        return None


def do_reads(symbols: tuple[str, ...], rates: BaseRates | None) -> list[dict]:
    """One read per symbol, on its latest stored session. Idempotent."""
    seen = {(r["symbol"], r["session"]) for r in load_journal()
            if r.get("action") == "read"}
    reads, written = [], 0
    for sym in symbols:
        sessions = sessions_in_store(STORE, sym)
        if not sessions:
            print(f"  {sym}: no sessions in the store")
            continue
        session = sessions[-1]
        try:
            chain = from_chain_store(STORE, sym, session, max_dte=MAX_DTE)
        except ValueError as exc:
            print(f"  {sym}: {exc}")
            continue
        rep = analyze(chain)
        r = read(chain, rep)
        reads.append(r)
        print(r.render())
        if (sym, session) in seen:
            continue                        # already recorded; re-running is safe
        append({
            "action": "read", "symbol": sym, "session": session,
            "written_at": datetime.now().isoformat(timespec="seconds"),
            "spot": r.spot, "regime": r.regime, "size_call": r.size_call,
            "conviction": r.conviction, "atm_iv": round(r.atm_iv, 6),
            "expected_move": round(r.expected_move, 4),
            "net_gex": r.net_gex, "call_wall_em": r.call_wall_em,
            "put_wall_em": r.put_wall_em, "flip_em": r.flip_em,
            "warned": bool(rep.warning), "notes": r.notes,
        })
        written += 1

    top = rank(reads, 3, rates)
    print(f"\n  shortlist: " + (", ".join(
        f"{x.symbol} {x.conviction:.2f}" for x in top) or "none"))
    if len(top) < len(reads):
        print("  (names drop for zero conviction or a coverage warning)")
    print(f"  journalled {written} new read(s)")
    return reads


def do_grades(rates: BaseRates | None) -> int:
    """Grade every read that now has a later session to be measured against."""
    rows = load_journal()
    graded = {(r["symbol"], r["session"]) for r in rows
              if r.get("action") == "grade"}
    stale: list[tuple[str, str, str, int]] = []
    n = 0
    for r in rows:
        if r.get("action") != "read":
            continue
        key = (r["symbol"], r["session"])
        if key in graded:
            continue
        sessions = sessions_in_store(STORE, r["symbol"])
        later = [s for s in sessions if s > r["session"]]
        if not later:
            continue                        # the outcome has not happened yet
        nxt = later[0]

        # THE NEXT STORED SESSION IS NOT NECESSARILY THE NEXT SESSION. The
        # store currently holds SPY for 2012-08-01, 2026-08-20, 2026-08-28 and
        # 2026-09-09, so "the following row" can be days or fourteen years
        # later. The expected move is scaled to ONE session; grading it against
        # a twelve-day gap is not a hard call to get wrong, it is a guaranteed
        # wrong answer that looks like a miss. A gap is a recorder outage, and
        # the right response is to decline rather than to invent a scale factor.
        gap_days = (date.fromisoformat(nxt)
                    - date.fromisoformat(r["session"])).days
        if gap_days > MAX_GAP_DAYS:
            stale.append((r["symbol"], r["session"], nxt, gap_days))
            continue
        s1 = spot_on(r["symbol"], nxt)
        if s1 is None or r.get("expected_move", 0) <= 0:
            continue

        # Rebuild just enough of the read to grade it. The stored record is the
        # source of truth, NOT a recomputation from today's chain -- the chain
        # has moved and regrading against it would be scoring a different call.
        from options.daily import DailyRead
        prior = DailyRead(
            symbol=r["symbol"], asof=r["session"], spot=r["spot"],
            regime=r["regime"], net_gex=r.get("net_gex", 0.0),
            atm_iv=r["atm_iv"], expected_move=r["expected_move"],
            call_wall_em=r.get("call_wall_em"), put_wall_em=r.get("put_wall_em"),
            flip_em=r.get("flip_em"), size_call=r["size_call"],
            conviction=r["conviction"])
        g = grade(prior, s1 - r["spot"], rates)
        g.update({"action": "grade", "symbol": r["symbol"],
                  "session": r["session"], "outcome_session": nxt,
                  "warned": r.get("warned", False),
                  "graded_at": datetime.now().isoformat(timespec="seconds")})
        append(g)
        n += 1
        print(f"  graded {r['symbol']} {r['session']} -> {nxt}: "
              f"{r['size_call']} ratio {g['ratio']:.2f} "
              f"{'HIT' if g['hit'] else 'miss'}")
    for sym, ses, nxt, gap in stale:
        print(f"  SKIPPED {sym} {ses}: next stored session is {nxt}, "
              f"{gap} days later -- recorder gap, not a session")
    return n


def report() -> None:
    rows = [r for r in load_journal() if r.get("action") == "grade"]
    print("=" * 70)
    print("SHADOW RECORD")
    print("=" * 70)
    if not rows:
        print("  nothing graded yet.")
        print("  Each session adds one record per symbol; `aggregate()` refuses")
        print("  a verdict under 30, so this is weeks away from an opinion.")
        return

    # Warned reads are excluded from the ranking; they are excluded here too,
    # for the same reason -- a regime that may be a pull artefact is not
    # evidence about the regime.
    clean = [r for r in rows if not r.get("warned")]
    print(f"  {len(rows)} graded, {len(rows) - len(clean)} excluded on a "
          f"coverage warning\n")
    for side in ("compression", "expansion"):
        arm = [r for r in clean if r.get("size_call") == side]
        if not arm:
            continue
        agg = aggregate(arm)
        if not agg.get("n"):
            continue
        print(f"  {side.upper():<12} n={agg['n']:<4} observed "
              f"{agg['observed']:.1%}  null {agg['expected_from_base_rates']:.1%}"
              f"  edge {agg['edge']:+.1%}  z={agg['z']}  -> {agg['verdict']}")
    overall = aggregate(clean)
    if overall.get("n"):
        print(f"\n  {'ALL':<12} n={overall['n']:<4} observed "
              f"{overall['observed']:.1%}  null "
              f"{overall['expected_from_base_rates']:.1%}"
              f"  edge {overall['edge']:+.1%}  z={overall['z']}"
              f"  -> {overall['verdict']}")
    if overall.get("skipped_no_base_rate"):
        print(f"  {overall['skipped_no_base_rate']} record(s) had no base rate "
              f"and were excluded rather than scored against 50%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    rates = BaseRates.load()
    if rates is None:
        print("NO BASE-RATE TABLE. Run `python -m research.em_base_rate` first;")
        print("without it every graded record is unscoreable later.")
    else:
        print(f"base rates: {rates.n:,} sessions, {rates.source}")

    if args.report:
        report()
        return 0

    print(f"\nreading {len(args.symbols)} symbol(s) from {STORE.name}\n")
    do_reads(tuple(s.upper() for s in args.symbols), rates)
    print()
    n = do_grades(rates)
    if n == 0:
        print("  nothing new to grade -- outcomes arrive one session later")
    print()
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
