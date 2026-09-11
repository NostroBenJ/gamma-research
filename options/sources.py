"""Chain loaders. One `Chain` out, several feeds in.

    from options.sources import from_chain_store
    chain = from_chain_store("../../Downloads/quantdesk/data/chains.sqlite", "INTC")

`options/gex.py` computes; this decides what it computes ON. Keeping them apart
is what lets the same GEX code run against a live CBOE snapshot, a recorded
session from months ago, and a vendor cross-check, without any of them knowing
about the others.

WHY NOT THE ROBINHOOD MCP. It works, and it is the wrong tool. Assembling one
expiry of INTC through it took ~14 round trips and produced 26 contracts, with
a null IV on the ITM strikes, a quote 90 minutes stale, and spot drifting
between the call leg and the put leg of the same pull. One CBOE request returns
**2,046 INTC contracts at a single timestamp**, free, with bid/ask and sizes.
The MCP is an order-entry API; this is a data feed.

THE ONE RULE THAT MAKES CBOE DATA USABLE: **take the IV from the OTM wing.**
Their per-contract IV is inverted for in-the-money options -- at SPY K=779 the
call and put gammas differed **51x** (0.00510 vs 0.00010) off the same strike,
because the put's IV had been solved from a 15.37/18.10 quote. Gamma is
identical for a call and a put by put-call parity, so a feed that disagrees
with itself at one strike is telling you which side to discard. The OTM wing is
the liquid one, so it is the side to keep, and its IV is then applied to BOTH
rights' open interest at that strike.

Standard library only.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from .gex import Chain, Contract

#: Default location of the quantdesk recorder's store, relative to this repo.
DEFAULT_STORE = Path(__file__).resolve().parents[2] / (
    "../Downloads/quantdesk/data/chains.sqlite")


def otm_iv(strike: float, spot: float, call_iv: float | None,
           put_iv: float | None) -> float:
    """Implied vol from the out-of-the-money wing at this strike.

    Below spot the put is OTM; above spot the call is. At the money either is
    fine and the average is used, which also smooths the discontinuity as spot
    crosses a strike.

    Falls back to whichever side exists when the preferred wing is missing --
    a missing OTM quote is a thin strike, not a corrupted one, and dropping the
    strike entirely would punch a hole in the profile.
    """
    call_iv = call_iv if (call_iv or 0) > 0 else None
    put_iv = put_iv if (put_iv or 0) > 0 else None
    if call_iv is None and put_iv is None:
        return 0.0
    if strike < spot:
        return put_iv if put_iv is not None else call_iv
    if strike > spot:
        return call_iv if call_iv is not None else put_iv
    if call_iv is not None and put_iv is not None:
        return 0.5 * (call_iv + put_iv)
    return call_iv if call_iv is not None else put_iv


def from_chain_store(db_path: str | Path, underlying: str,
                     session_date: str | date | None = None,
                     *, r: float = 0.04, max_dte: int | None = None) -> Chain:
    """Load one recorded session out of the quantdesk chain store.

    `session_date` defaults to the latest session held for that underlying.
    `max_dte` keeps only expiries within N days, which is what a daily read
    wants -- a two-year LEAP contributes almost no gamma and a lot of rows.

    T IS MEASURED FROM THE SESSION DATE, NOT FROM TODAY. Reading a recorded
    session with today's clock understates nothing and overstates everything:
    a chain recorded three months ago would price every contract as if expiry
    were three months nearer, and the whole point of the archive is to replay
    what the map looked like THEN.
    """
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        if session_date is None:
            row = con.execute(
                "SELECT MAX(session_date) AS d FROM chain_contracts "
                "WHERE underlying = ?", (underlying,)).fetchone()
            if row is None or row["d"] is None:
                raise ValueError(f"no sessions stored for {underlying}")
            session = str(row["d"])
        else:
            session = (session_date.isoformat()
                       if isinstance(session_date, date) else str(session_date))

        snap = con.execute(
            "SELECT spot, fetched_at FROM chain_snapshots "
            "WHERE underlying = ? AND session_date = ? "
            "ORDER BY fetched_at DESC LIMIT 1", (underlying, session)).fetchone()
        if snap is None:
            raise ValueError(f"no snapshot for {underlying} on {session}")
        spot = float(snap["spot"])

        rows = con.execute(
            "SELECT expiry, strike, right, open_interest, iv, bid, ask, "
            "vendor_delta FROM chain_contracts "
            "WHERE underlying = ? AND session_date = ?",
            (underlying, session)).fetchall()
    finally:
        con.close()

    session_d = date.fromisoformat(session)

    # Collapse to one record per (expiry, strike): the OTM wing's IV, and the
    # open interest of each right kept separately.
    per_strike: dict[tuple[str, float], dict] = {}
    for row in rows:
        key = (str(row["expiry"]), float(row["strike"]))
        slot = per_strike.setdefault(
            key, {"c_oi": 0.0, "p_oi": 0.0, "c_iv": None, "p_iv": None,
                  "c_q": (0.0, 0.0, None), "p_q": (0.0, 0.0, None)})
        side = "c" if str(row["right"]).lower().startswith("c") else "p"
        slot[f"{side}_oi"] += float(row["open_interest"] or 0.0)
        slot[f"{side}_q"] = (float(row["bid"] or 0.0), float(row["ask"] or 0.0),
                             row["vendor_delta"])
        iv = row["iv"]
        if iv is not None and float(iv) > 0:
            slot[f"{side}_iv"] = float(iv)

    contracts: list[Contract] = []
    for (expiry, strike), slot in per_strike.items():
        dte = (date.fromisoformat(expiry) - session_d).days
        if dte < 0 or (max_dte is not None and dte > max_dte):
            continue
        iv = otm_iv(strike, spot, slot["c_iv"], slot["p_iv"])
        for side, oi in (("c", slot["c_oi"]), ("p", slot["p_oi"])):
            # A strike with no open interest still carries a tradeable QUOTE,
            # and the spread builder needs those. GEX is unaffected: zero OI
            # contributes zero gamma either way.
            bid, ask, vd = slot[f"{side}_q"]
            if oi <= 0 and not (bid > 0 or ask > 0):
                continue
            contracts.append(Contract(
                strike=strike, right=side, expiry=expiry,
                t_years=dte / 365.0, iv=iv, oi=oi, bid=bid, ask=ask,
                vendor_delta=float(vd) if vd not in (None, "") else None))

    return Chain(symbol=underlying, asof=f"{session}T16:00:00", spot=spot,
                 contracts=tuple(contracts), r=r)


def sessions_in_store(db_path: str | Path, underlying: str) -> list[str]:
    """Every session date held for an underlying, ascending."""
    con = sqlite3.connect(str(db_path))
    try:
        return [str(r[0]) for r in con.execute(
            "SELECT DISTINCT session_date FROM chain_contracts "
            "WHERE underlying = ? ORDER BY session_date", (underlying,))]
    finally:
        con.close()
