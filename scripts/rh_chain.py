"""Turn raw Robinhood chain responses into a validated GEX snapshot.

    # 1. what to fetch -- run this FIRST, it prints the exact calls
    python scripts/rh_chain.py --plan --symbol INTC --spot 105.59 \
        --expiry 2026-09-11 --band 0.10

    # 2. having saved the raw MCP responses, join them
    python scripts/rh_chain.py --symbol INTC --spot 105.59 \
        --instruments raw/intc_instruments.json --quotes raw/intc_quotes.json

    # 3. check what landed
    python scripts/rh_chain.py --show --symbol INTC

There is no API call in this file, for the same reason `rh_snapshot.py` has
none: Robinhood publishes no official equities REST API and the agentic MCP is
an interactive-session capability, not an endpoint a scheduled process can
reach. The fetching is done by an agent or by hand; this script owns the part
that must not be done by hand.

WHAT MUST NOT BE DONE BY HAND, AND WHY THIS FILE EXISTS
-------------------------------------------------------
A full chain is roughly eighty strikes per right per expiry. Pulling all of it
for several names is a lot of requests, so any practical pull is BANDED to
strikes near spot -- and the moment a human picks the band, they will pick it
once, use it for the calls, and pick a slightly different one for the puts.

Net GEX is a difference of two sums. An uneven band makes it positive or
negative BY CONSTRUCTION, produces a clean-looking wall on the better-sampled
side and none on the other, and looks exactly like a real reading. That is not
hypothetical -- it is what the first live INTC pull did, and `options/gex.py`
grew `coverage_warning()` because of it.

So the band is computed here, once, and applied to both rights identically.
`--plan` prints the strike range so the fetch itself is symmetric, and the
ingest step REFUSES a pull whose two sides do not match rather than writing a
snapshot that will quietly mislead.

INPUT SHAPES. Both files are the raw `data` object from the MCP response, or a
bare list of the inner records -- `{"data": {"instruments": [...]}}`,
`{"instruments": [...]}`, and `[...]` are all accepted, because which one you
end up with depends on how the response was saved and guessing wrong should not
cost a re-fetch.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from options.gex import analyze, coverage_warning, load_chain  # noqa: E402

CHAIN_DIR = ROOT / "data" / "chains"
DEFAULT_BAND = 0.10


def unwrap(obj, *keys: str) -> list[dict]:
    """Accept {"data": {"k": [...]}}, {"k": [...]} or [...] alike."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], dict):
            obj = obj["data"]
        for k in keys:
            if k in obj:
                return obj[k]
        if "results" in obj:
            return obj["results"]
    raise SystemExit(f"could not find {keys} in the supplied JSON")


def plan(symbol: str, spot: float, expiry: str, band: float) -> int:
    """Print the strike range and the exact calls that produce a SYMMETRIC pull."""
    lo, hi = spot * (1 - band), spot * (1 + band)
    print(f"chain plan for {symbol}  spot {spot:,.2f}  expiry {expiry}")
    print(f"  band +/-{band:.0%}  ->  strikes {lo:,.2f} .. {hi:,.2f}")
    print()
    print("  Fetch BOTH rights over this SAME range. The whole point of this")
    print("  step is that the two sides match; an uneven pull makes net GEX")
    print("  positive or negative by construction.")
    print()
    print("  get_option_instruments  chain_symbol=%s expiration_dates=%s type=call"
          % (symbol, expiry))
    print("  get_option_instruments  chain_symbol=%s expiration_dates=%s type=put"
          % (symbol, expiry))
    print("  -> keep only ids whose strike_price is inside the range above")
    print("  -> get_option_quotes on those ids, in batches of ~20")
    print()
    print("  Save the instrument responses to one file and the quote responses")
    print("  to another (a JSON list of either shape is fine), then:")
    print(f"    python scripts/rh_chain.py --symbol {symbol} --spot {spot} \\")
    print(f"        --instruments <file> --quotes <file>")
    return 0


def ingest(symbol: str, spot: float, instruments_path: Path, quotes_path: Path,
           band: float, out: Path, r: float, allow_asymmetric: bool) -> int:
    raw_inst = unwrap(json.loads(instruments_path.read_text(encoding="utf-8")),
                      "instruments")
    raw_q = unwrap(json.loads(quotes_path.read_text(encoding="utf-8")), "results")

    # instrument id -> its static description
    meta: dict[str, dict] = {}
    for i in raw_inst:
        meta[str(i["id"])] = {
            "strike": float(i["strike_price"]),
            "right": "c" if str(i["type"]).lower().startswith("c") else "p",
            "expiry": str(i["expiration_date"]),
        }

    lo, hi = spot * (1 - band), spot * (1 + band)
    rows, skipped_band, unmatched = [], 0, 0
    for entry in raw_q:
        q = entry.get("quote", entry)
        iid = str(q.get("instrument_id") or q.get("id") or "")
        m = meta.get(iid)
        if m is None:
            unmatched += 1
            continue
        if not (lo <= m["strike"] <= hi):
            skipped_band += 1
            continue
        rows.append({
            "strike": m["strike"], "right": m["right"], "expiry": m["expiry"],
            "iv": float(q.get("implied_volatility") or 0.0),
            "oi": float(q.get("open_interest") or 0.0),
            "volume": float(q.get("volume") or 0.0),
        })

    if unmatched:
        print(f"  {unmatched} quotes had no matching instrument -- the two "
              f"files are from different pulls?")
    if not rows:
        raise SystemExit("no contracts inside the band; nothing to write")

    asof = datetime.now().isoformat(timespec="seconds")
    for row in rows:
        row["t_years"] = max(0.0, (date.fromisoformat(row["expiry"])
                                   - date.today()).days / 365.0)

    snap = {
        "symbol": symbol, "asof": asof, "spot": spot, "r": r,
        "band": band,
        "source": f"robinhood MCP, banded +/-{band:.0%} around {spot:,.2f}",
        "contracts": sorted(rows, key=lambda x: (x["right"], x["strike"])),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")

    chain = load_chain(out)
    warn = coverage_warning(chain)
    print(f"wrote {out}: {len(rows)} contracts inside the band "
          f"({skipped_band} outside, dropped)")
    if warn:
        # Refusing is the point. A snapshot that reads as a market signal but
        # is an artefact of the pull is worse than no snapshot, because nothing
        # downstream can tell the difference.
        print(f"\n  {warn}")
        if not allow_asymmetric:
            out.unlink()
            print(f"\n  REFUSED -- {out} removed. Re-fetch the missing side, or")
            print("  pass --allow-asymmetric if you know why and want it anyway.")
            return 1
        print("\n  KEPT anyway (--allow-asymmetric). The warning travels with "
              "the snapshot and renders in every report built from it.")
    print()
    print(analyze(chain).render())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--spot", type=float)
    ap.add_argument("--expiry")
    ap.add_argument("--band", type=float, default=DEFAULT_BAND,
                    help="fraction of spot to keep either side (default 0.10)")
    ap.add_argument("--instruments", type=Path)
    ap.add_argument("--quotes", type=Path)
    ap.add_argument("--r", type=float, default=0.04)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--allow-asymmetric", action="store_true",
                    help="write a lopsided chain anyway; the warning stays on it")
    args = ap.parse_args()

    out = args.out or (CHAIN_DIR / f"{args.symbol.upper()}.json")

    if args.show:
        if not out.exists():
            raise SystemExit(f"no chain at {out}")
        print(analyze(load_chain(out)).render())
        return 0

    if args.plan:
        if args.spot is None or not args.expiry:
            raise SystemExit("--plan needs --spot and --expiry")
        return plan(args.symbol.upper(), args.spot, args.expiry, args.band)

    if not (args.instruments and args.quotes and args.spot is not None):
        raise SystemExit("ingest needs --instruments, --quotes and --spot "
                         "(or use --plan / --show)")
    return ingest(args.symbol.upper(), args.spot, args.instruments, args.quotes,
                  args.band, out, args.r, args.allow_asymmetric)


if __name__ == "__main__":
    raise SystemExit(main())
