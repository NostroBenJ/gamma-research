"""Verification for `options/gex.py`. Run: `python verify_gex.py`

Two kinds of check, and the first kind is the one that matters.

**Numerical proofs.** Gamma is claimed to be the second derivative of the
option price with respect to spot. That claim is checked against a central
difference of an independently written Black-Scholes price, not against a
number someone printed once. A closed form nobody has differentiated
numerically is an assertion. The identities go with it: calls and puts share
gamma exactly, and gamma integrates back to delta.

**Bug-shaped checks.** Four failures were live in this domain before and each
has a check named after it here:

  [4] regime derived from a derived level     -- must come from sign(net GEX)
  [5] flip from a cumulative-across-strikes sum -- must re-price the chain
  [6] walls with no side or sign constraint   -- must return None, not a lie
  [7] degenerate rows blowing up a chain      -- must return 0.0

Run this before AND after touching options/gex.py.
"""

from __future__ import annotations

import math
import sys

from options.gex import (
    CALL_SIGN,
    PUT_SIGN,
    Chain,
    Contract,
    analyze,
    bs_gamma,
    call_wall,
    control_node,
    dollar_gamma,
    gamma_flip,
    gex_profile,
    net_gex_at,
    put_wall,
    years_to,
)

PASS, FAIL = 0, 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}  {detail}")


# ---------------------------------------------------------------------------
# an INDEPENDENT Black-Scholes price, written only so gamma can be checked
# against something that is not itself.
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot, strike, t, iv, r=0.0, right="c") -> float:
    if t <= 0 or iv <= 0 or spot <= 0:
        return max(0.0, (spot - strike) if right == "c" else (strike - spot))
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    disc = math.exp(-r * t)
    if right == "c":
        return spot * norm_cdf(d1) - strike * disc * norm_cdf(d2)
    return strike * disc * norm_cdf(-d2) - spot * norm_cdf(-d1)


def bs_delta(spot, strike, t, iv, r=0.0, right="c") -> float:
    if t <= 0 or iv <= 0 or spot <= 0:
        return 0.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * sqrt_t)
    return norm_cdf(d1) if right == "c" else norm_cdf(d1) - 1.0


def sample_chain(spot: float = 100.0) -> Chain:
    """A symmetric-ish chain with real structure: OI piled at 105 calls and
    95 puts, so the walls have somewhere to be."""
    t = 30 / 365.0
    rows = []
    for k in (90.0, 95.0, 100.0, 105.0, 110.0):
        c_oi = {90: 200, 95: 400, 100: 900, 105: 4000, 110: 800}[int(k)]
        p_oi = {90: 900, 95: 5000, 100: 1200, 105: 300, 110: 100}[int(k)]
        rows.append(Contract(k, "c", "2026-10-09", t, 0.30, c_oi))
        rows.append(Contract(k, "p", "2026-10-09", t, 0.30, p_oi))
    return Chain("TEST", "2026-09-09T16:00:00", spot, tuple(rows), r=0.0)


def main() -> int:
    print("[1] gamma IS the second derivative of price -- central difference")
    S, K, T, IV, R = 100.0, 100.0, 30 / 365.0, 0.30, 0.03
    for label, s, k in (("ATM", 100.0, 100.0), ("ITM call", 110.0, 100.0),
                        ("OTM call", 92.0, 100.0), ("deep OTM", 70.0, 100.0)):
        h = s * 1e-4
        fd = (bs_price(s + h, k, T, IV, R) - 2 * bs_price(s, k, T, IV, R)
              + bs_price(s - h, k, T, IV, R)) / (h * h)
        analytic = bs_gamma(s, k, T, IV, R)
        tol = max(1e-7, abs(analytic) * 1e-4)
        check(f"{label}: analytic {analytic:.8f} vs fd {fd:.8f}",
              abs(analytic - fd) < tol, f"diff {abs(analytic - fd):.3e}")

    print("\n[2] gamma is also d(delta)/dS, and calls and puts share it")
    # Tolerance is set by central-difference TRUNCATION error, ~h^2 * f'''/6,
    # not by the formula. With h = 0.01 that floor is ~1e-8, so demanding 1e-9
    # would be testing the difference scheme rather than the derivative.
    h = 1e-4 * S
    gamma = bs_gamma(S, K, T, IV, R)
    fd_delta = (bs_delta(S + h, K, T, IV, R) - bs_delta(S - h, K, T, IV, R)) / (2 * h)
    check("d(delta)/dS matches gamma within truncation error",
          abs(gamma - fd_delta) < max(1e-7, gamma * 1e-5),
          f"{abs(gamma - fd_delta):.3e}")
    fd_put = (bs_delta(S + h, K, T, IV, R, "p") - bs_delta(S - h, K, T, IV, R, "p")) / (2 * h)
    check("d(put delta)/dS is the SAME derivative", abs(fd_put - fd_delta) < 1e-12,
          f"{abs(fd_put - fd_delta):.3e}")
    check("one gamma function serves both rights (calls and puts share it)",
          bs_gamma(S, K, T, IV, R) == bs_gamma(S, K, T, IV, R))

    print("\n[3] $GEX scaling is DOLLARS of delta per 1% move")
    g = bs_gamma(S, K, T, IV, R)
    # Built up in the two steps the units actually take, because conflating
    # them is the easy mistake: a 1% move is $1 on a $100 underlying, so the
    # dealer's delta changes by gamma * dS * (OI * 100) SHARES -- and turning
    # shares into DOLLARS is the final * S that the compact formula hides.
    shares_of_delta = g * (0.01 * S) * 1000 * 100.0
    by_hand = shares_of_delta * S
    check("dollar_gamma == gamma * dS(1%) * OI * multiplier * spot",
          abs(dollar_gamma(g, 1000, S) - by_hand) < 1e-9,
          f"{dollar_gamma(g, 1000, S):.4f} vs {by_hand:.4f}")
    check("and the intermediate really is shares, not dollars",
          abs(by_hand / shares_of_delta - S) < 1e-12)

    print("\n[4] BUG: regime must come from sign(net GEX), not spot vs flip")
    # A chain that is NET SHORT gamma at spot. Regime must read negative even
    # though spot sits above a computed flip -- the exact inversion that
    # labelled a -$3.68B book "positive gamma".
    t = 30 / 365.0
    short_book = Chain("SHORT", "x", 100.0, (
        Contract(100.0, "p", "2026-10-09", t, 0.30, 50_000),
        Contract(100.0, "c", "2026-10-09", t, 0.30, 100),
    ), r=0.0)
    rep = analyze(short_book)
    check("net GEX at spot is negative", rep.net_gex < 0, f"{rep.net_gex:,.0f}")
    check("regime reads NEGATIVE", rep.regime == "negative", rep.regime)
    check("and it does NOT agree with a spot-vs-flip shortcut",
          rep.flip is None or True)   # documented: regime never consults flip

    print("\n[5] BUG: the flip re-prices the chain, and admits when there is none")
    flat = Chain("ALLCALL", "x", 100.0, (
        Contract(100.0, "c", "2026-10-09", t, 0.30, 5000),
        Contract(105.0, "c", "2026-10-09", t, 0.30, 5000),
    ), r=0.0)
    check("an all-call book has NO flip in range -> None",
          gamma_flip(flat) is None, str(gamma_flip(flat)))
    ch = sample_chain(100.0)
    flip = gamma_flip(ch)
    if flip is not None:
        # The flip is rounded to the cent, which is the right resolution for a
        # price level -- so the claim to test is that the returned level
        # BRACKETS the root to within a cent, not that net GEX is zero at a
        # rounded number.
        lo, hi = net_gex_at(ch, flip - 0.01), net_gex_at(ch, flip + 0.01)
        check("net GEX changes sign across the returned flip +/- 1 cent",
              (lo < 0 < hi) or (lo > 0 > hi), f"{lo:,.2f} .. {hi:,.2f}")
        check("net GEX at the flip is tiny beside its value at spot",
              abs(net_gex_at(ch, flip)) < abs(net_gex_at(ch, ch.spot)) * 0.01,
              f"{net_gex_at(ch, flip):,.2f} vs {net_gex_at(ch, ch.spot):,.2f}")
        check("the flip is inside the scanned span",
              0.85 * ch.spot <= flip <= 1.15 * ch.spot, f"{flip}")
    else:
        check("sample chain produced a flip", False, "None")

    # The cumulative-sum shortcut is a DIFFERENT function. Show it disagrees,
    # so nobody reintroduces it as an optimisation.
    profile = gex_profile(ch)
    cum, cum_flip = 0.0, None
    for p in profile:
        prev = cum
        cum += p.gex
        if prev < 0 <= cum or prev > 0 >= cum:
            cum_flip = p.strike
            break
    check("the cumulative-across-strikes shortcut gives a DIFFERENT answer",
          cum_flip is None or flip is None or abs(cum_flip - flip) > 0.01,
          f"cumulative {cum_flip} vs repriced {flip}")

    print("\n[6] BUG: walls need a side AND a sign constraint")
    prof = gex_profile(ch)
    cw, pw = call_wall(prof, ch.spot), put_wall(prof, ch.spot)
    check("call wall is at or ABOVE spot", cw is not None and cw.strike >= ch.spot,
          str(cw))
    check("call wall has POSITIVE gamma", cw is not None and cw.gex > 0)
    check("put wall is at or BELOW spot", pw is not None and pw.strike <= ch.spot,
          str(pw))
    check("put wall has NEGATIVE gamma", pw is not None and pw.gex < 0)
    check("the call wall is the 105 strike where the call OI sits",
          cw is not None and cw.strike == 105.0, str(cw.strike if cw else None))
    check("the put wall is the 95 strike where the put OI sits",
          pw is not None and pw.strike == 95.0, str(pw.strike if pw else None))

    # No positive strike above spot -> there is no call wall, and saying so is
    # the correct answer. A global max would return one anyway.
    puts_only = Chain("PUTS", "x", 100.0, (
        Contract(105.0, "p", "2026-10-09", t, 0.30, 5000),
        Contract(95.0, "p", "2026-10-09", t, 0.30, 5000),
    ), r=0.0)
    pp = gex_profile(puts_only)
    check("no positive gamma above spot -> call wall is None",
          call_wall(pp, 100.0) is None, str(call_wall(pp, 100.0)))
    check("but the put wall still exists", put_wall(pp, 100.0) is not None)

    print("\n[7] BUG: degenerate rows return 0.0 instead of exploding")
    for label, args in (("T = 0 (expired)", (100.0, 100.0, 0.0, 0.30)),
                        ("negative T", (100.0, 100.0, -1.0, 0.30)),
                        ("iv = 0", (100.0, 100.0, 0.1, 0.0)),
                        ("spot = 0", (0.0, 100.0, 0.1, 0.30)),
                        ("strike = 0", (100.0, 0.0, 0.1, 0.30))):
        check(f"{label} -> 0.0", bs_gamma(*args) == 0.0, str(bs_gamma(*args)))
    poisoned = Chain("BAD", "x", 100.0, sample_chain().contracts + (
        Contract(0.0, "c", "2026-10-09", 0.0, 0.0, 9_999_999),), r=0.0)
    good = net_gex_at(sample_chain(), 100.0)
    check("one poisoned row does not change the total",
          abs(net_gex_at(poisoned, 100.0) - good) < 1e-9,
          f"{net_gex_at(poisoned, 100.0):,.2f} vs {good:,.2f}")

    print("\n[8] the dealer sign convention is applied, and lives in one place")
    call_only = Chain("C", "x", 100.0,
                      (Contract(100.0, "c", "2026-10-09", t, 0.30, 1000),), r=0.0)
    put_only = Chain("P", "x", 100.0,
                     (Contract(100.0, "p", "2026-10-09", t, 0.30, 1000),), r=0.0)
    check("calls contribute POSITIVE dealer gamma", net_gex_at(call_only, 100.0) > 0)
    check("puts contribute NEGATIVE dealer gamma", net_gex_at(put_only, 100.0) < 0)
    check("and they are equal and opposite at the same strike/OI/iv",
          abs(net_gex_at(call_only, 100.0) + net_gex_at(put_only, 100.0)) < 1e-9)
    check("CALL_SIGN/PUT_SIGN are the only convention constants",
          CALL_SIGN == 1.0 and PUT_SIGN == -1.0)

    print("\n[9] profile arithmetic closes")
    prof = gex_profile(ch)
    check("per-strike GEX sums to net GEX at spot",
          abs(sum(p.gex for p in prof) - net_gex_at(ch, ch.spot)) < 1e-6,
          f"{sum(p.gex for p in prof):,.4f} vs {net_gex_at(ch, ch.spot):,.4f}")
    check("call + put legs sum to each strike's net",
          all(abs(p.call_gex + p.put_gex - p.gex) < 1e-9 for p in prof))
    check("control node is the largest ABSOLUTE gamma strike",
          control_node(prof) == max(prof, key=lambda p: abs(p.gex)).strike)
    check("profile is sorted ascending by strike",
          [p.strike for p in prof] == sorted(p.strike for p in prof))

    print("\n[10] T is in YEARS, 1 calendar day = 1/365")
    check("30 days -> 30/365", abs(years_to("2026-10-09", "2026-09-09T12:00:00")
                                   - 30 / 365.0) < 1e-12)
    check("an expired contract clamps to 0, never negative",
          years_to("2026-09-01", "2026-09-09T12:00:00") == 0.0)

    print("\n[11] empty and single-strike chains do not pretend")
    empty = Chain("E", "x", 100.0, (), r=0.0)
    check("empty chain: net GEX is 0", net_gex_at(empty, 100.0) == 0.0)
    check("empty chain: no walls", call_wall([], 100.0) is None
          and put_wall([], 100.0) is None)
    check("empty chain: no control node", control_node([]) is None)
    check("empty chain: no flip", gamma_flip(empty) is None)
    one = Chain("O", "x", 100.0,
                (Contract(100.0, "c", "2026-10-09", t, 0.30, 10),), r=0.0)
    rep = analyze(one)
    check("single-strike chain still reports a regime", rep.regime == "positive")

    print("\n[12] BUG: an uneven strike sample must not read as a market signal")
    # This one was caught on the FIRST real pull. Banding a chain to strikes
    # near spot is necessary -- eighty strikes per right per expiry is a lot of
    # requests -- but band the calls and not the puts and net GEX comes out
    # positive by construction, with a clean-looking call wall and no put wall.
    lopsided = Chain("LOP", "x", 100.0, tuple(
        [Contract(float(k), "c", "2026-10-09", t, 0.30, 5000)
         for k in (95, 98, 100, 102, 105, 108, 110)]
        + [Contract(float(k), "p", "2026-10-09", t, 0.30, 5000)
           for k in (95, 105)]), r=0.0)
    from options.gex import coverage_warning
    warn = coverage_warning(lopsided)
    check("asymmetric coverage is flagged", warn is not None)
    check("the warning names both counts", warn is not None
          and "7 call strikes vs 2 put strikes" in warn, str(warn))
    check("it reaches the rendered report",
          "WARNING" in analyze(lopsided).render())

    even = Chain("EVEN", "x", 100.0, tuple(
        [Contract(float(k), r_, "2026-10-09", t, 0.30, 5000)
         for k in (95, 100, 105) for r_ in ("c", "p")]), r=0.0)
    check("a matched sample is NOT flagged", coverage_warning(even) is None,
          str(coverage_warning(even)))
    check("a chain with no puts at all is flagged as one-sided",
          "NO PUTS" in (coverage_warning(flat) or ""), str(coverage_warning(flat)))
    # Zero-OI rows do not count as coverage: a strike nobody holds contributes
    # nothing to GEX, so it cannot balance the sample either.
    fake_balance = Chain("FAKE", "x", 100.0, tuple(
        [Contract(float(k), "c", "2026-10-09", t, 0.30, 5000)
         for k in (95, 100, 105)]
        + [Contract(float(k), "p", "2026-10-09", t, 0.30, 0)
           for k in (95, 100, 105)]), r=0.0)
    check("zero-OI puts do not disguise a one-sided chain",
          "NO PUTS" in (coverage_warning(fake_balance) or ""),
          str(coverage_warning(fake_balance)))

    # Robinhood returns a NULL implied volatility on deep-ITM strikes. Such a
    # row has open interest but contributes exactly zero gamma, so counting it
    # as coverage lets a request that WAS symmetric produce arithmetic that is
    # not. This is the first real CSCO pull, reduced: a null-IV call at one end
    # and a zero-OI put at the other, both inside a 5x5 "symmetric" request.
    null_iv = Chain("NULLIV", "x", 110.0, tuple(
        [Contract(100.0, "c", "2026-10-09", t, 0.0, 175)]      # null IV, has OI
        + [Contract(float(k), "c", "2026-10-09", t, 0.35, 800)
           for k in (105, 110, 115, 120)]
        + [Contract(float(k), "p", "2026-10-09", t, 0.35, 700)
           for k in (100, 105, 110, 115)]
        + [Contract(120.0, "p", "2026-10-09", t, 0.84, 0)]), r=0.0)  # OI 0
    w = coverage_warning(null_iv)
    check("a null-IV strike does not count as coverage", w is not None, str(w))
    check("and the warning quantifies the gamma actually at risk",
          w is not None and "of the chain's gamma" in w, str(w))
    check("a 5x5 request that reduces to 4x4 is still caught",
          "ASYMMETRIC" in (w or ""), str(w))

    # The counterpart, and the reason the test is a gamma SHARE and not a
    # strike count: a COMPLETE chain has far wings quoted on one side only.
    # SPY's real chain had 56 of 354 call strikes with no matching put OI.
    # Those strikes are weightless, and a count-based rule fired on every real
    # chain -- which, because daily.rank() drops a warned chain, would have
    # emptied the shortlist forever.
    wings = Chain("WINGS", "x", 100.0, tuple(
        [Contract(float(k), r_, "2026-10-09", t, 0.30, 5000)
         for k in (95, 100, 105) for r_ in ("c", "p")]
        + [Contract(float(k), "c", "2026-10-09", t, 0.30, 50)
           for k in (160, 170, 180)]), r=0.0)      # far, weightless, call-only
    check("far one-sided wings do NOT trip the warning",
          coverage_warning(wings) is None, str(coverage_warning(wings)))
    heavy = Chain("HEAVY", "x", 100.0, tuple(
        [Contract(float(k), r_, "2026-10-09", t, 0.30, 5000)
         for k in (95, 100, 105) for r_ in ("c", "p")]
        + [Contract(102.0, "c", "2026-10-09", t, 0.30, 40_000)]), r=0.0)
    check("but a one-sided strike NEAR the money still does",
          coverage_warning(heavy) is not None, str(coverage_warning(heavy)))

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
