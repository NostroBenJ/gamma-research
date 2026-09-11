# gamma-research

Dealer gamma exposure (GEX) computed from option chains, and pre-registered tests of whether it
tells you anything. Plain Python, standard library only.

## What's here

| file | what it does | checks |
|---|---|---:|
| `options/gex.py` | Black-Scholes gamma and delta; dollar GEX = gamma × OI × 100 × S² × 0.01; net GEX re-priced at any spot; the gamma flip; call and put walls; a coverage warning | `verify_gex.py` **57** |
| `options/daily.py` | expected move, ATM IV, a daily read per name, and ranking and grading **conditioned on implied vol** | `verify_daily.py` **56** |
| `options/spread.py` | vertical spread builder and finder: credit, max loss, max profit, breakeven, and the win rate needed to break even | `verify_spread.py` **39** |
| `options/sources.py` | loads a chain from a SQLite store of CBOE snapshots, taking IV from the out-of-the-money wing | |
| `research/em_base_rate.py` | how often SPY stays inside its expected move, by VIX bucket | |
| `research/gex_regime_test.py` | does the gamma regime change those odds? (pre-registered) | |
| `run_options_shadow.py` | a daily shadow journal, graded after the fact. **Places no orders** | |
| `scripts/rh_chain.py` | builds a chain from Robinhood instrument and quote dumps; refuses asymmetric pulls | |

## Design choices that each fixed a real bug

- **Regime comes from the sign of net GEX at spot**, never from comparing spot to a derived flip level.
- **The flip is found by re-pricing the whole chain at each candidate spot**, scanning outward from
  spot. When no crossing exists in range it returns `None` instead of inventing a level.
- **Walls have a side and a sign.** A call wall must sit at or above spot with positive gamma, a put
  wall at or below with negative gamma, or there is no wall.
- **Net GEX is a difference of two sums, so uneven sampling fakes a regime.** A chain pulled with more
  call strikes than put strikes reads positive by construction. Coverage is measured as a share of
  gamma (5% one-sided tolerance), counting only rows with open interest and IV, and warned chains are
  excluded from ranking. Catching this in my own data is the finding I'm proudest of.
- **Dealers are assumed long calls and short puts.** That's a convention, not a fact, and the first
  thing to revisit if levels look systematically wrong.

## What it found

- **Price stays inside one expected move 80.8% ± 0.6% of the time** (n = 5,029 sessions): 87.3% when
  VIX is at or below 13.9, 74.7% when it's above 22.6. Any "the range will hold" call has to beat that,
  not 50%.
- **The gamma regime barely moves those odds.** On a year of SPX data (216 session pairs, 7-day expiry
  pre-registered as primary): 89.4% inside in positive gamma vs 86.0% in negative, **z = +0.76**. The
  test could only have detected a gap of 10.7 points. The 30-day horizon gives z = +1.88 against a
  Bonferroni bar across three; the 1-day horizon had no usable pairs.
- **The right-sign effect (smaller moves in positive gamma) is only monetised by selling premium**,
  and the credit spreads that would do it measured poorly. See
  [options-research](https://github.com/waddwwdawad/options-research).

## Running it

Python 3.11+. From the repo root:

```
python verify_gex.py
python verify_daily.py
python verify_spread.py
python -m research.em_base_rate
python -m research.gex_regime_test
```

Data locations can be overridden with environment variables:

| variable | default | what |
|---|---|---|
| `QUANTDESK_CHAINS` | `../quantdesk/data/chains.sqlite` | CBOE chain snapshots, written by a separate recorder |
| `SPY_VIX_CSV` | `../options-research/spy_vix_daily.csv` | built by `vrp_data.py` in options-research |
| `SPX_GEX_CSV` | `../quantdesk/data/spx_gex_daily.csv` | SPX gamma series |

## Data

No market data is in this repo. `data/em_base_rate.json` is a small derived table of base rates. The
SPX gamma series was built from **purchased Databento OPRA data**, which is licensed and not
redistributed: the results are reported here, the data is not.

Research code for learning, not investment advice.
