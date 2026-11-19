[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Creepzone11/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Creepzone11/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# gta6-trophy-rarity-forecast-01

Entry for make.fun competition `gta6-trophy-rarity-forecast-01`
(user handle: creepzone). Forecasts, for every GTA VI trophy on
PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z, as five
quantiles per trophy, from the case file alone.

## What it is

`main.py` (Python 3.13, stdlib only, deterministic, ~0.07 s per case).
For each trophy it fits a straight line to its logit earn rate against
ln(1 + hours since launch) over the last 24 hours of reads, then reads
the line at the target -- damped and shrunk:

- Per-trophy slope, shrunk toward the pooled slope of all trophies
  (weight n / (n + 20)): a trophy with many reads keeps its own slope,
  a thin one borrows the panel's. Fast trophies saturate and slow ones
  keep rising, so one slope for all trophies is wrong.
- Damping 0.6 on the extrapolated increment: a launch week's logit
  curve is concave (saturation + dilution as fresh players join), so a
  straight line extrapolated 4.5 days out overshoots. The forecast is
  pulled 40% of the way back toward persistence at the last read.
- Spread calibrated from the case's own history: fit on the first 75%
  of reads, predict the last 25%, grid-search the sigma (logit spread
  per sqrt-day) that minimises pinball loss there. Falls back to a wide
  sigma (0.75) when the history is too short or constant -- which is
  exactly the practice cases, where every read is the same page.

## Validation

Practice cases (p00/p01): output valid (55 rows, monotone, 0-100),
two runs byte-identical, ~0.07 s per run. The practice loss itself
cannot be computed honestly -- the answers are the house's reads after
the checkpoints, unknowable from a constant history -- and the practice
pays nothing, so it is not chased.

Synthetic backtest (fake launch with known dynamics: player growth,
per-trophy progression timescales, dilution, 5-hour page refreshes;
15 cases across 3 truth regimes and 4 generator settings): this model
scores mean pinball 0.178 vs the house baseline's 0.284 (0.63x), winning
in every generator setting and on every 48-hour-checkpoint case. The
first version of this entry (pooled panel with ln N) scored 0.727 in
the same backtest and was discarded. Details in the workdir, not in
this repo.

## Run it locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
