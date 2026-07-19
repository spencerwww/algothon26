# Rolling-Cointegration Pair Strategy

## Outcome

The full 51-instrument universe contains 1,275 possible pairs. Four pairs have
raw two-direction Engle-Granger p-values below 0.05 in each non-overlapping
250-day block:

- AENO-NWIG
- HUXZ-ACAC
- EORC-NGTE
- SMAH-ILVX

CTGI-EELT and DIHO-AETS passed the first 500 discovery days but failed the
days 501-750 significance check, so neither was added. The 500-day
`prices.txt` is an exact prefix of the 750-day file and is not independent
validation data.

The repeated identity of four significant pairs is unlikely under the
instrument-label permutation null (5,000 permutations, empirical p=0.0002).
However, no individual pair survives a naive 5% Benjamini-Hochberg correction
in every block; this multiple-testing limitation should remain explicit.

## Final strategy

`updated_algorithm.py` contains the standalone implementation:

- rolling OLS log-spread trading for the four pairs;
- pair-specific lookback and entry thresholds;
- zero-crossing exits;
- a trailing 250-day, two-direction Engle-Granger health test;
- entry only when both p-values are below 0.05;
- immediate exit when rolling significance fails;
- continuous residual-level mean reversion on non-pair instruments;
- integer, cap-safe positions and a turnover deadband.

`teamName.py` imports the strategy for local `eval.py` use. For submission,
rename `updated_algorithm.py` to the registered team name so the submission
remains a single file.

## Exact local evaluation

On the 750-day file, with days 501-750 scored by the official evaluator:

| Metric | Result |
|---|---:|
| Mean daily P&L | 183.8 |
| P&L standard deviation | 1,377.37 |
| Annualised Sharpe | 2.11 |
| Dollar turnover | 15,304,449 |
| Score | 150.13 |

The 500-day prefix scores 279.54 on days 251-500, but is not independent.
At five times the official commission, the 750-day score remains positive at
122.67. A 10-day block bootstrap has a 95% score interval that includes zero,
so hidden-data performance is not guaranteed.

## Reproduction

```bash
python pair_universe_research.py
python validate_updated_algorithm.py
python eval.py
```

The research script performs the full-universe scan, 1,152 parameter
simulations and 504 subset simulations. Curated outputs are stored in
`pair_research_results/`.
