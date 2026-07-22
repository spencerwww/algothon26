# Algothon Research Constraints

These rules are mandatory for every strategy, backtest, notebook, and report in this directory.

## Data boundaries

- Days 1-500 are the only development data in the current 750-day dataset.
- Days 501-750 are a one-shot holdout. Never use them to select instruments, features, models, hyperparameters, thresholds, lookbacks, position sizes, sleeve combinations, or rollback decisions.
- The leaderboard window is forward evaluation evidence. Never tune a strategy in response to its score unless it is explicitly relabelled as leaderboard-adaptive and a new untouched evaluation window exists.
- Do not describe a repeatedly inspected window as test, holdout, out-of-sample, or validation data. Label it selection-contaminated.

## Model-selection protocol

- Perform all research and model selection inside days 1-500.
- Use consecutive expanding-window validation within days 1-500; never randomly shuffle time-series observations.
- Compare candidates using pooled score, worst-fold mean, standard deviation, turnover, and parameter stability—not one best fold or one best endpoint.
- Prefer a broad stable parameter region over the single backtest maximum.
- Do not add an instrument rule merely because it is positive after scanning many instruments. Account for the candidate search and require a pre-declared selection rule.
- Keep a final candidate frozen before any holdout evaluation. Record its file hash and parameters first.

## Holdout protocol

- Ask for explicit user approval before consuming an untouched holdout.
- Run the holdout once for the frozen candidate. Do not iterate against it.
- If the holdout has already influenced a decision, state that it is contaminated and do not use it as evidence of generalisation.
- Running the official `eval.py` verifies accounting only; it does not make a tuned result out-of-sample.

## Reporting

- Always state the exact train, validation, holdout, and leaderboard date-index ranges beside every result.
- Clearly separate verified external leaderboard results from local backtests.
- Never call a strategy robust solely because it was positive in folds that were also used repeatedly for candidate selection.
- Report negative results, uncertainty, multiple-testing risk, and regime sensitivity directly.
- Do not claim a recovered historical strategy is exact unless its submitted file or checksum has been verified.

## Change control

- Change one strategy family at a time: baseline, ridge, ALGO, pairs, then any additional sleeve.
- Do not combine changes until each has passed the pre-declared development-only protocol.
- Preserve the last externally best submission before experimenting.
- Never overwrite or delete the externally best version without an independently verified copy.
