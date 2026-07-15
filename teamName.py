import numpy as np

nInst = 51

# correlated baskets (column indices in prices.txt)
BASKETS = [
    [41, 49, 50],   # BLBT, MHRM, EAFC
    [9, 38],        # DUCT, HRND
    [1, 20],        # AENO, NWIG
    [10, 46],       # SMAH, ILVX
    [13, 45],       # EORC, NGTE
    [8, 27],        # HUXZ, ACAC
    [31, 43],       # ACIX, ITPA
]

LOOKBACK = 36
ENTRY_Z = 1.0
EXIT_Z = 0.2
DLR_POS_LIMIT = 10_000

currentPos = np.zeros(nInst, dtype=int)


def _trade_basket(prcSoFar, basket, pos):
    """Update positions for one correlated basket via z-score mean reversion."""
    logP = np.log(prcSoFar[basket, -LOOKBACK:])
    demeaned = logP - logP.mean(axis=1, keepdims=True)
    spread = demeaned - demeaned.mean(axis=0, keepdims=True)

    mu = spread.mean(axis=1)
    sd = spread.std(axis=1)
    sd[sd < 1e-8] = 1e-8
    z = (spread[:, -1] - mu) / sd

    for k, inst in enumerate(basket):
        maxShares = int(DLR_POS_LIMIT / prcSoFar[inst, -1])
        if z[k] > ENTRY_Z:
            pos[inst] = -maxShares  # rich vs basket -> short
        elif z[k] < -ENTRY_Z:
            pos[inst] = maxShares  # cheap vs basket -> long
        elif abs(z[k]) < EXIT_Z:
            pos[inst] = 0
        # otherwise hold existing position to avoid churn


def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape

    if nt < LOOKBACK + 1:
        return np.zeros(nins, dtype=int)

    newPos = currentPos.copy()
    for basket in BASKETS:
        _trade_basket(prcSoFar, basket, newPos)

    currentPos = newPos
    return currentPos
