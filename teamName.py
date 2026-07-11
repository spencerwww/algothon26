import numpy as np

nInst = 51

# correlated trio: BLBT, MHRM, EAFC (columns in prices.txt)
TRIO = [41, 49, 50]

LOOKBACK = 40
ENTRY_Z = 1.0
EXIT_Z = 0.25
DLR_POS_LIMIT = 10_000

currentPos = np.zeros(nInst, dtype=int)


def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape

    if nt < LOOKBACK + 1:
        return np.zeros(nins, dtype=int)

    # log prices of the trio over the lookback window
    logP = np.log(prcSoFar[TRIO, -LOOKBACK:])

    # normalize each series by its own rolling mean level so they're comparable,
    # then measure each one against the trio's average
    demeaned = logP - logP.mean(axis=1, keepdims=True)
    spread = demeaned - demeaned.mean(axis=0, keepdims=True)

    mu = spread.mean(axis=1)
    sd = spread.std(axis=1)
    sd[sd < 1e-8] = 1e-8
    z = (spread[:, -1] - mu) / sd

    newPos = currentPos.copy()
    for k, inst in enumerate(TRIO):
        price = prcSoFar[inst, -1]
        maxShares = int(DLR_POS_LIMIT / price)
        if z[k] > ENTRY_Z:
            newPos[inst] = -maxShares  # rich vs basket -> short
        elif z[k] < -ENTRY_Z:
            newPos[inst] = maxShares  # cheap vs basket -> long
        elif abs(z[k]) < EXIT_Z:
            newPos[inst] = 0
        # otherwise hold the existing position to avoid churn

    currentPos = newPos
    return currentPos
