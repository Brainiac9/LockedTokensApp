import numpy as np

def simulate_price_path(P0, shocks, price_floor=None):
    prices = [P0]
    for Z in shocks:
        Pt = prices[-1] * np.exp(Z)
        prices.append(max(Pt, price_floor) if price_floor else Pt)
    return prices