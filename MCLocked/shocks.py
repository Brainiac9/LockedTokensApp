import numpy as np
from scipy.stats import t as student_t


def generate_shocks(
    size=156,
    outlook="neutral",  # "bullish", "bearish", or "neutral"
    annual_vol=0.6,
    annual_drift=None,  # if None, inferred from outlook
    jump_prob=0.01,
    jump_scale=0.3,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    # Drift settings (annual return assumptions)
    if annual_drift is None:
        annual_drift = {
            "bearish": -0.3, 
            "neutral": 0.0,
            "bullish": 0.3
        }.get(outlook, 0.0)

    # Convert annual to weekly
    dt = 1 / 52
    mu = annual_drift
    sigma = annual_vol

    # Basic normal shocks
    shocks = np.random.normal(loc=(mu - 0.5 * sigma**2) * dt, scale=sigma * np.sqrt(dt), size=size)

    # Add jumps
    jumps = np.random.normal(0, jump_scale, size=size)
    shock_mask = np.random.rand(size) < jump_prob
    shocks += jumps * shock_mask

    return shocks
