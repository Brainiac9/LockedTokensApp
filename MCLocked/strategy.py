import numpy as np
import pandas as pd
from MCLocked.pricing import simulate_price_path
from MCLocked.margin import get_required_margin

try:
    import numpy_financial as npf
except ImportError:
    npf = None



def simulate_strategy_weekly(
    P0,
    investment_amount=None,
    num_tokens_granted=None,
    discount=0.0,
    hedge_ratio=1.0,
    T=156,
    annual_funding_rate=0.0,
    cliff_weeks=0,
    price_floor=None,
    seed=None,
    external_shocks=None,
    maint_margin_data=None,
    staking_apy=0.0,
    sell_frequency="weekly",
    outlook="bearish"
):
    if seed is not None:
        np.random.seed(seed)

    # Initial setup
    if (investment_amount is None) == (num_tokens_granted is None):
        raise ValueError("Provide either investment_amount or num_tokens_granted, not both.")

    Q = num_tokens_granted if num_tokens_granted is not None else investment_amount / (P0 * (1 - discount))
    token_cost = investment_amount if investment_amount is not None else Q * P0 * (1 - discount)

    prices = prices = simulate_price_path(P0, external_shocks, price_floor=price_floor)
    hedged_tokens = hedge_ratio * Q
    notional = hedged_tokens * P0
    collateral = get_required_margin(notional, maint_margin_data) if maint_margin_data else notional / 3.0
    cf0 = -token_cost - collateral

    details = [{
        "time": 0.0, "week": 0, "price": P0,
        "remaining_long": Q, "remaining_short": hedged_tokens,
        "collateral": collateral, "collateral_flow": -collateral,
        "funding_cash_flow": 0.0, "hedged_sale_flow": 0.0, "hedged_cover_flow": 0.0,
        "unhedged_sale_flow": 0.0, "vesting_total_flow": 0.0,
        "staking_tokens_tokens": 0.0, "total_cf": cf0, "note": "Initial Setup"
    }]

    remaining_long, remaining_short = Q, hedged_tokens
    total_funding_cost = 0.0
    accumulated_vested = 0.0
    vest_per_week = Q / max(T - cliff_weeks, 1)

    for t in range(1, T + 1):
        current_price = prices[t]
        price_prev = prices[t - 1]
        time_year = t / 52.0

        notional = remaining_short * price_prev
        new_collateral = get_required_margin(notional, maint_margin_data) if maint_margin_data else notional / 3.0
        collateral_flow = -(new_collateral - collateral)
        collateral = new_collateral

        funding_cost = (annual_funding_rate / 52.0) * notional
        total_funding_cost += funding_cost

        staking_reward = (staking_apy / 52.0) * remaining_long
        remaining_long += staking_reward

        vested = vest_per_week if t > cliff_weeks else 0.0
        accumulated_vested += vested
        sell = vested > 0 if sell_frequency == "weekly" else (t % 4 == 0 or t == T)

        if sell:
            hedged = min(accumulated_vested, remaining_short)
            hedged_sale = hedged * current_price
            hedged_cover = hedged * (P0 - current_price)
            unhedged = accumulated_vested - hedged
            unhedged_sale = unhedged * current_price
            vesting_flow = hedged_sale + hedged_cover + unhedged_sale
            remaining_short -= hedged
            remaining_long -= accumulated_vested
            accumulated_vested = 0.0
        else:
            hedged_sale = hedged_cover = unhedged_sale = vesting_flow = 0.0

        total_cf = collateral_flow + funding_cost + vesting_flow

        details.append({
            "time": time_year, "week": t, "price": current_price,
            "remaining_long": remaining_long, "remaining_short": remaining_short,
            "collateral": collateral, "collateral_flow": collateral_flow,
            "funding_cash_flow": funding_cost,
            "hedged_sale_flow": hedged_sale,
            "hedged_cover_flow": hedged_cover,
            "unhedged_sale_flow": unhedged_sale,
            "vesting_total_flow": vesting_flow,
            "staking_tokens_tokens": staking_reward,
            "total_cf": total_cf,
            "note": f"Week {t} - vested: {vested:.2f}, accumulated: {accumulated_vested:.2f}"
        })

    # Final settlement
    final_price = prices[-1]
    final_sale = remaining_long * final_price
    final_short_pnl = remaining_short * (P0 - final_price)
    collateral_release = collateral
    final_total_cf = final_sale + final_short_pnl + collateral_release

    details.append({
        "time": T / 52.0, "week": T, "price": final_price,
        "remaining_long": 0.0, "remaining_short": 0.0,
        "collateral": 0.0, "collateral_flow": collateral_release,
        "funding_cash_flow": 0.0,
        "hedged_sale_flow": 0.0,
        "hedged_cover_flow": final_short_pnl,
        "unhedged_sale_flow": final_sale,
        "vesting_total_flow": final_sale + final_short_pnl,
        "staking_tokens_tokens": 0.0,
        "total_cf": final_total_cf,
        "note": "Final Settlement"
    })

    df = pd.DataFrame(details)
    cash_flows = [(row["time"], row["total_cf"]) for _, row in df.iterrows()]
    values_arr = np.array([cf for _, cf in cash_flows])
    irr = npf.irr(values_arr) if npf else None
    annual_irr = (1 + irr) ** 52 - 1 if irr is not None else None
    total_pnl = np.sum(values_arr)

    return {
        "prices": prices,
        "collateral_series": df["collateral"].tolist(),
        "cash_flows": cash_flows,
        "IRR": irr,
        "Total_PnL": total_pnl,
        "Total_Funding_Cost": total_funding_cost,
        "simulation_details": df,
        "Annualized IRR": annual_irr
    }
