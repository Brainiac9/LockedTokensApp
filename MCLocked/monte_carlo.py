from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from MCLocked.strategy import simulate_strategy_weekly
from MCLocked.shocks import generate_shocks

def run_monte_carlo_simulation(P0, num_tokens, discounts, hedge_ratios, funding_rates, T, cliff_weeks, maint_margin_data, shock_params, num_runs=100, sell_frequency='monthly', staking_apy=0.0, price_outlook='bearish'):
    common_shocks = {run: generate_shocks(**shock_params) for run in range(num_runs)}

    def simulate_batch(discount, hedge_ratio, funding_rate, run_batch):
        results = []
        for run in run_batch:
            shocks = common_shocks[run]
            res = simulate_strategy_weekly(
                P0=P0,
                num_tokens_granted=num_tokens,
                discount=discount,
                hedge_ratio=hedge_ratio,
                T=T,
                annual_funding_rate=funding_rate,
                cliff_weeks=cliff_weeks,
                external_shocks=shocks,
                maint_margin_data=maint_margin_data,
                sell_frequency=sell_frequency,
                staking_apy=staking_apy,
                outlook=price_outlook
            )
            # Collect collateral_series from the results
            results.append((discount, hedge_ratio, funding_rate, res["Annualized IRR"], res["Total_PnL"], res["prices"], res["collateral_series"]))
        return results

    tasks = []
    batch_size = 500
    for d in discounts:
        for h in hedge_ratios:
            for f in funding_rates:
                for i in range(0, num_runs, batch_size):
                    batch = list(range(i, min(i + batch_size, num_runs)))
                    tasks.append((d, h, f, batch))

    # Run parallel simulation
    results_raw = Parallel(n_jobs=-1)(
        delayed(simulate_batch)(discount, hedge_ratio, funding_rate, run_batch)
        for discount, hedge_ratio, funding_rate, run_batch in tqdm(tasks, desc="Running Monte Carlo", total=len(tasks))
    )

    # Flatten the results
    flat_results = [item for batch in results_raw for item in batch]

    # Aggregation dictionary
    agg = defaultdict(list)
    for d, h, f, irr, pnl, price_path, collateral_series in flat_results:
        if irr is not None:  # Ensure that only valid results with a non-None IRR are included
            key = (round(d, 4), round(h, 3), round(f, 3))
            agg[key].append((irr, pnl, price_path, collateral_series))

    # Final results and detailed data structures
    results = []
    detailed_prices = defaultdict(list)
    detailed_ann_irr = defaultdict(list)
    detailed_pnl = defaultdict(list)
    detailed_collateral = defaultdict(list)

    # Create DataFrame for boxplot results
    results_df_box = pd.DataFrame([
        {
            "discount": d,
            "hedge_ratio": h,
            "funding_rate": f,
            "Annualized_IRR": irr,
            "Total_PnL": pnl
        }
        for d, h, f, irr, pnl, price_path, collateral_series in flat_results if irr is not None
    ])

    # Populate results and store detailed data
    for key, rows in agg.items():
        d, h, f = key
        irrs, pnls, prices, collaterals = zip(*rows)

        results.append({
            "discount": d,
            "hedge_ratio": h,
            "funding_rate": f,
            "avg_Annualized_IRR": np.mean([x for x in irrs if x is not None]),
            "median_Annualized_IRR": np.median([x for x in irrs if x is not None]),
            "avg_Total_PnL": np.mean(pnls),
            "median_Total_PnL": np.median(pnls),
            "num_runs": len(prices)
        })

        detailed_prices[key] = list(prices)
        detailed_ann_irr[key] = list(irrs)
        detailed_pnl[key] = list(pnls)
        detailed_collateral[key] = list(collaterals)  # Store collateral series

    return pd.DataFrame(results), detailed_prices, detailed_ann_irr, detailed_pnl, pd.DataFrame(results_df_box), detailed_collateral
