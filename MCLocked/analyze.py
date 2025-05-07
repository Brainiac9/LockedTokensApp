import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def summarize_price_paths(detailed_prices):
    """
    Summarize concatenated price paths across all runs.
    Returns statistics dict, average path array, and a matplotlib figure.
    """
    # Extract one scenario's list of price paths
    price_paths = next(iter(detailed_prices.values()), [])
    if not price_paths:
        return {}, [], None

    all_prices = np.concatenate(price_paths)
    stats = {
        "Mean": np.mean(all_prices),
        "Median": np.median(all_prices),
        "Min": np.min(all_prices),
        "Max": np.max(all_prices),
        "Std Dev": np.std(all_prices),
        "5th %ile": np.percentile(all_prices, 5),
        "95th %ile": np.percentile(all_prices, 95)
    }

    avg_path = np.mean(price_paths, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for run in price_paths:
        ax.plot(run, alpha=0.3, color='gray')
    ax.plot(avg_path, label='Average Path', color='black', linewidth=2)
    ax.set(title="Simulated Price Paths", xlabel="Week", ylabel="Price")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    return stats, avg_path, fig


def analyze_summary(detailed_ann_irr, detailed_pnl):
    """
    Build and return a summary DataFrame of statistics for each (disc, hedge, fund) key.
    """
    rows = []
    for (disc, hedge, fund), irr_list in detailed_ann_irr.items():
        pnl_list = detailed_pnl.get((disc, hedge, fund), [])
        if not irr_list or not pnl_list:
            continue
        arr_irr = np.array([x for x in irr_list if x is not None])
        arr_pnl = np.array(pnl_list)
        rows.append({
            "Discount": disc,
            "Hedge Ratio": hedge,
            "Funding Rate": fund,
            "Mean IRR": np.mean(arr_irr),
            "Median IRR": np.median(arr_irr),
            "5th %ile IRR": np.percentile(arr_irr, 5),
            "95th %ile IRR": np.percentile(arr_irr, 95),
            "Mean PnL": np.mean(arr_pnl),
            "Median PnL": np.median(arr_pnl),
            "5th %ile PnL": np.percentile(arr_pnl, 5),
            "95th %ile PnL": np.percentile(arr_pnl, 95),
        })

    df = pd.DataFrame(rows)
    df = df[['Discount','Hedge Ratio','Funding Rate',
             'Mean IRR','Median IRR','5th %ile IRR','95th %ile IRR',
             'Mean PnL','Median PnL','5th %ile PnL','95th %ile PnL']]
    return df


def find_best_discount(results_df: pd.DataFrame, target_irr: float) -> pd.DataFrame:
    """
    For each (hedge_ratio, funding_rate), determine the lowest discount
    that achieves at least the target IRR. If none meet the target, pick
    the discount whose median IRR is closest to the target.
    Display and return a DataFrame of recommended discounts.
    """
    combos = []
    grouped = results_df.groupby(['hedge_ratio','funding_rate'])
    for (hedge, fund), group in grouped:
        # Prepare DataFrame with median IRR per discount level
        med_df = group.groupby('discount')['median_Annualized_IRR'].median().reset_index()
        # Filter those meeting or exceeding target
        meets = med_df[med_df['median_Annualized_IRR'] >= target_irr]
        if not meets.empty:
            # pick smallest discount that meets target
            best_disc = meets.sort_values('discount').iloc[0]
        else:
            # fallback: closest to target
            med_df['irr_diff'] = (med_df['median_Annualized_IRR'] - target_irr).abs()
            best_disc = med_df.nsmallest(1, 'irr_diff').iloc[0]
        # record recommendation
        combos.append({
            'Hedge Ratio': hedge,
            'Funding Rate': fund,
            'Recommended Discount': best_disc['discount'],
            'Expected Median IRR': best_disc['median_Annualized_IRR'],
            # get corresponding PnL
            'Expected Median PnL': group[group['discount'] == best_disc['discount']]['median_Total_PnL'].median()
        })
    best_df = pd.DataFrame(combos)
    if best_df.empty:
        st.warning("No scenarios available. Run the simulation first.")
        return best_df
    best_df = best_df.set_index(['Hedge Ratio','Funding Rate'])
    # Display
    disp = best_df.copy()
    disp['Recommended Discount'] = disp['Recommended Discount'].apply(lambda x: f"{x:.2%}")
    disp['Expected Median IRR']  = disp['Expected Median IRR'].apply(lambda x: f"{x:.2%}")
    disp['Expected Median PnL']  = disp['Expected Median PnL'].apply(lambda x: f"{x:,.2f}")
    st.subheader("Recommended Discounts by Scenario")
    st.table(disp)
    return best_df


def plot_boxplots(results_df_box: pd.DataFrame, target_discount: float):
    filtered = results_df_box[results_df_box['discount']==target_discount]
    if filtered.empty:
        st.warning("No data for this discount.")
        return
    fig1, ax1 = plt.subplots(figsize=(12,6))
    sns.boxplot(data=filtered, x='hedge_ratio', y='Annualized_IRR', hue='funding_rate', ax=ax1)
    ax1.set(title="IRR Boxplot",xlabel="Hedge Ratio",ylabel="IRR")
    st.pyplot(fig1)
    fig2, ax2 = plt.subplots(figsize=(12,6))
    sns.boxplot(data=filtered, x='hedge_ratio', y='Total_PnL', hue='funding_rate', ax=ax2)
    ax2.set(title="Total PnL Boxplot",xlabel="Hedge Ratio",ylabel="Total PnL")
    st.pyplot(fig2)


def group_and_analyze_collateral(detailed_collateral):
    from collections import defaultdict
    rows = []
    grouped = defaultdict(list)
    for (disc, hedge, fund), paths in detailed_collateral.items():
        for series in paths:
            grouped[(hedge,fund)].extend(series)
    for (hedge,fund), vals in grouped.items():
        arr = np.array(vals)
        rows.append({
            'Hedge Ratio': hedge,
            'Funding Rate': fund,
            'Mean': np.mean(arr),
            'Median': np.median(arr),
            'Min': np.min(arr),
            'Max': np.max(arr),
            'Std': np.std(arr),
            '5th %ile': np.percentile(arr,5),
            '95th %ile': np.percentile(arr,95)
        })
    df = pd.DataFrame(rows)
    return df.set_index(['Hedge Ratio','Funding Rate'])


def plot_histograms_for_selected_keys(detailed_ann_irr: dict, detailed_pnl: dict, selected_keys: list):
    for key in selected_keys:
        irr_vals = detailed_ann_irr.get(key, [])
        pnl_vals = detailed_pnl.get(key, [])
        if not irr_vals or not pnl_vals:
            continue

        discount, hedge, fund = key
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.hist(irr_vals, bins=30, color='skyblue', edgecolor='black')
        ax1.set_title(f"IRR Histogram\nDiscount: {discount:.2%}, Hedge Ratio: {hedge:.2%}, Funding Rates: {fund:.2%}")
        ax1.set_xlabel("IRR")
        ax1.set_ylabel("Frequency")
        ax1.grid(True)

        ax2.hist(pnl_vals, bins=30, color='salmon', edgecolor='black')
        ax2.set_title(f"PnL Histogram\nDiscount: {discount:.2%}, Hedge: {hedge:.2%}, Fund: {fund:.2%}")
        ax2.set_xlabel("Total PnL")
        ax2.set_ylabel("Frequency")
        ax2.grid(True)

        plt.tight_layout()
        st.pyplot(fig)
