import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

from MCLocked.monte_carlo import run_monte_carlo_simulation
from MCLocked.analyze import (
    summarize_price_paths,
    find_best_discount,
    plot_boxplots,
    plot_histograms_for_selected_keys,
    group_and_analyze_collateral
)

# Streamlit config
st.set_page_config(page_title="Locked Token Simulation Dashboard", layout="wide")
st.title("Locked Token Monte Carlo Simulator")

# Sidebar Inputs
st.sidebar.header("Simulation Settings")

# Input method
col1, col2 = st.sidebar.columns(2)
input_method = col1.radio("Input Method", ["Investment Amount", "Number of Tokens"])
if input_method == "Investment Amount":
    investment_amount = col2.number_input("Amount (USD)", value=1_000_000.0, format="%.0f")
    num_tokens = None
else:
    num_tokens = col2.number_input("Number of Tokens", value=470_000, format="%.0f")

# Date range & lock/cliff
dcol1, dcol2 = st.sidebar.columns(2)
start_date = dcol1.date_input("Start Date", value=datetime.today())
end_date   = dcol2.date_input("End Date",   value=datetime.today())
lock_up_weeks = int((end_date - start_date).days / 7)
cliff_weeks   = st.sidebar.number_input("Cliff Weeks (weeks)", min_value=0, value=52)
st.sidebar.write(f"**Lock-Up:** {lock_up_weeks}w   **Cliff:** {cliff_weeks}w")

# Basic parameters
st.sidebar.markdown("---")
P0          = st.sidebar.number_input("Initial Price", value=10.0, format="%.2f")
num_runs    = st.sidebar.number_input("MC Runs", value=100)
staking_apy = st.sidebar.number_input("Staking APY", value=0.0, format="%.2f")
sell_freq   = st.sidebar.selectbox("Sell Frequency", ["weekly", "monthly"])

# Expander for Discount settings
with st.sidebar.expander("Discount Settings", expanded=True):
    disc_min   = st.number_input("Min Discount", 0.0, 1.0, 0.3, format="%.2f")
    disc_max   = st.number_input("Max Discount", 0.0, 1.0, 0.8, format="%.2f")
    disc_count = st.number_input("# Points", 1, 50, 10)
    discounts  = np.linspace(disc_min, disc_max, int(disc_count))

# Expander for Hedge Ratio settings
with st.sidebar.expander("Hedge Ratio Settings", expanded=False):
    hedge_min   = st.number_input("Min Hedge Ratio", 0.0, 1.0, 0.44, format="%.2f")
    hedge_max   = st.number_input("Max Hedge Ratio", 0.0, 1.0, 1.0, format="%.2f")
    hedge_count = st.number_input("# Points", 1, 50, 3)
    hedge_ratios= np.linspace(hedge_min, hedge_max, int(hedge_count))

# Expander for Funding Rate settings
with st.sidebar.expander("Funding Rate Settings", expanded=False):
    fund_min   = st.number_input("Min Funding Rate", -1.0, 1.0, -0.5, format="%.2f")
    fund_max   = st.number_input("Max Funding Rate", -1.0, 1.0, 0.5, format="%.2f")
    fund_count = st.number_input("# Points", 1, 50, 11)
    fund_rates = np.linspace(fund_min, fund_max, int(fund_count))

# Outlook selection
outlook = st.sidebar.selectbox("Price Outlook", ["bearish", "neutral", "bullish"])

margin_tiers = [
    {"notionalFloor": 0, "notionalCap": 10_000, "maintMarginRate": 0.01, "maintAmount": 0},
    {"notionalFloor": 10_000, "notionalCap": 60_000, "maintMarginRate": 0.015, "maintAmount": 50},
    {"notionalFloor": 60_000, "notionalCap": 300_000, "maintMarginRate": 0.02, "maintAmount": 350},
    {"notionalFloor": 300_000, "notionalCap": 600_000, "maintMarginRate": 0.025, "maintAmount": 1850},
    {"notionalFloor": 600_000, "notionalCap": 3_000_000, "maintMarginRate": 0.05, "maintAmount": 16850},
    {"notionalFloor": 3_000_000, "notionalCap": 6_000_000, "maintMarginRate": 0.10, "maintAmount": 166850},
    {"notionalFloor": 6_000_000, "notionalCap": 7_500_000, "maintMarginRate": 0.125, "maintAmount": 316850},
    {"notionalFloor": 7_500_000, "notionalCap": 15_000_000, "maintMarginRate": 0.25, "maintAmount": 1254350},
    {"notionalFloor": 15_000_000, "notionalCap": 30_000_000, "maintMarginRate": 0.50, "maintAmount": 5004350},
]

# Shock parameters
shock = {
    "size": lock_up_weeks,
    "outlook": outlook,
    "annual_vol": 0.5,
    "annual_drift": None,
    "jump_prob": 0.02,
    "jump_scale": 0.2
}

# Run Simulation button
token_run = st.sidebar.button("Run Simulation")
if token_run:
    if input_method == "Investment Amount":
        num_tokens = investment_amount / (P0 * (1 - discounts[0]))
    df, d_pr, d_irr, d_pnl, df_box, d_coll = run_monte_carlo_simulation(
        P0=P0,
        num_tokens=num_tokens,
        discounts=discounts,
        hedge_ratios=hedge_ratios,
        funding_rates=fund_rates,
        sell_frequency=sell_freq,
        staking_apy=staking_apy,
        T=lock_up_weeks,
        cliff_weeks=cliff_weeks,
        maint_margin_data=margin_tiers,
        shock_params=shock,
        num_runs=num_runs,
        price_outlook=outlook
    )
    st.session_state.update({
        "res_df": df,
        "d_pr": d_pr,
        "d_irr": d_irr,
        "d_pnl": d_pnl,
        "df_box": df_box,
        "d_coll": d_coll
    })

# Display results
if "res_df" in st.session_state:
    df     = st.session_state["res_df"]
    d_pr   = st.session_state["d_pr"]
    d_irr  = st.session_state["d_irr"]
    d_pnl  = st.session_state["d_pnl"]
    df_box = st.session_state["df_box"]
    d_coll = st.session_state["d_coll"]

    # Price Paths
    st.subheader("Price Paths")
    stats, avg, fig = summarize_price_paths(d_pr)
    stats_r = {k: round(v,2) for k,v in stats.items()}
    st.json(stats_r)
    st.pyplot(fig)

    # IRR & PnL Summary
    st.subheader("IRR & PnL Summary")
    rows = []
    for key, irr_list in d_irr.items():
        pnl_list = d_pnl[key]
        if not irr_list or not pnl_list:
            continue
        disc, hr, fr = key
        rows.append({
            "Discount": disc,
            "Hedge Ratio": hr,
            "Annualized Funding Rate": fr,
            "Mean IRR": np.mean(irr_list),
            "Median IRR": np.median(irr_list),
            "Mean PnL": np.mean(pnl_list),
            "Median PnL": np.median(pnl_list)
        })
    sum_df = pd.DataFrame(rows)
    disp_df = sum_df.copy()
    for col in ["Discount", "Hedge Ratio", "Annualized Funding Rate"]:
        disp_df[col] = disp_df[col].apply(lambda x: f"{x:.2%}")
    disp_df["Mean IRR"] = disp_df["Mean IRR"].apply(lambda x: f"{x:.2%}")
    disp_df["Median IRR"] = disp_df["Median IRR"].apply(lambda x: f"{x:.2%}")
    disp_df["Mean PnL"] = disp_df["Mean PnL"].apply(lambda x: f"{x:,.2f}")
    disp_df["Median PnL"] = disp_df["Median PnL"].apply(lambda x: f"{x:,.2f}")
    st.dataframe(disp_df, use_container_width=True)

    # Boxplots
    st.subheader("Boxplots")
    sel_disc = st.selectbox("Choose Discount", sorted(df["discount"].unique()))
    plot_boxplots(df_box, sel_disc)

    # Collateral Stats
    st.subheader("Collateral Stats")
    collat_df = group_and_analyze_collateral(d_coll)
    st.dataframe(collat_df, use_container_width=True)

    # Histograms
    st.subheader("Histograms of IRR & PnL for Selected Scenarios")
    labels = {key: f"Hedge Ratio={key[1]:.2%} | Funding Rate={key[2]:.2%} | Discount={key[0]:.2%}" for key in d_irr}
    inv_labels = {v: k for k, v in labels.items()}
    selected = st.multiselect("Select Scenario Combinations", options=list(inv_labels.keys()), default=list(inv_labels.keys())[:3])
    selected_keys = [inv_labels[label] for label in selected if label in inv_labels]
    if selected_keys:
        plot_histograms_for_selected_keys(d_irr, d_pnl, selected_keys)
    else:
        st.info("Select at least one scenario to display histograms.")

    # Best Discount Finder
    st.subheader("Recommended Discounts by Scenario")
    target_irr = st.slider(
        """
For each set of hedge ratios and funding rates simulated determines the lowest discount
that achieves in median at least the target IRR. If none meet the target, pick
the discount whose median IRR is closest to the target.
""",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        format="%.2f"
    )
    find_best_discount(df, target_irr)
else:
    st.warning("Run simulation first.")
