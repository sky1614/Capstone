import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from math import ceil, sqrt, erf
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -----------------------------
# Practical MSME settings
# -----------------------------
Z = 1.645  # 95% service level
ON_ORDER_BUFFER_PCT = 0.30
SAFETY_CAP_PCT = 0.60
RISK_HIGH = 0.30
RISK_MED = 0.10

st.set_page_config(page_title="AI Revenue Co-Pilot (Phase 1)", layout="wide")

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_tally_csv(path: str):
    t = pd.read_csv(path)
    # Expected columns: month, sales_amount
    t["month"] = pd.to_datetime(t["month"], errors="coerce")
    t["sales_amount"] = pd.to_numeric(t["sales_amount"], errors="coerce")
    t = t.dropna(subset=["month", "sales_amount"]).sort_values("month")
    return t


def make_canonical(df: pd.DataFrame):
    # Works for weekly and daily datasets
    date_col = "week" if "week" in df.columns else "date"
    sku_col = "sku"
    units_col = "units_sold"
    stock_col = "stock_available"
    deliv_col = "delivery_days"

    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, sku_col, units_col]).copy()

    # Normalize to Monday-start weekly buckets
    if date_col == "date":
        tmp["week"] = tmp[date_col].dt.to_period("W-MON").dt.start_time
    else:
        tmp["week"] = tmp[date_col].dt.to_period("W-MON").dt.start_time

    for c in [units_col, stock_col, deliv_col]:
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    canon = (tmp.groupby(["week", sku_col], as_index=False)
               .agg({
                   units_col: "sum",
                   stock_col: "last",
                   deliv_col: "median"
               })
               .rename(columns={
                   sku_col: "sku",
                   units_col: "y",
                   stock_col: "stock",
                   deliv_col: "delivery_days"
               }))

    canon = canon.dropna(subset=["week", "sku", "y"]).sort_values(["sku", "week"]).reset_index(drop=True)
    return canon

def demand_series(df_sku):
    s = df_sku.sort_values("week").set_index("week")["y"]
    s = s.resample("W-MON").sum().fillna(0.0)
    return s

def lead_time_weeks(df_sku):
    ld = float(df_sku["delivery_days"].median()) if df_sku["delivery_days"].notna().any() else 7.0
    if np.isnan(ld) or ld <= 0:
        ld = 7.0
    return max(1, int(ceil(ld / 7.0)))

def current_stock(df_sku):
    last = df_sku.sort_values("week").tail(1)["stock"].values
    return float(last[0]) if len(last) and not np.isnan(last[0]) else float("nan")

def ets_forecast_next(s: pd.Series) -> float:
    try:
        fit = ExponentialSmoothing(
            s, trend="add", seasonal=None, initialization_method="estimated"
        ).fit(optimized=True)
        return float(fit.forecast(1).iloc[0])
    except Exception:
        return float(s.iloc[-1])

def build_action_feed_practical(canon_df: pd.DataFrame, top_n=15):
    top_skus = (canon_df.groupby("sku")["y"].sum()
                .sort_values(ascending=False).head(top_n).index.tolist())

    rows = []
    for sku in top_skus:
        df_sku = canon_df[canon_df["sku"] == sku].copy()
        s = demand_series(df_sku)
        if len(s) < 20:
            continue

        f_next = ets_forecast_next(s)

        recent = s.iloc[-12:] if len(s) >= 12 else s
        sigma = float(recent.std())

        lt = lead_time_weeks(df_sku)
        stock_now = current_stock(df_sku)

        safety_raw = Z * sigma * sqrt(lt)
        safety_cap = SAFETY_CAP_PCT * max(0.0, f_next * lt)
        safety_stock = min(safety_raw, safety_cap)

        on_order = ON_ORDER_BUFFER_PCT * max(0.0, f_next * lt)

        if np.isnan(stock_now):
            stock_position = float("nan")
        else:
            stock_position = stock_now + on_order

        order_up_to = max(0.0, f_next * lt + safety_stock)

        if np.isnan(stock_position):
            order_qty = order_up_to
            coverage_weeks = None
            risk_before = None
            risk_after = None
            risk_flag = "UNKNOWN_STOCK"
        else:
            order_qty = max(0.0, order_up_to - stock_position)

            coverage_weeks = (stock_now / max(1e-9, f_next)) if f_next > 0 else float("inf")

            mean_lt = f_next * lt
            std_lt = max(1e-9, sigma * sqrt(lt))

            z_before = (stock_position - mean_lt) / std_lt
            service_before = norm_cdf(z_before)
            risk_before = 1.0 - service_before

            stock_position_after = stock_position + order_qty
            z_after = (stock_position_after - mean_lt) / std_lt
            service_after = norm_cdf(z_after)
            risk_after = 1.0 - service_after

            if risk_after >= RISK_HIGH:
                risk_flag = "HIGH"
            elif risk_after >= RISK_MED:
                risk_flag = "MED"
            else:
                risk_flag = "NORMAL"

        priority_score = (0 if risk_before is None else (risk_before * max(0.0, f_next)))
        lost_sales_avoided = (0 if risk_before is None else (risk_before * max(0.0, f_next)))  # proxy

        rows.append({
            "sku": sku,
            "forecast_next_week_units": round(f_next, 2),
            "lead_time_weeks": lt,
            "sigma_weekly": round(sigma, 2),
            "safety_stock_used": round(safety_stock, 2),
            "assumed_on_order": round(on_order, 2),
            "current_stock_units": None if np.isnan(stock_now) else round(stock_now, 2),
            "stock_position_proxy": None if np.isnan(stock_position) else round(stock_position, 2),
            "order_up_to_units": round(order_up_to, 2),
            "recommended_order_qty": round(order_qty, 2),
            "coverage_weeks_on_hand": None if coverage_weeks is None else round(coverage_weeks, 2),
            "stockout_risk_before_pct": None if risk_before is None else round(100*risk_before, 1),
            "stockout_risk_after_pct": None if risk_after is None else round(100*risk_after, 1),
            "priority_score": round(priority_score, 2),
            "lost_sales_avoided_proxy": round(lost_sales_avoided, 2),
            "risk_flag": risk_flag,
        })

    out = pd.DataFrame(rows)
    risk_rank = {"HIGH":0, "MED":1, "NORMAL":2, "UNKNOWN_STOCK":3}
    out["risk_rank"] = out["risk_flag"].map(risk_rank).fillna(9)
    out = out.sort_values(["risk_rank","priority_score","recommended_order_qty"], ascending=[True, False, False]) \
             .drop(columns=["risk_rank"])
    return out, top_skus

# -----------------------------
# UI
# -----------------------------
st.title("AI Revenue Co-Pilot — Phase 1 (Forecast → Reorder Decisions)")

with st.sidebar:
    st.header("Data")
    default_path = "FMCG_2022_2024.csv"
    data_path = st.text_input("CSV path", value=default_path)
    st.subheader("Real Client (Tally)")
    tally_path = st.text_input("Tally CSV path", value="tally_sales.csv")
    top_n = st.slider("Top SKUs to analyze", 5, 30, 15)
    st.header("Practical knobs")
    on_order_pct = st.slider("Assumed on-order %", 0.0, 1.0, float(ON_ORDER_BUFFER_PCT), 0.05)
    safety_cap_pct = st.slider("Safety stock cap (% of forecast*LT)", 0.0, 1.0, float(SAFETY_CAP_PCT), 0.05)

# Apply sliders
ON_ORDER_BUFFER_PCT = on_order_pct
SAFETY_CAP_PCT = safety_cap_pct

try:
    df_raw = load_data(data_path)
except Exception as e:
    st.error(f"Could not load file at '{data_path}'. Error: {e}")
    st.stop()

canon = make_canonical(df_raw)

# Quick stats
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (weekly x SKU)", f"{len(canon):,}")
c2.metric("Unique SKUs", f"{canon['sku'].nunique():,}")
c3.metric("Weeks", f"{canon['week'].nunique():,}")
c4.metric("Delivery days (median)", f"{canon['delivery_days'].median():.1f}")

action_feed, top_skus = build_action_feed_practical(canon, top_n=top_n)

# KPI strip
k1, k2, k3, k4 = st.columns(4)
k1.metric("MED/HIGH items", int((action_feed["risk_flag"].isin(["MED","HIGH"])).sum()))
k2.metric("Total order qty (top N)", f"{action_feed['recommended_order_qty'].sum():.0f}")
k3.metric("Avg risk after (%)", f"{action_feed['stockout_risk_after_pct'].dropna().mean():.1f}")
k4.metric("Avg coverage (weeks)", f"{action_feed['coverage_weeks_on_hand'].dropna().mean():.2f}")

tabs = st.tabs(["Action Feed", "SKU Drilldown", "Real Client (Tally)"])

with tabs[0]:
    st.subheader("Weekly Action Feed (Approve/Reject-ready)")
    st.markdown("### Business Impact (Estimated)")

    # Use the full action_feed (before risk filtering) for true KPI view
    tmp_imp = action_feed.copy()
    tmp_imp["risk_before"] = tmp_imp["stockout_risk_before_pct"].fillna(0) / 100.0
    tmp_imp["risk_after"]  = tmp_imp["stockout_risk_after_pct"].fillna(0) / 100.0

    # Units avoided proxy = forecast * (risk_before - risk_after)
    tmp_imp["units_avoided"] = (tmp_imp["forecast_next_week_units"] * (tmp_imp["risk_before"] - tmp_imp["risk_after"])).clip(lower=0)
    units_avoided = float(tmp_imp["units_avoided"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MED/HIGH items", int((action_feed["risk_flag"].isin(["MED","HIGH"])).sum()))
    c2.metric("Total order qty (top N)", f"{action_feed['recommended_order_qty'].sum():.0f}")
    c3.metric("Avg risk after (%)", f"{action_feed['stockout_risk_after_pct'].dropna().mean():.1f}")
    c4.metric("Potential lost sales avoided (units)", f"{units_avoided:,.0f}")

    avg_price = st.slider("Avg selling price per unit (₹) — impact simulator", 1, 500, 50)
    revenue_protected = units_avoided * avg_price
    st.metric("Estimated Revenue Protected (₹)", f"{revenue_protected:,.0f}")
    st.caption("Estimate = forecast × (risk_before − risk_after). Replace avg price with real SKU pricing later.")

    st.markdown("### Weekly Action Feed (Approve/Reject + Decision Log)")

    risk_filter = st.multiselect(
        "Filter risk",
        ["HIGH","MED","NORMAL","UNKNOWN_STOCK"],
        default=["HIGH","MED","NORMAL"]
    )

    view = action_feed[action_feed["risk_flag"].isin(risk_filter)].copy()

    # --- Decision workflow (persistent in session) ---
    if "decision_log" not in st.session_state:
        st.session_state.decision_log = {}

    # Add decision + note columns
    if "decision" not in view.columns:
        view["decision"] = "PENDING"
    if "note" not in view.columns:
        view["note"] = ""

    # Populate from session_state
    for i, r in view.iterrows():
        sku = r["sku"]
        if sku in st.session_state.decision_log:
            view.at[i, "decision"] = st.session_state.decision_log[sku].get("decision", "PENDING")
            view.at[i, "note"] = st.session_state.decision_log[sku].get("note", "")

    # Use Streamlit data editor for approve/reject
    edited = st.data_editor(
        view,
        use_container_width=True,
        height=520,
        column_config={
            "decision": st.column_config.SelectboxColumn(
                "decision",
                options=["APPROVE", "REJECT", "DEFER", "PENDING"],
                required=True,
            ),
            "note": st.column_config.TextColumn("note"),
        },
        disabled=[
            "sku","forecast_next_week_units","lead_time_weeks","sigma_weekly",
            "safety_stock_used","assumed_on_order","current_stock_units",
            "stock_position_proxy","order_up_to_units","recommended_order_qty",
            "coverage_weeks_on_hand","stockout_risk_before_pct","stockout_risk_after_pct",
            "priority_score","lost_sales_avoided_proxy","risk_flag"
        ],
    )

    # Save decisions back to session_state
    for _, r in edited.iterrows():
        sku = r["sku"]
        st.session_state.decision_log[sku] = {
            "decision": r["decision"],
            "note": r.get("note", ""),
            "timestamp": datetime.now().isoformat(timespec="seconds")
        }

    # Create downloadable decision log
    log_rows = []
    for sku, v in st.session_state.decision_log.items():
        log_rows.append({"sku": sku, **v})
    log_df = pd.DataFrame(log_rows).sort_values(["timestamp","sku"], ascending=[False, True]) if log_rows else pd.DataFrame(columns=["sku","decision","note","timestamp"])

    st.download_button(
        "Download Decision Log CSV",
        data=log_df.to_csv(index=False).encode("utf-8"),
        file_name="decision_log.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Action Feed (with decisions) CSV",
        data=edited.to_csv(index=False).encode("utf-8"),
        file_name="action_feed_with_decisions.csv",
        mime="text/csv"
    )


with tabs[1]:
    st.subheader("SKU Drilldown")

    # --- Select SKU from FULL dataset, not filtered view ---
    all_skus = sorted(df["sku"].unique())
    sel_sku = st.selectbox("Select SKU", all_skus)

    # --- Filter full history for selected SKU ---
    sku_hist = df[df["sku"] == sel_sku].sort_values("week")

    if sku_hist.empty:
        st.warning("No data available for this SKU.")
        st.stop()

    # --- Prepare series ---
    y = sku_hist.set_index("week")["units_sold"]

    # --- Forecast (same model as Tab 1) ---
    try:
        model = ExponentialSmoothing(y, trend="add", seasonal=None)
        fit = model.fit(optimized=True)
        f_next = fit.forecast(1).iloc[0]
    except Exception:
        f_next = y.iloc[-1]

    # --- Plot 1: Actual sales + forecast ---
    st.markdown("### Sales Trend (Actual vs Forecast)")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.index, y.values, label="Actual sales")
    ax.axhline(f_next, linestyle="--", color="orange", label="Next-week forecast")
    ax.legend()
    ax.set_ylabel("Units sold")
    st.pyplot(fig)

    # --- Stock numbers ---
    current_stock = sku_hist["stock_available"].iloc[-1]
    lead_time = max(1, int(np.ceil(sku_hist["delivery_days"].median() / 7)))

    sigma = y.tail(12).std()
    safety_stock = min(1.645 * sigma * np.sqrt(lead_time), 0.6 * f_next * lead_time)
    order_up_to = f_next * lead_time + safety_stock

    on_order = 0.3 * f_next * lead_time

    # --- Plot 2: Stock position vs target ---
    st.markdown("### Stock Position vs Target")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(
        ["On-hand", "On-order (assumed)", "Target (order-up-to)"],
        [current_stock, on_order, order_up_to]
    )
    st.pyplot(fig2)

    # --- Explain numbers ---
    st.markdown("### Key Numbers (Explainable)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast next week", f"{f_next:.0f}")
    c2.metric("Current stock", f"{current_stock:.0f}")
    c3.metric("Safety stock", f"{safety_stock:.0f}")
    c4.metric("Recommended target", f"{order_up_to:.0f}")

    st.caption(
        "This view explains *why* the system recommends a certain order quantity "
        "by showing demand trend, stock position, and safety buffer."
    )

    
with tabs[2]:
    st.subheader("Real Client Validation (Tally)")

    try:
        tally_df = load_tally_csv(tally_path)
        # MoM growth + run-rate metrics
        t = tally_df.copy()
        t["mom_growth_pct"] = t["sales_amount"].pct_change() * 100.0

        last_3 = t["sales_amount"].tail(3)
        run_rate = float(last_3.mean() * 12) if len(last_3) >= 1 else float("nan")

        best_row = t.loc[t["sales_amount"].idxmax()]
        worst_row = t.loc[t["sales_amount"].idxmin()]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sales (₹)", f"{t['sales_amount'].sum():,.0f}")
        c2.metric("Run-rate (₹/year, last 3 mo avg × 12)", f"{run_rate:,.0f}")
        c3.metric("Best Month", f"{best_row['month'].strftime('%Y-%m')} | ₹{best_row['sales_amount']:,.0f}")
        c4.metric("Worst Month", f"{worst_row['month'].strftime('%Y-%m')} | ₹{worst_row['sales_amount']:,.0f}")

        st.markdown("**Month-over-Month Growth (%)**")
        st.dataframe(
            t[["month","sales_amount","mom_growth_pct"]].assign(
                month=lambda x: x["month"].dt.strftime("%Y-%m")
            ),
            use_container_width=True,
            height=260
        )

    except Exception as e:
        st.error(f"Could not load Tally CSV at '{tally_path}'. Error: {e}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Months", len(tally_df))
    c2.metric("Total Sales (₹)", f"{tally_df['sales_amount'].sum():,.0f}")
    c3.metric("Avg Monthly Sales (₹)", f"{tally_df['sales_amount'].mean():,.0f}")

    st.markdown("**Monthly Sales Trend (Real Client)**")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 3))
    plt.plot(tally_df["month"], tally_df["sales_amount"])
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    st.dataframe(tally_df, use_container_width=True, height=350)

    st.info("This tab shows real MSME sales (Tally). The other tabs demonstrate the AI decision engine (forecast → reorder).")


    f_next = ets_forecast_next(s)
    lt = lead_time_weeks(df_sku)
    stock_now = current_stock(df_sku)

    recent = s.iloc[-12:] if len(s) >= 12 else s
    sigma = float(recent.std())
    safety_raw = Z * sigma * sqrt(lt)
    safety_cap = SAFETY_CAP_PCT * max(0.0, f_next * lt)
    safety_stock = min(safety_raw, safety_cap)
    on_order = ON_ORDER_BUFFER_PCT * max(0.0, f_next * lt)
    order_up_to = max(0.0, f_next * lt + safety_stock)

    left, right = st.columns(2)

    with left:
        st.markdown("**Forecast vs Actual (recent)**")
        # Plot last 52 weeks
        s_plot = s.iloc[-52:] if len(s) > 52 else s
        plt.figure(figsize=(8,3))
        plt.plot(s_plot.index, s_plot.values, label="Actual")
        plt.axhline(f_next, linestyle="--", label="Next-week forecast")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    with right:
        st.markdown("**Stock Position vs Target**")
        plt.figure(figsize=(8,3))
        x = ["On-hand", "On-order (assumed)", "Target (order-up-to)"]
        y = [0 if np.isnan(stock_now) else stock_now, on_order, order_up_to]
        plt.bar(x, y)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    st.write({
        "forecast_next_week_units": round(f_next, 2),
        "lead_time_weeks": lt,
        "sigma_weekly": round(sigma, 2),
        "safety_stock_used": round(safety_stock, 2),
        "assumed_on_order": round(on_order, 2),
        "current_stock_units": None if np.isnan(stock_now) else round(stock_now, 2),
        "order_up_to_units": round(order_up_to, 2),
    })
