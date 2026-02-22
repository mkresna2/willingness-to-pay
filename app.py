import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WTP Segmentation | Hotel RM",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────────
BLUES = ["#1A3A5C", "#2471A3", "#5DADE2", "#AED6F1", "#D6EAF8"]
QUAL  = px.colors.qualitative.Set2

# ── Load & Engineer Features ─────────────────────────────────────────────────────
def _engineer_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Booking_Date", "Check_in_Date", "Check_out_Date"]:
        df[col] = pd.to_datetime(df[col])

    # ── Derived fields
    df["Booking_Window"]    = (df["Check_in_Date"] - df["Booking_Date"]).dt.days
    df["Is_Cancelled"]      = (df["Cancellation_Status"] == "Cancelled").astype(int)
    df["Is_Member"]         = df["Rate_Plan"].str.contains("Member", na=False).astype(int)
    df["Check_in_Month"]    = df["Check_in_Date"].dt.month
    df["Check_in_DOW"]      = df["Check_in_Date"].dt.day_name()
    df["Booking_Month"]     = df["Booking_Date"].dt.month

    # Normalise rate-plan name (strip " + Member" suffix)
    df["Rate_Plan_Base"] = (
        df["Rate_Plan"]
        .str.replace(r"\s*\+\s*Member", "", regex=True)
        .str.strip()
    )

    # ── Booking-window buckets
    df["Window_Segment"] = pd.cut(
        df["Booking_Window"],
        bins=[-1, 0, 7, 21, 60, 9999],
        labels=[
            "Same Day",
            "Last Minute (1-7d)",
            "Short Advance (8-21d)",
            "Medium Advance (22-60d)",
            "Long Advance (61d+)",
        ],
    )

    # ── Proxy scores for WTP composite
    rate_plan_score_map = {
        "Non-Refundable": 1.00,
        "BAR":            0.75,
        "Corporate":      0.60,
        "Early Bird (>30)": 0.35,
    }
    channel_score_map = {
        "Walk-in": 1.00,
        "Direct":  0.80,
        "Website": 0.60,
        "OTA":     0.40,
    }

    df["Rate_Plan_Score"] = df["Rate_Plan_Base"].map(rate_plan_score_map).fillna(0.5)
    df["Channel_Score"]   = df["Booking_Channel"].map(channel_score_map).fillna(0.5)
    df["Window_Score"]    = 1 - (df["Booking_Window"].clip(0, 90) / 90)

    # ── WTP Score  (0–100, higher = greater willingness to pay)
    #    50% weight: accepted rate; 25%: rate-plan commitment; 15%: channel; 10%: urgency
    rate_min, rate_max = df["Booked_Rate"].min(), df["Booked_Rate"].max()
    rate_norm = (df["Booked_Rate"] - rate_min) / (rate_max - rate_min)

    df["WTP_Score"] = (
        0.50 * rate_norm
        + 0.25 * df["Rate_Plan_Score"]
        + 0.15 * df["Channel_Score"]
        + 0.10 * df["Window_Score"]
    ) * 100

    # ── Quartile tiers
    df["WTP_Tier"] = pd.qcut(
        df["WTP_Score"],
        q=4,
        labels=["Price-Sensitive", "Value-Seeker", "Comfort-Buyer", "Premium"],
    )

    return df


@st.cache_data
def load_data(path: str = "data/Bookings.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return _engineer_data(df)


def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return _engineer_data(df)


# ── Sidebar (upload section) ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏨 Hotel RM")
    st.subheader("WTP Segmentation")
    st.markdown("---")

    with st.expander("📤 Upload Data"):
        uploaded_file = st.file_uploader(
            "Upload booking data (CSV)",
            type=["csv"],
            help="Upload your own booking CSV. If no file is uploaded, sample data is used.",
        )
        with st.expander("Expected CSV format"):
            st.markdown("""
**Required columns:**
- `Booking_ID`, `Booking_Date`, `Check_in_Date`, `Check_out_Date`
- `Room_Type`, `Rate_Plan`, `Booked_Rate`, `Number_of_Nights`, `Number_of_Guests`
- `Booking_Channel`, `Cancellation_Status`, `Revenue_Generated`

**Date format:** YYYY-MM-DD  
**Cancellation_Status:** e.g. `Confirmed`, `Cancelled`  
**Rate_Plan:** May include ` + Member` suffix (e.g. `BAR + Member`).
            """.strip())

    st.markdown("---")

# ── Resolve data source ──────────────────────────────────────────────────────────
df = load_data_from_upload(uploaded_file) if uploaded_file else load_data()

# ── Sidebar (filters) ────────────────────────────────────────────────────────────
with st.sidebar:
    min_ci = df["Check_in_Date"].min().date()
    max_ci = df["Check_in_Date"].max().date()
    date_range = st.date_input(
        "Check-in Date Range",
        value=(min_ci, max_ci),
        min_value=min_ci,
        max_value=max_ci,
    )

    room_types = st.multiselect(
        "Room Type",
        options=sorted(df["Room_Type"].unique()),
        default=sorted(df["Room_Type"].unique()),
    )
    channels = st.multiselect(
        "Booking Channel",
        options=sorted(df["Booking_Channel"].unique()),
        default=sorted(df["Booking_Channel"].unique()),
    )
    rate_plans = st.multiselect(
        "Rate Plan (Base)",
        options=sorted(df["Rate_Plan_Base"].unique()),
        default=sorted(df["Rate_Plan_Base"].unique()),
    )
    cancel_opts = st.multiselect(
        "Cancellation Status",
        options=df["Cancellation_Status"].unique().tolist(),
        default=df["Cancellation_Status"].unique().tolist(),
    )
    member_opt = st.radio("Membership", ["All", "Member Only", "Non-Member Only"])

    st.markdown("---")

# ── Apply filters
if len(date_range) == 2:
    d0, d1 = date_range
else:
    d0, d1 = min_ci, max_ci

mask = (
    (df["Check_in_Date"].dt.date >= d0)
    & (df["Check_in_Date"].dt.date <= d1)
    & (df["Room_Type"].isin(room_types))
    & (df["Booking_Channel"].isin(channels))
    & (df["Rate_Plan_Base"].isin(rate_plans))
    & (df["Cancellation_Status"].isin(cancel_opts))
)
if member_opt == "Member Only":
    mask &= df["Is_Member"] == 1
elif member_opt == "Non-Member Only":
    mask &= df["Is_Member"] == 0

fdf = df[mask].copy()
confirmed = fdf[fdf["Cancellation_Status"] == "Confirmed"]

st.sidebar.metric("Filtered Bookings", f"{len(fdf):,}")
st.sidebar.metric("Confirmed Bookings", f"{len(confirmed):,}")

# ── Main Title ────────────────────────────────────────────────────────────────────
st.title("Willingness-to-Pay Dynamic Segmentation")
st.caption("Hotel Revenue Management · Ubud Property · IDR pricing")

tabs = st.tabs([
    "📊 Overview",
    "💰 WTP Distribution",
    "🎯 Segment Profiles",
    "🤖 Dynamic Clustering",
    "📈 Price Sensitivity",
])

# ═══════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Bookings",       f"{len(fdf):,}")
    c2.metric("Confirmed Revenue",    f"IDR {confirmed['Revenue_Generated'].sum()/1e9:.2f}B")
    c3.metric("Avg Booked Rate",      f"IDR {fdf['Booked_Rate'].mean():,.0f}")
    c4.metric("Cancellation Rate",    f"{fdf['Is_Cancelled'].mean()*100:.1f}%")
    c5.metric("Avg WTP Score",        f"{fdf['WTP_Score'].mean():.1f}/100")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        monthly = (
            fdf.groupby(fdf["Check_in_Date"].dt.to_period("M"))
            .agg(Bookings=("Booking_ID", "count"), Avg_Rate=("Booked_Rate", "mean"))
            .reset_index()
        )
        monthly["Check_in_Date"] = monthly["Check_in_Date"].astype(str)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=monthly["Check_in_Date"], y=monthly["Bookings"],
                   name="Bookings", marker_color=BLUES[1]),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=monthly["Check_in_Date"], y=monthly["Avg_Rate"],
                       name="Avg Rate", line=dict(color="#E67E22", width=2)),
            secondary_y=True,
        )
        fig.update_layout(title="Monthly Bookings & Avg Booked Rate", height=360,
                          legend=dict(orientation="h", y=1.1))
        fig.update_yaxes(title_text="Bookings", secondary_y=False)
        fig.update_yaxes(title_text="Avg Rate (IDR)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        tier_counts = fdf["WTP_Tier"].value_counts().reset_index()
        tier_counts.columns = ["WTP_Tier", "Count"]
        fig = px.pie(
            tier_counts, values="Count", names="WTP_Tier",
            title="WTP Tier Distribution",
            color_discrete_sequence=BLUES,
            hole=0.42,
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        ch_mix = (
            fdf.groupby(["Booking_Channel", "WTP_Tier"])
            .size().reset_index(name="Count")
        )
        fig = px.bar(
            ch_mix, x="Booking_Channel", y="Count", color="WTP_Tier",
            title="Bookings by Channel & WTP Tier",
            color_discrete_sequence=BLUES,
            barmode="stack",
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        with st.expander("ℹ️ Help — Revenue by Room Type"):
            st.markdown("""
- **Revenue**: Total revenue from confirmed bookings only, grouped by room type.
- **Bar color (Avg WTP)**: Average willingness-to-pay score for that room type; darker blue = higher WTP.
- **Labels**: Number of confirmed bookings (e.g. \"1,234 bkgs\") shown on each bar.
            """.strip())
        rt_rev = (
            confirmed.groupby("Room_Type")
            .agg(Revenue=("Revenue_Generated", "sum"),
                 Bookings=("Booking_ID", "count"),
                 Avg_WTP=("WTP_Score", "mean"))
            .reset_index()
        )
        fig = px.bar(
            rt_rev, x="Room_Type", y="Revenue", color="Avg_WTP",
            title="Revenue by Room Type (Confirmed Only)",
            color_continuous_scale="Blues",
            text=rt_rev["Bookings"].apply(lambda x: f"{x:,} bkgs"),
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════════
# TAB 2 — WTP Distribution
# ═══════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("WTP Score & Rate Distributions")

    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.histogram(
            fdf, x="WTP_Score", color="Room_Type",
            nbins=40, barmode="overlay", opacity=0.72,
            title="WTP Score Distribution by Room Type",
            color_discrete_sequence=BLUES,
        )
        mean_wtp = fdf["WTP_Score"].mean()
        fig.add_vline(x=mean_wtp, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_wtp:.1f}")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = px.histogram(
            fdf, x="Booked_Rate", color="Rate_Plan_Base",
            nbins=50, barmode="overlay", opacity=0.72,
            title="Booked Rate Distribution by Rate Plan",
            color_discrete_sequence=QUAL,
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("WTP Comparison Across Segments")

    c1, c2 = st.columns(2)
    with c1:
        seg_dim = st.selectbox(
            "Segment dimension",
            ["Booking_Channel", "Room_Type", "Rate_Plan_Base",
             "Window_Segment", "Is_Member"],
            key="dist_seg",
        )
    with c2:
        metric_sel = st.selectbox(
            "Metric", ["WTP_Score", "Booked_Rate"], key="dist_metric"
        )

    fig = px.box(
        fdf, x=seg_dim, y=metric_sel, color=seg_dim,
        points="outliers",
        title=f"{metric_sel} by {seg_dim}",
        color_discrete_sequence=QUAL,
    )
    fig.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.subheader("WTP Heatmap")
    c1, c2 = st.columns(2)
    with c1:
        heat_x = st.selectbox("X axis", ["Booking_Channel", "Room_Type", "Window_Segment"], key="hx")
    with c2:
        heat_y = st.selectbox("Y axis", ["Room_Type", "Rate_Plan_Base", "Booking_Channel"], key="hy")

    heat_pivot = (
        fdf.groupby([heat_y, heat_x])["WTP_Score"]
        .mean()
        .reset_index()
        .pivot(index=heat_y, columns=heat_x, values="WTP_Score")
    )
    fig = px.imshow(
        heat_pivot, text_auto=".1f",
        color_continuous_scale="Blues",
        title=f"Avg WTP Score: {heat_y} × {heat_x}",
        aspect="auto",
    )
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Segment Profiles
# ═══════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("WTP Tier Profiles")

    tier_profile = (
        fdf.groupby("WTP_Tier", observed=True)
        .agg(
            Bookings=("Booking_ID", "count"),
            Avg_Rate=("Booked_Rate", "mean"),
            Avg_WTP_Score=("WTP_Score", "mean"),
            Avg_LOS=("Number_of_Nights", "mean"),
            Avg_Guests=("Number_of_Guests", "mean"),
            Cancel_Rate=("Is_Cancelled", "mean"),
            Member_Rate=("Is_Member", "mean"),
            Avg_Window=("Booking_Window", "mean"),
            Total_Revenue=("Revenue_Generated", "sum"),
        )
        .round(2)
        .reset_index()
    )
    tier_profile["Cancel_Rate"] = (tier_profile["Cancel_Rate"] * 100).round(1)
    tier_profile["Member_Rate"] = (tier_profile["Member_Rate"] * 100).round(1)
    tier_profile["Total_Revenue"] = (tier_profile["Total_Revenue"] / 1e6).round(1)
    tier_profile.columns = [
        "WTP Tier", "Bookings", "Avg Rate (IDR)", "Avg WTP Score",
        "Avg LOS", "Avg Guests", "Cancel Rate (%)", "Member Rate (%)",
        "Avg Booking Window (d)", "Revenue (IDR M)",
    ]
    st.dataframe(tier_profile, use_container_width=True, hide_index=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        # Radar / spider chart per tier
        radar_metrics = [
            "Avg Rate (IDR)", "Avg WTP Score", "Avg LOS",
            "Avg Guests", "Member Rate (%)",
        ]
        radar_data = tier_profile[["WTP Tier"] + radar_metrics].copy()
        for m in radar_metrics:
            rng = radar_data[m].max() - radar_data[m].min()
            radar_data[m] = ((radar_data[m] - radar_data[m].min()) / rng * 100) if rng > 0 else 50.0

        fig = go.Figure()
        for i, row in radar_data.iterrows():
            vals = row[radar_metrics].tolist()
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_metrics + [radar_metrics[0]],
                fill="toself",
                name=row["WTP Tier"],
                line_color=BLUES[i % len(BLUES)],
                opacity=0.75,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Tier Profiles (Normalised)", height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        ch_tier = (
            fdf.groupby(["WTP_Tier", "Booking_Channel"], observed=True)
            .size().reset_index(name="Count")
        )
        totals = ch_tier.groupby("WTP_Tier", observed=True)["Count"].transform("sum")
        ch_tier["Pct"] = (ch_tier["Count"] / totals * 100).round(1)
        fig = px.bar(
            ch_tier, x="WTP_Tier", y="Pct", color="Booking_Channel",
            title="Channel Mix by WTP Tier (%)",
            color_discrete_sequence=QUAL,
            barmode="stack",
        )
        fig.update_layout(height=420, yaxis_title="Share (%)")
        st.plotly_chart(fig, use_container_width=True)

    # Deep dive
    st.subheader("Tier Deep Dive")
    sel_tier = st.selectbox(
        "Select WTP Tier",
        fdf["WTP_Tier"].dropna().cat.categories.tolist(),
    )
    tier_df = fdf[fdf["WTP_Tier"] == sel_tier]

    c1, c2, c3 = st.columns(3)

    with c1:
        fig = px.histogram(
            tier_df, x="Booked_Rate", nbins=30,
            title=f"Rate Distribution — {sel_tier}",
            color_discrete_sequence=[BLUES[1]],
        )
        fig.update_layout(height=310)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        win_counts = (
            tier_df["Window_Segment"]
            .value_counts()
            .reset_index()
        )
        win_counts.columns = ["Window_Segment", "Count"]
        fig = px.bar(
            win_counts, x="Window_Segment", y="Count",
            title=f"Booking Window Segments — {sel_tier}",
            color_discrete_sequence=[BLUES[0]],
        )
        fig.update_layout(height=310, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        rp_mix = tier_df["Rate_Plan_Base"].value_counts().reset_index()
        rp_mix.columns = ["Rate Plan", "Count"]
        fig = px.pie(
            rp_mix, values="Count", names="Rate Plan",
            title=f"Rate Plan Mix — {sel_tier}",
            color_discrete_sequence=QUAL, hole=0.42,
        )
        fig.update_layout(height=310)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Dynamic Clustering
# ═══════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("K-Means WTP Clustering")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_clusters = st.slider("Number of Clusters (k)", 2, 8, 4)
    with c2:
        incl_cancelled = st.checkbox("Include Cancelled Bookings", value=False)
    with c3:
        feature_set = st.multiselect(
            "Clustering Features",
            ["Booked_Rate", "Booking_Window", "Number_of_Nights",
             "Number_of_Guests", "Is_Member", "Channel_Score",
             "Rate_Plan_Score", "WTP_Score"],
            default=["Booked_Rate", "Booking_Window",
                     "Number_of_Nights", "Is_Member", "Channel_Score"],
        )

    cluster_src = fdf if incl_cancelled else confirmed

    if len(feature_set) < 2:
        st.warning("Select at least 2 features for clustering.")
    elif len(cluster_src) < n_clusters:
        st.warning("Not enough data rows for the chosen number of clusters.")
    else:
        feat_data = cluster_src[feature_set].dropna()
        idx = feat_data.index

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feat_data)

        # Elbow
        max_k = min(10, len(feat_data) - 1)
        k_range = range(2, max_k + 1)
        inertias = [
            KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_
            for k in k_range
        ]

        col_el, col_pca = st.columns(2)

        with col_el:
            fig = px.line(
                x=list(k_range), y=inertias, markers=True,
                title="Elbow Method — Inertia vs k",
                labels={"x": "k (clusters)", "y": "Inertia"},
            )
            fig.add_vline(x=n_clusters, line_dash="dash", line_color="red",
                          annotation_text=f"k={n_clusters}")
            fig.update_layout(height=330)
            st.plotly_chart(fig, use_container_width=True)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        cdf = cluster_src.loc[idx].copy()
        cdf["Cluster"] = [f"Cluster {l + 1}" for l in labels]

        with col_pca:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame({
                "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
                "Cluster": cdf["Cluster"],
                "Booked_Rate": cdf["Booked_Rate"],
                "WTP_Score": cdf["WTP_Score"],
            })
            fig = px.scatter(
                pca_df, x="PC1", y="PC2", color="Cluster",
                title=f"PCA Projection — {n_clusters} Clusters",
                color_discrete_sequence=px.colors.qualitative.Set1,
                hover_data=["Booked_Rate", "WTP_Score"],
                opacity=0.55,
            )
            fig.update_traces(marker=dict(size=4))
            fig.update_layout(height=330)
            st.plotly_chart(fig, use_container_width=True)

        # Cluster summary table
        st.subheader("Cluster Profiles")
        cp = (
            cdf.groupby("Cluster")
            .agg(
                Bookings=("Booking_ID", "count"),
                Avg_Rate=("Booked_Rate", "mean"),
                Avg_WTP=("WTP_Score", "mean"),
                Avg_Window=("Booking_Window", "mean"),
                Avg_LOS=("Number_of_Nights", "mean"),
                Member_Rate=("Is_Member", "mean"),
            )
            .round(1)
            .reset_index()
        )
        cp["Member_Rate"] = (cp["Member_Rate"] * 100).round(1)
        cp.columns = ["Cluster", "Bookings", "Avg Rate (IDR)", "Avg WTP Score",
                      "Avg Booking Window (d)", "Avg LOS", "Member Rate (%)"]
        st.dataframe(cp, use_container_width=True, hide_index=True)

        col_l, col_r = st.columns(2)

        with col_l:
            fig = px.box(
                cdf, x="Cluster", y="Booked_Rate", color="Cluster",
                title="Booked Rate by Cluster",
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            fig = px.box(
                cdf, x="Cluster", y="Booking_Window", color="Cluster",
                title="Booking Window by Cluster",
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        ch_cl = (
            cdf.groupby(["Cluster", "Booking_Channel"])
            .size().reset_index(name="Count")
        )
        fig = px.bar(
            ch_cl, x="Cluster", y="Count", color="Booking_Channel",
            title="Channel Mix by Cluster",
            color_discrete_sequence=QUAL,
            barmode="group",
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Price Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Price Sensitivity & WTP Curves")

    c1, c2 = st.columns(2)
    with c1:
        curve_seg = st.selectbox(
            "Segment dimension",
            ["Room_Type", "Booking_Channel", "Rate_Plan_Base", "Window_Segment"],
            key="curve_seg",
        )
    with c2:
        n_bins = st.slider("Price bins", 10, 50, 20, key="price_bins")

    # Demand curve (confirmed only)
    conf2 = fdf[fdf["Cancellation_Status"] == "Confirmed"].copy()
    conf2["Rate_Bin"] = pd.cut(conf2["Booked_Rate"], bins=n_bins)
    conf2["Rate_Mid"] = conf2["Rate_Bin"].apply(
        lambda x: x.mid if pd.notna(x) else np.nan
    )

    demand = (
        conf2.groupby([curve_seg, "Rate_Mid"], observed=True)
        .agg(Bookings=("Booking_ID", "count"))
        .reset_index()
    )

    fig = px.line(
        demand, x="Rate_Mid", y="Bookings", color=curve_seg,
        title=f"Demand Curve by {curve_seg} (Confirmed Bookings)",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set1,
        labels={"Rate_Mid": "Booked Rate (IDR)"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        # Cancellation rate vs price
        fdf_c = fdf.copy()
        fdf_c["Rate_Bin"] = pd.cut(fdf_c["Booked_Rate"], bins=20)
        fdf_c["Rate_Mid"] = fdf_c["Rate_Bin"].apply(
            lambda x: x.mid if pd.notna(x) else np.nan
        )
        cr = (
            fdf_c.groupby("Rate_Mid", observed=True)
            .agg(Cancel_Rate=("Is_Cancelled", "mean"),
                 Bookings=("Booking_ID", "count"))
            .reset_index()
        )
        cr["Cancel_Rate"] *= 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=cr["Rate_Mid"], y=cr["Cancel_Rate"],
                       name="Cancel Rate (%)",
                       line=dict(color="#E74C3C", width=2)),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(x=cr["Rate_Mid"], y=cr["Bookings"],
                   name="Bookings", marker_color="#85C1E9", opacity=0.6),
            secondary_y=True,
        )
        fig.update_layout(title="Cancellation Rate vs Booked Price", height=400)
        fig.update_yaxes(title_text="Cancel Rate (%)", secondary_y=False)
        fig.update_yaxes(title_text="Bookings", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Booking window vs rate scatter
        sample = fdf.sample(min(2000, len(fdf)), random_state=42)
        fig = px.scatter(
            sample, x="Booking_Window", y="Booked_Rate",
            color="Room_Type", opacity=0.5,
            trendline="ols",
            title="Booking Window vs Booked Rate",
            color_discrete_sequence=BLUES,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # WTP by booking window
    st.subheader("WTP by Booking Window Segment")

    win_wtp = (
        fdf.groupby("Window_Segment", observed=True)
        .agg(
            Bookings=("Booking_ID", "count"),
            Avg_Rate=("Booked_Rate", "mean"),
            Avg_WTP=("WTP_Score", "mean"),
            Cancel_Rate=("Is_Cancelled", "mean"),
        )
        .reset_index()
    )
    win_wtp["Cancel_Rate"] = (win_wtp["Cancel_Rate"] * 100).round(1)

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            win_wtp, x="Window_Segment", y="Avg_Rate",
            color="Avg_WTP", color_continuous_scale="Blues",
            title="Avg Booked Rate by Booking Window",
            text=win_wtp["Avg_Rate"].apply(lambda x: f"{x:,.0f}"),
        )
        fig.update_layout(height=360, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=win_wtp["Window_Segment"], y=win_wtp["Bookings"],
                   name="Bookings", marker_color=BLUES[1]),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=win_wtp["Window_Segment"], y=win_wtp["Cancel_Rate"],
                       name="Cancel Rate (%)",
                       line=dict(color="#E74C3C", width=2),
                       mode="lines+markers"),
            secondary_y=True,
        )
        fig.update_layout(title="Bookings & Cancel Rate by Window", height=360)
        fig.update_yaxes(title_text="Bookings", secondary_y=False)
        fig.update_yaxes(title_text="Cancel Rate (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # Member vs Non-Member WTP premium
    st.subheader("Member vs Non-Member WTP Premium")
    mem_comp = (
        fdf.groupby(["Room_Type", "Is_Member"])
        .agg(Avg_Rate=("Booked_Rate", "mean"), Avg_WTP=("WTP_Score", "mean"))
        .reset_index()
    )
    mem_comp["Membership"] = mem_comp["Is_Member"].map({1: "Member", 0: "Non-Member"})

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            mem_comp, x="Room_Type", y="Avg_Rate", color="Membership",
            barmode="group",
            title="Avg Rate: Member vs Non-Member by Room Type",
            color_discrete_sequence=[BLUES[0], BLUES[2]],
        )
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            mem_comp, x="Room_Type", y="Avg_WTP", color="Membership",
            barmode="group",
            title="Avg WTP Score: Member vs Non-Member",
            color_discrete_sequence=[BLUES[0], BLUES[2]],
        )
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
