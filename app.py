import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="PLN BI Dashboard", layout="wide")
st.title("⚡ PLN Business Intelligence Dashboard")

# Customizing Plotly layout for 1920x1080 PPT fitting
PPT_WIDTH = 1920
PPT_HEIGHT = 1080
PPT_MARGINS = dict(l=50, r=50, t=80, b=50)
TEMPLATE = "plotly_white"

# ==========================================
# DATA GENERATION (Cached for performance)
# ==========================================
@st.cache_data
def generate_data():
    np.random.seed(42)
    regions = ["Jawa Barat", "Jawa Tengah", "Jawa Timur", "Bali", "Sumatera Utara", "Kalimantan Timur", "Sulawesi Selatan"]
    segments = ["Rumah Tangga", "Bisnis", "Industri", "Publik"]
    cause_categories = ["Cuaca Ekstrem", "Kelebihan Beban", "Gangguan Jaringan", "Pemeliharaan", "Pohon/Vegetasi"]
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="MS")

    region_factor = {"Jawa Barat": 1.25, "Jawa Tengah": 1.00, "Jawa Timur": 1.18, "Bali": 0.75, "Sumatera Utara": 0.85, "Kalimantan Timur": 0.70, "Sulawesi Selatan": 0.68}
    segment_factor = {"Rumah Tangga": 1.00, "Bisnis": 0.78, "Industri": 1.35, "Publik": 0.35}
    tariff = {"Rumah Tangga": 1450, "Bisnis": 1600, "Industri": 1350, "Publik": 1500}

    rows = []
    for date in dates:
        month, year = date.month, date.year
        seasonal_factor = 1 + 0.08 * np.sin((month - 1) / 12 * 2 * np.pi)
        growth_factor = 1 + (year - 2024) * 0.06

        for region in regions:
            for segment in segments:
                base_consumption = 90000 * region_factor[region] * segment_factor[segment]
                consumption_mwh = base_consumption * seasonal_factor * growth_factor * np.random.normal(1, 0.06)
                peak_load_mw = consumption_mwh / 720 * np.random.normal(1.12, 0.04)
                customers = int(120000 * region_factor[region] * segment_factor[segment] * np.random.normal(1, 0.05))
                revenue_billion = consumption_mwh * tariff[segment] / 1_000_000
                distribution_loss_pct = max(np.random.normal(7.5 + (1.2 if region in ["Sumatera Utara", "Sulawesi Selatan"] else 0), 0.8), 3)
                outage_count = np.random.poisson(18 * region_factor[region] * (1.2 if month in [1, 2, 11, 12] else 1) * (1.15 if region in ["Kalimantan Timur", "Sulawesi Selatan"] else 1))
                saidi_hours = outage_count * np.random.normal(0.45, 0.10)
                ev_potential_index = np.random.normal(70 if region in ["Jawa Barat", "Jawa Timur", "Bali"] else 50, 8)
                green_energy_interest = np.random.normal(72 if segment in ["Industri", "Bisnis"] else 55, 7)

                rows.append({"date": date, "year": year, "month": month, "region": region, "segment": segment, "consumption_mwh": consumption_mwh, "peak_load_mw": peak_load_mw, "customers": customers, "revenue_billion_idr": revenue_billion, "distribution_loss_pct": distribution_loss_pct, "outage_count": outage_count, "saidi_hours": saidi_hours, "ev_potential_index": ev_potential_index, "green_energy_interest": green_energy_interest})

    df = pd.DataFrame(rows)
    cause_rows = []
    for _, row in df.iterrows():
        total_outages = int(row["outage_count"])
        if total_outages <= 0: continue
        probs = np.array([0.25, 0.22, 0.28, 0.15, 0.10])
        if row["region"] in ["Kalimantan Timur", "Sulawesi Selatan"]: probs += np.array([0.10, 0.00, 0.05, -0.05, -0.10])
        if row["peak_load_mw"] > df["peak_load_mw"].quantile(0.75): probs += np.array([0.00, 0.10, 0.00, -0.05, -0.05])
        probs = np.clip(probs, 0.01, None)
        probs = probs / probs.sum()
        cause_counts = np.random.multinomial(total_outages, probs)
        for cause, count in zip(cause_categories, cause_counts):
            cause_rows.append({"date": row["date"], "region": row["region"], "segment": row["segment"], "cause": cause, "outage_count": count})

    return df, pd.DataFrame(cause_rows)

df, df_cause = generate_data()

# ==========================================
# UI TABS SETUP
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Deskriptif", "🔍 Diagnostik", "📈 Prediktif", "💡 Preskriptif", "👔 Executive", "🎯 Executive One-Pager"])

# ==========================================
# 1. DESCRIPTIVE DASHBOARD
# ==========================================
with tab1:
    df_monthly = df.groupby("date", as_index=False).agg({"consumption_mwh": "sum", "revenue_billion_idr": "sum"})
    df_region = df.groupby("region", as_index=False).agg({"consumption_mwh": "sum", "outage_count": "sum"})
    
    # Reshaped from 3x2 to 2x3 for a better landscape fit
    fig_desc = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "indicator"}, {"type": "xy"}, {"type": "xy"}],
               [{"type": "indicator"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=["", "Tren Konsumsi Listrik Bulanan", "Konsumsi per Wilayah",
                        "", "Tren Revenue Bulanan", "Jumlah Gangguan per Wilayah"],
        vertical_spacing=0.15
    )

    fig_desc.add_trace(go.Indicator(mode="number", value=df["consumption_mwh"].sum(), number={"suffix": " MWh", "valueformat": ",.0f"}, title={"text": "Total Konsumsi"}), row=1, col=1)
    fig_desc.add_trace(go.Scatter(x=df_monthly["date"], y=df_monthly["consumption_mwh"], mode="lines+markers", line=dict(color='#1f77b4', width=3)), row=1, col=2)
    fig_desc.add_trace(go.Bar(x=df_region["region"], y=df_region["consumption_mwh"], marker_color='#1f77b4'), row=1, col=3)

    fig_desc.add_trace(go.Indicator(mode="number", value=df["revenue_billion_idr"].sum(), number={"suffix": " M IDR", "valueformat": ",.0f"}, title={"text": "Total Revenue"}), row=2, col=1)
    fig_desc.add_trace(go.Scatter(x=df_monthly["date"], y=df_monthly["revenue_billion_idr"], mode="lines+markers", line=dict(color='#2ca02c', width=3)), row=2, col=2)
    fig_desc.add_trace(go.Bar(x=df_region["region"], y=df_region["outage_count"], marker_color='#d62728'), row=2, col=3)

    fig_desc.update_layout(title_text="Dashboard Deskriptif PLN", title_font=dict(size=28, color='#000000'), title_x=0.5, title_xanchor="center", font=dict(size=14, color='#1a1a1a'), width=PPT_WIDTH, height=PPT_HEIGHT, margin=PPT_MARGINS, template=TEMPLATE, showlegend=False)
    fig_desc.for_each_annotation(lambda a: a.update(font=dict(size=18, color='#1a1a1a')))
    st.plotly_chart(fig_desc, use_container_width=True)

# ==========================================
# 2. DIAGNOSTIC DASHBOARD
# ==========================================
with tab2:
    df_diag = df.groupby(["region", "date"], as_index=False).agg({"peak_load_mw": "sum", "outage_count": "sum", "distribution_loss_pct": "mean", "saidi_hours": "mean"})
    df_cause_region = df_cause.groupby(["region", "cause"], as_index=False)["outage_count"].sum()
    df_segment_loss = df.groupby(["segment"], as_index=False)["distribution_loss_pct"].mean()

    fig_diag = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Penyebab Gangguan per Wilayah", "Beban Puncak vs Jumlah Gangguan",
                        "Susut Distribusi per Segmen", "Susut Distribusi vs SAIDI"],
        vertical_spacing=0.15
    )

    for cause in df_cause_region["cause"].unique():
        subset = df_cause_region[df_cause_region["cause"] == cause]
        fig_diag.add_trace(go.Bar(x=subset["region"], y=subset["outage_count"], name=cause), row=1, col=1)

    fig_diag.add_trace(go.Scatter(x=df_diag["peak_load_mw"], y=df_diag["outage_count"], mode="markers", text=df_diag["region"], marker=dict(size=8, opacity=0.7, color='#ff7f0e'), name="Beban vs Gangguan"), row=1, col=2)
    fig_diag.add_trace(go.Bar(x=df_segment_loss["segment"], y=df_segment_loss["distribution_loss_pct"], marker_color='#9467bd', showlegend=False), row=2, col=1)
    fig_diag.add_trace(go.Scatter(x=df_diag["distribution_loss_pct"], y=df_diag["saidi_hours"], mode="markers", text=df_diag["region"], marker=dict(size=8, opacity=0.7, color='#8c564b'), name="Loss vs SAIDI"), row=2, col=2)

    fig_diag.update_layout(title_text="Dashboard Diagnostik PLN", title_font=dict(size=28, color='#000000'), title_x=0.5, title_xanchor="center", font=dict(size=14, color='#1a1a1a'), barmode="stack", width=PPT_WIDTH, height=PPT_HEIGHT, margin=PPT_MARGINS, template=TEMPLATE)
    fig_diag.for_each_annotation(lambda a: a.update(font=dict(size=18, color='#1a1a1a')))
    st.plotly_chart(fig_diag, use_container_width=True)

# ==========================================
# 3. PREDICTIVE DASHBOARD
# ==========================================
with tab3:
    df_forecast = df.groupby("date", as_index=False).agg({"consumption_mwh": "sum"})
    df_forecast["month_index"] = np.arange(len(df_forecast))
    df_forecast["month"] = df_forecast["date"].dt.month
    df_forecast["sin_month"] = np.sin(2 * np.pi * df_forecast["month"] / 12)
    df_forecast["cos_month"] = np.cos(2 * np.pi * df_forecast["month"] / 12)

    model = LinearRegression()
    model.fit(df_forecast[["month_index", "sin_month", "cos_month"]], df_forecast["consumption_mwh"])

    future_dates = pd.date_range(start="2026-01-01", end="2026-06-01", freq="MS")
    future_df = pd.DataFrame({"date": future_dates})
    future_df["month_index"] = np.arange(len(df_forecast), len(df_forecast) + len(future_df))
    future_df["month"] = future_df["date"].dt.month
    future_df["sin_month"] = np.sin(2 * np.pi * future_df["month"] / 12)
    future_df["cos_month"] = np.cos(2 * np.pi * future_df["month"] / 12)
    future_df["forecast"] = model.predict(future_df[["month_index", "sin_month", "cos_month"]])
    
    residual_std = np.std(df_forecast["consumption_mwh"] - model.predict(df_forecast[["month_index", "sin_month", "cos_month"]]))
    future_df["lower"] = future_df["forecast"] - 1.96 * residual_std
    future_df["upper"] = future_df["forecast"] + 1.96 * residual_std

    df_region_latest = df[df["date"] >= "2025-07-01"].groupby("region", as_index=False).agg({"peak_load_mw": "sum", "outage_count": "sum", "distribution_loss_pct": "mean", "ev_potential_index": "mean"})
    df_region_latest["risk_score"] = (df_region_latest["peak_load_mw"]/df_region_latest["peak_load_mw"].max()*35 + df_region_latest["outage_count"]/df_region_latest["outage_count"].max()*25 + df_region_latest["distribution_loss_pct"]/df_region_latest["distribution_loss_pct"].max()*20 + df_region_latest["ev_potential_index"]/df_region_latest["ev_potential_index"].max()*20)

    fig_pred = make_subplots(
        rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=["Forecast Konsumsi Listrik (6 Bulan Kedepan)", "Komponen Risiko per Wilayah", "Prediksi Risiko Demand & Gangguan"],
        vertical_spacing=0.15
    )

    fig_pred.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["consumption_mwh"], name="Historis", line=dict(color="#1f77b4")), row=1, col=1)
    fig_pred.add_trace(go.Scatter(x=future_df["date"], y=future_df["forecast"], name="Forecast", line=dict(color="#ff7f0e", dash="dash")), row=1, col=1)
    fig_pred.add_trace(go.Scatter(x=future_df["date"], y=future_df["upper"], name="Upper", line=dict(width=0), showlegend=False), row=1, col=1)
    fig_pred.add_trace(go.Scatter(x=future_df["date"], y=future_df["lower"], name="Lower", fill="tonexty", fillcolor="rgba(255,127,14,0.2)", line=dict(width=0), showlegend=False), row=1, col=1)

    fig_pred.add_trace(go.Bar(x=df_region_latest["region"], y=df_region_latest["risk_score"], name="Risk Score", marker_color="#e377c2"), row=2, col=1)
    fig_pred.add_trace(go.Scatter(x=df_region_latest["peak_load_mw"], y=df_region_latest["outage_count"], mode="markers+text", text=df_region_latest["region"], textposition="top center", marker=dict(size=df_region_latest["risk_score"], sizemode="area", sizeref=0.1, color="#17becf"), name="Risk Bubble"), row=2, col=2)

    fig_pred.update_layout(title_text="Dashboard Prediktif PLN", title_font=dict(size=28, color='#000000'), title_x=0.5, title_xanchor="center", font=dict(size=14, color='#1a1a1a'), width=PPT_WIDTH, height=PPT_HEIGHT, margin=PPT_MARGINS, template=TEMPLATE)
    fig_pred.for_each_annotation(lambda a: a.update(font=dict(size=18, color='#1a1a1a')))
    st.plotly_chart(fig_pred, use_container_width=True)

# ==========================================
# 4. PRESCRIPTIVE DASHBOARD
# ==========================================
with tab4:
    invest_rows = []
    for region in df_region_latest["region"]:
        latest = df_region_latest[df_region_latest["region"] == region].iloc[0]
        impact = (latest["peak_load_mw"]/df_region_latest["peak_load_mw"].max()*30 + latest["outage_count"]/df_region_latest["outage_count"].max()*25 + latest["distribution_loss_pct"]/df_region_latest["distribution_loss_pct"].max()*20 + latest["ev_potential_index"]*0.25)
        need = np.random.randint(250, 950)
        uplift = impact * np.random.uniform(5.5, 8.5)
        risk_val = (latest["outage_count"]/df_region_latest["outage_count"].max()*100) * np.random.uniform(2.5, 5.5)
        priority = impact*0.45 + (uplift/8)*0.35 + (risk_val/5)*0.20
        
        invest_rows.append({"region": region, "investment": need, "impact": impact, "uplift": uplift, "priority": priority})
    
    df_invest = pd.DataFrame(invest_rows).sort_values("priority", ascending=False)
    df_invest["action"] = np.where(df_invest["priority"] >= df_invest["priority"].quantile(0.75), "Prioritas Tinggi", np.where(df_invest["priority"] >= df_invest["priority"].quantile(0.45), "Bertahap", "Optimasi Operasional"))

    fig_presc = make_subplots(
        rows=2, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "table"}]],
        subplot_titles=["Matriks Prioritas Investasi", "Skor Prioritas Wilayah", "Kebutuhan vs Potensi Revenue", "Keputusan Investasi"],
        vertical_spacing=0.15
    )

    fig_presc.add_trace(go.Scatter(x=df_invest["impact"], y=df_invest["investment"], mode="markers+text", text=df_invest["region"], textposition="top center", marker=dict(size=df_invest["priority"], sizemode="area", sizeref=0.15, color="#bcbd22")), row=1, col=1)
    fig_presc.add_trace(go.Bar(x=df_invest["region"], y=df_invest["priority"], marker_color="#8c564b"), row=1, col=2)
    fig_presc.add_trace(go.Scatter(x=df_invest["investment"], y=df_invest["uplift"], mode="markers+text", text=df_invest["region"], textposition="top center", marker=dict(size=16, color="#e377c2")), row=2, col=1)

    fig_presc.add_trace(go.Table(
        header=dict(values=["Wilayah", "Investasi (M IDR)", "Priority", "Rekomendasi"], align="left", fill_color='royalblue', font=dict(color='white')),
        cells=dict(values=[df_invest["region"], df_invest["investment"].round(0), df_invest["priority"].round(1), df_invest["action"]], align="left", fill_color='aliceblue')
    ), row=2, col=2)

    fig_presc.update_layout(title_text="Dashboard Preskriptif PLN", title_font=dict(size=28, color='#000000'), title_x=0.5, title_xanchor="center", font=dict(size=14, color='#1a1a1a'), width=PPT_WIDTH, height=PPT_HEIGHT, margin=PPT_MARGINS, template=TEMPLATE, showlegend=False)
    fig_presc.for_each_annotation(lambda a: a.update(font=dict(size=18, color='#1a1a1a')))
    st.plotly_chart(fig_presc, use_container_width=True)

# ==========================================
# 5. EXECUTIVE DASHBOARD
# ==========================================
with tab5:
    # Calculate executive metrics
    total_revenue = df["revenue_billion_idr"].sum()
    total_consumption = df["consumption_mwh"].sum()
    avg_reliability = 100 - df["distribution_loss_pct"].mean()
    total_customers = df["customers"].sum()
    
    yoy_growth = ((df[df["year"] == 2025]["consumption_mwh"].sum() - df[df["year"] == 2024]["consumption_mwh"].sum()) / df[df["year"] == 2024]["consumption_mwh"].sum() * 100)
    revenue_efficiency = total_revenue / (total_consumption / 1000)  # Revenue per GWh
    
    # Strategic KPIs
    df_latest = df[df["date"] >= "2025-10-01"]
    critical_regions = df_latest.groupby("region").agg({
        "peak_load_mw": "sum",
        "outage_count": "sum",
        "distribution_loss_pct": "mean",
        "ev_potential_index": "mean",
        "revenue_billion_idr": "sum"
    }).reset_index()
    critical_regions["reliability_score"] = 100 - critical_regions["distribution_loss_pct"]
    critical_regions["operational_risk"] = (critical_regions["outage_count"] / critical_regions["outage_count"].max() * 100)
    
    # ========== SECTION 1: EXECUTIVE KPI CARDS ==========
    st.markdown("### 📈 Strategic Performance Indicators (Q4 2025)")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.metric(
            "Total Revenue",
            f"IDR {total_revenue:.1f}M",
            f"{(total_revenue/1000000):.0f}% YTD",
            delta_color="normal"
        )
    with kpi_col2:
        st.metric(
            "Network Reliability",
            f"{avg_reliability:.1f}%",
            f"+{100-df[df['year']==2024]['distribution_loss_pct'].mean():.1f}% vs 2024",
            delta_color="normal"
        )
    with kpi_col3:
        st.metric(
            "Consumption Growth",
            f"{yoy_growth:.1f}%",
            "YoY Change",
            delta_color="normal"
        )
    with kpi_col4:
        st.metric(
            "Revenue Efficiency",
            f"IDR {revenue_efficiency:.0f}M/GWh",
            f"{revenue_efficiency/1300:.2f}x Industry Avg",
            delta_color="normal"
        )
    
    st.divider()
    
    # ========== SECTION 2: BALANCED SCORECARD ==========
    fig_scorecard = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=["Financial Health", "Operational Excellence", "Customer Satisfaction", "Strategic Growth"],
        vertical_spacing=0.25, horizontal_spacing=0.2
    )
    
    # Financial metrics
    financial_score = min((total_revenue / 800000) * 100, 100)  # Normalized to target
    fig_scorecard.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=financial_score,
        delta={"reference": 85},
        title={"text": "Revenue & Profitability"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#2ca02c"},
               "steps": [{"range": [0, 50], "color": "#ffcccc"},
                        {"range": [50, 85], "color": "#ffffcc"},
                        {"range": [85, 100], "color": "#ccffcc"}],
               "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90}}
    ), row=1, col=1)
    
    # Operational metrics
    operational_score = avg_reliability
    fig_scorecard.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=operational_score,
        delta={"reference": 93},
        title={"text": "Network Reliability & Efficiency"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#1f77b4"},
               "steps": [{"range": [0, 85], "color": "#ffcccc"},
                        {"range": [85, 93], "color": "#ffffcc"},
                        {"range": [93, 100], "color": "#ccffcc"}],
               "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 95}}
    ), row=1, col=2)
    
    # Customer metrics
    avg_saidi = df_latest["saidi_hours"].mean()
    customer_score = max(100 - (avg_saidi / 8 * 100), 0)  # Normalized SAIDI
    fig_scorecard.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=customer_score,
        delta={"reference": 78},
        title={"text": "Service Quality & Availability"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#ff7f0e"},
               "steps": [{"range": [0, 70], "color": "#ffcccc"},
                        {"range": [70, 78], "color": "#ffffcc"},
                        {"range": [78, 100], "color": "#ccffcc"}],
               "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 85}}
    ), row=2, col=1)
    
    # Growth metrics
    growth_score = min((yoy_growth + 10) * 5, 100)  # Normalized growth
    fig_scorecard.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=growth_score,
        delta={"reference": 60},
        title={"text": "Strategic Initiative Progress"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#9467bd"},
               "steps": [{"range": [0, 50], "color": "#ffcccc"},
                        {"range": [50, 60], "color": "#ffffcc"},
                        {"range": [60, 100], "color": "#ccffcc"}],
               "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 70}}
    ), row=2, col=2)
    
    fig_scorecard.update_layout(
        title_text="Balanced Scorecard - Strategic Performance View",
        title_font=dict(size=24, color='#000000'),
        title_x=0.5,
        title_xanchor="center",
        font=dict(size=12, color='#1a1a1a'),
        width=PPT_WIDTH,
        height=600,
        margin=PPT_MARGINS,
        template=TEMPLATE,
        showlegend=False
    )
    
    st.plotly_chart(fig_scorecard, use_container_width=True)
    
    st.divider()
    
    # ========== SECTION 3: STRATEGIC INITIATIVES & ROI ==========
    st.markdown("### 🎯 Strategic Initiatives - Impact & Accountability")
    
    initiatives_data = {
        "Initiative": [
            "Smart Grid Deployment",
            "Renewable Energy Integration",
            "Loss Reduction Program",
            "Customer Digital Transformation",
            "Demand Response Optimization"
        ],
        "Budget (M IDR)": [450, 280, 320, 195, 210],
        "ROI Target (%)": [28, 35, 42, 55, 31],
        "Actual Progress (%)": [85, 72, 78, 65, 81],
        "Health": ["On Track", "At Risk", "On Track", "At Risk", "On Track"]
    }
    
    df_initiatives = pd.DataFrame(initiatives_data)
    df_initiatives["Expected Value (M IDR)"] = (df_initiatives["Budget (M IDR)"] * df_initiatives["ROI Target (%)"] / 100).round(1)
    df_initiatives["Variance (%)"] = ((df_initiatives["Actual Progress (%)"] - df_initiatives["ROI Target (%)"]) / df_initiatives["ROI Target (%)"]) * 100
    
    fig_initiatives = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=["Initiative Performance vs Target", "Budget Utilization & ROI"],
        horizontal_spacing=0.15
    )
    
    colors_health = ["#2ca02c" if h == "On Track" else "#d62728" for h in df_initiatives["Health"]]
    
    fig_initiatives.add_trace(go.Bar(
        x=df_initiatives["Initiative"],
        y=df_initiatives["ROI Target (%)"],
        name="Target ROI %",
        marker_color="#1f77b4",
        opacity=0.6
    ), row=1, col=1)
    
    fig_initiatives.add_trace(go.Bar(
        x=df_initiatives["Initiative"],
        y=df_initiatives["Actual Progress (%)"],
        name="Actual Progress %",
        marker_color=colors_health
    ), row=1, col=1)
    
    fig_initiatives.add_trace(go.Scatter(
        x=df_initiatives["Budget (M IDR)"],
        y=df_initiatives["Expected Value (M IDR)"],
        mode="markers+text",
        text=df_initiatives["Initiative"],
        textposition="top center",
        marker=dict(size=15, color=df_initiatives["ROI Target (%)"], colorscale="Viridis", showscale=True),
        name="Initiative ROI"
    ), row=1, col=2)
    
    fig_initiatives.update_xaxes(title_text="Initiative", row=1, col=1)
    fig_initiatives.update_yaxes(title_text="Performance (%)", row=1, col=1)
    fig_initiatives.update_xaxes(title_text="Budget Investment (M IDR)", row=1, col=2)
    fig_initiatives.update_yaxes(title_text="Expected Value Return (M IDR)", row=1, col=2)
    
    fig_initiatives.update_layout(
        title_text="Strategic Initiatives - Execution & ROI Tracking",
        title_font=dict(size=24, color='#000000'),
        title_x=0.5,
        title_xanchor="center",
        font=dict(size=12, color='#1a1a1a'),
        width=PPT_WIDTH,
        height=500,
        margin=PPT_MARGINS,
        template=TEMPLATE,
        barmode="group",
        hovermode="closest"
    )
    
    st.plotly_chart(fig_initiatives, use_container_width=True)
    
    # Display initiative details table
    st.markdown("#### Initiative Details & Accountability Metrics")
    
    display_df = df_initiatives[["Initiative", "Budget (M IDR)", "ROI Target (%)", "Actual Progress (%)", "Health"]].copy()
    display_df["Budget (M IDR)"] = display_df["Budget (M IDR)"].apply(lambda x: f"IDR {x:.0f}M")
    display_df["ROI Target (%)"] = display_df["ROI Target (%)"].apply(lambda x: f"{x:.1f}%")
    display_df["Actual Progress (%)"] = display_df["Actual Progress (%)"].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # ========== SECTION 4: REGIONAL STRATEGIC HEATMAP ==========
    st.markdown("### 🗺️ Regional Strategic Assessment - Risk & Opportunity Matrix")
    
    fig_heatmap = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=["Risk-Opportunity Matrix", "Regional Performance Scorecard"],
        horizontal_spacing=0.15
    )
    
    # Risk-Opportunity Matrix
    fig_heatmap.add_trace(go.Scatter(
        x=critical_regions["reliability_score"],
        y=critical_regions["ev_potential_index"],
        mode="markers+text",
        text=critical_regions["region"],
        textposition="top center",
        marker=dict(
            size=critical_regions["peak_load_mw"] / 50,
            color=critical_regions["operational_risk"],
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Operational<br>Risk %", x=0.46)
        ),
        name="Regions",
        hovertemplate="<b>%{text}</b><br>Reliability: %{x:.1f}%<br>EV Potential: %{y:.1f}<extra></extra>"
    ), row=1, col=1)
    
    fig_heatmap.update_xaxes(title_text="Network Reliability Score", row=1, col=1)
    fig_heatmap.update_yaxes(title_text="Strategic Growth Opportunity Index", row=1, col=1)
    
    # Regional Performance Heatmap
    critical_regions_sorted = critical_regions.sort_values("revenue_billion_idr", ascending=False)
    heatmap_metrics = critical_regions_sorted[["reliability_score", "operational_risk"]].values.T
    
    fig_heatmap.add_trace(go.Heatmap(
        z=heatmap_metrics,
        x=critical_regions_sorted["region"],
        y=["Reliability Score", "Operational Risk"],
        colorscale="RdYlGn_r",
        text=np.round(heatmap_metrics, 1),
        texttemplate="%{text:.1f}",
        textfont={"size": 12},
        showscale=True,
        colorbar=dict(title="Score", x=1.02),
        name="Performance"
    ), row=1, col=2)
    
    fig_heatmap.update_xaxes(title_text="Region", row=1, col=2)
    fig_heatmap.update_yaxes(title_text="KPI", row=1, col=2)
    
    fig_heatmap.update_layout(
        title_text="Regional Assessment - Strategic Positioning",
        title_font=dict(size=24, color='#000000'),
        title_x=0.5,
        title_xanchor="center",
        font=dict(size=12, color='#1a1a1a'),
        width=PPT_WIDTH,
        height=550,
        margin=PPT_MARGINS,
        template=TEMPLATE,
        showlegend=False
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.divider()
    
    # ========== SECTION 5: EXECUTIVE SUMMARY & ALERTS ==========
    st.markdown("### ⚠️ Executive Alerts & Strategic Recommendations")
    
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    
    with alert_col1:
        st.warning(
            "🔴 **Critical:** 2 regions showing reliability below 90%. Immediate infrastructure assessment required for Kalimantan Timur and Sulawesi Selatan."
        )
    
    with alert_col2:
        st.info(
            "🟡 **Attention:** 2 strategic initiatives at risk of missing targets. Smart Grid Deployment (85% actual vs 28% target) and Customer DX (65% actual vs 55% target) require resource reallocation."
        )
    
    with alert_col3:
        st.success(
            "🟢 **Opportunity:** YoY growth at 6.8% exceeds projections. EV integration potential in 3 high-growth regions ready for expansion investment."
        )
    
    st.divider()
    
    # ========== SECTION 6: KEY BUSINESS DRIVERS ==========
    st.markdown("### 📊 Key Business Drivers Analysis - Correlation to Success")
    
    df_drivers = df_latest.groupby("region").agg({
        "peak_load_mw": "mean",
        "revenue_billion_idr": "sum",
        "outage_count": "sum",
        "distribution_loss_pct": "mean",
        "ev_potential_index": "mean"
    }).reset_index()
    
    fig_drivers = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=["Peak Load vs Revenue", "Network Loss vs Outages",
                        "Customer Growth Potential", "Efficiency Trend Analysis"],
        vertical_spacing=0.15, horizontal_spacing=0.15
    )
    
    # Peak Load vs Revenue
    fig_drivers.add_trace(go.Scatter(
        x=df_drivers["peak_load_mw"],
        y=df_drivers["revenue_billion_idr"],
        mode="markers+text",
        text=df_drivers["region"],
        textposition="top center",
        marker=dict(size=12, color="#1f77b4"),
        name="Load-Revenue"
    ), row=1, col=1)
    
    # Network Loss vs Outages
    fig_drivers.add_trace(go.Scatter(
        x=df_drivers["distribution_loss_pct"],
        y=df_drivers["outage_count"],
        mode="markers+text",
        text=df_drivers["region"],
        textposition="top center",
        marker=dict(size=12, color="#d62728"),
        name="Loss-Outage"
    ), row=1, col=2)
    
    # EV Potential (Customer Growth)
    fig_drivers.add_trace(go.Bar(
        x=df_drivers["region"],
        y=df_drivers["ev_potential_index"],
        marker_color="#2ca02c",
        name="EV Potential"
    ), row=2, col=1)
    
    # Efficiency Score
    efficiency_raw = df_drivers["revenue_billion_idr"] / df_drivers["distribution_loss_pct"]
    df_drivers["efficiency_score"] = ((efficiency_raw - efficiency_raw.min()) / (efficiency_raw.max() - efficiency_raw.min())) * 100
    
    fig_drivers.add_trace(go.Bar(
        x=df_drivers["region"],
        y=df_drivers["efficiency_score"],
        marker_color="#ff7f0e",
        name="Efficiency"
    ), row=2, col=2)
    
    fig_drivers.update_layout(
        title_text="Key Drivers Behind Business Performance",
        title_font=dict(size=24, color='#000000'),
        title_x=0.5,
        title_xanchor="center",
        font=dict(size=12, color='#1a1a1a'),
        width=PPT_WIDTH,
        height=700,
        margin=PPT_MARGINS,
        template=TEMPLATE,
        showlegend=False,
        hovermode="closest"
    )
    
    st.plotly_chart(fig_drivers, use_container_width=True)

# ==========================================
# 6. EXECUTIVE ONE-PAGER (1920x1080 OPTIMIZED)
# ==========================================
with tab6:
    st.markdown("""
    <style>
    .executive-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #000000;
        margin-bottom: 5px;
    }
    .executive-subtitle {
        text-align: center;
        font-size: 16px;
        color: #666666;
        margin-bottom: 25px;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #000000;
    }
    .metric-label {
        font-size: 12px;
        color: #666666;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ========== EXECUTIVE ONE-PAGER HEADER ==========
    col_title1, col_title2, col_title3 = st.columns([1, 2, 1])
    with col_title2:
        st.markdown('<div class="executive-title">PLN EXECUTIVE DASHBOARD</div>', unsafe_allow_html=True)
        st.markdown('<div class="executive-subtitle">Strategic Performance Summary | Q4 2025</div>', unsafe_allow_html=True)
    
    # ========== TOP KPI ROW ==========
    st.markdown("### Key Performance Indicators")
    
    kpi_row = st.columns(6)
    
    metrics_data = [
        ("Total Revenue", f"IDR {total_revenue:.0f}M", "+6.8% YoY", "#2ca02c"),
        ("Network Reliability", f"{avg_reliability:.1f}%", "+1.2% vs 2024", "#1f77b4"),
        ("Total Customers", f"{total_customers/1_000_000:.2f}M", "+3.5% Growth", "#ff7f0e"),
        ("Avg Efficiency", f"IDR {revenue_efficiency:.0f}/GWh", "+4.2% Improvement", "#9467bd"),
        ("Service Availability", f"{100-df_latest['saidi_hours'].mean()/8*100:.1f}%", "-2.1h SAIDI", "#e377c2"),
        ("Growth Trajectory", f"{yoy_growth:.1f}%", "Exceeding Target", "#17becf")
    ]
    
    for idx, (label, value, change, color) in enumerate(metrics_data):
        with kpi_row[idx]:
            st.markdown(f"""
            <div class="metric-box" style="border-left-color: {color};">
                <div class="metric-value">{value}</div>
                <div class="metric-label"><b>{label}</b></div>
                <div class="metric-label">{change}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # ========== MAIN STRATEGIC DASHBOARD (2x2 GRID) ==========
    st.markdown("### Strategic Performance Dashboard")
    
    # Prepare data for main visualizations
    df_monthly_compact = df.groupby("date", as_index=False).agg({"consumption_mwh": "sum", "revenue_billion_idr": "sum", "outage_count": "sum"})
    df_region_compact = df_latest.groupby("region", as_index=False).agg({
        "revenue_billion_idr": "sum",
        "outage_count": "sum",
        "distribution_loss_pct": "mean"
    }).sort_values("revenue_billion_idr", ascending=False)
    
    fig_one_pager = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "table"}]],
        subplot_titles=["Revenue & Consumption Trend", "Network Outages Trend",
                        "Regional Revenue Contribution", "Strategic Status"],
        vertical_spacing=0.20, horizontal_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    # Chart 1: Revenue & Consumption Trend
    fig_one_pager.add_trace(go.Scatter(
        x=df_monthly_compact["date"], 
        y=df_monthly_compact["revenue_billion_idr"],
        name="Revenue (M IDR)",
        line=dict(color="#2ca02c", width=3),
        yaxis="y"
    ), row=1, col=1)
    
    fig_one_pager.add_trace(go.Scatter(
        x=df_monthly_compact["date"],
        y=df_monthly_compact["consumption_mwh"],
        name="Consumption (MWh)",
        line=dict(color="#1f77b4", width=3, dash="dot"),
        yaxis="y2"
    ), row=1, col=1)
    
    # Chart 2: Network Outages Trend
    fig_one_pager.add_trace(go.Bar(
        x=df_monthly_compact["date"],
        y=df_monthly_compact["outage_count"],
        name="Outage Count",
        marker_color="#d62728",
        marker_opacity=0.7
    ), row=1, col=2)
    
    # Add trend line for outages
    z = np.polyfit(np.arange(len(df_monthly_compact)), df_monthly_compact["outage_count"], 2)
    p = np.poly1d(z)
    trend_values = p(np.arange(len(df_monthly_compact)))
    
    fig_one_pager.add_trace(go.Scatter(
        x=df_monthly_compact["date"],
        y=trend_values,
        name="Trend",
        line=dict(color="red", width=2, dash="dash"),
        mode="lines"
    ), row=1, col=2)
    
    # Chart 3: Regional Revenue Contribution
    fig_one_pager.add_trace(go.Bar(
        x=df_region_compact["region"],
        y=df_region_compact["revenue_billion_idr"],
        name="Revenue",
        marker_color="#1f77b4",
        text=df_region_compact["revenue_billion_idr"].round(0),
        textposition="outside",
        textfont=dict(size=10)
    ), row=2, col=1)
    
    # Chart 4: Strategic Status Table
    status_data = []
    regions_list = critical_regions.sort_values("revenue_billion_idr", ascending=False).head(5)
    
    for _, region_row in regions_list.iterrows():
        region = region_row["region"]
        reliability = region_row["reliability_score"]
        risk = region_row["operational_risk"]
        
        if reliability >= 95:
            status = "🟢 Optimal"
        elif reliability >= 90:
            status = "🟡 Good"
        else:
            status = "🔴 Alert"
        
        status_data.append([region, f"{reliability:.1f}%", f"{risk:.0f}%", status])
    
    fig_one_pager.add_trace(go.Table(
        header=dict(
            values=["<b>Region</b>", "<b>Reliability</b>", "<b>Risk</b>", "<b>Status</b>"],
            align="center",
            fill_color="#1f77b4",
            font=dict(color="white", size=12)
        ),
        cells=dict(
            values=list(zip(*status_data)),
            align="center",
            fill_color=[["#f0f0f0", "#ffffff"] * 3 for _ in range(4)],
            font=dict(size=11)
        )
    ), row=2, col=2)
    
    fig_one_pager.update_xaxes(title_text="Month", row=1, col=1, tickangle=-45)
    fig_one_pager.update_yaxes(title_text="Revenue (M IDR)", row=1, col=1, title_font=dict(color="#2ca02c"))
    fig_one_pager.update_yaxes(title_text="Consumption (MWh)", row=1, col=1, secondary_y=True, title_font=dict(color="#1f77b4"))
    
    fig_one_pager.update_xaxes(title_text="Month", row=1, col=2, tickangle=-45)
    fig_one_pager.update_yaxes(title_text="Outage Count", row=1, col=2)
    
    fig_one_pager.update_xaxes(title_text="Region", row=2, col=1, tickangle=-45)
    fig_one_pager.update_yaxes(title_text="Revenue (M IDR)", row=2, col=1)
    
    fig_one_pager.update_layout(
        title_text="",
        font=dict(size=11, color='#1a1a1a'),
        width=1920,
        height=1080,
        margin=dict(l=60, r=60, t=100, b=60),
        template=TEMPLATE,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            font=dict(size=9)
        ),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_one_pager, use_container_width=True)
    
    st.divider()
    
    # ========== BOTTOM SECTION: EXECUTIVE SUMMARY & ALERTS ==========
    summary_col1, summary_col2, summary_col3 = st.columns([1, 1, 1])
    
    with summary_col1:
        st.markdown("#### 🎯 Strategic Focus Areas")
        st.markdown("""
        - **Smart Grid Deployment**: 85% complete
        - **Loss Reduction**: On track for Q1 2026
        - **Renewable Integration**: 72% progress
        - **Digital Transformation**: Scaling up
        """)
    
    with summary_col2:
        st.markdown("#### ⚠️ Critical Alerts")
        st.markdown("""
        - **High Risk Regions**: 2 areas below 90% reliability
        - **Outage Trend**: Slight increase in Q4 2025
        - **Resource Constraints**: 3 initiatives need reallocation
        - **Action Required**: Infrastructure assessment needed
        """)
    
    with summary_col3:
        st.markdown("#### ✅ Key Achievements")
        st.markdown(f"""
        - **Revenue Growth**: +6.8% YoY
        - **Reliability Improvement**: +1.2%
        - **Customer Growth**: +3.5%
        - **Operational Efficiency**: +4.2%
        """)
    
    st.divider()
    
    # ========== FOOTER: STRATEGIC INITIATIVES SUMMARY ==========
    st.markdown("#### 📊 Strategic Initiatives Overview")
    
    footer_col1, footer_col2 = st.columns([0.6, 0.4])
    
    with footer_col1:
        initiatives_summary = df_initiatives[["Initiative", "Budget (M IDR)", "Actual Progress (%)"]].head(5).copy()
        initiatives_summary.columns = ["Strategic Initiative", "Investment (M IDR)", "Progress (%)"]
        
        st.dataframe(
            initiatives_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Investment (M IDR)": st.column_config.NumberColumn(format="IDR %.0f M"),
                "Progress (%)": st.column_config.ProgressColumn(min_value=0, max_value=100)
            }
        )
    
    with footer_col2:
        # Quick metrics
        total_investment = df_initiatives["Budget (M IDR)"].sum()
        avg_progress = df_initiatives["Actual Progress (%)"].mean()
        on_track = len(df_initiatives[df_initiatives["Health"] == "On Track"])
        
        st.metric("Total Investment", f"IDR {total_investment:.0f}M")
        st.metric("Average Progress", f"{avg_progress:.1f}%")
        st.metric("On Track Initiatives", f"{on_track} of 5")
    
    st.markdown("""
    ---
    <div style='text-align: center; font-size: 11px; color: #999999;'>
    <p>PLN Business Intelligence Dashboard | Data as of Q4 2025 | Prepared for Executive Review</p>
    <p>For detailed analysis, refer to Descriptive, Diagnostic, Predictive, and Prescriptive tabs</p>
    </div>
    """, unsafe_allow_html=True)