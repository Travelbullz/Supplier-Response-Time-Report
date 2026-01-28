import io
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

st.set_page_config(page_title="Supplier Response Time Analysis", layout="wide")

# ----------------------------
# Expected buckets (canonical)
# ----------------------------
CANON_BUCKETS = [
    "<0.5 sec",
    ">0.5 & <1.0",
    ">1 & <1.5",
    ">1.5 & <2.0",
    ">2.0 & 2.5",
    ">2.5 & <3.0",
    ">3 & <=4",
    ">4",
]

REQUIRED_COLS = ["Suppliers", "Total Requests"]

# Midpoints for estimated mean (open bucket >4 is an assumption)
MIDPOINTS = {
    "<0.5 sec": 0.25,
    ">0.5 & <1.0": 0.75,
    ">1 & <1.5": 1.25,
    ">1.5 & <2.0": 1.75,
    ">2.0 & 2.5": 2.25,
    ">2.5 & <3.0": 2.75,
    ">3 & <=4": 3.5,
    ">4": 5.0,
}


# ----------------------------
# Helpers
# ----------------------------
def _clean_header(h: str) -> str:
    h = str(h).strip()
    h = re.sub(r"\s+", " ", h)
    return h


def _canonicalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns:
        (df_normalized, ignored_original_columns)
    """
    df = df.copy()
    original_cols = [str(c) for c in df.columns]
    df.columns = [_clean_header(c) for c in df.columns]

    # Alias map for known messy headers (based on your sample)
    aliases = {
        "Suppliers": "Suppliers",
        "Supplier": "Suppliers",
        "Total Requests": "Total Requests",
        "TotalRequests": "Total Requests",
        "<0.5  sec": "<0.5 sec",
        "<0.5 sec": "<0.5 sec",
        ">0.5 & <1.0": ">0.5 & <1.0",
        ">1 & <1.5": ">1 & <1.5",
        ">1.5&<2.0": ">1.5 & <2.0",
        ">1.5 & <2.0": ">1.5 & <2.0",
        ">2.0&2.5": ">2.0 & 2.5",
        ">2.0 &2.5": ">2.0 & 2.5",
        ">2.0 & 2.5": ">2.0 & 2.5",
        ">2.5 & <3.0": ">2.5 & <3.0",
        ">3 & <=4": ">3 & <=4",
        ">4": ">4",
        # This exists in your sample but is redundant/inconsistent vs totals.
        "3.5 &4.0": "__IGNORE__",
        "3.5 & 4.0": "__IGNORE__",
    }

    new_cols = []
    for c in df.columns:
        new_cols.append(aliases.get(c, c))
    df.columns = new_cols

    ignored_cols = []
    if "__IGNORE__" in df.columns:
        # Track original columns that likely mapped to ignore keys
        for oc in original_cols:
            if str(oc).strip() in ("3.5 &4.0", "3.5 & 4.0"):
                ignored_cols.append(str(oc))
        df = df.drop(columns=["__IGNORE__"])

    return df, ignored_cols


def _parse_int_series(s: pd.Series) -> pd.Series:
    # Handles numbers like "68,02,009" and blanks
    return (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": "0", "nan": "0", "None": "0"})
        .astype(float)
        .fillna(0)
        .astype(int)
    )


def threshold_to_buckets(threshold: float) -> List[str]:
    """
    Defines "slow" as >= threshold using the available buckets.
    """
    if threshold == 1.0:
        return [">1 & <1.5", ">1.5 & <2.0", ">2.0 & 2.5", ">2.5 & <3.0", ">3 & <=4", ">4"]
    if threshold == 2.0:
        return [">2.0 & 2.5", ">2.5 & <3.0", ">3 & <=4", ">4"]
    if threshold == 3.0:
        return [">3 & <=4", ">4"]
    if threshold == 4.0:
        return [">4"]
    # fallback
    return [">2.0 & 2.5", ">2.5 & <3.0", ">3 & <=4", ">4"]


def detected_columns_panel(meta: Dict):
    with st.expander("Detected columns (for troubleshooting)", expanded=False):
        st.write("**Original columns:**")
        st.code(", ".join(meta.get("original_cols", [])) or "(none)")
        st.write("**After normalization:**")
        st.code(", ".join(meta.get("cleaned_cols", [])) or "(none)")

        ignored = meta.get("ignored_cols", [])
        missing_b = meta.get("missing_buckets", [])
        if ignored:
            st.warning(f"Ignored columns: {', '.join(ignored)}")
        if missing_b:
            st.warning(f"Missing bucket columns treated as 0: {', '.join(missing_b)}")


def build_sample_files() -> Tuple[bytes, bytes]:
    sample = pd.DataFrame(
        [
            {
                "Suppliers": "ExampleSupplierA",
                "Total Requests": 1000,
                "<0.5 sec": 600,
                ">0.5 & <1.0": 200,
                ">1 & <1.5": 80,
                ">1.5 & <2.0": 60,
                ">2.0 & 2.5": 30,
                ">2.5 & <3.0": 20,
                ">3 & <=4": 8,
                ">4": 2,
            },
            {
                "Suppliers": "ExampleSupplierB",
                "Total Requests": 500,
                "<0.5 sec": 200,
                ">0.5 & <1.0": 120,
                ">1 & <1.5": 80,
                ">1.5 & <2.0": 40,
                ">2.0 & 2.5": 30,
                ">2.5 & <3.0": 15,
                ">3 & <=4": 10,
                ">4": 5,
            },
        ]
    )

    # Ensure correct column order and all buckets exist
    for b in CANON_BUCKETS:
        if b not in sample.columns:
            sample[b] = 0
    sample = sample[["Suppliers", "Total Requests"] + CANON_BUCKETS]

    csv_bytes = sample.to_csv(index=False).encode("utf-8")

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        sample.to_excel(writer, index=False, sheet_name="Timeouts")
    xlsx_bytes = bio.getvalue()

    return csv_bytes, xlsx_bytes


@st.cache_data(show_spinner=False)
def load_dataframe(uploaded_file) -> Tuple[pd.DataFrame, Dict]:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError("Unsupported file type. Please upload .xlsx or .csv")

    original_cols = [str(c) for c in df.columns]

    df, ignored_cols = _canonicalize_columns(df)
    cleaned_cols = [str(c) for c in df.columns]

    # required columns hard fail (clear error)
    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {', '.join(missing_required)}")

    # bucket columns: warn + treat as 0
    missing_buckets = []
    for b in CANON_BUCKETS:
        if b not in df.columns:
            df[b] = 0
            missing_buckets.append(b)

    # Convert numeric columns
    df["Total Requests"] = _parse_int_series(df["Total Requests"])
    for b in CANON_BUCKETS:
        df[b] = _parse_int_series(df[b])

    # Clean supplier names
    df["Suppliers"] = df["Suppliers"].astype(str).str.strip()

    # Remove empty supplier rows
    df = df[df["Suppliers"].str.len() > 0].reset_index(drop=True)

    meta = {
        "original_cols": original_cols,
        "cleaned_cols": cleaned_cols,
        "ignored_cols": ignored_cols,
        "missing_buckets": missing_buckets,
    }

    return df, meta


def compute_metrics(df: pd.DataFrame, threshold: float, total_source_choice: str) -> pd.DataFrame:
    out = df.copy()

    # Sum of buckets (for consistency check / optional total source)
    out["Buckets Sum"] = out[CANON_BUCKETS].sum(axis=1)

    if total_source_choice == "Use sum of buckets":
        out["Total (Used)"] = out["Buckets Sum"]
    else:
        out["Total (Used)"] = out["Total Requests"]

    # mismatch magnitude (can be used for alerts/QA)
    out["_MismatchAbs"] = (out["Total Requests"] - out["Buckets Sum"]).abs()

    # Standard metrics: >4, <0.5
    out[">4s Count"] = out[">4"]
    out[">4s Rate"] = np.where(out["Total (Used)"] > 0, out[">4s Count"] / out["Total (Used)"], 0.0)

    out["<0.5s Count"] = out["<0.5 sec"]
    out["<0.5s Rate"] = np.where(out["Total (Used)"] > 0, out["<0.5s Count"] / out["Total (Used)"], 0.0)

    # Threshold-driven "slow" definition
    slow_buckets = threshold_to_buckets(threshold)
    slow_label = f">{int(threshold)}s"

    out[f"{slow_label} Count"] = out[slow_buckets].sum(axis=1)
    out[f"{slow_label} Rate"] = np.where(out["Total (Used)"] > 0, out[f"{slow_label} Count"] / out["Total (Used)"], 0.0)

    # Estimated mean latency from buckets
    weighted = 0.0
    for b in CANON_BUCKETS:
        weighted += out[b] * MIDPOINTS[b]
    out["Estimated Mean (sec)"] = np.where(out["Total (Used)"] > 0, weighted / out["Total (Used)"], 0.0)

    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def make_figures(
    filtered: pd.DataFrame,
    top_n: int,
    slow_label: str,
    slow_count_col: str,
    slow_rate_col: str,
    total_used_col: str,
) -> Dict[str, object]:
    if filtered.empty:
        return {}

    # Top by rate
    top_rate = filtered.sort_values(slow_rate_col, ascending=False).head(top_n)
    fig_rate = px.bar(
        top_rate,
        x="Suppliers",
        y=slow_rate_col,
        title=f"Top {len(top_rate)} suppliers by % requests {slow_label}",
    )
    fig_rate.update_yaxes(tickformat=".0%")

    # Top by impact
    top_impact = filtered.sort_values(slow_count_col, ascending=False).head(top_n)
    fig_impact = px.bar(
        top_impact,
        x="Suppliers",
        y=slow_count_col,
        title=f"Top {len(top_impact)} suppliers by # requests {slow_label}",
    )

    # Scatter: volume vs rate (risk quadrant)
    fig_scatter = px.scatter(
        filtered,
        x=total_used_col,
        y=slow_rate_col,
        size=slow_count_col,
        hover_name="Suppliers",
        title=f"Risk view: Total volume vs % {slow_label} (bubble=size of {slow_label} count)",
    )
    fig_scatter.update_yaxes(tickformat=".0%")

    # Distribution stacked (normalize per supplier by bucket sum)
    dist_cols = CANON_BUCKETS
    dist = filtered[["Suppliers"] + dist_cols].copy()
    dist["TotalBuckets"] = dist[dist_cols].sum(axis=1).replace({0: np.nan})
    for b in dist_cols:
        dist[b] = dist[b] / dist["TotalBuckets"]
    dist = dist.drop(columns=["TotalBuckets"])

    dist_melt = dist.melt(id_vars=["Suppliers"], var_name="Bucket", value_name="Share").dropna()

    # Show only top suppliers by slow rate to keep the plot readable
    dist_show = filtered.sort_values(slow_rate_col, ascending=False).head(min(top_n, 20))["Suppliers"]
    dist_melt = dist_melt[dist_melt["Suppliers"].isin(dist_show)]

    fig_dist = px.bar(
        dist_melt,
        x="Suppliers",
        y="Share",
        color="Bucket",
        title=f"Latency distribution (share of requests per bucket) - top suppliers by {slow_label} rate",
    )
    fig_dist.update_yaxes(tickformat=".0%")

    return {
        "fig_rate": fig_rate,
        "fig_impact": fig_impact,
        "fig_scatter": fig_scatter,
        "fig_dist": fig_dist,
    }


def build_html_report(
    df_filtered: pd.DataFrame,
    threshold: float,
    total_source: str,
    figs: List[object],
) -> bytes:
    slow_label = f">{int(threshold)}s"
    slow_rate_col = f"{slow_label} Rate"
    slow_count_col = f"{slow_label} Count"

    parts = []
    parts.append(
        f"""
<html><head><meta charset="utf-8"/>
<title>Supplier Response Time Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 24px; }}
  .meta {{ color:#555; margin-bottom: 12px; }}
  h1,h2 {{ margin: 0.2em 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 12px; }}
  th {{ background: #f5f5f5; }}
</style>
</head><body>
<h1>Supplier Response Time Report</h1>
<div class="meta">
  Threshold: {slow_label} | Total source: {total_source}
</div>
"""
    )

    parts.append("<h2>Top suppliers (filtered)</h2>")
    show_cols = [
        "Suppliers",
        "Total (Used)",
        slow_count_col,
        slow_rate_col,
        ">4s Rate",
        "Estimated Mean (sec)",
    ]
    top_table = (
        df_filtered.sort_values(slow_rate_col, ascending=False)
        .head(30)[show_cols]
        .copy()
    )
    # Make rates readable in HTML
    if slow_rate_col in top_table.columns:
        top_table[slow_rate_col] = (top_table[slow_rate_col] * 100).round(2).astype(str) + "%"
    if ">4s Rate" in top_table.columns:
        top_table[">4s Rate"] = (top_table[">4s Rate"] * 100).round(2).astype(str) + "%"

    parts.append(top_table.to_html(index=False))

    parts.append("<h2>Charts</h2>")
    for i, fig in enumerate(figs):
        parts.append(
            pio.to_html(
                fig,
                include_plotlyjs=("cdn" if i == 0 else False),
                full_html=False,
            )
        )

    parts.append("</body></html>")
    return "\n".join(parts).encode("utf-8")


# ----------------------------
# UI
# ----------------------------
st.title("Supplier Response Time Analysis")
st.caption(
    "Upload a .xlsx or .csv file. The app will compute latency bucket KPIs, highlight slow suppliers, and generate charts + exports."
)

# Sidebar: sample downloads + drilldown (once data exists)
with st.sidebar:
    st.header("Help")
    csv_sample, xlsx_sample = build_sample_files()
    st.download_button("Download sample CSV", data=csv_sample, file_name="sample_timeouts.csv", mime="text/csv")
    st.download_button(
        "Download sample Excel",
        data=xlsx_sample,
        file_name="sample_timeouts.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("Use this sample to match the expected columns/buckets.")

uploaded = st.file_uploader("Upload file", type=["xlsx", "csv"])

if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

try:
    df_raw, meta = load_dataframe(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

# Column troubleshooting panel
detected_columns_panel(meta)

# Total source choice (correctness control)
total_source = st.radio(
    "Total Requests source",
    ["Use 'Total Requests' column", "Use sum of buckets"],
    index=0,
    horizontal=True,
)

# Filters state
def init_state():
    st.session_state.setdefault("search", "")
    st.session_state.setdefault("min_rate_pct", 0)  # 0..100
    st.session_state.setdefault("top_n", 20)
    st.session_state.setdefault("threshold", 2.0)

def reset_filters():
    st.session_state["search"] = ""
    st.session_state["min_rate_pct"] = 0
    st.session_state["top_n"] = 20
    st.session_state["threshold"] = 2.0

init_state()

# Filters UI
st.subheader("Filters")
f0, f1, f2, f3 = st.columns([1.2, 2, 2, 2])

f0.button("Reset filters", on_click=reset_filters)

threshold = f1.selectbox(
    "Slow threshold",
    [1.0, 2.0, 3.0, 4.0],
    index=[1.0, 2.0, 3.0, 4.0].index(st.session_state["threshold"]),
    key="threshold",
    format_func=lambda x: f">{int(x)}s",
)

# compute metrics AFTER threshold + total_source
df = compute_metrics(df_raw, threshold=threshold, total_source_choice=total_source)

slow_label = f">{int(threshold)}s"
slow_count_col = f"{slow_label} Count"
slow_rate_col = f"{slow_label} Rate"
total_used_col = "Total (Used)"

# Data quality warning: mismatch counts (row-wise)
mismatch_rows = int((df["Total Requests"] != df["Buckets Sum"]).sum())
if mismatch_rows > 0:
    st.warning(
        f"{mismatch_rows} supplier rows have Total Requests ≠ sum(bucket columns). "
        f"Current setting: {total_source}."
    )

search = f2.text_input("Search supplier", key="search")
min_rate_pct = f3.slider(f"Min {slow_label} Rate (%)", 0, 100, st.session_state["min_rate_pct"], 1, key="min_rate_pct")
min_rate = min_rate_pct / 100.0

top_n = st.selectbox("Top N for charts", [10, 20, 30, 50], index=[10, 20, 30, 50].index(st.session_state["top_n"]), key="top_n")

st.caption(
    f"Active filters: search='{search}', minRate={min_rate_pct}%, topN={top_n}, threshold={slow_label}, totalSource='{total_source}'."
)

# Apply filters
filtered = df.copy()
if search.strip():
    filtered = filtered[filtered["Suppliers"].str.contains(search.strip(), case=False, na=False)]
filtered = filtered[filtered[slow_rate_col] >= min_rate]

# Global KPIs (based on filtered? better overall based on all)
total_requests_all = int(df[total_used_col].sum())
slow_all = int(df[slow_count_col].sum())
gt4_all = int(df[">4s Count"].sum())

slow_rate_all = (slow_all / total_requests_all) if total_requests_all else 0
gt4_rate_all = (gt4_all / total_requests_all) if total_requests_all else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Requests (Used)", f"{total_requests_all:,}")
c2.metric(f"{slow_label} Requests", f"{slow_all:,}", f"{slow_rate_all*100:.2f}%")
c3.metric(">4s Requests", f"{gt4_all:,}", f"{gt4_rate_all*100:.2f}%")
c4.metric("Suppliers", f"{df['Suppliers'].nunique():,}")

# Top issues cards (from FILTERED dataset)
def top_row(df_, col, ascending=False):
    if df_.empty:
        return None
    return df_.sort_values(col, ascending=ascending).iloc[0]

worst_rate = top_row(filtered, slow_rate_col, ascending=False)
worst_impact = top_row(filtered, slow_count_col, ascending=False)
worst_gt4 = top_row(filtered, ">4s Rate", ascending=False)
most_volume = top_row(filtered, total_used_col, ascending=False)

st.subheader("Top issues (based on current filters)")
t1, t2, t3, t4 = st.columns(4)
if worst_rate is not None:
    t1.metric("Worst slow %", f"{worst_rate[slow_rate_col]*100:.2f}%", worst_rate["Suppliers"])
else:
    t1.metric("Worst slow %", "—")
if worst_impact is not None:
    t2.metric("Highest slow volume", f"{int(worst_impact[slow_count_col]):,}", worst_impact["Suppliers"])
else:
    t2.metric("Highest slow volume", "—")
if worst_gt4 is not None:
    t3.metric("Worst >4s %", f"{worst_gt4['>4s Rate']*100:.2f}%", worst_gt4["Suppliers"])
else:
    t3.metric("Worst >4s %", "—")
if most_volume is not None:
    t4.metric("Most traffic", f"{int(most_volume[total_used_col]):,}", most_volume["Suppliers"])
else:
    t4.metric("Most traffic", "—")

# Sidebar drilldown selector
with st.sidebar:
    st.header("Supplier drilldown")
    if filtered.empty:
        st.info("No suppliers match the current filters.")
        selected_supplier = None
    else:
        selected_supplier = st.selectbox(
            "Select supplier",
            options=sorted(filtered["Suppliers"].unique().tolist()),
            index=0,
        )

# Tabs
tab1, tab2, tab3 = st.tabs(["Charts", "Table", "Export"])

figs = make_figures(
    filtered=filtered,
    top_n=top_n,
    slow_label=slow_label,
    slow_count_col=slow_count_col,
    slow_rate_col=slow_rate_col,
    total_used_col=total_used_col,
)

with tab1:
    st.subheader("Problem suppliers")
    if filtered.empty:
        st.info("No data to chart (filters removed all rows).")
    else:
        st.plotly_chart(figs["fig_rate"], use_container_width=True)
        st.plotly_chart(figs["fig_impact"], use_container_width=True)
        st.plotly_chart(figs["fig_scatter"], use_container_width=True)
        st.plotly_chart(figs["fig_dist"], use_container_width=True)

    # Drilldown details in main area (optional, but handy)
    if selected_supplier:
        st.divider()
        st.subheader(f"Drilldown: {selected_supplier}")
        srow = filtered[filtered["Suppliers"] == selected_supplier].iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total (Used)", f"{int(srow[total_used_col]):,}")
        k2.metric(slow_count_col, f"{int(srow[slow_count_col]):,}", f"{srow[slow_rate_col]*100:.2f}%")
        k3.metric(">4s Count", f"{int(srow['>4s Count']):,}", f"{srow['>4s Rate']*100:.2f}%")
        k4.metric("Est. Mean (sec)", f"{srow['Estimated Mean (sec)']:.3f}")

        dist = pd.DataFrame({"Bucket": CANON_BUCKETS, "Count": [int(srow[b]) for b in CANON_BUCKETS]})
        fig_one = px.bar(dist, x="Bucket", y="Count", title="Latency bucket distribution")
        st.plotly_chart(fig_one, use_container_width=True)

with tab2:
    st.subheader("Supplier metrics table")
    view_cols = [
        "Suppliers",
        "Total Requests",
        "Buckets Sum",
        "Total (Used)",
        slow_count_col,
        slow_rate_col,
        ">4s Count",
        ">4s Rate",
        "<0.5s Count",
        "<0.5s Rate",
        "Estimated Mean (sec)",
    ]

    table = filtered[view_cols].sort_values(slow_rate_col, ascending=False) if not filtered.empty else filtered[view_cols]
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            slow_rate_col: st.column_config.NumberColumn(format="%.4f"),
            ">4s Rate": st.column_config.NumberColumn(format="%.4f"),
            "<0.5s Rate": st.column_config.NumberColumn(format="%.4f"),
            "Estimated Mean (sec)": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption("Tip: Rates are decimals (0.10 = 10%).")

with tab3:
    st.subheader("Export")

    # KPI-only exports
    kpi_cols = [
        "Suppliers",
        "Total (Used)",
        slow_count_col,
        slow_rate_col,
        ">4s Count",
        ">4s Rate",
        "<0.5s Count",
        "<0.5s Rate",
        "Estimated Mean (sec)",
    ]

    st.download_button(
        "Download KPI CSV (filtered)",
        data=to_csv_bytes(filtered[kpi_cols]) if not filtered.empty else to_csv_bytes(pd.DataFrame(columns=kpi_cols)),
        file_name=f"supplier_kpis_filtered_{slow_label.replace('>', 'gt')}.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download KPI CSV (all suppliers)",
        data=to_csv_bytes(df[kpi_cols]),
        file_name=f"supplier_kpis_all_{slow_label.replace('>', 'gt')}.csv",
        mime="text/csv",
    )

    # Full data exports (includes buckets + totals)
    st.download_button(
        "Download FULL CSV (filtered)",
        data=to_csv_bytes(filtered),
        file_name=f"supplier_full_filtered_{slow_label.replace('>', 'gt')}.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download FULL CSV (all suppliers)",
        data=to_csv_bytes(df),
        file_name=f"supplier_full_all_{slow_label.replace('>', 'gt')}.csv",
        mime="text/csv",
    )

    # HTML report export (only if we have charts)
    if filtered.empty:
        st.info("HTML report is disabled because there is no filtered data.")
    else:
        html_bytes = build_html_report(
            df_filtered=filtered,
            threshold=threshold,
            total_source=total_source,
            figs=[figs["fig_rate"], figs["fig_impact"], figs["fig_scatter"], figs["fig_dist"]],
        )
        st.download_button(
            "Download HTML report",
            data=html_bytes,
            file_name=f"supplier_report_{slow_label.replace('>', 'gt')}.html",
            mime="text/html",
        )

st.caption(
    "Note: Estimated Mean is derived from bucket midpoints; it’s an approximation when only bucket counts are available."
)
