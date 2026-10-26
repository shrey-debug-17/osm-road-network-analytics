import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import pydeck as pdk
from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = Path.home() / "bda-osm"
load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="ROADSCOPE | Network Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- VISUAL DESIGN ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080e18;
    --panel: #101a29;
    --panel2: #142235;
    --line: #23354a;
    --text: #e8f0fa;
    --muted: #8fa4bb;
    --teal: #42e6c2;
    --amber: #ffbd69;
}
html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}
.stApp {
    background:
      radial-gradient(ellipse at 75% 0%, #132a39 0%, transparent 42%),
      #080e18;
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: #0b1320;
    border-right: 1px solid #23354a;
}
[data-testid="stSidebar"] * { color: #dce8f5; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1600px; }
h1, h2, h3 { letter-spacing: -0.045em; color: #edf5ff; }
h1 { font-size: 2.45rem !important; font-weight: 800 !important; }
h2 { font-size: 1.45rem !important; font-weight: 750 !important; }
h3 { font-size: 1.05rem !important; font-weight: 700 !important; }
p, label { color: #b7c7d9; }
small, [data-testid="stCaptionContainer"] { color: #8fa4bb !important; }
hr { border-color: #23354a; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #132236, #0e1827);
    border: 1px solid #263b51;
    border-radius: 14px;
    padding: 18px 20px;
    min-height: 112px;
}
div[data-testid="stMetricLabel"] { color: #9fb4c9; font-size: .78rem; }
div[data-testid="stMetricValue"] {
    color: #eaf6ff; font-weight: 800; font-size: 1.7rem;
}
div[data-testid="stMetricDelta"] { font-family: 'DM Mono', monospace; }
div[data-testid="stPlotlyChart"], div[data-testid="stVegaLiteChart"],
div[data-testid="stDataFrame"] {
    border: 1px solid #23354a;
    border-radius: 12px;
    overflow: hidden;
}
.stButton button, .stDownloadButton button {
    background: #17382f;
    border: 1px solid #2a806d;
    color: #78f5d9;
    border-radius: 9px;
    font-weight: 700;
}
.stButton button:hover, .stDownloadButton button:hover {
    background: #205143;
    border-color: #42e6c2;
    color: white;
}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    background: #111e2e;
    border-color: #30455c;
}
div[data-testid="stAlert"] { border-radius: 10px; }
.tag {
    display: inline-block;
    font: 500 11px 'DM Mono', monospace;
    letter-spacing: .12em;
    color: #42e6c2;
    border: 1px solid #27665d;
    background: #102923;
    border-radius: 5px;
    padding: 5px 8px;
    margin-bottom: 12px;
}
.hero {
    padding: 8px 0 22px 0;
    border-bottom: 1px solid #22364b;
    margin-bottom: 24px;
}
.hero-sub { color: #91a7be; font-size: 0.94rem; margin-top: -10px; }
.section-note {
    color: #8fa4bb;
    font-size: .82rem;
    margin-top: -8px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# ---------- DATA ACCESS ----------
@st.cache_data(ttl=300)
def load_collection(name):
    uri = os.getenv("MONGODB_URI")
    if not uri:
        try:
            uri = st.secrets["MONGODB_URI"]
        except Exception:
            uri = None
    if not uri:
        raise RuntimeError("MONGODB_URI is missing from ~/bda-osm/.env")
    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    try:
        client.admin.command("ping")
        docs = list(client["bda_osm"][name].find(
            {}, {"_id": 0, "exported_at": 0}
        ))
        return pd.DataFrame(docs)
    finally:
        client.close()


@st.cache_data(ttl=300)
def load_data():
    return {
        "temporal": load_collection("temporal_comparison"),
        "regional": load_collection("regional_metrics"),
        "anomalies": load_collection("regional_anomalies"),
    }


@st.cache_data(show_spinner="Loading road geometry...")
def load_roads(max_ways=42000):
    path = ROOT / "road_map_deploy.geojsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    # Evenly sample the file without loading all 136 MiB into memory.
    total = sum(1 for _ in path.open("r", encoding="utf-8"))
    stride = max(1, total // max_ways)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride:
                continue
            try:
                feature = json.loads(line)
                coords = feature["geometry"]["coordinates"]
                if len(coords) >= 2:
                    rows.append({
                        "coordinates": coords,
                        "highway": feature["properties"].get("highway", "road"),
                        "osm_id": feature["properties"].get("id"),
                    })
            except (ValueError, KeyError, TypeError):
                continue
            if len(rows) >= max_ways:
                break
    return pd.DataFrame(rows)


def render_road_map(roads, overlay=None, overlay_label=""):
    layers = []
    if not roads.empty:
        layers.append(pdk.Layer(
            "PathLayer",
            data=roads,
            get_path="coordinates",
            get_color=[48, 205, 181, 180],
            get_width=1.5,
            width_min_pixels=1,
            width_max_pixels=3,
            pickable=True,
            auto_highlight=True,
        ))
    if overlay is not None and not overlay.empty:
        points = overlay.rename(columns={
            "grid_lon": "longitude",
            "grid_lat": "latitude",
        }).copy()
        points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
        points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
        points = points.dropna(subset=["longitude", "latitude"])
        if not points.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=points,
                get_position="[longitude, latitude]",
                get_radius=1500,
                radius_min_pixels=4,
                radius_max_pixels=11,
                get_fill_color=[255, 174, 91, 220],
                get_line_color=[255, 221, 168, 230],
                line_width_min_pixels=1,
                stroked=True,
                filled=True,
                pickable=True,
            ))

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=24.7, longitude=79.3, zoom=4.0,
            pitch=0, bearing=0,
        ),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={
            "html": "<b>{highway}</b><br/>OSM way: {osm_id}",
            "style": {
                "backgroundColor": "#101a29",
                "color": "#e8f0fa",
                "fontFamily": "Manrope",
            },
        },
    )
    st.pydeck_chart(deck, use_container_width=True, height=520)


def header(kicker, title, subtitle):
    st.markdown(
        f'<div class="hero"><div class="tag">{kicker}</div>'
        f'<h1>{title}</h1><div class="hero-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(
        '<div style="font-size:12px;letter-spacing:.2em;color:#42e6c2;'
        'font-weight:800;margin:10px 0 8px">◈ ROADSCOPE</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Network intelligence")
    st.caption("BIG DATA ANALYTICS  /  OSM")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Road Network", "Temporal Evolution", "Anomaly Lab"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**DATA PIPELINE**")
    st.caption("01  HDFS · distributed storage")
    st.caption("02  Spark · large-scale processing")
    st.caption("03  MongoDB · analytical results")
    st.divider()
    if st.button("↻  Refresh dashboard", use_container_width=True):
        load_data.clear()
        load_collection.clear()
        load_roads.clear()
        st.rerun()
    st.caption("Central-zone India · OSM snapshots")


# ---------- LOAD ANALYTICS ----------
try:
    with st.spinner("Connecting to analytics store..."):
        data = load_data()
except Exception as exc:
    header("SYSTEM STATUS", "Data connection unavailable", "Check the database connection and retry.")
    st.error("Could not load MongoDB Atlas results.")
    st.code(str(exc))
    st.info("Verify internet connectivity, Atlas network access and the MONGODB_URI setting in .env. Keep the URI private.")
    st.stop()

temporal = data["temporal"].copy()
regional = data["regional"].copy()
anomalies = data["anomalies"].copy()

if temporal.empty or regional.empty or anomalies.empty:
    st.error("At least one required MongoDB collection is empty.")
    st.stop()

temporal["snapshot_date"] = pd.to_datetime(temporal["snapshot_date"], errors="coerce")
temporal = temporal.dropna(subset=["snapshot_date"]).sort_values("snapshot_date")
if temporal.empty:
    st.error("No valid snapshot dates were found.")
    st.stop()

latest = temporal.iloc[-1]
previous = temporal.iloc[-2] if len(temporal) > 1 else None
segments = int(latest.get("segment_count", 0))
length_km = float(latest.get("total_length_m", 0)) / 1000
candidate_flag = anomalies.get(
    "is_anomaly_candidate", pd.Series(False, index=anomalies.index)
).fillna(False).astype(bool)
candidates = anomalies[candidate_flag].copy()
snapshot = latest["snapshot_date"].strftime("%d %b %Y")


# ---------- OVERVIEW ----------
if page == "Overview":
    header(
        "NETWORK INTELLIGENCE / 01",
        "Road network, in focus.",
        "A regional view of mapped road evolution and statistical outliers across dated OpenStreetMap snapshots.",
    )
    st.markdown(
        '<div class="tag">● ANALYTICS STORE CONNECTED</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LATEST SNAPSHOT", snapshot)
    c2.metric("ROAD SEGMENTS", f"{segments:,}")
    c3.metric("MAPPED LENGTH", f"{length_km:,.0f} km")
    c4.metric("OUTLIER CANDIDATES", f"{len(candidates):,}")

    st.markdown("## Regional road system")
    st.markdown(
        '<div class="section-note">Major road classes · teal lines represent sampled OSM ways · amber markers represent statistical anomaly candidates.</div>',
        unsafe_allow_html=True,
    )
    try:
        roads = load_roads()
        if roads.empty:
            st.warning("Road-map dataset not found. Expected data/processed/road_map.geojsonl.")
        else:
            render_road_map(roads, candidates.head(1200))
            st.caption(f"Rendered {len(roads):,} sampled major-road ways for responsiveness. Full geometry remains stored locally.")
    except Exception as exc:
        st.warning(f"Road map could not be rendered: {exc}")

    st.markdown("## Evolution at a glance")
    left, right = st.columns(2)
    with left:
        st.markdown("**Mapped segment count**")
        chart = temporal.set_index("snapshot_date")[["segment_count"]]
        st.line_chart(chart, color=["#42e6c2"], x_label="Snapshot", y_label="Segments")
    with right:
        st.markdown("**Mapped road length**")
        chart = temporal.copy()
        chart["Length (km)"] = chart["total_length_m"] / 1000
        st.line_chart(
            chart.set_index("snapshot_date")[["Length (km)"]],
            color=["#ffbd69"], x_label="Snapshot", y_label="Kilometres",
        )

    st.markdown("## Snapshot ledger")
    display = temporal.copy()
    display["snapshot_date"] = display["snapshot_date"].dt.strftime("%Y-%m-%d")
    display["total_length_km"] = display["total_length_m"] / 1000
    cols = [c for c in [
        "snapshot_date", "segment_count", "total_length_km",
        "segment_change_pct", "length_change_pct",
    ] if c in display.columns]
    st.dataframe(display[cols], width="stretch", hide_index=True)

    st.info(
        "Interpretation note: snapshot differences measure changes in OSM data representation and coverage. "
        "They do not independently prove physical road construction."
    )


# ---------- ROAD NETWORK ----------
elif page == "Road Network":
    header(
        "SPATIAL EXPLORER / 02",
        "Explore the network.",
        "Inspect regional grid metrics alongside the actual geometry of major mapped roads.",
    )
    min_segments = int(pd.to_numeric(regional["segment_count"], errors="coerce").min())
    max_segments = int(pd.to_numeric(regional["segment_count"], errors="coerce").max())
    default_threshold = min(max(100, min_segments), max_segments)
    threshold = st.slider(
        "Minimum segments per grid cell",
        min_value=min_segments,
        max_value=max_segments,
        value=default_threshold,
    )
    filtered = regional[
        pd.to_numeric(regional["segment_count"], errors="coerce") >= threshold
    ].copy()

    a, b, c = st.columns(3)
    a.metric("MATCHING GRID CELLS", f"{len(filtered):,}")
    b.metric("SEGMENTS IN CELLS", f"{int(filtered['segment_count'].sum()):,}")
    c.metric("MAPPED LENGTH", f"{filtered['total_length_km'].sum():,.1f} km")

    try:
        roads = load_roads()
        if not roads.empty:
            render_road_map(roads)
        else:
            st.warning("No road geometry found.")
    except Exception as exc:
        st.warning(f"Map unavailable: {exc}")

    st.markdown("## Grid metrics")
    st.caption("Grid cells are 0.1° longitude/latitude bins, not equal-area cells.")
    options = [c for c in [
        "segment_count", "total_length_km", "junction_node_count",
        "avg_node_degree", "avg_segment_length_m",
    ] if c in filtered.columns]
    if options:
        sort_by = st.selectbox("Rank grid cells by", options)
        st.dataframe(
            filtered.sort_values(sort_by, ascending=False),
            width="stretch", hide_index=True,
        )
    st.download_button(
        "Export regional metrics CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="osm_regional_metrics.csv",
        mime="text/csv",
    )


# ---------- TEMPORAL EVOLUTION ----------
elif page == "Temporal Evolution":
    header(
        "TIME SERIES / 03",
        "Change, over time.",
        "Compare the mapped network across the available dated snapshots.",
    )
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("### Segment count")
        st.bar_chart(
            temporal.set_index("snapshot_date")[["segment_count"]],
            color=["#42e6c2"], x_label="Snapshot", y_label="Segments",
        )
    with t2:
        st.markdown("### Mapped length")
        length_data = temporal.copy()
        length_data["Length (km)"] = length_data["total_length_m"] / 1000
        st.bar_chart(
            length_data.set_index("snapshot_date")[["Length (km)"]],
            color=["#ffbd69"], x_label="Snapshot", y_label="Kilometres",
        )

    st.markdown("## Interval changes")
    change_cols = [c for c in [
        "snapshot_date", "segment_change_pct", "length_change_pct",
    ] if c in temporal.columns]
    change = temporal[change_cols].copy()
    change["snapshot_date"] = change["snapshot_date"].dt.strftime("%Y-%m-%d")
    st.dataframe(change, width="stretch", hide_index=True)
    st.info(
        "These are dated snapshots, not complete per-edit history. Changes can include additions, edits, "
        "reclassification, and coverage differences in OpenStreetMap."
    )


# ---------- ANOMALY LAB ----------
elif page == "Anomaly Lab":
    header(
        "STATISTICAL REVIEW / 04",
        "Investigate outliers.",
        "Find grid cells with unusual road-network metrics and inspect their geographic context.",
    )
    st.warning(
        "An anomaly candidate is a statistical outlier, not a confirmed road defect. "
        "Urban density, incomplete mapping, geography and tagging patterns may affect scores."
    )
    st.markdown("## Candidate overview")
    a, b, c = st.columns(3)
    a.metric("ELIGIBLE CELLS", f"{len(anomalies):,}")
    b.metric("CANDIDATE CELLS", f"{len(candidates):,}")
    c.metric(
        "CANDIDATE SHARE",
        f"{100 * len(candidates) / len(anomalies):.2f}%" if len(anomalies) else "N/A",
    )

    score_options = [c for c in [
        "segment_count_robust_z", "total_length_km_robust_z",
        "avg_segment_length_m_robust_z", "avg_node_degree_robust_z",
        "junction_node_count_robust_z",
    ] if c in candidates.columns]

    if not score_options:
        st.error("Robust z-score columns are not available in the anomaly collection.")
        st.stop()

    x, y = st.columns([2, 1])
    with x:
        chosen_score = st.selectbox("Score dimension", score_options)
    with y:
        min_score = st.slider("Minimum absolute robust z-score", 0.0, 10.0, 3.5, 0.5)

    ranked = candidates.copy()
    ranked[chosen_score] = pd.to_numeric(ranked[chosen_score], errors="coerce")
    ranked = ranked[ranked[chosen_score].abs() >= min_score]
    ranked = ranked.sort_values(chosen_score, key=lambda s: s.abs(), ascending=False)

    st.metric("MATCHING CANDIDATES", f"{len(ranked):,}")
    if not ranked.empty:
        st.markdown("## Geographic context")
        try:
            roads = load_roads()
            if not roads.empty:
                render_road_map(roads, ranked.head(1500))
        except Exception as exc:
            st.warning(f"Map unavailable: {exc}")

        st.markdown("## Ranked candidates")
        st.dataframe(ranked, width="stretch", hide_index=True)
        st.download_button(
            "Export candidates CSV",
            ranked.to_csv(index=False).encode("utf-8"),
            file_name="osm_anomaly_candidates.csv",
            mime="text/csv",
        )
    else:
        st.info("No candidates match the selected score threshold.")


# ---------- FOOTER ----------
st.divider()
st.caption(
    "ROADSCOPE  /  BDA PROJECT     •     OpenStreetMap snapshot analysis     •     "
    "Hadoop HDFS + Apache Spark + MongoDB Atlas"
)
