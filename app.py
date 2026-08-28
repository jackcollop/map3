"""Dash app: cotton planted acres by US county.

Filters: irrigation practice (irrigated / non-irrigated / all),
crop type (Upland / Pima), and as-of date.
"""
import json
import os

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# --- Paths (data files live alongside this script) ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "cotton_county_data.csv")
GEOJSON_PATH = os.path.join(BASE_DIR, "counties-fips.json")

# --- Load data once at startup ----------------------------------------------
COLS = ["as_of_date", "crop_year", "month", "State County Code",
        "Irrigation Practice", "crop_normalized", "Planted Acres",
        "Failed Acres", "Prevented Acres", "Planted and Failed Acres",
        "State", "County"]
df = pd.read_csv(CSV_PATH, usecols=COLS, low_memory=False)
df["fips"] = df["State County Code"].astype(int).map(lambda x: f"{x:05d}")

# fips -> "County, State" for hover labels
NAME_BY_FIPS = (
    df.drop_duplicates("fips")
      .assign(name=lambda x: x["County"].str.title() + ", " + x["State"].str.title())
      .set_index("fips")["name"]
)

with open(GEOJSON_PATH) as f:
    counties = json.load(f)

# --- Ag Statistics District geography (optional; built by build_asd.py) ------
ASD_GEOJSON_PATH = os.path.join(BASE_DIR, "asd_borders.json")
ASD_XW_PATH = os.path.join(BASE_DIR, "asd_crosswalk.json")
ASD_AVAILABLE = os.path.exists(ASD_GEOJSON_PATH) and os.path.exists(ASD_XW_PATH)
if ASD_AVAILABLE:
    with open(ASD_GEOJSON_PATH) as f:
        asd_geojson = json.load(f)
    with open(ASD_XW_PATH) as f:
        _asd_xw = json.load(f)
    df["asd_id"] = df["fips"].map(lambda x: (_asd_xw.get(x) or {}).get("id"))
    NAME_BY_ASD = {v["id"]: v["name"] for v in _asd_xw.values()}
else:
    asd_geojson, NAME_BY_ASD = None, {}

DATES = sorted(df["as_of_date"].unique(), reverse=True)

CROP_MAP = {"Upland": "Cotton, Upland", "Pima": "Cotton, ELS"}
IRRIG_LABELS = {"All": "All", "I": "Irrigated", "N": "Non-irrigated"}
VIEW_LABELS = {"abs": "Absolute acres", "yoy": "YoY change"}
METRIC_LABELS = {"Planted Acres": "Planted", "Failed Acres": "Failed",
                 "Failed %": "Failed %"}
GEO_LABELS = {"county": "County", "asd": "Ag district"}
# Diverging colormap for YoY change: red (decrease) -> white (0) -> green (increase)
DIV_RWG = ["#b2182b", "#f7f7f7", "#1a9850"]

# date -> (crop_year, month) for locating the prior-year comparison snapshot
DATE_META = (df.drop_duplicates("as_of_date")
               .set_index("as_of_date")[["crop_year", "month"]]
               .to_dict("index"))

# --- App --------------------------------------------------------------------
app = Dash(__name__)
app.title = "Cotton Planted Acres"

_control = {"marginBottom": "18px"}
_label = {"fontWeight": "600", "display": "block", "marginBottom": "6px"}

app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "maxWidth": "1200px",
           "margin": "0 auto", "padding": "20px"},
    children=[
        html.H1("Cotton Planted Acres by County", style={"marginBottom": "4px"}),
        html.P("USDA FSA acreage data", style={"color": "#666", "marginTop": 0}),
        html.Div(
            style={"display": "flex", "gap": "40px", "flexWrap": "wrap",

                   "alignItems": "flex-start"},
            children=[
                html.Div(style=_control, children=[
                    html.Label("Crop type", style=_label),
                    dcc.RadioItems(
                        id="crop", options=list(CROP_MAP.keys()),
                        value="Upland", inline=True,
                        labelStyle={"marginRight": "16px"}),
                ]),
                html.Div(style=_control, children=[
                    html.Label("Irrigation", style=_label),
                    dcc.RadioItems(
                        id="irrig",
                        options=[{"label": v, "value": k}
                                 for k, v in IRRIG_LABELS.items()],
                        value="All", inline=True,
                        labelStyle={"marginRight": "16px"}),
                ]),
                html.Div(style={**_control, "minWidth": "180px"}, children=[
                    html.Label("As-of date", style=_label),
                    dcc.Dropdown(
                        id="date", options=DATES, value=DATES[0],
                        clearable=False),
                ]),
                html.Div(style=_control, children=[
                    html.Label("Metric", style=_label),
                    dcc.RadioItems(
                        id="metric",
                        options=[{"label": v, "value": k}
                                 for k, v in METRIC_LABELS.items()],
                        value="Planted Acres", inline=True,
                        labelStyle={"marginRight": "16px"}),
                ]),
                html.Div(style=_control, children=[
                    html.Label("View", style=_label),
                    dcc.RadioItems(
                        id="view",
                        options=[{"label": v, "value": k}
                                 for k, v in VIEW_LABELS.items()],
                        value="abs", inline=True,
                        labelStyle={"marginRight": "16px"}),
                ]),
                html.Div(style=_control, children=[
                    html.Label("Geography", style=_label),
                    dcc.RadioItems(
                        id="geo",
                        options=[{"label": v, "value": k, "disabled":
                                  k == "asd" and not ASD_AVAILABLE}
                                 for k, v in GEO_LABELS.items()],
                        value="county", inline=True,
                        labelStyle={"marginRight": "16px"}),
                ]),
            ],
        ),
        dcc.Graph(id="map", style={"height": "640px"}),
    ],
)


def _filter(sub, crop, irrig):
    sub = sub[sub["crop_normalized"] == CROP_MAP[crop]]
    if irrig != "All":
        sub = sub[sub["Irrigation Practice"] == irrig]
    return sub


def _agg(sub, loc):
    """Per-geography (county fips or ASD id) sums of raw acreage columns."""
    return sub.groupby(loc)[
        ["Planted Acres", "Failed Acres", "Prevented Acres",
         "Planted and Failed Acres"]].sum()


def _metric_series(g, metric):
    """Per-county value for the chosen metric.

    Failed % is failed / "Planted and Failed Acres" * 100.
    """
    if metric == "Failed %":
        denom = g["Planted and Failed Acres"]
        return g["Failed Acres"] / denom.where(denom > 0) * 100
    return g[metric]


@app.callback(
    Output("map", "figure"),
    Input("crop", "value"),
    Input("irrig", "value"),
    Input("date", "value"),
    Input("view", "value"),
    Input("metric", "value"),
    Input("geo", "value"),
)
def update_map(crop, irrig, date, view, metric, geo):
    d = _filter(df[df["as_of_date"] == date], crop, irrig)
    mlabel = METRIC_LABELS[metric]
    is_pct = metric == "Failed %"
    unit = "%" if is_pct else " Acres"
    vfmt = ".1f" if is_pct else ",.0f"
    # For failed metrics, more is worse -> reverse so high/increase reads red.
    reverse = metric in ("Failed Acres", "Failed %")
    seq_scale = "YlGn_r" if reverse else "YlGn"
    # Diverging red -> white -> green so 0 is white (paired with symmetric
    # range_color and midpoint=0 below). Reversed for "more is worse" metrics.
    div_scale = DIV_RWG[::-1] if reverse else DIV_RWG

    # Geography: county (fips) or Ag Statistics District (asd_id).
    if geo == "asd" and ASD_AVAILABLE:
        loc, geoms, name_map = "asd_id", asd_geojson, NAME_BY_ASD
    else:
        geo = "county"
        loc, geoms, name_map = "fips", counties, NAME_BY_FIPS
    geo_word = GEO_LABELS[geo]

    if view == "abs":
        g = _agg(d, loc)
        g["val"] = _metric_series(g, metric)
        g["failed_pct"] = _metric_series(g, "Failed %")
        ca = g.reset_index()
        ca["name"] = ca[loc].map(name_map)

        top = ca["val"].quantile(0.98) if len(ca) else 1
        fig = px.choropleth(
            ca,
            geojson=geoms,
            locations=loc,
            color="val",
            color_continuous_scale=seq_scale,
            range_color=(0, top or 1),
            scope="usa",
            labels={"val": f"{mlabel}{unit if is_pct else ''}"},
            custom_data=["name", "Planted Acres", "Failed Acres",
                         "Prevented Acres", "failed_pct"],
        )
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Planted Acres: %{customdata[1]:,.0f}<br>"
                          "Failed Acres: %{customdata[2]:,.0f}<br>"
                          "Prevented Acres: %{customdata[3]:,.0f}<br>"
                          "Failed %: %{customdata[4]:.1f}%<extra></extra>",
        )
        fig.update_layout(
            title=f"{crop} — {IRRIG_LABELS[irrig]} — {mlabel}{unit} "
                  f"by {geo_word} — as of {date}",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    # --- Year-over-year change: this snapshot vs same month, prior crop year ---
    meta = DATE_META[date]
    cur_year, month = meta["crop_year"], meta["month"]
    base_year = cur_year - 1
    base = _filter(
        df[(df["crop_year"] == base_year) & (df["month"] == month)], crop, irrig)

    if base.empty:
        # No comparison snapshot for prior year / month
        fig = px.choropleth(scope="usa")
        fig.update_layout(
            title=f"No {base_year} {month} snapshot to compare against",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    m = pd.concat(
        [_metric_series(_agg(d, loc), metric).rename("cur"),
         _metric_series(_agg(base, loc), metric).rename("base")],
        axis=1).reset_index()
    # For pct, drop areas without both years (NaN); for acres treat missing as 0.
    m = m.dropna(subset=["cur", "base"]) if is_pct else m.fillna(0.0)
    m["delta"] = m["cur"] - m["base"]
    m["name"] = m[loc].map(name_map)

    # Change units: percentage points for pct, acres otherwise.
    chg_unit = " pp" if is_pct else ""
    lim = m["delta"].abs().quantile(0.98) if len(m) else 1
    lim = lim or 1
    fig = px.choropleth(
        m,
        geojson=geoms,
        locations=loc,
        color="delta",
        color_continuous_scale=div_scale,
        range_color=(-lim, lim),
        color_continuous_midpoint=0,
        scope="usa",
        labels={"delta": f"Δ {mlabel}"},
        custom_data=["name", "base", "cur"],
    )
    pu = "%" if is_pct else ""
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>"
                      + f"{base_year} {month}: " + "%{customdata[1]:" + vfmt + "}" + pu + "<br>"
                      + f"{cur_year} {month}: " + "%{customdata[2]:" + vfmt + "}" + pu + "<br>"
                      + f"Change in {mlabel}: " + "%{z:" + vfmt + "}" + chg_unit + "<extra></extra>",
    )
    fig.update_layout(
        title=f"{crop} — {IRRIG_LABELS[irrig]} — YoY change in {mlabel} "
              f"by {geo_word} ({cur_year} vs {base_year}, {month}, as of {date})",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
