"""
Website Phishing Risk Analytics Dashboard
=========================================
Dataset : UCI "Website Phishing" (id=379)
          https://archive.ics.uci.edu/dataset/379/website+phishing
Run     : streamlit run app.py

Fact -> Insight -> Decision -> Action
  * Fact     : how many websites are phishing / suspicious / legitimate
  * Insight  : which website signals separate phishing from legitimate sites
  * Decision : which signals and risk-score rule a security team should use
  * Action   : concrete screening recommendations

Note: this is DESCRIPTIVE analytics (no prediction model). All numbers are
computed live from the dataset, so they always match the data you load.
"""

import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Website Phishing Risk Analytics",
    page_icon="🛡️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/379/website+phishing"

CLASS_NAMES = {-1: "Phishing", 0: "Suspicious", 1: "Legitimate"}
CLASS_ORDER = ["Phishing", "Suspicious", "Legitimate"]
CLASS_COLORS = {
    "Phishing": "#d62728",
    "Suspicious": "#ff9f1c",
    "Legitimate": "#2ca02c",
}

# Signal values are coded -1 / 0 / 1. Their meaning is NOT assumed: for every
# signal the app learns which value is the riskiest directly from the data.
VALUES = [-1, 0, 1]


def vlabel(v) -> str:
    return f"Value {v}"


VALUE_ORDER = [vlabel(v) for v in VALUES]

FRIENDLY = {
    "SFH": "Server form handler",
    "popUpWidnow": "Pop-up window",
    "popUpWindow": "Pop-up window",
    "SSLfinal_State": "SSL certificate state",
    "Request_URL": "External object requests",
    "URL_of_Anchor": "URL of anchor tags",
    "web_traffic": "Website traffic",
    "URL_Length": "URL length",
    "age_of_domain": "Domain age",
    "having_IP_Address": "IP address in URL",
}

LOCAL_CANDIDATES = [
    "website_phishing.csv",
    "PhishingData.csv",
    "phishing.csv",
    "data/website_phishing.csv",
    "data/PhishingData.csv",
    "PhishingData.arff",
    "data/PhishingData.arff",
]
TARGET_CANDIDATES = ["result", "class", "label", "target"]


# --------------------------------------------------------------------------
# Data loading and preparation
# --------------------------------------------------------------------------
def read_table(name: str, raw: bytes) -> pd.DataFrame:
    """Read a CSV or ARFF file from raw bytes."""
    if name.lower().endswith(".arff"):
        from scipy.io import arff

        data, _ = arff.loadarff(io.StringIO(raw.decode("utf-8", errors="ignore")))
        df = pd.DataFrame(data)
        for col in df.columns:
            df[col] = df[col].apply(
                lambda v: v.decode("utf-8") if isinstance(v, bytes) else v
            )
        return df
    return pd.read_csv(io.BytesIO(raw))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names, find the target and add a readable Class."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    target = next(
        (c for c in df.columns if c.lower() in TARGET_CANDIDATES), df.columns[-1]
    )
    df = df.rename(columns={target: "Result"})

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    feats = [c for c in df.columns if c != "Result"]
    df = df.dropna(subset=["Result"] + feats).copy()
    df["Result"] = df["Result"].astype(int)
    df["Class"] = df["Result"].map(CLASS_NAMES).fillna(df["Result"].astype(str))
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Loading dataset...")
def load_data(upload_name, upload_bytes):
    """Order: uploaded file -> local file -> download from UCI via ucimlrepo."""
    if upload_bytes is not None:
        return prepare(read_table(upload_name, upload_bytes)), f"Uploaded file: {upload_name}"

    for path in LOCAL_CANDIDATES:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return prepare(read_table(path, f.read())), f"Local file: {path}"

    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=379)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    return prepare(df), "UCI repository (ucimlrepo, id=379)"


def feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in ("Result", "Class")]


def fname(col: str) -> str:
    return FRIENDLY.get(col, col)


# --------------------------------------------------------------------------
# Analytics helpers
# --------------------------------------------------------------------------
def signal_table(df: pd.DataFrame, feats, min_sites: int = 20) -> pd.DataFrame:
    """For each signal, find its riskiest and safest value from the data."""
    rows = []
    for f in feats:
        stats = {}
        for v in VALUES:
            sub = df[df[f] == v]
            if len(sub) > 0:
                stats[v] = ((sub["Result"] == -1).mean() * 100, len(sub))
        if not stats:
            continue
        eligible = {v: s for v, s in stats.items() if s[1] >= min_sites} or stats
        risky = max(eligible, key=lambda v: eligible[v][0])
        safest = min(eligible, key=lambda v: eligible[v][0])
        rows.append(
            {
                "Feature": fname(f),
                "Column": f,
                "Risky value": int(risky),
                "Phishing rate at risky value (%)": eligible[risky][0],
                "Sites with risky value (%)": eligible[risky][1] / len(df) * 100,
                "Safest value": int(safest),
                "Phishing rate at safest value (%)": eligible[safest][0],
                "Gap (pts)": eligible[risky][0] - eligible[safest][0],
            }
        )
    out = pd.DataFrame(rows).sort_values("Gap (pts)", ascending=False)
    return out.reset_index(drop=True)


def risky_map_from(sig: pd.DataFrame) -> dict:
    return dict(zip(sig["Column"], sig["Risky value"]))


def risk_scores(df: pd.DataFrame, feats, risky_map) -> pd.Series:
    """Risk score = number of signals where the site has that signal's risky value."""
    score = pd.Series(0, index=df.index)
    for f in feats:
        score += (df[f] == risky_map[f]).astype(int)
    return score


def risk_table(df: pd.DataFrame, feats, risky_map) -> pd.DataFrame:
    score = risk_scores(df, feats, risky_map)
    cap = min(5, len(feats))
    bucket = score.clip(upper=cap)
    rows = []
    for k in range(0, cap + 1):
        mask = bucket == k
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "Risky signals": f"{k}+" if k == cap else str(k),
                "Sites": int(mask.sum()),
                "Phishing rate (%)": (df.loc[mask, "Result"] == -1).mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def screening_rule(df: pd.DataFrame, feats, risky_map, target=90.0, min_sites=20):
    """Smallest 'k or more risky signals' rule with phishing rate >= target."""
    score = risk_scores(df, feats, risky_map)
    for k in range(1, len(feats) + 1):
        mask = score >= k
        if mask.sum() >= min_sites:
            rate = (df.loc[mask, "Result"] == -1).mean() * 100
            if rate >= target:
                return k, int(mask.sum()), rate
    return None


def build_insights(df, feats, sig, rule):
    n = len(df)

    def share(c):
        return (df["Class"] == c).mean() * 100

    lines = [
        f"{share('Phishing'):.1f}% of the {n:,} websites are phishing and "
        f"{share('Suspicious'):.1f}% are suspicious; only "
        f"{share('Legitimate'):.1f}% are clearly legitimate."
    ]
    for _, r in sig.head(3).iterrows():
        lines.append(
            f"{r['Feature']} is one of the strongest risk signals: "
            f"{r['Phishing rate at risky value (%)']:.0f}% of sites with value "
            f"{r['Risky value']} are phishing, versus "
            f"{r['Phishing rate at safest value (%)']:.0f}% of sites with value "
            f"{r['Safest value']}."
        )
    if rule:
        k, cnt, rate = rule
        lines.append(
            f"A simple rule works well: websites with {k} or more risky signals "
            f"({cnt:,} sites) are {rate:.0f}% phishing."
        )
    else:
        lines.append(
            "No simple 'count of risky signals' rule reaches 90% phishing, so "
            "combine signals with manual review."
        )
    sus = share("Suspicious")
    if sus > 0:
        lines.append(
            f"Suspicious sites ({sus:.1f}% of the data) carry mixed signals and "
            f"should go to a manual review queue."
        )
    return lines


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_overview(df, feats, sig, rule):
    st.title("🛡️ Website Phishing Risk Analytics")
    st.caption(
        "Executive overview: how big is the phishing problem and which signals matter most?"
    )

    n = len(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Websites analysed", f"{n:,}")
    c2.metric("Phishing rate", f"{(df['Class'] == 'Phishing').mean() * 100:.1f}%")
    c3.metric("Suspicious share", f"{(df['Class'] == 'Suspicious').mean() * 100:.1f}%")
    c4.metric("Legitimate share", f"{(df['Class'] == 'Legitimate').mean() * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        counts = df["Class"].value_counts().rename_axis("Class").reset_index(name="Sites")
        fig = px.pie(
            counts,
            names="Class",
            values="Sites",
            hole=0.55,
            color="Class",
            color_discrete_map=CLASS_COLORS,
            category_orders={"Class": CLASS_ORDER},
            title="Website class distribution",
        )
        st.plotly_chart(fig)
    with right:
        top = sig.sort_values("Gap (pts)")
        fig = px.bar(
            top,
            x="Gap (pts)",
            y="Feature",
            orientation="h",
            color="Gap (pts)",
            color_continuous_scale="Reds",
            title="Top risk signals (phishing-rate gap, riskiest vs safest value)",
            labels={"Gap (pts)": "Gap in phishing rate (percentage points)"},
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig)

    st.subheader("Key insights")
    for line in build_insights(df, feats, sig, rule):
        st.markdown(f"- {line}")


def page_drivers(df, feats, sig):
    st.title("🔍 Risk Drivers")
    st.caption("Why do websites end up phishing? Compare each signal across classes.")

    # Heatmap of phishing rate by feature and signal value
    data = {}
    for v in VALUES:
        data[vlabel(v)] = [
            (df.loc[df[f] == v, "Result"] == -1).mean() * 100 if (df[f] == v).any() else np.nan
            for f in feats
        ]
    mat = pd.DataFrame(data, index=[fname(f) for f in feats])
    fig = px.imshow(
        mat,
        text_auto=".0f",
        color_continuous_scale="Reds",
        aspect="auto",
        title="Phishing rate (%) for each signal value",
        labels={"color": "Phishing %"},
    )
    st.plotly_chart(fig)
    st.caption(
        "Read across a row: the darkest cell is the riskiest value for that signal. "
        "The bigger the difference between cells, the stronger the warning sign."
    )

    st.subheader("Explore one signal")
    choice = st.selectbox("Choose a signal", options=feats, format_func=fname)
    g = df.groupby([choice, "Class"]).size().reset_index(name="Sites")
    g["Share (%)"] = g["Sites"] / g.groupby(choice)["Sites"].transform("sum") * 100
    g["Signal value"] = g[choice].apply(lambda v: vlabel(int(v)))
    fig = px.bar(
        g,
        x="Signal value",
        y="Share (%)",
        color="Class",
        color_discrete_map=CLASS_COLORS,
        category_orders={"Class": CLASS_ORDER, "Signal value": VALUE_ORDER},
        text_auto=".0f",
        title=f"Class mix for: {fname(choice)}",
    )
    st.plotly_chart(fig)

    st.subheader("Signal ranking")
    show = sig.drop(columns=["Column"]).round(1)
    st.dataframe(show, hide_index=True)


def page_actions(df, feats, sig, rule):
    st.title("✅ Risk & Recommended Actions")
    st.caption("From insight to decision: what should a security team do?")

    risky_map = risky_map_from(sig)
    rt = risk_table(df, feats, risky_map)
    fig = px.bar(
        rt,
        x="Risky signals",
        y="Phishing rate (%)",
        text=rt["Phishing rate (%)"].round(0).astype(int).astype(str) + "%",
        hover_data=["Sites"],
        color="Phishing rate (%)",
        color_continuous_scale="Reds",
        title="Phishing rate by number of risky signals on a website",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig)

    # Risk / opportunity / action table
    st.subheader("Risks, opportunities and actions")
    rows = []
    if rule:
        k, cnt, rate = rule
        rows.append(
            {
                "Type": "Risk",
                "Finding": f"{cnt:,} sites have {k}+ risky signals and {rate:.0f}% are phishing",
                "Recommended action": f"Auto-flag or block websites with {k} or more risky signals",
            }
        )
    for _, r in sig.head(3).iterrows():
        rows.append(
            {
                "Type": "Risk",
                "Finding": f"{r['Feature']} = {r['Risky value']}: "
                f"{r['Phishing rate at risky value (%)']:.0f}% phishing, covers "
                f"{r['Sites with risky value (%)']:.0f}% of sites",
                "Recommended action": f"Add '{r['Feature']}' to the first-line screening checklist",
            }
        )
    zero = risk_scores(df, feats, risky_map) == 0
    if zero.sum() > 0:
        z_rate = (df.loc[zero, "Result"] == -1).mean() * 100
        if z_rate <= 10:
            rows.append(
                {
                    "Type": "Opportunity",
                    "Finding": f"{int(zero.sum()):,} sites have no risky signal "
                    f"(phishing rate {z_rate:.0f}%)",
                    "Recommended action": "Fast-track these sites to reduce analyst workload",
                }
            )
    sus = (df["Class"] == "Suspicious").mean() * 100
    if sus > 0:
        rows.append(
            {
                "Type": "Risk",
                "Finding": f"Suspicious sites are {sus:.1f}% of the data with mixed signals",
                "Recommended action": "Send to a manual review / sandbox queue",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    # Screening simulator
    st.subheader("Website screening simulator")
    st.caption("Pick values for the top 3 risk signals to see how similar sites behave.")
    cols = st.columns(3)
    mask = pd.Series(True, index=df.index)
    for col, (_, r) in zip(cols, sig.head(3).iterrows()):
        risky = str(r["Risky value"])

        def show_option(o, risky=risky):
            return o + (" (risky)" if o == risky else "")

        pick = col.selectbox(
            r["Feature"], ["Any", "-1", "0", "1"], format_func=show_option,
            key=f"sim_{r['Column']}",
        )
        if pick != "Any":
            mask &= df[r["Column"]] == int(pick)
    matched = df[mask]
    if len(matched) == 0:
        st.info("No websites in the dataset match this combination.")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Matching websites", f"{len(matched):,}")
        m2.metric(
            "Phishing rate among them",
            f"{(matched['Result'] == -1).mean() * 100:.1f}%",
        )

    # Download
    report = "# Website Phishing Risk Analytics - Key Insights\n\n"
    report += "\n".join(f"- {line}" for line in build_insights(df, feats, sig, rule))
    report += f"\n\nDataset: {DATASET_PAGE}\n"
    st.download_button(
        "⬇️ Download insights summary (.md)",
        data=report,
        file_name="insights_summary.md",
        mime="text/markdown",
    )


def page_data(df, feats, source):
    st.title("🗂️ Data Overview")
    st.caption(f"Source: {source}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Signals (features)", len(feats))
    c3.metric("Missing values", int(df[feats + ["Result"]].isna().sum().sum()))
    c4.metric("Duplicate rows", int(df[feats + ["Result"]].duplicated().sum()))

    st.subheader("Preview")
    st.dataframe(df.head(20), hide_index=True)

    st.subheader("Class counts")
    counts = df["Class"].value_counts().rename_axis("Class").reset_index(name="Sites")
    counts["Share (%)"] = (counts["Sites"] / len(df) * 100).round(1)
    st.dataframe(counts, hide_index=True)

    st.subheader("Feature dictionary")
    st.dataframe(
        pd.DataFrame({"Column": feats, "Readable name": [fname(f) for f in feats]}),
        hide_index=True,
    )
    st.caption(
        "Signal values are coded -1 / 0 / 1. The dashboard does not assume what each "
        "value means; it works out the riskiest value of every signal from the data."
    )
    st.markdown(f"Dataset page: [{DATASET_PAGE}]({DATASET_PAGE}) (license: CC BY 4.0)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    st.sidebar.title("🛡️ Phishing Analytics")
    upload = st.sidebar.file_uploader(
        "Optional: upload dataset (CSV or ARFF)", type=["csv", "arff"]
    )

    try:
        df, source = load_data(
            upload.name if upload else None,
            upload.getvalue() if upload else None,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Could not load the dataset automatically. Download it from "
            f"{DATASET_PAGE}, save it as 'website_phishing.csv' next to app.py "
            "(or upload it in the sidebar)."
        )
        st.exception(exc)
        st.stop()

    feats = feature_cols(df)
    sig = signal_table(df, feats)
    rule = screening_rule(df, feats, risky_map_from(sig))

    page = st.sidebar.radio(
        "Go to",
        ["Executive Overview", "Risk Drivers", "Risk & Actions", "Data Overview"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Data: {source}")

    if page == "Executive Overview":
        page_overview(df, feats, sig, rule)
    elif page == "Risk Drivers":
        page_drivers(df, feats, sig)
    elif page == "Risk & Actions":
        page_actions(df, feats, sig, rule)
    else:
        page_data(df, feats, source)


main()
