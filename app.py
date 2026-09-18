"""
NEBULA X — Rail Condition Monitoring

Run:
    python -m streamlit run app.py
"""

import io
import os
import sys
import shutil
import tempfile

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from door_predict import DoorPipeline, parse_dt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAIL_DIR = os.path.join(BASE_DIR, "rail data")

# Allow Python to import rail_predict.py from rail data/
if RAIL_DIR not in sys.path:
    sys.path.append(RAIL_DIR)

from rail_predict import RailPipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NEBULA X — Rail Condition Monitoring",
    layout="wide"
)

ABN = "Abnormal resistance"


# ============================================================
# MODEL LOADERS
# ============================================================

@st.cache_resource
def load_door_model():
    return DoorPipeline().fit(
        os.path.join(BASE_DIR, "Train.csv"),
        os.path.join(BASE_DIR, "Train_Segments_Answer.csv")
    )


@st.cache_resource
def load_rail_model():

    feature_csv = os.path.join(
        RAIL_DIR,
        "analysis",
        "rail_features_v2.csv"
    )

    return RailPipeline().fit_feature_table(feature_csv)


# ============================================================
# HEADER
# ============================================================

st.title("🚆 NEBULA X — Rail Condition Monitoring")

st.caption(
    "AI-assisted fault detection across railway subsystems."
)


# ============================================================
# SUBSYSTEM SELECTOR
# ============================================================

subsystem = st.selectbox(
    "Subsystem",
    [
        "Door",
        "Rail Corrugation",
        "ACV (soon)",
        "SHM (soon)"
    ]
)


# ============================================================
# DOOR
# ============================================================

if subsystem == "Door":

    st.markdown("## 🚪 Door Motor Monitoring")

    st.write(
        "Detect abnormal resistance during train door "
        "opening and closing cycles."
    )

    st.markdown("### 1 · Upload")

    up = st.file_uploader(
        "Upload door sensor CSV",
        type="csv",
        key="door_upload"
    )

    if up is None:
        st.stop()

    raw = pd.read_csv(up)

    st.success(
        f"Loaded {len(raw):,} rows."
    )

    # --------------------------------------------------------
    # RUN MODEL
    # --------------------------------------------------------

    with st.spinner("Finding cycles and classifying..."):

        model = load_door_model()

        tmp = io.StringIO()

        raw.to_csv(
            tmp,
            index=False
        )

        tmp.seek(0)

        result = model.predict_proba(tmp)

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    n_ab = int(
        (result.prediction == ABN).sum()
    )

    n_ok = int(
        (result.prediction == "Normal").sum()
    )

    st.markdown("### 2 · Result")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Cycles found",
        len(result)
    )

    c2.metric(
        "Normal",
        n_ok
    )

    c3.metric(
        "Abnormal resistance",
        n_ab
    )

    # --------------------------------------------------------
    # TIMELINE
    # --------------------------------------------------------

    st.markdown("#### Timeline")

    starts = [
        parse_dt(s)
        for s in result.start_time
    ]

    colors = [
        "#c1440e" if p == ABN else "#2d7d46"
        for p in result.prediction
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=starts,
            y=result.confidence,
            mode="markers",
            marker=dict(
                size=12,
                color=colors
            ),
            text=result.prediction,
            hovertemplate=(
                "%{text}<br>"
                "risk %{y:.0%}"
                "<extra></extra>"
            )
        )
    )

    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="decision boundary"
    )

    fig.update_layout(
        height=340,
        yaxis_title="Abnormal-resistance risk",
        xaxis_title="Cycle start",
        showlegend=False,
        margin=dict(t=20)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.markdown("#### Predicted segments")

    show = result.copy()

    show["flag"] = show.prediction.map(
        lambda p: "⚠️" if p == ABN else "✅"
    )

    show = show.sort_values(
        "confidence",
        ascending=False
    )

    st.dataframe(
        show[
            [
                "flag",
                "start_time",
                "end_time",
                "prediction",
                "confidence"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.markdown("### 3 · Download")

    csv = result[
        [
            "start_time",
            "end_time",
            "prediction"
        ]
    ].to_csv(index=False)

    st.download_button(
        "⬇ Download door_predictions.csv",
        csv,
        file_name="door_predictions.csv",
        mime="text/csv",
        type="primary"
    )


# ============================================================
# RAIL CORRUGATION
# ============================================================

elif subsystem == "Rail Corrugation":

    st.markdown("## 🛤️ Rail Corrugation Monitoring")

    st.write(
        "Analyse axle-box vibration recordings and classify "
        "each recording as **Normal**, **Side I**, or **Side II**."
    )

    st.markdown("### 1 · Upload")

    uploaded_files = st.file_uploader(
        "Upload one or more Rail Corrugation CSV files",
        type="csv",
        accept_multiple_files=True,
        key="rail_upload"
    )

    if not uploaded_files:

        st.info(
            "Upload Rail Corrugation CSV recordings to begin."
        )

        st.stop()

    st.success(
        f"{len(uploaded_files)} recording(s) uploaded."
    )

    # --------------------------------------------------------
    # ANALYSE BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyse Rail Recordings",
        type="primary"
    ):

        with st.spinner(
            "Extracting vibration features and classifying..."
        ):

            temp_dir = tempfile.mkdtemp()

            try:

                for uploaded_file in uploaded_files:

                    path = os.path.join(
                        temp_dir,
                        uploaded_file.name
                    )

                    with open(path, "wb") as f:

                        f.write(
                            uploaded_file.getbuffer()
                        )

                model = load_rail_model()

                result = model.predict_proba(
                    temp_dir
                )

                st.session_state[
                    "rail_result"
                ] = result

            finally:

                shutil.rmtree(
                    temp_dir,
                    ignore_errors=True
                )

    # --------------------------------------------------------
    # WAIT UNTIL ANALYSIS
    # --------------------------------------------------------

    if "rail_result" not in st.session_state:

        st.stop()

    result = st.session_state["rail_result"]

    st.success("Rail analysis complete.")

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.markdown("### 2 · Result")

    normal_count = int(
        (result["prediction"] == "Normal").sum()
    )

    side1_count = int(
        (result["prediction"] == "Side I").sum()
    )

    side2_count = int(
        (result["prediction"] == "Side II").sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Recordings analysed",
        len(result)
    )

    c2.metric(
        "Normal",
        normal_count
    )

    c3.metric(
        "Side I",
        side1_count
    )

    c4.metric(
        "Side II",
        side2_count
    )

    # --------------------------------------------------------
    # CONDITION DISTRIBUTION
    # --------------------------------------------------------

    st.markdown("#### Classification Summary")

    counts = (
        result["prediction"]
        .value_counts()
        .reindex(
            [
                "Normal",
                "Side I",
                "Side II"
            ],
            fill_value=0
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts.values,
            text=counts.values,
            textposition="auto"
        )
    )

    fig.update_layout(
        height=300,
        xaxis_title="Rail condition",
        yaxis_title="Recordings",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    st.markdown("#### Model Confidence")

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=result["file_id"],
            y=result["confidence"],
            mode="markers",
            marker=dict(size=12),
            text=result["prediction"],
            hovertemplate=(
                "%{x}<br>"
                "%{text}<br>"
                "confidence %{y:.0%}"
                "<extra></extra>"
            )
        )
    )

    fig2.update_layout(
        height=320,
        yaxis_title="Model confidence",
        xaxis_title="Recording",
        yaxis=dict(
            range=[0, 1]
        )
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.markdown("#### Predicted Rail Conditions")

    show = result.copy()

    show["flag"] = show["prediction"].map(
        lambda x: "✅" if x == "Normal" else "⚠️"
    )

    st.dataframe(
        show[
            [
                "flag",
                "file_id",
                "prediction",
                "confidence"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # WHY?
    # --------------------------------------------------------

    st.markdown("### Why was it flagged?")

    selected_file = st.selectbox(
        "Choose a recording",
        result["file_id"].tolist()
    )

    selected_row = result[
        result["file_id"] == selected_file
    ].iloc[0]

    selected_prediction = selected_row["prediction"]
    selected_confidence = selected_row["confidence"]

    if selected_prediction == "Normal":

        st.success(
            f"**{selected_file}** was classified as Normal "
            f"with {selected_confidence:.0%} model confidence. "
            "The vibration pattern did not show strong evidence "
            "of Side I or Side II corrugation."
        )

    elif selected_prediction == "Side I":

        st.warning(
            f"**{selected_file}** was classified as Side I "
            f"with {selected_confidence:.0%} model confidence. "
            "The model detected a stronger corrugation-related "
            "vibration signature on Side I axle-box sensors."
        )

    elif selected_prediction == "Side II":

        st.warning(
            f"**{selected_file}** was classified as Side II "
            f"with {selected_confidence:.0%} model confidence. "
            "The model detected a stronger corrugation-related "
            "vibration signature on Side II axle-box sensors."
        )

    # --------------------------------------------------------
    # MAINTENANCE RECOMMENDATION
    # --------------------------------------------------------

    st.markdown("### Maintenance Recommendation")

    if selected_prediction == "Normal":

        st.success(
            "✅ **Low priority** — Continue routine monitoring. "
            "No corrugation-related maintenance action is "
            "currently indicated."
        )

    elif selected_prediction == "Side I":

        st.error(
            "⚠️ **Inspection recommended** — Inspect the "
            "Side I rail for corrugation and verify the affected "
            "track section. If confirmed, assess whether "
            "rail grinding or milling is required."
        )

    elif selected_prediction == "Side II":

        st.error(
            "⚠️ **Inspection recommended** — Inspect the "
            "Side II rail for corrugation and verify the affected "
            "track section. If confirmed, assess whether "
            "rail grinding or milling is required."
        )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.markdown("### 3 · Download")

    csv = result[
        [
            "file_id",
            "prediction"
        ]
    ].to_csv(index=False)

    st.download_button(
        "⬇ Download rail_predictions.csv",
        csv,
        file_name="rail_predictions.csv",
        mime="text/csv",
        type="primary"
    )

    st.caption(
        "Submission file contains only file_id and prediction."
    )


# ============================================================
# OTHER SUBSYSTEMS
# ============================================================

else:

    st.info(
        "This subsystem is still being developed."
    )