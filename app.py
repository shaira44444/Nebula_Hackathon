"""
NEBULA X — Condition Monitoring app.

Compulsory deliverable. A non-technical user selects a subsystem,
drops in a CSV, sees the predictions, downloads the result.

Run:  streamlit run app.py
"""
import io
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from door_predict import DoorPipeline, parse_dt

st.set_page_config(page_title="NEBULA X — Condition Monitoring", layout="wide")

ABN = "Abnormal resistance"


@st.cache_resource
def load_door_model():
    return DoorPipeline().fit("Train.csv", "Train_Segments_Answer.csv")


st.title("Rail Condition Monitoring")
st.caption("Upload a subsystem data stream. The app finds each cycle, "
           "classifies it, and returns a downloadable report.")

subsystem = st.selectbox("Subsystem", ["Door", "Rail Corrugation (soon)",
                                       "ACV (soon)", "SHM (soon)"])

if subsystem != "Door":
    st.info("This subsystem isn't wired up yet. Select **Door**.")
    st.stop()

st.markdown("### 1 · Upload")
up = st.file_uploader("Drag a door data CSV here (a continuous stream "
                      "like Test.csv)", type="csv")

if up is None:
    st.stop()

# ---- run the pipeline
raw = pd.read_csv(up)
st.success(f"Loaded {len(raw):,} rows.")

with st.spinner("Finding cycles and classifying..."):
    model = load_door_model()
    tmp = io.StringIO()
    raw.to_csv(tmp, index=False)
    tmp.seek(0)
    result = model.predict_proba(tmp)

n_ab = int((result.prediction == ABN).sum())
n_ok = int((result.prediction == "Normal").sum())

st.markdown("### 2 · Result")
c1, c2, c3 = st.columns(3)
c1.metric("Cycles found", len(result))
c2.metric("Normal", n_ok)
c3.metric("Abnormal resistance", n_ab)

# ---- timeline
st.markdown("#### Timeline")
starts = [parse_dt(s) for s in result.start_time]
colors = ["#c1440e" if p == ABN else "#2d7d46" for p in result.prediction]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=starts, y=result.confidence, mode="markers",
    marker=dict(size=12, color=colors),
    text=result.prediction,
    hovertemplate="%{text}<br>risk %{y:.0%}<extra></extra>"))
fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
              annotation_text="decision boundary")
fig.update_layout(height=340, yaxis_title="abnormal-resistance risk",
                  xaxis_title="cycle start", showlegend=False,
                  margin=dict(t=20))
st.plotly_chart(fig, use_container_width=True)

# ---- table (abnormal first)
st.markdown("#### Predicted segments")
show = result.copy()
show["flag"] = show.prediction.map(lambda p: "⚠️" if p == ABN else "")
show = show.sort_values("confidence", ascending=False)
st.dataframe(show[["flag", "start_time", "end_time", "prediction",
                   "confidence"]],
             use_container_width=True, hide_index=True)

# ---- download (exact submission schema, no confidence col)
st.markdown("### 3 · Download")
csv = result[["start_time", "end_time", "prediction"]].to_csv(index=False)
st.download_button("⬇ Download door_predictions.csv", csv,
                   file_name="door_predictions.csv", mime="text/csv",
                   type="primary")
st.caption("This is the exact file to place in predictions.zip.")
