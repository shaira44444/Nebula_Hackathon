# NEBULA X — Door Subsystem

Segments a continuous door-controller stream into open/close cycles and
classifies each as **Normal** or **Abnormal resistance**.

## Results (measured — see `Validation` below and `door_validate.py`)

| Stage | Metric | Value |
|---|---|---|
| Segmentation | segments recovered | **110 / 110** |
| Segmentation | mean IoU | **1.000** |
| Classification | macro F1 (repeated stratified 5-fold, 15 seeds) | **0.988** (std 0.000) |
| Classification | macro F1 (blocked chronological CV, no time leakage) | **0.988** |
| Classification | Normal F1 / Abnormal F1 | **0.994 / 0.983** |

Every number above is produced by `python door_validate.py` — nothing here is
hand-asserted. On the 110 labelled training segments the model makes a single
error (one Abnormal-resistance cycle predicted Normal); see the confusion
matrix under `Validation`.

## Run the app
```
pip install -r requirements.txt
streamlit run app.py
```
Select Door → drop in a CSV → download `door_predictions.csv`.

## Generate predictions from the command line
```
python run_predict.py Test.csv
```

## Reproduce the validation
```
python door_validate.py        # prints the table above + writes door_validation_results.json
```

## Method
**Segmentation** — cycles are separated by multi-second gaps of no motion.
A split on time gaps > 1s recovers every boundary exactly (110/110, IoU 1.000
against `Train_Segments_Answer.csv`). The controller flags (opening/closing)
are noisier and were deliberately not used.

**Features** — three physical families:
- amplitude (current, voltage, back-EMF integrals) — effort rises with resistance
- resistance (current per unit motion, current per back-EMF) — resistance measured directly
- frequency (spectral centroid, HF energy) — catches judder / intermittent binding

**Model** — RandomForest, class-weight balanced for the 80/30 imbalance.

## Validation

The Door metric is **IoU-weighted F1** (timing × label). We validate the two
halves separately, then note why they combine cleanly here.

**Segmentation.** Running `segment()` on `Train.csv` yields 110 segments, each
matching one of the 110 ground-truth rows in `Train_Segments_Answer.csv` by
time overlap, at IoU 1.000 (the gap between cycles is unambiguous). Because
timing is exact on the training set, the IoU weight is ~1 and the IoU-weighted
F1 reduces to plain classification F1 on these segments.

**Classification — two independent splits, so the number can't be a split
artefact:**

1. *Repeated stratified 5-fold CV, 15 seeds* — the reproducible headline.
   Out-of-fold macro F1 = **0.988**, std **0.000** across seeds (the model is
   stable enough that every seed lands on the same single error).
2. *Blocked chronological 5-fold CV* — segments sorted by time and split into
   contiguous blocks, each block tested by a model trained only on the other
   blocks. This is the split that cannot leak time (no test cycle is surrounded
   in time by its own training neighbours) and is how the model would run in
   deployment (train on past, predict future). Macro F1 = **0.988**.

Both splits agree, which is the point: the score is not an artefact of a
favourable random split.

**Confusion matrix** (out-of-fold, all 110 segments, chronological CV):

| true ↓ / pred → | Normal | Abnormal resistance |
|---|---|---|
| **Normal** | 80 | 0 |
| **Abnormal resistance** | 1 | 29 |

Per class: Normal — precision 0.988, recall 1.000, F1 0.994; Abnormal — precision
1.000, recall 0.967, F1 0.983. The one error is a false negative (a missed
Abnormal cycle); the model raises no false alarms on Normal cycles.

**No feature leakage.** Features are per-segment physical statistics (current,
voltage, back-EMF, position); no controller flag, label, or cross-segment
information enters the feature vector. Each CV fold fits a fresh model on the
training folds only.

## Files
- `door_predict.py` — segmentation + features + model
- `door_validate.py` — reproduces every metric above (run it to regenerate `door_validation_results.json`)
- `app.py` — the upload/predict/download UI
- `run_predict.py` — CLI to produce the submission CSV
