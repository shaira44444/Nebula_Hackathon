"""
Door subsystem — honest, leakage-aware validation.

This does NOT touch door_predict.py's model choice or features; it reuses them
exactly (segment / features / the RandomForest config from DoorPipeline) and
only measures how well they generalise. Two things are validated:

  1. Segmentation  — do the time-gap splits recover the 110 ground-truth
     segments, and at what IoU (the timing half of the Door metric).

  2. Classification — Normal vs Abnormal-resistance, measured with
       (a) repeated stratified 5-fold CV   -> reproducible headline F1 + spread
       (b) blocked chronological CV         -> train-on-past / test-on-future,
                                               the split that can't leak time.
     Out-of-fold predictions from the chronological folds give one confusion
     matrix over all 110 segments (each predicted once, while held out).

Run:  python door_validate.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from door_predict import segment, features, parse_dt, NORMAL, ABNORMAL

TRAIN_CSV = "Train.csv"
ANSWER_CSV = "Train_Segments_Answer.csv"
LABELS = [NORMAL, ABNORMAL]


def new_model(seed):
    """Same config as DoorPipeline, fresh per fold so nothing leaks between folds."""
    return RandomForestClassifier(
        n_estimators=600, min_samples_leaf=2,
        class_weight="balanced", random_state=seed)


def interval_iou(a0, a1, b0, b1):
    """IoU of two time intervals given as pandas Timestamps."""
    lo = max(a0.value, b0.value)
    hi = min(a1.value, b1.value)
    inter = max(0, hi - lo)
    union = (max(a1.value, b1.value) - min(a0.value, b0.value))
    return inter / union if union else 0.0


def build_dataset():
    """Reproduce the pipeline's segmentation, attach ground-truth by overlap,
    and return features X, labels y, per-segment start times, and seg IoUs."""
    df = pd.read_csv(TRAIN_CSV)
    ans = pd.read_csv(ANSWER_CSV)
    ans = ans.copy()
    ans["s"] = ans["start_time"].map(parse_dt)
    ans["e"] = ans["end_time"].map(parse_dt)

    segs = segment(df)

    rows, labels, starts, ious = [], [], [], []
    matched_gt = set()
    for s, e, g in segs:
        hit = ans[(ans.s <= e) & (ans.e >= s)]
        if len(hit):
            gt = hit.iloc[0]
            labels.append(gt["status"])
            ious.append(interval_iou(s, e, gt["s"], gt["e"]))
            matched_gt.add(gt["segment_id"])
        else:
            labels.append(NORMAL)          # unmatched pred -> treated as Normal
            ious.append(0.0)
        rows.append(features(g))
        starts.append(s)

    X = pd.DataFrame(rows)
    y = np.array(labels)
    return X, y, np.array(starts), np.array(ious), len(segs), len(ans), len(matched_gt)


def repeated_stratified(X, y, seeds=range(15), k=5):
    """Repeated stratified k-fold. Reproducible headline macro-F1 + spread."""
    macro, per = [], {NORMAL: [], ABNORMAL: []}
    for seed in seeds:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        oof = np.empty(len(y), dtype=object)
        for tr, te in skf.split(X, y):
            m = new_model(seed).fit(X.iloc[tr].values, y[tr])
            oof[te] = m.predict(X.iloc[te].values)
        macro.append(f1_score(y, oof, labels=LABELS, average="macro"))
        f = f1_score(y, oof, labels=LABELS, average=None)
        per[NORMAL].append(f[0]); per[ABNORMAL].append(f[1])
    return {
        "macro_f1_mean": float(np.mean(macro)),
        "macro_f1_std": float(np.std(macro)),
        "normal_f1_mean": float(np.mean(per[NORMAL])),
        "abnormal_f1_mean": float(np.mean(per[ABNORMAL])),
        "seeds": len(list(seeds)), "k": k,
    }


def blocked_chronological(X, y, starts, k=5):
    """Blocked chronological CV: sort by time, split into k contiguous blocks,
    test each block on a model trained from the other blocks. No shuffling, so
    a test segment is never surrounded in time by its own training neighbours.
    OOF predictions (each segment predicted once, while held out) -> one
    confusion matrix over all segments."""
    order = np.argsort([t.value for t in starts])
    Xo, yo = X.iloc[order].reset_index(drop=True), y[order]
    folds = np.array_split(np.arange(len(yo)), k)
    oof = np.empty(len(yo), dtype=object)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(yo)), te)
        m = new_model(0).fit(Xo.iloc[tr].values, yo[tr])
        oof[te] = m.predict(Xo.iloc[te].values)
    cm = confusion_matrix(yo, oof, labels=LABELS)
    f = f1_score(yo, oof, labels=LABELS, average=None)
    return {
        "macro_f1": float(f1_score(yo, oof, labels=LABELS, average="macro")),
        "normal_f1": float(f[0]),
        "abnormal_f1": float(f[1]),
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
        "report": classification_report(yo, oof, labels=LABELS, digits=3),
        "k": k,
    }


def main():
    X, y, starts, ious, n_pred, n_gt, n_matched = build_dataset()

    seg = {
        "predicted_segments": n_pred,
        "ground_truth_segments": n_gt,
        "matched_1to1": n_matched,
        "mean_iou": float(np.mean(ious)),
        "min_iou": float(np.min(ious)),
    }

    strat = repeated_stratified(X, y)
    chrono = blocked_chronological(X, y, starts)

    print("=" * 62)
    print("SEGMENTATION VALIDATION")
    print("=" * 62)
    print(f"  ground-truth segments : {seg['ground_truth_segments']}")
    print(f"  segments recovered    : {seg['matched_1to1']}/{seg['ground_truth_segments']}")
    print(f"  predicted segments    : {seg['predicted_segments']}")
    print(f"  mean IoU              : {seg['mean_iou']:.4f}")
    print(f"  min  IoU              : {seg['min_iou']:.4f}")

    print("\n" + "=" * 62)
    print("CLASSIFICATION — repeated stratified 5-fold CV")
    print("=" * 62)
    print(f"  macro F1  : {strat['macro_f1_mean']:.3f}  (std {strat['macro_f1_std']:.3f}, "
          f"{strat['seeds']} seeds)")
    print(f"  Normal   F1: {strat['normal_f1_mean']:.3f}")
    print(f"  Abnormal F1: {strat['abnormal_f1_mean']:.3f}")

    print("\n" + "=" * 62)
    print("CLASSIFICATION — blocked chronological CV (no time leakage)")
    print("=" * 62)
    print(f"  macro F1  : {chrono['macro_f1']:.3f}")
    print(f"  Normal   F1: {chrono['normal_f1']:.3f}")
    print(f"  Abnormal F1: {chrono['abnormal_f1']:.3f}")
    print("\n  confusion matrix (rows = true, cols = pred)")
    print(f"                    pred:{LABELS[0]:<20} {LABELS[1]}")
    cm = chrono["confusion_matrix"]
    for i, lab in enumerate(LABELS):
        print(f"  true {lab:<20} {cm[i][0]:<24} {cm[i][1]}")
    print("\n" + chrono["report"])

    out = {"segmentation": seg, "stratified_cv": strat, "chronological_cv": chrono}
    with open("door_validation_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("Wrote door_validation_results.json")


if __name__ == "__main__":
    main()
