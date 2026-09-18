"""
Rail corrugation — honest, leakage-aware validation.

Task: classify each 1-second axle-box recording as Normal / Side I / Side II.
Scoring metric (PS3): macro F1 across the three classes.

This validates the SAME configuration the shipped pipeline uses. RailPipeline's
default is resample="borderline" (BorderlineSMOTE on training data + HGB with
class_weight=None), so this script's headline uses that same setting, and prints
the plain-baseline (class_weight="balanced", no resampling) alongside for
comparison. Set RESAMPLE=None below to make the baseline the headline instead.

Any resampling is applied INSIDE each fold, to the training split only — the
held-out fold is never resampled or seen — so the numbers are leakage-free.

It reads the verified feature table `rail_features_v2.csv` (272 rows: 101
features + filename + label), which was checked to reproduce from the raw
signals at ~1e-15 on the provided sample files and to be correctly aligned to
Train_Labels.csv, so no raw Train/ folder is needed to run this.

Run:  python rail_validate.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

FEATURES_CSV = "rail_features_v2.csv"
CLASSES = ["Normal", "Side I", "Side II"]
META = ["filename", "label"]
RESAMPLE = "borderline"     # matches RailPipeline default; None = plain baseline


def new_model(resample):
    """Same config as RailPipeline: class_weight balanced only when NOT resampling."""
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=2.0,
        class_weight=(None if resample else "balanced"), random_state=0)


def sampler_for(y, resample):
    """Same sampler RailPipeline builds; k clamped to the smallest class in this split."""
    if not resample:
        return None
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    k = max(1, min(5, pd.Series(y).value_counts().min() - 1))
    if resample == "smote":
        return SMOTE(random_state=0, k_neighbors=k)
    return BorderlineSMOTE(random_state=0, k_neighbors=k,
                           m_neighbors=max(2, min(10, k * 2)))


def _fit_predict(Xtr, ytr, Xte, resample):
    if resample:
        Xtr, ytr = sampler_for(ytr, resample).fit_resample(Xtr, ytr)   # train-only
    return new_model(resample).fit(Xtr, ytr).predict(Xte)


def load():
    df = pd.read_csv(FEATURES_CSV)
    y = df["label"].values
    X = df.drop(columns=[c for c in META if c in df.columns]).fillna(0.0)
    return X.values, y


def repeated_stratified(X, y, resample, seeds=range(8), k=5):
    """Repeated stratified k-fold. macro F1 + per-class F1, mean/std over seeds."""
    macro, per, rec = [], {c: [] for c in CLASSES}, []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        oof = np.empty(len(y), dtype=object)
        for tr, te in skf.split(X, y):
            oof[te] = _fit_predict(X[tr], y[tr], X[te], resample)
        macro.append(f1_score(y, oof, labels=CLASSES, average="macro"))
        f = f1_score(y, oof, labels=CLASSES, average=None)
        for i, c in enumerate(CLASSES):
            per[c].append(f[i])
        m = y == "Side I"
        rec.append((oof[m] == "Side I").mean())
    return {
        "macro_f1_mean": float(np.mean(macro)), "macro_f1_std": float(np.std(macro)),
        "macro_f1_min": float(np.min(macro)), "macro_f1_max": float(np.max(macro)),
        "per_class_f1_mean": {c: float(np.mean(per[c])) for c in CLASSES},
        "per_class_f1_std": {c: float(np.std(per[c])) for c in CLASSES},
        "sideI_recall_mean": float(np.mean(rec)),
        "seeds": len(list(seeds)), "k": k,
    }


def oof_confusion(X, y, resample, k=5, seed=0):
    """One stratified pass; each file predicted once while held out."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=object)
    for tr, te in skf.split(X, y):
        oof[te] = _fit_predict(X[tr], y[tr], X[te], resample)
    cm = confusion_matrix(y, oof, labels=CLASSES)
    return {
        "macro_f1": float(f1_score(y, oof, labels=CLASSES, average="macro")),
        "confusion_matrix": cm.tolist(), "labels": CLASSES,
        "report": classification_report(y, oof, labels=CLASSES, digits=3, zero_division=0),
    }


def main():
    X, y = load()
    counts = pd.Series(y).value_counts().to_dict()

    head = repeated_stratified(X, y, RESAMPLE)
    base = repeated_stratified(X, y, None)
    cm = oof_confusion(X, y, RESAMPLE)

    name = f"resample={RESAMPLE!r}" if RESAMPLE else "baseline (class_weight balanced)"
    print("=" * 64)
    print("RAIL CORRUGATION VALIDATION  —  shipped config:", name)
    print("=" * 64)
    print(f"  files: {len(y)}  | class counts: "
          + ", ".join(f"{c} {counts.get(c,0)}" for c in CLASSES))

    print(f"\n  repeated stratified 5-fold CV ({head['seeds']} seeds):")
    print(f"    macro F1 : {head['macro_f1_mean']:.3f}  "
          f"(std {head['macro_f1_std']:.3f}, range "
          f"{head['macro_f1_min']:.3f}-{head['macro_f1_max']:.3f})")
    for c in CLASSES:
        print(f"    {c:<8} F1: {head['per_class_f1_mean'][c]:.3f} "
              f"(std {head['per_class_f1_std'][c]:.3f})")
    print(f"    Side I recall: {head['sideI_recall_mean']:.3f}")

    print(f"\n  baseline for comparison (no resampling): "
          f"macro F1 {base['macro_f1_mean']:.3f} +/- {base['macro_f1_std']:.3f}, "
          f"Side I recall {base['sideI_recall_mean']:.3f}")

    print("\n  confusion matrix (out-of-fold, seed 0; rows=true, cols=pred)")
    print("            " + "".join(f"{c:>10}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:<10}" + "".join(f"{v:>10}" for v in cm["confusion_matrix"][i]))
    print("\n" + cm["report"])

    with open("rail_validation_results.json", "w") as fh:
        json.dump({"shipped_config": name, "class_counts": counts,
                   "stratified_cv": head, "baseline_cv": base,
                   "oof_confusion": cm}, fh, indent=2)
    print("Wrote rail_validation_results.json")


if __name__ == "__main__":
    main()
