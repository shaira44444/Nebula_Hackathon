"""
Rail — can we lift Side I (14 examples, F1 ~0.54) and raise macro F1?

Every method is evaluated under the SAME repeated stratified 5-fold CV as
rail_validate.py. CRITICAL: any resampling happens INSIDE each fold, on the
training split only — the held-out fold is never resampled or seen — so the
numbers are leakage-free and comparable to the baseline.

Methods compared:
  baseline        - HGB, class_weight="balanced" (current pipeline)
  weight_boost    - HGB, manual sample weights, Side I up-weighted extra
  smote           - SMOTE oversample minorities in the train fold, then HGB(no weight)
  borderline      - BorderlineSMOTE (synthesise near the decision boundary)
  smote_enn       - SMOTE + Edited-NN cleaning
  two_stage       - stage 1: fault vs Normal; stage 2: Side I vs Side II on flagged

Run:  python rail_sideI_experiments.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN

CLASSES = ["Normal", "Side I", "Side II"]
FEATURES_CSV = "rail_features_v2.csv"


def hgb(class_weight="balanced"):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=2.0, class_weight=class_weight, random_state=0)


def load():
    df = pd.read_csv(FEATURES_CSV)
    y = df["label"].values
    X = df.drop(columns=[c for c in ("filename", "label") if c in df.columns]).fillna(0.0)
    return X.values, y


# ---- per-fold predict functions: fit on (Xtr,ytr), return preds for Xte ----

def m_baseline(Xtr, ytr, Xte):
    return hgb().fit(Xtr, ytr).predict(Xte)


def m_weight_boost(Xtr, ytr, Xte):
    # inverse-frequency weights, with an extra x2 on Side I
    n = len(ytr)
    w = np.ones(n)
    for c in CLASSES:
        mask = ytr == c
        if mask.sum():
            w[mask] = n / (len(CLASSES) * mask.sum())
    w[ytr == "Side I"] *= 2.0
    return hgb(class_weight=None).fit(Xtr, ytr, sample_weight=w).predict(Xte)


def _resample_fit(sampler, Xtr, ytr, Xte):
    Xr, yr = sampler.fit_resample(Xtr, ytr)
    return hgb(class_weight=None).fit(Xr, yr).predict(Xte)


def m_smote(Xtr, ytr, Xte):
    k = min(5, (pd.Series(ytr).value_counts().min() - 1))
    return _resample_fit(SMOTE(random_state=0, k_neighbors=max(1, k)), Xtr, ytr, Xte)


def m_borderline(Xtr, ytr, Xte):
    k = min(5, (pd.Series(ytr).value_counts().min() - 1))
    return _resample_fit(
        BorderlineSMOTE(random_state=0, k_neighbors=max(1, k),
                        m_neighbors=max(2, min(10, k * 2))), Xtr, ytr, Xte)


def m_smote_enn(Xtr, ytr, Xte):
    k = min(5, (pd.Series(ytr).value_counts().min() - 1))
    return _resample_fit(SMOTEENN(random_state=0,
                                  smote=SMOTE(random_state=0, k_neighbors=max(1, k))),
                         Xtr, ytr, Xte)


def m_two_stage(Xtr, ytr, Xte):
    # stage 1: Normal vs Fault
    y1 = np.where(ytr == "Normal", "Normal", "Fault")
    s1 = hgb().fit(Xtr, y1)
    # stage 2: Side I vs Side II, trained on the true faults only
    fault = ytr != "Normal"
    s2 = hgb().fit(Xtr[fault], ytr[fault])
    p1 = s1.predict(Xte)
    out = np.array(["Normal"] * len(Xte), dtype=object)
    flag = p1 == "Fault"
    if flag.any():
        out[flag] = s2.predict(Xte[flag])
    return out


METHODS = {
    "baseline": m_baseline,
    "weight_boost": m_weight_boost,
    "smote": m_smote,
    "borderline": m_borderline,
    "smote_enn": m_smote_enn,
    "two_stage": m_two_stage,
}


def evaluate(fn, X, y, seeds=range(3), k=5):
    macro, sideI, sideI_rec, per = [], [], [], {c: [] for c in CLASSES}
    for seed in seeds:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        oof = np.empty(len(y), dtype=object)
        for tr, te in skf.split(X, y):
            oof[te] = fn(X[tr], y[tr], X[te])
        macro.append(f1_score(y, oof, labels=CLASSES, average="macro"))
        f = f1_score(y, oof, labels=CLASSES, average=None)
        for i, c in enumerate(CLASSES):
            per[c].append(f[i])
        # Side I recall
        m = y == "Side I"
        sideI_rec.append((oof[m] == "Side I").mean())
    return {
        "macro_f1": (float(np.mean(macro)), float(np.std(macro))),
        "sideI_f1": (float(np.mean(per["Side I"])), float(np.std(per["Side I"]))),
        "sideI_recall": (float(np.mean(sideI_rec)), float(np.std(sideI_rec))),
        "normal_f1": float(np.mean(per["Normal"])),
        "sideII_f1": float(np.mean(per["Side II"])),
    }


def main():
    X, y = load()
    results = {name: evaluate(fn, X, y) for name, fn in METHODS.items()}

    print(f"{'method':<14}{'macro F1':<18}{'Side I F1':<18}{'Side I recall':<18}"
          f"{'Normal':<8}{'SideII':<8}")
    print("-" * 82)
    base = results["baseline"]["macro_f1"][0]
    for name, r in results.items():
        mf, ms = r["macro_f1"]; s1f, s1s = r["sideI_f1"]; rr, rs = r["sideI_recall"]
        delta = mf - base
        tag = f"  ({delta:+.3f})" if name != "baseline" else ""
        print(f"{name:<14}{mf:.3f}+/-{ms:.3f}    {s1f:.3f}+/-{s1s:.3f}    "
              f"{rr:.3f}+/-{rs:.3f}    {r['normal_f1']:.3f}   {r['sideII_f1']:.3f}{tag}")

    with open("rail_sideI_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nWrote rail_sideI_results.json")


if __name__ == "__main__":
    main()
