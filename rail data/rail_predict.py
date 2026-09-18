"""
Rail corrugation — classify a 1-second axle-box recording as Normal / Side I / Side II.

Physics: corrugation is periodic wear, so it excites vibration at a characteristic
frequency that scales with train speed. We remove the speed dependence by ORDER
TRACKING (resampling each channel into the angular domain using the toothed-wheel
tacho), then describe each rail side with order-spectrum + envelope + time features.
Side faults are localised, so we aggregate PER CAR (max / std across cars), not by a
single per-side mean that would dilute a fault on a few cars.

Layout: col 0 = tacho (0/1 toothed-wheel toggle, 90 teeth/rev); cols 1..128 = car(0..7)
x position(1..8) x [vibration, shock]. Positions 1,3,5,7 -> Side I; 2,4,6,8 -> Side II.

    from rail_predict import RailPipeline
    p = RailPipeline().fit(TRAIN_DIR, "Train_Labels.csv")
    p.predict(TEST_DIR)          # -> rail_predictions.csv schema (file_id, prediction)
"""
import os, glob, numpy as np, pandas as pd
from scipy import signal, stats

FS = 10_000.0          # sampling rate (Hz)
TEETH = 90             # teeth on the speed wheel
SPR = 128              # samples per revolution after order resampling
MAXORD = 60            # highest order we keep
CLASSES = ["Normal", "Side I", "Side II"]


def vib_col(car, pos):   return 1 + car*16 + (pos-1)*2      # vibration column
def shock_col(car, pos): return 2 + car*16 + (pos-1)*2      # shock column
POS = {"I": [1, 3, 5, 7], "II": [2, 4, 6, 8]}


def _edges(tacho):
    s = (np.asarray(tacho) > 0.5).astype(int)
    return np.flatnonzero(np.diff(s) != 0) + 1    # sample indices of 0/1 transitions


def order_resample(x, edge_idx):
    """Resample x into the angular domain using tacho edges as angle markers.
    Each edge = 1/(2*TEETH) revolution. Returns a signal at SPR samples/rev."""
    if len(edge_idx) < 4:
        return None
    rev_at_edge = np.arange(len(edge_idx)) / (2.0 * TEETH)   # cumulative revolutions
    n_rev = rev_at_edge[-1]
    if n_rev < 0.5:
        return None
    grid = np.arange(0, n_rev, 1.0 / SPR)                    # uniform angle grid
    samp_at_rev = np.interp(grid, rev_at_edge, edge_idx)     # -> fractional sample index
    xi = np.interp(samp_at_rev, np.arange(len(x)), x)        # signal on the angle grid
    return xi - xi.mean()


def order_spectrum(xi):
    """Amplitude spectrum in orders (cycles per revolution)."""
    n = len(xi)
    amp = np.abs(np.fft.rfft(xi * np.hanning(n))) / n
    orders = np.fft.rfftfreq(n, d=1.0 / SPR)                 # in orders
    return orders, amp


def _time_feats(x):
    x = np.asarray(x, float); rms = np.sqrt(np.mean(x**2))
    return {"rms": rms,
            "kurt": float(stats.kurtosis(x)),
            "crest": float(np.max(np.abs(x)) / (rms + 1e-9)),
            "p2p": float(np.ptp(x))}


def _spec_feats(orders, amp):
    tot = amp.sum() + 1e-12
    bands = [(1, 5), (5, 12), (12, 25), (25, 60)]            # order bands
    d = {f"ord_{lo}_{hi}": amp[(orders >= lo) & (orders < hi)].sum() / tot for lo, hi in bands}
    d["ord_centroid"] = float((orders * amp).sum() / tot)
    d["ord_peak"] = float(orders[np.argmax(amp)])
    pn = amp / tot
    d["ord_entropy"] = float(-(pn * np.log(pn + 1e-12)).sum())
    d["ord_peakamp"] = float(amp.max())
    return d


def _plain_spectrum(v):
    """Fallback for low-speed files: FFT in Hz mapped onto a 0..MAXORD pseudo-order axis,
    so the same feature keys are produced for every file (no speed-dependent gaps)."""
    x = np.asarray(v, float) - np.mean(v)
    n = len(x)
    amp = np.abs(np.fft.rfft(x * np.hanning(n))) / n
    freq = np.fft.rfftfreq(n, d=1.0 / FS)
    axis = freq / (FS / 2.0) * MAXORD          # pseudo-orders 0..MAXORD
    return axis, amp, x


def _spectrum(v, edge):
    """Order spectrum when rotation allows, else a plain-FFT fallback. Same keys either way."""
    xi = order_resample(v, edge)
    if xi is not None and len(xi) > 8:
        orders, amp = order_spectrum(xi)
        return orders, amp, xi
    return _plain_spectrum(v)


def _env_feats(xi):
    """Envelope-spectrum energy — catches impact/modulation from corrugation."""
    env = np.abs(signal.hilbert(xi)); env = env - env.mean()
    orders, amp = order_spectrum(env); tot = amp.sum() + 1e-12
    return {"env_ord_1_5": amp[(orders >= 1) & (orders < 5)].sum() / tot,
            "env_ord_5_15": amp[(orders >= 5) & (orders < 15)].sum() / tot,
            "env_peak": float(orders[np.argmax(amp)])}


def features(df):
    """One file -> feature dict. Per-car features aggregated (mean/max/std) per side."""
    tacho = df.iloc[:, 0].values
    edge = _edges(tacho)
    out = {"revs": len(edge) / (2.0 * TEETH)}                # speed = revolutions in 1 s

    for side in ["I", "II"]:
        percar = {}   # feature -> list over cars
        for car in range(8):
            for pos in POS[side]:
                v = df.iloc[:, vib_col(car, pos)].values.astype(float)
                key = (car, pos)
                f = _time_feats(v)
                orders, amp, xi = _spectrum(v, edge)     # always returns a spectrum
                f.update(_spec_feats(orders, amp))
                f.update(_env_feats(xi))
                percar.setdefault(key, {}).update(f)
        # aggregate across the 32 channels of this side: mean, max, std
        allkeys = set().union(*[set(v) for v in percar.values()]) if percar else set()
        vals = {k: np.array([percar[c].get(k, np.nan) for c in percar]) for k in allkeys}
        for k, arr in vals.items():
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                continue
            out[f"{side}_{k}_mean"] = float(arr.mean())
            out[f"{side}_{k}_max"] = float(arr.max())
            out[f"{side}_{k}_std"] = float(arr.std())

    # side-contrast: the localisation signal (I vs II)
    for base in ["rms_mean", "rms_max", "ord_5_12_max", "ord_peakamp_max", "env_ord_5_15_max"]:
        a, b = out.get(f"I_{base}"), out.get(f"II_{base}")
        if a is not None and b is not None:
            out[f"contrast_{base}_ratio"] = a / (b + 1e-9)
            out[f"contrast_{base}_diff"] = a - b
    return out


class RailPipeline:
    def __init__(self, model=None, resample="borderline"):
        """
        resample:
          "borderline" (default) - BorderlineSMOTE the training set before the
              final fit, with class_weight=None. Validated best for the scored
              metric (rail_validate.py / rail_sideI_experiments.py, in-fold):
              macro-F1 0.804 -> 0.815 over 8 seeds, Side I recall 0.554 -> 0.625,
              Side I F1 0.553 -> 0.579. The lift is modest (within ~1 std) but
              consistent and never hurt macro-F1; Side I (14 files) stays the cap.
          "smote"  - plain SMOTE (essentially tied: macro-F1 0.814).
          None     - no resampling, class_weight="balanced" (the prior baseline,
              macro-F1 ~0.804). Quote any number WITH its spread.
        Resampling is applied only to training data (here and inside every CV
        fold in the validators) -> no leakage into evaluation.
        """
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.resample = resample
        cw = None if resample else "balanced"
        self.model = model or HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=2.0, class_weight=cw, random_state=0)
        self.feature_names = None

    def _sampler(self, y):
        if not self.resample:
            return None
        from imblearn.over_sampling import SMOTE, BorderlineSMOTE
        k = max(1, min(5, pd.Series(y).value_counts().min() - 1))
        if self.resample == "smote":
            return SMOTE(random_state=0, k_neighbors=k)
        return BorderlineSMOTE(random_state=0, k_neighbors=k,
                               m_neighbors=max(2, min(10, k * 2)))

    def _table(self, files):
        rows = [features(pd.read_csv(f, dtype=np.float32)) for f in files]
        return pd.DataFrame(rows)

    def fit(self, train_dir, labels_csv):
        lab = pd.read_csv(labels_csv)
        files = [os.path.join(train_dir, fn) for fn in lab["filename"]]
        X = self._table(files).fillna(0.0)
        self.feature_names = list(X.columns)
        Xv, yv = X.values, lab["label"].values
        sampler = self._sampler(yv)
        if sampler is not None:
            Xv, yv = sampler.fit_resample(Xv, yv)   # train-only oversampling
        self.model.fit(Xv, yv)
        return self

    def predict(self, test_dir, out_csv="rail_predictions.csv"):
        files = sorted(glob.glob(os.path.join(test_dir, "*.csv")),
                       key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
        X = self._table(files).reindex(columns=self.feature_names).fillna(0.0)
        pred = self.model.predict(X.values)
        res = pd.DataFrame({"file_id": [os.path.basename(f) for f in files], "prediction": pred})
        res.to_csv(out_csv, index=False)
        return res
