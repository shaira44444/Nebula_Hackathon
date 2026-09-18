"""
Rail corrugation — classify a 1-second axle-box recording
as Normal / Side I / Side II.

Physics:
Corrugation is periodic rail wear, so it excites vibration at a
characteristic frequency that scales with train speed.

We remove the speed dependence using ORDER TRACKING:
- use the toothed-wheel tacho
- resample vibration into angular domain
- extract order-spectrum features
- extract envelope features
- extract time-domain vibration features
- compare Side I vs Side II

Layout:
col 0 = tacho
cols 1..128 = car(0..7) x position(1..8) x [vibration, shock]

Positions:
1,3,5,7 -> Side I
2,4,6,8 -> Side II
"""

import os
import glob
import numpy as np
import pandas as pd

from scipy import signal, stats


# ============================================================
# CONSTANTS
# ============================================================

FS = 10_000.0
TEETH = 90
SPR = 128
MAXORD = 60

CLASSES = [
    "Normal",
    "Side I",
    "Side II"
]


# ============================================================
# COLUMN HELPERS
# ============================================================

def vib_col(car, pos):
    return 1 + car * 16 + (pos - 1) * 2


def shock_col(car, pos):
    return 2 + car * 16 + (pos - 1) * 2


POS = {
    "I": [1, 3, 5, 7],
    "II": [2, 4, 6, 8]
}


# ============================================================
# ORDER TRACKING
# ============================================================

def _edges(tacho):
    """
    Find 0/1 transitions from the toothed-wheel tacho.
    """

    s = (
        np.asarray(tacho) > 0.5
    ).astype(int)

    return (
        np.flatnonzero(
            np.diff(s) != 0
        ) + 1
    )


def order_resample(x, edge_idx):
    """
    Resample a vibration signal into the angular domain.

    Each tacho edge represents:
        1 / (2 * TEETH) revolution

    Returns a signal sampled at SPR samples per revolution.
    """

    if len(edge_idx) < 4:
        return None

    rev_at_edge = (
        np.arange(len(edge_idx))
        / (2.0 * TEETH)
    )

    n_rev = rev_at_edge[-1]

    if n_rev < 0.5:
        return None

    grid = np.arange(
        0,
        n_rev,
        1.0 / SPR
    )

    samp_at_rev = np.interp(
        grid,
        rev_at_edge,
        edge_idx
    )

    xi = np.interp(
        samp_at_rev,
        np.arange(len(x)),
        x
    )

    return xi - xi.mean()


def order_spectrum(xi):
    """
    Amplitude spectrum in orders
    (cycles per wheel revolution).
    """

    n = len(xi)

    amp = (
        np.abs(
            np.fft.rfft(
                xi * np.hanning(n)
            )
        )
        / n
    )

    orders = np.fft.rfftfreq(
        n,
        d=1.0 / SPR
    )

    return orders, amp


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def _time_feats(x):
    """
    Time-domain vibration features.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    rms = np.sqrt(
        np.mean(x ** 2)
    )

    return {
        "rms": float(rms),

        "kurt": float(
            stats.kurtosis(x)
        ),

        "crest": float(
            np.max(np.abs(x))
            / (rms + 1e-9)
        ),

        "p2p": float(
            np.ptp(x)
        )
    }


def _spec_feats(orders, amp):
    """
    Order-spectrum features.
    """

    tot = (
        amp.sum()
        + 1e-12
    )

    bands = [
        (1, 5),
        (5, 12),
        (12, 25),
        (25, 60)
    ]

    d = {}

    for lo, hi in bands:

        mask = (
            (orders >= lo)
            & (orders < hi)
        )

        d[f"ord_{lo}_{hi}"] = float(
            amp[mask].sum()
            / tot
        )

    d["ord_centroid"] = float(
        (orders * amp).sum()
        / tot
    )

    d["ord_peak"] = float(
        orders[
            np.argmax(amp)
        ]
    )

    pn = amp / tot

    d["ord_entropy"] = float(
        -(
            pn
            * np.log(
                pn + 1e-12
            )
        ).sum()
    )

    d["ord_peakamp"] = float(
        amp.max()
    )

    return d


def _plain_spectrum(v):
    """
    Fallback for low-speed files.

    Uses ordinary FFT and maps the frequency axis
    onto a pseudo-order axis so all files still
    produce the same feature names.
    """

    x = (
        np.asarray(
            v,
            dtype=float
        )
        - np.mean(v)
    )

    n = len(x)

    amp = (
        np.abs(
            np.fft.rfft(
                x * np.hanning(n)
            )
        )
        / n
    )

    freq = np.fft.rfftfreq(
        n,
        d=1.0 / FS
    )

    axis = (
        freq
        / (FS / 2.0)
        * MAXORD
    )

    return axis, amp, x


def _spectrum(v, edge):
    """
    Use order spectrum when tacho signal is sufficient.
    Otherwise fall back to ordinary FFT.
    """

    xi = order_resample(
        v,
        edge
    )

    if (
        xi is not None
        and len(xi) > 8
    ):

        orders, amp = order_spectrum(
            xi
        )

        return (
            orders,
            amp,
            xi
        )

    return _plain_spectrum(v)


def _env_feats(xi):
    """
    Envelope-spectrum features.

    Useful for identifying repeated impact /
    modulation patterns associated with corrugation.
    """

    env = np.abs(
        signal.hilbert(xi)
    )

    env = (
        env
        - env.mean()
    )

    orders, amp = order_spectrum(
        env
    )

    tot = (
        amp.sum()
        + 1e-12
    )

    mask_1_5 = (
        (orders >= 1)
        & (orders < 5)
    )

    mask_5_15 = (
        (orders >= 5)
        & (orders < 15)
    )

    return {
        "env_ord_1_5": float(
            amp[mask_1_5].sum()
            / tot
        ),

        "env_ord_5_15": float(
            amp[mask_5_15].sum()
            / tot
        ),

        "env_peak": float(
            orders[
                np.argmax(amp)
            ]
        )
    }


def features(df):
    """
    Convert one rail recording CSV into one feature dictionary.

    Features are extracted separately for Side I and Side II,
    then aggregated across axle-box channels.

    Side-contrast features help localise whether corrugation
    is stronger on Side I or Side II.
    """

    tacho = (
        df.iloc[:, 0]
        .values
    )

    edge = _edges(
        tacho
    )

    out = {
        "revs": (
            len(edge)
            / (2.0 * TEETH)
        )
    }

    # --------------------------------------------------------
    # Extract each side separately
    # --------------------------------------------------------

    for side in [
        "I",
        "II"
    ]:

        per_channel = {}

        for car in range(8):

            for pos in POS[side]:

                v = (
                    df.iloc[
                        :,
                        vib_col(
                            car,
                            pos
                        )
                    ]
                    .values
                    .astype(float)
                )

                key = (
                    car,
                    pos
                )

                f = _time_feats(
                    v
                )

                orders, amp, xi = (
                    _spectrum(
                        v,
                        edge
                    )
                )

                f.update(
                    _spec_feats(
                        orders,
                        amp
                    )
                )

                f.update(
                    _env_feats(
                        xi
                    )
                )

                per_channel[
                    key
                ] = f

        # ----------------------------------------------------
        # Aggregate features across channels
        # ----------------------------------------------------

        if per_channel:

            all_keys = set().union(
                *[
                    set(v.keys())
                    for v
                    in per_channel.values()
                ]
            )

        else:

            all_keys = set()

        for feature_name in all_keys:

            arr = np.array(
                [
                    per_channel[channel]
                    .get(
                        feature_name,
                        np.nan
                    )
                    for channel
                    in per_channel
                ],
                dtype=float
            )

            arr = arr[
                ~np.isnan(arr)
            ]

            if len(arr) == 0:
                continue

            out[
                f"{side}_{feature_name}_mean"
            ] = float(
                arr.mean()
            )

            out[
                f"{side}_{feature_name}_max"
            ] = float(
                arr.max()
            )

            out[
                f"{side}_{feature_name}_std"
            ] = float(
                arr.std()
            )

    # --------------------------------------------------------
    # Side I vs Side II contrast features
    # --------------------------------------------------------

    contrast_bases = [
        "rms_mean",
        "rms_max",
        "ord_5_12_max",
        "ord_peakamp_max",
        "env_ord_5_15_max"
    ]

    for base in contrast_bases:

        a = out.get(
            f"I_{base}"
        )

        b = out.get(
            f"II_{base}"
        )

        if (
            a is not None
            and b is not None
        ):

            out[
                f"contrast_{base}_ratio"
            ] = float(
                a
                / (b + 1e-9)
            )

            out[
                f"contrast_{base}_diff"
            ] = float(
                a - b
            )

    return out


# ============================================================
# MODEL PIPELINE
# ============================================================

class RailPipeline:

    def __init__(
        self,
        model=None,
        resample="borderline"
    ):
        """
        resample:
            "borderline"
                BorderlineSMOTE

            "smote"
                Standard SMOTE

            None
                No oversampling
        """

        from sklearn.ensemble import (
            HistGradientBoostingClassifier
        )

        self.resample = resample

        class_weight = (
            None
            if resample
            else "balanced"
        )

        self.model = (
            model
            or HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=2.0,
                class_weight=class_weight,
                random_state=0
            )
        )

        self.feature_names = None


    # ========================================================
    # SAMPLING
    # ========================================================

    def _sampler(
        self,
        y
    ):

        if not self.resample:
            return None

        from imblearn.over_sampling import (
            SMOTE,
            BorderlineSMOTE
        )

        counts = (
            pd.Series(y)
            .value_counts()
        )

        k = max(
            1,
            min(
                5,
                counts.min() - 1
            )
        )

        if self.resample == "smote":

            return SMOTE(
                random_state=0,
                k_neighbors=k
            )

        return BorderlineSMOTE(
            random_state=0,
            k_neighbors=k,
            m_neighbors=max(
                2,
                min(
                    10,
                    k * 2
                )
            )
        )


    # ========================================================
    # BUILD FEATURE TABLE
    # ========================================================

    def _table(
        self,
        files
    ):

        rows = []

        for f in files:

            df = pd.read_csv(
                f,
                dtype=np.float32
            )

            rows.append(
                features(df)
            )

        return pd.DataFrame(
            rows
        )


    # ========================================================
    # TRAIN FROM RAW FILES
    # ========================================================

    def fit(
        self,
        train_dir,
        labels_csv
    ):

        lab = pd.read_csv(
            labels_csv
        )

        files = [
            os.path.join(
                train_dir,
                fn
            )
            for fn
            in lab["filename"]
        ]

        X = (
            self._table(files)
            .fillna(0.0)
        )

        self.feature_names = list(
            X.columns
        )

        Xv = X.values

        yv = (
            lab["label"]
            .values
        )

        sampler = self._sampler(
            yv
        )

        if sampler is not None:

            Xv, yv = (
                sampler.fit_resample(
                    Xv,
                    yv
                )
            )

        self.model.fit(
            Xv,
            yv
        )

        return self


    # ========================================================
    # TRAIN FROM SAVED FEATURE TABLE
    # ========================================================

    def fit_feature_table(
        self,
        feature_csv
    ):
        """
        Train directly from rail_features_v2.csv.

        Expected columns:
        - filename (optional)
        - label
        - feature columns
        """

        df = pd.read_csv(
            feature_csv
        )

        if "label" not in df.columns:

            raise ValueError(
                "Feature CSV must contain "
                "a 'label' column."
            )

        drop_cols = [
            c
            for c in [
                "filename",
                "label"
            ]
            if c in df.columns
        ]

        X = (
            df
            .drop(
                columns=drop_cols
            )
            .fillna(0.0)
        )

        y = (
            df["label"]
            .values
        )

        self.feature_names = list(
            X.columns
        )

        Xv = X.values

        sampler = self._sampler(
            y
        )

        if sampler is not None:

            Xv, y = (
                sampler.fit_resample(
                    Xv,
                    y
                )
            )

        self.model.fit(
            Xv,
            y
        )

        return self


    # ========================================================
    # BASIC PREDICTION
    # ========================================================

    def predict(
        self,
        test_dir,
        out_csv="rail_predictions.csv"
    ):

        files = sorted(
            glob.glob(
                os.path.join(
                    test_dir,
                    "*.csv"
                )
            ),
            key=lambda p: int(
                "".join(
                    c
                    for c
                    in os.path.basename(p)
                    if c.isdigit()
                )
                or 0
            )
        )

        if not files:

            return pd.DataFrame(
                columns=[
                    "file_id",
                    "prediction"
                ]
            )

        X = (
            self._table(files)
            .reindex(
                columns=self.feature_names
            )
            .fillna(0.0)
        )

        pred = (
            self.model.predict(
                X.values
            )
        )

        result = pd.DataFrame(
            {
                "file_id": [
                    os.path.basename(f)
                    for f in files
                ],

                "prediction": pred
            }
        )

        result.to_csv(
            out_csv,
            index=False
        )

        return result


    # ========================================================
    # PREDICTION + CONFIDENCE
    # ========================================================

    def predict_proba(
        self,
        test_dir
    ):
        """
        Predict:
        - file_id
        - prediction
        - model confidence
        """

        files = sorted(
            glob.glob(
                os.path.join(
                    test_dir,
                    "*.csv"
                )
            ),
            key=lambda p: int(
                "".join(
                    c
                    for c
                    in os.path.basename(p)
                    if c.isdigit()
                )
                or 0
            )
        )

        if not files:

            return pd.DataFrame(
                columns=[
                    "file_id",
                    "prediction",
                    "confidence"
                ]
            )

        X = (
            self._table(files)
            .reindex(
                columns=self.feature_names
            )
            .fillna(0.0)
        )

        pred = (
            self.model.predict(
                X.values
            )
        )

        prob = (
            self.model.predict_proba(
                X.values
            )
        )

        confidence = (
            prob.max(
                axis=1
            )
        )

        result = pd.DataFrame(
            {
                "file_id": [
                    os.path.basename(f)
                    for f in files
                ],

                "prediction": pred,

                "confidence": confidence
            }
        )

        return result