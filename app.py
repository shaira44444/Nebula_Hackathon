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
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import KFold, cross_val_predict

from door_predict import DoorPipeline, parse_dt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# ACV MODEL CONFIGURATION
# ============================================================

ACV_DIR = Path(BASE_DIR) / "NebulaX_ACV"
DATA_DIR = ACV_DIR
TRAIN_FILES = [f"acv_case_{i:02d}.xlsx" for i in range(1, 7)]
LABEL_FILE = "Train_Labels.csv"
TEST_FILE = "acv_test_case.xlsx"
SEEDS = [11, 22, 33, 42, 55, 66, 77, 88, 99, 123, 321, 777, 1001, 2026, 9999]
OUTPUT_FILE = ACV_DIR / "acv_predictions.csv"
warnings.filterwarnings("ignore")

# =====================================================================
# EXPECTED COMMON ACV SIGNALS
# =====================================================================

EXPECTED_SIGNALS = {

    "acv_setting_mode": [
        "acv setting mode"
    ],

    "cooling_target": [
        "acv control temperature cooling"
    ],

    "heating_target": [
        "acv control temperature heating"
    ],

    "acv_running_mode": [
        "acv running mode"
    ],

    "outdoor_temperature": [
        "outdoor average temperature"
    ],

    "acv_information_valid": [
        "acv information valid"
    ],

    "load_halved": [
        "load halved"
    ],

    "indoor_temperature": [
        "indoor average temperature"
    ],
}


CATEGORICAL_SIGNALS = {
    "acv_setting_mode",
    "acv_running_mode",
    "acv_information_valid",
    "load_halved",
}


# =====================================================================
# HELPERS
# =====================================================================

def clean_text(value):

    value = str(value).lower().strip()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    return " ".join(
        value.split()
    )


def clean_feature_name(value):

    value = clean_text(value)

    return value.replace(
        " ",
        "_"
    )


def normalize_car(value):

    match = re.search(
        r"(\d+)",
        str(value)
    )

    if match:

        return f"{int(match.group(1)):02d}"

    return str(value)


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    )


# =====================================================================
# FILE CHECK
# =====================================================================

def check_files():

    required = (
        TRAIN_FILES
        +
        [LABEL_FILE, TEST_FILE]
    )

    print("\n" + "=" * 75)
    print("CHECKING FILES")
    print("=" * 75)

    missing = []

    for filename in required:

        path = DATA_DIR / filename

        if path.exists():

            print(
                f"[FOUND]   {filename}"
            )

        else:

            print(
                f"[MISSING] {filename}"
            )

            missing.append(
                filename
            )

    if missing:

        raise FileNotFoundError(
            f"Missing files: {missing}"
        )


# =====================================================================
# DETECT CAR COLUMNS
# =====================================================================

def find_car_columns(df):

    pattern = re.compile(
        r"^Car\s+(\d+)\s*-\s*(.+)$",
        re.IGNORECASE
    )

    cars = defaultdict(dict)

    for column in df.columns:

        match = pattern.match(
            str(column).strip()
        )

        if not match:
            continue

        car = (
            f"{int(match.group(1)):02d}"
        )

        parameter = (
            match.group(2).strip()
        )

        cars[car][parameter] = column

    return dict(cars)


# =====================================================================
# MAP REAL COLUMN NAME -> STANDARD SIGNAL
# =====================================================================

def identify_signal(parameter):

    cleaned = clean_text(
        parameter
    )

    for signal_name, aliases in EXPECTED_SIGNALS.items():

        for alias in aliases:

            alias_clean = clean_text(
                alias
            )

            if alias_clean in cleaned:

                return signal_name

    return None


# =====================================================================
# FIND AVAILABLE SIGNALS
# =====================================================================



# =====================================================================
# FIXED CATEGORICAL FEATURES
# =====================================================================

def categorical_features(
    series,
    prefix
):

    features = {}

    s = series.dropna()

    if len(s) == 0:

        return features

    features[
        f"{prefix}__missing_ratio"
    ] = (
        series.isna().mean()
    )

    features[
        f"{prefix}__unique_count"
    ] = s.nunique()

    # ---------------------------------------------------------
    # IMPORTANT:
    # Use actual state VALUE rather than state_0 meaning
    # "most frequent state".
    # ---------------------------------------------------------

    value_counts = (
        s.astype(str)
        .value_counts(
            normalize=True
        )
    )

    for value, ratio in (
        value_counts.items()
    ):

        value_name = (
            clean_feature_name(
                value
            )
        )

        if not value_name:

            value_name = "blank"

        features[
            f"{prefix}__value_{value_name}_ratio"
        ] = float(
            ratio
        )

    # ---------------------------------------------------------
    # Number of state changes
    # ---------------------------------------------------------

    string_series = (
        series
        .astype(str)
    )

    changes = (
        string_series
        !=
        string_series.shift(1)
    )

    if len(changes) > 1:

        features[
            f"{prefix}__change_ratio"
        ] = (
            changes.iloc[1:].mean()
        )

    return features


# =====================================================================
# NUMERIC FEATURES
# =====================================================================

def numeric_features(
    series,
    prefix
):

    features = {}

    s = safe_numeric(
        series
    )

    valid = s.dropna()

    features[
        f"{prefix}__missing_ratio"
    ] = (
        s.isna().mean()
    )

    if len(valid) == 0:

        return features

    stats = {

        "mean":
            valid.mean(),

        "median":
            valid.median(),

        "std":
            valid.std(),

        "min":
            valid.min(),

        "max":
            valid.max(),

        "range":
            valid.max()
            -
            valid.min(),

        "q10":
            valid.quantile(0.10),

        "q25":
            valid.quantile(0.25),

        "q75":
            valid.quantile(0.75),

        "q90":
            valid.quantile(0.90),

        "iqr":
            valid.quantile(0.75)
            -
            valid.quantile(0.25),

    }

    for name, value in stats.items():

        features[
            f"{prefix}__{name}"
        ] = value

    diff = (
        valid.diff().dropna()
    )

    if len(diff):

        features[
            f"{prefix}__diff_mean"
        ] = diff.mean()

        features[
            f"{prefix}__diff_abs_mean"
        ] = diff.abs().mean()

        features[
            f"{prefix}__diff_std"
        ] = diff.std()

    if len(valid) >= 2:

        try:

            x = np.arange(
                len(valid)
            )

            features[
                f"{prefix}__trend"
            ] = np.polyfit(
                x,
                valid.values,
                1
            )[0]

        except Exception:

            features[
                f"{prefix}__trend"
            ] = 0.0

    return features


# =====================================================================
# LONGEST CONSECUTIVE TRUE RUN
# =====================================================================

def longest_true_run(condition):

    condition = (
        pd.Series(condition)
        .fillna(False)
        .astype(bool)
        .values
    )

    longest = 0
    current = 0

    for value in condition:

        if value:

            current += 1

            longest = max(
                longest,
                current
            )

        else:

            current = 0

    return longest


# =====================================================================
# BOOLEAN CONDITION FEATURES
# =====================================================================

def condition_features(
    condition,
    prefix
):

    condition = (
        pd.Series(condition)
        .fillna(False)
        .astype(bool)
    )

    n = len(
        condition
    )

    if n == 0:

        return {}

    longest = (
        longest_true_run(
            condition
        )
    )

    return {

        f"{prefix}__ratio":
            condition.mean(),

        f"{prefix}__longest_run":
            longest,

        f"{prefix}__longest_run_ratio":
            longest / n,

    }


# =====================================================================
# EXTRACT STANDARD SIGNAL SERIES
# =====================================================================

def get_standard_signals(
    df,
    parameter_columns
):

    signals = {}

    for parameter, column in (
        parameter_columns.items()
    ):

        signal = identify_signal(
            parameter
        )

        if signal is not None:

            signals[
                signal
            ] = df[column]

    return signals


# =====================================================================
# EXTRACT ONE CAR
# =====================================================================

def extract_car_features(
    df,
    car,
    parameter_columns
):

    features = {
        "car": car
    }

    signals = (
        get_standard_signals(
            df,
            parameter_columns
        )
    )

    # =========================================================
    # RAW STANDARD SIGNAL FEATURES
    # =========================================================

    for signal_name, series in (
        signals.items()
    ):

        if signal_name in CATEGORICAL_SIGNALS:

            extracted = (
                categorical_features(
                    series,
                    signal_name
                )
            )

        else:

            extracted = (
                numeric_features(
                    series,
                    signal_name
                )
            )

        features.update(
            extracted
        )

    # =========================================================
    # DOMAIN VARIABLES
    # =========================================================

    indoor = None
    outdoor = None
    cooling_target = None
    heating_target = None

    if "indoor_temperature" in signals:

        indoor = safe_numeric(
            signals[
                "indoor_temperature"
            ]
        )

    if "outdoor_temperature" in signals:

        outdoor = safe_numeric(
            signals[
                "outdoor_temperature"
            ]
        )

    if "cooling_target" in signals:

        cooling_target = safe_numeric(
            signals[
                "cooling_target"
            ]
        )

    if "heating_target" in signals:

        heating_target = safe_numeric(
            signals[
                "heating_target"
            ]
        )

    # =========================================================
    # COOLING ERROR
    # =========================================================

    if (
        indoor is not None
        and
        cooling_target is not None
    ):

        cooling_error = (
            indoor
            -
            cooling_target
        )

        features.update(
            numeric_features(
                cooling_error,
                "ENG_cooling_error"
            )
        )

        # -----------------------------------------------------
        # Threshold persistence
        # -----------------------------------------------------

        for threshold in [
            0,
            0.5,
            1,
            1.5,
            2,
            2.5,
            3,
            4,
            5,
        ]:

            name = str(
                threshold
            ).replace(
                ".",
                "p"
            )

            condition = (
                cooling_error
                >
                threshold
            )

            features.update(
                condition_features(
                    condition,
                    f"ENG_error_gt_{name}"
                )
            )

        # -----------------------------------------------------
        # Cooling-error start vs end
        # -----------------------------------------------------

        valid_error = (
            cooling_error
            .dropna()
        )

        if len(valid_error) >= 10:

            chunk = max(
                1,
                len(valid_error) // 10
            )

            start_mean = (
                valid_error
                .iloc[:chunk]
                .mean()
            )

            end_mean = (
                valid_error
                .iloc[-chunk:]
                .mean()
            )

            features[
                "ENG_cooling_error_start_mean"
            ] = start_mean

            features[
                "ENG_cooling_error_end_mean"
            ] = end_mean

            features[
                "ENG_cooling_error_end_minus_start"
            ] = (
                end_mean
                -
                start_mean
            )

    # =========================================================
    # INDOOR VS OUTDOOR
    # =========================================================

    if (
        indoor is not None
        and
        outdoor is not None
    ):

        indoor_outdoor_delta = (
            indoor
            -
            outdoor
        )

        features.update(
            numeric_features(
                indoor_outdoor_delta,
                "ENG_indoor_minus_outdoor"
            )
        )

    # =========================================================
    # COOLING TARGET VS OUTDOOR
    # =========================================================

    if (
        cooling_target is not None
        and
        outdoor is not None
    ):

        cooling_demand = (
            outdoor
            -
            cooling_target
        )

        features.update(
            numeric_features(
                cooling_demand,
                "ENG_outdoor_minus_target"
            )
        )

    # =========================================================
    # TEMPERATURE RESPONSE
    # =========================================================

    if indoor is not None:

        valid_indoor = (
            indoor.dropna()
        )

        if len(valid_indoor) >= 2:

            indoor_diff = (
                valid_indoor.diff()
                .dropna()
            )

            features[
                "ENG_indoor_temp_falling_ratio"
            ] = (
                indoor_diff < 0
            ).mean()

            features[
                "ENG_indoor_temp_rising_ratio"
            ] = (
                indoor_diff > 0
            ).mean()

            features[
                "ENG_indoor_temp_change_abs_mean"
            ] = (
                indoor_diff
                .abs()
                .mean()
            )

            features[
                "ENG_indoor_temp_negative_change_mean"
            ] = (
                indoor_diff[
                    indoor_diff < 0
                ].mean()
                if (
                    indoor_diff < 0
                ).any()
                else
                0.0
            )

    # =========================================================
    # VALIDITY
    # =========================================================

    if "acv_information_valid" in signals:

        valid_series = (
            signals[
                "acv_information_valid"
            ]
        )

        features[
            "ENG_information_valid_unique"
        ] = (
            valid_series
            .dropna()
            .nunique()
        )

    return features


# =====================================================================
# EXTRACT CASE
# =====================================================================

def extract_case(
    file_path
):

    print(
        f"\nReading: {file_path.name}"
    )

    df = pd.read_excel(
        file_path,
        engine="openpyxl"
    )

    cars = find_car_columns(
        df
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    print(
        f"Cars: {sorted(cars.keys())}"
    )

    rows = []

    for car in sorted(cars):

        features = (
            extract_car_features(
                df,
                car,
                cars[car]
            )
        )

        features[
            "file_id"
        ] = file_path.name

        rows.append(
            features
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# LABELS
# =====================================================================

def load_labels():

    df = pd.read_csv(
        DATA_DIR / LABEL_FILE
    )

    file_col = None
    car_col = None

    for column in df.columns:

        lower = (
            str(column).lower()
        )

        if (
            "file" in lower
            or
            "case" in lower
        ):

            file_col = column

        if (
            "car" in lower
            or
            "fault" in lower
            or
            "label" in lower
        ):

            car_col = column

    if file_col is None:

        file_col = (
            df.columns[0]
        )

    if car_col is None:

        car_col = (
            df.columns[1]
        )

    return {

        str(row[file_col]).strip():
            normalize_car(
                row[car_col]
            )

        for _, row in df.iterrows()

    }


# =====================================================================
# RELATIVE FEATURES
# =====================================================================

def add_relative_features(
    df
):

    df = df.copy()

    ignore = {
        "file_id",
        "car",
        "fault"
    }

    numeric_columns = [

        column

        for column in df.columns

        if (
            column not in ignore
            and
            pd.api.types.is_numeric_dtype(
                df[column]
            )
        )
    ]

    new_columns = {}

    for column in numeric_columns:

        group = (
            df.groupby(
                "file_id"
            )[column]
        )

        median = (
            group.transform(
                "median"
            )
        )

        mean = (
            group.transform(
                "mean"
            )
        )

        std = (
            group.transform(
                "std"
            )
        )

        relative = (
            df[column]
            -
            median
        )

        new_columns[
            f"REL__{column}"
        ] = relative

        new_columns[
            f"ABSREL__{column}"
        ] = (
            relative.abs()
        )

        new_columns[
            f"ZREL__{column}"
        ] = (
            (
                df[column]
                -
                mean
            )
            /
            std.replace(
                0,
                np.nan
            )
        )

    relative_df = pd.DataFrame(
        new_columns,
        index=df.index
    )

    return pd.concat(
        [
            df,
            relative_df
        ],
        axis=1
    )


# =====================================================================
# BUILD TRAINING DATA
# =====================================================================

def build_training_data():

    labels = (
        load_labels()
    )

    cases = []

    print("\n" + "=" * 75)
    print("BUILDING TRAINING DATA")
    print("=" * 75)

    for filename in TRAIN_FILES:

        case = extract_case(
            DATA_DIR / filename
        )

        faulty = (
            labels[
                filename
            ]
        )

        case[
            "fault"
        ] = (
            case["car"]
            ==
            faulty
        ).astype(
            int
        )

        print(
            f"Faulty car: {faulty}"
        )

        cases.append(
            case
        )

    dataset = pd.concat(
        cases,
        ignore_index=True,
        sort=False
    )

    dataset = (
        add_relative_features(
            dataset
        )
    )

    return dataset


# =====================================================================
# FEATURE SELECTION
# =====================================================================

def get_features(
    dataset
):

    ignore = {
        "file_id",
        "car",
        "fault"
    }

    features = []

    for column in dataset.columns:

        if column in ignore:
            continue

        if not pd.api.types.is_numeric_dtype(
            dataset[column]
        ):
            continue

        # Must contain some real data
        if (
            dataset[column]
            .notna()
            .sum()
            ==
            0
        ):

            continue

        # -----------------------------------------------------
        # Avoid completely constant features
        # -----------------------------------------------------

        if (
            dataset[column]
            .nunique(
                dropna=True
            )
            <=
            1
        ):

            continue

        features.append(
            column
        )

    return features



# =====================================================================
# FINAL MODEL COMPARISON
# =====================================================================

SEED = 42
MODEL_NAMES = [
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "HistGradientBoosting",
]


def make_model(model_name, seed=SEED):
    if model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=1200,
            max_depth=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "Extra Trees":
        classifier = ExtraTreesClassifier(
            n_estimators=1200,
            max_depth=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "Gradient Boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=seed,
        )
    elif model_name == "HistGradientBoosting":
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=7,
            min_samples_leaf=2,
            l2_regularization=2.0,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", classifier),
    ])


def fit_model(model_name, X, y, seed=SEED):
    model = make_model(model_name, seed)

    # RF/ET already use class_weight='balanced'. For the two boosting models,
    # pass equivalent balanced sample weights during fitting.
    if model_name in {"Gradient Boosting", "HistGradientBoosting"}:
        weights = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, model__sample_weight=weights)
    else:
        model.fit(X, y)

    return model


def create_ranking(cars, scores):
    result = pd.DataFrame({"car": list(cars), "score": list(scores)})
    return result.sort_values(
        ["score", "car"], ascending=[False, True]
    ).reset_index(drop=True)


def competition_score(rank, n=8):
    return (n - (rank - 1)) / n


def loco_validate(dataset, features):
    print("\n" + "=" * 75)
    print("LEAVE-ONE-CASE-OUT MODEL COMPARISON")
    print("=" * 75)

    rows = []
    cases = list(dict.fromkeys(dataset["file_id"].tolist()))

    for model_name in MODEL_NAMES:
        print(f"\n{model_name}")
        print("-" * 75)

        scores = []
        ranks = []
        top1 = 0

        for held_out in cases:
            train = dataset[dataset["file_id"] != held_out]
            valid = dataset[dataset["file_id"] == held_out]

            model = fit_model(
                model_name,
                train[features],
                train["fault"],
                SEED,
            )
            probabilities = model.predict_proba(valid[features])[:, 1]
            ranking = create_ranking(valid["car"], probabilities)

            true_car = str(valid.loc[valid["fault"] == 1, "car"].iloc[0]).zfill(2)
            rank = int(ranking.index[ranking["car"].astype(str).str.zfill(2) == true_car][0]) + 1
            score = competition_score(rank)

            scores.append(score)
            ranks.append(rank)
            top1 += int(rank == 1)

            ranking_text = "|".join(ranking["car"].astype(str).str.zfill(2))
            print(
                f"{held_out:<20} True={true_car}  Rank={rank}  "
                f"Score={score:.3f}  {ranking_text}"
            )

        row = {
            "model": model_name,
            "average_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "top1": int(top1),
            "worst_rank": int(max(ranks)),
        }
        rows.append(row)
        print(
            f"AVERAGE={row['average_score']:.4f}  "
            f"MIN={row['min_score']:.4f}  TOP1={top1}/6  "
            f"WORST_RANK={row['worst_rank']}"
        )

    leaderboard = pd.DataFrame(rows)
    # Primary: official average competition score.
    # Tie-breaks: minimum score, Top-1 count, worst rank.
    # MODEL_NAMES order is preserved for a complete tie, keeping RF first.
    leaderboard["model_order"] = leaderboard["model"].map(
        {name: i for i, name in enumerate(MODEL_NAMES)}
    )
    leaderboard = leaderboard.sort_values(
        ["average_score", "min_score", "top1", "worst_rank", "model_order"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    print("\n" + "=" * 75)
    print("MODEL LEADERBOARD")
    print("=" * 75)
    print(
        leaderboard[
            ["model", "average_score", "min_score", "top1", "worst_rank"]
        ].to_string(index=False)
    )

    return leaderboard


def prepare_test(features):
    print("\n" + "=" * 75)
    print("PREPARING TEST CASE")
    print("=" * 75)

    test = extract_case(DATA_DIR / TEST_FILE)
    test = add_relative_features(test)

    for feature in features:
        if feature not in test.columns:
            test[feature] = np.nan

    return test


def train_final_and_predict(dataset, test, features, selected_model):
    print("\n" + "=" * 75)
    print("FINAL TEST PREDICTION - 15-SEED PROBABILITY AVERAGE")
    print("=" * 75)
    print(f"Selected model: {selected_model}")
    print(f"Seeds averaged: {len(SEEDS)}")

    # The model has already been selected using fixed-seed LOCO validation.
    # For the final hidden-test ranking only, retrain the selected model once
    # per seed and average its fault probabilities. This reduces the effect
    # of one arbitrary random seed on close ranking positions.
    all_probabilities = []

    for seed in SEEDS:
        model = fit_model(
            selected_model,
            dataset[features],
            dataset["fault"],
            seed,
        )
        probabilities = model.predict_proba(test[features])[:, 1]
        all_probabilities.append(probabilities)

    mean_probabilities = np.mean(np.vstack(all_probabilities), axis=0)
    ranking = create_ranking(test["car"], mean_probabilities)
    ranking_text = "|".join(ranking["car"].astype(str).str.zfill(2))

    print(f"\nFinal averaged ranking: {ranking_text}")
    for i, row in ranking.iterrows():
        print(
            f"  {i + 1}. Car {str(row['car']).zfill(2)}  "
            f"mean_score={row['score']:.4f}"
        )

    submission = pd.DataFrame({
        "file_id": [TEST_FILE],
        "ranked_cars": [ranking_text],
    })
    submission.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved only output file: {OUTPUT_FILE}")
    return ranking_text




RAIL_DIR = os.path.join(BASE_DIR, "rail data")

# Allow Python to import rail_predict.py from rail data/
if RAIL_DIR not in sys.path:
    sys.path.append(RAIL_DIR)

from rail_predict import RailPipeline



# ============================================================
# SHM — STRUCTURAL HEALTH MONITORING
# ============================================================

SHM_DIR = Path(BASE_DIR) / "SHM"
SHM_TRAIN_DIR = SHM_DIR / "Train"
SHM_LABEL_FILE = SHM_DIR / "Train_Labels.csv"


def _shm_numeric_signal(df):
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        converted = {}
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= max(10, int(len(df) * 0.5)):
                converted[c] = s
        numeric = pd.DataFrame(converted)

    if numeric.empty:
        raise ValueError("No usable numeric stress signal was found in the CSV.")

    stress_cols = [c for c in numeric.columns if "stress" in str(c).lower()]
    if stress_cols:
        s = numeric[stress_cols[0]]
    else:
        stds = numeric.std(numeric_only=True).replace([np.inf, -np.inf], np.nan)
        s = numeric[stds.idxmax()] if not stds.dropna().empty else numeric.iloc[:, 0]

    return pd.to_numeric(s, errors="coerce").dropna()


def extract_shm_features_from_df(df):
    s = _shm_numeric_signal(df)
    if len(s) < 5:
        raise ValueError("The uploaded stress signal contains too few numeric samples.")

    x = s.to_numpy(dtype=float)
    d = np.diff(x)
    absx = np.abs(x)
    qs = np.quantile(x, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    q01, q05, q10, q25, q50, q75, q90, q95, q99 = qs

    turning = int(np.sum(np.sign(d[1:]) != np.sign(d[:-1]))) if len(d) >= 2 else 0
    centered = x - np.mean(x)
    zero_crossings = int(np.sum(np.signbit(centered[1:]) != np.signbit(centered[:-1])))
    rms = float(np.sqrt(np.mean(x ** 2)))
    std = float(np.std(x))
    peak = float(np.max(absx))
    mean_abs = float(np.mean(absx))

    feats = {
        "n_samples": float(len(x)), "mean": float(np.mean(x)), "std": std,
        "rms": rms, "min": float(np.min(x)), "max": float(np.max(x)),
        "range": float(np.max(x) - np.min(x)), "median": float(q50),
        "q01": float(q01), "q05": float(q05), "q10": float(q10),
        "q25": float(q25), "q75": float(q75), "q90": float(q90),
        "q95": float(q95), "q99": float(q99), "iqr": float(q75 - q25),
        "mean_abs": mean_abs, "peak_abs": peak,
        "crest_factor": float(peak / rms) if rms > 0 else 0.0,
        "shape_factor": float(rms / mean_abs) if mean_abs > 0 else 0.0,
        "mean_abs_change": float(np.mean(np.abs(d))) if len(d) else 0.0,
        "std_change": float(np.std(d)) if len(d) else 0.0,
        "max_abs_change": float(np.max(np.abs(d))) if len(d) else 0.0,
        "turning_points": float(turning),
        "turning_ratio": float(turning / max(len(x) - 2, 1)),
        "zero_crossings": float(zero_crossings),
        "zero_crossing_ratio": float(zero_crossings / max(len(x) - 1, 1)),
    }

    if std > 0:
        z = (x - np.mean(x)) / std
        feats["skewness"] = float(np.mean(z ** 3))
        feats["kurtosis"] = float(np.mean(z ** 4))
    else:
        feats["skewness"] = feats["kurtosis"] = 0.0

    for chunks in (10, 20, 50, 100):
        if len(x) >= chunks:
            parts = np.array_split(x, chunks)
            ranges = np.array([np.max(p) - np.min(p) for p in parts if len(p)])
            feats[f"chunk{chunks}_range_mean"] = float(np.mean(ranges))
            feats[f"chunk{chunks}_range_max"] = float(np.max(ranges))
            feats[f"chunk{chunks}_range_std"] = float(np.std(ranges))

    return feats, s


def _find_shm_training_paths():
    train_candidates = [
        SHM_TRAIN_DIR, SHM_DIR / "train",
        Path(BASE_DIR) / "shm" / "Train", Path(BASE_DIR) / "shm" / "train"
    ]
    label_candidates = [
        SHM_LABEL_FILE, SHM_DIR / "train_labels.csv",
        Path(BASE_DIR) / "shm" / "Train_Labels.csv",
        Path(BASE_DIR) / "shm" / "train_labels.csv"
    ]
    return (
        next((p for p in train_candidates if p.exists()), None),
        next((p for p in label_candidates if p.exists()), None),
    )


@st.cache_resource(show_spinner=False)
def load_shm_model():
    train_dir, label_file = _find_shm_training_paths()
    if train_dir is None or label_file is None:
        raise FileNotFoundError(
            "SHM training data not found. Add SHM/Train/*.csv and SHM/Train_Labels.csv."
        )

    labels = pd.read_csv(label_file)
    lower = {str(c).strip().lower(): c for c in labels.columns}
    filename_col = lower.get("filename") or lower.get("file_id")
    damage_col = lower.get("damage") or lower.get("prediction")
    if filename_col is None or damage_col is None:
        raise ValueError("Train_Labels.csv must contain filename and damage columns.")

    rows, targets = [], []
    for _, label_row in labels.iterrows():
        filename = str(label_row[filename_col]).strip()
        path = train_dir / filename
        if not path.exists():
            continue
        feats, _ = extract_shm_features_from_df(pd.read_csv(path))
        rows.append(feats)
        targets.append(float(label_row[damage_col]))

    if len(rows) < 5:
        raise RuntimeError(f"Only {len(rows)} labelled SHM training files were found.")

    table = pd.DataFrame(rows)
    feature_cols = list(table.columns)
    X = table[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = np.asarray(targets, dtype=float)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=1200, max_depth=8, min_samples_leaf=2,
            max_features=0.8, random_state=42, n_jobs=-1
        )),
    ])

    folds = min(5, len(y))
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    cv_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
    mape = float(np.mean(np.abs(y - cv_pred) / np.maximum(np.abs(y), 1e-12)))
    score = max(0.0, 1.0 - mape)
    model.fit(X, y)

    return model, feature_cols, {
        "mape": mape, "competition_score": score,
        "n_train": len(y), "folds": folds
    }


def predict_shm(uploaded_file):
    raw = pd.read_csv(uploaded_file)
    feats, signal = extract_shm_features_from_df(raw)
    model, feature_cols, performance = load_shm_model()
    row = pd.DataFrame([{c: feats.get(c, np.nan) for c in feature_cols}])
    prediction = max(0.0, float(model.predict(row)[0]))
    x = signal.to_numpy(dtype=float)
    return {
        "prediction": prediction, "signal": signal, "features": feats,
        "performance": performance,
        "stress_range": float(np.max(x) - np.min(x)),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "std": float(np.std(x)),
        "mean_abs_change": float(np.mean(np.abs(np.diff(x)))) if len(x) > 1 else 0.0,
        "cycle_count": int(feats.get("turning_points", 0) / 2),
        "max_cycle_range": float(feats.get("chunk50_range_max", np.max(x) - np.min(x))),
    }


def shm_risk_label(damage):
    if damage >= 1.0:
        return "Critical", "🔴"
    if damage >= 0.67:
        return "High", "🟠"
    if damage >= 0.33:
        return "Moderate", "🟡"
    return "Low", "🟢"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NEBULA X — Condition Monitoring",
    layout="wide"
)


st.markdown("""
<style>
.stApp { background:#f6f9fc; color:#14233b; }
[data-testid="stHeader"] { background:rgba(246,249,252,.94); }
div[data-testid="stMetric"] {
    background:white; border:1px solid #e2e9f1; border-radius:10px;
    padding:12px 14px; box-shadow:0 2px 8px rgba(17,42,76,.04);
}
div[data-testid="stDataFrame"] {
    border:1px solid #e2e9f1; border-radius:10px; overflow:hidden;
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background:#1769e0; border-color:#1769e0; border-radius:8px; font-weight:700;
}
h1,h2,h3 { color:#14233b; }
</style>
""", unsafe_allow_html=True)

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

st.title("🚆 NEBULA X — Condition Monitoring")

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
        "ACV",
        "SHM"
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
# ACV LEAK LOCALISATION
# ============================================================

elif subsystem == "ACV":

    st.markdown("## ❄️ ACV Leak Localisation")
    st.write(
        "Rank the eight cars from most to least likely to have an ACV "
        "refrigerant leak using the validated V4 sensor-feature pipeline."
    )

    # Training data remains in NebulaX_ACV/. The uploaded workbook is the
    # test journey that will be ranked.
    st.markdown("### 1 · Upload Test Case")
    acv_upload = st.file_uploader(
        "Upload ACV test case (.xlsx)",
        type=["xlsx"],
        key="acv_upload",
    )

    if acv_upload is None:
        st.info("Upload an ACV Excel test case to begin.")
        st.stop()

    st.success(f"Loaded **{acv_upload.name}**")

    if st.button("▶ Run ACV Analysis", type="primary", key="run_acv"):
        with st.spinner("Building V4 features, validating models and ranking cars..."):
            suffix = Path(acv_upload.name).suffix or ".xlsx"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(acv_upload.getbuffer())
                test_path = Path(tmp.name)

            try:
                dataset = build_training_data()
                features = get_features(dataset)
                if len(features) != 624:
                    raise RuntimeError(
                        f"Expected 624 validated V4 features, found {len(features)}."
                    )

                leaderboard = loco_validate(dataset, features)
                selected_model = str(leaderboard.iloc[0]["model"])

                test = extract_case(test_path)
                test = add_relative_features(test)
                for feature in features:
                    if feature not in test.columns:
                        test[feature] = np.nan

                all_probabilities = []
                for seed in SEEDS:
                    model = fit_model(
                        selected_model, dataset[features], dataset["fault"], seed
                    )
                    all_probabilities.append(
                        model.predict_proba(test[features])[:, 1]
                    )

                mean_probabilities = np.mean(np.vstack(all_probabilities), axis=0)
                ranking = create_ranking(test["car"], mean_probabilities)
                ranking["car"] = ranking["car"].astype(str).str.zfill(2)
                ranking.insert(0, "rank", range(1, len(ranking) + 1))
                ranking_text = "|".join(ranking["car"])

                submission = pd.DataFrame({
                    "file_id": [acv_upload.name],
                    "ranked_cars": [ranking_text],
                })

                st.session_state["acv_results"] = {
                    "leaderboard": leaderboard,
                    "selected_model": selected_model,
                    "ranking": ranking,
                    "ranking_text": ranking_text,
                    "submission_csv": submission.to_csv(index=False),
                    "training_cases": len(TRAIN_FILES),
                    "total_examples": len(dataset),
                    "fault_examples": int(dataset["fault"].sum()),
                    "normal_examples": int((dataset["fault"] == 0).sum()),
                    "features": len(features),
                    "test_name": acv_upload.name,
                }
            finally:
                try:
                    test_path.unlink(missing_ok=True)
                except Exception:
                    pass

    if "acv_results" not in st.session_state:
        st.stop()

    r = st.session_state["acv_results"]
    st.success("✅ Analysis completed — ready for results.")

    st.markdown("### 2 · Analysis Overview")
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown("**① Load Data**  \n6 training cases + test case")
    p2.markdown(f"**② Build Features**  \n{r['features']} features (V4 pipeline)")
    p3.markdown("**③ Validate Models**  \nLeave-One-Case-Out")
    p4.markdown("**④ Predict Test Case**  \n15-seed probability average")

    left, right = st.columns([1, 2])

    with left:
        st.markdown("### Dataset Summary")
        m1, m2 = st.columns(2)
        m1.metric("Training Cases", r["training_cases"])
        m2.metric("Total Examples", r["total_examples"])
        m3, m4 = st.columns(2)
        m3.metric("Usable Features", r["features"])
        m4.metric("Cars per Case", 8)
        st.caption(
            f"{r['fault_examples']} faulty + {r['normal_examples']} normal training examples"
        )

        st.markdown("### 🏆 Selected Model")
        st.markdown(f"## {r['selected_model']}")
        best = r["leaderboard"].iloc[0]
        st.write(f"✓ Average LOCO score: **{best['average_score']:.4f}**")
        st.write(f"✓ Minimum LOCO score: **{best['min_score']:.4f}**")
        st.write(f"✓ Top-1 fault localisation: **{int(best['top1'])}/6**")
        st.write(f"✓ Worst validation rank: **{int(best['worst_rank'])}**")
        st.write(f"✓ Final prediction averaged across **{len(SEEDS)} seeds**")

        st.markdown("### Prediction Summary")
        st.success(f"Final Ranking: {r['ranking_text']}")
        st.metric("Highest-ranked fault candidate", f"Car {r['ranking'].iloc[0]['car']}")

    with right:
        st.markdown("### Model Comparison · Leave-One-Case-Out")
        board = r["leaderboard"][[
            "model", "average_score", "min_score", "top1", "worst_rank"
        ]].copy()
        board.columns = ["Model", "Average Score", "Min Score", "Top-1", "Worst Rank"]
        board["Top-1"] = board["Top-1"].astype(int).astype(str) + "/6"
        st.dataframe(
            board,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Average Score": st.column_config.NumberColumn(format="%.4f"),
                "Min Score": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        st.markdown(f"### Final Test Prediction · {len(SEEDS)}-Seed Average")
        display_rank = r["ranking"].rename(columns={
            "rank": "Rank", "car": "Car", "score": "Average Fault Score"
        })
        st.dataframe(
            display_rank,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "Car": st.column_config.TextColumn(width="small"),
                "Average Fault Score": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="%.4f"
                ),
            },
        )

    st.caption(
        "Ranking indicates relative model-estimated fault likelihood across the eight cars; "
        "the hidden test label is not available to the app."
    )

    st.markdown("### 3 · Download")
    st.download_button(
        "⬇ Download acv_predictions.csv",
        r["submission_csv"],
        file_name="acv_predictions.csv",
        mime="text/csv",
        type="primary",
    )


# ============================================================
# SHM — STRUCTURAL HEALTH MONITORING
# ============================================================

elif subsystem == "SHM":

    st.markdown("## 🏗️ Structural Health Monitoring")
    st.write(
        "Upload a dynamic-stress time-series file to predict its "
        "**cumulative fatigue damage**."
    )
    st.info(
        "ℹ️ This tool uses a machine-learning regression model to estimate "
        "cumulative fatigue damage from stress-signal characteristics."
    )

    upload_col, result_col = st.columns([1.05, 1.0], gap="large")

    with upload_col:
        st.markdown("### 1 · Upload Data")
        shm_upload = st.file_uploader(
            "Drag and drop or choose an SHM CSV file",
            type=["csv"],
            key="shm_upload",
        )
        if shm_upload is not None:
            st.success(f"✓ File uploaded: **{shm_upload.name}**")
            if st.button(
                "⚡ Predict Fatigue Damage",
                type="primary",
                use_container_width=True,
                key="run_shm",
            ):
                with st.spinner("Extracting stress features and predicting damage..."):
                    try:
                        shm_upload.seek(0)
                        result = predict_shm(shm_upload)
                        result["file_name"] = shm_upload.name
                        st.session_state["shm_result"] = result
                    except Exception as exc:
                        st.session_state.pop("shm_result", None)
                        st.error(f"SHM analysis failed: {exc}")
        else:
            st.caption("Supports .csv dynamic-stress files.")

    if "shm_result" in st.session_state:
        r = st.session_state["shm_result"]
        damage = float(r["prediction"])
        risk, risk_icon = shm_risk_label(damage)

        with result_col:
            st.markdown("### 2 · Prediction Result")
            st.markdown(
                f"""
                <div style="border:1px solid #cfe8d7;background:#effbf3;
                border-radius:12px;padding:20px 24px;margin-bottom:14px;">
                    <div style="font-size:14px;color:#496357;">
                    Predicted Cumulative Fatigue Damage</div>
                    <div style="font-size:40px;font-weight:800;color:#147a43;">
                    {damage:.6f}</div>
                    <div style="margin-top:8px;font-weight:700;color:#31443a;">
                    {risk_icon} {risk} risk indicator</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            a, b, c = st.columns(3)
            a.metric("Stress Range", f"{r['stress_range']:.2f}")
            b.metric("RMS", f"{r['rms']:.2f}")
            c.metric("Cycle Estimate", f"{r['cycle_count']:,}")
            d, e = st.columns(2)
            d.metric("Std. Deviation", f"{r['std']:.2f}")
            e.metric("Max Local Range", f"{r['max_cycle_range']:.2f}")

        chart_col, feature_col = st.columns([1.65, 1.0], gap="large")

        with chart_col:
            st.markdown("### 3 · Stress Signal (Time Series)")
            signal = r["signal"]
            max_points = 5000
            if len(signal) > max_points:
                sample_idx = np.linspace(0, len(signal) - 1, max_points).astype(int)
                y_plot = signal.iloc[sample_idx].to_numpy()
                x_plot = sample_idx
                st.caption(f"Showing {max_points:,} sampled points from {len(signal):,}.")
            else:
                y_plot = signal.to_numpy()
                x_plot = np.arange(len(signal))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_plot, y=y_plot, mode="lines",
                line=dict(width=1.15, color="#2477d4"),
                hovertemplate="Sample %{x}<br>Stress %{y:.3f}<extra></extra>",
            ))
            fig.update_layout(
                height=350, xaxis_title="Measurement Sequence",
                yaxis_title="Stress", showlegend=False,
                margin=dict(l=20, r=20, t=10, b=20),
                paper_bgcolor="white", plot_bgcolor="white",
            )
            fig.update_xaxes(showgrid=True, gridcolor="#edf1f5")
            fig.update_yaxes(showgrid=True, gridcolor="#edf1f5")
            st.plotly_chart(fig, use_container_width=True)

        with feature_col:
            st.markdown("### 4 · Key Signal Characteristics")
            names = ["Range", "RMS", "Std Dev", "Mean Abs Change"]
            vals = [r["stress_range"], r["rms"], r["std"], r["mean_abs_change"]]
            fig2 = go.Figure(go.Bar(
                x=names, y=vals,
                text=[f"{v:.2f}" for v in vals],
                textposition="outside",
                marker_color=["#2f7de1", "#35a66f", "#ef7f32", "#8b55c7"],
            ))
            fig2.update_layout(
                height=350, showlegend=False, yaxis_title="Value",
                margin=dict(l=15, r=15, t=10, b=20),
                paper_bgcolor="white", plot_bgcolor="white",
            )
            fig2.update_yaxes(showgrid=True, gridcolor="#edf1f5")
            st.plotly_chart(fig2, use_container_width=True)

        detail_col, model_col = st.columns([1.45, 1.0], gap="large")

        with detail_col:
            st.markdown("### 5 · Additional Feature Details")
            f = r["features"]
            details = pd.DataFrame({
                "Feature": [
                    "Mean", "Std Deviation", "Minimum", "Maximum",
                    "Interquartile Range (IQR)", "Mean Absolute Value",
                    "Mean Absolute Change", "Turning Points",
                ],
                "Value": [
                    f.get("mean", np.nan), f.get("std", np.nan),
                    f.get("min", np.nan), f.get("max", np.nan),
                    f.get("iqr", np.nan), f.get("mean_abs", np.nan),
                    f.get("mean_abs_change", np.nan), f.get("turning_points", np.nan),
                ],
            })
            st.dataframe(
                details, hide_index=True, use_container_width=True,
                column_config={"Value": st.column_config.NumberColumn(format="%.4f")},
            )

        with model_col:
            st.markdown("### 6 · Model Information")
            st.markdown("**Model Used**")
            st.markdown("#### 🌐 Random Forest Regressor")
            perf = r["performance"]
            st.caption(
                f"Validation performance · {perf['folds']}-fold CV on "
                f"{perf['n_train']} labelled training files"
            )
            m1, m2 = st.columns(2)
            m1.metric("MAPE", f"{perf['mape']:.2%}")
            m2.metric("Competition Score", f"{perf['competition_score']:.4f}")

            submission = pd.DataFrame({
                "file_id": [r["file_name"]],
                "prediction": [damage],
            })
            st.download_button(
                "⬇ Download shm_predictions.csv",
                submission.to_csv(index=False),
                file_name="shm_predictions.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

            if st.button("↻ Predict Another File", use_container_width=True, key="reset_shm"):
                st.session_state.pop("shm_result", None)
                st.rerun()

    else:
        with result_col:
            st.markdown("### 2 · Prediction Result")
            st.info("Upload an SHM CSV file and run the model to view the prediction.")

