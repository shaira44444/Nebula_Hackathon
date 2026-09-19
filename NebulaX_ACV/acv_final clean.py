import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.utils.class_weight import compute_sample_weight



warnings.filterwarnings("ignore")


# =====================================================================
# CONFIG
# =====================================================================

DATA_FOLDER = r"C:\NebulaX_ACV_Data"

DATA_DIR = Path(DATA_FOLDER)
OUTPUT_FILE = DATA_DIR / "acv_predictions.csv"

TRAIN_FILES = [
    "acv_case_01.xlsx",
    "acv_case_02.xlsx",
    "acv_case_03.xlsx",
    "acv_case_04.xlsx",
    "acv_case_05.xlsx",
    "acv_case_06.xlsx",
]

LABEL_FILE = "Train_Labels.csv"
TEST_FILE = "acv_test_case.xlsx"

SEEDS = [
    11,
    22,
    33,
    42,
    55,
    66,
    77,
    88,
    99,
    123,
    321,
    777,
    1001,
    2026,
    9999,
]


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


def main():

    check_files()
    dataset = build_training_data()
    features = get_features(dataset)

    print("\n" + "=" * 75)
    print("DATASET SUMMARY")
    print("=" * 75)
    print(f"Rows: {len(dataset)}")
    print(f"Fault examples: {int(dataset['fault'].sum())}")
    print(f"Normal examples: {int((dataset['fault'] == 0).sum())}")
    print(f"Usable features: {len(features)}")

    # Guard against accidentally changing the validated V4 feature pipeline.
    if len(features) != 624:
        raise RuntimeError(
            f"Expected 624 V4 features, but found {len(features)}. "
            "Stop and inspect the feature pipeline before using the ranking."
        )

    leaderboard = loco_validate(dataset, features)
    selected_model = leaderboard.iloc[0]["model"]

    test = prepare_test(features)
    train_final_and_predict(dataset, test, features, selected_model)


if __name__ == "__main__":
    main()
