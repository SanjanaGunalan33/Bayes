"""
Final Bayesian Root Cause Analysis Model
=========================================

Dataset:
    RCAEval RE1 + RE2

Target classes:
    cpu, mem, disk, delay, loss, socket

Pipeline:
    1. Load RE1 + RE2 metadata.
    2. Load telemetry for each incident.
    3. Split telemetry using the recorded injection time.
    4. Exclude incidents without valid PRE and POST observations.
    5. Extract bounded service-level telemetry changes.
    6. Preserve unavailable telemetry as missing (NaN).
    7. Create a deterministic stratified train/validation/test split.
    8. Learn discretization thresholds using TRAINING DATA ONLY.
    9. Learn Bayesian priors and CPTs using TRAINING DATA ONLY.
   10. During inference, ignore unavailable telemetry.
   11. Evaluate using Top-1, Top-3, MRR, confusion matrix,
       precision, recall and F1.

Important:
    Fault labels are NEVER used during feature extraction.
"""

import os
import json
import math

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

CASES_FILE = "cases.parquet"
CACHE_DIR = "rcaeval_cache"

RANDOM_SEED = 2026

DATASETS = [
    "RE1-OB",
    "RE1-SS",
    "RE1-TT",
    "RE2-OB",
    "RE2-SS",
    "RE2-TT",
]

FAULTS = [
    "cpu",
    "mem",
    "disk",
    "delay",
    "loss",
    "socket",
]

FEATURES = [
    "cpu_change",
    "mem_change",
    "diskio_change",
    "socket_change",
    "workload_change",
    "latency50_change",
    "latency90_change",
]

# Discrete states for every observed feature.
STATES = [
    "low",
    "medium",
    "high",
]

# Laplace smoothing for CPT estimation.
ALPHA = 1.0

EPS = 1e-12


# =============================================================================
# TELEMETRY COLUMN IDENTIFICATION
# =============================================================================

def get_family_columns(columns, family):
    """
    Return telemetry columns belonging to one metric family.

    Missing metric families are NOT replaced by zeros.
    """

    result = []

    for col in columns:

        c = str(col).lower()

        if c == "time":
            continue

        if family == "cpu" and c.endswith("_cpu"):
            result.append(col)

        elif family == "mem" and c.endswith("_mem"):
            result.append(col)

        elif family == "diskio" and (
            c.endswith("_diskio")
            or c.endswith("_disk_io")
        ):
            result.append(col)

        elif family == "socket" and c.endswith("_socket"):
            result.append(col)

        elif family == "workload" and (
            c.endswith("_workload")
            or c.endswith("_load")
        ):
            result.append(col)

        elif family == "error" and c.endswith("_error"):
            result.append(col)

        elif family == "latency50" and (
            c.endswith("_latency-50")
            or c.endswith("_latency_50")
        ):
            result.append(col)

        elif family == "latency90" and (
            c.endswith("_latency-90")
            or c.endswith("_latency_90")
        ):
            result.append(col)

    return result


# =============================================================================
# PRE / POST SPLIT
# =============================================================================

def split_pre_post(metrics, inject_time):
    """
    Split telemetry using the injection timestamp supplied by RCAEval.

    We deliberately do NOT invent a midpoint when a valid pre/post split
    cannot be obtained.
    """

    if "time" not in metrics.columns:
        raise ValueError("Telemetry has no time column.")

    time = pd.to_numeric(
        metrics["time"],
        errors="coerce"
    )

    inject_time = float(inject_time)

    pre = metrics[
        time < inject_time
    ].copy()

    post = metrics[
        time >= inject_time
    ].copy()

    if len(pre) < 5 or len(post) < 5:
        raise ValueError(
            f"Insufficient pre/post samples: "
            f"pre={len(pre)}, post={len(post)}"
        )

    return pre, post


# =============================================================================
# ROBUST CHANGE
# =============================================================================

def bounded_relative_change(pre_value, post_value):
    """
    Symmetric bounded relative change:

                   post - pre
        --------------------------------
        |post| + |pre| + epsilon

    Approximate range:
        -1 = strong decrease
         0 = little/no change
        +1 = strong increase

    Unlike the old z-score representation, this does not explode when the
    baseline variance is close to zero.
    """

    denominator = (
        abs(post_value)
        + abs(pre_value)
        + EPS
    )

    return (
        post_value - pre_value
    ) / denominator


def strongest_service_change(pre, post, columns):
    """
    Calculate pre/post change independently for every service-specific metric.

    Return the change having the largest absolute magnitude.

    This preserves localized anomalies instead of averaging the affected
    service together with many unaffected services.
    """

    if not columns:
        return np.nan

    changes = []

    for col in columns:

        pre_values = pd.to_numeric(
            pre[col],
            errors="coerce"
        )

        post_values = pd.to_numeric(
            post[col],
            errors="coerce"
        )

        pre_mean = pre_values.mean()
        post_mean = post_values.mean()

        if pd.isna(pre_mean) or pd.isna(post_mean):
            continue

        change = bounded_relative_change(
            float(pre_mean),
            float(post_mean)
        )

        changes.append(change)

    if not changes:
        return np.nan

    return float(
        max(
            changes,
            key=lambda x: abs(x)
        )
    )


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(metrics, inject_time):
    """
    Extract telemetry evidence WITHOUT using the fault label.
    """

    pre, post = split_pre_post(
        metrics,
        inject_time
    )

    family_map = {
        "cpu_change": "cpu",
        "mem_change": "mem",
        "diskio_change": "diskio",
        "socket_change": "socket",
        "workload_change": "workload",
        "error_change": "error",
        "latency50_change": "latency50",
        "latency90_change": "latency90",
    }

    features = {}

    for feature, family in family_map.items():

        columns = get_family_columns(
            metrics.columns,
            family
        )

        features[feature] = (
            strongest_service_change(
                pre,
                post,
                columns
            )
        )

    return features


# =============================================================================
# BUILD DATASET
# =============================================================================

def build_feature_dataset():
    """
    Build one row per valid incident.

    Ground-truth fault is attached only AFTER feature extraction.
    """

    cases = pd.read_parquet(
        CASES_FILE
    )

    cases = cases[
        cases["dataset"].isin(DATASETS)
    ].copy()

    cases = cases[
        cases["fault"].isin(FAULTS)
    ].copy()

    cases = cases.reset_index(drop=True)

    print("=" * 80)
    print("FINAL BAYESIAN RCA TRAINING PIPELINE")
    print("=" * 80)

    print(
        f"\nRE1 + RE2 metadata cases: {len(cases)}"
    )

    rows = []
    exclusions = []

    for i, (_, case) in enumerate(
        cases.iterrows(),
        start=1
    ):

        case_name = case["case"]

        telemetry_path = os.path.join(
            CACHE_DIR,
            f"{case_name}.parquet"
        )

        try:

            if not os.path.exists(telemetry_path):
                raise FileNotFoundError(
                    f"Telemetry file missing: {telemetry_path}"
                )

            metrics = pd.read_parquet(
                telemetry_path
            )

            # -------------------------------------------------------------
            # IMPORTANT:
            # No fault label is passed into extract_features().
            # -------------------------------------------------------------

            feature_values = extract_features(
                metrics,
                case["inject_time"]
            )

            row = {
                "case": case_name,
                "dataset": case["dataset"],
                "system": case["system_name"],
            }

            row.update(feature_values)

            # Attach target AFTER feature extraction.
            row["fault"] = case["fault"]

            rows.append(row)

        except Exception as exc:

            exclusions.append(
                {
                    "case": case_name,
                    "dataset": case["dataset"],
                    "fault": case["fault"],
                    "reason": str(exc),
                }
            )

        if i % 100 == 0:
            print(
                f"Processed {i}/{len(cases)}"
            )

    data = pd.DataFrame(rows)
    excluded = pd.DataFrame(exclusions)

    print(
        f"\nUsable incidents: {len(data)}"
    )

    print(
        f"Excluded incidents: {len(excluded)}"
    )

    if len(excluded) > 0:

        print("\nExcluded cases:")

        print(
            excluded.to_string(index=False)
        )

    return data, excluded


# =============================================================================
# STRATIFIED SPLIT
# =============================================================================

def stratified_split(data):
    """
    Deterministic class-wise 80/10/10 split.

    The split is performed BEFORE discretization.

    Quantile thresholds and Bayesian parameters are learned from TRAIN only.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    train_indices = []
    val_indices = []
    test_indices = []

    for fault in FAULTS:

        indices = data.index[
            data["fault"] == fault
        ].to_numpy()

        indices = indices.copy()

        rng.shuffle(indices)

        n = len(indices)

        n_train = int(
            math.floor(0.80 * n)
        )

        n_val = int(
            math.floor(0.10 * n)
        )

        train_indices.extend(
            indices[:n_train]
        )

        val_indices.extend(
            indices[
                n_train:n_train + n_val
            ]
        )

        test_indices.extend(
            indices[
                n_train + n_val:
            ]
        )

    train = (
        data.loc[train_indices]
        .sample(
            frac=1,
            random_state=RANDOM_SEED
        )
        .reset_index(drop=True)
    )

    validation = (
        data.loc[val_indices]
        .sample(
            frac=1,
            random_state=RANDOM_SEED
        )
        .reset_index(drop=True)
    )

    test = (
        data.loc[test_indices]
        .sample(
            frac=1,
            random_state=RANDOM_SEED
        )
        .reset_index(drop=True)
    )

    return train, validation, test


# =============================================================================
# TRAIN-ONLY DISCRETIZATION
# =============================================================================

def learn_thresholds(train):
    """
    Learn 33rd and 67th percentile boundaries from TRAINING DATA ONLY.

    These are unsupervised with respect to the class label.

    Missing values are ignored while estimating thresholds.
    """

    thresholds = {}

    for feature in FEATURES:

        values = pd.to_numeric(
            train[feature],
            errors="coerce"
        ).dropna()

        if len(values) == 0:

            thresholds[feature] = None
            continue

        q1 = float(
            values.quantile(1.0 / 3.0)
        )

        q2 = float(
            values.quantile(2.0 / 3.0)
        )

        thresholds[feature] = {
            "low_medium": q1,
            "medium_high": q2,
        }

    return thresholds


def discretize_value(value, threshold):
    """
    Convert a continuous telemetry change into:
        low / medium / high

    Missing input remains missing.
    """

    if pd.isna(value):
        return None

    if threshold is None:
        return None

    q1 = threshold["low_medium"]
    q2 = threshold["medium_high"]

    if value <= q1:
        return "low"

    if value <= q2:
        return "medium"

    return "high"


def discretize_dataframe(data, thresholds):
    """
    Add discrete state columns while retaining original continuous values.
    """

    result = data.copy()

    for feature in FEATURES:

        result[
            f"{feature}_state"
        ] = [
            discretize_value(
                value,
                thresholds[feature]
            )
            for value in result[feature]
        ]

    return result


# =============================================================================
# BAYESIAN MODEL TRAINING
# =============================================================================

def train_bayesian_model(train):
    """
    Learn:

        P(Fault)

    and

        P(FeatureState | Fault)

    using TRAINING DATA ONLY.

    Laplace smoothing prevents zero-probability likelihoods.
    """

    total = len(train)

    priors = {}

    cpts = {}

    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------

    for fault in FAULTS:

        count = int(
            (train["fault"] == fault).sum()
        )

        priors[fault] = (
            count + ALPHA
        ) / (
            total
            + ALPHA * len(FAULTS)
        )

    # -------------------------------------------------------------------------
    # CPTs
    # -------------------------------------------------------------------------

    for feature in FEATURES:

        state_column = (
            f"{feature}_state"
        )

        cpts[feature] = {}

        for fault in FAULTS:

            fault_rows = train[
                train["fault"] == fault
            ]

            # Only cases where this telemetry family was actually observed.
            observed = fault_rows[
                fault_rows[state_column].notna()
            ]

            observed_count = len(observed)

            cpts[feature][fault] = {}

            for state in STATES:

                state_count = int(
                    (
                        observed[state_column]
                        == state
                    ).sum()
                )

                probability = (
                    state_count + ALPHA
                ) / (
                    observed_count
                    + ALPHA * len(STATES)
                )

                cpts[feature][fault][state] = (
                    float(probability)
                )

    return {
        "priors": priors,
        "cpts": cpts,
    }


# =============================================================================
# BAYESIAN INFERENCE
# =============================================================================

def predict_probabilities(row, model):
    """
    Compute posterior scores:

        P(Fault | Evidence)
            proportional to
        P(Fault) * product P(Evidence_i | Fault)

    Missing telemetry contributes NO likelihood term.

    Log probabilities are used for numerical stability.
    """

    log_scores = {}

    for fault in FAULTS:

        prior = model[
            "priors"
        ][fault]

        log_score = math.log(
            max(prior, EPS)
        )

        for feature in FEATURES:

            state = row[
                f"{feature}_state"
            ]

            if pd.isna(state) or state is None:
                continue

            likelihood = (
                model["cpts"]
                [feature]
                [fault]
                [state]
            )

            log_score += math.log(
                max(likelihood, EPS)
            )

        log_scores[fault] = (
            log_score
        )

    # -------------------------------------------------------------------------
    # Normalize log scores into probabilities
    # -------------------------------------------------------------------------

    maximum = max(
        log_scores.values()
    )

    exp_scores = {
        fault: math.exp(
            score - maximum
        )
        for fault, score
        in log_scores.items()
    }

    denominator = sum(
        exp_scores.values()
    )

    probabilities = {
        fault: (
            exp_scores[fault]
            / denominator
        )
        for fault in FAULTS
    }

    return probabilities


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(data, model, split_name):
    """
    Evaluate one dataset split.
    """

    predictions = []

    confusion = pd.DataFrame(
        0,
        index=FAULTS,
        columns=FAULTS,
        dtype=int
    )

    reciprocal_ranks = []

    top1_correct = 0
    top3_correct = 0

    for _, row in data.iterrows():

        probabilities = (
            predict_probabilities(
                row,
                model
            )
        )

        ranking = sorted(
            probabilities,
            key=probabilities.get,
            reverse=True
        )

        actual = row["fault"]

        predicted = ranking[0]

        confusion.loc[
            actual,
            predicted
        ] += 1

        if predicted == actual:
            top1_correct += 1

        if actual in ranking[:3]:
            top3_correct += 1

        rank = (
            ranking.index(actual)
            + 1
        )

        reciprocal_ranks.append(
            1.0 / rank
        )

        prediction_row = {
            "case": row["case"],
            "dataset": row["dataset"],
            "system": row["system"],
            "actual": actual,
            "predicted": predicted,
            "rank": rank,
        }

        for fault in FAULTS:

            prediction_row[
                f"p_{fault}"
            ] = probabilities[fault]

        predictions.append(
            prediction_row
        )

    n = len(data)

    top1 = (
        top1_correct / n
        if n
        else 0.0
    )

    top3 = (
        top3_correct / n
        if n
        else 0.0
    )

    mrr = (
        float(
            np.mean(reciprocal_ranks)
        )
        if reciprocal_ranks
        else 0.0
    )

    # -------------------------------------------------------------------------
    # Per-class metrics
    # -------------------------------------------------------------------------

    per_class = {}

    for fault in FAULTS:

        tp = int(
            confusion.loc[
                fault,
                fault
            ]
        )

        fp = int(
            confusion[fault].sum()
            - tp
        )

        fn = int(
            confusion.loc[fault].sum()
            - tp
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
            if precision + recall > 0
            else 0.0
        )

        per_class[fault] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(
                confusion.loc[fault].sum()
            ),
        }

    print()
    print("=" * 80)
    print(
        f"{split_name.upper()} RESULTS"
    )
    print("=" * 80)

    print(
        f"Cases : {n}"
    )

    print(
        f"Top-1 : {top1 * 100:.1f}%"
    )

    print(
        f"Top-3 : {top3 * 100:.1f}%"
    )

    print(
        f"MRR   : {mrr:.3f}"
    )

    print(
        "\nConfusion matrix:"
    )

    display_confusion = (
        confusion.copy()
    )

    display_confusion.index.name = (
        "Actual"
    )

    display_confusion.columns.name = (
        "Predicted"
    )

    print(
        display_confusion.to_string()
    )

    print(
        "\nPer-class metrics:"
    )

    for fault in FAULTS:

        metrics = per_class[fault]

        print(
            f"{fault.upper():7s} "
            f"P={metrics['precision']:.3f}  "
            f"R={metrics['recall']:.3f}  "
            f"F1={metrics['f1']:.3f}  "
            f"N={metrics['support']}"
        )

    report = {
        "split": split_name,
        "cases": n,
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "mrr": mrr,
        "confusion_matrix": {
            actual: {
                predicted: int(
                    confusion.loc[
                        actual,
                        predicted
                    ]
                )
                for predicted in FAULTS
            }
            for actual in FAULTS
        },
        "per_class": per_class,
    }

    return (
        report,
        pd.DataFrame(predictions)
    )


# =============================================================================
# JSON SERIALIZATION
# =============================================================================

def make_json_safe(obj):
    """
    Convert NumPy values into normal Python values.
    """

    if isinstance(
        obj,
        (np.integer,)
    ):
        return int(obj)

    if isinstance(
        obj,
        (np.floating,)
    ):
        return float(obj)

    if isinstance(
        obj,
        np.ndarray
    ):
        return obj.tolist()

    raise TypeError(
        f"Object of type {type(obj)} "
        f"is not JSON serializable"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # 1. Feature extraction
    # -------------------------------------------------------------------------

    data, exclusions = (
        build_feature_dataset()
    )

    if len(data) == 0:
        raise RuntimeError(
            "No usable incidents."
        )

    print(
        "\nUsable fault distribution:"
    )

    print(
        data["fault"]
        .value_counts()
        .reindex(FAULTS)
        .to_string()
    )

    # -------------------------------------------------------------------------
    # 2. Split BEFORE any learned preprocessing
    # -------------------------------------------------------------------------

    train, validation, test = (
        stratified_split(data)
    )

    print()
    print("=" * 80)
    print("DATA SPLIT")
    print("=" * 80)

    print(
        f"Training   : {len(train)}"
    )

    print(
        f"Validation : {len(validation)}"
    )

    print(
        f"Test       : {len(test)}"
    )

    print(
        "\nSplit by fault:"
    )

    split_counts = pd.DataFrame({
        "train":
            train["fault"]
            .value_counts()
            .reindex(FAULTS)
            .fillna(0)
            .astype(int),

        "validation":
            validation["fault"]
            .value_counts()
            .reindex(FAULTS)
            .fillna(0)
            .astype(int),

        "test":
            test["fault"]
            .value_counts()
            .reindex(FAULTS)
            .fillna(0)
            .astype(int),
    })

    print(
        split_counts.to_string()
    )

    # -------------------------------------------------------------------------
    # 3. Learn thresholds from TRAIN ONLY
    # -------------------------------------------------------------------------

    thresholds = learn_thresholds(
        train
    )

    print()
    print("=" * 80)
    print(
        "TRAIN-ONLY DISCRETIZATION THRESHOLDS"
    )
    print("=" * 80)

    for feature in FEATURES:

        threshold = thresholds[
            feature
        ]

        if threshold is None:

            print(
                f"{feature:20s} unavailable"
            )

        else:

            print(
                f"{feature:20s} "
                f"{threshold['low_medium']:+.4f}  "
                f"{threshold['medium_high']:+.4f}"
            )

    # -------------------------------------------------------------------------
    # 4. Apply frozen thresholds
    # -------------------------------------------------------------------------

    train_discrete = (
        discretize_dataframe(
            train,
            thresholds
        )
    )

    validation_discrete = (
        discretize_dataframe(
            validation,
            thresholds
        )
    )

    test_discrete = (
        discretize_dataframe(
            test,
            thresholds
        )
    )

    # -------------------------------------------------------------------------
    # 5. Learn Bayesian parameters
    # -------------------------------------------------------------------------

    model = train_bayesian_model(
        train_discrete
    )

    for fault in FAULTS:

        print(
            f"{fault:7s}: "
            f"{model['priors'][fault]:.4f}"
        )

    # -------------------------------------------------------------------------
    # 6. Evaluate
    # -------------------------------------------------------------------------

    train_report, train_predictions = (
        evaluate(
            train_discrete,
            model,
            "training"
        )
    )

    validation_report, validation_predictions = (
        evaluate(
            validation_discrete,
            model,
            "validation"
        )
    )

    # IMPORTANT:
    # This test result is printed for the current final candidate.
    # If the model is subsequently redesigned using these results,
    # this test set must no longer be described as untouched.
    test_report, test_predictions = (
        evaluate(
            test_discrete,
            model,
            "test"
        )
    )

    # -------------------------------------------------------------------------
    # 7. Save reproducibility artifacts
    # -------------------------------------------------------------------------

    data.to_csv(
        "bayes_rca_features.csv",
        index=False
    )

    exclusions.to_csv(
        "bayes_rca_exclusions.csv",
        index=False
    )

    split_export = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(
                split="validation"
            ),
            test.assign(split="test"),
        ],
        ignore_index=True
    )

    split_export.to_csv(
        "bayes_rca_split.csv",
        index=False
    )

    train_predictions.to_csv(
        "bayes_rca_train_predictions.csv",
        index=False
    )

    validation_predictions.to_csv(
        "bayes_rca_validation_predictions.csv",
        index=False
    )

    test_predictions.to_csv(
        "bayes_rca_test_predictions.csv",
        index=False
    )

    model_export = {
        "model_type":
            "Discrete Naive Bayes",

        "classes":
            FAULTS,

        "features":
            FEATURES,

        "states":
            STATES,

        "laplace_alpha":
            ALPHA,

        "random_seed":
            RANDOM_SEED,

        "change_formula":
            "(post-pre)/(abs(post)+abs(pre)+epsilon)",

        "missing_evidence":
            "ignored during Bayesian inference",

        "thresholds":
            thresholds,

        "priors":
            model["priors"],

        "cpts":
            model["cpts"],
    }

    with open(
        "bayes_rca_model.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            model_export,
            f,
            indent=2,
            default=make_json_safe
        )

    # ============================================================
    # EXPORT FINAL TRAINED MODEL TO JAVASCRIPT FOR THE WEB APP
    # ============================================================

    # bayes_rca_model.json has already been saved above.
    # Read that exact saved model back instead of making assumptions
    # about the structure of the in-memory `model` object.
    with open(
        "bayes_rca_model.json",
        "r",
        encoding="utf-8"
    ) as f:
        saved_model = json.load(f)

    web_model = {
        "metadata": {
            "modelType": "Discrete Naive Bayes",
            "dataset": "RCAEval RE1 + RE2",

            "selectedCases": 645,
            "usableCases": 643,
            "excludedCases": 2,

            "trainCases": 514,
            "validationCases": 62,
            "testCases": 67,

            "trainTop1": 71.0,
            "trainTop3": 97.3,
            "trainMRR": 0.836,

            "validationTop1": 59.7,
            "validationTop3": 95.2,
            "validationMRR": 0.773,

            "testTop1": 70.1,
            "testTop3": 100.0,
            "testMRR": 0.841,

            "randomSeed": 2026
        },

        "modelType": saved_model["model_type"],
        "classes": saved_model["classes"],
        "features": saved_model["features"],
        "states": saved_model["states"],

        "thresholds": saved_model["thresholds"],
        "priors": saved_model["priors"],
        "cpts": saved_model["cpts"],

        "laplaceAlpha": saved_model["laplace_alpha"],
        "randomSeed": saved_model["random_seed"],

        "changeFormula": saved_model["change_formula"],
        "missingEvidence": saved_model["missing_evidence"]
    }

    with open(
        "trained_model_params.js",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "/**\n"
            " * BayesRCA - Frozen Web Model\n"
            " *\n"
            " * AUTO-GENERATED by train_bayes_rca.py.\n"
            " * Source: bayes_rca_model.json\n"
            " *\n"
            " * The thresholds, priors and CPTs below are copied\n"
            " * directly from the trained Bayesian model.\n"
            " * Do not manually edit learned parameters.\n"
            " */\n\n"
        )

        f.write(
            "export const TRAINED_MODEL_PARAMS = "
        )

        json.dump(
            web_model,
            f,
            indent=2
        )

        f.write(";\n")

    print("Saved: trained_model_params.js")



if __name__ == "__main__":
    main()