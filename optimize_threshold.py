import pandas as pd
import numpy as np

FEATURE_FILE = "candidate_features_memory_safe.csv"

print("=" * 70)
print("F0.5 THRESHOLD OPTIMIZATION")
print("=" * 70)

print("\nLoading candidate features...")

df = pd.read_csv(FEATURE_FILE)

print("Rows:", len(df))
print("Columns:")
print(df.columns.tolist())


# ============================================================
# FIND IMPORTANT COLUMNS
# ============================================================

label_candidates = [
    "label",
    "is_match",
    "match",
    "target",
    "y"
]

s1_candidates = [
    "source1_entity_id",
    "s1_entity_id",
    "source1_id"
]

target_candidates = [
    "candidate_entity_id",
    "entity_id",
    "target_entity_id"
]


label_col = None
for c in label_candidates:
    if c in df.columns:
        label_col = c
        break


s1_col = None
for c in s1_candidates:
    if c in df.columns:
        s1_col = c
        break


target_col = None
for c in target_candidates:
    if c in df.columns:
        target_col = c
        break


print("\nDetected columns:")
print("S1 column    :", s1_col)
print("Target column:", target_col)
print("Label column :", label_col)


if label_col is None:
    raise ValueError(
        "Could not find the match label column."
    )

if s1_col is None:
    raise ValueError(
        "Could not find the Source 1 ID column."
    )


# ============================================================
# SCORE-E
# ============================================================

def score_e(df):

    score = (
        15 * df["name_exact"]
        + 15 * df["address_exact"]
        + 1 * np.minimum(df["name_overlap"], 8) / 8
        + 1 * np.minimum(df["address_overlap"], 20) / 20
        + 10 * df["name_jaccard"]
        + 10 * df["address_jaccard"]
    )

    return score


print("\nCalculating Score-E...")

df["score"] = score_e(df)

df[label_col] = df[label_col].astype(int)


# ============================================================
# MACRO F0.5
# ============================================================

def f05(precision, recall):

    if precision == 0 and recall == 0:
        return 0.0

    return (
        1.25 * precision * recall
        / (0.25 * precision + recall)
    )


def evaluate_threshold(threshold):

    total_f05 = 0.0
    entities = 0

    for s1_id, group in df.groupby(s1_col, sort=False):

        true_matches = set(
            group.loc[
                group[label_col] == 1,
                target_col
            ]
        )

        predicted_matches = set(
            group.loc[
                group["score"] >= threshold,
                target_col
            ]
        )

        tp = len(true_matches & predicted_matches)

        fp = len(predicted_matches - true_matches)

        fn = len(true_matches - predicted_matches)

        if len(predicted_matches) == 0:
            precision = 1.0 if len(true_matches) == 0 else 0.0
        else:
            precision = tp / len(predicted_matches)

        if len(true_matches) == 0:
            recall = 1.0 if len(predicted_matches) == 0 else 0.0
        else:
            recall = tp / len(true_matches)

        score = f05(
            precision,
            recall
        )

        total_f05 += score
        entities += 1

    return total_f05 / entities


# ============================================================
# TEST THRESHOLDS
# ============================================================

thresholds = [
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    34,
    36,
    38,
    40,
    42,
    45,
    48,
    50
]


results = []


print("\n" + "=" * 70)
print("TESTING THRESHOLDS")
print("=" * 70)

for threshold in thresholds:

    score = evaluate_threshold(
        threshold
    )

    results.append(
        {
            "threshold": threshold,
            "F0.5": score
        }
    )

    print(
        f"Threshold {threshold:>2} "
        f"-> F0.5 = {score:.6f}"
    )


# ============================================================
# BEST THRESHOLD
# ============================================================

results_df = pd.DataFrame(results)

best = results_df.loc[
    results_df["F0.5"].idxmax()
]


print("\n" + "=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    "Threshold:",
    int(best["threshold"])
)

print(
    "F0.5:",
    f"{best['F0.5']:.6f}"
)


results_df.to_csv(
    "threshold_optimization_results.csv",
    index=False
)

print(
    "\nSaved:",
    "threshold_optimization_results.csv"
)

print("\nDone.")