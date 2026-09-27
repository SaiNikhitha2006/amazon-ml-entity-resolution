import pandas as pd
import numpy as np

INPUT_FILE = "candidate_features_memory_safe.csv"

print("Loading candidate features...")
df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))

# ------------------------------------------------------------
# Features
# ------------------------------------------------------------

features = [
    "name_exact",
    "address_exact",
    "name_overlap",
    "address_overlap",
    "name_jaccard",
    "address_jaccard"
]

print("\nChecking columns...")

missing = [f for f in features if f not in df.columns]

if missing:
    print("Missing columns:", missing)
    print("Available columns:")
    print(df.columns.tolist())
    raise SystemExit

# ------------------------------------------------------------
# Make sure numeric
# ------------------------------------------------------------

for f in features:
    df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0)

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
).fillna(0).astype(int)

# ------------------------------------------------------------
# Normalize overlap features
# ------------------------------------------------------------

# Cap token overlaps so extremely large overlaps
# don't dominate the score.

df["name_overlap_norm"] = (
    df["name_overlap"].clip(0, 8) / 8.0
)

df["address_overlap_norm"] = (
    df["address_overlap"].clip(0, 20) / 20.0
)

# ------------------------------------------------------------
# Different scoring combinations
# ------------------------------------------------------------

weights = {

    "Score_A": {
        "name_exact": 5,
        "address_exact": 5,
        "name_overlap_norm": 2,
        "address_overlap_norm": 2,
        "name_jaccard": 5,
        "address_jaccard": 5
    },

    "Score_B": {
        "name_exact": 8,
        "address_exact": 8,
        "name_overlap_norm": 2,
        "address_overlap_norm": 2,
        "name_jaccard": 6,
        "address_jaccard": 6
    },

    "Score_C": {
        "name_exact": 10,
        "address_exact": 10,
        "name_overlap_norm": 1,
        "address_overlap_norm": 1,
        "name_jaccard": 7,
        "address_jaccard": 7
    },

    "Score_D": {
        "name_exact": 12,
        "address_exact": 12,
        "name_overlap_norm": 2,
        "address_overlap_norm": 2,
        "name_jaccard": 8,
        "address_jaccard": 8
    },

    "Score_E": {
        "name_exact": 15,
        "address_exact": 15,
        "name_overlap_norm": 1,
        "address_overlap_norm": 1,
        "name_jaccard": 10,
        "address_jaccard": 10
    }
}

# ------------------------------------------------------------
# Evaluate a score
# ------------------------------------------------------------

def evaluate(score_name, w):

    score = np.zeros(len(df), dtype=np.float32)

    for feature, weight in w.items():
        score += (
            df[feature].to_numpy(dtype=np.float32)
            * weight
        )

    temp = df[
        [
            "source1_entity_id",
            "label"
        ]
    ].copy()

    temp["score"] = score

    # Sort candidates within each S1
    temp = temp.sort_values(
        [
            "source1_entity_id",
            "score"
        ],
        ascending=[
            True,
            False
        ]
    )

    # --------------------------------------------------------
    # Top K
    # --------------------------------------------------------

    results = {}

    for k in [1, 3, 5, 10, 20]:

        topk = (
            temp
            .groupby("source1_entity_id")
            .head(k)
        )

        true_found = int(topk["label"].sum())

        total_true = int(df["label"].sum())

        recall = (
            true_found / total_true * 100
            if total_true > 0
            else 0
        )

        results[k] = recall

    return results


# ------------------------------------------------------------
# Run all scoring methods
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MATCHING SCORE OPTIMIZATION")
print("=" * 70)

all_results = {}

for name, w in weights.items():

    print("\nTesting:", name)

    result = evaluate(name, w)

    all_results[name] = result

    print(
        f"Top-1  : {result[1]:.2f}%"
    )

    print(
        f"Top-3  : {result[3]:.2f}%"
    )

    print(
        f"Top-5  : {result[5]:.2f}%"
    )

    print(
        f"Top-10 : {result[10]:.2f}%"
    )

    print(
        f"Top-20 : {result[20]:.2f}%"
    )


# ------------------------------------------------------------
# Summary table
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

summary = []

for name, result in all_results.items():

    summary.append({
        "Score": name,
        "Top1": result[1],
        "Top3": result[3],
        "Top5": result[5],
        "Top10": result[10],
        "Top20": result[20]
    })

summary_df = pd.DataFrame(summary)

print(
    summary_df.to_string(
        index=False,
        formatters={
            "Top1": "{:.2f}%".format,
            "Top3": "{:.2f}%".format,
            "Top5": "{:.2f}%".format,
            "Top10": "{:.2f}%".format,
            "Top20": "{:.2f}%".format
        }
    )
)

# ------------------------------------------------------------
# Best score according to Top-10 recall
# ------------------------------------------------------------

best = summary_df.loc[
    summary_df["Top10"].idxmax()
]

print("\n" + "=" * 70)
print("BEST SCORE")
print("=" * 70)

print(
    "Method :", best["Score"]
)

print(
    f"Top-1  : {best['Top1']:.2f}%"
)

print(
    f"Top-3  : {best['Top3']:.2f}%"
)

print(
    f"Top-5  : {best['Top5']:.2f}%"
)

print(
    f"Top-10 : {best['Top10']:.2f}%"
)

print(
    f"Top-20 : {best['Top20']:.2f}%"
)

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

summary_df.to_csv(
    "score_optimization_results.csv",
    index=False
)

print(
    "\nSaved: score_optimization_results.csv"
)

print("\nDone.")