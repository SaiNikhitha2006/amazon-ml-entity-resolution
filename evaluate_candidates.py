import pandas as pd

FEATURE_FILE = "candidate_features_memory_safe.csv"

print("Loading generated features...")

df = pd.read_csv(FEATURE_FILE)

print("Rows:", len(df))

print("\n" + "=" * 70)
print("CANDIDATE FEATURE EVALUATION")
print("=" * 70)

# ------------------------------------------------------------
# Positive / negative pairs
# ------------------------------------------------------------

positive = df[df["label"] == 1]

negative = df[df["label"] == 0]

print("\nPositive candidate pairs:", len(positive))
print("Negative candidate pairs:", len(negative))

# ------------------------------------------------------------
# S1 coverage
# ------------------------------------------------------------

total_s1 = df["source1_entity_id"].nunique()

s1_with_positive = positive["source1_entity_id"].nunique()

print("\nS1 records represented:", total_s1)
print("S1 records with true candidates:", s1_with_positive)

# ------------------------------------------------------------
# Candidate recall
# ------------------------------------------------------------

# Number of true pairs available for these 20K S1 records
# comes from the label column.

true_pairs = len(positive)

print("\nTrue matches retrieved:", true_pairs)

# ------------------------------------------------------------
# Feature statistics
# ------------------------------------------------------------

print("\nFeature statistics:")

features = [
    "country_match",
    "name_exact",
    "address_exact",
    "name_overlap",
    "address_overlap",
    "name_jaccard",
    "address_jaccard"
]

print(
    df[features].describe()
)

# ------------------------------------------------------------
# Positive feature statistics
# ------------------------------------------------------------

print("\nPositive-match feature statistics:")

print(
    positive[features].describe()
)

# ------------------------------------------------------------
# Negative feature statistics
# ------------------------------------------------------------

print("\nNegative-match feature statistics:")

print(
    negative[features].describe()
)

# ------------------------------------------------------------
# Strong matches
# ------------------------------------------------------------

strong = df[
    (
        (df["name_exact"] == 1) |
        (df["address_exact"] == 1)
    )
]

print(
    "\nCandidates having exact name OR exact address:",
    len(strong)
)

print(
    "Of these, true matches:",
    int(strong["label"].sum())
)

# ------------------------------------------------------------
# Combined score
# ------------------------------------------------------------

df["score"] = (

    df["country_match"] * 1.0

    + df["name_exact"] * 5.0

    + df["address_exact"] * 5.0

    + df["name_overlap"] * 1.0

    + df["address_overlap"] * 1.0

    + df["name_jaccard"] * 3.0

    + df["address_jaccard"] * 3.0
)

# ------------------------------------------------------------
# Top candidates per S1
# ------------------------------------------------------------

top = (
    df.sort_values(
        [
            "source1_entity_id",
            "score"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby("source1_entity_id")
    .head(10)
)

top_true = int(top["label"].sum())

print(
    "\nTrue matches inside top 10 candidates per S1:",
    top_true
)

print(
    "Top-10 candidate recall:",
    round(
        top_true / true_pairs * 100,
        2
    ),
    "%"
)

# ------------------------------------------------------------
# Top 5
# ------------------------------------------------------------

top5 = (
    df.sort_values(
        [
            "source1_entity_id",
            "score"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby("source1_entity_id")
    .head(5)
)

top5_true = int(top5["label"].sum())

print(
    "\nTrue matches inside top 5 candidates per S1:",
    top5_true
)

print(
    "Top-5 candidate recall:",
    round(
        top5_true / true_pairs * 100,
        2
    ),
    "%"
)

print("\nDone.")