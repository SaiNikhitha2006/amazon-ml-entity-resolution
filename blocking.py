import pandas as pd
from collections import defaultdict
from normalize import normalize_business_name


TRAIN = "dataset/train"

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

# Only S1 is sampled.
# We still need all S2/S3 records to find candidates.
SAMPLE_SIZE = 100_000


# --------------------------------------------------
# 1. LOAD SOURCE 1 SAMPLE
# --------------------------------------------------

print("Loading Source 1...")

s1 = pd.read_csv(
    f"{TRAIN}/train_source1.tsv",
    sep="\t",
    nrows=SAMPLE_SIZE,
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

print("Source 1 loaded:", len(s1))


# --------------------------------------------------
# 2. LOAD SOURCE 2
# --------------------------------------------------

print("Loading Source 2...")

s2 = pd.read_csv(
    f"{TRAIN}/train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

print("Source 2 loaded:", len(s2))


# --------------------------------------------------
# 3. LOAD SOURCE 3
# --------------------------------------------------

print("Loading Source 3...")

s3 = pd.read_csv(
    f"{TRAIN}/train_source3.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

print("Source 3 loaded:", len(s3))


# --------------------------------------------------
# 4. LOAD GROUND TRUTH
# --------------------------------------------------

print("Loading ground truth...")

gt = pd.read_csv(
    f"{TRAIN}/train_ground_truth.tsv",
    sep="\t",
    usecols=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)

print("Ground truth loaded:", len(gt))


# --------------------------------------------------
# 5. CREATE FAST GROUND-TRUTH LOOKUP
# --------------------------------------------------

print("Creating ground-truth lookup...")

# Only keep ground truth for our sampled S1 records
sample_ids = set(s1["entity_id"])

gt_sample = gt[
    gt["source1_entity_id"].isin(sample_ids)
].copy()

# Convert ground truth into:
#
# S1_ID -> set of true matching IDs
#
# Example:
#
# S1-123 -> {S2-111, S3-555, S3-777}

ground_truth = {}

for _, row in gt_sample.iterrows():

    s1_id = row["source1_entity_id"]

    value = row["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        ground_truth[s1_id] = set()

    else:
        ground_truth[s1_id] = set(
            str(value).split(",")
        )


print(
    "Ground-truth records for sample:",
    len(ground_truth)
)


# --------------------------------------------------
# 6. NORMALIZE BUSINESS NAMES
# --------------------------------------------------

print("\nNormalizing Source 1 names...")

s1["name_norm"] = (
    s1["business_name"]
    .fillna("")
    .map(normalize_business_name)
)

print("Source 1 names normalized.")


print("\nNormalizing Source 2 names...")

s2["name_norm"] = (
    s2["business_name"]
    .fillna("")
    .map(normalize_business_name)
)

print("Source 2 names normalized.")


print("\nNormalizing Source 3 names...")

s3["name_norm"] = (
    s3["business_name"]
    .fillna("")
    .map(normalize_business_name)
)

print("Source 3 names normalized.")


# --------------------------------------------------
# 7. BUILD SOURCE 2 BLOCKING INDEX
# --------------------------------------------------

print("\nBuilding Source 2 index...")

s2_index = defaultdict(set)

for country, name, entity_id in zip(
    s2["country"],
    s2["name_norm"],
    s2["entity_id"]
):

    if name:

        key = (
            country,
            name
        )

        s2_index[key].add(entity_id)


print(
    "Source 2 index keys:",
    len(s2_index)
)


# --------------------------------------------------
# 8. BUILD SOURCE 3 BLOCKING INDEX
# --------------------------------------------------

print("\nBuilding Source 3 index...")

s3_index = defaultdict(set)

for country, name, entity_id in zip(
    s3["country"],
    s3["name_norm"],
    s3["entity_id"]
):

    if name:

        key = (
            country,
            name
        )

        s3_index[key].add(entity_id)


print(
    "Source 3 index keys:",
    len(s3_index)
)


# --------------------------------------------------
# 9. EVALUATE BLOCKING
# --------------------------------------------------

print("\n" + "=" * 70)
print("BLOCKING RECALL TEST")
print("=" * 70)


total_true_matches = 0
retrieved_true_matches = 0

s1_with_true_matches = 0
s1_with_retrieved_true_match = 0

total_candidates = 0
s1_with_candidates = 0


for country, name, s1_id in zip(
    s1["country"],
    s1["name_norm"],
    s1["entity_id"]
):

    # ----------------------------------------------
    # Generate candidates
    # ----------------------------------------------

    key = (
        country,
        name
    )

    candidates_s2 = s2_index.get(
        key,
        set()
    )

    candidates_s3 = s3_index.get(
        key,
        set()
    )

    candidates = (
        candidates_s2 |
        candidates_s3
    )

    total_candidates += len(candidates)

    if len(candidates) > 0:
        s1_with_candidates += 1


    # ----------------------------------------------
    # Get ground truth
    # ----------------------------------------------

    true_matches = ground_truth.get(
        s1_id,
        set()
    )


    # ----------------------------------------------
    # Skip singleton/no-match records
    # ----------------------------------------------

    if len(true_matches) == 0:
        continue


    s1_with_true_matches += 1

    total_true_matches += len(true_matches)


    # ----------------------------------------------
    # Check how many true matches were retrieved
    # ----------------------------------------------

    retrieved = (
        true_matches &
        candidates
    )

    retrieved_true_matches += len(retrieved)


    if len(retrieved) > 0:
        s1_with_retrieved_true_match += 1


# --------------------------------------------------
# 10. RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)


print(
    "\nS1 records evaluated:",
    len(s1)
)


print(
    "S1 records with candidates:",
    s1_with_candidates
)


print(
    "S1 records with true matches:",
    s1_with_true_matches
)


print(
    "Total true matches:",
    total_true_matches
)


print(
    "True matches retrieved:",
    retrieved_true_matches
)


# --------------------------------------------------
# PAIR-LEVEL RECALL
# --------------------------------------------------

if total_true_matches > 0:

    pair_recall = (
        retrieved_true_matches
        /
        total_true_matches
        *
        100
    )

    print(
        "\nPAIR-LEVEL BLOCKING RECALL:",
        round(pair_recall, 2),
        "%"
    )


# --------------------------------------------------
# ENTITY-LEVEL RECALL
# --------------------------------------------------

if s1_with_true_matches > 0:

    entity_recall = (
        s1_with_retrieved_true_match
        /
        s1_with_true_matches
        *
        100
    )

    print(
        "ENTITY-LEVEL BLOCKING RECALL:",
        round(entity_recall, 2),
        "%"
    )


# --------------------------------------------------
# CANDIDATE STATISTICS
# --------------------------------------------------

print(
    "\nTotal generated candidates:",
    total_candidates
)


print(
    "Average candidates per S1:",
    round(
        total_candidates / len(s1),
        2
    )
)


if s1_with_candidates > 0:

    print(
        "Average candidates per S1 with candidates:",
        round(
            total_candidates / s1_with_candidates,
            2
        )
    )


print("\nDone.")