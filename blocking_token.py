import pandas as pd
from collections import defaultdict, Counter
from normalize import normalize_business_name


TRAIN = "dataset/train"
SAMPLE_SIZE = 100_000

# Tokens appearing in more than this many records
# are ignored because they create huge candidate sets.
MAX_TOKEN_FREQUENCY = 500


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    f"{TRAIN}/train_source1.tsv",
    sep="\t",
    nrows=SAMPLE_SIZE,
    usecols=["entity_id", "business_name", "country"]
)

print("Source 1 loaded:", len(s1))


print("Loading Source 2...")

s2 = pd.read_csv(
    f"{TRAIN}/train_source2.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "country"]
)

print("Source 2 loaded:", len(s2))


print("Loading Source 3...")

s3 = pd.read_csv(
    f"{TRAIN}/train_source3.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "country"]
)

print("Source 3 loaded:", len(s3))


print("Loading ground truth...")

gt = pd.read_csv(
    f"{TRAIN}/train_ground_truth.tsv",
    sep="\t",
    usecols=["source1_entity_id", "matched_entity_ids"]
)


# ============================================================
# 2. GROUND TRUTH LOOKUP
# ============================================================

print("Creating ground-truth lookup...")

sample_ids = set(s1["entity_id"])

gt_sample = gt[
    gt["source1_entity_id"].isin(sample_ids)
]

ground_truth = {}

for _, row in gt_sample.iterrows():

    value = row["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        ground_truth[row["source1_entity_id"]] = set()
    else:
        ground_truth[row["source1_entity_id"]] = set(
            str(value).split(",")
        )


# ============================================================
# 3. NORMALIZATION
# ============================================================

print("\nNormalizing Source 1...")

s1["name_norm"] = (
    s1["business_name"]
    .fillna("")
    .map(normalize_business_name)
)

print("Source 1 done.")


print("Normalizing Source 2...")

s2["name_norm"] = (
    s2["business_name"]
    .fillna("")
    .map(normalize_business_name)
)

print("Source 2 done.")


print("Normalizing Source 3...")

s3["name_norm"] = (
    s3["business_name"]
    .fillna("")
    .map(normalize_business_name)
)

print("Source 3 done.")


# ============================================================
# 4. EXACT NAME INDEX
# ============================================================

print("\nBuilding exact-name indexes...")

exact_index = defaultdict(set)

for country, name, entity_id in zip(
    s2["country"],
    s2["name_norm"],
    s2["entity_id"]
):

    if name:
        exact_index[(country, name)].add(entity_id)


for country, name, entity_id in zip(
    s3["country"],
    s3["name_norm"],
    s3["entity_id"]
):

    if name:
        exact_index[(country, name)].add(entity_id)


print("Exact index keys:", len(exact_index))


# ============================================================
# 5. COUNT TOKEN FREQUENCIES
# ============================================================

print("\nCounting token frequencies...")

token_frequency = Counter()


def get_tokens(name):

    return {
        token
        for token in name.split()
        if len(token) >= 4
    }


for name in s2["name_norm"]:

    if name:
        tokens = get_tokens(name)

        # Count once per record
        for token in tokens:
            token_frequency[token] += 1


for name in s3["name_norm"]:

    if name:
        tokens = get_tokens(name)

        for token in tokens:
            token_frequency[token] += 1


print(
    "Unique tokens:",
    len(token_frequency)
)


# ============================================================
# 6. BUILD FILTERED TOKEN INDEX
# ============================================================

print("\nBuilding filtered token index...")

token_index = defaultdict(set)


for country, name, entity_id in zip(
    s2["country"],
    s2["name_norm"],
    s2["entity_id"]
):

    if not name:
        continue

    tokens = get_tokens(name)

    for token in tokens:

        if token_frequency[token] <= MAX_TOKEN_FREQUENCY:

            token_index[
                (country, token)
            ].add(entity_id)


for country, name, entity_id in zip(
    s3["country"],
    s3["name_norm"],
    s3["entity_id"]
):

    if not name:
        continue

    tokens = get_tokens(name)

    for token in tokens:

        if token_frequency[token] <= MAX_TOKEN_FREQUENCY:

            token_index[
                (country, token)
            ].add(entity_id)


print(
    "Filtered token keys:",
    len(token_index)
)


# ============================================================
# 7. BLOCKING TEST
# ============================================================

print("\n" + "=" * 70)
print("FAST TOKEN BLOCKING RECALL TEST")
print("=" * 70)


total_true_matches = 0
retrieved_true_matches = 0

s1_with_true_matches = 0
s1_with_retrieved_true_match = 0

total_candidates = 0
s1_with_candidates = 0


# ============================================================
# 8. TEST EACH S1
# ============================================================

for i, row in enumerate(
    s1.itertuples(index=False),
    start=1
):

    country = row.country
    name = row.name_norm
    s1_id = row.entity_id


    # --------------------------------------------------------
    # Candidate set
    # --------------------------------------------------------

    candidates = set()


    # Exact normalized name candidate
    if name:

        candidates.update(
            exact_index.get(
                (country, name),
                set()
            )
        )


    # Token candidates
    if name:

        tokens = get_tokens(name)

        for token in tokens:

            # Ignore common tokens
            if token_frequency[token] <= MAX_TOKEN_FREQUENCY:

                candidates.update(
                    token_index.get(
                        (country, token),
                        set()
                    )
                )


    total_candidates += len(candidates)


    if candidates:
        s1_with_candidates += 1


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    true_matches = ground_truth.get(
        s1_id,
        set()
    )


    if true_matches:

        s1_with_true_matches += 1

        total_true_matches += len(true_matches)

        retrieved = (
            true_matches &
            candidates
        )

        retrieved_true_matches += len(retrieved)

        if retrieved:
            s1_with_retrieved_true_match += 1


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if i % 10000 == 0:

        print(
            f"Processed {i:,}/{len(s1):,} "
            f"S1 records..."
        )


# ============================================================
# 9. RESULTS
# ============================================================

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


# ============================================================
# 10. RECALL
# ============================================================

if total_true_matches > 0:

    pair_recall = (
        retrieved_true_matches
        / total_true_matches
        * 100
    )

    print(
        "\nPAIR-LEVEL BLOCKING RECALL:",
        round(pair_recall, 2),
        "%"
    )


if s1_with_true_matches > 0:

    entity_recall = (
        s1_with_retrieved_true_match
        / s1_with_true_matches
        * 100
    )

    print(
        "ENTITY-LEVEL BLOCKING RECALL:",
        round(entity_recall, 2),
        "%"
    )


# ============================================================
# 11. CANDIDATE STATISTICS
# ============================================================

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
            total_candidates /
            s1_with_candidates,
            2
        )
    )


print("\nDone.")