import pandas as pd
import re
from collections import defaultdict
from normalize import normalize_business_name


TRAIN = "dataset/train"
SAMPLE_SIZE = 100_000

# Maximum number of candidates allowed from one n-gram
MAX_NGRAM_FREQUENCY = 300

# Character n-gram size
N = 3


# ============================================================
# CHARACTER N-GRAMS
# ============================================================

def get_ngrams(text):
    if not text:
        return set()

    # Remove spaces so spelling similarity is captured
    text = re.sub(r"[^a-z0-9]", "", text)

    if len(text) < N:
        return {text}

    return {
        text[i:i + N]
        for i in range(len(text) - N + 1)
    }


# ============================================================
# LOAD DATA
# ============================================================

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


print("Loading ground truth...")

gt = pd.read_csv(
    f"{TRAIN}/train_ground_truth.tsv",
    sep="\t",
    usecols=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)


# ============================================================
# GROUND TRUTH
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

print(
    "Ground-truth records:",
    len(ground_truth)
)


# ============================================================
# NORMALIZE NAMES
# ============================================================

print("\nNormalizing Source 1 names...")

s1["name_norm"] = (
    s1["business_name"]
    .fillna("")
    .map(normalize_business_name)
)

print("Source 1 done.")


print("Normalizing Source 2 names...")

s2["name_norm"] = (
    s2["business_name"]
    .fillna("")
    .map(normalize_business_name)
)

print("Source 2 done.")


print("Normalizing Source 3 names...")

s3["name_norm"] = (
    s3["business_name"]
    .fillna("")
    .map(normalize_business_name)
)

print("Source 3 done.")


# ============================================================
# BUILD N-GRAM FREQUENCY
# ============================================================

print("\nCounting character n-gram frequencies...")

ngram_frequency = defaultdict(int)


def count_ngrams(name):

    # Count each n-gram once per record
    for gram in get_ngrams(name):
        ngram_frequency[gram] += 1


for name in s2["name_norm"]:
    count_ngrams(name)


for name in s3["name_norm"]:
    count_ngrams(name)


print(
    "Unique character n-grams:",
    len(ngram_frequency)
)


# ============================================================
# BUILD FILTERED N-GRAM INDEX
# ============================================================

print("\nBuilding Source 2 character index...")

ngram_index = defaultdict(set)


for country, name, entity_id in zip(
    s2["country"],
    s2["name_norm"],
    s2["entity_id"]
):

    for gram in get_ngrams(name):

        if ngram_frequency[gram] <= MAX_NGRAM_FREQUENCY:

            ngram_index[
                (country, gram)
            ].add(entity_id)


print("Source 2/3 character index building...")


for country, name, entity_id in zip(
    s3["country"],
    s3["name_norm"],
    s3["entity_id"]
):

    for gram in get_ngrams(name):

        if ngram_frequency[gram] <= MAX_NGRAM_FREQUENCY:

            ngram_index[
                (country, gram)
            ].add(entity_id)


print(
    "Filtered n-gram keys:",
    len(ngram_index)
)


# ============================================================
# RECALL TEST
# ============================================================

print("\n" + "=" * 70)
print("FUZZY CHARACTER BLOCKING RECALL TEST")
print("=" * 70)


total_true_matches = 0
retrieved_true_matches = 0

s1_with_true_matches = 0
s1_with_retrieved_true_match = 0

total_candidates = 0
s1_with_candidates = 0


# ============================================================
# EVALUATE
# ============================================================

for i, row in enumerate(
    s1.itertuples(index=False),
    start=1
):

    country = row.country
    name = row.name_norm
    s1_id = row.entity_id

    candidates = set()

    # --------------------------------------------------------
    # Character n-gram blocking
    # --------------------------------------------------------

    grams = get_ngrams(name)

    for gram in grams:

        if ngram_frequency[gram] <= MAX_NGRAM_FREQUENCY:

            candidates.update(
                ngram_index.get(
                    (country, gram),
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
            f"Processed {i:,}/{len(s1):,} S1 records..."
        )


# ============================================================
# RESULTS
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
# RECALL
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
# CANDIDATE STATISTICS
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