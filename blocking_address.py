import pandas as pd
import re
from collections import defaultdict, Counter


TRAIN = "dataset/train"
SAMPLE_SIZE = 100_000

# Ignore extremely common address tokens
MAX_TOKEN_FREQUENCY = 500


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

def normalize_address(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Replace common address abbreviations
    replacements = {
        r"\bst\b": "street",
        r"\bstreet\b": "street",
        r"\brd\b": "road",
        r"\bave\b": "avenue",
        r"\bav\b": "avenue",
        r"\bdr\b": "drive",
        r"\bln\b": "lane",
        r"\bblvd\b": "boulevard",
        r"\bhwy\b": "highway",
        r"\bctr\b": "center",
        r"\bpkwy\b": "parkway",
        r"\bpl\b": "place",
        r"\bct\b": "court",
        r"\bapt\b": "apartment",
        r"\bste\b": "suite",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


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
        "business_address",
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
        "business_address",
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
        "business_address",
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


# ============================================================
# NORMALIZE ADDRESSES
# ============================================================

print("\nNormalizing Source 1 addresses...")

s1["address_norm"] = (
    s1["business_address"]
    .fillna("")
    .map(normalize_address)
)

print("Source 1 done.")


print("Normalizing Source 2 addresses...")

s2["address_norm"] = (
    s2["business_address"]
    .fillna("")
    .map(normalize_address)
)

print("Source 2 done.")


print("Normalizing Source 3 addresses...")

s3["address_norm"] = (
    s3["business_address"]
    .fillna("")
    .map(normalize_address)
)

print("Source 3 done.")


# ============================================================
# EXACT ADDRESS INDEX
# ============================================================

print("\nBuilding exact address index...")

exact_address_index = defaultdict(set)

for country, address, entity_id in zip(
    s2["country"],
    s2["address_norm"],
    s2["entity_id"]
):

    if address:
        exact_address_index[
            (country, address)
        ].add(entity_id)


for country, address, entity_id in zip(
    s3["country"],
    s3["address_norm"],
    s3["entity_id"]
):

    if address:
        exact_address_index[
            (country, address)
        ].add(entity_id)


print(
    "Exact address keys:",
    len(exact_address_index)
)


# ============================================================
# TOKEN FREQUENCY
# ============================================================

print("\nCounting address token frequencies...")

token_frequency = Counter()


def get_tokens(address):

    return {
        token
        for token in address.split()
        if len(token) >= 3
    }


for address in s2["address_norm"]:

    if address:

        for token in get_tokens(address):
            token_frequency[token] += 1


for address in s3["address_norm"]:

    if address:

        for token in get_tokens(address):
            token_frequency[token] += 1


print(
    "Unique address tokens:",
    len(token_frequency)
)


# ============================================================
# BUILD FILTERED TOKEN INDEX
# ============================================================

print("\nBuilding address token index...")

address_token_index = defaultdict(set)


for country, address, entity_id in zip(
    s2["country"],
    s2["address_norm"],
    s2["entity_id"]
):

    if not address:
        continue

    for token in get_tokens(address):

        if token_frequency[token] <= MAX_TOKEN_FREQUENCY:

            address_token_index[
                (country, token)
            ].add(entity_id)


for country, address, entity_id in zip(
    s3["country"],
    s3["address_norm"],
    s3["entity_id"]
):

    if not address:
        continue

    for token in get_tokens(address):

        if token_frequency[token] <= MAX_TOKEN_FREQUENCY:

            address_token_index[
                (country, token)
            ].add(entity_id)


print(
    "Filtered address token keys:",
    len(address_token_index)
)


# ============================================================
# TEST BLOCKING
# ============================================================

print("\n" + "=" * 70)
print("ADDRESS BLOCKING RECALL TEST")
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
    address = row.address_norm
    s1_id = row.entity_id

    candidates = set()


    # --------------------------------------------------------
    # Exact address blocking
    # --------------------------------------------------------

    if address:

        candidates.update(
            exact_address_index.get(
                (country, address),
                set()
            )
        )


    # --------------------------------------------------------
    # Address token blocking
    # --------------------------------------------------------

    if address:

        tokens = get_tokens(address)

        for token in tokens:

            if token_frequency[token] <= MAX_TOKEN_FREQUENCY:

                candidates.update(
                    address_token_index.get(
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