import pandas as pd
import re
import os
from collections import defaultdict, Counter

# ============================================================
# SETTINGS
# ============================================================

TRAIN = "dataset/train"

# Start small for testing
SAMPLE_SIZE = 20000

# Process S1 in small batches
BATCH_SIZE = 250

OUTPUT_FILE = "candidate_features_memory_safe.csv"

# IMPORTANT:
# Ignore tokens that occur too frequently.
# This prevents huge candidate lists.
MAX_TOKEN_FREQUENCY = 1000

# Maximum candidates retained for one S1 record
MAX_CANDIDATES_PER_RECORD = 500


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def tokens(value):

    if not value:
        return set()

    return {
        x for x in value.split()
        if len(x) >= 3
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
        "business_address",
        "country"
    ]
)

print("Source 1:", len(s1))


print("Loading Source 2...")

s2 = pd.read_csv(
    f"{TRAIN}/train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("Source 2:", len(s2))


print("Loading Source 3...")

s3 = pd.read_csv(
    f"{TRAIN}/train_source3.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("Source 3:", len(s3))


# ============================================================
# GROUND TRUTH
# ============================================================

print("Loading ground truth...")

gt = pd.read_csv(
    f"{TRAIN}/train_ground_truth.tsv",
    sep="\t",
    usecols=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)

sample_ids = set(s1["entity_id"])

gt = gt[
    gt["source1_entity_id"].isin(sample_ids)
]

ground_truth = {}

for row in gt.itertuples(index=False):

    if pd.isna(row.matched_entity_ids):

        ground_truth[row.source1_entity_id] = set()

    else:

        ground_truth[row.source1_entity_id] = set(
            str(row.matched_entity_ids).split(",")
        )

print(
    "Ground-truth records:",
    len(ground_truth)
)


# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing data...")

for df in [s1, s2, s3]:

    df["name_norm"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_text)
    )

    df["address_norm"] = (
        df["business_address"]
        .fillna("")
        .map(normalize_text)
    )

print("Normalization complete.")


# ============================================================
# ENTITY INFORMATION
# ============================================================

print("Creating compact entity lookup...")

entities = {}

for row in s2.itertuples(index=False):

    entities[row.entity_id] = (
        row.name_norm,
        row.address_norm,
        row.country
    )

for row in s3.itertuples(index=False):

    entities[row.entity_id] = (
        row.name_norm,
        row.address_norm,
        row.country
    )

print(
    "Entities:",
    len(entities)
)


# ============================================================
# EXACT NAME INDEX
# ============================================================

print("\nBuilding exact-name index...")

name_index = defaultdict(list)

for row in s2.itertuples(index=False):

    if row.name_norm:

        name_index[
            (row.country, row.name_norm)
        ].append(row.entity_id)


for row in s3.itertuples(index=False):

    if row.name_norm:

        name_index[
            (row.country, row.name_norm)
        ].append(row.entity_id)


print(
    "Exact name keys:",
    len(name_index)
)


# ============================================================
# EXACT ADDRESS INDEX
# ============================================================

print("Building exact-address index...")

address_index = defaultdict(list)

for row in s2.itertuples(index=False):

    if row.address_norm:

        address_index[
            (row.country, row.address_norm)
        ].append(row.entity_id)


for row in s3.itertuples(index=False):

    if row.address_norm:

        address_index[
            (row.country, row.address_norm)
        ].append(row.entity_id)


print(
    "Exact address keys:",
    len(address_index)
)


# ============================================================
# TOKEN FREQUENCY
# ============================================================

print("\nCounting token frequencies...")

name_frequency = Counter()
address_frequency = Counter()


for row in s2.itertuples(index=False):

    for token in tokens(row.name_norm):
        name_frequency[token] += 1

    for token in tokens(row.address_norm):
        address_frequency[token] += 1


for row in s3.itertuples(index=False):

    for token in tokens(row.name_norm):
        name_frequency[token] += 1

    for token in tokens(row.address_norm):
        address_frequency[token] += 1


print(
    "Name tokens:",
    len(name_frequency)
)

print(
    "Address tokens:",
    len(address_frequency)
)


# ============================================================
# FILTERED TOKEN INDEXES
# ============================================================

print("\nBuilding filtered token indexes...")

name_token_index = defaultdict(list)

address_token_index = defaultdict(list)


# Source 2
for row in s2.itertuples(index=False):

    country = row.country

    for token in tokens(row.name_norm):

        if name_frequency[token] <= MAX_TOKEN_FREQUENCY:

            name_token_index[
                (country, token)
            ].append(row.entity_id)


    for token in tokens(row.address_norm):

        if address_frequency[token] <= MAX_TOKEN_FREQUENCY:

            address_token_index[
                (country, token)
            ].append(row.entity_id)


# Source 3
for row in s3.itertuples(index=False):

    country = row.country

    for token in tokens(row.name_norm):

        if name_frequency[token] <= MAX_TOKEN_FREQUENCY:

            name_token_index[
                (country, token)
            ].append(row.entity_id)


    for token in tokens(row.address_norm):

        if address_frequency[token] <= MAX_TOKEN_FREQUENCY:

            address_token_index[
                (country, token)
            ].append(row.entity_id)


print(
    "Name token keys:",
    len(name_token_index)
)

print(
    "Address token keys:",
    len(address_token_index)
)


# ============================================================
# JACCARD
# ============================================================

def jaccard(a, b):

    if not a or not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


def overlap(a, b):

    if not a or not b:
        return 0.0

    return len(a & b)


# ============================================================
# OUTPUT
# ============================================================

if os.path.exists(OUTPUT_FILE):

    os.remove(OUTPUT_FILE)


columns = [
    "source1_entity_id",
    "candidate_entity_id",
    "country_match",
    "name_exact",
    "address_exact",
    "name_overlap",
    "address_overlap",
    "name_jaccard",
    "address_jaccard",
    "label"
]


print("\n")
print("=" * 70)
print("MEMORY-SAFE FEATURE GENERATION")
print("=" * 70)


first_write = True

total_candidates = 0


# ============================================================
# PROCESS BATCHES
# ============================================================

for start in range(
    0,
    len(s1),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(s1)
    )

    batch = s1.iloc[start:end]

    output_rows = []


    for row in batch.itertuples(index=False):

        s1_id = row.entity_id

        country = row.country

        name = row.name_norm

        address = row.address_norm

        name_tokens = tokens(name)

        address_tokens = tokens(address)


        # ----------------------------------------------------
        # Candidate generation
        # ----------------------------------------------------

        candidate_scores = defaultdict(int)


        # EXACT NAME
        exact_name_candidates = name_index.get(
            (country, name),
            []
        )

        for candidate in exact_name_candidates:

            candidate_scores[candidate] += 10


        # EXACT ADDRESS
        exact_address_candidates = address_index.get(
            (country, address),
            []
        )

        for candidate in exact_address_candidates:

            candidate_scores[candidate] += 10


        # NAME TOKENS
        for token in name_tokens:

            candidate_list = name_token_index.get(
                (country, token),
                []
            )

            for candidate in candidate_list:

                candidate_scores[candidate] += 1


        # ADDRESS TOKENS
        for token in address_tokens:

            candidate_list = address_token_index.get(
                (country, token),
                []
            )

            for candidate in candidate_list:

                candidate_scores[candidate] += 1


        # ----------------------------------------------------
        # KEEP ONLY STRONGEST CANDIDATES
        # ----------------------------------------------------

        if len(candidate_scores) > MAX_CANDIDATES_PER_RECORD:

            candidates = sorted(
                candidate_scores,
                key=candidate_scores.get,
                reverse=True
            )[:MAX_CANDIDATES_PER_RECORD]

        else:

            candidates = list(
                candidate_scores.keys()
            )


        total_candidates += len(candidates)


        true_matches = ground_truth.get(
            s1_id,
            set()
        )


        # ----------------------------------------------------
        # FEATURE GENERATION
        # ----------------------------------------------------

        for candidate in candidates:

            target = entities.get(candidate)

            if target is None:
                continue


            target_name = target[0]

            target_address = target[1]

            target_country = target[2]


            target_name_tokens = tokens(
                target_name
            )

            target_address_tokens = tokens(
                target_address
            )


            name_exact = int(
                bool(name) and
                name == target_name
            )


            address_exact = int(
                bool(address) and
                address == target_address
            )


            name_overlap = overlap(
                name_tokens,
                target_name_tokens
            )


            address_overlap = overlap(
                address_tokens,
                target_address_tokens
            )


            name_j = jaccard(
                name_tokens,
                target_name_tokens
            )


            address_j = jaccard(
                address_tokens,
                target_address_tokens
            )


            country_match = int(
                country == target_country
            )


            label = int(
                candidate in true_matches
            )


            output_rows.append({

                "source1_entity_id":
                    s1_id,

                "candidate_entity_id":
                    candidate,

                "country_match":
                    country_match,

                "name_exact":
                    name_exact,

                "address_exact":
                    address_exact,

                "name_overlap":
                    name_overlap,

                "address_overlap":
                    address_overlap,

                "name_jaccard":
                    round(name_j, 4),

                "address_jaccard":
                    round(address_j, 4),

                "label":
                    label
            })


    # ========================================================
    # WRITE BATCH
    # ========================================================

    if output_rows:

        result = pd.DataFrame(
            output_rows,
            columns=columns
        )

        result.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False

        del result

        del output_rows


    print(
        f"Processed {end:,}/{len(s1):,} "
        f"S1 records | "
        f"Candidates so far: "
        f"{total_candidates:,}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    "Total candidates generated:",
    total_candidates
)

print(
    "Output:",
    OUTPUT_FILE
)

if os.path.exists(OUTPUT_FILE):

    size_mb = (
        os.path.getsize(OUTPUT_FILE)
        / (1024 * 1024)
    )

    print(
        f"Output size: {size_mb:.2f} MB"
    )

print("\nDone.")