import pandas as pd
import re
import os
from collections import defaultdict

# ============================================================
# SETTINGS
# ============================================================

TRAIN = "dataset/train"

# Start with 20,000 only
SAMPLE_SIZE = 20000

# Number of S1 records processed before writing to disk
BATCH_SIZE = 500

OUTPUT_FILE = "candidate_features_stream.csv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_tokens(text):
    if not text:
        return set()

    return {
        x for x in text.split()
        if len(x) >= 3
    }


# ============================================================
# FAST SIMILARITY
# ============================================================

def jaccard(a, b):

    if not a or not b:
        return 0.0

    intersection = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0.0

    return intersection / union


def containment(a, b):

    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


# ============================================================
# LOAD SOURCE 1
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


# ============================================================
# LOAD SOURCE 2
# ============================================================

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


# ============================================================
# LOAD SOURCE 3
# ============================================================

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
# LOAD GROUND TRUTH
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

print("Ground-truth records:", len(ground_truth))


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
# ENTITY LOOKUP
# ============================================================

print("Creating entity lookup...")

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

print("Entities:", len(entities))


# ============================================================
# BUILD INDEXES
# ============================================================

print("Building indexes...")

name_index = defaultdict(set)
address_index = defaultdict(set)

name_token_index = defaultdict(set)
address_token_index = defaultdict(set)


# ----------------------------
# SOURCE 2
# ----------------------------

for row in s2.itertuples(index=False):

    country = row.country

    if row.name_norm:

        name_index[
            (country, row.name_norm)
        ].add(row.entity_id)

        for token in get_tokens(row.name_norm):

            name_token_index[
                (country, token)
            ].add(row.entity_id)

    if row.address_norm:

        address_index[
            (country, row.address_norm)
        ].add(row.entity_id)

        for token in get_tokens(row.address_norm):

            address_token_index[
                (country, token)
            ].add(row.entity_id)


# ----------------------------
# SOURCE 3
# ----------------------------

for row in s3.itertuples(index=False):

    country = row.country

    if row.name_norm:

        name_index[
            (country, row.name_norm)
        ].add(row.entity_id)

        for token in get_tokens(row.name_norm):

            name_token_index[
                (country, token)
            ].add(row.entity_id)

    if row.address_norm:

        address_index[
            (country, row.address_norm)
        ].add(row.entity_id)

        for token in get_tokens(row.address_norm):

            address_token_index[
                (country, token)
            ].add(row.entity_id)


print("Indexes ready.")


# ============================================================
# OUTPUT FILE
# ============================================================

if os.path.exists(OUTPUT_FILE):

    os.remove(OUTPUT_FILE)

print("\n" + "=" * 70)
print("STREAMING FEATURE GENERATION")
print("=" * 70)


# ============================================================
# PROCESS IN BATCHES
# ============================================================

total_rows = len(s1)

columns = [
    "source1_entity_id",
    "candidate_entity_id",
    "country_match",
    "name_exact",
    "address_exact",
    "name_jaccard",
    "address_jaccard",
    "name_containment",
    "address_containment",
    "label"
]

first_write = True


for start in range(0, total_rows, BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        total_rows
    )

    batch = s1.iloc[start:end]

    output_rows = []

    for row in batch.itertuples(index=False):

        s1_id = row.entity_id
        country = row.country

        name = row.name_norm
        address = row.address_norm

        name_set = get_tokens(name)
        address_set = get_tokens(address)

        candidates = set()

        # Exact name
        candidates.update(
            name_index.get(
                (country, name),
                set()
            )
        )

        # Exact address
        candidates.update(
            address_index.get(
                (country, address),
                set()
            )
        )

        # Name tokens
        for token in name_set:

            candidates.update(
                name_token_index.get(
                    (country, token),
                    set()
                )
            )

        # Address tokens
        for token in address_set:

            candidates.update(
                address_token_index.get(
                    (country, token),
                    set()
                )
            )

        true_matches = ground_truth.get(
            s1_id,
            set()
        )

        # ----------------------------------------------------
        # FEATURE CALCULATION
        # ----------------------------------------------------

        for candidate in candidates:

            target = entities.get(candidate)

            if target is None:
                continue

            target_name = target[0]
            target_address = target[1]
            target_country = target[2]

            target_name_set = get_tokens(
                target_name
            )

            target_address_set = get_tokens(
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

            name_j = jaccard(
                name_set,
                target_name_set
            )

            address_j = jaccard(
                address_set,
                target_address_set
            )

            name_c = containment(
                name_set,
                target_name_set
            )

            address_c = containment(
                address_set,
                target_address_set
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

                "name_jaccard":
                    round(name_j, 4),

                "address_jaccard":
                    round(address_j, 4),

                "name_containment":
                    round(name_c, 4),

                "address_containment":
                    round(address_c, 4),

                "label":
                    label
            })

    # ========================================================
    # WRITE BATCH TO DISK
    # ========================================================

    if output_rows:

        batch_df = pd.DataFrame(
            output_rows,
            columns=columns
        )

        batch_df.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False

        del batch_df
        del output_rows

    print(
        f"Processed {end:,}/{total_rows:,} "
        f"S1 records | "
        f"Saved to {OUTPUT_FILE}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STREAMING FEATURE GENERATION COMPLETE")
print("=" * 70)

if os.path.exists(OUTPUT_FILE):

    size_mb = (
        os.path.getsize(OUTPUT_FILE)
        / (1024 * 1024)
    )

    print(
        f"Output file: {OUTPUT_FILE}"
    )

    print(
        f"File size: {size_mb:.2f} MB"
    )

    print(
        "\nFeatures were written directly "
        "to disk in batches."
    )

print("\nDone.")