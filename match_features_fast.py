import pandas as pd
import re
from collections import defaultdict

TRAIN = "dataset/train"

# Start with 20,000 for fast testing.
# After everything works, change to 100000.
SAMPLE_SIZE = 20000

OUTPUT_FILE = "candidate_features_fast.csv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokens(text):
    if not text:
        return set()

    return {
        x for x in text.split()
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


# ============================================================
# NORMALIZE
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


# ============================================================
# FAST INDEXES
# ============================================================

print("Building indexes...")

name_index = defaultdict(set)
address_index = defaultdict(set)


for row in s2.itertuples(index=False):

    if row.name_norm:
        name_index[
            (row.country, row.name_norm)
        ].add(row.entity_id)

    if row.address_norm:
        address_index[
            (row.country, row.address_norm)
        ].add(row.entity_id)


for row in s3.itertuples(index=False):

    if row.name_norm:
        name_index[
            (row.country, row.name_norm)
        ].add(row.entity_id)

    if row.address_norm:
        address_index[
            (row.country, row.address_norm)
        ].add(row.entity_id)


print("Indexes ready.")


# ============================================================
# FAST TOKEN INDEX
# ============================================================

name_token_index = defaultdict(set)
address_token_index = defaultdict(set)


for row in s2.itertuples(index=False):

    for token in tokens(row.name_norm):
        name_token_index[
            (row.country, token)
        ].add(row.entity_id)

    for token in tokens(row.address_norm):
        address_token_index[
            (row.country, token)
        ].add(row.entity_id)


for row in s3.itertuples(index=False):

    for token in tokens(row.name_norm):
        name_token_index[
            (row.country, token)
        ].add(row.entity_id)

    for token in tokens(row.address_norm):
        address_token_index[
            (row.country, token)
        ].add(row.entity_id)


print("Token indexes ready.")


# ============================================================
# FAST FEATURE FUNCTIONS
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
# GENERATE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FAST FEATURE GENERATION")
print("=" * 70)

results = []

processed = 0


for row in s1.itertuples(index=False):

    processed += 1

    s1_id = row.entity_id
    country = row.country

    name = row.name_norm
    address = row.address_norm

    name_set = tokens(name)
    address_set = tokens(address)

    candidates = set()

    # --------------------------------------------------------
    # Exact name
    # --------------------------------------------------------

    candidates.update(
        name_index.get(
            (country, name),
            set()
        )
    )

    # --------------------------------------------------------
    # Exact address
    # --------------------------------------------------------

    candidates.update(
        address_index.get(
            (country, address),
            set()
        )
    )

    # --------------------------------------------------------
    # Name token candidates
    # --------------------------------------------------------

    for token in name_set:

        candidates.update(
            name_token_index.get(
                (country, token),
                set()
            )
        )

    # --------------------------------------------------------
    # Address token candidates
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Calculate CHEAP features only
    # --------------------------------------------------------

    for candidate in candidates:

        target = entities.get(candidate)

        if target is None:
            continue

        target_name = target[0]
        target_address = target[1]
        target_country = target[2]

        target_name_set = tokens(
            target_name
        )

        target_address_set = tokens(
            target_address
        )

        name_exact = int(
            name != "" and
            name == target_name
        )

        address_exact = int(
            address != "" and
            address == target_address
        )

        name_jaccard = jaccard(
            name_set,
            target_name_set
        )

        address_jaccard = jaccard(
            address_set,
            target_address_set
        )

        name_containment = containment(
            name_set,
            target_name_set
        )

        address_containment = containment(
            address_set,
            target_address_set
        )

        country_match = int(
            country == target_country
        )

        label = int(
            candidate in true_matches
        )

        results.append({

            "source1_entity_id": s1_id,

            "candidate_entity_id": candidate,

            "country_match": country_match,

            "name_exact": name_exact,

            "address_exact": address_exact,

            "name_jaccard":
                round(name_jaccard, 4),

            "address_jaccard":
                round(address_jaccard, 4),

            "name_containment":
                round(name_containment, 4),

            "address_containment":
                round(address_containment, 4),

            "label": label
        })

    if processed % 1000 == 0:

        print(
            f"Processed "
            f"{processed:,}/{SAMPLE_SIZE:,}..."
        )


# ============================================================
# SAVE
# ============================================================

print("\nCreating output...")

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FAST FEATURE GENERATION COMPLETE")
print("=" * 70)

print(
    "Candidate pairs:",
    len(df)
)

print(
    "Positive matches:",
    int(df["label"].sum())
)

print(
    "Negative pairs:",
    int((df["label"] == 0).sum())
)

print(
    "Output:",
    OUTPUT_FILE
)

print("\nDone.")