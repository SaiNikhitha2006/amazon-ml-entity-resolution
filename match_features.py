import pandas as pd
import re
import os
from collections import defaultdict, Counter
from difflib import SequenceMatcher

from normalize import normalize_business_name


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN = "dataset/train"
SAMPLE_SIZE = 100_000

MAX_NAME_TOKEN_FREQUENCY = 500
MAX_ADDRESS_TOKEN_FREQUENCY = 500
MAX_PREFIX_FREQUENCY = 300

PREFIX_LENGTHS = (3, 4)

OUTPUT_FILE = "candidate_features_100k.csv"


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

def normalize_address(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    replacements = {
        r"\bst\b": "street",
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

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# TOKEN FUNCTIONS
# ============================================================

def name_tokens(text):

    if not text:
        return set()

    return {
        token
        for token in text.split()
        if len(token) >= 3
    }


def address_tokens(text):

    if not text:
        return set()

    return {
        token
        for token in text.split()
        if len(token) >= 3
    }


# ============================================================
# CHARACTER PREFIXES
# ============================================================

def prefixes(text):

    if not text:
        return set()

    compact = re.sub(
        r"[^a-z0-9]",
        "",
        text
    )

    if not compact:
        return set()

    result = set()

    for length in PREFIX_LENGTHS:

        if len(compact) >= length:
            result.add(
                compact[:length]
            )

    return result


# ============================================================
# SIMILARITY FUNCTIONS
# ============================================================

def sequence_similarity(a, b):

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def token_jaccard(a, b):

    if not a or not b:
        return 0.0

    intersection = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0.0

    return intersection / union


def token_containment(a, b):

    if not a or not b:
        return 0.0

    intersection = len(a & b)

    smaller = min(
        len(a),
        len(b)
    )

    if smaller == 0:
        return 0.0

    return intersection / smaller


def numeric_tokens(text):

    if not text:
        return set()

    return set(
        re.findall(
            r"\d+",
            text
        )
    )


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

print(
    "Source 1 loaded:",
    len(s1)
)


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

print(
    "Source 2 loaded:",
    len(s2)
)


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

print(
    "Source 3 loaded:",
    len(s3)
)


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

sample_ids = set(
    s1["entity_id"]
)

gt = gt[
    gt["source1_entity_id"].isin(
        sample_ids
    )
]

ground_truth = {}

for _, row in gt.iterrows():

    value = row["matched_entity_ids"]

    if pd.isna(value):

        ground_truth[
            row["source1_entity_id"]
        ] = set()

    else:

        value = str(value).strip()

        if value:

            ground_truth[
                row["source1_entity_id"]
            ] = set(
                value.split(",")
            )

        else:

            ground_truth[
                row["source1_entity_id"]
            ] = set()


# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing names...")

s1["name_norm"] = (
    s1["business_name"]
    .fillna("")
    .map(normalize_business_name)
)

s2["name_norm"] = (
    s2["business_name"]
    .fillna("")
    .map(normalize_business_name)
)

s3["name_norm"] = (
    s3["business_name"]
    .fillna("")
    .map(normalize_business_name)
)


print("Normalizing addresses...")

s1["address_norm"] = (
    s1["business_address"]
    .fillna("")
    .map(normalize_address)
)

s2["address_norm"] = (
    s2["business_address"]
    .fillna("")
    .map(normalize_address)
)

s3["address_norm"] = (
    s3["business_address"]
    .fillna("")
    .map(normalize_address)
)


# ============================================================
# ENTITY LOOKUP
# ============================================================

print("Creating entity lookup...")

entity_lookup = {}

for row in s2.itertuples(index=False):

    entity_lookup[
        row.entity_id
    ] = {
        "name": row.name_norm,
        "address": row.address_norm,
        "country": row.country
    }


for row in s3.itertuples(index=False):

    entity_lookup[
        row.entity_id
    ] = {
        "name": row.name_norm,
        "address": row.address_norm,
        "country": row.country
    }


# ============================================================
# BUILD BLOCKING INDEXES
# ============================================================

print("\nBuilding blocking indexes...")


# ------------------------------------------------------------
# Exact name
# ------------------------------------------------------------

exact_name_index = defaultdict(set)

for row in s2.itertuples(index=False):

    if row.name_norm:

        exact_name_index[
            (row.country, row.name_norm)
        ].add(row.entity_id)


for row in s3.itertuples(index=False):

    if row.name_norm:

        exact_name_index[
            (row.country, row.name_norm)
        ].add(row.entity_id)


# ------------------------------------------------------------
# Name token frequency
# ------------------------------------------------------------

name_frequency = Counter()

for name in s2["name_norm"]:

    for token in name_tokens(name):
        name_frequency[token] += 1


for name in s3["name_norm"]:

    for token in name_tokens(name):
        name_frequency[token] += 1


# ------------------------------------------------------------
# Name token index
# ------------------------------------------------------------

name_token_index = defaultdict(set)

for row in s2.itertuples(index=False):

    for token in name_tokens(row.name_norm):

        if (
            name_frequency[token]
            <= MAX_NAME_TOKEN_FREQUENCY
        ):

            name_token_index[
                (row.country, token)
            ].add(row.entity_id)


for row in s3.itertuples(index=False):

    for token in name_tokens(row.name_norm):

        if (
            name_frequency[token]
            <= MAX_NAME_TOKEN_FREQUENCY
        ):

            name_token_index[
                (row.country, token)
            ].add(row.entity_id)


# ------------------------------------------------------------
# Exact address
# ------------------------------------------------------------

exact_address_index = defaultdict(set)

for row in s2.itertuples(index=False):

    if row.address_norm:

        exact_address_index[
            (row.country, row.address_norm)
        ].add(row.entity_id)


for row in s3.itertuples(index=False):

    if row.address_norm:

        exact_address_index[
            (row.country, row.address_norm)
        ].add(row.entity_id)


# ------------------------------------------------------------
# Address token frequency
# ------------------------------------------------------------

address_frequency = Counter()

for address in s2["address_norm"]:

    for token in address_tokens(address):
        address_frequency[token] += 1


for address in s3["address_norm"]:

    for token in address_tokens(address):
        address_frequency[token] += 1


# ------------------------------------------------------------
# Address token index
# ------------------------------------------------------------

address_token_index = defaultdict(set)

for row in s2.itertuples(index=False):

    for token in address_tokens(
        row.address_norm
    ):

        if (
            address_frequency[token]
            <= MAX_ADDRESS_TOKEN_FREQUENCY
        ):

            address_token_index[
                (row.country, token)
            ].add(row.entity_id)


for row in s3.itertuples(index=False):

    for token in address_tokens(
        row.address_norm
    ):

        if (
            address_frequency[token]
            <= MAX_ADDRESS_TOKEN_FREQUENCY
        ):

            address_token_index[
                (row.country, token)
            ].add(row.entity_id)


# ------------------------------------------------------------
# Prefix frequency
# ------------------------------------------------------------

name_prefix_frequency = Counter()
address_prefix_frequency = Counter()


for name in s2["name_norm"]:

    for p in prefixes(name):
        name_prefix_frequency[p] += 1


for name in s3["name_norm"]:

    for p in prefixes(name):
        name_prefix_frequency[p] += 1


for address in s2["address_norm"]:

    for p in prefixes(address):
        address_prefix_frequency[p] += 1


for address in s3["address_norm"]:

    for p in prefixes(address):
        address_prefix_frequency[p] += 1


# ------------------------------------------------------------
# Prefix indexes
# ------------------------------------------------------------

name_prefix_index = defaultdict(set)

address_prefix_index = defaultdict(set)


for row in s2.itertuples(index=False):

    for p in prefixes(row.name_norm):

        if (
            name_prefix_frequency[p]
            <= MAX_PREFIX_FREQUENCY
        ):

            name_prefix_index[
                (row.country, p)
            ].add(row.entity_id)

    for p in prefixes(row.address_norm):

        if (
            address_prefix_frequency[p]
            <= MAX_PREFIX_FREQUENCY
        ):

            address_prefix_index[
                (row.country, p)
            ].add(row.entity_id)


for row in s3.itertuples(index=False):

    for p in prefixes(row.name_norm):

        if (
            name_prefix_frequency[p]
            <= MAX_PREFIX_FREQUENCY
        ):

            name_prefix_index[
                (row.country, p)
            ].add(row.entity_id)

    for p in prefixes(row.address_norm):

        if (
            address_prefix_frequency[p]
            <= MAX_PREFIX_FREQUENCY
        ):

            address_prefix_index[
                (row.country, p)
            ].add(row.entity_id)


print("Blocking indexes ready.")


# ============================================================
# FEATURE GENERATION
# ============================================================

print("\n" + "=" * 70)
print("GENERATING MATCHING FEATURES")
print("=" * 70)


feature_rows = []

positive_count = 0
negative_count = 0

processed = 0


for row in s1.itertuples(index=False):

    processed += 1

    s1_id = row.entity_id
    country = row.country

    s1_name = row.name_norm
    s1_address = row.address_norm

    s1_name_set = name_tokens(
        s1_name
    )

    s1_address_set = address_tokens(
        s1_address
    )

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    candidates = set()

    # Exact name
    candidates.update(
        exact_name_index.get(
            (country, s1_name),
            set()
        )
    )

    # Name token
    for token in s1_name_set:

        if (
            name_frequency[token]
            <= MAX_NAME_TOKEN_FREQUENCY
        ):

            candidates.update(
                name_token_index.get(
                    (country, token),
                    set()
                )
            )

    # Exact address
    candidates.update(
        exact_address_index.get(
            (country, s1_address),
            set()
        )
    )

    # Address token
    for token in s1_address_set:

        if (
            address_frequency[token]
            <= MAX_ADDRESS_TOKEN_FREQUENCY
        ):

            candidates.update(
                address_token_index.get(
                    (country, token),
                    set()
                )
            )

    # Name prefix
    for p in prefixes(s1_name):

        if (
            name_prefix_frequency[p]
            <= MAX_PREFIX_FREQUENCY
        ):

            candidates.update(
                name_prefix_index.get(
                    (country, p),
                    set()
                )
            )

    # Address prefix
    for p in prefixes(s1_address):

        if (
            address_prefix_frequency[p]
            <= MAX_PREFIX_FREQUENCY
        ):

            candidates.update(
                address_prefix_index.get(
                    (country, p),
                    set()
                )
            )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    true_matches = ground_truth.get(
        s1_id,
        set()
    )

    # --------------------------------------------------------
    # Generate features
    # --------------------------------------------------------

    for candidate_id in candidates:

        target = entity_lookup.get(
            candidate_id
        )

        if target is None:
            continue

        target_name = target["name"]
        target_address = target["address"]
        target_country = target["country"]

        target_name_set = name_tokens(
            target_name
        )

        target_address_set = address_tokens(
            target_address
        )

        # ====================================================
        # NAME FEATURES
        # ====================================================

        name_seq = sequence_similarity(
            s1_name,
            target_name
        )

        name_jaccard = token_jaccard(
            s1_name_set,
            target_name_set
        )

        name_containment = token_containment(
            s1_name_set,
            target_name_set
        )

        name_exact = int(
            bool(s1_name)
            and s1_name == target_name
        )

        # ====================================================
        # ADDRESS FEATURES
        # ====================================================

        address_seq = sequence_similarity(
            s1_address,
            target_address
        )

        address_jaccard = token_jaccard(
            s1_address_set,
            target_address_set
        )

        address_containment = token_containment(
            s1_address_set,
            target_address_set
        )

        address_exact = int(
            bool(s1_address)
            and s1_address == target_address
        )

        # ====================================================
        # NUMERIC ADDRESS FEATURES
        # ====================================================

        s1_numbers = numeric_tokens(
            s1_address
        )

        target_numbers = numeric_tokens(
            target_address
        )

        number_overlap = token_jaccard(
            s1_numbers,
            target_numbers
        )

        # ====================================================
        # COUNTRY
        # ====================================================

        country_match = int(
            country == target_country
        )

        # ====================================================
        # LENGTH FEATURES
        # ====================================================

        name_length_difference = abs(
            len(s1_name)
            -
            len(target_name)
        )

        address_length_difference = abs(
            len(s1_address)
            -
            len(target_address)
        )

        # ====================================================
        # LABEL
        # ====================================================

        label = int(
            candidate_id in true_matches
        )

        if label:
            positive_count += 1
        else:
            negative_count += 1

        # ====================================================
        # SAVE
        # ====================================================

        feature_rows.append({

            "source1_entity_id":
                s1_id,

            "candidate_entity_id":
                candidate_id,

            "country_match":
                country_match,

            "name_exact":
                name_exact,

            "name_sequence_similarity":
                round(name_seq, 4),

            "name_token_jaccard":
                round(name_jaccard, 4),

            "name_token_containment":
                round(name_containment, 4),

            "address_exact":
                address_exact,

            "address_sequence_similarity":
                round(address_seq, 4),

            "address_token_jaccard":
                round(address_jaccard, 4),

            "address_token_containment":
                round(address_containment, 4),

            "address_number_overlap":
                round(number_overlap, 4),

            "name_length_difference":
                name_length_difference,

            "address_length_difference":
                address_length_difference,

            "label":
                label
        })

    if processed % 10000 == 0:

        print(
            f"Processed "
            f"{processed:,}/{len(s1):,} "
            f"S1 records..."
        )


# ============================================================
# SAVE FEATURES
# ============================================================

print("\nCreating DataFrame...")

features = pd.DataFrame(
    feature_rows
)

print(
    "Total candidate pairs:",
    len(features)
)

print(
    "Positive pairs:",
    positive_count
)

print(
    "Negative pairs:",
    negative_count
)


features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE GENERATION COMPLETE")
print("=" * 70)

print(
    "\nOutput file:",
    OUTPUT_FILE
)

print(
    "Rows:",
    len(features)
)

print(
    "Columns:",
    len(features.columns)
)

print("\nFeature columns:")

for column in features.columns:

    print(
        " -",
        column
    )


print("\nDone.")