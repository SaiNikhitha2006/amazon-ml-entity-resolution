import pandas as pd
import re
from collections import defaultdict, Counter
from normalize import normalize_business_name


TRAIN = "dataset/train"
SAMPLE_SIZE = 100_000

MAX_NAME_TOKEN_FREQUENCY = 500
MAX_ADDRESS_TOKEN_FREQUENCY = 500


# ============================================================
# NORMALIZATION
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


def get_name_tokens(text):

    if not text:
        return set()

    return {
        x for x in text.split()
        if len(x) >= 4
    }


def get_address_tokens(text):

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
# CREATE LOOKUPS
# ============================================================

print("Creating entity lookup...")

source2_lookup = s2.set_index("entity_id").to_dict("index")
source3_lookup = s3.set_index("entity_id").to_dict("index")


print("Creating ground-truth lookup...")

sample_ids = set(s1["entity_id"])

gt = gt[
    gt["source1_entity_id"].isin(sample_ids)
]

ground_truth = {}

for _, row in gt.iterrows():

    value = row["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        ground_truth[row["source1_entity_id"]] = set()
    else:
        ground_truth[row["source1_entity_id"]] = set(
            str(value).split(",")
        )


# ============================================================
# NORMALIZE
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
# BUILD NAME INDEXES
# ============================================================

print("\nBuilding name indexes...")

exact_name_index = defaultdict(set)
name_frequency = Counter()

for country, name, entity_id in zip(
    s2["country"],
    s2["name_norm"],
    s2["entity_id"]
):

    if name:
        exact_name_index[
            (country, name)
        ].add(entity_id)

    for token in get_name_tokens(name):
        name_frequency[token] += 1


for country, name, entity_id in zip(
    s3["country"],
    s3["name_norm"],
    s3["entity_id"]
):

    if name:
        exact_name_index[
            (country, name)
        ].add(entity_id)

    for token in get_name_tokens(name):
        name_frequency[token] += 1


name_token_index = defaultdict(set)

for country, name, entity_id in zip(
    s2["country"],
    s2["name_norm"],
    s2["entity_id"]
):

    for token in get_name_tokens(name):

        if name_frequency[token] <= MAX_NAME_TOKEN_FREQUENCY:
            name_token_index[
                (country, token)
            ].add(entity_id)


for country, name, entity_id in zip(
    s3["country"],
    s3["name_norm"],
    s3["entity_id"]
):

    for token in get_name_tokens(name):

        if name_frequency[token] <= MAX_NAME_TOKEN_FREQUENCY:
            name_token_index[
                (country, token)
            ].add(entity_id)


# ============================================================
# BUILD ADDRESS INDEXES
# ============================================================

print("Building address indexes...")

exact_address_index = defaultdict(set)
address_frequency = Counter()

for country, address, entity_id in zip(
    s2["country"],
    s2["address_norm"],
    s2["entity_id"]
):

    if address:
        exact_address_index[
            (country, address)
        ].add(entity_id)

    for token in get_address_tokens(address):
        address_frequency[token] += 1


for country, address, entity_id in zip(
    s3["country"],
    s3["address_norm"],
    s3["entity_id"]
):

    if address:
        exact_address_index[
            (country, address)
        ].add(entity_id)

    for token in get_address_tokens(address):
        address_frequency[token] += 1


address_token_index = defaultdict(set)

for country, address, entity_id in zip(
    s2["country"],
    s2["address_norm"],
    s2["entity_id"]
):

    for token in get_address_tokens(address):

        if address_frequency[token] <= MAX_ADDRESS_TOKEN_FREQUENCY:
            address_token_index[
                (country, token)
            ].add(entity_id)


for country, address, entity_id in zip(
    s3["country"],
    s3["address_norm"],
    s3["entity_id"]
):

    for token in get_address_tokens(address):

        if address_frequency[token] <= MAX_ADDRESS_TOKEN_FREQUENCY:
            address_token_index[
                (country, token)
            ].add(entity_id)


# ============================================================
# ANALYZE MISSED MATCHES
# ============================================================

print("\n" + "=" * 70)
print("ANALYZING MISSED TRUE MATCHES")
print("=" * 70)


missed_pairs = []

missed_by_country = Counter()

missed_name_exact = 0
missed_address_exact = 0

missed_name_token_overlap = 0
missed_address_token_overlap = 0

missed_both_empty = 0

total_true_pairs = 0
retrieved_pairs = 0


# ============================================================
# PROCESS S1
# ============================================================

for i, row in enumerate(
    s1.itertuples(index=False),
    start=1
):

    s1_id = row.entity_id
    country = row.country
    name = row.name_norm
    address = row.address_norm

    # --------------------------------------------------------
    # Generate candidates using current combined blocker
    # --------------------------------------------------------

    candidates = set()

    # Exact name
    candidates.update(
        exact_name_index.get(
            (country, name),
            set()
        )
    )

    # Name tokens
    for token in get_name_tokens(name):

        if name_frequency[token] <= MAX_NAME_TOKEN_FREQUENCY:

            candidates.update(
                name_token_index.get(
                    (country, token),
                    set()
                )
            )

    # Exact address
    candidates.update(
        exact_address_index.get(
            (country, address),
            set()
        )
    )

    # Address tokens
    for token in get_address_tokens(address):

        if address_frequency[token] <= MAX_ADDRESS_TOKEN_FREQUENCY:

            candidates.update(
                address_token_index.get(
                    (country, token),
                    set()
                )
            )

    # --------------------------------------------------------
    # True matches
    # --------------------------------------------------------

    true_matches = ground_truth.get(
        s1_id,
        set()
    )

    for true_id in true_matches:

        total_true_pairs += 1

        if true_id in candidates:

            retrieved_pairs += 1

            continue

        # ====================================================
        # MISSED TRUE MATCH
        # ====================================================

        if true_id.startswith("S2-"):
            target = source2_lookup.get(true_id)
        else:
            target = source3_lookup.get(true_id)

        if target is None:
            continue

        target_name = normalize_business_name(
            target.get("business_name", "")
        )

        target_address = normalize_address(
            target.get("business_address", "")
        )

        # ----------------------------------------------------
        # Name comparison
        # ----------------------------------------------------

        source_name_tokens = get_name_tokens(name)
        target_name_tokens = get_name_tokens(target_name)

        name_overlap = (
            source_name_tokens &
            target_name_tokens
        )

        exact_name = (
            name != "" and
            name == target_name
        )

        # ----------------------------------------------------
        # Address comparison
        # ----------------------------------------------------

        source_address_tokens = get_address_tokens(address)
        target_address_tokens = get_address_tokens(
            target_address
        )

        address_overlap = (
            source_address_tokens &
            target_address_tokens
        )

        exact_address = (
            address != "" and
            address == target_address
        )

        # ----------------------------------------------------
        # Classify
        # ----------------------------------------------------

        if exact_name:
            missed_name_exact += 1

        if exact_address:
            missed_address_exact += 1

        if name_overlap:
            missed_name_token_overlap += 1

        if address_overlap:
            missed_address_token_overlap += 1

        if not name and not address:
            missed_both_empty += 1

        missed_by_country[country] += 1

        missed_pairs.append({
            "source1_id": s1_id,
            "matched_id": true_id,
            "country": country,
            "source1_name": name,
            "matched_name": target_name,
            "source1_address": address,
            "matched_address": target_address,
            "name_overlap": " ".join(
                sorted(name_overlap)
            ),
            "address_overlap": " ".join(
                sorted(address_overlap)
            ),
        })

    if i % 10000 == 0:
        print(
            f"Processed {i:,}/{len(s1):,} S1 records..."
        )


# ============================================================
# RESULTS
# ============================================================

missed_count = len(missed_pairs)

print("\n" + "=" * 70)
print("MISSED MATCH ANALYSIS")
print("=" * 70)

print(
    "\nTotal true pairs:",
    total_true_pairs
)

print(
    "Retrieved true pairs:",
    retrieved_pairs
)

print(
    "Missed true pairs:",
    missed_count
)

print(
    "Observed recall:",
    round(
        retrieved_pairs / total_true_pairs * 100,
        2
    ),
    "%"
)


# ============================================================
# MISSED PAIR CHARACTERISTICS
# ============================================================

print("\nMissed pair characteristics:")

print(
    "Missed pairs with exact normalized name:",
    missed_name_exact
)

print(
    "Missed pairs with exact normalized address:",
    missed_address_exact
)

print(
    "Missed pairs with at least one common name token:",
    missed_name_token_overlap
)

print(
    "Missed pairs with at least one common address token:",
    missed_address_token_overlap
)

print(
    "Missed pairs with both name and address empty:",
    missed_both_empty
)


# ============================================================
# COUNTRY
# ============================================================

print("\nMissed pairs by country:")

for country, count in missed_by_country.items():

    print(
        f"{country}: {count:,}"
    )


# ============================================================
# SAVE SAMPLE
# ============================================================

if missed_pairs:

    missed_df = pd.DataFrame(
        missed_pairs
    )

    output_file = "missed_matches_sample.csv"

    # Save first 5,000 missed pairs
    missed_df.head(5000).to_csv(
        output_file,
        index=False
    )

    print(
        "\nSaved first 5,000 missed pairs to:",
        output_file
    )

    print("\nSample missed matches:")

    print(
        missed_df.head(20).to_string(
            index=False
        )
    )


print("\nDone.")