import pandas as pd
import re
from collections import defaultdict, Counter
from normalize import normalize_business_name


TRAIN = "dataset/train"

# Test on 100K S1 records first
SAMPLE_SIZE = 100_000

# Ignore very common tokens
MAX_NAME_TOKEN_FREQUENCY = 500
MAX_ADDRESS_TOKEN_FREQUENCY = 500


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

def get_name_tokens(name):

    if not name:
        return set()

    return {
        token
        for token in name.split()
        if len(token) >= 4
    }


def get_address_tokens(address):

    if not address:
        return set()

    return {
        token
        for token in address.split()
        if len(token) >= 3
    }


# ============================================================
# 1. LOAD SOURCE 1
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

print("Source 1 loaded:", len(s1))


# ============================================================
# 2. LOAD SOURCE 2
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

print("Source 2 loaded:", len(s2))


# ============================================================
# 3. LOAD SOURCE 3
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

print("Source 3 loaded:", len(s3))


# ============================================================
# 4. LOAD GROUND TRUTH
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


# ============================================================
# 5. GROUND TRUTH LOOKUP
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

        ground_truth[
            row["source1_entity_id"]
        ] = set()

    else:

        ground_truth[
            row["source1_entity_id"]
        ] = set(
            str(value).split(",")
        )


print(
    "Ground-truth records:",
    len(ground_truth)
)


# ============================================================
# 6. NORMALIZE BUSINESS NAMES
# ============================================================

print("\nNormalizing Source 1 names...")

s1["name_norm"] = (
    s1["business_name"]
    .fillna("")
    .map(normalize_business_name)
)

print("Source 1 names done.")


print("Normalizing Source 2 names...")

s2["name_norm"] = (
    s2["business_name"]
    .fillna("")
    .map(normalize_business_name)
)

print("Source 2 names done.")


print("Normalizing Source 3 names...")

s3["name_norm"] = (
    s3["business_name"]
    .fillna("")
    .map(normalize_business_name)
)

print("Source 3 names done.")


# ============================================================
# 7. NORMALIZE ADDRESSES
# ============================================================

print("\nNormalizing Source 1 addresses...")

s1["address_norm"] = (
    s1["business_address"]
    .fillna("")
    .map(normalize_address)
)

print("Source 1 addresses done.")


print("Normalizing Source 2 addresses...")

s2["address_norm"] = (
    s2["business_address"]
    .fillna("")
    .map(normalize_address)
)

print("Source 2 addresses done.")


print("Normalizing Source 3 addresses...")

s3["address_norm"] = (
    s3["business_address"]
    .fillna("")
    .map(normalize_address)
)

print("Source 3 addresses done.")


# ============================================================
# 8. EXACT NAME INDEX
# ============================================================

print("\nBuilding exact-name index...")

exact_name_index = defaultdict(set)


for country, name, entity_id in zip(
    s2["country"],
    s2["name_norm"],
    s2["entity_id"]
):

    if name:

        exact_name_index[
            (country, name)
        ].add(entity_id)


for country, name, entity_id in zip(
    s3["country"],
    s3["name_norm"],
    s3["entity_id"]
):

    if name:

        exact_name_index[
            (country, name)
        ].add(entity_id)


print(
    "Exact name keys:",
    len(exact_name_index)
)


# ============================================================
# 9. NAME TOKEN FREQUENCY
# ============================================================

print("\nCounting name token frequencies...")

name_frequency = Counter()


for name in s2["name_norm"]:

    for token in get_name_tokens(name):

        name_frequency[token] += 1


for name in s3["name_norm"]:

    for token in get_name_tokens(name):

        name_frequency[token] += 1


print(
    "Unique name tokens:",
    len(name_frequency)
)


# ============================================================
# 10. NAME TOKEN INDEX
# ============================================================

print("\nBuilding name token index...")

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


print(
    "Name token keys:",
    len(name_token_index)
)


# ============================================================
# 11. EXACT ADDRESS INDEX
# ============================================================

print("\nBuilding exact-address index...")

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
# 12. ADDRESS TOKEN FREQUENCY
# ============================================================

print("\nCounting address token frequencies...")

address_frequency = Counter()


for address in s2["address_norm"]:

    for token in get_address_tokens(address):

        address_frequency[token] += 1


for address in s3["address_norm"]:

    for token in get_address_tokens(address):

        address_frequency[token] += 1


print(
    "Unique address tokens:",
    len(address_frequency)
)


# ============================================================
# 13. ADDRESS TOKEN INDEX
# ============================================================

print("\nBuilding address token index...")

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


print(
    "Address token keys:",
    len(address_token_index)
)


# ============================================================
# 14. COMBINED BLOCKING TEST
# ============================================================

print("\n" + "=" * 70)
print("COMBINED BLOCKING RECALL TEST")
print("=" * 70)


total_true_matches = 0
retrieved_true_matches = 0

s1_with_true_matches = 0
s1_with_retrieved_true_match = 0

total_candidates = 0
s1_with_candidates = 0

# Extra statistics
exact_name_hits = 0
name_token_hits = 0
exact_address_hits = 0
address_token_hits = 0


# ============================================================
# 15. EVALUATE EACH S1
# ============================================================

for i, row in enumerate(
    s1.itertuples(index=False),
    start=1
):

    country = row.country
    name = row.name_norm
    address = row.address_norm
    s1_id = row.entity_id

    candidates = set()


    # --------------------------------------------------------
    # BLOCK 1: EXACT NAME
    # --------------------------------------------------------

    if name:

        result = exact_name_index.get(
            (country, name),
            set()
        )

        if result:

            exact_name_hits += 1

            candidates.update(result)


    # --------------------------------------------------------
    # BLOCK 2: NAME TOKEN
    # --------------------------------------------------------

    if name:

        for token in get_name_tokens(name):

            if name_frequency[token] <= MAX_NAME_TOKEN_FREQUENCY:

                result = name_token_index.get(
                    (country, token),
                    set()
                )

                if result:

                    name_token_hits += 1

                    candidates.update(result)


    # --------------------------------------------------------
    # BLOCK 3: EXACT ADDRESS
    # --------------------------------------------------------

    if address:

        result = exact_address_index.get(
            (country, address),
            set()
        )

        if result:

            exact_address_hits += 1

            candidates.update(result)


    # --------------------------------------------------------
    # BLOCK 4: ADDRESS TOKEN
    # --------------------------------------------------------

    if address:

        for token in get_address_tokens(address):

            if address_frequency[token] <= MAX_ADDRESS_TOKEN_FREQUENCY:

                result = address_token_index.get(
                    (country, token),
                    set()
                )

                if result:

                    address_token_hits += 1

                    candidates.update(result)


    # --------------------------------------------------------
    # Candidate statistics
    # --------------------------------------------------------

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
# 16. FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("COMBINED BLOCKING RESULTS")
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
# 17. RECALL
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
# 18. CANDIDATE STATISTICS
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


# ============================================================
# 19. BLOCK HIT STATISTICS
# ============================================================

print("\nBlock hit statistics:")

print(
    "Exact name block hits:",
    exact_name_hits
)

print(
    "Name token block hits:",
    name_token_hits
)

print(
    "Exact address block hits:",
    exact_address_hits
)

print(
    "Address token block hits:",
    address_token_hits
)


print("\nDone.")