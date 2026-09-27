import pandas as pd
import re
import os
from collections import defaultdict

# ============================================================
# SETTINGS
# ============================================================

DATASET = "dataset/test"
OUTPUT = "output"

os.makedirs(OUTPUT, exist_ok=True)

# Maximum candidates considered for each Source 1 record
MAX_CANDIDATES = 100

# Maximum matches written for each Source 1 record
TOP_MATCHES = 20

# Optimized threshold from validation
MATCH_THRESHOLD = 12


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = text.replace("&", " and ")

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokens(text):
    if not text:
        return set()

    return set(text.split())


def jaccard(a, b):
    if not a or not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading test Source 1...")

s1 = pd.read_csv(
    f"{DATASET}/test_source1.tsv",
    sep="\t"
)

print("Source 1:", len(s1))


print("Loading test Source 2...")

s2 = pd.read_csv(
    f"{DATASET}/test_source2.tsv",
    sep="\t"
)

print("Source 2:", len(s2))


print("Loading test Source 3...")

s3 = pd.read_csv(
    f"{DATASET}/test_source3.tsv",
    sep="\t"
)

print("Source 3:", len(s3))


# ============================================================
# NORMALIZE DATA
# ============================================================

print("\nNormalizing data...")

for df in [s1, s2, s3]:

    df["name_norm"] = df["business_name"].map(normalize)

    df["address_norm"] = df["business_address"].map(normalize)

    df["name_tokens"] = df["name_norm"].map(tokens)

    df["address_tokens"] = df["address_norm"].map(tokens)


print("Normalization complete.")


# ============================================================
# COMBINE SOURCE 2 + SOURCE 3
# ============================================================

targets = pd.concat(
    [
        s2,
        s3
    ],
    ignore_index=True
)

print(
    "Total target records:",
    len(targets)
)


# ============================================================
# BUILD BLOCKING INDEXES
# ============================================================

print("\nBuilding indexes...")


name_index = defaultdict(set)

address_index = defaultdict(set)


for idx, row in targets.iterrows():

    # --------------------------------------------------------
    # Exact normalized name index
    # --------------------------------------------------------

    name = row["name_norm"]

    if name:
        name_index[name].add(idx)


    # --------------------------------------------------------
    # Address token index
    # --------------------------------------------------------

    for token in row["address_tokens"]:

        if len(token) >= 4:

            address_index[token].add(idx)


print(
    "Name index keys:",
    len(name_index)
)

print(
    "Address index keys:",
    len(address_index)
)


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(row):

    candidates = set()


    # --------------------------------------------------------
    # Exact normalized name
    # --------------------------------------------------------

    name = row["name_norm"]

    if name and name in name_index:

        candidates.update(
            name_index[name]
        )


    # --------------------------------------------------------
    # Address token blocking
    # --------------------------------------------------------

    for token in row["address_tokens"]:

        if len(token) < 4:
            continue

        ids = address_index.get(token)

        if ids:

            candidates.update(ids)


        # Prevent very large candidate sets
        if len(candidates) >= MAX_CANDIDATES * 5:

            break


    # --------------------------------------------------------
    # Limit candidates
    # --------------------------------------------------------

    if len(candidates) > MAX_CANDIDATES:

        candidates = list(candidates)[
            :MAX_CANDIDATES
        ]

    else:

        candidates = list(candidates)


    return candidates


# ============================================================
# SCORE-E
# ============================================================

def score_match(s1row, target):

    name1 = s1row["name_norm"]
    name2 = target["name_norm"]

    addr1 = s1row["address_norm"]
    addr2 = target["address_norm"]


    name_tokens_1 = s1row["name_tokens"]
    name_tokens_2 = target["name_tokens"]

    addr_tokens_1 = s1row["address_tokens"]
    addr_tokens_2 = target["address_tokens"]


    # --------------------------------------------------------
    # Exact name match
    # --------------------------------------------------------

    name_exact = (
        1
        if name1 and name1 == name2
        else 0
    )


    # --------------------------------------------------------
    # Exact address match
    # --------------------------------------------------------

    address_exact = (
        1
        if addr1 and addr1 == addr2
        else 0
    )


    # --------------------------------------------------------
    # Token overlap
    # --------------------------------------------------------

    name_overlap = len(
        name_tokens_1 &
        name_tokens_2
    )


    address_overlap = len(
        addr_tokens_1 &
        addr_tokens_2
    )


    # --------------------------------------------------------
    # Jaccard similarity
    # --------------------------------------------------------

    name_jaccard = jaccard(
        name_tokens_1,
        name_tokens_2
    )


    address_jaccard = jaccard(
        addr_tokens_1,
        addr_tokens_2
    )


    # ========================================================
    # SCORE-E
    # ========================================================

    score = (

        15 * name_exact

        + 15 * address_exact

        + 1 * min(name_overlap, 8) / 8

        + 1 * min(address_overlap, 20) / 20

        + 10 * name_jaccard

        + 10 * address_jaccard
    )


    return score


# ============================================================
# OUTPUT FILES
# ============================================================

matching_file = (
    f"{OUTPUT}/matching_results.tsv"
)

candidate_file = (
    f"{OUTPUT}/candidate_pairs.tsv"
)


matching_out = open(
    matching_file,
    "w",
    encoding="utf-8"
)

candidate_out = open(
    candidate_file,
    "w",
    encoding="utf-8"
)


# ============================================================
# HEADERS
# ============================================================

matching_out.write(
    "source1_entity_id\tmatched_entity_ids\n"
)

candidate_out.write(
    "source1_entity_id\tcandidate_entity_ids\n"
)


# ============================================================
# FINAL MATCHING
# ============================================================

print("\n" + "=" * 70)

print("FINAL MATCHING")

print("=" * 70)


for counter, (_, row) in enumerate(
    s1.iterrows(),
    start=1
):

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    candidate_ids = generate_candidates(row)


    # --------------------------------------------------------
    # Score candidates
    # --------------------------------------------------------

    scored = []


    for idx in candidate_ids:

        target = targets.iloc[idx]

        score = score_match(
            row,
            target
        )

        scored.append(
            (
                score,
                target["entity_id"]
            )
        )


    # --------------------------------------------------------
    # Sort candidates by score
    # --------------------------------------------------------

    scored.sort(
        reverse=True
    )


    # ========================================================
    # CANDIDATE OUTPUT
    # ========================================================

    candidate_entity_ids = []

    seen = set()


    for _, entity_id in scored:

        if entity_id not in seen:

            candidate_entity_ids.append(
                entity_id
            )

            seen.add(entity_id)


    candidate_out.write(
        f"{row['entity_id']}\t"
        f"{','.join(candidate_entity_ids)}\n"
    )


    # ========================================================
    # MATCHING OUTPUT
    # ========================================================

    matches = []


    for score, entity_id in scored[:TOP_MATCHES]:

        # Optimized threshold = 12
        if score >= MATCH_THRESHOLD:

            matches.append(
                entity_id
            )


    # --------------------------------------------------------
    # Remove duplicate IDs
    # --------------------------------------------------------

    matches = list(
        dict.fromkeys(matches)
    )


    matching_out.write(
        f"{row['entity_id']}\t"
        f"{','.join(matches)}\n"
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if counter % 5000 == 0:

        percentage = (
            counter / len(s1)
        ) * 100

        print(
            f"Processed "
            f"{counter:,}/"
            f"{len(s1):,} S1 records "
            f"({percentage:.1f}%)"
        )


# ============================================================
# CLOSE FILES
# ============================================================

matching_out.close()

candidate_out.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print("COMPLETE")

print("=" * 70)


print(
    "Matching file:",
    matching_file
)

print(
    "Candidate file:",
    candidate_file
)

print(
    "\nThreshold used:",
    MATCH_THRESHOLD
)

print("\nDone.")