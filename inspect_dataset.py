import pandas as pd

TRAIN = "dataset/train"

# --------------------------------------------------
# 1. COUNTRY DISTRIBUTION
# --------------------------------------------------

files = {
    "SOURCE1": f"{TRAIN}/train_source1.tsv",
    "SOURCE2": f"{TRAIN}/train_source2.tsv",
    "SOURCE3": f"{TRAIN}/train_source3.tsv",
}

for name, path in files.items():

    print("\n" + "=" * 70)
    print(name, "- COUNTRY DISTRIBUTION")
    print("=" * 70)

    df = pd.read_csv(path, sep="\t")

    print(df["country"].value_counts(dropna=False))

    print("\nCountry percentages:")
    print(
        (df["country"].value_counts(normalize=True) * 100)
        .round(2)
    )


# --------------------------------------------------
# 2. GROUND TRUTH MATCH COUNTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("GROUND TRUTH MATCH ANALYSIS")
print("=" * 70)

gt = pd.read_csv(
    f"{TRAIN}/train_ground_truth.tsv",
    sep="\t"
)

# Empty/NaN = zero matches
gt["match_count"] = (
    gt["matched_entity_ids"]
    .fillna("")
    .apply(
        lambda x: 0 if x.strip() == ""
        else len(x.split(","))
    )
)

print("\nMatch count distribution:")
print(gt["match_count"].value_counts().sort_index())

print("\nBasic statistics:")
print(gt["match_count"].describe())


# --------------------------------------------------
# 3. SINGLETONS
# --------------------------------------------------

singletons = (gt["match_count"] == 0).sum()

print("\nNumber of singleton S1 entities:", singletons)

print(
    "Percentage of S1 entities that are singletons:",
    round(singletons / len(gt) * 100, 2),
)


# --------------------------------------------------
# 4. MULTIPLE MATCHES
# --------------------------------------------------

print("\nNumber with at least one match:",
      (gt["match_count"] > 0).sum())

print("Number with multiple matches:",
      (gt["match_count"] > 1).sum())


# --------------------------------------------------
# 5. SAMPLE MATCHES
# --------------------------------------------------

print("\nSample ground-truth matches:")

print(
    gt[
        gt["matched_entity_ids"].notna()
    ][
        ["source1_entity_id", "matched_entity_ids", "match_count"]
    ].head(10).to_string(index=False)
)