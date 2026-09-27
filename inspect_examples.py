import pandas as pd

TRAIN = "dataset/train"

# Load only the columns we need
s1 = pd.read_csv(
    f"{TRAIN}/train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

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

print("\n" + "=" * 80)
print("SOURCE 1 EXAMPLES")
print("=" * 80)

print(
    s1[
        ["entity_id", "business_name", "business_address", "country"]
    ].head(20).to_string(index=False)
)

print("\n" + "=" * 80)
print("SOURCE 2 EXAMPLES")
print("=" * 80)

print(
    s2[
        ["entity_id", "business_name", "business_address", "country"]
    ].head(20).to_string(index=False)
)

print("\n" + "=" * 80)
print("SOURCE 3 EXAMPLES")
print("=" * 80)

print(
    s3[
        ["entity_id", "business_name", "business_address", "country"]
    ].head(20).to_string(index=False)
)

# --------------------------------------------------
# Missing-value examples
# --------------------------------------------------

print("\n" + "=" * 80)
print("SOURCE 2 - MISSING ADDRESS EXAMPLES")
print("=" * 80)

print(
    s2[
        s2["business_address"].isna()
    ][
        ["entity_id", "business_name", "business_address", "country"]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 80)
print("SOURCE 3 - MISSING ADDRESS EXAMPLES")
print("=" * 80)

print(
    s3[
        s3["business_address"].isna()
    ][
        ["entity_id", "business_name", "business_address", "country"]
    ].head(10).to_string(index=False)
)