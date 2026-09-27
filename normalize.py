import re
import unicodedata


# --------------------------------------------------
# BASIC TEXT NORMALIZATION
# --------------------------------------------------

def normalize_text(text):
    """
    Basic normalization while preserving Unicode characters.
    """

    if text is None:
        return ""

    text = str(text)

    # Handle NaN
    if text.lower() == "nan":
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# BUSINESS NAME NORMALIZATION
# --------------------------------------------------

def normalize_business_name(name):

    name = normalize_text(name)

    if not name:
        return ""

    tokens = name.split()

    cleaned_tokens = []

    for token in tokens:

        # Remove immediately repeated words
        if not cleaned_tokens or token != cleaned_tokens[-1]:
            cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)


# --------------------------------------------------
# ADDRESS NORMALIZATION
# --------------------------------------------------

def normalize_address(address):

    address = normalize_text(address)

    if not address:
        return ""

    # Replace abbreviations only when they are
    # complete tokens.
    replacements = {
        "rd": "road",
        "st": "street",
        "ave": "avenue",
        "dr": "drive",
        "ln": "lane",
        "hwy": "highway",
    }

    tokens = address.split()

    normalized_tokens = []

    for token in tokens:

        if token in replacements:
            token = replacements[token]

        normalized_tokens.append(token)

    return " ".join(normalized_tokens)


# --------------------------------------------------
# TOKENIZATION
# --------------------------------------------------

def tokenize(text):

    if not text:
        return []

    return text.split()


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    examples = [
        "ABC Pvt. Ltd.",
        "Foundation  Excel Agency Private  Limited",
        "Crestline Crestline Clean LP",
        "Shivshakti Vidyalaya Vidyalaya Overseas Corporation",
        "New Delhi Techn0logies Private Limited",
        "Apollo Hospirlg",
    ]

    print("=" * 70)
    print("BUSINESS NAME NORMALIZATION")
    print("=" * 70)

    for name in examples:

        normalized = normalize_business_name(name)

        print(f"\nOriginal   : {name}")
        print(f"Normalized : {normalized}")


    addresses = [
        "12 MG Road, Mumbai",
        "105 ELM ST, MORGANTON, NC",
        "7100BRIGHAM ROAD, HENRICO COUNTY, VA",
        "46 White Rock Dr, Hurricanne, West Virginia",
    ]

    print("\n" + "=" * 70)
    print("ADDRESS NORMALIZATION")
    print("=" * 70)

    for address in addresses:

        normalized = normalize_address(address)

        print(f"\nOriginal   : {address}")
        print(f"Normalized : {normalized}")