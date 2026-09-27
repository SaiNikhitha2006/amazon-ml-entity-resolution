# Amazon ML Challenge 2026 – Business Entity Resolution

## Overview

This project was developed for the Amazon ML Challenge 2026 Business Entity Resolution problem.

The objective is to identify matching business records across multiple data sources despite noisy and inconsistent business names and addresses.

## Approach

The solution uses:

- Data normalization
- Blocking and candidate generation
- Exact matching
- Name similarity
- Address similarity
- Token overlap
- Jaccard similarity
- Feature engineering
- Threshold-based matching

## Matching Pipeline

1. Load and preprocess business records
2. Normalize names and addresses
3. Generate candidate pairs using blocking
4. Calculate matching features
5. Score candidate matches
6. Apply a matching threshold
7. Generate the final submission files

## Features Used

- Country match
- Exact name match
- Exact address match
- Name overlap
- Address overlap
- Name Jaccard similarity
- Address Jaccard similarity

## Threshold Optimization

The local training evaluation tested multiple thresholds.

The best result in the local threshold experiment was:

- Threshold: 12
- F0.5: 0.578705

This local evaluation result is separate from the challenge leaderboard score.

## Challenge Result

Public leaderboard score at the time of documentation:

**0.442**

The generated submission passed the provided submission validator with ID checking enabled.

## Technologies

- Python
- Pandas
- Entity Resolution
- Blocking
- Feature Engineering
- Similarity Matching
- Jaccard Similarity

## Project Structure

```text
├── final_matcher.py
├── optimize_threshold.py
├── match_features_stream.py
├── match_features_memory_safe.py
├── evaluate_candidates.py
├── blocking.py
├── blocking_token.py
├── blocking_address.py
├── blocking_combined.py
├── blocking_fuzzy.py
├── analyze_missed_matches.py
├── Documentation_template.md
└── utils/
    └── validate_submission.py