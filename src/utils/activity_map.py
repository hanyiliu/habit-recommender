# Maps ATUS TUTIER1CODE (2-digit) to one of 11 consolidated activity categories.
# Tier-1 codes are the first two digits of the 6-digit TRCODE.
#
# ATUS Tier-1 structure (from official ATUS lexicon):
#   01 - Personal Care (sleep, grooming, health)
#   02 - Household Activities
#   03 - Caring for Household Members
#   04 - Caring for Non-Household Members
#   05 - Work & Work-Related
#   06 - Education
#   07 - Consumer Purchases
#   08 - Professional & Personal Care Services
#   09 - Household Services
#   10 - Government Services & Civic Obligations
#   11 - Eating & Drinking
#   12 - Socializing, Relaxing & Leisure
#   13 - Sports, Exercise & Recreation
#   14 - Religious & Spiritual Activities
#   15 - Volunteer Activities
#   16 - Telephone Calls
#   18 - Travel
#   50 - Data collection (interviewer/coding artifacts — treat as Other)

# ---------------------------------------------------------------------------
# Tier-1 → 11-category mapping
# ---------------------------------------------------------------------------
# Within Tier-1 = 01 (Personal Care), we split Sleep vs Grooming using
# Tier-2 codes:
#   0101 = Sleeping  → Sleep
#   0102 = Sleeplessness → Sleep
#   0103 = Medical care → Other
#   0104 = Personal activities (grooming, etc.) → Grooming
#   0105 = Personal/private activities → Grooming

TIER1_TO_CATEGORY = {
    1:  "Sleep",        # Personal Care — default; refined by TIER2 below
    2:  "Household",    # Household Activities
    3:  "Household",    # Caring for Household Members (childcare, eldercare)
    4:  "Household",    # Caring for Non-Household Members
    5:  "Work",         # Work & Work-Related Activities
    6:  "Education",    # Education
    7:  "Other",        # Consumer Purchases (shopping)
    8:  "Other",        # Professional & Personal Care Services
    9:  "Other",        # Household Services (plumber, contractor, etc.)
    10: "Other",        # Government Services & Civic Obligations
    11: "Eating",       # Eating & Drinking
    12: "Socializing",  # Socializing, Relaxing & Leisure
    13: "Exercise",     # Sports, Exercise & Recreation
    14: "Other",        # Religious & Spiritual Activities
    15: "Other",        # Volunteer Activities
    16: "Socializing",  # Telephone Calls (treated as social)
    18: "Travel",       # Travel
    50: "Other",        # Data collection artifacts
}

# Tier-2 overrides within Tier-1 = 01 (Personal Care)
# Key = TUTIER2CODE (integer), Value = category
TIER1_01_TIER2_OVERRIDES = {
    1: "Sleep",     # 0101 Sleeping
    2: "Sleep",     # 0102 Sleeplessness
    3: "Other",     # 0103 Medical and care activities
    4: "Grooming",  # 0104 Personal activities
    5: "Grooming",  # 0105 Personal/private activities
}

# Tier-2 overrides within Tier-1 = 12 (Socializing/Leisure)
# Key = TUTIER2CODE (integer), Value = category
TIER1_12_TIER2_OVERRIDES = {
    1:  "Socializing",      # 1201 Socializing and communicating
    2:  "Leisure/Screen",   # 1202 Attending/hosting social events
    3:  "Leisure/Screen",   # 1203 Relaxing and leisure (TV, reading, etc.)
    4:  "Leisure/Screen",   # 1204 Arts and entertainment (not sports)
    5:  "Socializing",      # 1205 Waiting (socializing context)
    99: "Other",            # 1299 Unspecified leisure
}


def get_category(tier1: int, tier2: int = None) -> str:
    """
    Return one of 11 consolidated activity categories given ATUS tier codes.

    Args:
        tier1: TUTIER1CODE as integer (e.g. 1, 5, 13)
        tier2: TUTIER2CODE as integer (e.g. 1, 3); used to refine Tier-1 = 01 and 12

    Returns:
        One of: Sleep, Grooming, Work, Education, Eating, Socializing,
                Leisure/Screen, Household, Exercise, Travel, Other
    """
    tier1 = int(tier1) if tier1 is not None else -1

    if tier1 == 1 and tier2 is not None:
        return TIER1_01_TIER2_OVERRIDES.get(int(tier2), "Other")

    if tier1 == 12 and tier2 is not None:
        return TIER1_12_TIER2_OVERRIDES.get(int(tier2), "Leisure/Screen")

    return TIER1_TO_CATEGORY.get(tier1, "Other")


# ---------------------------------------------------------------------------
# Convenience: apply to a whole DataFrame row
# ---------------------------------------------------------------------------

def map_activity_category(row) -> str:
    """
    Apply to a DataFrame row with TUTIER1CODE and TUTIER2CODE columns.

    Usage:
        df["CATEGORY"] = df.apply(map_activity_category, axis=1)
    """
    return get_category(row["TUTIER1CODE"], row.get("TUTIER2CODE"))


# ---------------------------------------------------------------------------
# All 11 valid category labels
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Sleep",
    "Grooming",
    "Work",
    "Education",
    "Eating",
    "Socializing",
    "Leisure/Screen",
    "Household",
    "Exercise",
    "Travel",
    "Other",
]

CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}