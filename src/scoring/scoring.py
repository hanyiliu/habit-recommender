# Requires from preprocessing:
#   sequences  — (N, 48) int numpy array, one row per respondent, values 0–10
#                produced by src/data/preprocessing/preprocessing.py
#                (category mapping must match the constants defined below)

import numpy as np
from sklearn.cluster import KMeans

SLEEP, EXERCISE, WORK, EDUCATION, EATING = 0, 1, 2, 3, 4
SOCIALIZING, LEISURE, HOUSEHOLD, GROOMING, TRAVEL, OTHER = 5, 6, 7, 8, 9, 10

CATEGORY_NAMES = (
    "Sleep", "Exercise", "Work", "Education", "Eating",
    "Socializing", "Leisure/Screen", "Household", "Grooming", "Travel", "Other",
)

NUM_CATEGORIES = 11
NUM_SLOTS = 48  # slot n = 04:00 AM + n × 30 min  (ATUS diary starts at 4:00 AM)

# Meal anchors: nearest Eating slot is scored by Gaussian proximity
BREAKFAST_ANCHOR = 6    # 07:00 AM
LUNCH_ANCHOR = 16       # 12:00 PM
DINNER_ANCHOR = 28      # 06:00 PM
MEAL_ANCHORS = (BREAKFAST_ANCHOR, LUNCH_ANCHOR, DINNER_ANCHOR)
MEAL_SIGMA = 2.0        # ~1-hour tolerance (half-width in slots)

LATE_EVENING_START = 32   # 08:00 PM — screen-time penalty window begins
LATE_EVENING_END = 40     # 12:00 AM
LATE_NIGHT_START = 36     # 10:00 PM — work-bleed penalty window begins

CDC_SLEEP_SLOTS = 14      # 7 h, CDC adult minimum
EXERCISE_SATURATION = 2   # 1 h, beyond the WHO 30-min/day minimum

# Component order: sleep_contiguity, exercise_presence, meal_regularity,
#                  screen_time, work_structure
DEFAULT_WEIGHTS = np.array([0.30, 0.25, 0.20, 0.15, 0.10])


# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------

def _longest_run(seq, category):
    max_run = current = 0
    for s in seq:
        if s == category:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


def score_sleep_contiguity(seq):
    return min(_longest_run(seq, SLEEP) / CDC_SLEEP_SLOTS, 1.0)


def score_exercise_presence(seq):
    count = sum(1 for s in seq if s == EXERCISE)
    return min(count / EXERCISE_SATURATION, 1.0)


def score_meal_regularity(seq):
    seq = np.asarray(seq)
    eating_slots = np.where(seq == EATING)[0]
    if len(eating_slots) == 0:
        return 0.0
    total = 0.0
    for anchor in MEAL_ANCHORS:
        nearest = np.abs(eating_slots - anchor).min()
        total += np.exp(-0.5 * (nearest / MEAL_SIGMA) ** 2)
    return total / len(MEAL_ANCHORS)


def score_screen_time(seq):
    seq = np.asarray(seq)
    count = np.sum(seq[LATE_EVENING_START:LATE_EVENING_END] == LEISURE)
    return 1.0 - count / (LATE_EVENING_END - LATE_EVENING_START)


def score_work_structure(seq):
    seq = np.asarray(seq)
    count = np.sum(seq[LATE_NIGHT_START:] == WORK)
    return 1.0 - count / (NUM_SLOTS - LATE_NIGHT_START)


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def compute_health_score(seq, weights=None):
    """Return a scalar health score in [0, 1] for a 48-slot activity sequence."""
    w = DEFAULT_WEIGHTS if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    components = np.array([
        score_sleep_contiguity(seq),
        score_exercise_presence(seq),
        score_meal_regularity(seq),
        score_screen_time(seq),
        score_work_structure(seq),
    ])
    return float(w @ components)


# ---------------------------------------------------------------------------
# Routine builder
# ---------------------------------------------------------------------------

def build_routines(sequences, K, threshold=0.5, min_cluster_size=30,
                   weights=None, random_state=42):
    """
    Cluster respondents, score individuals, and average the top-scorers in each
    cluster to produce one optimal routine vector per cluster.

    Args:
        sequences        (N, 48) int array — one row per respondent
        K                number of k-means clusters
        threshold        top fraction of scorers kept per cluster (e.g. 0.5 = top 50%)
        min_cluster_size clusters with fewer qualifying respondents are merged into
                         the nearest valid cluster
        weights          health-score component weights (length 5); uses DEFAULT_WEIGHTS if None
        random_state     passed to KMeans for reproducibility

    Returns:
        routines  (R, 48) float — one routine per surviving cluster (R ≤ K)
        labels    (N,)    int   — cluster index per respondent after merging
        scores    (N,)    float — per-respondent health score
    """
    sequences = np.asarray(sequences, dtype=float)

    kmeans = KMeans(n_clusters=K, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(sequences).copy()
    centroids = kmeans.cluster_centers_

    scores = np.array([compute_health_score(seq, weights) for seq in sequences])

    def _qualifying_count(k, current_labels):
        mask = current_labels == k
        if not mask.any():
            return 0
        cutoff = np.quantile(scores[mask], 1.0 - threshold)
        return int(np.sum(scores[mask] >= cutoff))

    valid = {k for k in range(K) if _qualifying_count(k, labels) >= min_cluster_size}
    invalid = set(range(K)) - valid

    # reassign each invalid cluster's respondents to the nearest valid centroid
    if invalid and valid:
        valid_arr = np.array(sorted(valid))
        for k in invalid:
            dists = np.linalg.norm(centroids[valid_arr] - centroids[k], axis=1)
            labels[labels == k] = valid_arr[np.argmin(dists)]

    # build one routine per valid cluster using final (possibly merged) labels
    routines = []
    for k in sorted(valid):
        mask = labels == k
        cluster_scores = scores[mask]
        cluster_seqs = sequences[mask]
        cutoff = np.quantile(cluster_scores, 1.0 - threshold)
        top_seqs = cluster_seqs[cluster_scores >= cutoff]
        routines.append(top_seqs.mean(axis=0))

    return np.array(routines), labels, scores


# ---------------------------------------------------------------------------
# Ablation sweep
# ---------------------------------------------------------------------------

def run_routine_sweep(sequences, K_values=(5, 10, 20, 30),
                      thresholds=(0.25, 0.50, 0.75)):
    """
    Run build_routines over every (K, threshold) combination.

    Returns:
        dict keyed by (K, threshold) → (routines, labels, scores)
    """
    results = {}
    for K in K_values:
        for t in thresholds:
            results[(K, t)] = build_routines(sequences, K=K, threshold=t)
    return results