"""
Central configuration: paths, constants, and model hyperparameters.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
TRAIN_CSV = "data/train.csv"
TEST_CSV  = "data/test.csv"

# ── Category mapping ──────────────────────────────────────────────────────────
# YouTube video categories A–H encoded as integers
CATEGORY_MAP = {"A": 1, "B": 2, "C": 3, "D": 4,
                "E": 5, "F": 6, "G": 7, "H": 8}

# ── Preprocessing ─────────────────────────────────────────────────────────────
NUMERIC_COLS   = ["views", "likes", "dislikes", "comment"]
ADVIEW_CAP     = 2_000_000   # filter rows with adview > this value (outliers)
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

# ── Random Forest hyperparameters ─────────────────────────────────────────────
RF_N_ESTIMATORS    = 200
RF_MAX_DEPTH       = 25
RF_MIN_SAMPLES_SPLIT = 15
RF_MIN_SAMPLES_LEAF  = 2

# ── ANN hyperparameters ───────────────────────────────────────────────────────
ANN_EPOCHS     = 100
ANN_HIDDEN     = 6       # neurons in each hidden layer

# ── Saved model paths ─────────────────────────────────────────────────────────
DT_MODEL_PATH  = "models/decisiontree_youtubeadview.pkl"
ANN_MODEL_PATH = "models/ann_youtubeadview.keras"
SCALER_PATH    = "models/scaler.pkl"

# ── Output ────────────────────────────────────────────────────────────────────
PREDICTIONS_CSV = "reports/PredictedAdview.csv"
