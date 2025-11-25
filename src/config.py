# --- PATHS ---
RAW_DATA_DIR = 'data/raw/'
PROCESSED_BASE_DIR = 'data/processed/'
MODEL_BASE_DIR = 'models/'

# --- SIGNAL PROCESSING ---
FS = 360
WINDOW_SIZE = 2000 

# --- EXPERIMENT PROTOCOL (High Score Mode) ---
# We switch from specific patients to a random percentage.
TEST_SPLIT_RATIO = 0.2  # 20% for testing
RANDOM_SEED = 42
NUM_CORES = 4