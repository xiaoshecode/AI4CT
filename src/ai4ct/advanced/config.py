# ai4ct/advanced/config.py
import numpy as np

from ai4ct.paths import DICOM_DIR, LABEL_PATH, ADVANCED_OUTPUT

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = ADVANCED_OUTPUT
GPR_THRESHOLD = 95.0
N_SAMPLES = 644
TEST_SIZE  = 0.2
VAL_SIZE   = 0.125  # Historical setting: split_data divides this by (1 - TEST_SIZE).
