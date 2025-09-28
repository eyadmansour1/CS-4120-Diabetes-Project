import numpy as np
import random

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
