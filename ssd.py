import numpy as np
import tensorflow as tf

CLASSES = ["background", "boat", "person"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading Model...")
