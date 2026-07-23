import random
import numpy as np

def set_seed(seed: int = 369):
    random.seed(seed)
    np.random.seed(seed)

def uniform_probability_centered(center: float, r:float, size=1, rounding=5):
    samples = [round(random.uniform(center - r, center + r), rounding) for _ in range(size)]
    return samples if size != 1 else samples[0]

def uniform_probability_range(low: float, high: float, size=1, rounding=5):
    samples = [round(random.uniform(low, high), rounding) for _ in range(size)]
    return samples if size != 1 else samples[0]

def choice(options: list, size=1, probabilities: list = None):
    if probabilities is not None and len(options) != len(probabilities):
        raise ValueError("Length of options and probabilities must be the same.")
    return random.choices(options, weights=probabilities, k=size)

def probability_event(prob: float) -> bool:
    return random.random() < prob