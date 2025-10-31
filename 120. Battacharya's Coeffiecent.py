import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    if len(p) != len(q) or len(p) == 0:
        return 0.0
    p, q = np.array(p), np.array(q)
    bc = np.sum(np.sqrt(p * q))
    if bc <= 0:
        return 0.0
    return round(float(-np.log(bc)), 4)
