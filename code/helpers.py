import numpy as np

def interp(start, end, ratio):
    """
    Description: interpolation between start point set and end point set
    Input:
        start: start point set, (N, 2) or (2,)
        end: end point set, (N, 2) or (2,)
        ratio: ratio, (N, 2) or (N,)
    Output:
        mid_point: interpolated point set, (N, 2)
    Note: if ratio is (N,) vector or start/end is single point (2,), use broadcast
    """
    N = len(ratio)
    start = np.array(start).reshape(-1, 2)
    end = np.array(end).reshape(-1, 2)
    ratio = np.array(ratio).reshape(N, -1)
    
    assert len(start) in [1, N]
    assert len(end) in [1, N]
    assert ratio.shape[1] in [1, 2]
    
    return start + (end - start) * ratio


def euler_dist(pt1, pt2):
    """
    Description: distance between two point sets
    Input:
        pt1: point set 1, (N, 2) or (2, N)
        pt2: point set 1, (N, 2) or (2, N)
    Output:
        dist: distance, (N, 2)
    """
    pt1 = np.array(pt1).reshape(-1, 2)
    pt2 = np.array(pt2).reshape(-1, 2)
    dist = np.sqrt(np.sum((pt2 - pt1) ** 2), axis=1)
    
    return dist