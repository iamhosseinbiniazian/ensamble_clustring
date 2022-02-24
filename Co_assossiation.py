import numpy as np
def ComputeCoassossiation(l):
    res = [[int(x == y) for y in l] for x in l]
    return np.array(res)