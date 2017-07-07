# coding=utf-8

import numpy as np
import scipy.sparse as sp

def projection_by_random(w, dims, step_size, u=None):
    """ weight is projected using randomly sampled unit vector.
    Params:
        w(csr_matrix): weight parameter
        dims(int): # of dimension of weight parameter
        step_size(float): exploration (delta) or exploitation (ganma) parameter
        u(csr_matrix): update direction (default: None). if u==None, random direction will be sampled from normal distribution
    Returns:
        w'(csr_matrix): new weight parameter
        u(csr_matrix): update direction
    """
    if u == None:
        u = np.random.uniform(size=(1, dims)) # direction for update
        u = sp.csr_matrix(u / np.linalg.norm(u), dtype=np.float32) # normalize so as to unit vector
   
    return w + step_size * u, u
