# coding=utf-8

"""
A python implementation of Dueling Bandit Gradient Descent (DBGD).
"""

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed

from update_func import dueling_bandits

class Updater():
    """ This class support DBGD updater using Iterative Parameter Mixture (IPM).
    Note that, original paper doesn't use IPM, but I believe that IPM is also useful for DBGD training like other online learning technique.
    """

    def __init__(self, delta=1.0, ganma=0.01, process_num=1, metric="MAP")
        """
        Params:
            delta(float): exploration parameter
            ganma(float): exploitation parameter
            process_num(int): # of parallerization
            metric(str): metric for optimization (MAP or MRR)
        """
        self.delta = delta
        self.ganma = ganma
        self.METRIC = metric
        self.PROCESS_NUM = process_num
        assert self.METRIC in ["MAP", "MRR"], "invalid metric name {}".format(self.METRIC)

    def __make_minibatch(self, x_dict, y_dict):
        """
        Params:
            x_dict(dict): dict of csr_matrix of feature vectors.
            y_dict(dict): dict of np.ndarray of labels corresponding to each feature vector
        Returns:
            x_batch(list): batch of feature vectors
            y_batch(list): batch of labels
        """

        x_batch = []
        y_batch = []
        qids = x_dict.keys()
        N = len(qids) # # of qids
        np.random.seed(0) # set seed for permutation
        perm = np.random.permutation(N)

        for p in xrange(self.PROCESS_NUM):
            ini = N * (p) / self.PROCESS_NUM
            fin = N * (p + 1) / self.PROCESS_NUM
            x_batch.append({qids[idx]:x_dict[qids[idx]] for idx in perm[ini:fin]})
            y_batch.append({qids[idx]:y_dict[qids[idx]] for idx in perm[ini:fin]})

        return x_batch, y_batch

    def __iterative_parameter_mixture(self, callback, weight):
        """
        Params:
            callback: callback for parallerized process
            weight(Weight): current weight class
        """
        _w_sum = sp.csr_matrix((1, weight.dims), dtype=np.float32)
        for _w in callback:
            _w_sum += _w

        # insert updated weight
        weight.set_weight(1.0 / self.PROCESS_NUM * _w_sum)
        weight.epoch += 1

    def update(self, x_dict, y_dict, weight):
        """ Update weight parameter using DBGD.
        Params:
            x_dict(dict): dict of csr_matrix of feature vectors.
            y_dict(dict): dict of np.ndarray of labels corresponding to each feature vector
            weight(Weight): class of weight
        """
        assert len(x_dict) == len(y_dict), "invalid # of qids"
        
        x_batch, y_batch = self.__make_minibatch(x_dict, y_dict)

        callback = Parallel(n_jobs=self.PROCESS_NUM)( \
                delayed(dueling_bandits)(x_batch[i], y_batch[i], weight.get_weight(), weight., self.delta, self.ganma, self.METRIC) for i in range(self.PROCESS_NUM)) 
        self.__iterative_parameter_mixture(callback, weight)
