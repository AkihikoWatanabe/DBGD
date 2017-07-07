# coding=utf-8

import numpy as np
import scipy.sparse as sp
import gzip
import cPickle

class Weight():
    """ Class for weight parameter for DBGD. 
    """

    def __init__(self, dims=100000):
        """
        Params:
            dims(int): # of dimension for weight vector (default:100000)
        """
        self.dims = dims
        self.w = sp.csr_matrix((1, dims), dtype=np.float32) # weight parameter
        self.epoch = 0

    def set_weight(self, new_weight):
        """
        Params:
            new_weight(csr_marix): new weight parameter to set
        """
        self.w = new_weight

    def get_weight(self):

        return self.w

    def dump_weight(self, path):
        """ Dump weight vector
        Params:
            path(str): path to dump directory
        """        
        np.savez(path+".epoch{epoch_num}".format(epoch_num=self.epoch), data=self.w.data, indices=self.w.indices, indptr=self.w.indptr, dims=[self.dims])

    def load_weight(self, path, epoch):
        """ Dump weight vector
        Params:
            path(str): path to dump directory
            epoch(int): number of epochs
        """        
        data = np.load(path+".epoch{epoch_num}.npz".format(epoch_num=epoch))
        dims = data["dims"][0]
        self.w = sp.csr_matrix((data["data"], data["indices"], data["indptr"]), (1, dims))
        self.epoch = epoch

    def extend_weight_dims(self, dims):
        """ Extend # of dimensions on weight vector.
        Params:
            dims(int): # of dimensions to extend
        """
        self.w = sp.csr_matrix((self.w.data, self.w.indices, self.w.indptr), (1, dims))
        self.dims = dims
