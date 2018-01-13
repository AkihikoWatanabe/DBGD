# coding=utf-8

import numpy as np
import unittest

def reciprocal_rank(rels):
    """ Calculate Reciprocal Rank (RR). This function considers nonzero values in rels as relevant items.

    Args:
        rels(list): a list of binary relevance scores (e.g. [1, 0, 1, 0], [0, 0, 0, 1])

    Returns:
        float: Reciprocal Rank score

    For instance, the RR score of binary releance scores [1, 0, 1, 0] equals 1.0
    and [0, 0, 0, 1] equals 0.25.
    """

    nonzero_indices = np.asarray(rels).nonzero()[0]
    if has_no_relevant_items(nonzero_indices):
        # RR equals 0.0 if there is no relevant items
        return 0.0

    return 1.0 / (1.0 + nonzero_indices[0])


def precision_k(rels, k):
    """ Calculate Precision@k. This function considers nonzero values in rels as relevant items.

    Args:
        rels(list): a list of binary relevance scores (e.g. [1, 0, 0, 1, 0, 0]). 
        k(int): top k num that are used for calculating precision@k

    Returns:
        float: precision@k score

    For instance, the precision@(4, 6) of binary relevance scores [1, 0, 0, 1, 0, 0] equals 0.500 and 0.333 respectively
    """

    return np.sum([1.0 if r != 0.0 else 0.0 for r in rels[:k]]) / float(k)


def average_precision(rels):
    """ Calculate Average Precision (AP). This function considers nonzero values in rels as relevant items.

    Args:
        rels(list): a list of binary relevance scores (e.g. [1, 0, 0, 1, 0, 0]).

    Returns:
        float: Average Precision score 

    For instance, the AP score of the binary scores [1, 0, 0, 1, 0, 0] equals 0.75.
    """

    nonzero_indices = np.asarray(rels).nonzero()[0] 
    if has_no_relevant_items(nonzero_indices):
        # AP equals 0.0 if there is no relevant items
        return 0.0

    return np.sum([precision_k(rels, k+1) for k, r in enumerate(rels) if r != 0.0]) / float(len(nonzero_indices))


def mean_reciprocal_rank(rel_lists):
    """ Calculate Mean Reciprocal Rank (MRR). This function considers nonzero values in rel_lists as relevant items.

    Args:
        rel_lists(list): a two-dimensional list of binary relevance scores (e.g. [[1, 0, 1, 0], [0, 0, 0, 1]])

    Returns:
        float: Mean Reciprocal Rank score

    For instance, the MRR score of binary relevance scores [[1, 0, 1, 0], [0, 0, 0, 1]] equals 0.625.
    """

    _rel_lists = np.asarray(rel_lists)

    return np.average([reciprocal_rank(rels) for rels in _rel_lists])


def mean_average_precision(rel_lists):
    """ Calculate Mean Average Precision (MRR). This function considers nonzero values in rel_lists as relevant items.

    Args:
        rel_lists(list): a two-dimensional list of binary relevance scores (e.g. [[1, 0, 1, 0], [0, 0, 0, 1]])

    Returns:
        float: Mean Average Precision score

    For instance, the MAP score of binary relevance scores [[1, 0, 1, 0], [0, 0, 0, 1]] equals 0.542
    """

    return np.average([average_precision(rels) for rels in rel_lists])


def has_no_relevant_items(nonzero_indices):
    
    if len(nonzero_indices) == 0:
        return True
    else:
        return False


class TestMetrics(unittest.TestCase):

    def test_metrics(self):
        self.assertEqual(round(reciprocal_rank([1, 0, 1, 0]), 3), 1.000)
        self.assertEqual(round(reciprocal_rank([0, 0, 0, 1]), 3), 0.250)
        self.assertEqual(round(precision_k([1, 0, 0, 1, 0, 0], 4), 3), 0.500)
        self.assertEqual(round(precision_k([1, 0, 0, 1, 0, 0], 6), 3), 0.333)
        self.assertEqual(round(average_precision([1, 0, 0, 1, 0, 0]), 3), 0.750)
        self.assertEqual(round(mean_reciprocal_rank([[1, 0, 1, 0], [0, 0, 0, 1]]), 3), 0.625)
        self.assertEqual(round(mean_average_precision([[1, 0, 1, 0], [0, 0, 0, 1]]), 3), 0.542)


if __name__ == "__main__":
    unittest.main()
