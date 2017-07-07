# coding=utf-8

from projection import projection_by_random
from predictor import Predictor
from rank_metrics import mean_reciprocal_rank, mean_average_precision,  

def dueling_bandits(x_dict, y_dict, weight, dims, delta, ganma, metric):
    """ Update function for DBGD.
    Params:
        x_dict(dict): dict of csr_matrix of feature vectors.
        y_dict(dict): dict of np.ndarray of labels corresponding to each feature vector
        weight: weight vector
        dims: # of dimensions of weight vector
        delta: exploration parameter
        ganma: exploitation parameter
        metric: metric for optimization (MAP or MRR)
    Returns:
        weight: updated weight
    """

    if metric == "MRR":
        calc_metric = mean_reciprocal_rank
    elif metric == "MAP":
        calc_metric = mean_average_precision
    predictor = Predictor()

    for qid, features in x_dict.items():
        cand_w, u = projection_by_random(weight, dims, delta)
        # make rankings
        current_rank = [gold for (gold, _, _) in predictor.predict_and_ranks(features, y_dict[qid], weight)]
        cand_rank = [gold for (gold, _, _) in predictor.predict_and_ranks(features, y_dict[qid], cand_w)
        if calc_metric([current_rank]) < calc_metric([cand_rank]):
            weight, _ = projection_by_random(weight, dims, ganma, u=u) # update weight

    return weight
