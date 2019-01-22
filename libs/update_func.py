# coding=utf-8

from projection import projection_by_random
from predictor import Predictor
from metrics import mean_reciprocal_rank, mean_average_precision
from tqdm import tqdm
import numpy as np
from random import random

def dueling_bandits(x_dict, y_dict, weight, dims, delta, ganma, metric):
    """ Update function for DBGD.
    Params:
        x_dict(dict): dict of csr_matrix of feature vectors.
        y_dict(dict): dict of np.ndarray of labels corresponding to each feature vector
        weight(csr_matrix): weight vector
        dims(int): # of dimensions of weight vector
        delta(float): exploration parameter
        ganma(float): exploitation parameter
        metric(str): metric for optimization (MAP or MRR)
    Returns:
        weight: updated weight

    Note that 
    """

    if metric == "MRR":
        calc_metric = mean_reciprocal_rank
    elif metric == "MAP":
        calc_metric = mean_average_precision

    for qid, features in tqdm(x_dict.items()):
        cand_w, u = projection_by_random(weight, dims, delta)
        # If you would like to use your own duel function,
        # you need to replace "duel_by_offline_data" with your duel function.
        if duel_by_offline_data(features, y_dict[qid], cand_w, weight, calc_metric):
            weight, _ = projection_by_random(weight, dims, ganma, u=u) # update weight

    return weight

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def is_cand_beats_curr(prob):
    if prob > random():
        return True
    else:
        return False
    
def duel_by_offline_data(features, true_labels, cand_w, current_w, calc_metric):
    """ Duel function to compare candidate weight with current weight by offline data.
    Note that, this offline duel might be not standard settings for dueling bandit gradient descent (DBGD).
    In DBGD, models should be trained based on real-time user feedback data using some interleaved lists.
    If you would like to train models based on such setting,
    you need to implement your own duel function and compare candidate weight with current weight.

    Args:
        features(csr_matrix): csr_matrix of feature vectors. Each vector is represented by np.ndarray
        true_labels(list): List of true labels 
        cand_w(csr_matrix): candidate weight vector 
        current_w(csr_matrix): current weight vector  
        calc_metric(function): function that calculate ranking metric
    Returns:
        bool: booling value whether candidate weight wins current weight or not .
    """
    C = 10.0
    
    # make rankings 
    predictor = Predictor()
    current_rank = [gold for (gold, _, _) in predictor.predict_and_ranks(features, true_labels, current_w)]
    cand_rank = [gold for (gold, _, _) in predictor.predict_and_ranks(features, true_labels, cand_w)]
 
    v_curr = C * calc_metric([current_rank])  # value for current weight
    v_cand = C * calc_metric([cand_rank])  # value for candidate weight
    prob_cand_beats_curr = 1.0 - sigmoid(v_curr - v_cand)
    if is_cand_beats_curr(prob_cand_beats_curr):
        # candidate weight wins current weight
        return True

    return False
