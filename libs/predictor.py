# coding=utf-8

class Predictor():
    """ This class predicts confidence of each cases based on given weight class.
    """
    
    def __init__(self):
        pass

    def predict(self, x_list, w):
        """
        Params:
            x_list(csr_matrix): csr_matrix of feature vectors. Each vector is represented by np.ndarray
            w(csr_matrix): weight vector for prediction 
        Returns:
            y_list(list): List of result labels (or confidence score) on the prediction
        """
        #w = weight.get_weight()
        return [w.multiply(x_list[j]).sum() for j in xrange(x_list.shape[0])]

    def predict_and_ranks(self, x_list, y_list, w):
        """ Predict and make rankings using result of prediction.
        Params:
            x_list(csr_matrix): csr_matrix of feature vectors. Each vector is represented by np.ndarray
            y_list(list): List of true labels
            w(csr_matrix): weight vector for prediction
        Returns:
            ranks(list): ranking, each element is composed of (true_label, case_id, score)
        """
        scores = self.predict(x_list, w)

        return sorted(zip(y_list, range(len(scores)), scores), key=lambda x:x[2], reverse=True)
