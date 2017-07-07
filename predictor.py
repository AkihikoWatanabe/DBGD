# coding=utf-8

class Predictor():
    """ This class predicts confidence of each cases based on given weight class.
    """
    
    def __init__(self):
        pass

    def predict(self, x_list, weight):
        """
        Params:
            x_list(csr_matrix): csr_matrix of feature vectors. Each vector is represented by np.ndarray
            weight(Weight): weight class to use prediction
        Returns:
            y_list(list): List of result labels (or confidence score) on the prediction
        """
        w = weight.get_weight()
        return [w.multiply(x_list[j]).sum() for j in xrange(x_list.shape[0])]

    def predict_and_ranks(self, x_list, y_list, weight):
        """ Predict and make rankings using result of prediction.
        Params:
            x_list(csr_matrix): csr_matrix of feature vectors. Each vector is represented by np.ndarray
            y_list(list): List of true labels
            weight(Weight): weight class to use prediction
        Returns:
            ranks(list): ranking, each element is composed of (true_label, case_id, score)
        """
        scores = self.predict(x_list, weight)

        return sorted(zip(y_list, range(len(scores)), scores), key=lambda x:x[1], reverse=True)
