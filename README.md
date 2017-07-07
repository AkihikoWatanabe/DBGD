# DBGD
A python implementation of Dueling Bandit Gradient Descent (DBGD).

DBGD is a list-wise online learning approach based on dueling bandits problem using user's implicit feedback.

Note that, this method is commonly used in online setting (i.e. models are trained using user's realtime implicit feedback using some interleaved list), but this implementation is for offline setting (i.e. models are trained using offline training data (not realtime user's feedback)).

For details about DBGD, see the following papers:
```
Interactively optimizing information retrieval systems as a dueling bandits problem, Yue+, ICML'09.
```

# Example
## Training
```python
from updater import Updater
from weight import Weight

# number of maximum epochs
epochs = 100

# number of maximum number of features
max_feature_num = 5

# exploration parameter
delta = 1.0

# exploitation parameter
ganma = 0.1

# number of parallerization
parallel_num = 6

# metric that you want to optimize
# you can choose MAP or MRR
metric = "MAP"

# make training data
# x_train represents feature vector using dict
#       - key: qid
#	- value: feature vectors using scipy.sparse.csr_matrix
# y_train represents relevancy labels (e.g. 5 scale ratings or binary) corresponding to each feature_vector using dict
#	- key: qid
#	- value: relevancy vectors
x_train, y_train = make_data()

weight = Weight(max_feature_num)

updater = Updater(delta=delta, ganma=ganma, process_num=prallel_num, metric=metric)

for _ in xrange(epochs):
	# update weight using DBGD
	updater.update(x_train, y_train, weight)
	# dump weight parameter
	weight.dump_weight("./models/dbgd")
```

## Testing
```pythohn
from weight import Weight
import predictor import Predictor

# make test data
x_test, y_test = make_data()

# load trained weight parameters from model file
# second argument means number of epochs for weight that you want to load
weight = Weight()
weight.load_weight("./moidels/pa", 30)

predictor = Predictor()

# get result rankings for x_test
for qid, features in x_test.items():
	labels = y_test[qid]
	# ranking is represented as list and its element is composed of (true_label, case_id, score) by descending order of score
	ranking = predictor.predict_and_ranks(features, labels, weight)
```
