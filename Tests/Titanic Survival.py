# Titanic Surival Chance data tests
# From tflearn.org at "http://tflearn.org/tutorials/quickstart.html"
# Reverse Learning, 2017

import numpy as np
import tflearn
import IOA
import tensorflow as tf

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('../DataSets/titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)

# Preprocessing function
def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
      # Converting 'sex' field to float (id is 1 after removing labels column)
      data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore=[1, 6]

# Preprocess data
data = preprocess(data, to_ignore)

# Build neural network
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch = 10, batch_size = 16, show_metric = False)

test = preprocess([[3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]], to_ignore)
testTF = [tf.constant(i) for i in test]
print(model.predict([testTF]))

# Apply IOA
optimizer = IOA.IOA(model.predict, 6)
final, digest = optimizer.optimize([0.0, 1.0], epochs = -1, debug = True, debug_interval = 1,
                                   restrictions = {1: (0, 1), 2: (0, 100), 3: (0, 4), 4: (0, 2)})