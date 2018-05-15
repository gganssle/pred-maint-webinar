import time
import os

import pandas as pd
import numpy as np

import functools

from tqdm import tqdm, trange

from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
tf.enable_eager_execution()

import sequency
seq = sequency.sequencer()

train_df = pd.read_csv('../dat/clean_train.csv')
test_df  = pd.read_csv('../dat/clean_test.csv')

# pick a large window size of 50 cycles
sequence_length = 50

# pick the feature columns
sensor_cols = ['s' + str(i) for i in range(1,22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)

# instantiate generators
seq_gen = (list(seq.gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols))
           for id in train_df['id'].unique())
label_gen = [seq.gen_labels(train_df[train_df['id']==id], sequence_length, ['label1'])
             for id in train_df['id'].unique()]

# generate sequences
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
label_array = np.concatenate(label_gen).astype(np.int32)

# build model
batch_size = 15631

class simplelstm():
    def __init__(self, batch_size=batch_size, lstm1_size=100, lstm2_size=50):
        super(simplelstm, self).__init__()
        self.lstm1_size = lstm1_size
        self.lstm2_size = lstm2_size

    def forward(self, inpt, train):
        if train:
            prob = 0.2
        else:
            prob = 1

        network = tf.nn.rnn_cell.LSTMCell(self.lstm1_size)
        network = tf.contrib.rnn.DropoutWrapper(network, output_keep_prob=prob)
        network = tf.nn.rnn_cell.LSTMCell(self.lstm2_size)
        network = tf.contrib.rnn.DropoutWrapper(network, output_keep_prob=prob)
        output, lstm_state = tf.nn.dynamic_rnn(network, inpt, dtype=tf.float32)

        output = tf.layers.dense(inputs=lstm_state.h, units=2, activation=None)
        output = tf.contrib.layers.softmax(output)

        return output, lstm_state

# instantiate model and optimizer
model = simplelstm()
optim = tf.train.AdamOptimizer(learning_rate=0.001)

# one hot encoding
enc = OneHotEncoder()

labels = enc.fit_transform(label_array).todense()

# batching
samples = tf.constant(seq_array[0:batch_size], dtype='float32')
labels = tf.constant(label_array[0:batch_size], dtype='float32')

# loss fn
def loss_fn(model, labels, samples):
    samples = tf.constant(seq_array[0:batch_size], dtype='float32') # vsh
    labels = tf.constant(label_array[0:batch_size], dtype='float32') # vsh
    # above, vsh stands for very shitty hack. Shwhy does functools.partial
    # destroy tensor shape?

    pred, _ = model.forward(samples, train=True)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=pred)
    return loss

# gradient tape
#def grad(model, samples, labels):
#  with tf.GradientTape() as tape:
#    loss, state = loss_fn(model, labels, samples)
#  return tape.gradient(loss, state)

#grad(model, samples, labels)

# output accuracy of untrained model on train data
correct = 0
for i in trange(100):
    if model.forward(tf.expand_dims(samples[1], axis=0), train=False) == labels[i]:
        correct += 1
print('percent corect guesses:', 100 * correct/1000, '%\n')

# train
epochs = 1

for epoch in trange(epochs):
    func_loss = functools.partial(loss_fn, model, samples, labels)
    optim.minimize(func_loss)

# output "accuracy" of trained model on train data
correct = 0
for i in trange(100):
    if model.forward(tf.expand_dims(samples[1], axis=0), train=False) == labels[i]:
        correct += 1
print('percent corect guesses:', 100 * correct/1000, '%\n')


checkpoint_dir = '../dat/checkpoints/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tf.Checkpoint(model=model)
root.save(file_prefix=checkpoint_prefix)
