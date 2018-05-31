import time
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

# output accuracy of untrained model on train data
correct = 0
for i in trange(100):
    temp_pred = np.argmax(model.forward(tf.expand_dims(samples[i], axis=0), train=False)[0], axis=1)[0]
    temp_true = labels[i].numpy()[0]
    tqdm.write('{},{}'.format(temp_pred, temp_true))
    if temp_pred == temp_true:
        correct += 1
print('percent corect guesses:', 100 * correct/100, '%\n')

# train ############################################################################
epochs = 100

for epoch in trange(epochs):
    func_loss = functools.partial(loss_fn, model, samples, labels)
    optim.minimize(func_loss)
    #tqdm.write('{}'.format(loss_fn(model, labels, samples)))
#######################################################################################

# output "accuracy" of trained model on train data
correct = 0
for i in trange(100):
    temp_pred = np.argmax(model.forward(tf.expand_dims(samples[i], axis=0), train=False)[0], axis=1)[0]
    temp_true = labels[i].numpy()[0]
    tqdm.write(temp_pred, temp_true)
    if model.forward(tf.expand_dims(samples[1], axis=0), train=False) == labels[i]:
        correct += 1
print('percent corect guesses:', 100 * correct/100, '%\n')

# plot
raw = pd.read_csv('../dat/clean_train.csv')
dat = raw[raw['id'] == 92]

fig,ax = plt.subplots(1, figsize=(15,10))

for i,sensor in enumerate(list(raw.columns)[5:-4]):
    plt.plot(dat[sensor], label=sensor, linestyle='-')

rect1 = patches.Rectangle((18824,0), width=25, height=1,
                          alpha=0.4, edgecolor='yellow',facecolor='yellow')
ax.add_patch(rect1)
rect2 = patches.Rectangle((18849,0), width=6, height=1,
                          alpha=0.4, edgecolor='red',facecolor='red')
ax.add_patch(rect2)

plt.tight_layout()
plt.legend()
plt.show()

# plot
dat = raw[raw['id'] == 69]

fig,ax = plt.subplots(1, figsize=(15,10))

for i,sensor in enumerate(list(raw.columns)[5:-4]):
    plt.plot(dat[sensor], label=sensor, linestyle='-')

rect1 = patches.Rectangle((13950,0), width=15, height=1,
                          alpha=0.4, edgecolor='yellow',facecolor='yellow')
ax.add_patch(rect1)
rect2 = patches.Rectangle((13966,0), width=25, height=1,
                          alpha=0.4, edgecolor='red',facecolor='red')
ax.add_patch(rect2)

plt.tight_layout()
plt.legend()
plt.show()
