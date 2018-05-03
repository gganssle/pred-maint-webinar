import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

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
label_array = np.concatenate(label_gen).astype(np.float32)



################################################################################

ftrs = seq_array.shape[2]
batch_size = 20

class simplelstm():
    def __init__(self, batch_size=batch_size, lstm_size=ftrs):
        super(simplelstm, self).__init__()
        self.lstm_size = lstm_size
        self.hidden_state1 = tf.zeros([sequence_length, lstm_size], dtype='float32')
        self.current_state1 = tf.zeros([sequence_length, lstm_size], dtype='float32')
        self.state1 = self.hidden_state1, self.current_state1
        self.hidden_state2 = tf.zeros([sequence_length, lstm_size], dtype='float32')
        self.current_state2 = tf.zeros([sequence_length, lstm_size], dtype='float32')
        self.state2 = self.hidden_state2, self.current_state2

    def forward(self, inpt):
        network = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
        network = tf.contrib.rnn.DropoutWrapper(network, output_keep_prob=0.2)
        network = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
        network = tf.contrib.rnn.DropoutWrapper(network, output_keep_prob=0.2)
        output = tf.nn.dynamic_rnn(network, inpt, dtype=tf.float32)

        #output = tf.layers.dense(inputs=output, units=2, activation=None)
        output = tf.contrib.layers.fully_connected(output, num_outputs=2, activation_fn=None)
        output = tf.contrib.layers.softmax(output)

        return output

model = simplelstm()

seq_array.shape
samples = tf.constant(seq_array[0:batch_size], dtype='float32') # this is the batching operation
samples.shape
output = model.forward(samples)
output




################################################################################


class simplelstm():
    def __init__(self, batch_size=2, lstm_size=200):
        super(simplelstm, self).__init__()
        self.hidden_state1 = tf.zeros([batch_size, lstm_size], dtype='float32')
        self.current_state1 = tf.zeros([batch_size, lstm_size], dtype='float32')
        self.state1 = self.hidden_state1, self.current_state1
        self.hidden_state2 = tf.zeros([batch_size, lstm_size], dtype='float32')
        self.current_state2 = tf.zeros([batch_size, lstm_size], dtype='float32')
        self.state2 = self.hidden_state2, self.current_state2

        self.lstm1 = tf.nn.rnn_cell.LSTMCell(lstm_size)
        self.lstm2 = tf.nn.rnn_cell.LSTMCell(lstm_size)

    def forward(self, inpt):
        output, _ = self.lstm1(inpt, self.state1)
        output = tf.nn.dropout(output, keep_prob=0.2)
        output, _ = self.lstm2(output, self.state2)
        output = tf.nn.dropout(output, keep_prob=0.2)
        output = tf.contrib.layers.fully_connected(output, num_outputs=2, activation_fn=None)
        output = tf.contrib.layers.softmax(output)

        return output

testin = tf.constant(-1, shape=(50,25), dtype='float32')
samples = testin[0:batch_size] # this is the batching operation
samples.shape
model = simplelstm()
output = model.forward(samples)
output


################################################################################



class simplelstm():
    def __init__(self, batch_size=2, lstm_size=2):
        super(simplelstm, self).__init__()
        self.hidden_state = tf.zeros([1, lstm_size], dtype='float32')
        self.current_state = tf.zeros([1, lstm_size], dtype='float32')
        self.state1 = self.hidden_state, self.current_state
        self.state2 = self.hidden_state, self.current_state

        self.lstm1 = tf.nn.rnn_cell.LSTMCell(lstm_size)
        self.lstm2 = tf.nn.rnn_cell.LSTMCell(lstm_size)

    def forward(self, inpt):
        output, _ = self.lstm1(inpt, self.state1)
        output = tf.nn.dropout(output, keep_prob=0.2)
        output, _ = self.lstm2(output, self.state2)
        output = tf.nn.dropout(output, keep_prob=0.2)
        output = tf.contrib.layers.fully_connected(output, num_outputs=2, activation_fn=None)
        output = tf.contrib.layers.softmax(output)

        return output

testin = tf.constant(-1, shape=(4,2), dtype='float32')
onesample = testin[1]
model = simplelstm()
output = model.forward(tf.reshape(onesample, shape=[1, onesample.shape[0]]))
output


################################################################################
class simplelstm():
    def __init__(self, batch_size=2, lstm_size=2):
        super(simplelstm, self).__init__()
        hidden_state = tf.zeros([1, lstm_size], dtype='float32')
        current_state = tf.zeros([1, lstm_size], dtype='float32')
        state = hidden_state, current_state
        lstm1 = tf.nn.rnn_cell.LSTMCell(lstm_size)

    def forward(self, inpt):
        output, ostate = lstm(inpt, state)
        return output, ostate

testin = tf.constant(-1, shape=(4,2), dtype='float32')
onesample = testin[1]
model = simplelstm()
output, ostate = model.forward(tf.reshape(onesample, shape=[1, onesample.shape[0]]))
output
ostate


################################################################################


batch_size = 2
lstm = tf.nn.rnn_cell.LSTMCell(2)
lstm.state_size
hidden_state = tf.zeros([1,2], dtype='float32')
current_state = tf.zeros([1,2], dtype='float32')
state = hidden_state, current_state
inpt = tf.constant(-1, shape=(1,2), dtype='float32')
output, ostate = lstm(inpt, state)

output

ostate
