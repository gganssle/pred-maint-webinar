import numpy as np
import pandas as pd

class sequencer():
    def __init__(self):
        super(sequencer, self).__init__()

    def gen_sequence(self, id_df, seq_length, seq_cols):
        data_array = id_df[seq_cols].values
        num_elements = data_array.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data_array[start:stop, :]

    def gen_labels(self, id_df, seq_length, label):
        data_array = id_df[label].values
        num_elements = data_array.shape[0]
        return data_array[seq_length:num_elements, :]
