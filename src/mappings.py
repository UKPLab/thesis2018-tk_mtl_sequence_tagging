"""Mappings between constants and tensorflow object"""

import tensorflow as tf

from constants import ACTIVATION_LINEAR, ACTIVATION_RELU, ACTIVATION_SIGMOID, ACTIVATION_TANH, \
    OPTIMIZER_SGD, OPTIMIZER_ADAM, OPTIMIZER_ADAGRAD, OPTIMIZER_ADADELTA, RNN_UNIT_TYPE_LSTM, RNN_UNIT_TYPE_GRU, \
    RNN_UNIT_TYPE_SIMPLE

ACTIVATION_MAPPING = {
    ACTIVATION_RELU: tf.nn.relu,
    ACTIVATION_LINEAR: None,
    ACTIVATION_SIGMOID: tf.nn.sigmoid,
    ACTIVATION_TANH: tf.nn.tanh,
}

OPTIMIZER_MAPPING = {
    OPTIMIZER_SGD: tf.train.GradientDescentOptimizer,
    OPTIMIZER_ADAM: tf.train.AdamOptimizer,
    OPTIMIZER_ADAGRAD: tf.train.AdagradOptimizer,
    OPTIMIZER_ADADELTA: tf.train.AdadeltaOptimizer,
}

RNN_CELL_MAPPING = {
    RNN_UNIT_TYPE_SIMPLE: tf.contrib.rnn.BasicRNNCell,
    RNN_UNIT_TYPE_GRU: tf.contrib.rnn.GRUCell,
    RNN_UNIT_TYPE_LSTM: tf.contrib.rnn.LSTMCell,
}
