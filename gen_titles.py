import tensorflow as tf

import numpy as np
import os
import time

import telebot
from telebot import types
import random
import datetime
import sqlite3
# import all the modules




text = ''
with open('tutby_titles.txt', 'r') as infile:
    for line in infile:
        text += line
# writing all the titles from file to variable 'text'
# doing it line by line cause my server wasn't powerful enough to do it in one time

vocab = sorted(set(text))
# creating vocabulary
print(f'unique characters {len(vocab)}')

char2idx = {u:i for i, u in enumerate(vocab)}

idx2char = np.array(vocab)
# these two are converting characters to numbers and vice versa
# cause neural networks work only with numbers


text_as_int = np.array([])
for i in range(150):
    text_as_int = np.append(text_as_int, np.array([char2idx[c] for c in text[i * 240231]]))
# making np.array out of all the text as integers
# doing it in 150 times cause, again, my server wasn't powerful enough


# The maximum length sentence you want for a single input in characters
seq_length = 265



# creating dataset from array with all letters as integers
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
# It was done for neural network to understand it

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text





# Batch size
BATCH_SIZE = 72

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000




# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

# function for builing model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
    batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
    return_sequences=True,
    stateful=True,
    recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
    ])
    return model



# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'



#building model
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)


#loading weights of latest checkpoint
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

#building model
model.build(tf.TensorShape([1, None]))

print(model.summary())


# main func to generate text
def generate_text(model, start_string, temper = 0.45):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = temper

    # Here batch size == 1
    model.reset_states()
    predictions = model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    while idx2char[predicted_id] != '₪':
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


global new_title
new_title = ''

global last_temp
last_temp = 0.53

try:
    while True:
        tempp = input()
        if not tempp:
            if last_temp:
                tempp = last_temp
            else:
                tempp = tempp
        else:
            tempp = float(tempp)
        new_title = generate_text(model, start_string=u"♣", temper = tempp)
        print(new_title.replace('♣', '').replace('₪', ''))
        print()
        with open('phrases_logs.txt', 'a') as log_file:
            log_file.write(f'{new_title} [{tempp}]\n\n')
        last_temp = tempp
except KeyboardInterrupt:
    pass
