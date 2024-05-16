# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 21:25:55 2023

@author: timshen2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import KFold
import random
from keras import optimizers
from keras.layers import Conv1D, MaxPool1D, Flatten,Input
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense,Bidirectional,concatenate,GlobalMaxPooling1D,multiply
from keras import optimizers,Model
from keras.layers import Add
from keras.layers import SimpleRNN

# CNN_AA
vocabulary = 100
embedding_dim = 32
word_num = 24
state_dim = 32


# Define input shape for the model
inputs = Input(shape=(word_num,))

# Add an embedding layer with vocabulary size, embedding dimension, and input length
embedding = Embedding(vocabulary, embedding_dim, input_length=word_num)(inputs)

# Add a 1D convolutional layer with 256 filters, kernel size 5, 'same' padding, and ReLU activation
conv1 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(embedding)

# Add a max pooling layer with pool size 5, stride 1, and 'same' padding
pool1 = MaxPool1D(pool_size=5, strides=1, padding="same")(conv1)

# Add another 1D convolutional layer with 256 filters, kernel size 5, 'same' padding, and ReLU activation
conv2 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(pool1)

# Add another max pooling layer with pool size 5, stride 1, and 'same' padding
pool2 = MaxPool1D(pool_size=5, strides=1, padding="same")(conv2)

# Add another 1D convolutional layer with 256 filters, kernel size 5, 'same' padding, and ReLU activation
conv3 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(pool2)

# Add another max pooling layer with pool size 5, stride 1, and 'same' padding
pool3 = MaxPool1D(pool_size=5, strides=1, padding="same")(conv3)

# Add another 1D convolutional layer with 256 filters, kernel size 5, 'same' padding, and ReLU activation
conv4 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(pool3)

# Add another max pooling layer with pool size 5, stride 1, and 'same' padding
pool4 = MaxPool1D(pool_size=5, strides=1, padding="same")(conv4)

# Flatten the output of the last convolutional layer
flatten = Flatten()(pool4)

# Add a dense layer with sigmoid activation to output a binary classification
outputs = Dense(1, activation='sigmoid')(flatten)

# Create the model
model_cnn = Model(inputs=inputs, outputs=outputs)

# Print a summary of the model architecture
model_cnn.summary()




#--------------------------------------------------------------------------------------------

# cnn_lstm_res

input_layer = Input(shape=(24,))
embed = Embedding(vocabulary, embedding_dim, input_length=word_num)(input_layer)

# First convolutional block
cnn1 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(embed)
cnn1 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn1)
cnn1 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn1)
cnn1 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn1)

# Second convolutional block with residual connection from the first block
cnn2 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn1)
cnn2 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn2)
cnn2 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn2)
cnn2 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn2)
cnn2_res = Add()([cnn1, cnn2])

# Third convolutional block with residual connection from the second block
cnn3 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn2_res)
cnn3 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn3)
cnn3 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn3)
cnn3 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn3)
cnn3_res = Add()([cnn2_res, cnn3])

# Fourth convolutional block with residual connection from the third block
cnn4 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn3_res)
cnn4 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn4)
cnn4 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn4)
cnn4 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn4)
cnn4_res = Add()([cnn3_res, cnn4])

cnn = Flatten()(cnn4_res)
cnn = Dense(16, activation='sigmoid')(cnn)

lstm = Bidirectional(LSTM(state_dim, return_sequences=True, dropout=0.2))(embed)
lstm = Bidirectional(LSTM(state_dim, return_sequences=True, dropout=0.2))(lstm)
lstm = Bidirectional(LSTM(state_dim, return_sequences=False, dropout=0.2))(lstm)
lstm = Dense(16, activation='sigmoid')(lstm)

merge = concatenate([cnn, lstm], axis=1)
preds = Dense(1, activation='sigmoid')(merge)

cnn_lstm_res = Model(input_layer, preds)
cnn_lstm_res.summary()


#--------------------------------------------------------------------------------------------

# cnn_lstm_res
# Define the input layer for text data
input_layer_1 = Input(shape=(24,))

# Define the embedding layer for text data
embed_1 = Embedding(vocabulary, embedding_dim, input_length=word_num)(input_layer_1)

# First convolutional block
cnn1 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(embed_1)
cnn1 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn1)
cnn1 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn1)
cnn1 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn1)

# Second convolutional block with residual connection from the first block
cnn2 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn1)
cnn2 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn2)
cnn2 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn2)
cnn2 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn2)
cnn2_res = Add()([cnn1, cnn2])

# Third convolutional block with residual connection from the second block
cnn3 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn2_res)
cnn3 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn3)
cnn3 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn3)
cnn3 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn3)
cnn3_res = Add()([cnn2_res, cnn3])

# Fourth convolutional block with residual connection from the third block
cnn4 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn3_res)
cnn4 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn4)
cnn4 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn4)
cnn4 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn4)
cnn4_res = Add()([cnn3_res, cnn4])

cnn = Flatten()(cnn4_res)
cnn = Dense(16, activation='sigmoid')(cnn)

lstm = Bidirectional(LSTM(state_dim, return_sequences=True, dropout=0.2))(embed_1)
lstm = Bidirectional(LSTM(state_dim, return_sequences=True, dropout=0.2))(lstm)
lstm = Bidirectional(LSTM(state_dim, return_sequences=False, dropout=0.2))(lstm)
lstm = Dense(16, activation='sigmoid')(lstm)

# Define the input layer for gene feature
input_layer_2 = Input(shape=(1,))

# Define the embedding layer for gene feature
embed_2 = Embedding(100, 20, input_length=1)(input_layer_2)

# Flatten the output of the embedding layer for gene feature
flatten = Flatten()(embed_2)

# Concatenate the output of the CNN, LSTM and the flattened gene feature
merge = concatenate([cnn, lstm, flatten], axis=1)

preds = Dense(1, activation='sigmoid')(merge)

cnn_lstm_res_gene = Model([input_layer_1, input_layer_2], preds)
cnn_lstm_res_gene.summary()

#--------------------------------------------------------------------------------------------

# cnn_lstm
input_layer = Input(shape=(24,))
embed = Embedding(vocabulary, embedding_dim, input_length=word_num)(input_layer)

# CNN layers
cnn1 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(embed)
cnn1 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn1)
cnn1 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn1)
cnn1 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn1)

cnn2 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn1)
cnn2 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn2)
cnn2 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn2)
cnn2 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn2)

cnn3 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn2)
cnn3 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn3)
cnn3 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn3)
cnn3 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn3)

cnn4 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn3)
cnn4 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn4)
cnn4 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn4)
cnn4 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn4)

cnn = Flatten()(cnn4)
cnn = Dense(16, activation='sigmoid')(cnn)

# LSTM layers
lstm = Bidirectional(LSTM(state_dim, return_sequences=True, dropout=0.2))(embed)
lstm = Bidirectional(LSTM(state_dim, return_sequences=True, dropout=0.2))(lstm)
lstm = Bidirectional(LSTM(state_dim, return_sequences=False, dropout=0.2))(lstm)
lstm = Dense(16, activation='sigmoid')(lstm)

# Concatenate CNN and LSTM outputs
merge = concatenate([cnn, lstm], axis=1)

# Output layer
preds = Dense(1, activation='sigmoid')(merge)

# Create the model
cnn_lstm = Model(input_layer, preds)
cnn_lstm.summary()

#--------------------------------------------------------------------------------------------

# cnn_lstm
# Define the input layer for text data
input_layer_1 = Input(shape=(24,))

# Define the embedding layer for text data
embed_1 = Embedding(vocabulary, embedding_dim, input_length=word_num)(input_layer_1)

# CNN layers
cnn1 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(embed_1)
cnn1 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn1)
cnn1 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn1)
cnn1 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn1)

cnn2 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn1)
cnn2 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn2)
cnn2 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn2)
cnn2 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn2)

cnn3 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn2)
cnn3 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn3)
cnn3 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn3)
cnn3 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn3)

cnn4 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn3)
cnn4 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn4)
cnn4 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(cnn4)
cnn4 = MaxPool1D(pool_size=5, strides=1, padding="same")(cnn4)

cnn = Flatten()(cnn4)
cnn = Dense(16, activation='sigmoid')(cnn)

# LSTM layers
lstm = Bidirectional(LSTM(state_dim, return_sequences=True, dropout=0.2))(embed_1)
lstm = Bidirectional(LSTM(state_dim, return_sequences=True, dropout=0.2))(lstm)
lstm = Bidirectional(LSTM(state_dim, return_sequences=False, dropout=0.2))(lstm)
lstm = Dense(16, activation='sigmoid')(lstm)

# Define the input layer for gene feature
input_layer_2 = Input(shape=(1,))

# Define the embedding layer for gene feature
embed_2 = Embedding(100, 20, input_length=1)(input_layer_2)

# Flatten the output of the embedding layer for gene feature
flatten = Flatten()(embed_2)

# Concatenate the output of the CNN, LSTM and the flattened gene feature
merge = concatenate([cnn, lstm, flatten], axis=1)

preds = Dense(1, activation='sigmoid')(merge)

cnn_lstm_gene = Model([input_layer_1, input_layer_2], preds)
cnn_lstm_gene.summary()

#--------------------------------------------------------------------------------------------

# cnn_AA_Gene/GeneFam

# Define the input layer for text data
input_layer_1 = Input(shape=(24,))

# Add an embedding layer for text data
embed_1 = Embedding(vocabulary, embedding_dim, input_length=word_num)(input_layer_1)

# Add four 1D convolutional layers with max pooling
cnn = Conv1D(filters=256,kernel_size=5,padding = 'same', activation = 'relu')(embed_1)
cnn = MaxPool1D(pool_size=5,strides=1, padding="same")(cnn)
cnn = Conv1D(filters=256,kernel_size=5,padding = 'same', activation = 'relu')(cnn)
cnn = MaxPool1D(pool_size=5,strides=1, padding="same")(cnn)
cnn = Conv1D(filters=256,kernel_size=5,padding = 'same', activation = 'relu')(cnn)
cnn = MaxPool1D(pool_size=5,strides=1, padding="same")(cnn)
cnn = Conv1D(filters=256,kernel_size=5,padding = 'same', activation = 'relu')(cnn)
cnn = MaxPool1D(pool_size=5,strides=1, padding="same")(cnn)

# Flatten the output of the last convolutional layer
cnn = Flatten()(cnn)

# Add a dense layer with sigmoid activation for binary classification
cnn = Dense(1,activation='sigmoid')(cnn)

# Define the input layer for categorical data
input_layer_2 = Input(shape=(1,))

# Add an embedding layer for categorical data
embed_2 = Embedding(100, 20, input_length=1)(input_layer_2)

# Flatten the output of the embedding layer
flatten = Flatten()(embed_2)

# Concatenate the output of the last convolutional layer and the flattened categorical data
merge = concatenate([cnn,flatten],axis = 1)

# Add a dense layer with sigmoid activation for binary classification
preds = Dense(1,activation='sigmoid')(merge)

# Define the model with two inputs and one output
cnn_gene = Model([input_layer_1,input_layer_2],preds)

# Print the model summary
cnn_gene.summary()



#--------------------------------------------------------------------------------------------

# logist

# Create a sequential model
model_logist = Sequential() 

# Add an embedding layer with specified vocabulary, embedding dimension, and input length
model_logist.add(Embedding(vocabulary, embedding_dim, input_length=word_num,)) 

# Flatten the output of the embedding layer
model_logist.add(Flatten())

# Add a dense layer with 1 neuron and relu activation function
model_logist.add(Dense(1, activation='relu'))

# Print a summary of the model's architecture
model_logist.summary()

#--------------------------------------------------------------------------------------------

# logist
# Logistic Regression

# Define the input layer for text data
input_layer_1 = Input(shape=(word_num,))

# Define the embedding layer for text data
embed_1 = Embedding(vocabulary, embedding_dim, input_length=word_num)(input_layer_1)

# Flatten the output of the embedding layer for text data
flatten_1 = Flatten()(embed_1)

# Define the input layer for gene feature
input_layer_2 = Input(shape=(1,))

# Define the embedding layer for gene feature
embed_2 = Embedding(100, 20, input_length=1)(input_layer_2)

# Flatten the output of the embedding layer for gene feature
flatten_2 = Flatten()(embed_2)

# Concatenate the flattened text data and the flattened gene feature
merge = concatenate([flatten_1,flatten_2],axis = 1)

# Add a dense output layer with relu activation
output_layer = Dense(1, activation='relu')(merge)

# Define the model with two inputs and one output
logist_gene = Model(inputs=[input_layer_1,input_layer_2], outputs=output_layer)

# Print the model summary
logist_gene.summary()

#--------------------------------------------------------------------------------------------


# SimpleRNN

# Define the input layer
input_layer = Input(shape=(word_num,))

# Add the embedding layer
embed = Embedding(vocabulary, embedding_dim, input_length=word_num)(input_layer)

# Add a SimpleRNN layer, with dropout and return sequences set to False
rnn = SimpleRNN(state_dim, return_sequences=False, dropout=0.2)(embed)

# Add a dense output layer with relu activation
output_layer = Dense(1, activation='relu')(rnn)

# Create the model with input and output layers
model_SimpleRNN = Model(inputs=input_layer, outputs=output_layer)

# Print the summary of the model
model_SimpleRNN.summary()


#--------------------------------------------------------------------------------------------

# SimpleRNN

# Define the input layer for text data
input_layer_1 = Input(shape=(word_num,))

# Add the embedding layer for text data
embed_1 = Embedding(vocabulary, embedding_dim, input_length=word_num)(input_layer_1)

# Add a SimpleRNN layer, with dropout and return sequences set to False
rnn = SimpleRNN(state_dim, return_sequences=False, dropout=0.2)(embed_1)

# Define the input layer for gene feature
input_layer_2 = Input(shape=(1,))

# Add an embedding layer for gene feature
embed_2 = Embedding(100, 20, input_length=1)(input_layer_2)

# Flatten the output of the embedding layer
flatten = Flatten()(embed_2)

# Concatenate the output of the last SimpleRNN layer and the flattened gene feature
merge = concatenate([rnn,flatten],axis = 1)

# Add a dense output layer with relu activation
output_layer = Dense(1, activation='relu')(merge)

# Create the model with two inputs and one output
SimpleRNN_gene = Model(inputs=[input_layer_1,input_layer_2], outputs=output_layer)

# Print the summary of the model
SimpleRNN_gene.summary()



#--------------------------------------------------------------------------------------------
# Define input layer
input_layer = Input(shape=(word_num,))

# Define embedding layer
embed = Embedding(vocabulary, embedding_dim, input_length=word_num)(input_layer)

# Define LSTM layer
lstm = LSTM(state_dim, return_sequences=False, dropout=0.2)(embed)

# Define output layer
output_layer = Dense(1, activation='relu')(lstm)

# Define the model
model_lstm = Model(inputs=input_layer, outputs=output_layer)

# Print model summary
model_lstm.summary()



#--------------------------------------------------------------------------------------------

# LSTM_gene

# Define the input layer for text data
input_layer_1 = Input(shape=(word_num,))

# Define the embedding layer for text data
embed_1 = Embedding(vocabulary, embedding_dim, input_length=word_num)(input_layer_1)

# Define LSTM layer for text data
lstm = LSTM(state_dim, return_sequences=False, dropout=0.2)(embed_1)

# Define the input layer for gene feature
input_layer_2 = Input(shape=(1,))

# Define the embedding layer for gene feature
embed_2 = Embedding(100, 20, input_length=1)(input_layer_2)

# Flatten the output of the embedding layer
flatten = Flatten()(embed_2)

# Concatenate the output of the LSTM layer and the flattened gene feature
merge = concatenate([lstm, flatten], axis=1)

# Define the output layer with relu activation
output_layer = Dense(1, activation='relu')(merge)

# Define the model with two inputs and one output
lstm_gene = Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer)

# Print the model summary
lstm_gene.summary()




