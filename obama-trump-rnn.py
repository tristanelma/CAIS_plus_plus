
# coding: utf-8

# # Recurrent Neural Networks: Obama/Trump Tweet Classifier
# 
# For this workshop, you will be training a Recurrent Neural Network to classify between Obama's tweets and Trump's tweets. You will need to be familiar with the theory behind RNNs and/or LSTMs before starting. Visit our lesson [here](http://caisplusplus.usc.edu/blog/curriculum/lesson8) for more info. Only fill in the TODO sections!
# 
# ---

# ## Step 1: Load the Data
# 1. Load in Tweets and corresponding labels (Obama = 0, Trump = 1)
# 2. Display some sample tweets

# In[1]:


TWEETS_DIR = './obama-trump-data/tweets'
LABELS_DIR = './obama-trump-data/labels_np'

######################################
# TODO: specify your own word embeddings file directory here
EMBEDDINGS_DIR = 'glove.6B.50d.txt'
######################################


# In[2]:


import pickle

# Load tweets, labels
tweets = pickle.load(open(TWEETS_DIR,'rb'))
labels = pickle.load(open(LABELS_DIR,'rb'))

# Sample some tweets to display
for i in range(0,100,10):
    print("Tweet: ", tweets[i], ". Label: ", labels[i])


# ## Step 2: Data Preprocessing
# 
# Relevant Keras documentation [here](https://keras.io/preprocessing/text/).
# 
# 1. **Tokenize** the tweets: convert each tweet into a sequence of word indices (Documentation .)
# 2. **Pad** the input sequences with blank spots at the end to make sure they're all the same length
# 3. Load in pre-trained word embeddings so that we can convert each word index into a unique **word embedding vector**
# 
# **Word embeddings** are a way of encoding words into n-dimensional vectors with continuous values so that the vector contains some information about the words' meanings. **Embedded word vectors** are often more useful than **one-hot vectors** in natural language processing applications, because the vectors themselves contain some embedded information about the word's meaning, instead of just identifying which word is there. 
# 
# For example, similar words (e.g. "him" and "her") tend to have similar embedded vectors, whereas if we were just using one-hot vectors (i.e. only one "1" for "him" and one "1" for "her"), no notion of the words' actual meanings would be conveyed. Although this may sound crazy at first, word embeddings can even convey relationships between analogous words: for example: `king-man+woman≈queen`.
# 
# For more info on word embeddings, check out this introductory [blog post](https://www.springboard.com/blog/introduction-word-embeddings/). Two of the most common word embeddings methods are [word2vec](https://www.tensorflow.org/tutorials/representation/word2vec) and [GloVe](https://nlp.stanford.edu/projects/glove/).

# In[3]:


import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Tokenize the tweets (convert sentence to sequence of words)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)

sequences = tokenizer.texts_to_sequences(tweets)
word_index = tokenizer.word_index

print('Found %s unique tokens (words).' % len(word_index))

# Pad sequences to ensure samples are the same size
training_data = pad_sequences(sequences)

print("Training data size is (%d,%d)"  % training_data.shape) # shape = (num. tweets, max. length of tweet)
print("Labels are size %d"  % labels.shape)


# In[4]:


# Show effect of tokenization, padding
print("Original tweet: ", tweets[0])
print("Tweet after tokenization and padding: ", training_data[0])


# In[5]:


# Convert words to word embedding vectors

EMBEDDING_DIM = 50
print('Indexing word vectors.')

import os
embeddings_index = {}
f = open(EMBEDDINGS_DIR) # NOTE: if using Windows and getting errors: change this to open(EMBEDDINGS_DIR, 'rb')
for line in f:
    values = line.split()
    word = values[0] # NOTE: if you made that 'rb' change: add to this line, values[0].decode('UTF-8')
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# prepare word embedding matrix
num_words = len(word_index)+1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[6]:


# Sample word embedding vector:
print(word_index["computer"]) # retrieve word index
print(embedding_matrix[2645]) # use word index to retrieve word embedding vector


# ## Step 3: Creating the Classifier Model
# Here are some useful resources to help you guys out here:
# 
# #### Keras Documentation
# * [Embedding layer](https://keras.io/layers/embeddings/) (our word embeddings are pre-trained, so you won't have to worry too much about this)
# * [**Recurrent layers (e.g. RNN, LSTM)**](https://keras.io/layers/recurrent/) (Feel free to play around with different layers.)
# * [Dense layer](https://keras.io/layers/core/#dense) (Use this for your final classification "vote")
# * [Activation functions](https://keras.io/activations/)
# * [Dropout](https://keras.io/layers/core/#dropout), [Batch Normalization](https://keras.io/layers/normalization/)
# * [Optimizers](https://keras.io/optimizers/)
# 
# #### Examples
# * [Keras sequential model guide](https://keras.io/getting-started/sequential-model-guide/) (Scroll down to "Sequence classification with LSTM")
# * [Keras LSTM for IMDB review sentiment analysis](https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py)
# 
# **Hint**: if you have multiple recurrent layers, remember to use `return_sequences=True` if and only if you're adding another recurrent layer after the current one. This makes it so that the recurrent layer spits out an output after each timestep (or element in the sequence), instead of just at the very end of the sequence.

# In[7]:


from keras.models import Sequential
from keras.layers import Embedding, Input
from keras.layers.merge import Concatenate
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Dropout, concatenate
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import metrics
from keras.models import Model
import pickle


# In[19]:


model = Sequential()

# Add pre-trained embedding layer 
    # converts word indices to GloVe word embedding vectors as they're fed in
model.add(Embedding(len(word_index) + 1,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=training_data.shape[1],
                    trainable=False))

# At this point, each individual training sample is now a sequence of word embedding vectors

######################################
# TODO: define the rest of the network!
model.add(LSTM(101, dropout=0.01, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

######################################

print(model.summary())


######################################
# TODO: pick out a loss function and corresponding optimizer.
    # Hint: there's one "correct" loss function we have in mind,
        # since we're just classifying between 0 and 1.
    # You can pick your own optimizer, but Keras's documentation recommends one for RNNs 

LOSS = 'binary_crossentropy'
OPTIMIZER = 'rmsprop'
######################################


model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics = [metrics.binary_accuracy])


# ## 3. Train/Evaluate the Model
# Train the network. Look to make sure that the loss is decreasing and the accuracy is decreasing, but be on the lookout for overfitting. 
# 
# Aim for **90%** final validation accuracy! (Pretty much the same as using a separate test set)

# In[20]:


#####################################
# TODO: pick number of epochs and batch size

EPOCHS = 6
BATCH_SIZE = 100
#####################################

model.fit(training_data, labels, 
          epochs = EPOCHS, 
          batch_size = BATCH_SIZE, 
          validation_split =.2)

