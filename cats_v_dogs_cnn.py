
# coding: utf-8

# # Cats vs Dogs

# For this workshop you will be building a Convolutional neural network to classify cats vs dogs. You will need to be familiar with the theory of CNNs. Visit our lesson [here](http://caisplusplus.usc.edu/blog/curriculum/lesson7) for more info. Only fill in the TODO sections. 

# In[12]:


# Imports, make sure you have cv2 installed!
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sklearn.utils

# DO NOT CHANGE ANY OF THIS

DATA_PATH = './data/'
TEST_PERCENT = 0.2
# This is just for sake of time. In real situations of course you would use the whole dataset.
SELECT_SUBSET_PERCENT = 0.15

# The cat and dog images are of variable size we have to resize them to all the same size.
# DO NOT CHANGE
RESIZE_WIDTH=64
RESIZE_HEIGHT=64
# We are setting this to be 5 epochs for fast training times. In practice we would have many more epochs. 
EPOCHS = 5


# ## Load the Data
# Load the train and test data sets. Do not modify this code at all. Make sure that your data for cats and dogs images is in ``./data``. You can find that data at https://www.kaggle.com/c/dogs-vs-cats/data

# In[6]:


# Lets get started by loading the data.
# Make sure you have the data downloaded to ./data
# To download the data go to https://www.kaggle.com/c/dogs-vs-cats/data and download train.zip

X = []
Y = []

files = os.listdir(DATA_PATH)
# Shuffle so we are selecting about an equal number of dog and cat images.
shuffled_files = sklearn.utils.shuffle(files)
select_count = int(len(shuffled_files) * SELECT_SUBSET_PERCENT)

print('Going to load %i files' % select_count)

subset_files_select = shuffled_files[:select_count]

DISPLAY_COUNT = 1000

for i, input_file in enumerate(subset_files_select):
    if i % DISPLAY_COUNT == 0 and i != 0:
        print('Have loaded %i samples' % i)
        
    img = plt.imread(DATA_PATH + input_file)
    # Resize the images to be the same size.
    img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    X.append(img)
    if 'cat' == input_file.split('.')[0]:
        Y.append(0.0)
    else:
        Y.append(1.0)
        
X = np.array(X)
Y = np.array(Y)

test_size = int(len(X) * TEST_PERCENT)

test_X = X[:test_size]
test_Y = Y[:test_size]
train_X = X[test_size:]
train_Y = Y[test_size:]

print('Train set has dimensionality %s' % str(train_X.shape))
print('Test set has dimensionality %s' % str(test_X.shape))

# Apply some normalization here.
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X /= 255
test_X /= 255


# ## Preprocessing
# While not necessary for this problem you can go ahead and try some preprocessing steps to try to get higher accuracies.

# In[ ]:


######################################
#TODO: (Optional)
# Perform any data preprocessing steps



######################################


# ### Defining the network
# Here are some useful resources to help with defining a powerful network.
# - Convolution layers (use the 2D convolution) https://keras.io/layers/convolutional/
# - Batch norm layer https://keras.io/layers/normalization/
# - Layer initializers https://keras.io/initializers/
# - Dense layer https://keras.io/layers/core/#dense
# - Activation functions https://keras.io/layers/core/#activation
# - Regulizers: 
#     - https://keras.io/layers/core/#dropout
#     - https://keras.io/regularizers/
#     - https://keras.io/callbacks/#earlystopping
#     - https://keras.io/constraints/

# In[10]:


######################################
#TODO:
# Import necessary layers.
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras import optimizers
from keras import losses

######################################

model = Sequential()

######################################
#TODO:
# Define the network

#Layer 1
model.add(Conv2D(filters=10, kernel_size=(3,3), input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Layer 2
model.add(BatchNormalization())
model.add(Conv2D(filters=10, kernel_size=(3,3), input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Layer 3
model.add(BatchNormalization())
model.add(Conv2D(filters=10, kernel_size=(3,3), input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))


######################################


######################################
#TODO:
# Define your loss and your objective
optimizer = optimizers.RMSprop()
loss = 'binary_crossentropy'
######################################


model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])


# ## Train Time
# Train the network. Be on the lookout for the validation loss and accuracy. Don't change any of the parameters here except for the batch size.

# In[11]:


######################################
#TODO:
# Define the batch size
batch_size = 1000
######################################


model.fit(train_X, train_Y, batch_size=batch_size, epochs=EPOCHS, validation_split=0.2, verbose=1, shuffle=True)


# ## Test Time
# Now it's time to actually test the network. 
# 
# Get above **65%**!

# In[ ]:


loss, acc = model.evaluate(test_X, test_Y, batch_size=batch_size, verbose=1)

print('')
print('Got %.2f%% accuracy' % (acc * 100.))

