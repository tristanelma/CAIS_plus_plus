
# coding: utf-8

# # CAIS++ Logistic Regression Workshop!
# 
# Before you go through this code, make sure you read [lesson 3]('http://caisplusplus.usc.edu/blog/curriculum/lesson3') from our curriculum

# ## Part 1: Import Dataset & Preprocessing 

# In[1]:


# Import Statemements

import random
import pickle # used to save and restore python objects
import gzip
import numpy as np
import tensorflow as tf


# In[2]:


# Load the Dataset

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding = 'latin1')
f.close()

# What does our dataset look like?
print(type(train_set))
print(type(test_set))


# In[3]:


# A Closer Look at Our Dataset

print("Inputs:")
print(train_set[0])

# 50,000 MNIST images: each represented as a vector of length 784 (28 x 28).
# 0 corresponds to a dark pixel, 1 corresponds to a light pixel
print("Inputs shape is " + str(train_set[0].shape)) 
print("Input type is " + str(type(train_set[0])))
print("Labels:")

# There's 50,000 labels (one for each example)
print(train_set[1])
print("Labels shape is" + str(train_set[1].shape))
print("Labels type is " + str(type(train_set[1])))


# In[4]:


# Function: to_categorical -- One-Hot Vector Encoding 

# Converts class labels (integers from 0 to nb_classes) to one-hot vector
# Example: 5 => [0 0 0 0 0 1 0 0 0 0]

# Arguments:
    # y: array, class labels to convert
    # nb_classes: integer, total number of classes 

def to_categorical(y, nb_classes):
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    answer = np.zeros((len(y), nb_classes))
    answer[np.arange(len(y)),y] = 1.
    return answer


# In[5]:


# Train-Test Split, One-Hot Encoding for our data

train_x = train_set[0]
train_y = to_categorical(train_set[1], 10)
test_x = test_set[0]
test_y = to_categorical(test_set[1],10)

# Result of One-Hot Encoding Class Labels 
print(test_y[:5])


# ## Part 2: Create Logistic Regression Model using TensorFlow

# <img src="https://www.tensorflow.org/versions/r1.1/images/softmax-regression-scalargraph.png" style="width: 600px;"/>
# <br />
# <br />
# <img src="https://www.tensorflow.org/versions/r1.1/images/softmax-regression-vectorequation.png" style="width: 600px;"/>
# <br />
# <br />
# 
# 
# $$y = \text{softmax}(Wx + b)$$

# In[6]:


# Set up Model Variables in TensorFlow

# This just helps with using tensorflow inside jupyter (reset/clear all tf variables)
tf.reset_default_graph()

# Weights Variable (xavier initializer -- random values centered around zero)
W = tf.get_variable("W", shape=[784, 10], initializer = tf.contrib.layers.xavier_initializer())

# Biases Variable (initialized to zero)
b = tf.Variable(tf.zeros([10]))

### TODO: Create Model Placeholders for x and y_

# x = Input Parameter
# y_ = Actual Labels (the one-hot vector with digit labels)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


# In[7]:


### TODO: Define hypothesis fxn (y)

# Hint: Use tf.nn.softmax() and tf.matmul() functions to matrix multiply x and W, then add b 
y = tf.nn.softmax(tf.matmul(x, W)+b)


# ## Part 3: Training the Model
# 
# Using the given training examples, our goal is to find the best possible hypothesis function. In other words, we need to find the weights that minimize the cost function

# In[8]:


# Logistic Regression Cost Function: Cross-Entropy Loss

# cost increases as predicted probability diverges from actual label 
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# In[9]:


# Optimization -- Batch Gradient Descent

# At each step, we'll use a random subset from the training set
# Instead of traditional gradient descent, which looks at every example at every step 

# Function: generate_batch: takes in desired batch size and returns a "batch" of data (MNIST images)
def generate_batch(batch_size):
    ### TODO: complete the "indexes = " line, where we want to randomly choose a set of images from the training set
    # The size of this set should be equal to the batch_size
    # Hint: use the random.sample() function to intialize the "indexes" variable.

    indexes = random.sample(range(50000), batch_size)
    
    # We'll return the images from train_x that correspond to the indexes in this "batch"
    return train_x[indexes], train_y[indexes]


# In[10]:


# Set up Training in TensorFlow

### TODO: Decide model hyperparameters 
learning_rate = .4
iterations = 1000
batch_size = 1000

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
sess = tf.InteractiveSession() # create the sesion object
tf.global_variables_initializer().run() # initialize variables


# In[11]:


# Training Algorithm: TensorFlow Automatically Takes Care of Gradient Calculations :) 

### TODO: Create a loop to run the training algorithm
# Inside the loop:
# 1) create variables batch_xs, and batch_ys, which will be our x training batch and y training batch. Use generate_batch()
# 2) call sess.run() (info here: https://www.tensorflow.org/api_docs/python/tf/InteractiveSession#run)
    # i) in sess.run(), the fetches argument will be the train_step defined above
    # ii) The feed_dict will pass batch_xs for the x placeholder, and batch_ys for the y_ placeholder

for i in range(iterations):
    batch_xs, batch_ys = generate_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
    # write code here! 


# ## Part 4: Evaluate the Model
# Now, let's see how accurrate our hypothesis function is on our test dataset

# In[12]:


# Model Evaluation

# store correct_predictions list and calculate accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

# look at predictions on our test dataset
print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

