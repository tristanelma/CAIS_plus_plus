
# coding: utf-8

# # CAIS++ Linear Regression Workshop
# Before you go through this code, make sure you read [Lesson 2](http://caisplusplus.usc.edu/blog/curriculum/lesson2) from our curriculum!
# 
# ---

# ## Part 1: Importing the Data

# In[1]:


##importing numpy and the boston data set:

import numpy as np
from sklearn.datasets import load_boston


# In[2]:


boston = load_boston()
print(boston.keys())


# In[31]:


print(boston.DESCR)


# In[4]:


# Investigate shape of the input data array
data = boston.data
target = boston.target ## according to the description above, the target is the median price of the houses

print(data.shape)
print(target.shape)
print(boston.feature_names)

num_features = len(boston.feature_names) #13 features
num_samples = data.shape[0] # 506 training examples


# In[5]:


# Use Pandas to get an overview of the training data

import pandas as pd
bos_dataframe = pd.DataFrame(boston.data)
bos_dataframe.columns = boston.feature_names
bos_dataframe.head()


# In[6]:


# Add in the target variable: price

bos_dataframe['PRICE'] = target
bos_dataframe.head()


# ## Part 2: Setting up the Machine Learning Objective

# In[7]:


# 1. Randomly initalize a weights vector between (-1,  1). Keep in mind: what should the size of this vector be?
# 2. Call it weights_init. 
# 3. Print weights_init
#############################################

weights_init = np.random.uniform(-1, 1, num_features)
print(weights_init)


# In[8]:


# Create a variable for the bias, called bias_init.Initalize the bias to 0
#############################################

bias_init = 0


# ### 2.1: Normalize the input data. We do this because so that we can get all of our data in the same scale.
# More information can be found [here](https://stats.stackexchange.com/questions/41704/how-and-why-do-normalization-and-feature-scaling-work)

# In[9]:


# 1. For each feature (coloumn in the data set), calculate the mean and the max. Use the amax function to calculate the max.
# 2. For each data point in that feature, subtract the mean and then divide by the max to normalize.

for i in range(num_features):
    # complete this for loop
    feature_avg = np.mean(data[:, i])
    feature_max = np.amax(data[:, i])
    data[:, i] = (data[:, i]-feature_avg)/feature_max

# now the values should be normalized:
bos_dataframe.head()


# ### 2.2 Defining the hypothesis and the cost function:
# The Hypothesis function returns a vector of predicted prices.
# 1. Since we are working with multiple features, we need to dot product the input data with the weights vector. Use the numpy  dot() function!
# $$h_{w}(x) = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$$
# 
# 2. Now we need to add our bias to each input value. use numpy's repeat function to create a vector of length 'num_samples' of the bias_init.
# 3. Return the dot product of the input data and weights summed with the bias vector.
# 
# The function header has been defined for you, but you need to complete it.

# In[10]:


def hypothesis(weights, bias):
    return data.dot(weights) + np.repeat(bias, num_samples)


# In[11]:


# Run this cell to see the shape of the return value of the hypothesis function:
# (BONUS: try to think of what the shape would be before printing it out)
hypothesis(weights_init, bias_init).shape


# 1. Define the cost function, which is just subtracting the actual target from our hypothesis, and squaring (use np.square()) that error.
# 2. We then take the mean (use np.mean()) of all these squared errors. Remember that we dvide by 2 to make the math easier later on:
# 
# $$MSE \;Cost = J(w_0, w_1) = {\frac1{2m}}\sum_{i=0}^m(h_w(x^{(i)})-y^{(i)})^2$$
# 3. The function header has been defined for you again, but you need to complete it:

# In[12]:


def cost(weights, bias):
    return np.mean(np.square(hypothesis(weights, bias) - target))/2


# In[13]:


# Run this cell to print out the inital cost. It's really large right now!
cost(weights_init, bias_init)


# The gradient function has been defined for you.
# It calculates the partial derivative for the weights and bias (look at the red and blue rectangles:
# <img src = "image_9.png"/>

# In[14]:


# Gradient: return weight gradient vector, bias gradient at current step

def gradient(weights, bias):
    weight_gradients = []
    
    for (weight_num, weight) in enumerate(weights):
        grad = np.mean((hypothesis(weights, bias)-target) * data[:, weight_num])
        weight_gradients.append(grad)
        
    weight_gradients = np.array(weight_gradients)
    
    bias_gradient = np.mean(hypothesis(weights, bias) - target)
    
    return (weight_gradients, bias_gradient)


# In[15]:


# Check to make sure it works
# Initial gradient should be large

gradient(weights_init, bias_init)


# ### 2.3: Run Gradient Descent
# 1. You want to update the weights by subtracting the partial derivative * some learning rate alpha.
# 2. Do the same for the bias
# 3. Append the cost of the new weights and bias to an array of costs using np.append()
# 4. Repeat for some number (we call this the number of epochs, or iterations of steps we're completing during gradient descent)
# 5. As always, the function header is defined for you. Complete the rest!

# In[16]:


# Gradient descent algorithm:
# Repeat for desired iterations: Calculate gradient, move down one step
# Cost should decrease over time

LEARNING_RATE = 0.01

def gradient_descent(weights, bias, num_epochs):
    costs = []
    weights = weights
    bias = bias
    
    for i in range(num_epochs):
        weights_gradient, bias_gradient = gradient(weights, bias)
    
        # write your code here:
        weights = weights - LEARNING_RATE * weights_gradient
        bias = bias - LEARNING_RATE * bias_gradient
        costs.append(cost(weights, bias))
        
    return costs, weights, bias


# In[17]:


costs, trained_weights, trained_bias = gradient_descent(weights_init, bias_init, 1000)


# In[18]:


print(trained_weights)
print(trained_bias)


# ## Part 4: Evaluating the Model

# In[19]:


import matplotlib.pyplot as plt


# In[20]:


plt.plot(costs)
plt.xlabel("Iteration number (Epoch)")
plt.ylabel("Cost")
plt.show()


# In[21]:


# Final predicted prices
new_hypotheses = hypothesis(trained_weights, trained_bias)


# In[22]:


# Make sure predictions, actual values are correlated

plt.scatter(target, new_hypotheses)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.show()


# ### Congrats! You just did machine learning
# 
# ---

# ## Part 5: Using sklearn's built-in linear regression functionality:

# In[23]:


from sklearn import linear_model
regr = linear_model.LinearRegression()


# In[24]:


## call the .fit() function on regr using data and target. Yes it's that easy:
##################################

regr.fit(data, target)


# In[25]:


plt.scatter(target, regr.predict(data))
plt.xlabel("Actual prices (Test)")
plt.ylabel("Predicted prices (Test)")
plt.show()


# ## Train Test Split:
# What we often do in machine learning is split our data into a training set and a testing set. This is so that once we train our model on our training set, we aren't making predictions on the same input, as that would give us "too-good" answers, so instead we put aside some data into a testing set and make predictions on that once we've trained our model
# 

# In[26]:


from sklearn.model_selection import train_test_split

## Using sklearn's train_test_split() function, create 4 variables X_train, X_test, Y_train, Y_test. 
## For function parameters, the test size will be 0.25, and the random_state will be 5. 
## Print each of these variables:

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size = 0.25, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[27]:


## use .fit() to train the regression model below
regr.fit(X_train, Y_train)


# In[28]:


plt.scatter(Y_test, regr.predict(X_test))
plt.xlabel("Actual prices (Test)")
plt.ylabel("Predicted prices (Test)")
plt.show()

