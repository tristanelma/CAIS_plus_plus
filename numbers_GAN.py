
# coding: utf-8

# In[1]:


# All the necesssary imports.
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Activation, Dropout
from keras.layers import LeakyReLU
from keras import initializers
from keras.datasets import mnist
from keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Load in the data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize to [-1, 1] (easier to work with)
x_train = (x_train.astype(np.float32) - 127.5) / 127.5

# For the sake of time don't work with the entire dataset.
x_train = x_train[:10000]
y_train = x_train[:10000]

# Flatten the data.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))


# In[3]:


# Plot out a sample image (reshaped to 28 x 28)
plt.imshow(x_train[0].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.show()


# In[5]:


# How many noise dimensions our generator should take in
NOISE_DIM = 100 # Feel free to tweak this and see what changes

def generator():
    ###################################
    #TODO: Implement
    model = Sequential()
    model.add(Dense(256, input_dim=NOISE_DIM, kernel_initializer=initializers.RandomNormal(stddev=.02)))
    model.add(Dense(512, activation=LeakyReLU(.2)))
    model.add(Dense(1024, activation=LeakyReLU(.2)))
    model.add(Dense(784, activation='tanh'))
    
    return model
    
    ###################################


# In[6]:


def discriminator():
    ###################################
    #TODO: Implement
    model = Sequential()
    model.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=.02)))
    model.add(Dense(512, input_dim=1024, activation=LeakyReLU(.2)))
    model.add(Dense(256, input_dim=512, activation=LeakyReLU(.2)))
    model.add(Dense(1, activation='sigmoid'))
    
    return model
    ###################################


# In[7]:


def combine(generator, discriminator):
    ###################################
    # TODO: Implement
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model
    ###################################


# In[8]:


# Compile the discriminator, generator, and full GAN. 

# Use this optimizer for each of the models
opt = Adam(lr=.0002, beta_1=0.5)

#############################################
# TODO: Compile generator and discriminator
g = generator()
g.compile(loss='binary_crossentropy', optimizer=opt)

d = discriminator()
d.compile(loss='binary_crossentropy', optimizer=opt)

gd = combine(g, d)
gd.compile(loss='binary_crossentropy', optimizer=opt)
#############################################


# In[9]:


# Helper function to display sample from the network
def disp_sample(g):
    noise = np.random.uniform(-1, 1, size=(batch_size, NOISE_DIM))
    generated_images = g.predict(noise, verbose=0)
    show_im = generated_images[0]
    show_im = (show_im + 1) / 2.0
    show_im = show_im.reshape(28, 28)
    plt.imshow(show_im, cmap='gray')
    plt.show()


# In[10]:


batch_size = 128

for epoch in range(100):
    print('Epoch #%d' % epoch)
    
    # Generate an image and display it.
    disp_sample(g)
    
    num_batches = int(x_train.shape[0] / batch_size)
    print('Number batches %i' % num_batches)
    for i in range(num_batches):
        # A training iteration
        
        # Generate noise.
        noise = np.random.uniform(-1, 1, size=(batch_size, NOISE_DIM))
        
        # Generate images from the noise using the generator.
        generated_images = g.predict(noise)
        
        # Grab the image batch for this iteration. 
        real_images = x_train[i * batch_size: (i+1) * batch_size]
        
        # Contains the real and fake images.
        X = np.concatenate([generated_images, real_images])
        
        # Labels if the sample is real (1) or not real (0). 
        y = np.concatenate([np.zeros(generated_images.shape[0]), np.ones(real_images.shape[0])])

        # Train the discriminator using the generated images and the real images.
        d.trainable = True
        d_loss = d.train_on_batch(X, y)
        d.trainable = False
        
        # Generate more noise to feed into the full gan network to train the generative portion. 
        noise = np.random.uniform(-1, 1, size=(batch_size, NOISE_DIM))

        # Get the g_loss
        g_loss = gd.train_on_batch(noise, np.ones(noise.shape[0]))
        
        print('%i(%i/%i) D: %.4f, G: %.4f' % (epoch, i, num_batches, d_loss, g_loss))

