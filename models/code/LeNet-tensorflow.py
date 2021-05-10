#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np


### Load MNIST Data

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28, 28, 1) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

print("Image Shape: {}".format(X_train[0].shape))
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_valid)))
print("Test Set:       {} samples".format(len(X_test)))

'''
# The MNIST data that TensorFlow pre-loads comes as 28x28x1 images.
# However, the LeNet architecture only accepts 32x32xC images, where C is the
# number of color channels.
# In order to reformat the MNIST data into a shape that LeNet will accept, we
# pad the data with two rows of zeros on the top and bottom, and two columns
# of zeros on the left and right (28+2+2 = 32).
'''
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_valid      = np.pad(X_valid, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))


### Visualize Data
## View a sample from the dataset.
#import random
#import matplotlib.pyplot as plt
#
#index = random.randint(0, len(X_train))
#image = X_train[index].squeeze()
#
#plt.figure(figsize=(1,1))
#plt.imshow(image, cmap="gray")
#print(y_train[index])


## Preprocess Data
# Shuffle the training data.
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)


### Setup TensorFlow

# `EPOCH` and `BATCH_SIZE` values affect the training speed and model accuracy.
EPOCHS = 10
BATCH_SIZE = 128

'''
 ## LeNet-5 Architecture
 **Layer 1: Convolutional.** The output shape should be 28x28x6.
 
 **Activation.** Your choice of activation function.
 
 **Pooling.** The output shape should be 14x14x6.
 
 **Layer 2: Convolutional.** The output shape should be 10x10x16.
 
 **Activation.** Your choice of activation function.
 
 **Pooling.** The output shape should be 5x5x16.
 
 **Flatten.** Flatten the output shape of the final pooling layer such that
             it's 1D instead of 3D. The easiest way to do is by using 
             `tf.contrib.layers.flatten`, which is already imported for you.
 
 **Layer 3: Fully Connected.** This should have 120 outputs.
 
 **Activation.** Your choice of activation function.
 
 **Layer 4: Fully Connected.** This should have 84 outputs.
 
 **Activation.** Your choice of activation function.
 
 **Layer 5: Fully Connected (Logits).** This should have 10 outputs.
 
 ## Output
 Return the result of the 2nd fully connected layer.
 
'''

from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for
    # the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), \
                                              mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], \
                           padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], \
                           padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), \
                                              mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], \
                           padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], \
                           padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120),\
                                            mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), \
                                             mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 10), \
                                             mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


### Features and Labels
    
# Train LeNet to classify [MNIST](http://yann.lecun.com/exdb/mnist/) data.
# `x` is a placeholder for a batch of input images.
# `y` is a placeholder for a batch of output labels.
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)


### Training Pipeline

# Create a training pipeline that uses the model to classify MNIST data.
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y,\
                                                        logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


### Model Evaluation

# Evaluate how well the loss and accuracy of the model for a given dataset.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], \
        y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, \
                            feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


### Train the Model
    
# Run the training data through the training pipeline to train the model.
# Before each epoch, shuffle the training set.
# After each epoch, measure the loss and accuracy of the validation set.
# Save the model after training.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, "./tf_models/my_model.ckpt")
    print("Model saved")


### Evaluate the Model
    
# Once you are completely satisfied with your model, evaluate the performance
# of the model on the test set.
# Be sure to only do this once!
# If you were to measure the performance of your trained model on the test set,
# then improve your model, and then measure the performance of your model on
# the test set again, that would invalidate your test results. You wouldn't 
#get a true measure of how well your model would perform against real data.

with tf.Session() as sess:
    saver.restore(sess, "./tf_models/my_model.ckpt")

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

