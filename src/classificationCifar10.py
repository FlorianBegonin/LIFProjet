# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:58:07 2017

@author: Florian
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


number_by_batch = 10000
global number_used
number_used = 0

d1 = unpickle("../data/data_batch_1")
d2 = unpickle("../data/data_batch_2")
d3 = unpickle("../data/data_batch_3")
d4 = unpickle("../data/data_batch_4")
d5 = unpickle("../data/data_batch_5")

d = np.array([d1,d2,d3,d4,d5])

def next_batch(n):
    batch = np.zeros((n,32,32,3))
    labels = np.array([], dtype = int)
    global number_used
    number_used += n
    for i in range(number_used,number_used + n):
        index = int(np.floor(i/number_by_batch))
        temp =d[index][b'data'][i-(index*number_by_batch)]
        temp2 = np.reshape(temp, (3,1024))
        temp2 = np.transpose(temp2)
        temp2 = np.reshape(temp2, (32,32,3))
        batch[i-number_used]= temp2
        labels = np.append(labels, convertToPlaceholder(d[index][b'labels'][i-(index*number_by_batch)]))
    labels = labels.reshape((n,10))
    return batch , labels


def generate_image (batch, labels, batch_size):
    
    for i in range (batch_size):
        image = batch[i]
        plt.imshow(image)
        plt.show()
        print(labels[i])
        new_image = tf.image.random_flip_left_right(image)
        new_image = tf.image.random_brightness(new_image,50)
        new_image = tf.image.random_contrast (new_image,0.2,1.5)
        final_image = tf.image.per_image_standardization(new_image)
        print(final_image)
        print(image.shape)
        print(new_image)
        print(batch.shape)
        batch[batch_size + i] = tf.cast(final_image, dtype = tf.float32)
        labels[batch_size + i] = labels[i]

    return batch,labels


def convertToPlaceholder(i):
    result = [0]*10
    result[i] = 1
    return result

def conv2D (x,W):
    #Input : x - données d'entrée / w - tableau de poids
    #Out : 2D convolution layer
    
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def pool2x2 (x):
    #Input : x - données d'entrée
    #Out : 2x2 max pooling layer
    
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
        

def weight_variable(shape):
    #Input : shape
    #Output : poids au format demandé
  
    initial = tf.truncated_normal(shape, stddev = 0.5)
    return tf.Variable(initial)


def bias_variable(shape):
    #Input : shape
    #Output : poids au format demandé
    
    initial = tf.constant(0.5, shape=shape)
    return tf.Variable(initial)


def createNetwork(x):

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 32, 32, 3])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2D(x_image, W_conv1) + b_conv1)
        
    with tf.name_scope('pool1'):
        #image 16*16 en sortie
       h_pool1 = pool2x2(h_conv1)
       
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2D(h_pool1, W_conv2) + b_conv2)
    
    with tf.name_scope('pool2'):
        #image 8*8 en sortie
        h_pool2 = pool2x2(h_conv2)
        
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([5, 5, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2D(h_pool2, W_conv3) + b_conv3)
        
    with tf.name_scope('pool3'):
        #image 4*4 en sortie
        h_pool3 = pool2x2(h_conv3)
        
    with tf.name_scope('flat'):
        h_flat = tf.reshape(h_pool3,[-1,4*4*128])
    
        
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([4*4*128 , 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024 , 10])
        b_fc2 = bias_variable([10])
        y_out = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        
    return y_out,keep_prob


def main ():
    
    
    #placeholder données d'entrée
    x = tf.placeholder(tf.float32, [None,32,32,3])
    
    #labels
    y = tf.placeholder(tf.float32, [None,10])
    
    y_out,keep_prob = createNetwork(x)
    
    with tf.name_scope('loss'):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_out)
        loss = tf.reduce_mean(loss)
        
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
        
    with tf.name_scope('accuracy'):
        #Compare les résultats aux labels 
        correct_pred = tf.equal(tf.argmax(y_out, 1), tf.argmax(y,1))
        correct_pred = tf.cast (correct_pred, tf.float32)
    
    session = tf.Session()
        
    session.run(tf.global_variables_initializer())
    #50000 images => 48000 training + 2000 validate
    num_iteration = 1200
    batch_size_train = 40
    batch_size_validate = 100
    accuracy = tf.reduce_mean(correct_pred)

    
    
    total_iteration = 0
    
    for i in range (total_iteration, total_iteration + num_iteration):
        batch = next_batch(batch_size_train)
        x_batch = batch[0]
        y_batch = batch[1]
        
        feed_dict_train = {x : x_batch , y : y_batch, keep_prob : 0.5}
        
        session.run(optimizer, feed_dict = feed_dict_train)
        
        if i%100 == 0:
            val_loss = session.run(loss, feed_dict = feed_dict_train)
            acc_train = session.run(accuracy, feed_dict = feed_dict_train)
            #acc_val = session.run(accuracy, feed_dict = feed_dict_val)
            print ("step : ",i," accuracy training: ",acc_train," loss : ",val_loss)
                
        
        total_iteration += num_iteration
            
        
    validation_batch = next_batch(batch_size_validate)
    x_batch = validation_batch[0]
    y_batch = validation_batch[1]
    final_accuracy = session.run(accuracy, {x : x_batch , y : y_batch, keep_prob : 1.0})
    print("final accuracy :",final_accuracy)

    

    
        
    


main()
