'''
Hayleigh Sanders - Final Project
RIN# 661195735
'''
import tarfile
import pickle
import random
import numpy as np
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot as plt
import random

from scipy.ndimage.filters import convolve as filter2
import numpy as np
from typing import Tuple
from PIL import Image
import numpy as np
from numpy import array
import matplotlib.pyplot as plt

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from random import seed
from random import random
seed(1)

import tarfile
import pickle
import numpy as np

import warnings

from tensorflow.contrib import rnn

import pickle

HSKERN = np.array([[1/12, 1/6, 1/12],
                   [1/6,    0, 1/6],
                   [1/12, 1/6, 1/12]], float)

kernelX = np.array([[-1, 1],
                    [-1, 1]]) * .25  # kernel for computing d/dx

kernelY = np.array([[-1, -1],
                    [1, 1]]) * .25  # kernel for computing d/dy

kernelT = np.ones((2, 2))*.25

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def HornSchunck(im1: np.ndarray, im2: np.ndarray, *,
                alpha: float = 0.001, Niter: int = 8,
                verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    im1: numpy.ndarray
        image at t=0
    im2: numpy.ndarray
        image at t=1
    alpha: float
        regularization constant
    Niter: int
        number of iteration
    """
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # set up initial velocities
    uInitial = np.zeros([im1.shape[0], im1.shape[1]])
    vInitial = np.zeros([im1.shape[0], im1.shape[1]])

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)
    #print (fx,fy,ft)

    if verbose:
        from .plots import plotderiv
        plotderiv(fx, fy, fz)

#    print(fx[100,100],fy[100,100],ft[100,100])

        # Iteration to reduce error
    for _ in range(Niter):
        # %% Compute local averages of the flow vectors
        uAvg = filter2(U, HSKERN)
        vAvg = filter2(V, HSKERN)
        
# %% common part of update step
        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
# %% iterative step
        u = uAvg - fx * der
        v = vAvg - fy * der
    #print(U[50][25])
    #fig, ax = plt.subplots()
    #plt.imshow(im1)
    thresh = 0
    #print(u,v)
    for x in range(0,im1.shape[0],5):
        for y in range(0,im1.shape[1],5):
            U = u[x][y]
            V = v[x][y]
            #print(U,V)
            #q = ax.quiver(y,x,U,V,width=.002,color='red')
            
    #plt.show()
    return u,v


def computeDerivatives(im1: np.ndarray, im2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx = filter2(im1, kernelX) + filter2(im2, kernelX)
    fy = filter2(im1, kernelY) + filter2(im2, kernelY)
    # ft = im2 - im1
    ft = filter2(im1, kernelT) + filter2(im2, -kernelT)
    return fx, fy, ft

#reset graph
tf.reset_default_graph()

#turn off annoying warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

labels = ["basketball shooting", "biking/cycling", "diving", "golf swinging", "horse back riding",
          "soccer juggling", "swinging", "tennis swinging", "trampoline jumping", "volleyball spiking", "walking with a dog"]

# load the dataset into memory
data_file1  =  open('youtube_action_train_data_part1.pkl',  'rb')
data_file2  =  open('youtube_action_train_data_part2.pkl',  'rb')
train_data1, train_labels1 = pickle.load(data_file1)
train_data2, train_labels2 = pickle.load(data_file2)
data_file1.close()
data_file2.close()

#concatenate separate parts
train_data = np.concatenate((train_data1,train_data2),axis=0)
train_labels = np.concatenate((train_labels1,train_labels2),axis=0)


#Convolution function
def convolution(x):
    keep=.7
    #initialize filters
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 32], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.08))
    conv3_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.08))

    #First convolutional layer
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    print(conv1.shape)
    conv1 = tf.nn.relu(conv1) #activation
    print(conv1.shape)
    #First pooling layer
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    print("conv1 pool:",conv1_pool.shape)
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    #second convolutional layer
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
    print(conv2.shape)
    conv2 = tf.nn.relu(conv2) #activation
    #second pooling layer
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    print("conv2 pool:",conv2_pool.shape)
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    #third convolutional layer
    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
    print("conv3: ",conv3.shape)
    conv3 = tf.nn.relu(conv3) #activation
    conv3_bn = tf.layers.batch_normalization(conv3)

    #flatten 
    flat = tf.contrib.layers.flatten(conv3)
    print("flat: ",flat.shape)

    #fully connected layer 1
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep)
    full1 = tf.layers.batch_normalization(full1)
    print("full 1: ",full1.shape)

    #fully connected layer 2
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep)
    full2 = tf.layers.batch_normalization(full2)
    print("full 2: ",full2.shape)

    #fully connected layer 3
    full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
    full3 = tf.nn.dropout(full3, keep)
    full3 = tf.layers.batch_normalization(full3)
    print("full 3: ",full3.shape)

    #output
    out = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=11, activation_fn=None)
    return out

def one_hot(x):
    enc = np.zeros((11))
    enc[x] = 1
    return enc

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

save_model_path = './my_model'

#Load data from the pickle file, normalize and save numpy arrays
#Do this once and use the saved normalized arrays because the normalization process takes a long time

#print(train_data.shape)

batch_size = 5
#Hyperparameters
epochs = 50
kp = 0.7
learning_rate = 0.01 #learning rate for optimizer


Ntrain = 5000 # Size of the testing dataset. Maximum = 7000
order = np.arange(train_data.shape[0])
np.random.shuffle(order)
train_data = train_data[order[:Ntrain]]
train_labels = train_labels[order[:Ntrain]]

test_data = train_data[5000:]
test_labels = train_labels[5000:]

num_classes = 11

#reset graph
tf.reset_default_graph()

#input
input_frames = tf.placeholder(dtype=tf.uint8, shape=[None, 30, 64, 64, 3], name='input_frames')
#input_img = tf.placeholder(tf.float32, shape=(None, 64, 64, 3), name='input_img')
input_img = tf.placeholder(tf.float32, shape=(None, 64,64,3), name='input_img')

y =  tf.placeholder(tf.float32, shape=(None, 11), name='output_y')
keep = tf.placeholder(tf.float32, name='keep')

#batch = tf.placeholder(tf.float32,shape=(batch_size, 10, 65536))
label = tf.placeholder(tf.float32, shape=(None,11), name='label')
logits = tf.placeholder(tf.float32, shape=(None,11), name='logits')

# Build model
logits = convolution(input_img)
action_lbl = tf.argmax(logits, axis = 1, name='action_lbl')

#loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

#initialize adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))


# Create the collection.
tf.get_collection("validation_nodes")
# Add stuff to the collection.
tf.add_to_collection("validation_nodes",input_frames)     
tf.add_to_collection("validation_nodes",action_lbl)
#accuracy - check if the predicted argmax(logits,1) from the model
#is the same as the labels tf.argmax(y,1)
#correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
# start training
saver = tf.train.Saver()
correct=0

iters=[]
trainacc=[]
testacc=[]

trainloss=[]
testloss=[]

with tf.compat.v1.Session() as sess:
    # Initialize global variables
    sess.run(tf.compat.v1.global_variables_initializer())

    input_frames = train_data
    z=0
    #epochs=int(train_data.shape[0]/batch_size)
    
    batch = np.zeros((0,64,64,3))
    l = np.zeros((0,11))
    batchHS = np.zeros((0,64,64,3))
    
    batcht = np.zeros((0,64,64,3))
    lt = np.zeros((0,11))
    batchHSt = np.zeros((0,64,64,3))
    
    for epoch in range(0,epochs):
        print("_____")
        print("epoch:",epoch)
        #get a batch of shuffled data
        q = train_data[z:z+batch_size]
        ql = train_labels[z:z+batch_size]
        print("q shape:",q.shape)
        z=z+batch_size
        for i in range(0,q.shape[0]):
            features = q[i][15]
            features=normalize(features)
            feat2 = q[i][16]
            feat2=normalize(feat2)

            #get optical flow
            u,v = HornSchunck(rgb2gray(features),rgb2gray(feat2))
            uv = np.zeros((64,64,3))
            for xx in range(0,64):
                for yy in range(0,64):
                    #print(u[x][y])
                    uv[xx][yy] = [u[xx][yy], v[xx][yy], 0]
            batchHS = np.append(batchHS,[uv],axis=0)
            
            ll = one_hot(ql[i])
            l = np.append(l,[ll],axis=0)
            batch = np.append(batch,[features],axis=0)
            

        #print("labels:",l)
        print("batch shape:",batch.shape)
        print("batchHS shape:",batchHS.shape)

        #Get CNN of both streams
        logs=sess.run(logits,feed_dict={input_img: batch, y: l, keep: 1.})
        logshs=sess.run(logits,feed_dict={input_img: batchHS, y: l, keep: 1})
        print("logs shape:",logs.shape)
        correct=0
        tot=0
        
        #print("logits:",np.argmax(logs[0]),"labels:",np.argmax(l[0]))
        #print("logitsHS:",np.argmax(logshs[0]),"labels:",np.argmax(l[0]))

        #Average streams
        loglog = np.add(logs,logshs)/2
        #print("loglog",loglog.shape)
        '''
        for zz in range(0,batch_size):
            if np.argmax(loglog[0]) == np.argmax(l[0]):
                correct=correct+1
            tot=tot+1
        #print("Correct: ",(correct/tot))
        '''

        loss = sess.run(cost,feed_dict={input_img: batch, y: l, keep: 1.})
        
        trainloss.append(loss)
                
        sess.run(optimizer,feed_dict={input_img: batch, y: l, keep: 1.})
        #loss = sess.run(cost,feed_dict={input_img: batch,y: l,keep: 1.})
        
        cp = sess.run(correct_prediction,feed_dict={input_img: batch,y: l,keep: 1.})
        cp2 = (np.count_nonzero(cp==True))/cp.shape[0]

        trainacc.append(cp2)

        counts=[0,0,0,0,0,0,0,0,0,0,0]
        corrects=[0,0,0,0,0,0,0,0,0,0,0]
        if(epoch==epochs-1):
            for lab in range(0,l.shape[0]):
                #print(cp[lab],l[lab],np.argmax(l[lab]),np.argmax(logs[lab]))
                lbl = np.argmax(l[lab])
                
                counts[lbl]=counts[lbl]+1
                guess = np.argmax(logs[lab])
                if cp[lab]==True:
                    corrects[lbl]=corrects[lbl]+1
                

        print("Train Loss:",loss,"Train Accuracy:",cp2)

        #Compute testing loss and accuracy vs epochs
            
       #get a batch of shuffled data
        for i in range(0,test_data.shape[0]):
            features = test_data[i][15]
            features = normalize(features)
            feat2 = q[i][14]
            feat2 = normalize(feat2)

            #get optical flow
            u,v = HornSchunck(rgb2gray(features),rgb2gray(feat2))
            uv = np.zeros((64,64,3))
            for xx in range(0,64):
                for yy in range(0,64):
                    #print(u[x][y])
                    uv[xx][yy] = [u[xx][yy], v[xx][yy], 0]
            batchHSt = np.append(batchHSt,[uv],axis=0)
            
            llt = one_hot(test_labels[i])
            lt = np.append(lt,[llt],axis=0)
            batcht = np.append(batcht,[features],axis=0)

        #print("labels:",l)
        print("batch shape:",batcht.shape)
        print("batchHS shape:",batchHSt.shape)
        logst=sess.run(logits,feed_dict={input_img: batcht, y: lt, keep: 1.})
        logshst=sess.run(logits,feed_dict={input_img: batchHSt, y: lt, keep: 1})
        print("logs shape:",logs.shape)
        correct=0
        tot=0
        
        #print("logits:",np.argmax(logst[0]),"labels:",np.argmax(lt[0]))
        #print("logitsHS:",np.argmax(logshst[0]),"labels:",np.argmax(lt[0]))
        loglogt = np.add(logst,logshst)/2
        print("loglog",loglogt.shape)
        '''
        for zz in range(0,batch_size):
            if np.argmax(loglogt[zz]) == np.argmax(lt[zz]):
                correct=correct+1
            tot=tot+1
        '''

        loss2 = sess.run(cost,feed_dict={input_img: batcht, y: lt, keep: 1.})
        
        testloss.append(loss2)
        
        cp = sess.run(correct_prediction,feed_dict={input_img: batch,y: l,keep: 1.})
        cp2 = (np.count_nonzero(cp==True))/cp.shape[0]
        
        testacc.append(cp2)
        
        print("Test Loss:",loss2,"Test Accuracy:",cp2)

        iters.append(epoch)

    #print(testloss)
    #plot data
    title = "Epochs vs. Training and Testing Loss"
    plt.plot(iters,trainloss,label="Training Loss")
    plt.plot(iters,testloss,label="Testing Loss")
    plt.legend(loc=1, prop={'size': 10})
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.title(title)
    #plt.show()
    
    '''
    #plot data
    title = "Epochs vs. Training and Testing Accuracy"
    plt.plot(iters,trainacc,label="Training Accuracy")
    plt.plot(iters,testacc,label="Testing Accuracy")
    plt.legend(loc=1, prop={'size': 10})
    plt.ylabel('Classification Accuracy')
    plt.xlabel('Number of Epochs')
    plt.title(title)
    plt.show()
    '''
    saver = tf.compat.v1.train.Saver()
    save_path = saver.save(sess, save_model_path)








        

