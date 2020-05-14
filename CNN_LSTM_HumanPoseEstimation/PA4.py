'''
Hayleigh Sanders
RIN# 661195735
Programming Assignment 4
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

#turn off annoying warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def convolution(x):
    q = tf.zeros(shape=[0,10,65536])
    for i in range(0,batch_size):
        keep=.7
        #initialize filters
        conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 16], mean=0, stddev=0.08))
        #print(x)
        #First convolutional layer
        conv1 = tf.nn.conv2d(x[i], conv1_filter, strides=[1,1,1,1], padding='SAME')
        #print(conv1.shape)
        conv1 = tf.nn.relu(conv1) #activation
        #print(conv1.shape)

        #flatten
        #print("conv3_bn:",conv1.shape)
        flat = tf.contrib.layers.flatten(conv1)
        #print("flat: ",flat.shape)
        flat = tf.reshape(flat,[1,10,65536])

        #output
        #out = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=14, activation_fn=None)
        q = tf.concat([q,flat], axis=0)
    return q

def RNN(x):
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)
    # define your RNN network, the length of the sequence will be automatically retrieved
    h_val, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    # collection of all the final output
    final_output = tf.zeros(shape=[batch_size, 0, 7, 2])
    for i in np.arange(10):
        temp = tf.reshape(h_val[:, i, :], [batch_size, num_units])
        output = tf.matmul(temp, weights['out']) + biases['out']
        output = tf.reshape(output, [-1, 1, 7, 2])
        final_output = tf.concat([final_output, output], axis=1)
    final_output = tf.reshape(final_output, [batch_size, 10, 14])
    return final_output

def normalize(td):
    # normalize the training data over each image
    td = td - np.mean(td, axis=(2, 3, 4), keepdims=True)
    td = td / ( np.std(td, axis=(2, 3, 4), keepdims=True))
    return td

save_model_path = './my_model'

#Load data from the pickle file, normalize and save numpy arrays
#Do this once and use the saved normalized arrays because the normalization process takes a long time

#print(train_data.shape)

#Hyperparameters
batch_size = 10
epochs = 50
kp = 0.7
learning_rate = 0.01 #learning rate for optimizer

#set data subset for training and get randomly shuffled indexes
data_subset = 5000 #choose a subset out of 7000 data samples for our training set
rand_indexes = np.random.permutation(data_subset) #shuffle indexes of training data subset
#print(train_data[rand_indexes[0]])

num_units = 100
num_classes = 14

#reset graph
tf.reset_default_graph()

#input
input_frames = tf.placeholder(dtype=tf.uint8, shape=[None, 10, 64, 64, 3], name='input_frames')
#input_frames = tf.placeholder(tf.float32, shape=(data_subset, 10, 64, 64, 3), name='input_frames')
input_batch = tf.placeholder(tf.float32, shape=(None, 10, 64, 64, 3), name='input_batch')
batch_labels =  tf.placeholder(tf.float32, shape=(batch_size, 10, 14), name='batch_labels')
#joint_pos = tf.placeholder(tf.float32, shape=(batch_size, 10, 7, 2), name='joint_pos')
#y =  tf.placeholder(tf.float32, shape=(None, 14), name='output_y')
keep = tf.placeholder(tf.float32, name='keep')
#batch = tf.placeholder(tf.float32,shape=(batch_size, 10, 65536))

#initialize weights
weights = {
    'out': tf.Variable(tf.random_normal([num_units, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Build model
conv_fcl = convolution(input_batch) #return fully connected leayer of the CNN
#batch =  tf.Variable(tf.float32, shape=(0,10,65536))

lstm_out = RNN(conv_fcl)

#Feed value to joint_pos
joint_pos = tf.identity(lstm_out, name='joint_pos')
#joint_pos = lstm_out
#predict_lbl = tf.identity(tf.argmax(logits, 1), name='predict_lbl') #Prediction model

#loss
cost = tf.reduce_mean(tf.square(batch_labels-lstm_out))

#initialize adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Create the collection.
tf.get_collection("validation_nodes")
# Add stuff to the collection.
tf.add_to_collection("validation_nodes",input_frames)
tf.add_to_collection("validation_nodes",joint_pos)
# start training
saver = tf.train.Saver()

losses = []
iters=[]
trainerror = []
testerror=[]

e0=[]
e1=[]
e2=[]
e3=[]
e4=[]
e5=[]
e6=[]

with tf.compat.v1.Session() as sess:
    # Initialize global variables
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # load the dataset into memory
    data_file = open('youtube_train_data.pkl', 'rb')
    train_data, train_labels = pickle.load(data_file)
    data_file.close()

    input_frames = train_data[:data_subset]
    #normalize data
    train_data_n = normalize(input_frames)
    #Train over epochs
    for epoch in range(epochs):
        print("Epoch:",epoch)
        #loop through minibatches
        batch = np.zeros((0,10,64,64,3),dtype=np.float32)
        b = np.zeros((0,10,65536),dtype=np.float32)
        y = np.zeros((0,10,14),dtype=np.float32)
        #batch = np.zeros(0,10,64,64,3)
        #q = tf.reshape(train_data, (10,64,64,3))
        #print("Reshaped:",q.shape)
        z=0
        for i in range(z,z+batch_size):
            #train the neural network
            
            #get a batch of shuffled data
            seq = train_data_n[rand_indexes[i]]
            batch = np.append(batch,[seq],axis=0)
            l = train_labels[rand_indexes[i]]
            l = np.reshape(l,(10,14))
            #print("l shape:",l.shape)
            #print("y shape:",y.shape)
            y = np.append(y,[l],axis=0)
            z=z+batch_size
        print(batch.shape)
        x = sess.run(conv_fcl,feed_dict={input_batch: batch, keep: kp})
        #print(x.shape)
        #get CNN output
        z = sess.run(lstm_out,feed_dict={conv_fcl: x})
        #get LSTM output
        c = sess.run(cost,feed_dict={lstm_out: z, batch_labels: y})
        losses.append(c)
        iters.append(epoch)
        print("Cost:",c)
        print("Y-hat - Y:",abs(sum(z[0][0]-y[0][0])/len(z[0][0])))
        diff = 0
        tot=0
        #print("z shape:",z.shape)
        z2 = np.reshape(z,(10,10,7,2))
        #print("y shape:",y.shape)
        y2 = np.reshape(y,(10,10,7,2))
        #find average error
        for a in range(0,z2.shape[0]):
            for b in range(0,10):
                for c in range(0,7):
                    diff = diff+abs(z2[a][b][c][0] - y2[a][b][c][0])+abs(z2[a][b][c][1] - y2[a][b][c][1])
                    tot = tot+2
                    if c==0:
                        e0.append(diff)
                    if c==1:
                        e1.append(diff)
                    if c==2:
                        e2.append(diff)
                    if c==3:
                        e3.append(diff)
                    if c==4:
                        e4.append(diff)
                    if c==5:
                        e5.append(diff)
                    if c==6:
                        e6.append(diff)
        print("avg training pixel error:",diff/tot,"for epoch",epoch)
        trainerror.append(diff/tot)
        
        #Compute testing pixel error for this epoch
        test_data = train_data[data_subset:]
        test_data_n = normalize(test_data)
        print("test data",test_data.shape)
        test_labels = train_labels[data_subset:]
        #loop through minibatches
        batch = np.zeros((0,10,64,64,3),dtype=np.float32)
        b = np.zeros((0,10,65536),dtype=np.float32)
        y = np.zeros((0,10,14),dtype=np.float32)
        #batch = np.zeros(0,10,64,64,3)
        #q = tf.reshape(train_data, (10,64,64,3))
        #print("Reshaped:",q.shape)
        z=0
        for i in range(z,z+batch_size):
            #train the neural network
            seq = test_data_n[i]
            batch = np.append(batch,[seq],axis=0)
            l = test_labels[i]
            l = np.reshape(l,(10,14))
            #print("l shape:",l.shape)
            #print("y shape:",y.shape)
            y = np.append(y,[l],axis=0)
            z=z+batch_size
        print(batch.shape)
        x = sess.run(conv_fcl,feed_dict={input_batch: batch, keep: kp})
        #print(x.shape)
        z = sess.run(lstm_out,feed_dict={conv_fcl: x})
        diff = 0
        tot=0
        #print("z shape:",z.shape)
        z2 = np.reshape(z,(10,10,7,2))
        #print("y shape:",y.shape)
        y2 = np.reshape(y,(10,10,7,2))
        for a in range(0,z2.shape[0]):
            for b in range(0,10):
                for c in range(0,7):
                    diff = diff+abs(z2[a][b][c][0] - y2[a][b][c][0])+abs(z2[a][b][c][1] - y2[a][b][c][1])
                    tot = tot+2
        print("avg testing pixel error:",diff/tot,"for epoch",epoch)
        testerror.append(diff/tot)
        
        #Plot predicted coordinates of the last epoch
        if(epoch == epochs-1):
            img = batch[0][0]
            coord = np.reshape(z[0][0],(7,2))
            gt = np.reshape(y[0][0],(7,2))
            coord = coord.T
            gt = gt.T
            print(coord)
            plt.scatter(coord[0],coord[1],color='red')
            plt.scatter(gt[0],gt[1],color='blue')
            print(gt)
            plt.imshow(img)
            #plt.show()
        #print("Y-hat - Y:",abs(sum(z[1][0]-y[0][0])/len(z[1][0])))
        #print("Y-hat - Y:",abs(sum(z[2][0]-y[0][0])/len(z[2][0])))

        #Optimizer - train model
        sess.run(optimizer,feed_dict={input_batch: batch, batch_labels: y, keep: kp})

    tdata = normalize(input_frames)
    data_size = tdata.shape[0]
    tdata = tdata.astype('float32')
    print("tdata shape:",tdata.shape)
    print(data_size/batch_size)
    
    p=np.zeros((0,10,7,2),dtype=np.float32)
    
    for i in range(0,int(data_size/batch_size)):
        chunk = tdata[i:i+batch_size]
        #print(tdata[i:i+batch_size].shape)
        #print(i)
        x = sess.run(conv_fcl,feed_dict={input_batch: tdata, keep: kp})
        #print(x.shape)
        jp = sess.run(lstm_out,feed_dict={conv_fcl: x})
        jp = jp.astype('float32')
        jp = np.reshape(jp,(batch_size,10,7,2))
        #print("jp shape:",jp.shape)
        #print("p shape:",p.shape)
        for s in range(0,batch_size):
            p = np.append(p,[jp[s]],axis=0)
        i=i+batch_size
    
    print("p shape:",p.shape)
    #joint_pos = tf.identity(p, name='joint_pos')
    ys = train_labels[0:data_subset]
    print("ys shape:",ys.shape)
    diff = 0
    tot=0
    for a in range(0,p.shape[0]):
        for b in range(0,10):
            for c in range(0,7):
                diff = diff+abs(p[a][b][c][0] - ys[a][b][c][0])+abs(p[a][b][c][1] - ys[a][b][c][1])
                tot = tot+2
    print("avg pixel error:",diff/tot)
    '''
    #plot data
    title = "Epochs vs. Loss"
    plt.plot(iters,losses,label="Loss")
    plt.legend(loc=1, prop={'size': 10})
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.title(title)
    #plt.show()
    '''
    '''
    #plot data
    title = "Epochs vs. Average Pixel Distance Error"
    plt.plot(iters,trainerror,label="Training Data Error")
    plt.plot(iters,testerror,label="Testing Data Error")
    plt.legend(loc=1, prop={'size': 10})
    plt.ylabel('Average Pixel Distance Error')
    plt.xlabel('Number of Epochs')
    plt.title(title)
    plt.show()
    '''
    p0 = np.zeros(20)
    p1=np.zeros(20)
    p2=np.zeros(20)
    p3=np.zeros(20)
    p4=np.zeros(20)
    p5=np.zeros(20)
    p6=np.zeros(20)
    for p in range(0,20):
        p0[p] = sum(i>p for i in e0)/len(e0)
        p1[p] = sum(i>p for i in e1)/len(e1)
        p2[p] = sum(i>p for i in e2)/len(e2)
        p3[p] = sum(i>p for i in e3)/len(e3)
        p4[p] = sum(i>p for i in e4)/len(e4)
        p5[p] = sum(i>p for i in e5)/len(e5)
        p6[p] = sum(i>p for i in e6)/len(e6)
    '''
    #plot data
    iters=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    title = "Prediction accuracy within 20 plx"
    plt.plot(iters,p0,label="Head")
    plt.plot(iters,p1,label="Right Wrist")
    plt.plot(iters,p2,label="Left Wrist")
    plt.plot(iters,p3,label="Right Elbow")
    plt.plot(iters,p4,label="Left Elbow")
    plt.plot(iters,p5,label="Right Shoulder")
    plt.plot(iters,p6,label="Left Shoulder")

    plt.legend(loc=1, prop={'size': 10})
    plt.ylabel('accuracy[%]')
    plt.xlabel('pixel distance from GT')
    plt.title(title)
    plt.show()
    '''
    
   #Compute testing pixel error for this epoch
    test_data = train_data[data_subset:]
    test_data_n = normalize(test_data)
    print("test data",test_data.shape)
    test_labels = train_labels[data_subset:]
    #loop through minibatches
    batch = np.zeros((0,10,64,64,3),dtype=np.float32)
    b = np.zeros((0,10,65536),dtype=np.float32)
    y = np.zeros((0,10,14),dtype=np.float32)
    #batch = np.zeros(0,10,64,64,3)
    #q = tf.reshape(train_data, (10,64,64,3))
    #print("Reshaped:",q.shape)
    z=0
    for i in range(z,z+batch_size):
        #train the neural network
        seq = test_data_n[i]
        batch = np.append(batch,[seq],axis=0)
        l = test_labels[i]
        l = np.reshape(l,(10,14))
        #print("l shape:",l.shape)
        #print("y shape:",y.shape)
        y = np.append(y,[l],axis=0)
        z=z+batch_size
    print(batch.shape)
    x = sess.run(conv_fcl,feed_dict={input_batch: batch, keep: kp})
    #print(x.shape)
    z = sess.run(lstm_out,feed_dict={conv_fcl: x})
    diff = 0
    tot=0
    #print("z shape:",z.shape)
    z2 = np.reshape(z,(10,10,7,2))
    #print("y shape:",y.shape)
    jerror = np.zeros(7)
    counts=np.zeros(7)
    y2 = np.reshape(y,(10,10,7,2))
    for a in range(0,z2.shape[0]):
        for b in range(0,10):
            for c in range(0,7):
                diff = diff+abs(z2[a][b][c][0] - y2[a][b][c][0])+abs(z2[a][b][c][1] - y2[a][b][c][1])
                jerror[c] = jerror[c]+diff
                counts[c] = counts[c]+1
                tot = tot+2
    print("avg testing pixel error:",diff/tot)
    testerror.append(diff/tot)

    jerror = jerror/tot
    print(counts)
    '''
    #plots
    labs = ['Head','Right Wrist','Left Wrist','Right Elbow','Left Elbow','Right Shoulder','Left Shoulder']
    labs2=[0,1,2,3,4,5,6]
    plt.title('Pixel Error for Each Joint')
    plt.bar(labs2,jerror/counts*2,align='center',alpha=.5, width=.5)
    plt.ylabel('Average Pixel Distance Error')
    plt.xlabel('Class')
    plt.show()
    '''
    
    saver = tf.compat.v1.train.Saver()
    save_path = saver.save(sess, save_model_path)








        
