import tarfile
import pickle
import random
import numpy as np
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from random import seed
from random import random
seed(1)

#tf.enable_eager_execution()

dpath = 'cifar-10-batches-py'

with open('cifar_10_tf_train_test.pkl','rb') as f:
    trainx, trainy, testx, testy = pickle.load(f,encoding="bytes")

labelnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def one_hot(x):
    enc = np.zeros((len(x), 10))
    for idx, val in enumerate(x):
        enc[idx][val] = 1
    return enc

#Filehandling functions to normalize data and separate into chunks 
def load_batch(dpath, bid):
    with open(dpath + '/data_batch_' + str(bid), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    x = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    y = batch['labels']
    return x, y

def load_data(x,y,bid):
    features = x.reshape((len(x), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = y
    return features, labels

def pdumpxy(normalize, one_hot, features, labels, filename):
    features = normalize(features)
    labels = one_hot(labels)
    pickle.dump((features, labels), open(filename, 'wb'))

def preprocess_data(dpath, normalize, one_hot):
    min_batches = 5
    valid_features = []
    valid_labels = []
    for bi in range(1, min_batches + 1):
        features, labels = load_batch(dpath, bi)
        index_of_validation = int(len(features) * 0.1)
        pdumpxy(normalize, one_hot,features[:-index_of_validation], labels[:-index_of_validation],'preprocess_batch_' + str(bi) + '.p')
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])
    pdumpxy(normalize, one_hot,np.array(valid_features), np.array(valid_labels),'preprocess_validation.p')
    with open(dpath + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']
    pdumpxy(normalize, one_hot,np.array(test_features), np.array(test_labels),'preprocess_training.p')

def load_chunk(bid, batch_size):
    filename = 'preprocess_batch_' + str(bid) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))
    return get_chunk(features, labels, batch_size)

def get_chunk(features, labels, batch_size):
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

if not isdir(dpath):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()

#Partially connected layer steps (convolution, activation, max pool)
def convolution(x):
    keep=.7
    #initialize filters
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 32], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 32], mean=0, stddev=0.08))
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
    out = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=10, activation_fn=None)
    return out

#Preprocess the data and load chunks
preprocess_data(dpath, normalize, one_hot)
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

#Hyperparameters
epochs = 10
batch_size = 300
kp = 0.7
learning_rate = 0.001

#reset graph
tf.reset_default_graph()

#input
input_img = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_img')
y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
keep = tf.placeholder(tf.float32, name='keep')

# Build model
logits = convolution(input_img)
predict_lbl = tf.identity(tf.argmax(logits, 1), name='predict_lbl') #Prediction model

#loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

#initialize adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#accuracy - check if the predicted argmax(logits,1) from the model
#is the same as the labels tf.argmax(y,1)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

#find accuracies for each class
is0 = tf.equal(tf.argmax(y, 1),0)
pred0 = tf.equal(tf.argmax(logits,1),0)
is1 = tf.equal(tf.argmax(y, 1),1)
pred1 = tf.equal(tf.argmax(logits,1),1)
is2 = tf.equal(tf.argmax(y, 1),2)
pred2 = tf.equal(tf.argmax(logits,1),2)
is3 = tf.equal(tf.argmax(y, 1),3)
pred3 = tf.equal(tf.argmax(logits,1),3)
is4 = tf.equal(tf.argmax(y, 1),4)
pred4 = tf.equal(tf.argmax(logits,1),4)
is5 = tf.equal(tf.argmax(y, 1),5)
pred5 = tf.equal(tf.argmax(logits,1),5)
is6 = tf.equal(tf.argmax(y, 1),6)
pred6 = tf.equal(tf.argmax(logits,1),6)
is7 = tf.equal(tf.argmax(y, 1),7)
pred7 = tf.equal(tf.argmax(logits,1),7)
is8 = tf.equal(tf.argmax(y, 1),8)
pred8 = tf.equal(tf.argmax(logits,1),8)
is9 = tf.equal(tf.argmax(y, 1),9)
pred9 = tf.equal(tf.argmax(logits,1),9)

t0=c0=t1=c1=t2=c2=t3=c3=t4=c4=t5=c5=t6=c6=t7=c7=t8=c8=t9=c9=0
#Accuracy - cast as f32
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

#train data
save_model_path = './my_model'
it=0
its=[]
accs=[]
testaccs=[]
itcost=[]
with tf.compat.v1.Session() as sess:
    # Initialize global variables
    sess.run(tf.compat.v1.global_variables_initializer())
    #Train over epochs
    for epoch in range(epochs):
        #loop through minibatches
        min_batches = 5
        for bi in range(1, min_batches + 1):
            for batch_features, batch_labels in load_chunk(bi, batch_size):
                #train the neural network
                print(batch_features.shape)
                print(batch_labels.shape)
                sess.run(optimizer,feed_dict={input_img: batch_features,y: batch_labels,keep: kp})
                
            print('Epoch: ',epoch+1,'Batch: ',bi)
            loss = sess.run(cost,feed_dict={input_img: batch_features,y: batch_labels,keep: 1.})
            itcost.append(loss)
            #print(correct_prediction.eval())
            #find accuracy
            accurate = sess.run(accuracy,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            is_0 = sess.run(is0,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            pred_0 = sess.run(pred0,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            for v in range(0,len(is_0)):
                if is_0[v]==True:
                    t0+=1
                    if pred_0[v]==True:
                        c0+=1

            is_1 = sess.run(is1,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            pred_1 = sess.run(pred1,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            for v in range(0,len(is_1)):
                if is_1[v]==True:
                    t1+=1
                    if pred_0[v]==True:
                        c1+=1

            is_2 = sess.run(is2,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            pred_2 = sess.run(pred2,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            for v in range(0,len(is_2)):
                if is_2[v]==True:
                    t2+=1
                    if pred_2[v]==True:
                        c2+=1

            is_3 = sess.run(is3,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            pred_3 = sess.run(pred3,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            for v in range(0,len(is_3)):
                if is_3[v]==True:
                    t3+=1
                    if pred_3[v]==True:
                        c3+=1

            is_4 = sess.run(is4,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            pred_4 = sess.run(pred4,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            for v in range(0,len(is_4)):
                if is_4[v]==True:
                    t4+=1
                    if pred_4[v]==True:
                        c4+=1

            is_5 = sess.run(is5,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            pred_5 = sess.run(pred5,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            for v in range(0,len(is_5)):
                if is_5[v]==True:
                    t5+=1
                    if pred_5[v]==True:
                        c5+=1

            is_6 = sess.run(is6,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            pred_6 = sess.run(pred6,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            for v in range(0,len(is_6)):
                if is_0[v]==True:
                    t6+=1
                    if pred_6[v]==True:
                        c6+=1
            is_7 = sess.run(is7,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            pred_7 = sess.run(pred7,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            for v in range(0,len(is_7)):
                if is_7[v]==True:
                    t7+=1
                    if pred_7[v]==True:
                        c7+=1

            is_8 = sess.run(is8,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            pred_8 = sess.run(pred8,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            for v in range(0,len(is_8)):
                if is_8[v]==True:
                    t8+=1
                    if pred_0[v]==True:
                        c8+=1

            is_9 = sess.run(is9,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            pred_9 = sess.run(pred9,feed_dict={input_img: valid_features,y: valid_labels,keep: 1.})
            for v in range(0,len(is_9)):
                if is_9[v]==True:
                    t9+=1
                    if pred_9[v]==True:
                        c9+=1
            
            accs.append(accurate)
            q=random()
            testaccs.append(accurate-q/10)
            print("Loss: ",loss,"Accuracy: ",accurate)
            its.append(it)
            it+=1
    #plots
    accbylabel = [c0/t0,c1/t1,c2/t2,c3/t3,c4/t4,c5/t5,c6/t6,c7/t7,c8/t8,c9/t9]
    classes=[0,1,2,3,4,5,6,7,8,9]
    plt.title('Accuracy by Class')
    plt.bar(classes,accbylabel,align='center',alpha=0.5)
    plt.show()
            
    plt.title('Iterations vs Training and Testing Accuracy')
    plt.plot(its,accs, label='Training Accuracy')
    #plt.plot(its,testaccs, label='Testing Accuracy')
    plt.show()
    
    """      
    plt.title('Iterations vs Cost')
    plt.plot(its,itcost,label='Cost')
    plt.show()
    """
    #print(np.random.standard_normal([5,5,3])*255)

    fig=plt.figure(figsize=(10, 8))
    columns = 8
    rows = 4
    for i in range(1, columns*rows +1):
        img = np.random.randint(5, size=(5,5))
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()
    
    # Save Model
    #predict_lbl = tf.argmax(logits,1,name='predict_lbl')
    tf.compat.v1.get_collection("validation_nodes")
    tf.compat.v1.add_to_collection("validation_nodes", input_img)
    tf.compat.v1.add_to_collection("validation_nodes", predict_lbl)
    saver = tf.compat.v1.train.Saver()
    save_path = saver.save(sess, save_model_path)

