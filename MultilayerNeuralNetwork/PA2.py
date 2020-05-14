import tensorflow as tf
import numpy as np
from PIL import Image
from numpy import array
import random
import math
import matplotlib.pyplot as plt
import pickle

#Programming Assignment 2 - Hayleigh Sanders RIN# 661195735

#populate test label list
testlabel=[]
trainlabel=[]
testdata = []
traindata = []
f=open(r"C:\Users\Hayleigh Sanders\Documents\ECSE4850 Deep Learning\ProgrammingAssignment2\data_prog2Spring18\labels\test_label.txt", "r")
f1 = f.readlines()
for x in f1:
    if int(x) == 0:
        testlabel.append([1,0,0,0,0,0,0,0,0,0])
    if int(x) == 1:
        testlabel.append([0,1,0,0,0,0,0,0,0,0])
    if int(x) == 2:
        testlabel.append([0,0,1,0,0,0,0,0,0,0])
    if int(x) == 3:
        testlabel.append([0,0,0,1,0,0,0,0,0,0])
    if int(x) == 4:
        testlabel.append([0,0,0,0,1,0,0,0,0,0])
    if int(x) == 5:
        testlabel.append([0,0,0,0,0,1,0,0,0,0])
    if int(x) == 6:
        testlabel.append([0,0,0,0,0,0,1,0,0,0])
    if int(x) == 7:
        testlabel.append([0,0,0,0,0,0,0,1,0,0])
    if int(x) == 8:
        testlabel.append([0,0,0,0,0,0,0,0,1,0])
    if int(x) == 9:
        testlabel.append([0,0,0,0,0,0,0,0,0,1])
f.close()

#populate training label list using one of k notation
f=open(r"C:\Users\Hayleigh Sanders\Documents\ECSE4850 Deep Learning\ProgrammingAssignment2\data_prog2Spring18\labels\train_label.txt", "r")
f1 = f.readlines()
for x in f1:
    if int(x) == 0:
        trainlabel.append([1,0,0,0,0,0,0,0,0,0])
    if int(x) == 1:
        trainlabel.append([0,1,0,0,0,0,0,0,0,0])
    if int(x) == 2:
        trainlabel.append([0,0,1,0,0,0,0,0,0,0])
    if int(x) == 3:
        trainlabel.append([0,0,0,1,0,0,0,0,0,0])
    if int(x) == 4:
        trainlabel.append([0,0,0,0,1,0,0,0,0,0])
    if int(x) == 5:
        trainlabel.append([0,0,0,0,0,1,0,0,0,0])
    if int(x) == 6:
        trainlabel.append([0,0,0,0,0,0,1,0,0,0])
    if int(x) == 7:
        trainlabel.append([0,0,0,0,0,0,0,1,0,0])
    if int(x) == 8:
        trainlabel.append([0,0,0,0,0,0,0,0,1,0])
    if int(x) == 9:
        trainlabel.append([0,0,0,0,0,0,0,0,0,1])
f.close()

#populate test data list
for i in range(1,5001):
    i = str(i)
    i = "0000"+i
    i = i[-5:]
    imname=r"C:\\Users\\Hayleigh Sanders\\Documents\\ECSE4850 Deep Learning\\ProgrammingAssignment2\\data_prog2Spring18\\test_data\\"+i+".jpg"
    img = Image.open(imname).convert('L') #convert to grayscale (pixels 0 to 255)
    img = array(img)
    img = img/255
    img=img.flatten()
    #img = np.append(img,[1],axis=0)
    testdata.append(img)

#populate training data list
for i in range(1,50001):
    i = str(i)
    i = "0000"+i
    i = i[-5:]
    imname=r"C:\\Users\\Hayleigh Sanders\\Documents\\ECSE4850 Deep Learning\\ProgrammingAssignment2\\data_prog2Spring18\\train_data\\"+i+".jpg"
    img = Image.open(imname).convert('L') #convert to grayscale (pixels 0 to 255)
    img = array(img)
    img=img/255
    img=img.flatten()
    #img = np.append(img,[1],axis=0)
    traindata.append(img)

#s = random.sample(range(10000),50) #get a subset of random non-repeating samples from the data set
#print(s)
trainlabel = np.asarray(trainlabel)
traindata = np.asarray(traindata)
testlabel=np.array(testlabel)
testdata = np.array(testdata)

traindata = traindata.T
trainlabel = trainlabel.T
testdata=testdata.T
testlabel=testlabel.T
rand_indexes = np.random.permutation(50000)
traindata = traindata[:, rand_indexes]
trainlabel = trainlabel[:, rand_indexes]

#hyperparameters
stepsize = 0.5
epochs = 50
xnodes = 784
hidden_nodes = 100
minibatch = 50

# initialize weights to small random values
weights = {"W1": np.random.randn(hidden_nodes, xnodes)*np.sqrt(1/xnodes),
          "w01": np.zeros((hidden_nodes, 1))*np.sqrt(1/xnodes),        
          "W2": np.random.randn(10, hidden_nodes)*np.sqrt(1/hidden_nodes),
          "w02": np.zeros((10, 1))*np.sqrt(1/hidden_nodes),
           "W3": np.random.randn(10, hidden_nodes)*np.sqrt(1/hidden_nodes),
          "w03": np.zeros((10, 1))*np.sqrt(1/hidden_nodes)}

#ReLu function
def ReLu(z):
    #print(np.maximum(0,z))
    return np.maximum(0,z)

#Derivative of the relu function
def dReLu(z):
    for x in range(0,len(z)):
        for y in range(0,len(z[x])):
            if z[x][y]<0:
                z[x][y]=0
            if z[x][y]>0:
                z[x][y]=1
    return z

#Softmax function
def softmax(z):
    result = np.exp(z) / np.sum(np.exp(z), axis=0)
    return result

#loss function
def loss(y, y_hat):
    ltot = np.sum(np.multiply(y, np.log(y_hat)))
    m = y.shape[1]
    loss = -(1./m) * ltot
    return loss

#forward propagation
def forward_propagation(X, weights):
    temp = {}
    #Use ReLu for the hidden layers, and softmax for the output layer
    temp["Z1"] = np.matmul(weights["W1"], X) + weights["w01"]
    temp["H1"] = ReLu(temp["Z1"])
    temp["Z2"] = np.matmul(weights["W2"], X) + weights["w02"]
    temp["H2"] = ReLu(temp["Z2"])
    temp["Z3"] = np.matmul(weights["W3"], temp["H2"]) + weights["w03"]
    temp["H3"] = softmax(temp["Z3"])
    return temp

#Backward propagation
def backward_propagation(X, Y, weights, temp, stoch):
    #output layer error
    dZ3 = temp["H3"] - Y
    
    #output layer gradients
    dW3 = (1/stoch) * np.matmul(dZ3, temp["H2"].T)
    dw03 = (1/stoch) * np.sum(dZ3, axis=1, keepdims=True)
    
    #second layer back propagation
    dH2 = np.matmul(weights["W3"].T, dZ3)
    dZ2 = dH2 * dReLu(temp["Z2"])
    
    #second layer gradients
    dW2 = (1/stoch) * np.matmul(dZ2, temp["H1"].T)
    dw02 = (1/stoch) * np.sum(dZ2, axis=1, keepdims=True)

    #first layer back propagation
    dH1 = np.matmul(weights["W2"].T, dZ2)
    dZ1 = dH1 * dReLu(temp["Z1"])
    
    #first layer gradients
    dW1 = (1/stoch) * np.matmul(dZ1, X.T)
    dw01 = (1/stoch) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dW1": dW1, "dw01": dw01, "dW2": dW2, "dw02": dw02, "dW3": dW3, "dw03": dw03}
    #print(q)
    return gradients

iters=[]
trainacc=[]
testacc=[]

for i in range(epochs):
    #Get a batch of training data
    rands = np.random.permutation(traindata.shape[1])
    randtraindata = traindata[:, rands]
    randtrainlabel = trainlabel[:, rands]
    num_batch = int(50000/minibatch)   
    #Perform Stochastic GD over 50 samples
    for j in range(num_batch): 
        # get mini-batch
        i1 = j * minibatch
        i2 = min(i1 + minibatch, traindata.shape[1] - 1)
        X = randtraindata[:, i1:i2]
        Y = randtrainlabel[:, i1:i2]
        m_batch = i2 - i1
        #Get values from forward propagation
        temp = forward_propagation(X, weights)
        #Get gradients from backward propagation
        gradients = backward_propagation(X, Y, weights, temp, m_batch)
        #Update weights
        weights["W1"] = weights["W1"] - stepsize * gradients["dW1"]
        weights["w01"] = weights["w01"] - stepsize * gradients["dw01"]
        weights["W2"] = weights["W2"] - stepsize * gradients["dW2"]
        weights["w02"] = weights["w02"] - stepsize * gradients["dw02"]
        weights["W3"] = weights["W3"] - stepsize * gradients["dW3"]
        weights["w03"] = weights["w03"] - stepsize * gradients["dw03"]
        #Decrease step size
        stepsize=stepsize-.000001

    #Perform forward propagation again to get the final Y (Training data)
    temp = forward_propagation(traindata, weights)
    #Find the training data loss
    train_loss = loss(trainlabel, temp["H3"])
    acc=0
    count=0
    print("-----")
    print("Epoch #",i+1)
    classacc_tr=[0,0,0,0,0,0,0,0,0,0]
    classtot_tr=[0,0,0,0,0,0,0,0,0,0]
    #Find training data accuracy
    for y in temp["H3"].T:
        yt = trainlabel.T
        guess = np.where(y==(max(y)))
        #print(yt[count])
        if(yt[count][guess]==1):
            acc+=1
            classacc_tr=classacc_tr+yt[count]
        classtot_tr=classtot_tr+yt[count]
        count+=1
    print("Train Accuracy: ",acc/50000)
    trainacc.append(1-(acc/50000))
    #Perform forward propagation again to get the final Y (Test data)
    temp = forward_propagation(testdata, weights)
    #find test data loss
    test_loss = loss(testlabel, temp["H3"])
    acc=0
    count=0
    classacc_te=[0,0,0,0,0,0,0,0,0,0]
    classtot_te=[0,0,0,0,0,0,0,0,0,0]
    #print(temp["Z2"].shape)
    for y in temp["H3"].T:
        yt = testlabel.T
        guess = np.where(y==(max(y)))
        if(yt[count][guess]==1):
            acc+=1
            classacc_te=classacc_te+yt[count]
        classtot_te=classtot_te+yt[count]
        count+=1
    print("Test Accuracy: ",acc/5000)
    testacc.append(1-(acc/5000))
    print("training loss: ",train_loss)
    print("testing loss: ",test_loss)
    iters.append(i)

#Dump weights
Theta = [weights["W1"],weights["w01"],weights["W2"],weights["w02"],weights["W3"],weights["w03"]]
filehandler = open("nn_parameters.txt","wb")
pickle.dump(Theta, filehandler, protocol=2)
filehandler.close()

#plot data
title = "Test and Training Data - Iterations vs. Error"
plt.plot(iters,testacc)
plt.plot(iters,trainacc)
plt.ylabel('Percent Error')
plt.xlabel('Number of Iterations')
plt.title(title)
plt.show()

classes = [0,1,2,3,4,5,6,7,8,9]
ca = classacc_te/classtot_te
title = "Error By Class"
plt.bar(classes,1-ca,align='center',alpha=.5)
plt.ylabel('Percent Error')
plt.xlabel('Class')
plt.title(title)
plt.show()
