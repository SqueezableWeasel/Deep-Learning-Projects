###
#Deep Learning Programming assignment 1
#Multiclass Logistic Regression 
#Hayleigh Sanders 661195735
###

import tensorflow as tf
import numpy
import matplotlib
from PIL import Image
import numpy as np
from numpy import array
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pickle

#populate test label list
testlabel=[]
trainlabel=[]
testdata = []
traindata = []
f=open(r"C:\Users\Hayleigh Sanders\Documents\ECSE4850 Deep Learning\ProgrammingAssignment1\data_prog2\labels\test_label.txt", "r")
f1 = f.readlines()
for x in f1:
    testlabel.append(int(x))
f.close()

#populate training label list
f=open(r"C:\Users\Hayleigh Sanders\Documents\ECSE4850 Deep Learning\ProgrammingAssignment1\data_prog2\labels\train_label.txt", "r")
f1 = f.readlines()
for x in f1:
    if int(x) == 1:
        trainlabel.append([1,0,0,0,0])
    if int(x) == 2:
        trainlabel.append([0,1,0,0,0])
    if int(x) == 3:
        trainlabel.append([0,0,1,0,0])
    if int(x) == 4:
        trainlabel.append([0,0,0,1,0])
    if int(x) == 5:
        trainlabel.append([0,0,0,0,1])
f.close()

#populate test data list
for i in range(1,4983):
    i = str(i)
    i = "000"+i
    i = i[-4:]
    imname=r"C:\\Users\\Hayleigh Sanders\\Documents\\ECSE4850 Deep Learning\\ProgrammingAssignment1\\data_prog2\\test_data\\"+i+".jpg"
    img = Image.open(imname).convert('L') #convert to grayscale (pixels 0 to 255)
    img = array(img)
    img = img/255
    img=img.flatten()
    img = np.append(img,[1],axis=0)
    testdata.append(img)

#populate training data list
for i in range(1,25113):
    i = str(i)
    i = "0000"+i
    i = i[-5:]
    imname=r"C:\\Users\\Hayleigh Sanders\\Documents\\ECSE4850 Deep Learning\\ProgrammingAssignment1\\data_prog2\\train_data\\"+i+".jpg"
    img = Image.open(imname).convert('L') #convert to grayscale (pixels 0 to 255)
    img = array(img)
    img=img/255
    img=img.flatten()
    img = np.append(img,[1],axis=0)
    traindata.append(img)

#print(traindata[0])
#print(trainlabel[0])

ninputs = 784
noutputs = 5
nexamples = 35112

Theta = np.random.normal(0,1,(785,5)) #populate theta with random Gaussian values
#Theta = np.zeros((785,5))

def softmax(z): #define softmax function
    z=z/1000
    return np.exp(z) / float(sum(np.exp(z)))

#print(len(traindata))
#print(np.dot(Theta.T, traindata[0]))
#print(softmax(Theta.T @ traindata[0]))
#print(sum(softmax(Theta.T @ traindata[0])))
theta_new = np.empty((0,785)) 
rate=.05
epsilon=0
#calculate the gradient for each theta k
for k in range(0,5):
    theta_k = Theta.T[k]
    gradient = 0
    #repeat for n iterations
    for i in range(0,100):
        #print("thetak: ",theta_k.shape)
        for m in range(0,len(traindata)):
            tmk = trainlabel[m][k] #get value of t[m][k] (0 or 1)
            #print("tmk: ",tmk)
            #print("traindata",len(traindata))
            td = traindata[m] #training data vector
            #print(td.T @ Theta)
            sm = softmax(td @ Theta) #compute softmax
            #print("softmax: ",sm)
            z = tmk - sm[k]
            z = z * traindata[m]
            gradient+=z #sum from 0 to M
            #print(gradient.shape)
        gradient=gradient*-1 #multiply total by -1
        theta_k = theta_k - (rate*gradient) #update theta

    #print(theta_k)
    theta_new = np.append(theta_new,[theta_k],axis=0) #put new thetas in a list
print(theta_new.shape)
#print(theta_new @ testdata[0])
correct=0
c1 = 0
c1t = 0
c2 = 0
c2t = 0
c3 = 0
c3t = 0
c4 = 0
c4t = 0
c5 = 0
c5t = 0
#compute error
for m in range(0,len(testdata)):
    guess = np.argmax(softmax(theta_new @ testdata[m]))+1
    if guess == testlabel[m]:
        correct+=1
        if guess == 1:
            c1+=1
        if guess == 2:
            c2+=1
        if guess == 3:
            c3+=1
        if guess == 4:
            c4+=1
        if guess == 5:
            c5+=1
    if testlabel[m] == 1:
        c1t+=1
    if testlabel[m] == 2:
        c2t+=1
    if testlabel[m] == 3:
        c3t+=1
    if testlabel[m] == 4:
        c4t+=1
    if testlabel[m] == 5:
        #print(guess,testlabel[m])
        c5t+=1
        
print("Total test accuracy: ",correct/len(testdata))
print("Class 1 accuracy: ", c1/c1t)
print("Class 2 accuracy: ", c2/c2t)
print("Class 3 accuracy: ", c3/c3t)
print("Class 4 accuracy: ", c4/c4t)
print("Class 5 accuracy: ", c5/c5t)

acc = [100*(1-c1/c1t),100*(1-c2/c2t),100*(1-c3/c3t),100*(1-c4/c4t),100*(1-c5/c5t)]
classes = [1,2,3,4,5]
testacc = [100-13.368,100-73.665,100-73.906,100-74.568]
trainacc = [100-19.118,100-72.065,100-71.579,100-72.2]
iterations = [0,10,50,100]

tgraph = []
for c in range(0,784):
    #print(theta_new[1][c])
    tgraph.append(theta_new[4][c])
tgraph=np.asarray(tgraph)
img = tgraph.reshape(28,28)
plt.imshow(img)
plt.colorbar()
plt.show()

#plot data
title = "Test Data - Iterations vs. Error"
plt.plot(iterations,testacc)
plt.ylabel('Percent Error')
plt.xlabel('Number of Iterations')
plt.title(title)
plt.show()

title = "Training Data - Iterations vs. Error"
plt.plot(iterations,trainacc)
plt.ylabel('Percent Error')
plt.xlabel('Number of Iterations')
plt.title(title)
plt.show()

title = "Error By Class"
plt.bar(classes,acc,align='center',alpha=.5)
plt.ylabel('Percent Error')
plt.xlabel('Class')
plt.title(title)
plt.show()

print(theta_new.shape) #make sure W is in the right shape!
filehandler = open("multiclass_parameters.txt","wb")
pickle.dump(theta_new.T, filehandler)
filehandler.close()










    


    
