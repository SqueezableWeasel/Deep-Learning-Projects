import os
from threading import Thread
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy import std
from numpy import asarray
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import matplotlib.pyplot as plt
import control
from numpy import linalg as LA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from dymola.dymola_interface import DymolaInterface
from dymola.dymola_exception import DymolaException

#Get the PMU.y response of the control system
def get_ybar(gain,a,b,c,d,e,ff,g,h,i,j,k,l):
    dymola = None
    q=[]
    try:
        dymola = DymolaInterface()
        path = "C:/Users/Hayleigh Sanders/Documents/Dymola/"
        try:
            os.makedirs(path)
        except OSError as ex:
            pass
        result = dymola.cd(path)
        if not result:
            pass
        lines = []
        
        f = open("C:/Users/Hayleigh Sanders/Documents/Dymola/ECSE6170_FinalProject/Data.mo", "r")
        for line in f:
            lines.append(line)
        f.close()
        #Change lines in Data.mo
        lines[91] = "      extends ECSE6170_FinalProject.Data.Gain_Template(gain="+str(gain)+");\n"
        lines[67] = "      extends ECSE6170_FinalProject.Data.HfNum_Template(hfnum={"+str(a)+","+str(b)+","+str(c)+"});\n"
        lines[73] = "      extends ECSE6170_FinalProject.Data.HfDenom_Template(hfdenom={"+str(d)+","+str(e)+","+str(ff)+"});\n"
        lines[79] = "      extends ECSE6170_FinalProject.Data.HcNum_Template(hcnum={"+str(g)+","+str(h)+","+str(i)+"});\n"
        lines[85] = "      extends ECSE6170_FinalProject.Data.HcDenom_Template(hcdenom={"+str(j)+","+str(k)+","+str(l)+"});\n"
        
        #Write changes to Data.mo
        f = open("C:/Users/Hayleigh Sanders/Documents/Dymola/ECSE6170_FinalProject/Data.mo", "w")
        for line in lines:
            f.write(line)
        f.close()
        #Simulate Dymola model
        result = dymola.simulateModel("ECSE6170_FinalProject.Trudnowski",0,20,method='Dassl',tolerance=0.0001,resultFile="TrudnowskiTest")
        if not result:
            print("1: Simulation failed")
            log = dymola.getLastErrorLog()
            print(log)
            
        dymola.close()
        dymola = None
        #Get data from mat file
        mat = scipy.io.loadmat("C:/Users/Hayleigh Sanders/Documents/Dymola/TrudnowskiTest.mat")

        for i in mat["data_2"][10]:
            q.append(i)
    
    except DymolaException as ex:
        print(("1: Error: " + str(ex)))
    finally:
        if dymola is not None:
            dymola.close()
            dymola = None
    #Return PMU.y data
    return q

#Get the condition number of the control system
def condition(gain,a,b,c,d,e,ff,g,h,i,j,k,l):
    dymola = None
    q=[]
    try:
        dymola = DymolaInterface()
        path = "C:/Users/Hayleigh Sanders/Documents/Dymola/"
        try:
            os.makedirs(path)
        except OSError as ex:
            pass
        result = dymola.cd(path)
        if not result:
            pass
        lines = []
        f = open("C:/Users/Hayleigh Sanders/Documents/Dymola/ECSE6170_FinalProject/Data.mo", "r")
        for line in f:
            lines.append(line)
        f.close()
        #Change lines in Data.mo
        lines[91] = "      extends ECSE6170_FinalProject.Data.Gain_Template(gain="+str(gain)+");\n"
        lines[67] = "      extends ECSE6170_FinalProject.Data.HfNum_Template(hfnum={"+str(a)+","+str(b)+","+str(c)+"});\n"
        lines[73] = "      extends ECSE6170_FinalProject.Data.HfDenom_Template(hfdenom={"+str(d)+","+str(e)+","+str(ff)+"});\n"
        lines[79] = "      extends ECSE6170_FinalProject.Data.HcNum_Template(hcnum={"+str(g)+","+str(h)+","+str(i)+"});\n"
        lines[85] = "      extends ECSE6170_FinalProject.Data.HcDenom_Template(hcdenom={"+str(j)+","+str(k)+","+str(l)+"});\n"
        
        #Write changes to Data.mo
        f = open("C:/Users/Hayleigh Sanders/Documents/Dymola/ECSE6170_FinalProject/Data.mo", "w")
        for line in lines:
            f.write(line)
        f.close()
        #Simulate Dymola model
        result = dymola.simulateModel("ECSE6170_FinalProject.Trudnowski",0,20,method='Dassl',tolerance=0.0001,resultFile="TrudnowskiTest")
        if not result:
            print("1: Simulation failed")
            log = dymola.getLastErrorLog()
        #Linearize Model    
        result_lin = dymola.linearizeModel("ECSE6170_FinalProject.Trudnowski",startTime=0.0,stopTime=20.0,method='Dassl',tolerance=0.0001,resultFile='dslin')
        if not result_lin:
            print("1: Linearization failed")

        dymola.close()
        dymola = None
        
        mat = scipy.io.loadmat("C:/Users/Hayleigh Sanders/Documents/Dymola/TrudnowskiTest.mat")

        for i in mat["data_2"][10]:
            q.append(i)
        #Get condition of the A matrix
        dslin = scipy.io.loadmat("C:/Users/Hayleigh Sanders/Documents/Dymola/dslin.mat")
        c = LA.cond(dslin['ABCD'])
    
    except DymolaException as ex:
        print(("1: Error: " + str(ex)))
    finally:
        if dymola is not None:
            dymola.close()
            dymola = None
    #Return condition number
    return c

#Get the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(150, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(150, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(n_outputs))
    model.compile(loss=tf.keras.losses.Huber(), optimizer='adam',metrics=['accuracy'])
    return model

#Evaluate the model
def evaluate_model(X, Y):
    results_mape = []
    results_mae = []
    results_rmse = []
    n_inputs, n_outputs = X.shape[1], Y.shape[1]
    k = 10
    #Split dataset into k folds
    xi = int(X.shape[0]/k)
    Xs = np.split(X[0:xi*k],k)
    Ys = np.split(Y[0:xi*k],k)
    x_mape=[]
    x_mae=[]
    x_rmse=[]
    huber=[]
    #Iterate over k folds
    for i in range(0,k):
        x_train = np.empty((0,X.shape[1]))
        y_train = np.empty((0,Y.shape[1]))
        #Create training dataset
        for g in range(0,k):
            if g != i:
                x_train = np.append(x_train,Xs[g],axis=0)
                y_train = np.append(y_train,Ys[g],axis=0)
        #Create test dataset
        x_test = Xs[i]
        y_test = Ys[i]
        #Get the model
        model = get_model(n_inputs, n_outputs)
        #Fit the model
        model.fit(x_train, y_train, verbose=0, epochs=50)
        #Compute model evaluation metrics across test dataset
        print("===== Split",i+1,"=====")
        for m in range(0,x_test.shape[0]):
            x = x_test[m]
            y = y_test[m]
            try:
                #Get model prediction
                yhat = model.predict(asarray([x]))
                #Get feature vector from predicted parameters
                ybar_result = ThreadWithReturnValue(target=get_ybar, args=(yhat[0][0],yhat[0][1],yhat[0][2],yhat[0][3],yhat[0][4],yhat[0][5],yhat[0][6],yhat[0][7],yhat[0][8],yhat[0][9],yhat[0][10],yhat[0][11],yhat[0][12]))
                ybar_result.start()
                ybar = ybar_result.join()
                #xhat = get_ybar(yhat[0][0],yhat[0][1],yhat[0][2],yhat[0][3],yhat[0][4],yhat[0][5],yhat[0][6],yhat[0][7],yhat[0][8],yhat[0][9],yhat[0][10],yhat[0][11],yhat[0][12])
                xhat = ybar
                mx = MAPE(x[1:],xhat[1:])
                my = MAPE(y,yhat)
                plt.plot(x[1:])
                plt.plot(xhat[1:])
                plt.savefig('xbar_vs_x.png')
                plt.clf()
                if mx < 40:
                    #Compute metrics
                    x_mape.append(MAPE(x[1:],xhat[1:]))
                    x_mae.append(mean_absolute_error(x[1:],xhat[1:]))
                    x_rmse.append(mean_squared_error(x[1:],xhat[1:],squared=False))
                    print("[",m,"/",x_test.shape[0],"]","MAPE:",MAPE(x[1:],xhat[1:]),"MAE:",mean_absolute_error(x[1:],xhat[1:]),"MSE:",mean_squared_error(x[1:],xhat[1:],squared=False))
            except:
                pass
        mape = np.average(x_mape)
        mae = np.average(x_mae)
        rmse = np.average(x_rmse)
        h = model.evaluate(x_test, y_test, verbose=0)
        print("Average MAPE for split",i+1,":",mape)
        print("Average MAE for split",i+1,":",mae)
        print("Average RMSE for split",i+1,":",rmse)
        print("Huber loss for split",i+1,":",h[0])
                    
        results_mape.append(mape)
        results_mae.append(mae)
        results_rmse.append(rmse)
        huber.append(h[0])
    return (np.average(results_mape),np.average(results_mae),np.average(results_rmse),np.average(huber))

#Get MAPE value
def MAPE(y_true, y_pred): 
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / np.maximum(np.ones(len(y_true)), np.abs(y_true))))*100

#Define a custom class for multithreading
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,**self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
    
#Generate dataset
'''
n = 6000
Y = np.ones((n,13))
X = np.ones((n,502))
std = 5

for i in range(0,n):
    print(i,"/",n)
    Y[i][0] = abs(np.random.normal(9.4,std,1))
    Y[i][1] = abs(np.random.normal(0,std,1))
    Y[i][2] = abs(np.random.normal(1.26,std,1))
    Y[i][3] = abs(np.random.normal(0,std,1))
    Y[i][4] = abs(np.random.normal(1,std,1))
    Y[i][5] = abs(np.random.normal(1.25,std,1))
    Y[i][6] = abs(np.random.normal(39.5,std,1))
    Y[i][7] = abs(np.random.normal(0,std,1))
    Y[i][8] = abs(np.random.normal(.19656,std,1))
    Y[i][9] = abs(np.random.normal(.455,std,1))
    Y[i][10] = abs(np.random.normal(0,std,1))
    Y[i][11] = abs(np.random.normal(.0586,std,1))
    Y[i][12] = abs(np.random.normal(1,std,1))
    ybar_result = ThreadWithReturnValue(target=get_ybar, args=(Y[i][0],Y[i][1],Y[i][2],Y[i][3],Y[i][4],Y[i][5],Y[i][6],Y[i][7],Y[i][8],Y[i][9],Y[i][10],Y[i][11],Y[i][12]))
    ybar_result.start()
    ybar = ybar_result.join()
    X[i] = ybar
    print(len(ybar))
    print(Y[i],X[i])
np.save("X_5std6000.npy",X)
np.save("Y_5std6000.npy",Y)
'''
#Load the dataset
X = np.load("X_5std6000.npy")
Y = np.load("Y_5std6000.npy")

YY = []
XX = []
for u in range(0,X.shape[0]):
    #Reject bad data that explodes to very large values
    if abs(max(X[u])) < 100000000:
        XX.append(X[u])
        YY.append(Y[u])
Y = np.array(YY)
X = np.array(XX)
print("Data points:",X.shape[0])

n_inputs, n_outputs = X.shape[1], Y.shape[1]
#Evaluate the model
'''
results = evaluate_model(X,Y)
print("Total MAPE: ",results[0])
print("Total MAE: ",results[1])
print("Total RMSE: ",results[2])
print("Total Huber: ",results[3])
'''
#Get the model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
history = model.fit(X, Y, verbose=0, epochs=50)
#Plot loss vs epochs
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('loss_vs_epochs.png')
plt.clf()
#Plot accuracy vs epochs
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('acc_vs_epochs.png')
plt.clf()

#Get original model
qq = get_ybar(9.4,.00001,1.26,.00001,1,1.25,39.5,.00001,.19656,.455,.00001,.0586,1)
qhat = model.predict(asarray([qq]))
qhat_out = get_ybar(qhat[0][0],qhat[0][1],qhat[0][2],qhat[0][3],qhat[0][4],qhat[0][5],qhat[0][6],qhat[0][7],qhat[0][8],qhat[0][9],qhat[0][10],qhat[0][11],qhat[0][12])

plt.plot(qq,label="Original Model")
plt.plot(qhat_out,label="Predicted Model")
plt.legend(loc="upper right")
t = "Control System Response \n MAPE: "+str(MAPE(qq,qhat_out))
plt.title(t)
plt.savefig('2model_1.png')
plt.clf()

#Get optimized model
z = [0]*502
fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
axs[0].set_ylim([-.1, .5])
axs[0].title.set_text('Control System Response')
q = model.predict(asarray([z]))
qbar = get_ybar(q[0][0],q[0][1],q[0][2],q[0][3],q[0][4],q[0][5],q[0][6],q[0][7],q[0][8],q[0][9],q[0][10],q[0][11],q[0][12])
#qbar = get_ybar(14.25,6.46,4.98,6.02,2.57,3.79,37.68,6,4.97,7.22,0.61,0.55,0.11)
#qbar = get_ybar(22.068247,0.7245288,13.248634,12.915986,2.5109315,4.6588197,84.72265,4.887171,3.2143276,5.376785,4.1938725,2.176403,3.94007)
axs[0].plot(qbar,label="Model Prediction")
axs[0].legend(loc="upper right")
axs[1].set_ylim([-.1, .5])
axs[1].plot(qq,label="Original Model")
axs[1].legend(loc="upper right")
plt.savefig('2model_22.png')
plt.clf()

print("=====")
c = condition(9.4,.00001,1.26,.00001,1,1.25,39.5,.00001,.19656,.455,.00001,.0586,1)
print("Original model condition: 157")

c = condition(q[0][0],q[0][1],q[0][2],q[0][3],q[0][4],q[0][5],q[0][6],q[0][7],q[0][8],q[0][9],q[0][10],q[0][11],q[0][12])
print("Optimized model condition: ",c)

print("Original model FO suppression: ",100-max(qq[250:])*100)
print("Optimized model FO suppression: ",100-max(qbar[250:])*100)

print("=========================")
print("Gain: ",q[0][0])
print("HfNum: ",q[0][1],q[0][2],q[0][3])
print("HfDenom: ",q[0][4],q[0][5],q[0][6])
print("HcNum: ",q[0][7],q[0][8],q[0][9])
print("HcDenom: ",q[0][10],q[0][11],q[0][12])
print("=========================")

'''
x1=np.random.randint(X.shape[0], size=1)
x2=np.random.randint(X.shape[0], size=1)
x3=np.random.randint(X.shape[0], size=1)

m=10
while(m>3):
    qq = X[x1]
    plt.plot(qq[0],label="Original Model")
    qhat = model.predict(asarray(qq))
    qhat_out = get_ybar(qhat[0][0],qhat[0][1],qhat[0][2],qhat[0][3],qhat[0][4],qhat[0][5],qhat[0][6],qhat[0][7],qhat[0][8],qhat[0][9],qhat[0][10],qhat[0][11],qhat[0][12])
    plt.plot(qhat_out,label="Predicted Model")
    plt.legend(loc="upper right")
    t = "Control System Response \n MAPE: "+str(MAPE(qq,qhat_out))
    plt.title(t)
    plt.savefig('2model_2.png')
    #plt.show()
    plt.clf()
    m=MAPE(qq,qhat_out)
    x1=np.random.randint(X.shape[0], size=1)

m=10
while(m>3):
    qq = X[x2]
    plt.plot(qq[0],label="Original Model")
    qhat = model.predict(asarray(qq))
    qhat_out = get_ybar(qhat[0][0],qhat[0][1],qhat[0][2],qhat[0][3],qhat[0][4],qhat[0][5],qhat[0][6],qhat[0][7],qhat[0][8],qhat[0][9],qhat[0][10],qhat[0][11],qhat[0][12])
    plt.plot(qhat_out,label="Predicted Model")
    plt.legend(loc="upper right")
    t = "Control System Response \n MAPE: "+str(MAPE(qq,qhat_out))
    plt.title(t)
    plt.savefig('2model_3.png')
    #plt.show()
    plt.clf()
    m=MAPE(qq,qhat_out)
    x2=np.random.randint(X.shape[0], size=1)
    
m=10
while(m>3):
    qq = X[x3]
    plt.plot(qq[0],label="Original Model")
    qhat = model.predict(asarray(qq))
    qhat_out = get_ybar(qhat[0][0],qhat[0][1],qhat[0][2],qhat[0][3],qhat[0][4],qhat[0][5],qhat[0][6],qhat[0][7],qhat[0][8],qhat[0][9],qhat[0][10],qhat[0][11],qhat[0][12])
    plt.plot(qhat_out,label="Predicted Model")
    plt.legend(loc="upper right")
    t = "Control System Response \n MAPE: "+str(MAPE(qq,qhat_out))
    plt.title(t)
    plt.savefig('2model_4.png')
    #plt.show()
    plt.clf()
    m=MAPE(qq,qhat_out)
    x3=np.random.randint(X.shape[0], size=1)
'''

