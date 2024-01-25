from numpy import mean
from numpy import std
from numpy import asarray
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#Get power flow results for the noiseless case
def get_yhat(a,b,c):
    zl_12 = .02*1j
    zl_13 = .01*1j
    zl_32 = .04*1j

    y12 = a
    y23 = b
    y13 = c

    S1 = 1
    S2 = -1*(500 + 350*1j)/100
    S3 = -1*(400 + 310*1j)/100

    V1=1
    V2=0
    V3=0

    v20 = 1
    v30 = 1

    Ybus = [[y12+y13, -1*y12, -1*y13],[-1*y12, y12+y23, -1*y23],[-1*y13, -1*y23, y13+y23]]

    stop_bit=1
    thresh=10**-9
    counter=1
    while (stop_bit):
        v2 = (((S2.real - S2.imag*1j)/(np.conj(v20)))+y12*V1+y23*v30)/(y12+y23)
        v3 = (((S3.real - S3.imag*1j)/(np.conj(v30)))+y13*V1+y23*v30)/(y13+y23)
        if abs(v2-v20)<thresh and abs(v3-v30)<thresh:
            V2=v2
            V3=v3
            stop_bit=0
        if counter > 1000: #Stop after 1000 iterations
            stop_bit=0
        v20=v2
        v30=v3
        counter=counter+1

    Psb = np.conj(V1)*(V1*(y12+y13)-(y12*V2+y13*V3))

    i12 = y12*(V1-V2)
    i21 = -1*i12
    i13 = y13*(V1-V3)
    i31 = -1*i13
    i23 = y23*(V2-V3)
    i32 = -1*i23

    S12 = V1*np.conj(i12)
    S21 = V2*np.conj(i21)
    S13 = V1*np.conj(i13)
    S31 = V3*np.conj(i31)
    S23 = V2*np.conj(i23)
    S32 = V3*np.conj(i32)

    SL12 = S12+S21
    SL13 = S13+S31
    SL23 = S23+S32

    return [V1.real,V1.imag,V2.real,V2.imag,V3.real,V3.imag,Psb.real,Psb.imag,i12.real,i12.imag,i21.real,i21.imag,i13.real,i13.imag,i31.real,i31.imag,i23.real,i23.imag,i32.real,i32.imag]

#Get power flow results for the noisy case
def get_yhat_noisy(a,b,c):
    zl_12 = .02*1j
    zl_13 = .01*1j
    zl_32 = .04*1j

    y12 = a
    y23 = b
    y13 = c

    S1 = 1
    S2 = -1*(500 + 350*1j)/100
    S3 = -1*(400 + 310*1j)/100

    V1=1
    V2=0
    V3=0

    v20 = 1
    v30 = 1

    Ybus = [[y12+y13, -1*y12, -1*y13],[-1*y12, y12+y23, -1*y23],[-1*y13, -1*y23, y13+y23]]

    stop_bit=1
    thresh=10**-9
    counter=1
    while (stop_bit):
        v2 = (((S2.real - S2.imag*1j)/(np.conj(v20)))+y12*V1+y23*v30)/(y12+y23)
        v3 = (((S3.real - S3.imag*1j)/(np.conj(v30)))+y13*V1+y23*v30)/(y13+y23)
        if abs(v2-v20)<thresh and abs(v3-v30)<thresh:
            V2=v2
            V3=v3
            stop_bit=0
        if counter > 1000: #Stop after 1000 iterations
            stop_bit=0
        v20=v2
        v30=v3
        counter=counter+1

    Psb = np.conj(V1)*(V1*(y12+y13)-(y12*V2+y13*V3))

    i12 = y12*(V1-V2)
    i21 = -1*i12
    i13 = y13*(V1-V3)
    i31 = -1*i13
    i23 = y23*(V2-V3)
    i32 = -1*i23

    S12 = V1*np.conj(i12)
    S21 = V2*np.conj(i21)
    S13 = V1*np.conj(i13)
    S31 = V3*np.conj(i31)
    S23 = V2*np.conj(i23)
    S32 = V3*np.conj(i32)

    SL12 = S12+S21
    SL13 = S13+S31
    SL23 = S23+S32

    #Generate noisy values
    V1re = np.random.uniform(low=V1.real-V1.real*.15, high=V1.real+V1.real*.15)
    V1im = np.random.uniform(low=V1.imag-V1.imag*.15, high=V1.imag+V1.imag*.15)
    V2re = np.random.uniform(low=V2.real-V2.real*.15, high=V2.real+V2.real*.15)
    V2im = np.random.uniform(low=V2.imag-V2.imag*.15, high=V2.imag+V2.imag*.15)
    V3re = np.random.uniform(low=V3.real-V3.real*.15, high=V3.real+V3.real*.15)
    V3im = np.random.uniform(low=V3.imag-V3.imag*.15, high=V3.imag+V3.imag*.15)

    Psbre = np.random.uniform(low=Psb.real-Psb.real*.15, high=Psb.real+Psb.real*.15)
    Psbim = np.random.uniform(low=Psb.imag-Psb.imag*.15, high=Psb.imag+Psb.imag*.15)

    i12re = np.random.uniform(low=i12.real-i12.real*.15, high=i12.real+i12.real*.15)
    i12im = np.random.uniform(low=i12.imag-i12.imag*.15, high=i12.imag+i12.imag*.15)
    i21re = np.random.uniform(low=i21.real-i21.real*.15, high=i21.real+i21.real*.15)
    i21im = np.random.uniform(low=i21.imag-i21.imag*.15, high=i21.imag+i21.imag*.15)
    i13re = np.random.uniform(low=i13.real-i13.real*.15, high=i13.real+i13.real*.15)
    i13im = np.random.uniform(low=i13.imag-i13.imag*.15, high=i13.imag+i13.imag*.15)
    i31re = np.random.uniform(low=i31.real-i31.real*.15, high=i31.real+i31.real*.15)
    i31im = np.random.uniform(low=i31.imag-i31.imag*.15, high=i31.imag+i31.imag*.15)
    i23re = np.random.uniform(low=i23.real-i23.real*.15, high=i23.real+i23.real*.15)
    i23im = np.random.uniform(low=i23.imag-i23.imag*.15, high=i23.imag+i23.imag*.15)
    i32re = np.random.uniform(low=i32.real-i32.real*.15, high=i32.real+i32.real*.15)
    i32im = np.random.uniform(low=i32.imag-i32.imag*.15, high=i32.imag+i32.imag*.15)

    return [V1re,V1im,V2re,V2im,V3re,V3im,Psbre,Psbim,i12re,i12im,i21re,i21im,i13re,i13im,i31re,i31im,i23re,i23im,i32re,i32im]

#Get model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(250, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(250, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(250, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(250, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
    return model

#Get MAPE value
def MAPE(y_true, y_pred): 
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / np.maximum(np.ones(len(y_true)), np.abs(y_true))))*100

#Evaluate model
def evaluate_model(X, Y):
    results_mape = []
    results_rmse = []
    results_ybus_mape = np.zeros((9,1))
    results_ybus_rmse = np.zeros((9,1))
    results_xmape=np.zeros((X.shape[1],1))
    results_xrmse=np.zeros((X.shape[1],1))
    n_inputs, n_outputs = X.shape[1], Y.shape[1]
    k = 10
    #split dataset into k folds
    xi = int(X.shape[0]/k)
    Xs = np.split(X[0:xi*k],k)
    Ys = np.split(Y[0:xi*k],k)
    x_mape=np.zeros((X.shape[1],1))
    x_rmse=np.zeros((X.shape[1],1))
    ybus_mape=np.zeros((9,1))
    ybus_rmse=np.zeros((9,1))
    cy_mape=np.zeros((3,1))
    cy_rmse=np.zeros((3,1))
    #Iterate over k folds
    for i in range(0,k):
        y_mape=[]
        y_rmse=[]
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
        #get the model
        model = get_model(n_inputs, n_outputs)
        #fit the model
        history = model.fit(x_train, y_train, verbose=0, epochs=400)
        #plot loss vs epochs
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig('loss_vs_epochs.png')
        plt.clf()
        #compute model evaluation metrics across test dataset
        print("===== Split",i+1,"=====")
        for m in range(0,x_test.shape[0]):
            x = x_test[m]
            y = y_test[m]
            #Get model prediction
            yhat = model.predict(asarray([x]))
            #Get feature vector from predicted admittance parameters
            xhat = get_yhat(yhat[0][0]+yhat[0][1]*1j,yhat[0][2]+yhat[0][3]*1j,yhat[0][4]+yhat[0][5]*1j)
            #Get Y, Yhat, Ybus and Ybus-hat in complex form
            y1 = y[0]+y[1]*1j
            y2 = y[2]+y[3]*1j
            y3 = y[4]+y[5]*1j
            y1_hat = yhat[0][0]+yhat[0][1]*1j
            y2_hat = yhat[0][2]+yhat[0][3]*1j
            y3_hat = yhat[0][4]+yhat[0][5]*1j
            ybus_test = [y1+y3,-1*y1,-1*y3,
                         -1*y1,y1+y2,-1*y2,
                         -1*y3,-1*y2,y3+y2]
            ybus_hat = [y1_hat+y3_hat,-1*y1_hat,-1*y3_hat,
                        -1*y1_hat,y1_hat+y2_hat,-1*y2_hat,
                        -1*y3_hat,-1*y2_hat,y3_hat+y2_hat]
            cy = [y1,y2,y3]
            cyhat = [y1_hat,y2_hat,y3_hat]
            y_mape.append(MAPE(cy,cyhat))
            y_rmse.append(mean_squared_error(y,yhat[0],squared=False))
            #Get Ybus results
            for d in range(0,9):
                results_ybus_mape[d] += MAPE([ybus_test[d]],[ybus_hat[d]])
                results_ybus_rmse[d] += (abs(ybus_test[d]-ybus_hat[d]))**2
            #Get Feature Results
            for u in range(2,len(x)):
                results_xmape[u] += MAPE([x[u]],[xhat[u]])
                results_xrmse[u] += (abs(x[u]-xhat[u]))**2
            #Get general Y results
            for s in range(0,3):
                cy_mape[s] += MAPE([cy[s]],[cyhat[s]])
                cy_rmse[s] += (abs(cy[s]-cyhat[s]))**2
            print("[",m,"/",x_test.shape[0],"]","MAPE:",MAPE(cy,cyhat),"RMSE:",mean_squared_error(y,yhat[0],squared=False))
        mape = np.average(y_mape)
        rmse = np.average(y_rmse)

        print("Average MAPE for split",i+1,":",mape)
        print("Average RMSE for split",i+1,":",rmse)
                    
        results_mape.append(mape)
        results_rmse.append(rmse)
    #Get average over total number of iterations
    for p in range(0,Y.shape[1]):
        results_ybus_mape[p] = results_ybus_mape[p]/(Xs[0].shape[0]*k)
        results_ybus_rmse[p] = np.sqrt(results_ybus_rmse[p]/(Xs[0].shape[0]*k))
    for q in range(2,X.shape[1]):
        results_xmape[q] = results_xmape[q]/(Xs[0].shape[0]*k)
        results_xrmse[q] = np.sqrt(results_xrmse[q]/(Xs[0].shape[0]*k))
    cy_mape = cy_mape/(Xs[0].shape[0]*k)
    cy_rmse = np.sqrt(cy_rmse/(Xs[0].shape[0]*k))
    y1 = cy_mape[0]
    y2 = cy_mape[1]
    y3 = cy_mape[2]
    results_ybus_mape = [(y1+y3)/2,y1,y3,y1,(y1+y2)/2,y2,y3,y2,(y3+y2)/2]
    y1 = cy_rmse[0]
    y2 = cy_rmse[1]
    y3 = cy_rmse[2]
    results_ybus_rmse = [(y1+y3)/2,y1,y3,y1,(y1+y2)/2,y2,y3,y2,(y3+y2)/2]
    return (np.average(results_mape),np.average(results_rmse),results_ybus_mape,results_ybus_rmse,results_xmape,results_xrmse)

#Generate training data
n = 6000

zl_12 = .02*1j
zl_13 = .01*1j
zl_32 = .04*1j

y12 = 1/zl_12
y13 = 1/zl_13
y23 = 1/zl_32

Y1=np.zeros((n,1)).astype(complex)
Y2=np.zeros((n,1)).astype(complex)
Y3=np.zeros((n,1)).astype(complex)
X = np.zeros((n,20))
X_noisy = np.zeros((n,20))
Y = np.zeros((n,6))

#Compute Y values along normal distribution
Y1re = np.random.normal(y12.real,.1,n)
Y1im = np.random.normal(y12.imag,.1,n)
Y2re = np.random.normal(y23.real,.1,n)
Y2im = np.random.normal(y23.imag,.1,n)
Y3re = np.random.normal(y13.real,.1,n)
Y3im = np.random.normal(y13.imag,.1,n)

#Form Y, X and X_noisy
for k in range(0,n):
    Y1[k] = Y1re[k] + Y1im[k]*1j
    Y2[k] = Y2re[k] + Y2im[k]*1j
    Y3[k] = Y3re[k] + Y3im[k]*1j
    Y[k] = (Y1re[k],Y1im[k],Y2re[k],Y2im[k],Y3re[k],Y3im[k])
    X[k] = get_yhat(Y1[k],Y2[k],Y3[k])
    X_noisy[k] = get_yhat_noisy(Y1[k],Y2[k],Y3[k])

#Get model evaluation results for noiseless case
results = evaluate_model(X, Y)
print("Noiseless Case - Total MAPE: ",results[0])
print("Noiseless Case - Total RMSE: ",results[1])
ybus_mape = results[2]
ybus_rmse = results[3]
x_mape = results[4]
x_rmse = results[5]
print("Noiseless Case - Ybus MAPE:",ybus_mape)
print("Noiseless Case - Ybus RMSE:",ybus_rmse)
print("Noiseless Case - X MAPE:",x_mape)
print("Noiseless Case - X RMSE:",x_rmse)

#Scale the results because matplotlib is stupid and won't plot float values for some reason
s = 100000000
s2 = 1000000000000
ybus_mape_scaled = []
ybus_rmse_scaled = []
for i in range(0,len(ybus_mape)):
    ybus_mape_scaled.append(int(ybus_mape[i]*s))
    ybus_rmse_scaled.append(int(ybus_rmse[i]*s))
x_mape_scaled = []
x_rmse_scaled = []
for i in range(0,len(x_mape)):
    x_mape_scaled.append(int(x_mape[i]*s))
    x_rmse_scaled.append(int(x_rmse[i]*s2))

#plot results
labels = ['V1.real','V1.imag','V2.real','V2.imag','V3.real','V3.imag','Psb.real','Psb.imag','i12.real','i12.imag','i21.real','i21.imag','i13.real','i13.imag','i31.real','i31.imag','i23.real','i23.imag','i32.real','i32.imag']
plt.ylim(0, max(x_mape_scaled)+s)
plt.axhline(y=s,linewidth=1, color='red', ls='dotted')
plt.bar(labels, x_mape_scaled,bottom=500)
plt.xticks(rotation='vertical')
plt.title("MAPE for Estimated Features")
plt.xlabel("Parameters")
plt.ylabel("MAPE (%)")
plt.savefig('MAPE_X.png',bbox_inches="tight")
plt.clf()

labels = ['V1.real','V1.imag','V2.real','V2.imag','V3.real','V3.imag','Psb.real','Psb.imag','i12.real','i12.imag','i21.real','i21.imag','i13.real','i13.imag','i31.real','i31.imag','i23.real','i23.imag','i32.real','i32.imag']
plt.ylim(0, max(x_rmse_scaled)+s2/100)
plt.bar(labels, x_rmse_scaled,bottom=500)
plt.xticks(rotation='vertical')
plt.title("RMSE for Estimated Features")
plt.xlabel("Parameters")
plt.ylabel("RMSE")
plt.savefig('RMSE_X.png',bbox_inches="tight")
plt.clf()

labels = ['Y11','Y12','Y13','Y21','Y22','Y23','Y31','Y32','Y33']
plt.ylim(0, max(ybus_mape_scaled)+s)
plt.axhline(y=s,linewidth=1, color='red', ls='dotted')
plt.bar(labels, ybus_mape_scaled,bottom=15)
plt.xticks(rotation='vertical')
plt.title("MAPE for Estimated Ybus")
plt.xlabel("Parameters")
plt.ylabel("MAPE (%)")
plt.savefig('MAPE_Ybus.png',bbox_inches="tight")
plt.clf()

labels = ['Y11','Y12','Y13','Y21','Y22','Y23','Y31','Y32','Y33']
plt.ylim(0, max(ybus_rmse_scaled)+s)
plt.bar(labels, ybus_rmse_scaled,bottom=15)
plt.xticks(rotation='vertical')
plt.title("RMSE for Estimated Ybus")
plt.xlabel("Parameters")
plt.ylabel("RMSE")
plt.savefig('RMSE_Ybus.png',bbox_inches="tight")
plt.clf()

#Noisy Case
results = evaluate_model(X_noisy, Y)
print("Noisy Case - Total MAPE: ",results[0])
print("Noisy Case - Total RMSE: ",results[1])
ybus_mape = results[2]
ybus_rmse = results[3]
x_mape = results[4]
x_rmse = results[5]
print("Noisy Case - Ybus MAPE:",ybus_mape)
print("Noisy Case - Ybus RMSE:",ybus_rmse)
print("Noisy Case - X MAPE:",x_mape)
print("Noisy Case - X RMSE:",x_rmse)

#Scale the results because matplotlib is stupid and won't plot float values for some reason
s = 100000000
s2 = 1000000000000
ybus_mape_scaled = []
ybus_rmse_scaled = []
for i in range(0,len(ybus_mape)):
    ybus_mape_scaled.append(int(ybus_mape[i]*s))
    ybus_rmse_scaled.append(int(ybus_rmse[i]*s))
x_mape_scaled = []
x_rmse_scaled = []
for i in range(0,len(x_mape)):
    x_mape_scaled.append(int(x_mape[i]*s))
    x_rmse_scaled.append(int(x_rmse[i]*s2))

#plot results
labels = ['V1.real','V1.imag','V2.real','V2.imag','V3.real','V3.imag','Psb.real','Psb.imag','i12.real','i12.imag','i21.real','i21.imag','i13.real','i13.imag','i31.real','i31.imag','i23.real','i23.imag','i32.real','i32.imag']
plt.ylim(0, max(x_mape_scaled)+s)
plt.axhline(y=s,linewidth=1, color='red', ls='dotted')
plt.bar(labels, x_mape_scaled,bottom=500)
plt.xticks(rotation='vertical')
plt.title("MAPE for Estimated Features")
plt.xlabel("Parameters")
plt.ylabel("MAPE (%)")
plt.savefig('Noisy_MAPE_X.png',bbox_inches="tight")
plt.clf()

labels = ['V1.real','V1.imag','V2.real','V2.imag','V3.real','V3.imag','Psb.real','Psb.imag','i12.real','i12.imag','i21.real','i21.imag','i13.real','i13.imag','i31.real','i31.imag','i23.real','i23.imag','i32.real','i32.imag']
plt.ylim(0, max(x_rmse_scaled)+s2/100)
plt.bar(labels, x_rmse_scaled,bottom=500)
plt.xticks(rotation='vertical')
plt.title("RMSE for Estimated Features")
plt.xlabel("Parameters")
plt.ylabel("RMSE")
plt.savefig('Noisy_RMSE_X.png',bbox_inches="tight")
plt.clf()

labels = ['Y11','Y12','Y13','Y21','Y22','Y23','Y31','Y32','Y33']
plt.ylim(0, max(ybus_mape_scaled)+s)
plt.axhline(y=s,linewidth=1, color='red', ls='dotted')
plt.bar(labels, ybus_mape_scaled,bottom=15)
plt.xticks(rotation='vertical')
plt.title("MAPE for Estimated Ybus")
plt.xlabel("Parameters")
plt.ylabel("MAPE (%)")
plt.savefig('Noisy_MAPE_Ybus.png',bbox_inches="tight")
plt.clf()

labels = ['Y11','Y12','Y13','Y21','Y22','Y23','Y31','Y32','Y33']
plt.ylim(0, max(ybus_rmse_scaled)+s)
plt.bar(labels, ybus_rmse_scaled,bottom=15)
plt.xticks(rotation='vertical')
plt.title("RMSE for Estimated Ybus")
plt.xlabel("Parameters")
plt.ylabel("RMSE")
plt.savefig('Noisy_RMSE_Ybus.png',bbox_inches="tight")
plt.clf()


