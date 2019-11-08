import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import timeit
import time
import datetime
from tqdm import tqdm

def activation(input_array,function='sigmoid'):
    if function =='sigmoid':
        return 1/(1 + np.exp(-input_array))      

np.random.seed(8888)
x1 = np.linspace(-5,5, 400)
x2 = np.linspace(-5,5, 400)
np.random.shuffle(x1) 
np.random.shuffle(x2) 
d = x1**2 + x2**2    
# Normalize d 0.2~0.8
d_max = np.max(d)
d_min = np.min(d)
d = (d-d_min)/(d_max-d_min)*(0.8-0.2)+0.2 
#---------------- Input data ------------------------------
num_in  = 2

#----------------Hiddent Layer 1 ---------------------
num_L1  = 10
bias_L1 = np.random.uniform(-0.5,0.5,[num_L1,1])#5 1
w_L1    = np.random.uniform(-0.5,0.5,[num_in,num_L1])#2 5

#---------------- Output -----------------------------
num_out  = 1
bias_out = np.random.uniform(-0.5,0.5,[num_out,1])# 1 1 
w_out    = np.random.uniform(-0.5,0.5,[num_L1,num_out])# 5 1

#---------------- Parameter --------------------------
eta   = 0.01
mom   = 0.9
epoch = 250000

Eav_train = np.zeros([epoch])
Eav_test = np.zeros([epoch])

dw_out    = temp1 = np.zeros([num_L1,num_out]) #5 1 
dbias_out = temp2 = np.zeros([num_out,1])#1 1
dw_L1     = temp3 = np.zeros([num_in,num_L1])#2 5
dbias_L1  = temp4 = np.zeros([num_L1,1])# 5 1

#---------------- Traning ----------------------------
t0 = timeit.default_timer()
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
pbar = tqdm(total =epoch)
for i in range(epoch):         
    #--------------- Feed Forward  -------------------    
    e = np.zeros([300])
    E_train = np.zeros([300])
    for j in range(300):
        #X   = np.array([x1[j],x2[j]]).reshape(2,1)# 2 1
        X   = np.array([x1[j],x2[j]]).reshape(2,1)# 2 1
        L1  = activation(np.dot(np.transpose(w_L1),X) + bias_L1,'sigmoid')#5 1
        out = activation(np.dot(np.transpose(L1),w_out) + bias_out,'sigmoid')#1 1    
        #--------------- Back Propagation-----------------
        e[j] = (d[j]-out) #1 1
        E_train[j] = 0.5 * e[j]**2
        locg_k = e[j] * (out*(1-out))# 1 1
        temp2  = temp2 + mom * dbias_out + eta * locg_k * 1  #1 1     
        temp1  = temp1 + mom * dw_out + eta * locg_k * L1 #5 1                 
        locg_j = L1*(1-L1) * locg_k * w_out# 5 1        
        temp4  = temp4 + mom * dbias_L1 + eta * locg_j * 1 # 5 1        
        temp3  = temp3 + mom * dw_L1 + eta * np.dot(X,np.transpose(locg_j))#2 5

    dbias_out = temp2/300
    dw_out    = temp1/300       
    dbias_L1  = temp4/300
    dw_L1     = temp3/300        
    temp1 = np.zeros([num_L1,num_out]) #5 1 
    temp2 = np.zeros([num_out,1])#1 1
    temp3 = np.zeros([num_in,num_L1])#2 5
    temp4 = np.zeros([num_L1,1])# 5 1
    #----------  New weight --------------
    bias_out  = bias_out + dbias_out
    w_out     = w_out + dw_out        
    bias_L1   = bias_L1 + dbias_L1
    w_L1      = w_L1 + dw_L1
    #---------- Eave_train
    Eav_train[i]  = np.mean(E_train)

    #---------- Test data loss ---------------     
    E_test = np.zeros([100])
    for j in range(100):
        X      = np.array([x1[300+j],x2[300+j]]).reshape(2,1)# 2 1
        L1     = activation(np.dot(np.transpose(w_L1),X) + bias_L1,'sigmoid')#5 1
        out    = activation(np.dot(np.transpose(L1),w_out) + bias_out,'sigmoid')#1 1
        E_test = 0.5*( d[300+j] - out )**2        
    Eav_test[i] = np.mean(E_test)
    if i % 1000==0:
        pbar.update(1000)
        
pbar.close()
t1 =(timeit.default_timer()-t0)
print('Training time: {} min'.format((t1/60)))

#--------- Predict data --------------

y_predict = np.zeros([100])
E_predict = np.zeros([100])
for j in range(100):
        X      = np.array([x1[300+j],x2[300+j]]).reshape(2,1)# 2 1
        L1     = activation(np.dot(np.transpose(w_L1),X) + bias_L1,'sigmoid')#5 1
        out    = activation(np.dot(np.transpose(L1),w_out) + bias_out,'sigmoid')#1 1
        y_predict[j] = out
        E_predict[j] = 0.5*( d[300+j] - out )**2
Eav_predict = np.mean(E_predict)
#----------- Return the data they were normolized before ----------------------
y_predict = (y_predict-0.2)/(0.8-0.2)*(d_max-d_min)+d_min 

#------------ Record the result ------------------

import csv
table = [
    #['TimeStamp','Unit', 'Eta', 'Alpha','Training_loss','Predict_loss','Epoch','Time(min)'],
    [ now,num_L1, eta, mom, Eav_train[epoch-1], Eav_predict,epoch, int(t1/60)]    
]
with open('BPNN_output.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(table)

#------------ Scattering Data set ----------------------
#------- Return the data they were normolized before ---
d = (d-0.2)/(0.8-0.2)*(d_max-d_min)+d_min 

fig1 = plt.figure(num ='Data_Set', figsize=(10,5))
ax = fig1.add_subplot(121, projection='3d')
ax.scatter(x1[:300], x2[:300], d[:300], c='b', marker='o', s=5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.title('Training Data')

ax2 = fig1.add_subplot(122, projection='3d')
ax2.scatter(x1[300:], x2[300:], d[300:], c='r', marker='o', s=5)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')
plt.title('Testing Data')
plt.show()

#------------ Scattering y_Predict data -----------------

fig2 = plt.figure(num = now, figsize=(14,6))
ax1 = fig2.add_subplot(122, projection='3d')
ax1.scatter(x1[300:], x2[300:], y_predict[:],c='g', marker='o', s=15)
#ax1.scatter(x1[300:], x2[300:],d[300:])
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')
plt.title('Predict Data : y = x1^2 +x2^2')

#------------ plot training and testing Loss -----------------
ax3 = fig2.add_subplot(121)
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Loss')
ax3.plot(range(epoch),Eav_train,label='Train Set :'+str(Eav_train[epoch-1]))
ax3.plot(range(epoch),Eav_test, color='red', linewidth=1.0, linestyle='--', label='Test Set :'+ str(Eav_test[epoch-1]))
plt.legend(loc='upper right')
plt.title('Unit:'+str(num_L1)+', Eta:'+str(eta)+', Alpha:'+str(mom))
plt.show()



