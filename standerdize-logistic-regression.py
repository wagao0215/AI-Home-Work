


import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv
import math
from sklearn.metrics import confusion_matrix
from numpy import loadtxt, where, zeros, e, array, log, ones, append, linspace
from pylab import scatter, show, legend, xlabel, ylabel, contour, title
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import copy
from sklearn.metrics import f1_score



x = None
count = 0
labels = None
with open('question2-num.csv', 'rt') as csvfile:
    spamreader = csv.reader(csvfile,dialect=csv.excel)
    for row in spamreader:
        if(count == 0):
            labels = row
        else:
            if x is None:
                x = np.array([row])
            else:
                x = np.append(x,[row],axis=0)
        count=count+1


y = []
for temp_ele in x:
    y.append(temp_ele[24])

X=x[0:400,0:24]

X_train,y_train=X[0:320],y[0:320]
X_test,y_test=X[320:400],y[320:400]

X_train=np.array(X_train,dtype=float)
y_train=np.array(y_train,dtype=float)
X_test=np.array(X_test,dtype=float)
y_test=np.array(y_test,dtype=float)

def sigmoid(X,theta):


    X=X.dot(theta)
    X = np.array(X, dtype=float)
    sig=[]
    for it in range(X.shape[0]):
        h=1.0/(1.0+np.exp(-1.0*X[it]))
        sig.append(h)
    for item in range(len(sig)):
        if sig[item] > 0.5:
            sig[item] = 1.0
        else:
            sig[item] = 0.0
    sig=np.array(sig,dtype=float)
    return sig
##################################################################
#this is the cost function


def costfunction(theta,X,y,r):

    theta = np.array(theta, dtype=float)
    hc = sigmoid(X,theta)
    m = X.shape[0]

    midj=[]

    for i in range(hc.shape[0]):
        midcha=y[i]*log(hc[i])+(1-y[i])*log(1-hc[i])
        midj.append(midcha)

# calculate the sum of y*log(h(z))+(1-y)*log(1-h(z))

    mid1sum=0

    for item in midj:
        mid1sum=mid1sum+item

    #print mid1sum

# calculate the first part of of (-1/m)*sum of (y*log(h(z))+(1-y)*log(1-h(z)))

    J1=(-1*mid1sum)/m

# calculate the sum of w*w

    mid2sum=0

    for item in theta:

        mid2sum = mid2sum+item*item

    #print mid2sum

# calculate the second part of J

    J2 = r*mid2sum/(2*m)

# calculate the whole  COST VALUE

    J = J1+J2

    return J

# calculate gradient descent value

#intheta=ones(24)
#temperal=0.03

#print costfunction(intheta,X_test,y_test,temperal)



def gradient(theta,X,y,r):
    theta = np.array(theta, dtype=float)
    hc = sigmoid(X,theta)
    m = X.shape[0]
    afa=0.1
    midg1=[]
    for k in range(hc.shape[0]):
        midg1.append(hc[k]-y[k])
    newtheta = []
    for j in range(theta.shape[0]):
        midg2 = []
        for i in range(X.shape[0]):

            midg2.append(midg1[i]*X[i,j])
        newtheta.append(theta[j]-(afa/m)*(sum(midg2)+r*theta[j]))
    return newtheta

#intheta=ones(24)
#new=gradient(intheta,X_train,y_train,2)
#print intheta
#print new
#print len(new)
#print X_train,y_train


def compare(theta1,theta2):

    theta1 = np.array(theta1, dtype=float)
    theta2 = np.array(theta2, dtype=float)
    temp=[]
    min=0.02
    for i in range(theta1.shape[0]):
        cha=theta2[i]-theta1[i]
        cha=cha*cha
        temp.append(cha)

    value=math.sqrt(sum(temp))

    print value

    if value<=min:
        return True
    else:
        return False

#print compare(new,intheta)

def generatetheta(X,Y,r):
     final =[]
     intheta = ones(24)
     final.append(intheta)
     while True:
       intheta1= copy.deepcopy(intheta)
       NEWtheta1= gradient(intheta1,X,Y,r)
       temptheta= compare(intheta1,NEWtheta1)
       if temptheta:
          final.append(NEWtheta1)
          break
       else :
         intheta=copy.deepcopy(NEWtheta1)
         continue

     return final[1]


def generatethetanormal(X, Y, r):

    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
    X=scaler.transform(X)
    print X
    final = []
    intheta = ones(24)
    final.append(intheta)
    while True:
        intheta1 = copy.deepcopy(intheta)
        NEWtheta1 = copy.deepcopy(gradient(intheta1, X, Y, r))
        temptheta = compare(intheta1, NEWtheta1)
        if temptheta:
            final.append(NEWtheta1)
            break
        else:
            intheta = copy.deepcopy(NEWtheta1)
            continue

    return final[1]


#print generatetheta(X_train,y_train,2)

for lamda in np.arange(-2,4,0.2):
#while True:

    #lamda = 1

    finaltheta=generatethetanormal(X_train, y_train, lamda)

    #print finaltheta

    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_test)
    X_test=scaler.transform(X_test)
    finaly= sigmoid(X_test, finaltheta)

    #print finaly

    #print y_test

    #print confusion_matrix(y_test,finaly)

    #print f1_score(y_test, finaly, average='macro')

    break
