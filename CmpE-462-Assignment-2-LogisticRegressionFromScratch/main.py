import numpy as np
import matplotlib.pyplot as plt
import csv
import math

CLASSES_DEMANDED = ["saab","van"]

BATCH_SIZE = 26
NUMBER_OF_FOLDS = 5


def read(fileName):
    with open(fileName,'r') as f:
        temp = f.read()
        temp = temp.split('\n')
        i = 0
        while True:
            if i >= len(temp):
                break
            
            temp[i] = temp[i].split(',')
            if len(temp[i]) ==1 or temp[i][-1] not in CLASSES_DEMANDED:
                del temp[i]
                i-=1
            i+= 1
            
        temp = np.array(temp)
        x = temp[:,:-1].astype(float)
        
        y = np.array([1.0 if car == "saab" else -1.0 for car in temp[:,-1]])[:,np.newaxis]
        # y = y.reshape((len(y),1))
        
        x = minMaxNormalization(x)
        return x,y
        
def minMaxNormalization(x):
    for i in range(len(x[1])):
        mx = max(x[:,i])
        mn = min(x[:,i])
        x[:,i] = (x[:,i] -mn) /(mx-mn)
    return x
def returnEstimation(x,w):
    global h_value
    h_value = np.dot(x,w)
    h_value = 1/(1+np.exp(-h_value))

    return [1 if i > 0.5 else -1 for i in h_value]
def giveMeStatistics(x,w,yy):
    
    y = returnEstimation(x, w)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(len(y)):
        if y[i] == 1 and yy[i] == 1:
            tp += 1
        elif y[i] == 1 and yy[i] == -1:
            fp += 1
        elif y[i] == -1 and yy[i] == 1:
            fn += 1
        elif y[i] == -1 and yy[i] == -1:
            tn += 1
    # print(tp,tn,fp,fn)
    accuracy = (tp + tn) / len(y)
    return accuracy
            
def logisticLoss(x,w,y):
    # global e,mean
    # a = np.dot(x,w)
    # # print("atype: ", type(a),type(y))
    # b = -np.multiply(y,a)
    # e = np.log(np.exp(b) + 1)
    # mean = np.mean(e)
    # # return mean
    return np.mean(np.log(np.exp(np.multiply(-y,np.dot(x,w)))  + 1))

    
    
    

    
def gradientLoss(x,y,w):
    # # global e,mean,a,b,pay,payda,c     

    # payda = (np.exp(np.multiply(y,np.dot(x,w)))  + 1)
    
    # pay = np.multiply(-y,x)
    
    # c = pay/payda
    # mean = np.mean(pay/payda,0)
    # # return np.resize(mean,(18,1))
    return np.resize(np.mean(-np.multiply(y,x)/(np.exp(np.multiply(y,np.dot(x,w)))  + 1),axis = 0),(18,1))

    
def regress(x,y,w,stepSize,errorThreshold):
    global v,i
    global debug
    losses = []
    currError = logisticLoss(x, w, y)
    w = np.array([0 for i in range(18)])[:,np.newaxis]
    i = 0
    while True:
        losses.append(currError)
        i+= 1
        g = gradientLoss(x,y,w) * stepSize
        v = -g
        w =  w + v
        prevError = currError
        currError = logisticLoss(x, w, y)
        change = abs(prevError - currError)
        if change < errorThreshold:
            break
    debug = returnEstimation(x, w)
    return w,np.array(losses)
def stochasticRegress(x,y,w,stepSize,errorThreshold):
    global v,i
    losses = []
    currError = logisticLoss(x, w, y)
    w = np.array([1.0 for i in range(18)])[:,np.newaxis]
    i = 0
    while True:
        losses.append(currError)
        i+= 1
        for b in range(len(x)//BATCH_SIZE):
            batch = x[np.random.randint(x.shape[0], size=BATCH_SIZE), :]
            
            g = gradientLoss(x,y,w) * stepSize
            v = -g
            w =  w + v
        
        prevError = currError
        currError = logisticLoss(x, w, y)
        change = abs(prevError - currError)
        if change < errorThreshold:
            break
    return w,np.array(losses)

def plotTheLosses(arr):
    plt.plot(arr)


def step1(xsList,ysList):
    global x,y
    global losses
    global results
    w = np.array([1.0 for i in range(18)])[:,np.newaxis]
    
    
    # w,losses = regress(x, y, w, 0.01, 0.00001)
    # print(giveMeStatistics(x, w, y))
    
    results = []
    
    for i in range(len(xsList)-1):
        w,losses = regress(xsList[i], ysList[i], w, 0.01, 0.00001)
        results.append(giveMeStatistics(xsList[4], w, ysList[4]))
        if i == 0:
            plt.plot(losses,'b',label = "0.01")
        plt.plot(losses,'b')
        # print("Batch Accuracy of fold",i," for 0.01: ",results[i])
    
    print("Batch Average Accuracy for 0.01 steps: ",sum(results)/len(results))
    
    
    
    results = []
    for i in range(len(xsList)-1):
        w,losses = regress(xsList[i], ysList[i], w, 0.10, 0.00001)
        results.append(giveMeStatistics(xsList[4], w, ysList[4]))
        if i == 0:
            plt.plot(losses,'c', label = "0.10")
        plt.plot(losses,'c')
        # print("Batch Accuracy of fold",i," for 0.10: ",results[i])
    print("Batch Average Accuracy for 0.10 steps: ",sum(results)/len(results))
    
    
    results = []    
    for i in range(len(xsList)-1):
        w,losses = regress(xsList[i], ysList[i], w, 0.25, 0.00001)
        results.append(giveMeStatistics(xsList[4], w, ysList[4]))
        if i == 0:
            plt.plot(losses,'g', label = "0.25")
        plt.plot(losses,'g')
        
        # print("Batch Accuracy of fold",i," for 0.25: ",results[i])
    print("Batch Average Accuracy for 0.25 steps: ",sum(results)/len(results))
    
    legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')
    plt.title("step1")
    plt.savefig("step1.png")
def step2(xsList,ysList):
    
    w = np.array([1.0 for i in range(18)])[:,np.newaxis]
    
    
    results = []
    for i in range(len(xsList)-1):    
        w,losses = stochasticRegress(xsList[i], ysList[i], w, 0.01, 0.00001)
        results.append(giveMeStatistics(xsList[4], w, ysList[4]))
        if i == 0:
            plt.plot(losses,'b',label = "0.01")
        plt.plot(losses,'b')
        # print("Stochastic Accuracy of fold",i," for 0.01: ",results[i])
    print("Stochastic Average Accuracy for 0.01 steps: ",sum(results)/len(results))
    results = []
    for i in range(len(xsList)-1):
        w,losses = stochasticRegress(xsList[i], ysList[i], w, 0.10, 0.00001)
        results.append(giveMeStatistics(xsList[4], w, ysList[4]))
        if i == 0:
            plt.plot(losses,'c', label = "0.10")
        plt.plot(losses,'c')
        # print("Stochastic Accuracy of fold",i," for 0.10: ",results[i])
    print("Stochastic Average Accuracy for 0.10 steps: ",sum(results)/len(results))    
    results = []
    for i in range(len(xsList)-1):
        w,losses = stochasticRegress(xsList[i], ysList[i], w, 0.25, 0.00001)
        results.append(giveMeStatistics(xsList[4], w, ysList[4]))
        if i == 0:
            
            plt.plot(losses,'g', label = "0.25")
        plt.plot(losses,'g')
        # print("Stochastic Accuracy of fold",i," for 0.25: ",results[i])
    print("Stochastic Average Accuracy for 0.25 steps: ",sum(results)/len(results))
    legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')
    plt.title("step2")
    plt.savefig("step2.png")

losses = []

# w = np.arange(1,19)

x,y = read("vehicle.csv")
length = len(x) // NUMBER_OF_FOLDS
xsList = np.split(x,[length,length*2,length*3,length*4])
ysList = np.split(y,[length,length*2,length*3,length*4])
# plotTheLosses(losses)
# w = np.array([1.0 for i in range(18)])[:,np.newaxis]
# w,losses = stochasticRegress(x,y,w,0.1,0.00001)
# plotTheLosses(losses)
# mean = gradientLoss()
# w = np.add(w,-mean).reshape(18,1)
step1(xsList, ysList)
step2(xsList, ysList)


# a = np.dot(x,w).reshape((416,1))
# e = np.exp(np.multiply(-y,a))

# pay = -np.multiply(y,x)

# mean = np.mean(pay / (e+1))


# an = np.dot(x,w)

# anan = np.multiply(np.dot(x,w),y)
# ananki = np.exp(-anan) + 1


















