import numpy as np
from operator import itemgetter
from numpy import genfromtxt
import math
import matplotlib.pyplot as plt

from decisionNode import decision_node 
from part2 import *
import sys
MAX_DEPTH = 5
dataset = []
labels = ["sepal-length","sepal-width","petal-length","petal-width"]


def readp1():
    with open("iris.csv","r") as f:
        text = f.read()
        lines = text.split("\n")
        dataset = lines[1:]
        
    for i in range(len(dataset)):
        dataset[i] = dataset[i].split(",")
        dataset[i][-1] = -1 if dataset[i][-1] == "Iris-setosa" else 1
        
    
    dataset = [[float(j) for j in i]for i in dataset]
        
    dataset = np.array(dataset)
    return dataset

def getGain(left,right,length,lcount):
    if lcount == 0:
        e1 = 0
    else:
        e1 = getEntrophy((left+lcount)/(lcount*2), (-left+lcount)/(lcount*2))
    rcount = length - lcount
    e2 = getEntrophy((right+rcount)/(rcount*2), (-right+rcount)/(rcount*2))
    
    return (e1*lcount + e2*rcount)/length
    
def getEntrophy(a,b):
    # print(a,b)
    if a*b != 0 :
        return -0.5 * math.log10(a) -0.5 * math.log10(b)
    else :
        return 1

def thresholdFinder1(data,index):
    
    global l,left,right,abc
    abc = data
    data = np.array(sorted(data,key = itemgetter(index)))
    
    
    l = []
    
    left = 0  
    right = sum(data[:,-1]) 
    
    
    gain = getGain(left, right, len(data), 0)
    l.append((gain,0,index,left,right))
    
    
    if abs(right) == len(data):
        #print(left,right,len(data))
        return (1,0,index,left,right)
    
    
    for i in range(len(data)-1):
        if data[i,index] != data[i+1,index]:
            
            curr = (data[i,index]+data[i+1,index])/2
            
            left = sum(data[:i+1,-1])    
            right = sum(data[i+1:,-1]) 
            
            gain = getGain(left, right, len(data), i+1)
            # gain = abs(left) + abs(right)
            l.append((gain,curr,index,left,right))
    
    
    return max(l)
def getGainRatio(gain,lcount,rcount):
    if lcount == 0:
        return gain
    total = lcount + rcount
    payda = lcount/total * math.log10(lcount/total) 
    payda+= rcount/total * math.log10(rcount/total )
    return -gain / payda
def thresholdFinder2(data,index):
    
    global l,left,right,abc
    abc = data
    data = np.array(sorted(data,key = itemgetter(index)))
    
    
    l = []
    
    left = 0  
    right = sum(data[:,-1]) 
    
    
    gain = getGain(left, right, len(data), 0)
    gratio = getGainRatio(gain, 0, len(data))
    l.append((gain,gratio,0,index,left,right))
    
    
    if abs(right) == len(data):
        # print(left,right,len(data))
        return (999999,0,index,left,right)
    
    
    for i in range(len(data)-1):
        if data[i,index] != data[i+1,index]:
            
            curr = (data[i,index]+data[i+1,index])/2
            
            left = sum(data[:i+1,-1])    
            right = sum(data[i+1:,-1]) 
            
            gain = getGain(left, right, len(data), i+1)
            gratio = getGainRatio(gain, i+1, len(data)-1-i)
            # gain = abs(left) + abs(right)
            l.append((gain,gratio,curr,index,left,right))
    
    bestGain = max(l)
    
    return (bestGain[1],bestGain[2],bestGain[3],bestGain[4],bestGain[5])



def tupleOpening(result):
    return result[0],result[1],result[2],result[3],result[4]
def createGainTree(data,depth,val,):
    global result
    if depth == MAX_DEPTH:
        return 1 if val > 0 else -1
    
    results = [thresholdFinder1(data, i) for i in range(len(data[0])-1)]
    
        
    result = max(results)
    # print(result)
    gain,threshold,column,leftVal,rightVal = tupleOpening(result)
    
    root = decision_node(threshold, column, val)
    
    if threshold == 0:
        return 1 if rightVal > 0 else -1
    elif gain == 1:
        root.left = 1 if leftVal > 0 else -1
        root.right = 1 if rightVal > 0 else -1
        
    else:
        sorter = data[:,result[2]] < result[1]
        leftData,rightData = data[sorter], data[~sorter]
        root.left  =    createGainTree(leftData, depth+1, left)
        root.right =    createGainTree(rightData,depth+1, right)
        
    return root
    
    
def createGainRatioTree(data,depth,val):
    global result
    if depth == MAX_DEPTH:
        return 1 if val > 0 else -1
    
    
    results = [thresholdFinder2(data, i) for i in range(len(data[0])-1)]
    result = max(results)
    # print(result)
    gain,threshold,column,leftVal,rightVal = tupleOpening(result)
    
    root = decision_node(threshold, column, val)
    
    if threshold == 0:
        return 1 if rightVal >= 0 else -1
    elif gain == 999999:
        root.left = 1 if leftVal > 0 else -1
        root.right = 1 if rightVal > 0 else -1
        
    else:
        sorter = data[:,result[2]] < result[1]
        leftData,rightData = data[sorter], data[~sorter]
        root.left  =    createGainRatioTree(leftData, depth+1, left)
        root.right =    createGainRatioTree(rightData,depth+1, right)
        
    return root
    
    
    
    

def testValue(value,root,clas):
    
    # print("left" if clas == -1 else "right" if clas == 1 else "\nstart")
    if root == 1:
        return 1
    elif root == -1:
        return -1
    
    
    # print(value[root.column],root.threshold,root.column,root.val)
    # if root.order:
    if value[root.column] < root.threshold:
        return testValue(value, root.left, -1)
    else:
        return testValue(value, root.right, 1)
    # else:
    #     if value[root.column] > root.threshold:
    #         return testValue(value, root.left, -1)
    #     else:
    #         return testValue(value, root.right, 1)




part = sys.argv[1]
step = sys.argv[2]

if part == "part1":
    dataset = readp1()            

    train_data = np.concatenate((dataset[:40], dataset[50:90]))
    test_data = np.concatenate((dataset[40:50], dataset[90:]))
    if step == "step1":
        
    
            
        
    
    
        root = createGainTree(train_data, 0,0)
        print("DT",labels[root.column],root.threshold)
    
    
    
        
        
        # hatali = []
        # for i in test_data:
            
        #     A = testValue(i, root, 0)
        #     if A != i[4]:
        #         hatali.append(i)
        # A = testValue([5.1, 3.5, 1.4, 0.2, -1.0], root, 0)
            
        # higher =train_data[:,0] <5.9
        # lower = 5.7 < train_data[:,0]
        # shower = train_data[higher * lower]
        # shower = sorted([[float(j) for j in i]for i in shower])
    
    elif step == "step2":
        
    
    
        
    
    
        root = createGainRatioTree(train_data, 0,0)
        print("DT",labels[root.column],root.threshold)
    
    
        
        
        # aga = [[5.5, 2.6, 4.4, 1.2, 1.0], [6.1, 3.0, 4.6, 1.4, 1.0], [5.8, 2.6, 4.0, 1.2, 1.0], [5.0, 2.3, 3.3, 1.0, 1.0], [5.6, 2.7, 4.2, 1.3, 1.0], [5.7, 3.0, 4.2, 1.2, 1.0], [5.7, 2.9, 4.2, 1.3, 1.0], [6.2, 2.9, 4.3, 1.3, 1.0], [5.1, 2.5, 3.0, 1.1, 1.0], [5.7, 2.8, 4.1, 1.3, 1.0]]
        # hatali = []
        # for i in test_data:
        #     print(i)
        #     A = testValue(i, root, 0)
        #     if A != i[4]:
        #         hatali.append(i)
        # A = testValue([5.1, 3.5, 1.4, 0.2, -1.0], root, 0)
        
        # higher =train_data[:,0] <5.9
        # lower = 5.7 < train_data[:,0]
        # shower = train_data[higher * lower]
        # shower = sorted([[float(j) for j in i]for i in shower])
        # print("asdasfaga")
    else:
        print("Unknown command")
elif part == "part2":
    read("wbcd.csv")

    if      step == "step1":
        step1()
    elif    step == "step2":
        step2()
    elif    step == "step1doAll":
        step1doAll()
    elif    step == "step2doAll":
        step2doAll()
    else:
        print("Unknown command...")
else:
    print("Unknown Command...")
# thresholdFinder1(dataset, 3)
    













    