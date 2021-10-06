


import numpy as np
from libsvm.svmutil import *
import matplotlib.pyplot as plt
import sys
from contextlib import contextmanager
import sys, os
param = svm_parameter('-q')
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            
            
def minMaxNormalization(x):
    for i in range(len(x[1])):
        mx = max(x[:,i])
        mn = min(x[:,i])
        x[:,i] = (x[:,i] -mn) /(mx-mn)
    return x
Cs = ["0.1", "1","50","500","1000"]
kernels = [" -t 0"," -t 1"," -t 2"," -t 3"]
names = ["linear","polynomial","radial","sigmoid"]
def read(file):
    global train_label,train_data,test_label,test_data,asd
    dataset = []
    with open(file) as f:
        lines = f.read().split("\n")
        
    
    asd = np.array([(lambda x : [1] + x[2:] if x[1]=="B" else [-1] + x[2:])( line.split(",")) for line in lines[1:]])
    dataset = np.array([[float(j) for j in i]for i in asd[:]])
    dataset = minMaxNormalization(dataset)
    train_label,train_data,test_label,test_data= dataset[:400,0],dataset[:400,1:],dataset[400:,0],dataset[400:,1:]
    
    
    
def svmFunc(parameters):
    # tempstd = sys.stdout
    # sys.stdout = open('dump.txt', 'w')

    global train_label,train_data,test_label,test_data
    # print(parameters)
    svmModel = svm_train(train_label, train_data, parameters) 
    
    train_lab, train_acc, train_val = svm_predict(train_label, train_data, svmModel, '-q')
    length = len(svmModel.get_SV())
    
    test_lab, test_acc, test_val  = svm_predict(test_label, test_data, svmModel, '-q')
    
    # sys.stdout.close()
    
    # sys.stdout = tempstd
    return (length,str((round(test_acc[0]/100, 4))))




def step2():
    global results
    
    for k in range(len(kernels)):
        with suppress_stdout():
            length,acc = svmFunc(kernels[k] + " -c "+ "1 -q")
        print("SVM kernel=" + names[k] + " C=1 acc=" + acc + " n=" + str(length))
        
        
        
def step1():
    global results
    i = 0
    for c in Cs:
        with suppress_stdout():
            length,acc = svmFunc(kernels[3] + " -c "+ c + " -q")
        print("SVM kernel=" + names[3] + " C=" +c +" acc=" + acc + " n=" + str(length))
        
        
def step1doAll():
    
    for c in Cs:
        for k in range(len(kernels)):
            length,acc = svmFunc(kernels[k] + " -c "+ c + " -q")
            print("SVM kernel=" + names[k] + " C=" +c +" acc=" + acc + " n=" + str(length))
        print()
        
def step2doAll():
    
    for k in range(len(kernels)):
        for c in Cs:
    
            length,acc = svmFunc(kernels[k] + " -c "+ c + " -q")
            print("SVM kernel=" + names[k] + " C=" +c +" acc=" + acc + " n=" + str(length))
        print()
            




# step1()
# print()
# step2()
# print()
# doAll()

    
    
    









    
    