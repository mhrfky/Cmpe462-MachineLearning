import sys
import math
import random
import numpy as  np
import matplotlib.pyplot as plt
import csv
import time

class dataPointGenerator:
    
    def __init__(self,num,w):
        self.num = num
        self.w = w
        self.create()
    def create(self):
        x = random.uniform(-self.num, self.num)
        y = random.uniform(-self.num, self.num)
        c = self.checkClass([1,x,y])
        self.data = np.array([[1,x,y,c]])
        for i in range(self.num-1):
            x = random.uniform(-self.num, self.num)
            y = random.uniform(-self.num, self.num)
            
            c = self.checkClass([1,x,y])
            if c == 0:
                i-= 1
            else:
                self.data = np.concatenate((self.data,[[1,x,y,c]]))    
            
    def checkClass(self,x):
        c = 0
        for i in range(len(x)):
            c += self.w[i] * x[i]
        return 0 if abs(c) <1 else 1 if c<0 else -1
    
    def returnPoints(self):
        return self.data
    def colorPalette(self):
        cols = []
        for x in self.data:
            if x[-1] == 1:
                cols.append("blue")
            else:
                cols.append("red")
        return cols
    def plot(self):
        plt.xlim((-self.num,self.num))
        plt.ylim((-self.num,self.num))
        cols = self.colorPalette()
        plt.scatter(self.data[:,1],self.data[:,2],c = cols)
        x = np.array(range(-self.num,self.num))
        y = [+self.w[0] - self.w[1]*i for i in x]
        plt.plot(x, y)
        plt.show()
        
        
        
        
class perceptron:
    def __init__(self,data,num,part,step):
        self.num = num
        self.data = data
        self.w = np.array([-10,10,10])
        self.findTheSeperator()
        self.part = part
        self.step = step
        self.plot()
    def findTheSeperator(self):
        while True:
            
            # print(self.w)
            if self.checkClass():
                break
            
            
    def checkClass(self):
        tempList = []
        for i in range(len(self.data)):
            temp = self.data[i]
            
            c = np.dot(temp[:3],self.w)
            c = 1 if c > 0 else -1
            
            if c * temp[-1] == -1:
                tempList.append(i)
        
        if len(tempList) == 0:
            return True
        
        temp = self.data[random.choice(tempList)]
        self.w = np.add(self.w, temp[:3] * temp[3])
        return False
    def colorPalette(self):
        cols = []
        for x in self.data:
            if x[-1] == 1:
                cols.append("blue")
            else:
                cols.append("red")
        return cols
    def plot(self):
        plt.xlim((-self.num,self.num))
        plt.ylim((-self.num,self.num))
        cols = self.colorPalette()
        plt.scatter(self.data[:,1],self.data[:,2],c = cols,marker = 'x')
        x = np.array(range(-self.num,self.num))
        y = [(self.w[0]/self.w[2]) - (self.w[1]/self.w[2])*i for i in x]
        yy = [+ww[0] - ww[1]*i for i in x]
        plt.plot(x, y,color = 'm')
        plt.plot(x, yy,color = 'g')
        plt.savefig(self.part + "_" + self.step)

class linearRegression:
    def __init__(self,fileName,alpha):
        self.read(fileName)
        self.alpha = alpha
        start = time.time()
        self.regress()
        end = time.time()
        print(self.w)
        print("Took ",end-start," seconds to finish.")
        
    def read(self,fileName):
        with open(fileName,'r') as f:
            temp = f.read()
            temp = temp.split('\n')
            for i in range(len(temp)):
                
                temp[i] = temp[i].split(',')
                if len(temp[i]) ==1:
                    del temp[i]
            temp = np.array(temp)
            self.x = temp[:,:-1].astype(float)
            self.t = temp[:,-1].astype(float)
            self.transposeX = np.transpose(self.x).astype(float)

    def regress(self):
        temp = np.matmul(self.transposeX,self.x) + self.alpha * np.identity(self.x.shape[1])
        temp = np.linalg.inv(  temp)
        temp = np.matmul(temp,self.transposeX)
        temp = np.matmul(temp,self.t)
        self.w = temp
    # (T*x)^-1*T*t
    
    
ww = np.array([-1,3,1])    

if len(sys.argv) > 1:
    if sys.argv[1] == "part1":
        if sys.argv[2] == "step1":
            
            ret = dataPointGenerator(50, ww)
            data = ret.returnPoints()
            
            del ret
            pp = perceptron(data,50,sys.argv[1],sys.argv[2])
            
        elif sys.argv[2] == "step2":
            
            ret = dataPointGenerator(100, ww)
            data = ret.returnPoints()
            
            del ret
            pp = perceptron(data,100,sys.argv[1],sys.argv[2])
            
        elif sys.argv[2] == "step3":
            ret = dataPointGenerator(5000, ww)
            data = ret.returnPoints()
            
            del ret
            pp = perceptron(data,5000,sys.argv[1],sys.argv[2])
        else:
            pass
    elif sys.argv[1] == "part2":
        if sys.argv[2] == "step1":
            lr = linearRegression("ds1.csv",0)
            x = lr.x
            tX = lr.transposeX
            t = lr.t
            a = lr.w
        elif sys.argv[2] == "step2":
            lr = linearRegression("ds2.csv",0)
            x = lr.x
            tX = lr.transposeX
            t = lr.t
            a = lr.w
        elif sys.argv[2] == "step3":
            lr = linearRegression("ds2.csv",math.exp(-18))
            x = lr.x
            tX = lr.transposeX
            t = lr.t
            a = lr.w
        else:
            pass
    else:
        pass
    
    
    
