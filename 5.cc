import csv
import random
import math

def loadcsv(filename):
    lines=csv.reader(open(filename,"r"))
    dataset=list(lines)
    for i in range(len(dataset)):
        dataset[i]=[float(x) for x in dataset[i]]
    return dataset

def splitdata(dataset,splitratio):
    line=int(len(dataset)*splitratio)
    trainset=dataset[:line]
    testset=dataset[line:]
    return [trainset,testset]

def mean(numbers):
    return sum(numbers)/(len(numbers))

def stdev(numbers):
    avg=mean(numbers)
    v=0
    for x in numbers:
        v+=(x-avg)**2
    return math.sqrt(v/(len(numbers)-1))

def summarize(dataset):
    separated={}
    for i in range(len(dataset)):
        vector=dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]]=[]
        separated[vector[-1]].append(vector)
        
    summary={}
    for classval,instances in separated.items():
        summary[classval]=[(mean(attribute),stdev(attribute)) for attribute in zip(*instances)][:-1]
    print(summary)
    return summary

def prob(x,mean,std):
    exp=math.exp((-(x-mean)**2)/(2*(std**2)))
    return (1/((2*math.pi)**(1/2)))*exp

def predict(summary,instance):
    p_summary={}
    for classval,inst in summary.items():
        p_summary[classval]=1
        for i in range(len(inst)):
            mean,std=inst[i]
            x=instance[i]
            p_summary[classval]*=prob(x,mean,std)
    bestc,bestp=None,-1
    for classval,pro in p_summary.items():
        if bestc is None or bestp<pro:
            bestc=classval
            bestp=pro
    return bestc

def get_prediction(summary,testset):
    prediction=[]
    for i in range(len(testset)):
        p=predict(summary,testset[i])
        prediction.append(p)
    return prediction

def get_accuracy(pred,testset):
    count=0
    for i in range(len(pred)):
        if pred[i]==testset[i][-1]:
            count+=1
    return ((count/len(pred))*100)

filename="ConceptLearning.csv"
dataset=loadcsv(filename)
splitratio=0.90
trainset,testset=splitdata(dataset,splitratio)
summary=summarize(trainset)
pred=get_prediction(summary,testset)
print("\nPredicted :", pred)
print("\nAccuracy:",get_accuracy(pred,testset))
