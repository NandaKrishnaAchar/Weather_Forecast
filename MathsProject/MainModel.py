#Importing Librabries
import pandas as pd
import numpy as np
import math
import collections
import matplotlib.pyplot as plt
from sklearn.externals import joblib

#importing dataset
dataset=pd.read_csv("Data.csv")
df=pd.DataFrame(dataset)

#calculating frequency
frequency=collections.Counter(df.iloc[:,-1])

#to find mean and probability
probability=[]
i=0
mean=0
for key,value in frequency.items():
    probability.insert(i,float(value/28))
    mean=mean+probability[i]*key
    i=i+1

#Median
d=dict(zip(frequency.keys(),probability))
median_value=int(d.__len__()/2)
median=list(d.keys())[median_value]

#Mode
mode=list(d.keys())[list(d.values()).index(max(d.values()))]

#Plotting Probability Distribution:
T=np.array(list(frequency.keys()))
power=np.array(probability)
from scipy.interpolate import spline
xnew = np.linspace(T.min(),T.max(),300) 
power_smooth = spline(T,power,xnew)
plt.plot(xnew,power_smooth)
plt.axvline(x=mean,color='red',label='Mean')
plt.title('Probability Distribution Graph')
plt.xlabel('Temperature')
plt.ylabel('Probability Distribution Function')
plt.show()

#to find variance
i=0
var=0
for key,value in frequency.items():
    var=var+((key-mean)**2)*probability[i]
    i=i+1
    
#Standard deviation
sd=math.sqrt(var)

#Printing
print("\n\n\nMeasure of Central Tendency:\n")
print("Mean of probability distribution is %.3f"%mean)
print("Median of probability distribution is %.3f"%median)
print("Mode of probability distribution is %.3f"%mode)
print("Standard deviation of probability distribution is %.3f"%sd)
print("Variance of probability distribution is %.3f"%var)

#Probability Distribution Of Test Random Variable:
print("\n\n\nRandom Variable Testing:")
rand_temp=float(input("Enter the random temperature"))
prob_rand_temp=np.interp(float(rand_temp),T,power)
print("Probability function of Random temperature is %.1f" %prob_rand_temp)

#Probability Application in Machine Learning:
print("\n\n\nApplication of probability function to analyze weather using Logisstic Regression Classifiation(ML Model) of each weekday and predict the weather of current weekday\n")
from datetime import datetime
from datetime import timedelta  
week = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
sample=np.array([week[datetime.today().weekday()]])

#LabelEncoder and OneHotEncoder
labelencoder=joblib.load('LabelEncoderCategories')
sample=labelencoder.transform(sample)

onehotencoder=joblib.load('OneHotEncoderCategories')
sample=onehotencoder.transform(sample.reshape(1,-1)).toarray()

#Predicting
classifier=joblib.load("MLModelLogisticRegression")
prob=classifier.predict_proba(sample)
print("The probability this weekday being Rainy=" + str(prob[0][0]) +"\nThe probability this weekday being Sunny=" +str(prob[0][1]))

        