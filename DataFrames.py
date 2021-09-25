#!/usr/bin/env python
# coding: utf-8

# # This code will generate the data-frames to three folders and histogram to each folder.

# In[1]:


#importing the respective libraries
import numpy as np
import os
#generating array of file names of respective folders
trainingSetPath = '/Users/ashokchakravarthynara/Documents/vehicle-x/train/'
trainingList = os.listdir(trainingSetPath)
valPath = '/Users/ashokchakravarthynara/Documents/vehicle-x/val/'
valList = os.listdir(valPath)
testPath = '/Users/ashokchakravarthynara/Documents/vehicle-x/test/'
testList = os.listdir(testPath)


# In[4]:



import sys
import numpy

temp=[]#for overall dataset
temp1=[]#for training data folder
temp2=[]#for validation data folder
temp3=[]#for test data folder
for i in trainingList:
    temp.append(int(i[1:5]))
    temp1.append(int(i[1:5]))
for i in valList:
    temp.append(int(i[1:5]))
    temp2.append(int(i[1:5]))
for i in testList:
    temp.append(int(i[1:5]))
    temp3.append(int(i[1:5]))
print(len(temp))

arr=np.array(temp)
arr1=np.array(temp1)
arr2=np.array(temp2)
arr3=np.array(temp3)


# In[11]:


import matplotlib.pyplot as plt
# histogram for overall dataset
_ = plt.hist(arr, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram of class labels in overall Dataset")

plt.xlabel('class labels')
plt.ylabel('frequency')
plt.show()


# In[10]:


import matplotlib.pyplot as plt
# histogram for training data
_ = plt.hist(arr1, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram of class labels in Training Data")
plt.xlabel('class labels')
plt.ylabel('frequency')
plt.show()


# In[9]:


import matplotlib.pyplot as plt
# histogram for validation data
_ = plt.hist(arr2, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram of class labels in Validation Data")
plt.xlabel('class labels')
plt.ylabel('frequency')
plt.show()


# In[8]:


import matplotlib.pyplot as plt
# histogram for test data
_ = plt.hist(arr3, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram of class labels in Test Data")
plt.xlabel('class labels')
plt.ylabel('frequency')
plt.show()


# In[3]:


import numpy as np
import pandas as pd
import scipy.sparse as sparse
count=0
df=pd.DataFrame()#creation of empty dataframe

temp=[]
for i in trainingList:
    temp.append(int(i[1:5]))
arra = pd.DataFrame(np.array(temp)[np.newaxis]).transpose()# class label data frame
#print(arra.shape)
count=0
for i in trainingList:
    p = trainingSetPath + i
        df1=pd.DataFrame(np.load(p)).transpose()#loading of feature arrays 
        df = df.append(df1)
df = df.reset_index()#reseting index for appending the class labels
df['filename'] = pd.DataFrame(arra)


#print(df)
df.to_csv('Training.CSV')


# In[7]:


temp=[]
for i in trainingList:
        #print(i[1:5],i)
    temp.append(int(i[1:5]))
#print(temp)
arra = pd.DataFrame(np.array(temp)[np.newaxis]).transpose()
print(min(temp),max(temp))


# In[2]:


import numpy as np
import pandas as pd
import scipy.sparse as sparse
count=0
df=pd.DataFrame()

#valList,testList
#valPath,testPath

Val=[]
for i in valList:
     Val.append(int(i[1:5]))

arra_val = pd.DataFrame(np.array(Val)[np.newaxis]).transpose()

print(arra_val)
count=0
for i in valList:
        p = valPath + i
        df1=pd.DataFrame(np.load(p)).transpose()
        df = df.append(df1)
        count=count+1
        if(count%500==0):
            print(count)
    
df = df.reset_index()
df['filename'] = pd.DataFrame(arra_val)

print(df)
df.to_csv('/Users/ashokchakravarthynara/Documents/vehicle-x/Validation_dataframe.CSV')


# In[ ]:


import numpy as np
import pandas as pd
import scipy.sparse as sparse


#valList,testList
#valPath,testPath

test=[]
for i in testList:
     test.append(int(i[1:5]))
arra_test = pd.DataFrame(np.array(test)[np.newaxis]).transpose()
print(arra_test.shape)

df=pd.DataFrame()
count=0
for i in testList:
        p = testPath + i
        df1=pd.DataFrame(np.load(p)).transpose()
        df = df.append(df1)
        count=count+1
        if(count%500==0):
            print(count)
    
df = df.reset_index()
df['filename'] = pd.DataFrame(arra_test)

print(df)
df.to_csv('/Users/ashokchakravarthynara/Documents/vehicle-x/test_dataframe.CSV')

