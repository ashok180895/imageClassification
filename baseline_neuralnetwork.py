# -*- coding: utf-8 -*-
"""Baseline NeuralNetwork.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18qqbIEmaiMVjUtRt6uxtf86055L7hWDI
"""

from google.colab import drive

drive.mount('/content/gdrive')
root_path = 'gdrive/MyDrive/someyh.CSV'

# Commented out IPython magic to ensure Python compatibility.


# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.utils import prune
from xml.etree import ElementTree as ET

# Hyper Parameters
input_size = 2048
hidden_size = 1024
num_classes = 1362
num_epochs = 500
batch_size = 100
learning_rate = 0.01




# define a function to plot confusion matrix
def plot_confusion(input_sample, num_classes, des_output, actual_output):
    confusion = torch.zeros(num_classes, num_classes)
    for i in range(input_sample):
        actual_class = actual_output[i]
        predicted_class = des_output[i]

        confusion[actual_class][predicted_class] += 1

    return confusion


"""
Step 1: Load data and pre-process data
Here we use data loader to read data
"""


# define a customise torch dataset
        #b = torch.from_numpy(df)
class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data_tensor = torch.Tensor(df.values.astype(np.float32))
        

    # a function to get items by index
    def __getitem__(self, index):
        obj = self.data_tensor[index]
        input = self.data_tensor[index][0:-1]
        target = self.data_tensor[index][-1] -1 ## please check why it is doing -1 here. should only take the last column

        return input, target

    # a function to count samples
    def __len__(self):
        n, _ = self.data_tensor.shape
        return n

# load all data
#data = np.load('/Users/ashokchakravarthynara/Desktop/Lab2-20210422/vehicle-x/train')
data_training = pd.read_csv('/content/gdrive/MyDrive/Training_dataframe.CSV', header=None, index_col=False)# here someyh.CSV is training data_frame
data_val = pd.read_csv('/content/gdrive/MyDrive/Validation_dataframe.CSV', header=None, index_col=False)
data_test = pd.read_csv('/content/gdrive/MyDrive/test_dataframe.CSV', header=None, index_col=False)

#dropping the index columns in the data-frames
data_training=data_training.drop(0)
data_training = data_training.drop([0,1], axis=1)  
data_val=data_val.drop(0)
data_val= data_val.drop([0,1], axis=1)
data_test=data_test.drop(0)
data_test= data_test.drop([0,1], axis=1)

## normalise input data
for column in data_training.columns[:-1]:
    # the last column is target
  data_training[column] = data_training.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())
## normalise input data
for column in data_val.columns[:-1]:
    # the last column is target
  data_val[column] = data_val.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

## normalise input data
for column in data_test.columns[:-1]:
    # the last column is target
  data_test[column] = data_test.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())


# randomly split data into training set (80%) and testing set (20%)
msk = np.random.rand(len(data_val)) < 0.8
#train_data = data[msk]
#test_data = data[~msk]
train_data = data_training
val_data = data_val
test_data = data_test

# define train dataset and a data loader
train_dataset = DataFrameDataset(df=train_data)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


"""
Step 2: Define a neural network 

Here we build a neural network with one hidden layer.
    input layer: 8 neurons, representing the features of Glass
    hidden layer: 50 neurons, using Sigmoid as activation function
    output layer: 1362 neurons, representing the type of glass
"""


# Neural Network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out

net = Net(input_size, hidden_size, num_classes)

'''
parameters_to_prune = ((net.fc1, "weight"), (net.fc2, "weight"))

prune.global_unstructured(
    parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2
)
print(net.fc1.weight)


prune.remove(net.fc1, "weight")

'''


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) ## change the optimizer

# store all losses for visualisation
all_losses = []

# train the model by batch
for epoch in range(num_epochs):
    total = 0
    correct = 0
    total_loss = 0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        X = batch_x
        Y = batch_y.long()

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(X)
        loss = criterion(outputs, Y)
        all_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if (epoch % 50 == 0):
            _, predicted = torch.max(outputs, 1)
            # calculate and print accuracy
            total = total + predicted.size(0)
            correct = correct + sum(predicted.data.numpy() == Y.data.numpy())
            total_loss = total_loss + loss
    if (epoch % 50 == 0):
        print('Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f %%'
#               % (epoch + 1, num_epochs,
                 total_loss, 100 * correct/total))

# Optional: plotting historical loss from ``all_losses`` during network learning
# Please uncomment me from next line to ``plt.show()`` if you want to plot loss

import matplotlib.pyplot as plt
#
plt.figure()
plt.plot(all_losses)
plt.show()

"""
Evaluating the Results

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every glass (rows)
which class the network guesses (columns).

"""

train_input = train_data.iloc[:, :input_size]
train_target = train_data.iloc[:, input_size]

inputs = torch.Tensor(train_input.values).float()
targets = torch.Tensor(train_target.values.astype(np.int) - 1).long()

outputs = net(inputs)
_, predicted = torch.max(outputs, 1)

print('Confusion matrix for training:')
print(plot_confusion(train_input.shape[0], num_classes, predicted.long().data, targets.data))

"""
Step 3: Test the neural network

Pass testing data to the built neural network and get its performance
"""
# get testing data
test_input = test_data.iloc[:, :input_size]
test_target = test_data.iloc[:, input_size]

inputs = torch.Tensor(test_input.values).float()
targets = torch.Tensor(test_target.values.astype(np.int) - 1).long()

outputs = net(inputs)
_, predicted = torch.max(outputs, 1)

total1 = predicted.size(0)
correct1 = predicted.data.numpy() == targets.data.numpy()

print('Testing Accuracy: %.2f %%' % (100 * sum(correct1)/total1))

print('Confusion matrix for testing:')
print(plot_confusion(test_input.shape[0], num_classes, predicted.long().data, targets.data))

# get validation data
val_input = val_data.iloc[:, :input_size]
val_target = val_data.iloc[:, input_size]

inputs = torch.Tensor(val_input.values).float()
targets = torch.Tensor(val_target.values.astype(np.int) - 1).long()

outputs = net(inputs)
_, predicted = torch.max(outputs, 1)

total2 = predicted.size(0)
correct2 = predicted.data.numpy() == targets.data.numpy()

print('validation Accuracy: %.2f %%' % (100 * sum(correct2)/total2))

"""
Evaluating the Results

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every glass (rows)
which class the network guesses (columns).

"""



print('Confusion matrix for validation:')
print(plot_confusion(val_input.shape[0], num_classes, predicted.long().data, targets.data))