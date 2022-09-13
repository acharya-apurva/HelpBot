import json
import random
import numpy as np
from nltk_script import *

#importing things needed for pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNetwork

with open('phrases.json', 'r') as f: #with statements open resources and guarantees that they will be closed after the block completes
    phrases = json.load(f)

#print(phrases)

tokens = []
tags = []
final = []
punctuation =['.','?','!',',']


#tokenizing

for phrase in phrases['phrases']:
    tag = phrase['tag']#the tag key
    tags.append(tag)
    for pattern in phrase['patterns']:
        token = tokenize(pattern) 
        tokens.extend(token) #since token is an array
        final.append((token, tag)) #appending a tuple

#stemming and excluding punctuation 

tokens = [stem(token) for token in tokens if token not in punctuation]
tokens = sorted(set(tokens)) #set to remove duplicates and the sorted will return the list
tags = sorted(set(tags))

X_train = []
Y_train = []
#creating the bag of words
for (pattern_sen, tag) in final:
    bag = bagOfWords(pattern_sen, tokens)
    X_train.append(bag)

    label = tags.index(tag) #to make sure we get number for our labels
    Y_train.append(label)

#converting X_train and Y_train to numpy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)

#defining hyperparameters
batch_size=8
hiddenSize=8
outputSize= len(tags)
inputSize= len(X_train[0])
learningRate = 0.001
numEpochs = 1000

#creating a dataset
class ChatDataSet(Dataset):
    def __init__(self):
        self.numSamples = len(X_train)
        self.xData = X_train
        self.yData = Y_train

    def __getitem__(self, index): #to access dataset[index]
        return self.xData[index], self.yData[index]
    
    def __len__(self):
        return self.numSamples

dataset = ChatDataSet()
trainLoader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0) #num_workers for multi threading makes it a bit faster

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#use the GPU if available
nnModel = NeuralNetwork(inputSize, hiddenSize, outputSize).to(device) #push the model to device 

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nnModel.parameters(), lr=learningRate)

#training the model
for epoch in range(numEpochs):
    for (words, labels) in trainLoader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        #forward pass
        outputs =nnModel(words)
        loss = criterion(outputs, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward() #calculate the back propagation
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{numEpochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

#save the data
data = { #dictionary
    "model_state": nnModel.state_dict(),
    "input_size": inputSize,
    "output_size": outputSize,
    "hidden_size": hiddenSize,
    "tags": tags,
    "tokens": tokens
}

FILE = "data.pth" #pytorch file type
torch.save(data,FILE)

print(f'training complete and file is saved to {FILE}')

