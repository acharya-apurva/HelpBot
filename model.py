import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClasses):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(inputSize, hiddenSize) #Neural Network layer 1 the parameters are input size and output size of the layer
        self.l2 = nn.Linear(hiddenSize, hiddenSize)
        self.l3 = nn.Linear(hiddenSize, numClasses) 
        self.relu = nn.ReLU() #the RELU activation funtion is used

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out