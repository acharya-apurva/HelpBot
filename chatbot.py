import json
import random
import torch
from nltk_script import tokenize, bagOfWords
from model import NeuralNetwork

device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('phrases.json','r') as f:
    phrases = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)


inputSize = data["input_size"]
outputSize = data["output_size"]
hiddenSize = data["hidden_size"]
tokens = data["tokens"]
tags = data["tags"]
modelState = data["model_state"]


nnModel = NeuralNetwork(inputSize, hiddenSize, outputSize).to(device) 
nnModel.load_state_dict(modelState)
nnModel.eval() #evaluation mode

#implementing the chatbot
bot_name = "Apurva"
print("Let's talk. You can type 'quit' to exit")
while True:
    sen = input('You: ')
    if sen == "quit":
        break
    sen = tokenize(sen)
    X = bagOfWords(sen, tokens)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = nnModel(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    #using the softmax function on the output from the last layer to get the probabilities
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]


    if prob.item() > 0.75:
        for phrase in phrases["phrases"]:
            if tag == phrase["tag"]:
                print(f"{bot_name}: {random.choice(phrase['botResponses'])}")
    else:
        print(f"{bot_name}: I am sorry. I do not understand that. ")



