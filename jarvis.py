import random
import json
import torch
import numpy as np
import sys
from Brain import NeuralNet
from NeuralNetwork import bag_of_words, tokenize
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("intents.json", 'r') as json_data:
    intents = json.load(json_data)
    
FILE = "Traindata.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

Name = "Jarvis"
from Listen import Listen
from Speak import Say

def Main():
    sentence = Listen()
    
    if sentence == "bye":
        exit()
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    
    if X is None:
        print("Error: bag_of_words function returned None.")
        return
    
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                reply = random.choice(intent["responses"])
                Say(reply)
                
Main()
