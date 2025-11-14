# define class for PyTorch neural network

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from utils.utils import skip_if, generate_dataset
from sklearn.model_selection import train_test_split

class NeuralNetwork(nn.Module): 
    def __init__(self, num_inputs: int, num_nodes: list[int], activation_funcs: list[nn.Module]):
        super().__init__()
        num_hidden_layers = len(num_nodes)
        model_layers = [
            nn.Linear(num_inputs,num_nodes[0]),
            activation_funcs[0]
        ]
        for i in range(1,num_hidden_layers): 
            model_layers.append(nn.Linear(num_nodes[i-1],num_nodes[i]))
            model_layers.append(activation_funcs[i])

        self.layers = nn.Sequential(*model_layers)

    def forward(self, x):   # forward pass of the model
        out = self.layers(x)
        return out

class NeuralNetworkPDE(nn.Module):
    
    
    def train_model_regression(self,lr=0.01, num_epochs=3000, lambd=0,training_method = "SGD",step_method = "ADAM"):   # train NN model for regression, and return testing MSE
        criterion = nn.MSELoss(reduction='mean')

        if training_method == "SGD": 
            loader = train_loader_SGD
            n_batches = 5   # define number of batches, so the learning rate can be divided by it (gives correct correspondence with our neural network code)
        else: 
            loader = train_loader_GD
            n_batches = 1

        if step_method == "ADAM":
            optimizer = optim.Adam(self.parameters(), lr=lr/n_batches, weight_decay=lambd) # lambd is regularization parameter for L2 regularization
        if step_method=="RMSprop": 
            optimizer = optim.RMSprop(self.parameters(), alpha = 0.9,lr=lr/n_batches, weight_decay=lambd) # lambd is regularization parameter for L2 regularization

        for _ in range(num_epochs):
            self.train()  # set model to training mode

            for input, target in loader: 
                optimizer.zero_grad()            # reset gradients to zero
                outputs = self.linear_stack(input)          # forward pass: compute predictions
                loss = criterion(outputs,target)  # compute MSE
                loss.backward()                 # backpropagate to compute gradients
                optimizer.step()                # update weights using SGD step 

        return self.testing_MSE()