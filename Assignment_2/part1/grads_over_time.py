from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

import argparse
import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

def make_plot(grad_norms):
    plt.plot(np.arange(len(grad_norms)), grad_norms, 'b-', label='Norm')
    plt.title('Norm of gradients at different timesteps for %s'%(config.model_type))
    plt.ylabel('Norm')
    plt.xlabel('T')
    plt.show()

def get_grad_norms(model, x, y):
    criterion = torch.nn.CrossEntropyLoss()
    out = model.forward(x)
    loss = criterion(out, y)
    for h in model.h_list:
        h.retain_grad()
    loss.backward()
    grad_norms = []
    for h in model.h_list:
        grad_norms += [h.grad.norm()]
    return grad_norms

def get_data(config):
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    x, y = next(iter(data_loader))

    return x.to(config.device), y.to(config.device)

def get_model(config):
    T = config.input_length
    D = config.input_dim
    H = config.num_hidden
    K = config.num_classes
    if config.model_type=='RNN':
        model = VanillaRNN(T,D,H,K).to(config.device)
    else:
        model = LSTM(T,D,H,K).to(config.device)

    return model

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()
    config.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    model = get_model(config)
    x, y = get_data(config)
    grad_norms = get_grad_norms(model, x, y)
    make_plot(grad_norms)
