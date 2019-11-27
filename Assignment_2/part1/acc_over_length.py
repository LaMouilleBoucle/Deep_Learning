import argparse
import torch

from train import train
from dataset import PalindromeDataset
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

def make_plot(avg_accuracies):
    plt.plot(np.arange(len(avg_accuracies[0])), avg_accuracies[0], 'b--', label='Raw')
    plt.plot(np.arange(len(avg_accuracies[1])), avg_accuracies[1], 'g-', label='One-Hot')
    plt.title('Accuracy of %s for different sequence lengths T' %(config.model_type))
    plt.ylabel('Accuracy')
    plt.xlabel('T')
    plt.legend()
    plt.show()

def get_data(config):
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    x, y = next(iter(data_loader))

    return x.to(config.device), y.to(config.device)

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()
    config.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    for i, dim in enumerate([1,10]):
        config.input_dim = dim
        avg_accuracies = [[],[]]
        for t_minus_one in range(4,30):
            config.input_length = t_minus_one
            accuracy = 0
            for s in [7, 13, 23, 42, 420]:
                np.random.seed(s)
                model = train(config)
                x, y = get_data(config)
                out = model.forward(x)
                accuracy += torch.sum((out.argmax(dim=1) == y).float())/x.size(0)
            avg_accuracies[i] += [accuracy/5]

    make_plot(avg_accuracies)
