"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 1#5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.

  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  correct = sum(np.argmax(predictions,axis=1) == np.argmax(targets,axis=1))
  accuracy = correct/predictions.shape[0]
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model.

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  cifar10 = cifar10_utils.get_cifar10(one_hot=True)
  n_channels = cifar10['train'].images.shape[1]
  n_classes = cifar10['train'].labels.shape[1]

  convnet = ConvNet(n_channels, n_classes).to(device)
  logloss = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(convnet.parameters(), FLAGS.learning_rate)

  print('\nTraining has started...\n')
  avg_losses = []
  accuracies = []
  for i in range(FLAGS.max_steps):
      x, y = cifar10['train'].next_batch(FLAGS.batch_size)
      x = torch.from_numpy(x).to(device); y = torch.from_numpy(y).to(device)

      optimizer.zero_grad()
      out = convnet.forward(x)
      avg_loss = logloss(out, y.argmax(dim=1))
      avg_loss.backward()
      optimizer.step()

      if (i+1) % FLAGS.eval_freq == 0:
          avg_losses += [avg_loss]
          accuracies += [accuracy(out.cpu().detach().numpy(), y.cpu().detach().numpy())]
          print('Average loss of batch %i is %f' %(i+1, avg_losses[-1]))
          print('Accuracy of batch %i is %f' %(i+1, accuracies[-1]))
          print('')
  print('Training has ended...\n')

  print('Plotting learning curve...\n')
  import matplotlib.pylab as plt
  plt.plot(np.arange(1,len(avg_losses)+1)*100, avg_losses, 'r-', label='Log-loss')
  plt.plot(np.arange(1,len(avg_losses)+1)*100, accuracies, 'g-', label='Accuracy')
  plt.title('lr=%.4f, max_steps=%i, B=%i' \
          %(FLAGS.learning_rate, FLAGS.max_steps, FLAGS.batch_size))
  plt.ylabel('Accuracy / Loss')
  plt.xlabel('Batch')
  plt.legend()
  plt.show()

  x_test = cifar10['test'].images; y_test = cifar10['test'].labels
  x_test = torch.from_numpy(x_test).to(device);
  accuracy_test = accuracy(convnet.forward(x_test).cpu().detach().numpy(), y_test)
  print('The accuracy of the trained ConvNet on the test set is %f' %(accuracy_test))
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
