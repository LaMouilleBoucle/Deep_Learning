"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
# import adabound

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20, 30, 50, 100'
LEARNING_RATE_DEFAULT = 7e-3
MAX_STEPS_DEFAULT = 2000
BATCH_SIZE_DEFAULT = 700
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

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
  Performs training and evaluation of MLP model.

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  cifar10 = cifar10_utils.get_cifar10(one_hot=True)
  x_test = cifar10['test'].images.reshape(cifar10['test'].images.shape[0], -1)
  x_test = torch.from_numpy(x_test); y_test = cifar10['test'].labels

  in_features = cifar10['train'].images.shape[1] * \
                cifar10['train'].images.shape[2] * \
                cifar10['train'].images.shape[3]
  n_classes = cifar10['train'].labels.shape[1]

  mlp = MLP(in_features, dnn_hidden_units, n_classes, neg_slope)
  logloss = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(mlp.parameters(), FLAGS.learning_rate, amsgrad=True, weight_decay=0.01)
  #optimizer = adabound.AdaBound(mlp.parameters(), lr=FLAGS.learning_rate, final_lr=0.1)

  print('\nTraining has started...\n')
  avg_losses = []; accuracies = []
  avg_losses_test= []; accuracies_test = []
  for i in range(FLAGS.max_steps):
      x, y = cifar10['train'].next_batch(FLAGS.batch_size)
      x = x.reshape((x.shape[0],-1))
      x = torch.from_numpy(x); y = torch.from_numpy(y)

      optimizer.zero_grad()
      out = mlp.forward(x)
      avg_loss = logloss(out, y.argmax(dim=1))
      avg_loss.backward()
      optimizer.step()

      if (i+1) % FLAGS.eval_freq == 0:
          avg_losses += [avg_loss]
          accuracies += [accuracy(out.detach().numpy(), y.detach().numpy())]
          out_test = mlp.forward(x_test)
          avg_losses_test += [logloss(out_test, torch.from_numpy(y_test).argmax(dim=1))]
          accuracies_test += [accuracy(out_test.detach().numpy(), y_test)]
          print('Average loss of batch %i is %f' %(i+1, avg_losses[-1]))
          print('Accuracy of batch %i is %f' %(i+1, accuracies[-1]))
          print('')
  print('Training has ended...\n')

  print('The accuracy of the trained MLP on the test set is %f \n' %(accuracies_test[-1]))

  print('Plotting learning curve...\n')
  import matplotlib.pylab as plt
  plt.plot(np.arange(1,len(avg_losses)+1), avg_losses, 'r--', label='Log-loss Train')
  plt.plot(np.arange(1,len(avg_losses_test)+1), avg_losses_test, 'r-', label='Log-loss Test')
  plt.plot(np.arange(1,len(accuracies)+1), accuracies, 'g--', label='Accuracy Train')
  plt.plot(np.arange(1,len(accuracies_test)+1), accuracies_test, 'g-', label='Accuracy Test')
  plt.title('n_hidden=%s, lr=%.3f, max_steps=%i, B=%i, a=%.3f' \
          %(dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.batch_size, neg_slope))
  plt.ylabel('Accuracy / Loss')
  plt.xlabel('Evaluation Step')
  plt.legend()
  plt.show()
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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  FLAGS, unparsed = parser.parse_known_args()

  main()
