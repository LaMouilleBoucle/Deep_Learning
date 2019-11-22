"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    """
    Initializes MLP object.

    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
      neg_slope: negative slope parameter for LeakyReLU

    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # Initializes all layers of the MLP.
    self.hidden_layers = []
    in_features = n_inputs
    for l in n_hidden:
        linear_module = LinearModule(in_features, l)
        lrelu_module = LeakyReLUModule(neg_slope)
        self.hidden_layers += [(linear_module, lrelu_module)]
        in_features = l

    linear_module = LinearModule(in_features, n_classes)
    softmax_module = SoftMaxModule()
    self.output_layer = (linear_module, softmax_module)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through
    several layer transformations.

    Args:
      x: input to the network
    Returns:
      out: outputs of the network

    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    a = x
    for l in self.hidden_layers:
        z = l[0].forward(a)
        a = l[1].forward(z)
    o = self.output_layer[0].forward(a)
    out = self.output_layer[1].forward(o)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss.

    Args:
      dout: gradients of the loss

    TODO:
    Implement backward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    da = self.output_layer[0].backward(self.output_layer[1].backward(dout))
    for l in self.hidden_layers[::-1]:
        lrelu_module = l[1]
        dz = lrelu_module.backward(da)

        linear_module = l[0]
        da = linear_module.backward(dz)
    ########################
    # END OF YOUR CODE    #
    #######################

    return

  def update(self, eta):
    """
    Performs parameter updates of each hidden layer

    Args:
      eta: the learning rate

    TODO:
    Implement update rules for parameters.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.output_layer[0].params['weight'] -= eta * self.output_layer[0].grads['weight']
    self.output_layer[0].params['bias'] -= eta * self.output_layer[0].grads['bias']

    for l in self.hidden_layers[::-1]:
        l[0].params['weight'] -= eta * l[0].grads['weight']
        l[0].params['bias'] -= eta * l[0].grads['bias']
    ########################
    # END OF YOUR CODE    #
    #######################

    return
