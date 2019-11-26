from train import train
import argparse
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=1, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model for different on raw/one_hot data,
    #   for different values of T,
    #       with different seeds for Initialization.
    for dim, input_type in zip([1, 10],['raw', 'one-hot']):
        config.input_dim = dim
        for t_minus_one in range(4,25):
            config.input_length = t_minus_one
            avg_accuracies_train = []
            avg_grad_norms = []
            for s in [7,13,23,42,420]:
                np.random.seed(s)
                avg_accuracy_train, avg_grad_norms = train(config)
                avg_accuracies_train += [avg_accuracy_train]
                avg_grad_norms += [avg_grad_norm]

        plt.subplot(1,2,1)
        plt.plot(np.arange(4,25), avg_accuracies_train, 'g-', label='Accuracy')
        plt.title('Average accuracy on train batches for different values of T')
        plt.ylabel('Accuracy')
        plt.xlabel('T')

        plt.subplot(1,2,2)
        plt.plot(np.arange(4,25), avg_grad_norms, 'b-', label='Norm')
        plt.title('Average norm of gradients for different values of T')
        plt.ylabel('Norm')
        plt.xlabel('T')

        plt.show()

    # # Train the model for different values of T on the raw data
    # config.input_dim = 10
    # for t_minus_one in range(4,22):
    #     config.input_length = t_minus_one
    #     accuracies_train, accuracy_test = train(config)
    #     avg_accuracies_train += np.array(accuracies_train)
    #
    #     plt.plot(np.arange(len(accuracies_train)), accuracies, 'g--', label='Accuracy, T=%i' %(t_minus_one+1))
    # print('Done with training on raw data for T=%i test accuracy =%i.' %(t_minus_one+1, accuracy_test))
    # plt.title('Learning curve for training on one-hot data')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Batch')
    # plt.legend()
    # plt.show()
