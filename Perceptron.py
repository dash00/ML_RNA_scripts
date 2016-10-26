#
# File directly inspired from the content of 
# Raschka, Sebastian. Python machine learning. Birmingham, UK: Packt Publishing, 2015. Print.
#

from common import *


class Perceptron(object):
    """ Perceptron classifier """

    def __init__(self, eta=0.01, iter_max=10):
        """
        Class constructor

        :param eta: Learning rate (between 0.0 and 1.0)
        :param iter_max: Passes over the training dataset.
        :return:
        """
        self.eta = eta
        self.iter_max = iter_max
    
    def fit(self, X, y):
        """
        Fit training data.

        :param X: Training data [n_samples, n_features], where n_samples
                  is the number of samples and n_features is the number of features.
        :param y: Target values [n_samples]
        :return: self
        """
        # Weights after fitting
        self.w_ = np.zeros(1 + X.shape[1])
        # Number of misclassifications in every epoch.
        self.errors_ = []
        # Let's suppose we converge during n_iter iterations
        for k in range(self.iter_max):
            # At each iteration, the error is computed as the number of updates
            # when running the learning process over the whole dataset
            errors = 0
            # For each train data xi and associated label target
            z = 0
            for xi, target in zip(X, y):
                # we compute (DW) and apply this update to each weight and to the bias
                prediction = self.predict(xi)
                update = self.eta * (target - prediction)
                self.w_[1:] += update * xi
                self.w_[0] += update
                # The sum of effective updates is updated
                errors += int(update != 0.0)
                z += 1
            # The error vector is updated for this epoch
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input wT.x

        :param X: Input vector
        :return: product
        """
        weighted_input = np.dot(X, self.w_[1:]) + self.w_[0]
        return weighted_input

    def predict(self, X):
        """Return class label after unit step

        :param X: Input vector
        :return: predicted class
        """
        prediction = np.where(self.net_input(X) >= 0.0, 1, -1)
        return prediction



