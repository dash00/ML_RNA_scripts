#
# File directly inspired from the content of 
# Raschka, Sebastian. Python machine learning. Birmingham, UK: Packt Publishing, 2015. Print.
#


from common import *



class Adaline(object):
    """ADAptive LInear NEuron classifier."""

    def __init__(self, eta=0.01, iter_max=50):
        """
        Class constructor

        :param eta: Learning rate (between 0.0 and 1.0)
        :param iter_max: Passes over the training dataset.
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
        # Weights after fitting.
        self.w_ = np.zeros(1 + X.shape[1])
        # Number of misclassifications in every epoch.
        self.cost_ = []
        # Let's suppose we converge during n_iter iterations
        for i in range(self.iter_max):
            # get the net input wT.x
            output = self.net_input(X)
            # We update each weight with gradient components
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # And we compute the corresponding global error J(w)
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input wT.x

        :param X: Input vector
        :return: product
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation

        :param X: Input vector
        :return: predicted class
        """
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step

        :param X: Input vector
        :return: predicted class
        """
        return np.where(self.activation(X) >= 0.0, 1, -1)

