from numpy.random import seed
from common import *

class AdalineS(object):
    """Stochastic ADAptive LInear NEuron classifier."""

    def __init__(self, eta=0.01, iter_max=10, shuffle=True, random_state=None):
        """
        Class constructor

        :param eta: Learning rate (between 0.0 and 1.0)
        :param iter_max: Passes over the training dataset.
        :param shuffle: use shuffle on data
        :param random_state: predefined random state
        """
        self.eta = eta
        self.iter_max = iter_max
        self.w_initialized = False
        self.shuffle = shuffle

        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """
        Fit training data.

        :param X: Training data [n_samples, n_features], where n_samples
                  is the number of samples and n_features is the number of features.
        :param y: Target values [n_samples]
        :return: self
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.iter_max):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """
        Fit training data without reinitializing the weights

        :param X: Training data [n_samples, n_features], where n_samples
                  is the number of samples and n_features is the number of features.
        :param y: Target values [n_samples]
        :return: self
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """
        Shuffle training data

        :param X: Training data [n_samples, n_features], where n_samples
                  is the number of samples and n_features is the number of features.
        :param y: Target values [n_samples]
        :return: shuffled training data
        """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

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

if __name__ == '__main__':

    import pandas as pd

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

    ada = AdalineS(iter_max=15, eta=0.01, random_state=1)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.title('Adaline - SGD - Learning rate 0.01')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.show()