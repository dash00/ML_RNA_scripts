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


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ada1 = Adaline(iter_max=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')
    ada2 = Adaline(iter_max=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()


    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    ada = Adaline(iter_max=15, eta=0.01)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.title('Adaline - Learning rate 0.01')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.show()
