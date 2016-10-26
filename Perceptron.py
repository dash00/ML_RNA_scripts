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


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # Visualize data
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()

    # Train Perceptron
    ppn = Perceptron(eta = 0.1, iter_max= 10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()
    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

# if __name__ == '__main__':
#
#     import pandas as pd
#
#     # Visualization
#     names = ['Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash',
#              'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols',
#              'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280_OD315_of_diluted_wines',
#              'Proline']
#     df = pd.read_csv('./wine/wine.data', names = names)
#     fig, ax = plt.subplots()
#     print df.describe()
#     groups = df.groupby('Class')
#     for name, group in groups:
#         ax.plot(group[['Alcohol']], group[['Color_intensity']], marker='o', linestyle='',ms=12, label=name)
#     ax.legend(numpoints=1, loc='upper left')
#     plt.show()
#
#     def shuffle(ldf, n=1, axis=0):
#         ldf = ldf.copy()
#         for _ in range(n):
#             ldf.apply(np.random.shuffle, axis=axis)
#         return ldf
#
#     # Fit
#     df12 = df[df['Class'].isin([1, 2])]
#     #shuffle(df12)
#     X = df12[['Alcohol', 'Color_intensity']].values
#     #X = (X - X.mean()) / (X.max() - X.min())
#     y = df12['Class'].values
#     y = np.where(y == 1, -1, 1)
#     print X, y
#     #print np.concatenate((X, y), axis=1)
#     ppn = Perceptron(eta = 0.01, n_iter = 10000)
#     ppn.fit(X, y)
#     plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
#     plt.xlabel('Epochs')
#     plt.ylabel('Number of misclassifications')
#     plt.show()
#
#
#     plot_decision_regions(X, y, classifier=ppn)
#     plt.xlabel('sepal length [cm]')
#     plt.ylabel('petal length [cm]')
#     plt.legend(loc='upper left')
#     plt.show()
#
#     for xi, target in zip(X, y):
#         print "({}) -> {} vs. {}".format(xi, ppn.predict(xi), target)
