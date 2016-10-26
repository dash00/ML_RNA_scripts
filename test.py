import unittest

from common import *
from Perceptron import *
from Adaline import *
from AdalineS import *

import pandas as pd

class FullTestCase(unittest.TestCase):

    def test_perceptron(self):

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
        ppn = Perceptron(eta=0.1, iter_max=10)
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

    def test_adaline(self):

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
        X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
        X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
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

    def test_adaline_s(self):
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
        y = df.iloc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', -1, 1)
        X = df.iloc[0:100, [0, 2]].values
        X_std = np.copy(X)
        X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
        X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

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

    def test_wine(self):

        # Visualization
        names = ['Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash',
                 'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols',
                 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280_OD315_of_diluted_wines',
                 'Proline']
        df = pd.read_csv('./wine/wine.data', names = names)
        fig, ax = plt.subplots()
        print df.describe()
        groups = df.groupby('Class')
        for name, group in groups:
            ax.plot(group[['Alcohol']], group[['Color_intensity']], marker='o', linestyle='',ms=12, label=name)
        ax.legend(numpoints=1, loc='upper left')
        plt.show()

        # Fit
        df12 = df[df['Class'].isin([1, 2])]
        shuffle(df12)
        X = df12[['Alcohol', 'Color_intensity']].values
        X = (X - X.mean()) / (X.max() - X.min())
        y = df12['Class'].values
        y = np.where(y == 1, -1, 1)
        ppn = AdalineS(iter_max=150, eta=0.01, random_state=345)
        ppn.fit(X, y)
        plt.plot(range(1, len(ppn.cost_) + 1), ppn.cost_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.show()


        plot_decision_regions(X, y, classifier=ppn)
        plt.xlabel('sepal length [cm]')
        plt.ylabel('petal length [cm]')
        plt.legend(loc='upper left')
        plt.show()



if __name__ == '__main__':
    unittest.main()
