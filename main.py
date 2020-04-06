from classification import *
from clustering import *
from common import *
from optimization import *

from sklearn import datasets
import matplotlib.pyplot as plt

n_samples = 1000
n_outliers = 50
X, y, coef = datasets.make_regression(n_samples=10, n_features=1,
                                      n_informative=1, noise=10,
                                      coef=True, random_state=0)

print(X)

# Plot data
plt.plot(X, y, "x")
plt.savefig('skl.png')
