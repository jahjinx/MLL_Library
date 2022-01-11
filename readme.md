# The Machine Learning Learning Library

## Description

The Machine Learning Learning Library is a small learning resource I wrote to better understand the structure and implementation of production machine learning packages such as scikit-learn. It currently contains the K-Nearest Neighbors classification algorithm as well as common scoring metrics (Accuracy, Recall, Precision, F1) with support for both binary and multiclass scoring. To support the K-Nearest Neighbors Algorithm, it also contains several distance metrics (Minkowski, Manhattan, Euclidean, Chebyshev).

## Usage
This repository contains two example files to run the K-Nearest Neighbors—one for binary and one for multiclass classification. See ex_binary_KNN.py and ex_multiclass_KNN.py respectively.

For Binary Classification with K-Nearest Neighbors:

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from MLL_Library.classification.MLLL_knn_classifier import *
# load breast cancer dataset 
load_data = datasets.load_breast_cancer()

# assign variables
X, y = load_data.data, load_data.target

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# instantiate model
knn = KNeighborsClassifier(n_neighbors=1, metric=chebyshev)

# fit model
knn.fit(X_train, y_train)

# make predictions
pred = knn.predict(X_test)
accuracy, recall, precision, F1 = metrics(y_test, pred, class_algo='binary', average=None)
```

For Multiclass Classification with K-Nearest Neighbors:

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from MLL_Library.classification.MLLL_knn_classifier import *
# load breast cancer dataset 
load_data = datasets.load_iris()

# assign variables
X, y = load_data.data, load_data.target

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# instantiate model
knn = KNeighborsClassifier(n_neighbors=1, metric=chebyshev)

# fit model
knn.fit(X_train, y_train)

# make predictions
pred = knn.predict(X_test)
accuracy, recall, precision, F1 = metrics(y_test, pred, class_algo='binary', average=None)
```
## MLL_Library.classification.KNeighborsClassifier
### Parameters
    n_neighbors : int, default=3
    metric : callable, default=minkowski, distance metric to be used for measuring distance between points. 
             Available metrics:   manhattan = minkowski @ order-p = 1 
                                  euclidean = minkowski @ order-p = 2 
                                  chebyshev = minkowski @ order-p = ∞
                                  minkowski
    p : int, default=2, power parameter for minkowski metric
