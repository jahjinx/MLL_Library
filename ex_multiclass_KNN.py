import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from MLL_Library.classification.KNeighborsClassifier import *

if __name__ == '__main__':
    print ('\nLoading the iris dataset courtesy of sklearn datasets.\nThis example applies the K- Nearest Neighbors algorithm to the iris for multiclass classification.')
    # load breast cancer dataset 
    load_data = datasets.load_iris()
    # assign variables
    X, y = load_data.data, load_data.target

    print(f'\nX Shape:{X.shape}, y Shape: {y.shape}')

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    # instantiate model
    knn = KNeighborsClassifier(n_neighbors=1, metric=chebyshev)

    # fit model
    knn.fit(X_train, y_train)

    # make predictions
    pred = knn.predict(X_test)

    print(f'\nPredictions: {pred}')
    print(f'\nGround-Truth: {y_test}')

    accuracy, recall, precision, F1 = metrics(y_test, pred, class_algo='multiclass', average=None)

    print('')
    print(f'Accuracy: {accuracy}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')
    print(f'F1: {F1}')