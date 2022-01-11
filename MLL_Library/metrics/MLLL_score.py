"""
=============================================================================
Scoring Metrics
=============================================================================
Description


build these first for classification only computes.
  Binary
    -any one or two unique value array
    -numbers are whole
    -can take strings

  Multiclass
    -any 3 or more classification array
    -numbers are whole
    -can take strings. we're not doing math, just compairng values.

  BINARY SCORES CANNOT BE SCRINGS BECAUSE OF FALSE NEGATIVE, ETC.
  MULTICLASS CORSE CAN BE STIRNGS BECAUSE "FALSE NEGATIVE" JUST MEANS "WRONG"

  create a def that gets all TP, FP, TN, FN and maybe use it to calc each metric
"""

import numpy as np

def _len_check(*args):
    """
    Checks that given arguments are of the same length.

    Application
    ---------------
    When evaluating machine learning models, the number of test and predicted
    classifications must be equal.

    Parameters
    ---------------
    *args : array, list, dictionary, tuple, set

    Returns
    ---------------
    Raise ValueError if arguements are not of the same lenght.

    Examples
    ---------------
    >>> array_pred = np.array([0,1,1,1,0,1,1,0])
    >>> array_test_1 = np.array([0,1,0,1,0,0,1])
    >>> array_test_2 = np.array([0,1,0,1,0,0,1])
    >>> _len_check(array_pred, array_test_1, array_test_2)
    ValueError: Arguments are of unequal length
    """

    size = []
    [size.append(len(arg)) for arg in args if len(arg) not in size]

    if len(np.unique(size)) > 1:
        raise ValueError("Arguments are of unequal length")


def _type_check(*args):
    """
    Determines if given arguments are binary or multiclass classifications. For
    the purpose of binary and multiclass machiene learning models, a binary
    classification can be a multiclass classification but a multiclass
    classification cannot be binary.

    Currently, this function only supports binary and multiclass arguments.

    Application
    ---------------
    Some scoring metrics use different equations for binary than for
    multiclass inputs, so we check that the types are compatable.

    Parameters
    ---------------
    *args : array, list, dictionary, tuple, set

    Returns
    ---------------
    List of unique argument types among all given arguments

    Examples
    ---------------
    >>> array_pred = np.array([0,1,1,1,0,1,1,0])
    >>> array_test_1 = np.array([0,1,0,1,0,0,1])
    >>> array_test_2 = np.array([0,1,0,1,0,0,1])
    >>> _type_check(array_pred, array_test_1, array_test_2)
    array(['binary', 'multiclass'], dtype='<U10')
    """

    arg_types = []
    for arg in args:
        if len(np.unique(arg)) <= 2:
            arg_types.append('binary')

        else:
            arg_types.append('multiclass')

    return np.unique(arg_types)


def _array_check(*args):
    """
    Determines if given arguments are a numpy array. If any of the given arguemnts
    are not numpy arrays, this function converts them to numpy arrays.

    Application
    ---------------
    Currently, our metrics() function uses numpy to perform element-wise
    operations, compareing arrays to calculate our metrics. array_check() ensures
    that arguements are numpy arrays.

    Parameters
    ---------------
    *args : array, list, dictionary, tuple, set

    Returns
    ---------------
    Given arguements as numpy arrays, stored in a list

    Examples
    ---------------
    >>> tuple_pred = (0,1,0,0,0,0,1)
    >>> tuple_test_1 = (0,1,0,1,0,0,1)
    >>> tuple_test_2 = (0,1,0,1,0,0,1)
    >>> _array_check(tuple_pred, tuple_test_1, tuple_test_2)
    [array([0, 1, 0, 0, 0, 0, 1]),
     array([0, 1, 0, 1, 0, 0, 1]),
     array([0, 1, 0, 1, 0, 0, 1])]
    """

    arg_store = []

    for arg in args:
        if type(arg) == np.ndarray:
            arg_store.append(arg)

        else:
            arg = np.array(arg)
            arg_store.append(arg)

    return arg_store


def _binary_scoring(true, pred):
    true_pos = np.sum((pred == 1) & (pred == true))
    false_pos = np.sum((pred == 1) & (pred != true))

    false_neg = np.sum((pred == 0) & (pred != true))

    # scoring metrics for binary classifications
    accuracy = np.sum(true == pred) / len(pred)
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, recall, precision, f1


def _multiclass_scoring(true, pred, average=None):
    # get correct and incorrect predictions as true/false
    pos_neg = np.equal(true, pred)
    # sum number of trues
    num_correct = np.sum(pos_neg)
    # get number of falses
    num_incorrect = len(pos_neg) - np.sum(pos_neg)

    # accuracy score for all multiclass scoring
    accuracy = num_correct / len(pos_neg)

    # average=None and average='macro' share some variables
    if average is None or average == 'macro':

        # determine our classes
        classes = np.unique(true)

        # define empty lists to store indiviual class score
        class_precision = []
        class_recall = []

        # individual class precision and recall scores
        for i in classes:
            class_precision.append(np.sum((true == i) & (true == pred)) / np.sum(pred == i))
            class_recall.append(np.sum((true == i) & (true == pred)) / np.sum(true == i))

        # F1 score used for average==None, the mean of which is F1 average='macro'
        f1 = 2 * np.divide(np.multiply(class_precision, class_recall), np.add(class_precision, class_recall))

        # precision and recall for average=None
        if average is None:
            # individual class precision scores
            precision = class_precision
            # individual classs recall scores
            recall = class_recall

        # precision, recall, and F1 for average='macro'
        elif average == 'macro':
            # mean of class's individual precision scores
            precision = np.mean(class_precision)
            # mean of class's individual recall scores
            recall = np.mean(class_recall)
            # mean of class's individual f1 scores
            f1 = np.mean(f1)

    # precision, recall, and F1 for precision='micro'
    elif average == 'micro':
        # precision calculated globally across all classes
        precision = num_correct / len(pos_neg)
        # recall calculated globally across all classes
        recall = num_correct / (num_correct + num_incorrect)
        # f1 calculated globally across all classes
        f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, recall, precision, f1


def metrics(true, pred, class_algo='binary', average=None):
    """
    Parameters
    ---------------
    true : array_like, predicted labels encoded as 1s and 0s

    pred : array_like, correct labels encoded as 1s and 0s

    class_algo : string, {'binary', 'multiclass'}, defines the classification type for scoring

    average :  string, {None, 'micro', 'macro'}, reqired for multiclass classifications. 
               If None, scores for each class returned. otherwise:
                    'micro': Counts  total true positives, false negatives and false positives.
                    'macro': Calc metrics for each label and returns their unweighted mean.

    Returns
    ---------------
    accuracy, recall, precision, F1

    Examples
    ---------------
    >>> list_pred = [0,1,0,0,0,0,1]
    >>> list_test_1 = [0,1,0,1,0,0,1]
    >>> accuracy, recall, precision, F1 = metrics(list_pred, list_test_1)
    >>> print(accuracy, recall, precision, F1)
    0.8571428571428571 0.6666666666666666 1.0 0.8

    Potential Improvements
    ---------------
    No Numpy : Adjust to use without Numpy. Element-wise operations such as below
      can work, but are signicantly slower than using Numpy.
        # if a (list 1) and b (list 2) are positive (1) and equal, set to 1 (true)
        and sum

        true_positive =
                np.sum([1 for a,b in zip(predictions, test) if a == 1 and b == a])

    Stringy : Make compaitable with string-like values. Currently, it only allows
    for binary integers.
    """

    # check that args of equal length
    _len_check(true, pred)

    # convert args to array if not already
    true, predicted = _array_check(true, pred)

    # if class_algo='binary' at least one arg is multiclass, raise error
    if class_algo == 'binary' and ('multiclass' in _type_check(true, pred)):
        raise ValueError("Select multiclass metric for multiclass types")

    # metrics for binary classifications
    if class_algo == 'binary':

        return _binary_scoring(true, pred)

    # for multiclass classification scoring
    elif class_algo == 'multiclass':

        return _multiclass_scoring(true, pred, average)
