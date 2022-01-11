"""
=============================================================================
Distance/Similarity Metrics
=============================================================================
When measuring distance/similarity between arrays, we have a number of metrics
at our disposal.

Each metric is described in their respective function below. However, let's
first note that each calculation below is performed in normed vector space 
rather than by following what may be a more fimilar distance equation. This is 
for several reasons:
  1. Easier to read and write in Pyhon
  2. One less library to import
  3. Minknowski makes it difficult, but also easier

In order for the Minkowski Distance below to work properly, we need a limiting 
function as the required order-p approaches ∞. Scypy offers a limiting 
function, but we can also use Numpy's linalg.norm to the same result.

Additionally, we can pass Minkowski to our Manhattan and Euclidean functions as: 
  Manhattan Distance = Minkowski Distance @ order-p = 1 
  Euclidean Distance = Minkowski Distance @ order-p = 2 
  Chebyshev Distance = Minkowski Distance @ order-p = ∞

Refrence
---------------
  https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

"""

import numpy as np

def minkowski(x, y, p, w=None):
    """
    Computes the Minkowski distance between two 1-D arrays.

    Parameters
    ---------------
    x : (N,) array_like
        Input array.
    y : (N,) array_like
        Input array.

    Returns
    ---------------
    The Minkowski distance between vectors 'x' and 'y' at order-p.

    KaTeX Formula
    ---------------
    d(x,y) =  \|x-y\|_p = \lim\limits_{p→∞}(\displaystyle\sum_{i=1}^n|x_i-y_i|^p)^\frac{1}{p}

    Examples
    ---------------
    >>> x = np.array([6,14,-10])
    >>> y = np.array([-8,12,3])
    >>> minkowski(x, y, 2)
    19.209372712298546
    """
    x_y = x - y
    dist = np.linalg.norm(x_y, ord=p)
    return dist


def manhattan(x, y):
    """
    Computes the Manhattan distance between two 1-D arrays using Minkowski
    Distance at order-p = 1.

    Manhattan, or 'City Block' Distance, can be understood in 2D space as the
    distance between two points measured along the axes at right angles--as if
    walking from point A to point B in a blocked city.

    Parameters
    ---------------
    x : (N,) array_like
        Input array.
    y : (N,) array_like
        Input array.

    Returns
    ---------------
    The Manhattan distance between vectors 'x' and 'y'.

    KaTeX Formula
    ---------------
    d(x,y) = \|x-y\|_1 = \displaystyle\sum_{i=1}^n|x_i-y_i|

    Alterntive Function
    ---------------
    np.sum(np.absolute((x-y)))

    Examples
    ---------------
    >>> x = np.array([6,14,-10])
    >>> y = np.array([-8,12,3])
    >>> manhattan(x, y)
    29.0
    """
    return minkowski(x, y, p=1)


def euclidean(x, y):
    """
    Computes the Euclidean distance between two 1-D arrays using Minkowski
    Distance at order-p = 2.

    Euclidean Distance can be understood in 2D space as the distance between two
    points as represented by a straight line. It is exemplified by the addages 
    "As the crow flies" and "The shortest distance between two points is a
    straight line". Recalling basic geometry, it can also be calulated and
    undersood by Pythagorean Teorem.

    Parameters
    ---------------
    x : (N,) array_like
        Input array.
    y : (N,) array_like
        Input array.

    Returns
    ---------------
    The Euclidean distance between vectors 'x' and 'y'.

    KaTeX Formula
    ---------------
    d(x,y) = \|x-y\|_2 = \sqrt{\displaystyle\sum_{i=1}^n|x_i-y_i|^2}

    Alterntive Function
    ---------------
    np.sqrt(np.sum((q-z)**2))

    Examples
    ---------------
    >>> x = np.array([6,14,-10])
    >>> y = np.array([-8,12,3])
    >>> euclidean(x, y)
    19.209372712298546
    """
    return minkowski(x, y, p=2)


def chebyshev(x, y):
    """
    Computes the Chebyshev distance between two 1-D arraysusing Minkowski
    Distance at order-p = ∞.

    Chebyshev distance in 2D can be loosly visualized as the number of moves it
    takes for a King to move from point A to point B in chess, as a King in chess
    can move horizontally, vertically, and diagonally. This explination assumes
    that we measure the distance between the centers of the squares, the squares
    have side length one, and the axes are aligned to the edges of the board.

    Parameters
    ---------------
    x : (N,) array_like
        Input array.
    y : (N,) array_like
        Input array.

    Returns
    ---------------
    The Chebyshev distance between vectors 'x' and 'y'.

    KaTeX Formula
    ---------------
    d(x,y) =  \|x-y\|_∞ = \lim\limits_{∞}(\displaystyle\sum_{i=1}^n|x_i-y_i|^∞)^\frac{1}{∞}=  \max_{i=1}^n|x_i-y_i|

    Alterntive Function
    ---------------
    np.max(np.absolute(x-y))

    Examples
    ---------------
    >>> x = np.array([6,14,-10])
    >>> y = np.array([-8,12,3])
    >>> chebyshev(x, y)
    14
    """
    return minkowski(x, y, p=float('inf'))