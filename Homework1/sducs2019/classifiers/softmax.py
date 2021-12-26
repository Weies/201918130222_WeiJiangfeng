from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # s = np.exp(X.dot(W))
    # sm = np.sum(s, axis=1)
    # for i in range(X.shape[0]):
    #     loss += -(np.log(s[i, y[i]] / sm[i]))
    # loss = loss / X.shape[0] + np.sum(W * W) * reg
    #
    # dW += 2 * reg * W

    scores = X.dot(W)
    scores = scores - np.max(scores, axis=1)[:, np.newaxis]
    scores = np.exp(scores)
    scores /= np.sum(scores, axis=1)[:, np.newaxis]

    for i in range(X.shape[0]):
        loss += -np.log(scores[i][y[i]])
        for j in range(dW.shape[1]):
            dW[:, j] += (scores[i][j] - (j == y[i])) * X[i]

    loss = loss / X.shape[0] + np.sum(W * W) * reg
    dW = dW / X.shape[0] + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # s = np.exp(X.dot(W))
    # sm = np.sum(s, axis=1)
    # ran = np.arange(X.shape[0])
    # a = s[ran, y[ran]]
    # loss = np.sum(-np.log(a / sm)) / X.shape[0] + np.sum(W * W) * reg
    #
    #
    #
    #
    # dW += 2 * reg * W

    scores = X.dot(W)
    scores = scores - np.max(scores, axis=1)[:, np.newaxis]
    scores = np.exp(scores)
    scores /= np.sum(scores, axis=1)[:, np.newaxis]
    loss = np.sum(-np.log(scores[np.arange(scores.shape[0]), y]))

    dW = dW.T

    scores_temp = np.zeros(scores.shape)
    for i in range(scores.shape[1]):  # record the cladd lable
        scores_temp[:, i] = i

    scores_temp = [y[:, np.newaxis].flatten() == scores_temp[:, i]
                   for i in range(W.shape[1])]

    scores_temp = np.array(scores_temp)
    scores_temp = scores_temp.T

    dW = ((scores - (scores_temp)).T).dot(X)
    dW = dW.T

    loss /= X.shape[0]
    dW /= X.shape[0]
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
