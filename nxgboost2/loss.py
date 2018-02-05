# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
from autograd import elementwise_grad

elementwise_hess = lambda func: elementwise_grad(elementwise_grad(func))

class BaseLoss(object):
    def __init__(self):
        pass

    def grad(self, preds, labels):
        raise NotImplementedError()

    def hess(self, preds, labels):
        raise NotImplementedError()

class CustomiseLoss(BaseLoss):
    def __init__(self, loss):
        super(CustomiseLoss, self).__init__()
        self.loss = loss

    def grad(self, preds, labels):
        preds = self.transform(preds)
        return elementwise_grad(self.loss)(preds, labels)

    def hess(self, preds, labels):
        preds = self.transform(preds)
        return elementwise_hess(self.loss)(preds, labels)

    def transform(self, preds):
        return 1.0/(1.0+np.exp(-preds))

class SquareLoss(BaseLoss):
    def transform(self, preds):
        return preds

    def grad(self, preds, labels):
        return preds - labels

    def hess(self, preds, labels):
        return np.ones_like(labels)

class LogisticLoss(BaseLoss):
    def transform(self, preds):
        return 1.0/(1.0 + np.exp(-preds))

    def grad(self, preds, labels):
        raise NotImplementedError()

    def hess(self, preds, labels):
        raise NotImplementedError()