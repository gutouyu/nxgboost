# -*- coding: utf-8 -*-
import sys
import numpy as np
sys.path.append('..')
from nxgboost.loss import CustomiseLoss, LogisticLoss

preds = np.array([1,2,3,4,5])
y = np.array([2,3,4,5,6])
square_loss = lambda preds, y: 0.5*(preds-y)**2
custom_square_loss = CustomiseLoss(square_loss)
for (pred, label) in zip(preds, y):
    print custom_square_loss.grad(pred, label) # preds-labels
    print custom_square_loss.hess(pred, label) # 1

logistic_loss_obj = LogisticLoss()
print logistic_loss_obj.grad(preds, y)
print logistic_loss_obj.hess(preds, y)

from autograd import elementwise_grad, grad
# y = 3*x*x + 3
func = lambda x: 3.0*x*x + 3.0
grad_func = grad(func)
x = 10.0
print grad_func(x) # 参数个数和原函数一样；计算原函数在x处的导数 dy/dx|x=10 60

x = np.array([10.0, 5.0, 1.0])
grad_func2 = elementwise_grad(func)
print grad_func2(x) # vector版本


