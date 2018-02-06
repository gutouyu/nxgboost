# nxgboost

# 介绍

# 文件
* `loss.py` 
    * 支持自定义损失函数
    * 默认提供SquareLoss,LogisticLoss

* `metirc.py`
    * accuracy
    * error
    * mean_square_error
    * mean_absolute_error
    * auc

* `tree_node.py`
    * 定义一棵树中的一个节点，包括叶节点还有内部节点
    * 根节点编号1，missing_value可能分配到左右子树中，也可能单独的分配到一个分支中
    * 根节点depth=1
