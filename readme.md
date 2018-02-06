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

* `tree.py`
    * 计算 weights for each leaf
    * calculate splits gain for one feature。对于missing value, 尝试三种策略：left,right,单独一个分支。选取三种分法中gain最大的一种。
    * 未完待续

* `binning.py`
    * 根据设置的max_bins得到每一个feature的upper bounder
    * 按照分位数的思想来划分的
    * 此处实现：global的方式事先划分； 没有考虑hess权重
    * 流程：针对每一个feature col都进行如下操作：
    （1）过滤掉对应feature为nan的样本。
    （2）从中进行行抽样。
    （3）统计value_counts。如果distinct_value小于max_bin，那么实际的bin退化为distinct_value,并以这些值作为bin的上限。 
     注意，这种时候可能导致每个bin中样本数量不均衡。
    （4）根据max_bin和value_counts，`均匀的填充bin(类似于选取分位数)`。要保证相同取值的样本在同一个bin中，所以bin中样本的数量可能也不是严格相等的。

* `sampling.py`
    * 按行采样
    * 按列采样

