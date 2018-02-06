from tree_node import TreeNode
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
import copyreg
import types
from time import time

# use copyreg to make the instance method picklable
# because multiprocessing must pickle things to sling them among process
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copyreg.pickle(types.MethodType, _pickle_method)

class Tree(object):
    def __init__(self,
                 min_sample_split,
                 min_child_weight,
                 max_depth,
                 colsample,
                 rowsample,
                 reg_lambda,
                 gamma,
                 num_thread):
        self.root = None
        self.min_sample_split = min_sample_split
        self.min_child_weight = min_child_weight
        self.max_depth = max_depth
        self.colsample = colsample
        self.rowsampel = rowsample
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.alive_nodes = []
        self.name_to_nodes = {}
        # number of tree node of this tree
        self.nodes_cnt = 0
        # number of nan tree node of this tree
        # nan tree node is the third child of the tree node
        self.nan_nodes_cnt = 0

        if num_thread == -1:
            self.num_thread = cpu_count()
        else:
            self.num_thread = num_thread

        # to avoid divide zero
        self.reg_lambda = max(self.reg_lambda, 0.00001)

    def calculate_leaf_score(self, G, H):
        """
        According to xgboost, the leaf score is ; - G / (H+lambda)
        """
        return - G / (H + self.reg_lambda)

    def calculate_split_gain(self, G_left, H_left, G_nan, H_nan, G_total, H_total):
        """
        According to xgboost, the scoring function is:
            gain = 0.5 * (GL^2/(HL+lambda) + GR^2/(HR+lambda) - (GL+GR)^2/(GL+GR+lambda)) - gamma

        this gain is the loss reduction, We want it to be as large as possible.

        """
        G_right = G_total - G_left
        H_right = H_total - H_left

        # if we let those with missing value go to a nan child
        gain_1 = 0.5 * (G_left**2/(H_left+self.reg_lambda)
                    +   G_right**2/(H_right+self.reg_lambda)
                    +   G_nan**2/(H_nan+self.reg_lambda)
                    -   G_total**2/(H_total+self.reg_lambda)) - 2 * self.gamma

        # if we let those with missing value go to left child
        gain_2 = 0.5 * ((G_left+G_nan)**2/(H_left+H_nan+self.reg_lambda)
                    +   G_right**2/(H_right+self.reg_lambda)
                    -   G_total**2/(H_total+self.reg_lambda)) - self.gamma

        # if we let those with missing value go to right child
        gain_3 = 0.5 * (G_left**2/(H_left+self.reg_lambda)
                    +   (G_right+G_nan)**2/(H_right+H_nan+self.reg_lambda)
                    -   G_total**2)/(H_total+self.reg_lambda) - self.gamma

        nan_go_to = None
        gain = None
        if gain_1 == max(gain_1, gain_2, gain_3):
            nan_go_to = 0 # nan child
            gain = gain_1
        elif gain_2 == max(gain_1, gain_2, gain_3):
            nan_go_to = 1 # left child
            gain = gain_2
        else:
            nan_go_to = 2 # right child
            gain = gain_3

        # in this case, the trainset does not contains nan samples
        if H_nan == 0 and G_nan == 0:
            nan_go_to = 3

        return nan_go_to, gain

    def _process_one_attribute_list(self, class_list, (col_attribute_list, col_attribute_list_cutting_index, col)):
        """
        this function is base for parallel using multiprocessing,
        so all operation are read-only

        """
        ret = []
        # linear scan this column's attribute list, bin by bin
        for uint8_threshold in range(len(col_attribute_list_cutting_index) - 1):
            start_ind = col_attribute_list_cutting_index[uint8_threshold]
            end_ind = col_attribute_list_cutting_index[uint8_threshold + 1]
            inds = col_attribute_list["index"][start_ind:end_ind]
            tree_node_G_H = class_list.statistic_given_inds(inds)
            ret.append((col, uint8_threshold, tree_node_G_H))
        return ret

    def build(self, attribute_list, class_list, col_sampler, bin_structure):
        raise NotImplementedError()

    def fit(self, attribute_list, class_list, row_sampler, col_sampler, bin_structure):
        raise NotImplementedError()

