import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from nxgboost.binning import BinStructure

"""
features = np.random.random((1000000, 50))
feature_dim = features.shape[1]
print features[0]

bs = BinStructure(features, max_bin=10, estimation_sampling=1000000)
print "size of bs:", sys.getsizeof(bs) # 64

bs1 = BinStructure(features, max_bin=10, estimation_sampling=100000)
print "size of bs1:", sys.getsizeof(bs1)

print bs[0]
print bs1[0]
"""

train = pd.read_csv("../data/train2.csv")
print train.head()
bs = BinStructure(train.values)
for i in range(train.shape[1]):
    print "=" * 20
    print bs[i]

