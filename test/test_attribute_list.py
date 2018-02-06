import sys
sys.path.append("..")
import numpy as np
from nxgboost.binning import BinStructure
from nxgboost.attribute_list import AttributeList
np.random.seed(2017)
features = np.random.randint(1000, size=(10, 1))
print features.nbytes / 1
feature_dim = features.shape[1]

bin_structure = BinStructure(features)
attribute_list = AttributeList(features, bin_structure)

print "features", features
print "BinStructure", bin_structure[0]
for i in range(feature_dim):
    print "attribute_list[i]", attribute_list[i]
    print "index", attribute_list[i]["index"]
    print "attri", attribute_list[i]["attribute"]
    print type(attribute_list[i])
    print attribute_list.attribute_list_cutting_index[i]
