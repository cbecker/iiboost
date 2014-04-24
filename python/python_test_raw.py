###################################################################################
# Simple test script, doesn't make use of the python wrapper class, just raw ctypes
#  Only use this to test basic functionality, otherwise use python_test_class.py
###################################################################################

import numpy as np
from sklearn.externals import joblib
import ctypes

# load data
print "--- Loading data ---"
gt = joblib.load("gt.jlb")
img = joblib.load("img.jlb")

print "--- Loading lib ---"
boostLib = ctypes.CDLL("libiiboost_python.so")

print "--- Calling train() ---"
width = gt.shape[2]
height = gt.shape[1]
depth = gt.shape[0]

model = ctypes.c_void_p(
	boostLib.train( ctypes.c_void_p(img.ctypes.data), ctypes.c_void_p(gt.ctypes.data), ctypes.c_int(width), ctypes.c_int(height), ctypes.c_int(depth) ) )

# pre-alloc prediction
pred = np.empty_like( img, dtype=np.dtype("float32") )

boostLib.predict( model, 
				  ctypes.c_void_p(img.ctypes.data), 
				  ctypes.c_int(width), ctypes.c_int(height), ctypes.c_int(depth),
				  ctypes.c_void_p(pred.ctypes.data) )

