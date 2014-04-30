###################################################################################
# Simple test script, doesn't make use of the python wrapper class, just raw ctypes
#  Only use this to test basic functionality, otherwise use python_test_class.py
###################################################################################

import numpy as np
from sklearn.externals import joblib
import ctypes

# gets 'prop' from every element in L
#  and puts it in an array of element type cArrayElType
def propToCArray( L, prop, cArrayElType ):
	N = len(L)
	arr = (cArrayElType * N)()

	for idx,e in enumerate(L):
		arr[idx] = cArrayElType( eval( "e." + prop ) )

	return arr


# load data
print "--- Loading data ---"
gts = [joblib.load("gt.jlb")]
imgs = [joblib.load("img.jlb")]


print "--- Loading lib ---"
boostLib = ctypes.CDLL("libiiboost_python.so")

# this returns a python string
boostLib.serializeModel.restype = ctypes.py_object


print "--- Calling train() ---"

# we need to pass an array to the C call
widthList =  propToCArray( imgs, "shape[2]", ctypes.c_int)
heightList =  propToCArray( imgs, "shape[1]", ctypes.c_int)
depthList =  propToCArray( imgs, "shape[0]", ctypes.c_int)

# list of img and gt (pointers)
imgList = propToCArray( imgs, "ctypes.data", ctypes.c_void_p )
gtList = propToCArray( gts, "ctypes.data", ctypes.c_void_p )

debugOutput = 1
numStumps = 10

model = ctypes.c_void_p(
 	boostLib.train( imgList, gtList,
 					widthList, heightList, depthList,
 					ctypes.c_int(1),
 					ctypes.c_int(numStumps),
 					ctypes.c_int(debugOutput) ) )

serStr = ctypes.py_object( boostLib.serializeModel( model ) )

# pre-alloc prediction
pred = np.empty_like( imgs[0], dtype=np.dtype("float32") )

boostLib.predict( model, 
				  ctypes.c_void_p(imgs[0].ctypes.data), 
				  widthList[0], heightList[0], depthList[0],
				  ctypes.c_void_p(pred.ctypes.data) )

