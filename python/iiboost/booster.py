##////////////////////////////////////////////////////////////////////////////////
## Copyright (c) 2013 Carlos Becker                                             ##
## Ecole Polytechnique Federale de Lausanne                                     ##
## Contact <carlos.becker@epfl.ch> for comments & bug reports                   ##
##                                                                              ##
## This program is free software: you can redistribute it and/or modify         ##
## it under the terms of the version 3 of the GNU General Public License        ##
## as published by the Free Software Foundation.                                ##
##                                                                              ##
## This program is distributed in the hope that it will be useful, but          ##
## WITHOUT ANY WARRANTY; without even the implied warranty of                   ##
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU             ##
## General Public License for more details.                                     ##
##                                                                              ##
## You should have received a copy of the GNU General Public License            ##
## along with this program. If not, see <http://www.gnu.org/licenses/>.         ##
##////////////////////////////////////////////////////////////////////////////////

import os
import collections
import numpy as np
import ctypes

from exceptions import RuntimeError

# libiiboost_python.so must reside in the same directory as this module.
libName = os.path.join(os.path.split(__file__)[0], "libiiboost_python.so")
libPtr = ctypes.CDLL( libName )

def as_carray(l, carray_el_type):
	return (carray_el_type * len(l))(*l)

# gets 'prop' from every element in L
#  and puts it in an array of element type cArrayElType
def propToCArray( L, prop, cArrayElType ):
	N = len(L)
	arr = (cArrayElType * N)()

	for idx,e in enumerate(L):
		arr[idx] = cArrayElType(eval("e." + prop))

	return arr

def propListToCArray( L, prop, cArrayElType ):
	M = len(L)
	N = len(L[0])
	T = M * N
	arr = (cArrayElType * T)()

	for idm, m in enumerate(L):
		for idx,e in enumerate(m):
			arr[idm*N + idx] = cArrayElType(eval("e." + prop))

	return arr


## --- For orientation, computes 9 float values per pixel ---
class EigenVectorsOfHessianImage(np.ndarray):
    """
    A subclass of numpy.ndarray returned by computeEigenVectorsOfHessianImage()
    When the array is deleted, it will free the underlying buffer allocated from C++.
    """
    def __new__(cls, ptr_address, shape):
        ArrayType = ctypes.c_float*int(np.prod(shape))
        obj = np.frombuffer(ArrayType.from_address(ptr_address), dtype=np.float32).reshape(shape).view(cls)
        obj.imgPtr = ctypes.c_void_p(ptr_address)
        return obj

    def __array_finalize__(self, obj):
        self.imgPtr = None

    def __del__(self):
        if self.imgPtr:
            assert self.imgPtr != -1, "Double-delete detected!"
            libPtr.freeEigenVectorsOfHessianImage( self.imgPtr )
            self.imgPtr = -1		

# sigma is the smoothing factor, in pixels
libPtr.computeEigenVectorsOfHessianImage.restype = ctypes.c_void_p
def computeEigenVectorsOfHessianImage( imgStack, zAnisotropyFactor, sigma=3.5 ):
    """
    Computes an orientation matrix (3x3) per pixel.
    """
    if imgStack.dtype != np.dtype("uint8"):
        raise RuntimeError("image must be of uint8 type")

    if not imgStack.flags["C_CONTIGUOUS"]:
        raise RuntimeError("image must be C_CONTIGUOUS, and must be provided in z-y-x order.")

	if not imgStack.ndim != 3:
		raise RuntimeError("image must be 3D.")
	
	# 'mangle' dimensions to deal with storage order (assuming C-style)
    width = imgStack.shape[2]
    height = imgStack.shape[1]
    depth = imgStack.shape[0]

    ptr_address = libPtr.computeEigenVectorsOfHessianImage( 
					ctypes.c_void_p(imgStack.ctypes.data),
					ctypes.c_int(width), ctypes.c_int(height), ctypes.c_int(depth),
					ctypes.c_double(zAnisotropyFactor),
					ctypes.c_double(sigma) )

    return EigenVectorsOfHessianImage(ptr_address, imgStack.shape + (3,3))

def computeIntegralImage( input_image ):
	"""
	Compute the integral image for the given volume.
	Roughly equivalent to:
	
	output = input_image.copy()
    for i in range(input_image.ndim):
        np.add.accumulate(output, axis=i, out=output)
	"""
	if input_image.dtype != np.dtype("float32"):
		raise RuntimeError("image must be of float32 type")

	if not input_image.flags["C_CONTIGUOUS"]:
		raise RuntimeError("image must be C_CONTIGUOUS, and must be provided in z-y-x order.")

	# 'mangle' dimensions to deal with storage order (assuming C-style)
	width = input_image.shape[2]
	height = input_image.shape[1]
	depth = input_image.shape[0]

	# pre-alloc integral image
	integralImage = np.empty_like( input_image, dtype=np.dtype("float32") )


	libPtr.computeIntegralImage( ctypes.c_void_p(input_image.ctypes.data),
								 ctypes.c_int(width), 
								 ctypes.c_int(height), 
								 ctypes.c_int(depth),
								 ctypes.c_void_p(integralImage.ctypes.data) )

	return integralImage


## --- Booster Class ---
class Booster(object):
	""" Booster class based on context cue boosting """

	# will hold C pointer to model
	modelPtr = None

	# holds ptr to library (ctypes)
	libPtr = None

	def __init__(self):
		self.libPtr = ctypes.CDLL( libName )
		self.modelPtr = None

		self.libPtr.serializeModel.restype = ctypes.py_object
		self.libPtr.deserializeModel.restype = ctypes.c_void_p
		self.libPtr.train.restype = ctypes.c_void_p
		self.libPtr.trainWithChannel.restype = ctypes.c_void_p
		self.libPtr.trainWithChannels.restype = ctypes.c_void_p
		self.libPtr.wlpredictWithChannels.restype = ctypes.c_int
		self.libPtr.wlAlphas.restype = ctypes.py_object

	# returns a string representation of the model
	def serialize( self ):
		if self.modelPtr is None:
			raise RuntimeError("Tried to serialize(), but no model available.")

		return self.libPtr.serializeModel( self.modelPtr )

	# construct model from string representation
	def deserialize( self, modelString ):
		if not isinstance(modelString, basestring):
			raise RuntimeError("Tried to deserialize(), but modelString must be a string.")

		newModelPtr = ctypes.c_void_p( self.libPtr.deserializeModel( ctypes.c_char_p(modelString) ) )

		if newModelPtr.value is None:
			raise RuntimeError("Error deserializing.")

		self.freeModel()
		self.modelPtr = newModelPtr

	def __getstate__(self):
		return self.serialize()

	def __setstate__(self, state):
		self.__init__()
		self.deserialize(state)
	
	def number_of_weaklearners(self):
		if self.modelPtr is None:
			raise ValueError("No model available") 
		
		return self.libPtr.numberOfWeakLearners(self.modelPtr)
	
	def wl_alphas(self):
		if self.modelPtr is None:
			raise ValueError("No model available") 
		
		return np.array(self.libPtr.wlAlphas(self.modelPtr))

	def trainWithChannels( self, imgStackList, eigVecOfHessianImgList,
                           gtStackList, chStackListList, 
                           zAnisotropyFactor, numStumps, gtNegativeLabel, gtPositiveLabel, debugOutput = False ):
			""" Train a boosted classifier """
			"""   imgStackList: list of images, of type uint8 """
			"""   eigVecOfHessianImgList: list of hessian eigenvector list (orientation)                  """
			"""   gtStackList:  list of GT, of type uint8. Negative = 1, Positive = 2, Ignore = else      """
			"""	  chStackListList: list of the channels/integral images for each image in imgStackList.   """
			"""    				   The number of channels per image must be the same for all images. 	  """
			"""    				   Use computeIntegralImage() to calculate integral images from channels. """
			"""   zAnisotropyFactor: ratio between z voxel size and x/y voxel size """
			"""   numStumps:    integer, number of stumps to train 										  """
			"""   gtPositive/NegativeLabel:    value of positive and negative training data in ground truth """
			""" WARNING: it assumes stacks are in C ordering """
			for param in (imgStackList, gtStackList, chStackListList, eigVecOfHessianImgList):
				if not isinstance(param, collections.Iterable):
					raise RuntimeError("imgStackList, eigVecOfHessianImgList gtStackList, "
									   "and chStackListList must each be iterable (e.g. a list)")

			# check shape/type of img and gt
			if not (len(imgStackList) == len(gtStackList) == len(chStackListList) == len(eigVecOfHessianImgList)):
				raise RuntimeError("image, eig vec, gt stack and channels list must of be of same size,",
									len(imgStackList)," ",len(gtStackList)," ",len(chStackListList), " ",len(eigVecOfHessianImgList))

			for chStackList in chStackListList :
				if not isinstance(chStackList, collections.Iterable):
					raise RuntimeError("Channel stacks must be provided as a sequence.")
				if isinstance(chStackList, np.ndarray) and chStackList.ndim != 4:
					raise RuntimeError("Channel stacks must be provided as a sequence of 3D volumes.")

				if len(chStackList) != len(chStackListList[0]):
					raise RuntimeError("Number of channels for each image must be the same")

			for img,gt,chStackList,eigvec in zip(imgStackList, gtStackList, chStackListList, eigVecOfHessianImgList):
				for ch in chStackList:
					if img.shape != gt.shape or gt.shape != ch.shape:
						raise RuntimeError("image, ground truth and channels must be of same size,",img.shape," ",gt.shape," ",ch.shape)

					if (img.dtype != np.dtype("uint8")) or \
	                    (gt.dtype != np.dtype("uint8")) or \
	                    (ch.dtype != np.dtype("float32")):
						raise RuntimeError("image and ground truth must be of uint8 type and channels of float32 type")
					
					if 0 in img.shape or 0 in gt.shape or 0 in ch.shape or 0 in eigvec.shape:
						raise RuntimeError("One of the inputs has a zero shape.")

				if not eigvec.flags["C_CONTIGUOUS"]:
					raise RuntimeError("eigVecImg must be C-contiguous")
				if eigvec.shape != img.shape + (3,3):
					raise RuntimeError("eigVecImg shape is {}, which doesn't correspond to raw image shape: {}.".format( eigvec.shape, img.shape ))

			# 'mangle' dimensions to deal with storage order (assuming C-style)
			width  = propToCArray( imgStackList, "shape[2]", ctypes.c_int )
			height = propToCArray( imgStackList, "shape[1]", ctypes.c_int )
			depth  = propToCArray( imgStackList, "shape[0]", ctypes.c_int )

			# C array of pointers
			imgs  = propToCArray( imgStackList, "ctypes.data", ctypes.c_void_p )
			gts   = propToCArray(  gtStackList, "ctypes.data", ctypes.c_void_p )
			evecs = propToCArray(  eigVecOfHessianImgList, "ctypes.data", ctypes.c_void_p )

			chans = propListToCArray( chStackListList, "ctypes.data", ctypes.c_void_p )

			numStacks = len(chStackListList)
			numChannels = len(chStackListList[0])
			print "Number of stacks: ",numStacks,". Each with ",numChannels," channels."

			if debugOutput:
				dbgOut = ctypes.c_int(1)
			else:
				dbgOut = ctypes.c_int(0)

			newModelPtr = ctypes.c_void_p(
								self.libPtr.trainWithChannels(
											imgs, evecs, gts,
											width, height, depth,
											ctypes.c_int( len(imgs) ),
											chans,
											ctypes.c_int( numChannels ), ctypes.c_double(zAnisotropyFactor),
											ctypes.c_int(numStumps), ctypes.c_int( gtNegativeLabel ), ctypes.c_int( gtPositiveLabel ), dbgOut ) )

			if newModelPtr.value == None:
				raise RuntimeError("Error training model.")

			self.freeModel()
			self.modelPtr = newModelPtr


	def predictWithChannels( self, imgStack, eigVecImg, chStackList, zAnisotropyFactor, useEarlyStopping = True):
		""" Per-pixel prediction for single ROI/image 	"""
		"""   imgStack: 		 image itself 					 """
		"""   eigVecImg:		 an EigenVectorsOfHessianImage instance """
		"""   chStackList:  	 list of integral images/channels """
		"""   zAnisotropyFactor: ratio between z voxel size and x/y voxel size """
		"""   useEarlyStopping:  speeds up prediction considerably by approximating the prediction score """
		
		self.__check_predict_params(imgStack, eigVecImg, chStackList)

		# C array of pointers
		chans = propToCArray(  chStackList, "ctypes.data", ctypes.c_void_p )

		# 'mangle' dimensions to deal with storage order (assuming C-style)
		width  = imgStack.shape[2]
		height = imgStack.shape[1]
		depth  = imgStack.shape[0]

		# pre-alloc prediction
		pred = np.empty_like(imgStack, dtype=np.float32)

		if useEarlyStopping:
				useEarlyStoppa = ctypes.c_int(1)
		else:
				useEarlyStoppa = ctypes.c_int(0)

		# Run prediction
		ret = self.libPtr.predictWithChannels( self.modelPtr,
				ctypes.c_void_p(imgStack.ctypes.data),
				ctypes.c_void_p(eigVecImg.ctypes.data),
				ctypes.c_int(width), ctypes.c_int(height), ctypes.c_int(depth),
				chans,
				ctypes.c_int( len(chans) ), ctypes.c_double(zAnisotropyFactor),
				useEarlyStoppa,
				ctypes.c_void_p(pred.ctypes.data) )

		ret = ctypes.c_int(ret)
		
		if ret.value != 0:
			raise RuntimeError("Error during prediction, see above.");

		return pred

	def wlpredictWithChannels(self, imgStack, eigVecImg, chStackList, zAnisotropyFactor):
		"""
		Per-pixel predictions for every weak learner.
		
		Parameters are the same than those in predictWithChannels.
		"""
		
		# Check parameters
		self.__check_predict_params(imgStack, eigVecImg, chStackList)
		
		# C array of pointers
		chans = as_carray([i.ctypes.data for i in chStackList], ctypes.c_void_p)

		# 'mangle' dimensions to deal with storage order (assuming C-style)
		width  = imgStack.shape[2]
		height = imgStack.shape[1]
		depth  = imgStack.shape[0]
		
		num_wl = self.number_of_weaklearners()
		
		# pre-alloc prediction
		pred = np.empty((num_wl,) + imgStack.shape, dtype=np.int8)
		pred_c = as_carray([i.ctypes.data for i in pred], ctypes.c_void_p)
		
		ret = self.libPtr.wlpredictWithChannels(self.modelPtr,
				ctypes.c_void_p(imgStack.ctypes.data),
				ctypes.c_void_p(eigVecImg.ctypes.data),
				ctypes.c_int(width), ctypes.c_int(height), ctypes.c_int(depth),
				chans,
				ctypes.c_int( len(chans) ),
				ctypes.c_double(zAnisotropyFactor),
				pred_c
			)
		
		ret = ctypes.c_int(ret)
		
		if ret.value != 0:
			raise RuntimeError("Error during prediction, see above.");
		
		return pred
	
	def __check_predict_params(self, imgStack, eigVecImg, chStackList):
		if self.modelPtr == None:
			raise RuntimeError("Tried to predict(), but no model available.")

		""" returns confidence stack of pixel type float """
		if imgStack.dtype != np.dtype("uint8"):
			raise ValueError("image must be of uint8 type")

		if not imgStack.flags["C_CONTIGUOUS"]:
			raise ValueError("image must be C_CONTIGUOUS, and must be provided in z-y-x order.")

		if not eigVecImg.flags["C_CONTIGUOUS"] or eigVecImg.dtype != np.float32:
			raise ValueError("eigVecImg must be a contiguous float32 array, provided in z-y-x order.")
		
		if eigVecImg.shape != imgStack.shape + (3,3):
			raise RuntimerError("eigVecImg has unexpected shape: {} for raw image of shape: {}".format( eigVecImg.shape, imgStack.shape ))

		if not isinstance(chStackList, collections.Iterable):
			raise ValueError("Channel stack must be provided as a sequence.")
		if isinstance(chStackList, np.ndarray) and chStackList.ndim != 4:
			raise ValueError("Channel stack must be provided as a sequence of 3D volumes.")

		# check shape/type of img and gt
		for ch in chStackList:
			if not imgStack.flags["C_CONTIGUOUS"]:
				raise ValueError("channels must be C_CONTIGUOUS, and must be provided in z-y-x order.")

			if imgStack.shape != ch.shape :
				raise ValueError("image and channels must be of same size,",imgStack.shape," ",ch.shape)

			if (ch.dtype != np.dtype("float32")):
				raise ValueError("channels must be of loat32 type")

	# if model is not null, free it
	def freeModel(self):
		if self.modelPtr is not None:
			self.libPtr.freeModel( self.modelPtr )
			self.modelPtr = None

	# we need a proper destructor to delete the C pointer
	# (because we love hacking code and dirty pointers)
	def __del__(self):
		self.freeModel()




	###### ---------- BEGIN OLD FUNCTIONALITY ---------------- #####

	def train( self, imgStackList, gtStackList, numStumps, debugOutput = False ):
		
		raise RuntimeError("You shouldn't be calling this function, this is old functionality")

		""" Train a boosted classifier """
		"""   imgStackList: list of images, of type uint8 """
		"""   gtStackList:  list of GT, of type uint8. Negative = 1, Positive = 2, Ignore = else """
		"""   numStumps:    integer """
		""" WARNING: it assumes stacks are in C ordering """

		if (type(imgStackList) != list) or (type(gtStackList) != list):
			raise RuntimeError("image and gt stack list must of be of type LIST")

		# check shape/type of img and gt
		if len(imgStackList) != len(gtStackList):
			raise RuntimeError("image and gt stack list must of be of same size")

		for img,gt in zip(imgStackList, gtStackList):
			if img.shape != gt.shape:
				raise RuntimeError("image and ground truth must be of same size")

			if (img.dtype != np.dtype("uint8")) or (gt.dtype != np.dtype("uint8")):
				raise RuntimeError("image and ground truth must be of uint8 type")

		# 'mangle' dimensions to deal with storage order (assuming C-style)
		width = propToCArray( imgStackList, "shape[2]", ctypes.c_int )
		height = propToCArray( imgStackList, "shape[1]", ctypes.c_int )
		depth = propToCArray( imgStackList, "shape[0]", ctypes.c_int )

		# C array of pointers
		imgs = propToCArray( imgStackList, "ctypes.data", ctypes.c_void_p )
		gts = propToCArray(  gtStackList,  "ctypes.data", ctypes.c_void_p )

		if debugOutput:
			dbgOut = ctypes.c_int(1)
		else:
			dbgOut = ctypes.c_int(0)

		newModelPtr = ctypes.c_void_p(
							self.libPtr.train(
										imgs, gts,
										width, height, depth,
										ctypes.c_int( len(imgs) ),
										ctypes.c_int(numStumps), dbgOut ) )

		if newModelPtr.value == None:
			raise RuntimeError("Error training model.")

		self.freeModel()
		self.modelPtr = newModelPtr

	def trainWithChannel( self, imgStackList, gtStackList, chStackList, numStumps, debugOutput = False ):
			
		raise RuntimeError("You shouldn't be calling this function, this is old functionality")

		if (type(imgStackList) != list) or (type(gtStackList) != list) or (type(chStackList) != list):
			raise RuntimeError("image, gt and channels stack list must of be of type LIST")

		# check shape/type of img and gt
		if len(imgStackList) != len(gtStackList) or len(gtStackList) != len(chStackList):
			raise RuntimeError("image, gt stack and channels list must of be of same size,",
													len(imgStackList)," ",len(gtStackList)," ",len(chStackList))

		for img,gt,ch in zip(imgStackList, gtStackList, chStackList):
			if img.shape != gt.shape or gt.shape != ch.shape:
				raise RuntimeError("image, ground truth and channels must be of same size,",img.shape," ",gt.shape," ",ch.shape)

			if (img.dtype != np.dtype("uint8")) or (gt.dtype != np.dtype("uint8")) or (ch.dtype != np.dtype("float32")):
				raise RuntimeError("image and ground truth must be of uint8 type and channels of float32 type")

		# 'mangle' dimensions to deal with storage order (assuming C-style)
		width  = propToCArray( imgStackList, "shape[2]", ctypes.c_int )
		height = propToCArray( imgStackList, "shape[1]", ctypes.c_int )
		depth  = propToCArray( imgStackList, "shape[0]", ctypes.c_int )

		# C array of pointers
		imgs  = propToCArray( imgStackList, "ctypes.data", ctypes.c_void_p )
		gts   = propToCArray(  gtStackList, "ctypes.data", ctypes.c_void_p )
		chans = propToCArray(  chStackList, "ctypes.data", ctypes.c_void_p )

		if debugOutput:
			dbgOut = ctypes.c_int(1)
		else:
			dbgOut = ctypes.c_int(0)

		newModelPtr = ctypes.c_void_p(
							self.libPtr.trainWithChannel(
										imgs, gts, chans,
										width, height, depth,
										ctypes.c_int( len(imgs) ),
										ctypes.c_int(numStumps), dbgOut ) )

		if newModelPtr.value == None:
			raise RuntimeError("Error training model.")

		self.freeModel()
		self.modelPtr = newModelPtr


	def predict( self, imgStack ):

		raise RuntimeError("You shouldn't be calling this function, this is old functionality")

		if self.modelPtr == None:
			raise RuntimeError("Tried to predict(), but no model available.")

		""" returns confidence stack of pixel type float """
		if imgStack.dtype != np.dtype("uint8"):
			raise RuntimeError("image must be of uint8 type")

		# 'mangle' dimensions to deal with storage order (assuming C-style)
		width  = imgStack.shape[2]
		height = imgStack.shape[1]
		depth  = imgStack.shape[0]

		# pre-alloc prediction
		pred = np.empty_like( imgStack, dtype=np.dtype("float32") )

		# Run prediction

		self.libPtr.predict( self.modelPtr,
								ctypes.c_void_p(imgStack.ctypes.data),
								ctypes.c_int(width), ctypes.c_int(height), ctypes.c_int(depth),
								ctypes.c_void_p(pred.ctypes.data) )

		return pred

	def predictWithChannel( self, imgStack, chStack ):

		raise RuntimeError("You shouldn't be calling this function, this is old functionality")

		if self.modelPtr == None:
			raise RuntimeError("Tried to predict(), but no model available.")

		""" returns confidence stack of pixel type float """
		if imgStack.dtype != np.dtype("uint8"):
			raise RuntimeError("image must be of uint8 type")

		if imgStack.shape != chStack.shape:
			raise RuntimeError("image and channels must be of same size (",imgStack.shape,"!=",chStack.shape)

		if chStack.dtype != np.dtype("float32"):
			raise RuntimeError("Channel must be of float32 type")

		# 'mangle' dimensions to deal with storage order (assuming C-style)
		width  = imgStack.shape[2]
		height = imgStack.shape[1]
		depth  = imgStack.shape[0]

		# pre-alloc prediction
		pred = np.empty_like( imgStack, dtype=np.dtype("float32") )

		# Run prediction
		self.libPtr.predictWithChannel( self.modelPtr,
				ctypes.c_void_p(imgStack.ctypes.data),
				ctypes.c_void_p(chStack.ctypes.data),
				ctypes.c_int(width), ctypes.c_int(height), ctypes.c_int(depth),
				ctypes.c_void_p(pred.ctypes.data) )

		return pred
