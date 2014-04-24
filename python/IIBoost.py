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

import numpy as np
import ctypes

class Booster:
	""" Booster class based on context cue boosting """

	libName = "libiiboost_python.so"

	# will hold C pointer to model
	modelPtr = None

	# holds ptr to library (ctypes)
	libPtr = None


	def __init__(self):
		self.libPtr = ctypes.CDLL( self.libName )
		self.modelPtr = None

	def train( self, imgStack, gtStack, numStumps, debugOutput = False ):
		""" Train requires the image itself (uint8) and ground truth stack (uint8) """
		if imgStack.shape != gtStack.shape:
			raise RuntimeError("image and ground truth must be of same size")

		if (imgStack.dtype != np.dtype("uint8")) or (gtStack.dtype != np.dtype("uint8")):
			raise RuntimeError("image and ground truth must be of uint8 type")

		# 'mangle' dimensions to deal with storage order (assuming C-style)
		width = imgStack.shape[2]
		height = imgStack.shape[1]
		depth = imgStack.shape[0]

		if debugOutput:
			dbgOut = ctypes.c_int(1)
		else:
			dbgOut = ctypes.c_int(0)

		self.modelPtr = ctypes.c_void_p(
						self.libPtr.train( ctypes.c_void_p(imgStack.ctypes.data), 
										ctypes.c_void_p(gtStack.ctypes.data), 
										ctypes.c_int(width), ctypes.c_int(height), ctypes.c_int(depth),
										ctypes.c_int(numStumps), dbgOut ) )

	def predict( self, imgStack ):
		""" returns confidence stack of pixel type float """
		if imgStack.dtype != np.dtype("uint8"):
			raise RuntimeError("image must be of uint8 type")

		# 'mangle' dimensions to deal with storage order (assuming C-style)
		width = imgStack.shape[2]
		height = imgStack.shape[1]
		depth = imgStack.shape[0]

		# pre-alloc prediction
		pred = np.empty_like( imgStack, dtype=np.dtype("float32") )

		self.libPtr.predict( self.modelPtr, 
				  ctypes.c_void_p(imgStack.ctypes.data), 
				  ctypes.c_int(width), ctypes.c_int(height), ctypes.c_int(depth),
				  ctypes.c_void_p(pred.ctypes.data) )

		return pred


	# we need a proper destructor to delete the C pointer
	# (because we love hacking code and dirty pointers)
	def __del__(self):
		if self.modelPtr != None:
			self.libPtr.freeModel( self.modelPtr )
			self.modelPtr = None
