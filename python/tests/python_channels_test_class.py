###################################################################################
# Test for the IIBoost wrapper class
###################################################################################

from iiboost import Booster, EigenVectorsOfHessianImage, computeEigenVectorsOfHessianImage, computeIntegralImage
from sklearn.externals import joblib	# to load data

import numpy as np

# to show something
import matplotlib.pyplot as plt

# load data
gt = joblib.load("../../testData/gt.jlb")
img = joblib.load("../../testData/img.jlb")

# let's pretend we have 3 image stacks with different number of ROIs
# with its corresponding gt and 2 feature channels

img3 = img2 = img1 = img
gt3  =  gt2 =  gt1 = gt

model = Booster()

imgFloat = np.float32(img)
iiImage = computeIntegralImage( imgFloat )

# again, this is stupid, just presume the second channel is a different feature
channel1 = iiImage
channel2 = iiImage
channels3 = channels2 = channels1 = [channel1,channel2]

# anisotropy factor is the ratio between z voxel size and x/y voxel size.
# if Isotropic -> 1.0
zAnisotropyFactor = 1.0;

# this is typically a good value, but it depends on the voxel size of the data
hessianSigma = 3.5

eigV1 = computeEigenVectorsOfHessianImage( img1, zAnisotropyFactor, hessianSigma )
eigV2 = computeEigenVectorsOfHessianImage( img2, zAnisotropyFactor, hessianSigma )
eigV3 = computeEigenVectorsOfHessianImage( img3, zAnisotropyFactor, hessianSigma )

# Train: note that we pass a list of stacks
model.trainWithChannels( [img1,img2,img3], [eigV1, eigV2, eigV3], [gt1,gt2,gt3], [channels1,channels2,channels3], 
                         zAnisotropyFactor, numStumps=100, gtNegativeLabel=1, gtPositiveLabel=2, debugOutput=True)

pred = model.predictWithChannels( img, eigV1, channels1, zAnisotropyFactor, useEarlyStopping=True)

# show image & prediction side by side
plt.ion()
plt.figure()

plt.subplot(1,2,1)
plt.imshow(img[:,:,10],cmap="gray")
plt.title("Click on the image to exit")

plt.subplot(1,2,2)
plt.imshow(pred[:,:,10],cmap="gray")
plt.title("Click on the image to exit")

plt.ginput(1)

print "Serializing..."
ss = model.serialize()
print "Deserializing..."
model.deserialize( ss )
print "DONE."
#print "Serialization string: " + model.serialize()
