###################################################################################
# Test for the IIBoost wrapper class
###################################################################################

from IIBoost import Booster
from sklearn.externals import joblib	# to load data

import numpy as np

# to show something
import matplotlib.pyplot as plt

# load data
gt = joblib.load("gt.jlb")
img = joblib.load("img.jlb")

# let's pretend we have 3 image stacks with different number of ROIs
# with its corresponding gt and 2 feature channels

img3 = img2 = img1 = img
gt3  =  gt2 =  gt1 = gt

model = Booster()

imgFloat = np.float32(img)
iiImage = model.computeIntegralImage( imgFloat )

# again, this is stupid, just presume the second channel is a different feature
channel1 = iiImage
channel2 = iiImage
channels3 = channels2 = channels1 = [channel1,channel2]

zAnisotropyFactor = 5.0;

# Train: note that we pass a list of stacks
model.trainWithChannels( [img1,img2,img3], [gt1,gt2,gt3], [channels1,channels2,channels3], zAnisotropyFactor, numStumps=100, gtNegativeLabel=1, gtPositiveLabel=2, debugOutput=True)

pred = model.predictWithChannels( img, channels1, zAnisotropyFactor )

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

ss = model.serialize()
model.deserialize( ss )

#print "Serialization string: " + model.serialize()
