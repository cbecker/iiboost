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

model = Booster()

imgFloat = np.float32(img)
iiImage = model.computeIntegralImage( imgFloat )
#this is stupid, just presume the second iiImage is a different feature
channel1 = [iiImage, iiImage, iiImage]
channel2 = [iiImage, iiImage, iiImage]

# Train: note that we pass a list of stacks
model.trainWithChannels( [img,img,img], [gt,gt,gt], [[iiImage, iiImage, iiImage],[iiImage, iiImage, iiImage]], numStumps=5, debugOutput=True)
#model.trainWithChannels( [img], [gt], [[iiImage]], numStumps=5, debugOutput=True)
#model.trainWithChannel( [img], [gt], [iiImage], numStumps=5, debugOutput=True)
#model.train( [img], [gt], numStumps=5, debugOutput=True)

imgFloat = np.float32(img)
iiImage = model.computeIntegralImage( imgFloat )

#this is stupid, just presume the second iiImage is a different feature
pred = model.predictWithChannels( img, [iiImage] )

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
