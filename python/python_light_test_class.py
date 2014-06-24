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

# Train: note that we pass a list of stacks
model.trainWithChannels( [img], [gt], [iiImage], numStumps=100, debugOutput=True)

# again, that doesn't make sense either, just to test
imgFloat = np.float32(img)
iiImage = model.computeIntegralImage( imgFloat )

pred = model.predictWithChannel( img, iiImage )

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
