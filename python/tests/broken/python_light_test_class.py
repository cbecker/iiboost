###################################################################################
# Test for the IIBoost wrapper class
###################################################################################

from iiboost import Booster, computeIntegralImage
from sklearn.externals import joblib	# to load data

import numpy as np

# to show something
import matplotlib.pyplot as plt


# load data
gt = joblib.load("../../testData/gt.jlb")
img = joblib.load("../../testData/img.jlb")

model = Booster()

imgFloat = np.float32(img)
iiImage = computeIntegralImage( imgFloat )

# Train: note that we pass a list of stacks
model.trainWithChannel( [img], [gt], [iiImage], numStumps=100, debugOutput=True)

imgFloat = np.float32(img)
iiImage = computeIntegralImage( imgFloat )

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
