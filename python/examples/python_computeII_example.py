###################################################################################
# Test for the IIBoost wrapper class
###################################################################################

from IIBoost import computeIntegralImage
from sklearn.externals import joblib	# to load data

import numpy as np

# to show something
import matplotlib.pyplot as plt


# load data
gt = joblib.load("gt.jlb")
img = joblib.load("img.jlb")

imgFloat = np.float32(img)

iiImage = computeIntegralImage( imgFloat )

print "And now for the show."

# show image & prediction side by side
plt.ion()
plt.figure()

plt.subplot(1,2,1)
plt.imshow(imgFloat[:,:,10],cmap="gray")
plt.title("Click on any image to exit")

plt.subplot(1,2,2)
plt.imshow(iiImage[:,:,10],cmap="gray")
plt.title("Click on any image to exit")

plt.ginput(1)

