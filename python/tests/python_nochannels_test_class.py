###################################################################################
# Test for the IIBoost wrapper class
###################################################################################

from IIBoost import Booster
from sklearn.externals import joblib	# to load data

# to show something
import matplotlib.pyplot as plt


# load data
gt = joblib.load("gt.jlb")
img = joblib.load("img.jlb")


model = Booster()

# Train: note that we pass a list of stacks
model.train( [img], [gt], numStumps=100, debugOutput=True)

pred = model.predict( img )


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
