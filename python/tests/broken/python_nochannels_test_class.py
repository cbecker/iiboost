###################################################################################
# Test for the IIBoost wrapper class
###################################################################################
import os
from iiboost import Booster
from sklearn.externals import joblib	# to load data

# to show something
import matplotlib.pyplot as plt
import matplotlib.colors

import numpy

import faulthandler
faulthandler.enable()

#dir = os.path.split(__file__)[0]
#os.chdir("/Users/bergs/Documents/workspace/iiboost/build")

# load data
gt = joblib.load("../../testData/gt.jlb")
img = joblib.load("../../testData/img.jlb")

import threading
model = Booster()
model.train( [img], [gt], numStumps=2, debugOutput=True)

img2 = joblib.load("../../testData/img.jlb")

def predict():
    model.predict( img2 )

prediction_thread = threading.Thread(target=predict)
prediction_thread.start()
prediction_thread.join()

prediction_thread = threading.Thread(target=predict)
prediction_thread.start()
prediction_thread.join()

prediction_thread = threading.Thread(target=predict)
prediction_thread.start()
prediction_thread.join()



# load data
#gt = joblib.load("gt.jlb")
#img = joblib.load("img.jlb")

train_img = numpy.load( '/tmp/training_img.npy' )
label_img = numpy.load( '/tmp/training_labels.npy' )
prediction_input_img = numpy.load( '/tmp/predict_img.npy' )

print train_img.shape, train_img.dtype
print label_img.shape, label_img.dtype
print prediction_input_img.shape, prediction_input_img.dtype

#print gt.shape, gt.dtype
#print numpy.unique(gt)

#numpy.save('/magnetic/iiboost_test_grayscale.npy', img)
#numpy.save('/magnetic/iiboost_test_labels.npy', gt)

#print img.shape, img.dtype

models = []

import threading
model = Booster()
model.train( [img], [gt], numStumps=2, debugOutput=True)

def predict():
    pred = model.predict( prediction_input_img )
prediction_thread = threading.Thread(target=predict)
prediction_thread.start()
prediction_thread.join()
print "DONE WITH ALL THAT"

pred = predictions[0]

from PyQt4.QtGui import QApplication, QColor
import volumina

app = QApplication(sys.argv)

viewer = volumina.viewer.Viewer()
viewer.show()
raw_layer = viewer.addGrayscaleLayer(train_img)
raw_layer.name = "Raw"

gt_colortable = [ QColor(0,0,0,0),      # clear
                  QColor(255,0,0,255),  # red
                  QColor(0,255,0,255) ] # green
gt_colortable = map( lambda c: c.rgba(), gt_colortable )
gt_layer = viewer.addColorTableLayer(label_img, colortable=gt_colortable)
gt_layer.name = "Ground-truth"

pred = model.predict( prediction_input_img )
print "output shape:", pred.shape

pred_layer = viewer.addAlphaModulatedLayer(pred)
pred_layer.tintColor = QColor(255,255,0)
pred_layer.name = "Prediction"

app.exec_()

sys.exit(0)
###########################################################################
# show image & prediction side by side
plt.ion()
plt.figure()

plt.subplot(1,3,1)
plt.imshow(img[:,:,10],cmap="gray")
plt.title("Click on the image to exit")

plt.subplot(1,3,2)
plt.imshow(pred[:,:,10],cmap="gray")
plt.title("Click on the image to exit")

# rgba
label_colors = [[0,0,0,1], # black
                [1,0,0,1], # red
                [0,1,0,1]] # green
label_colors = numpy.asarray( label_colors, dtype=numpy.float32 )
label_cmap = matplotlib.colors.ListedColormap( label_colors )

plt.subplot(1,3,3)
plt.imshow(gt[:,:,10],cmap=label_cmap)
plt.title("Click on the image to exit")

plt.ginput(1)

ss = model.serialize()
model.deserialize( ss )



#print "Serialization string: " + model.serialize()
