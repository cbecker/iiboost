from IIBoost import EigenVectorsOfHessianImage
from sklearn.externals import joblib    # to load data

import numpy as np

# to show something
import matplotlib.pyplot as plt

# load data
gt = joblib.load("gt.jlb")
img = joblib.load("img.jlb")

# compute eigen vector image
eigvecImg = EigenVectorsOfHessianImage()
eigvecImg.compute( gt, zAnisotropyFactor=1.0 )

