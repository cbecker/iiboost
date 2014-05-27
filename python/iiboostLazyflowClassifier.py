import os
import tempfile
import cPickle as pickle
import threading

import numpy
import h5py

import logging
logger = logging.getLogger(__name__)

import IIBoost

class IIBoostLazyflowClassifierFactory(object):
    """
    This class adheres to the LazyflowPixelwiseClassifierFactoryABC interface, 
    which means it can be used by the standard classifier operators defined in lazyflow.
    
    Instances of this class can create trained instances of IIBoostLazyflowClassifier,
    which adheres to the LazyflowPixelwiseClassifierABC interface.
    """
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    
    def create_and_train_pixelwise(self, images, label_images):
        logger.debug( 'training with IIBoost' )

        # IIBoost requires both images and labels to be uint8, 3D only
        converted_images = []
        for image in images:
            assert len(image.shape) == 4, "IIBoost expects 3D data."
            assert image.shape[-1] == 1, "IIBoost expects exactly one channel"
            converted = numpy.array( numpy.asarray(image[...,0], dtype=numpy.uint8) )
            converted_images.append( converted )

        converted_labels = []
        for label_image in label_images:
            assert len(label_image.shape) == 4, "IIBoost expects 3D data."
            assert label_image.shape[-1] == 1, "IIBoost expects exactly one channel"
            converted = numpy.array( numpy.asarray(label_image[...,0], dtype=numpy.uint8) )
            converted_labels.append( converted )

        model = IIBoost.Booster()
        model.train( converted_images, converted_labels, *self._args, **self._kwargs )

        # Save for future reference
        flattened_labels = map( numpy.ndarray.flatten, converted_labels )
        all_labels = numpy.concatenate(flattened_labels)
        known_labels = numpy.unique(all_labels)
        if known_labels[0] == 0:
            known_labels = known_labels[1:]

        return IIBoostLazyflowClassifier( model, known_labels )

    def get_halo_shape(self, data_axes):
        # FIXME: What halo does IIBoost require?
        halo_shape = (100,) * (len(data_axes)-1)
        halo_shape += (0,) # no halo for channel
        return halo_shape

    @property
    def description(self):
        return "IIBoost Classifier"

# This assertion should pass if lazyflow is available.
#from lazyflow.classifiers import LazyflowPixelwiseClassifierFactoryABC
#assert issubclass( IIBoostLazyflowClassifierFactory, LazyflowPixelwiseClassifierFactoryABC )

class IIBoostLazyflowClassifier(object):
    """
    Adapt the IIBoost classifier to the interface lazyflow expects.
    """
    def __init__(self, model, known_labels):
        self._known_labels = known_labels
        self._model = model
        self._lock = threading.Lock()
    
    def predict_probabilities_pixelwise(self, image):
        logger.debug( 'predicting with IIBoost' )
        assert len(image.shape) == 4, "IIBoost expects 3D data."
        assert image.shape[-1] == 1, "IIBoost expects exactly one channel"

        # IIBoost requires both images and labels to be uint8
        image = numpy.asarray(image, dtype=numpy.uint8)[...,0]
        image = numpy.array( image )
        with self._lock:
            prediction_img = self._model.predict( image )
            # Image from model prediction has no channels,
            #  but lazyflow expects classifiers to produce one channel for each 
            #  label class.  Here, we simply insert zero-channels for all but the last channel.
            prediction_img_reshaped = numpy.zeros( prediction_img.shape + (len(self._known_labels),), dtype=numpy.float32 )
            prediction_img_reshaped[...,-1] = prediction_img
            
            assert prediction_img_reshaped.shape == image.shape + (len(self._known_labels),), \
                "Output image had wrong shape. Expected: {}, Got {}"\
                "".format( image.shape + len(self._known_labels), prediction_img_reshaped.shape )
            return prediction_img_reshaped
    
    @property
    def known_classes(self):
        return self._known_labels

    def get_halo_shape(self, data_axes):
        # FIXME: What halo does IIBoost require?
        halo_shape = (100,) * (len(data_axes)-1)
        halo_shape += (0,) # no halo for channel
        return halo_shape

    def serialize_hdf5(self, h5py_group):
        # FIXME: save the classifier
        
        h5py_group['known_labels'] = self._known_labels
        
        # This field is required for all classifiers
        h5py_group['pickled_type'] = pickle.dumps( type(self) )

    @classmethod
    def deserialize_hdf5(cls, h5py_group):
        known_labels = list(h5py_group['known_labels'][:])
        
        # FIXME: Implement deserialization
        raise NotImplementedError
        #return IIBoostClassifier()

# This assertion should pass if lazyflow is available.
#from lazyflow.classifiers import LazyflowPixelwiseClassifierABC
#assert issubclass( IIBoostLazyflowClassifier, LazyflowPixelwiseClassifierABC )
