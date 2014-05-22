import os
import tempfile
import cPickle as pickle

import numpy
import h5py

import logging
logger = logging.getLogger(__name__)

import IIBoost

class IIBoostLazyflowClassifierFactory(object):
    """
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
            converted_images.append( numpy.asarray(image[...,0], dtype=numpy.uint8) )

        converted_labels = []
        for label_image in label_images:
            assert len(label_image.shape) == 4, "IIBoost expects 3D data."
            assert label_image.shape[-1] == 1, "IIBoost expects exactly one channel"
            converted_labels.append( numpy.asarray(label_image[...,0], dtype=numpy.uint8) )

        model = IIBoost.Booster()
        model.train( converted_images, converted_labels, *self._args, **self._kwargs )

        # Save for future reference
        flattened_labels = map( numpy.ndarray.flatten, converted_labels )
        all_labels = numpy.concatenate(flattened_labels)
        known_labels = numpy.unique(all_labels)

        return IIBoostLazyflowClassifier( model, known_labels )

    def get_halo_shape(self, data_axes):
        # FIXME: What halo does IIBoost require?
        return (0,) * len(data_axes)

    @property
    def description(self):
        return "IIBoost Classifier"

class IIBoostLazyflowClassifier(object):
    """
    Adapt the IIBoost classifier to the interface lazyflow expects.
    """
    def __init__(self, model, known_labels):
        self._known_labels = known_labels
        self._model = model
    
    def predict_probabilities_pixelwise(self, image):
        logger.debug( 'predicting with IIBoost' )
        assert len(image.shape) == 4, "IIBoost expects 3D data."
        assert image.shape[-1] == 1, "IIBoost expects exactly one channel"

        # IIBoost requires both images and labels to be uint8
        image = numpy.asarray(image, dtype=numpy.uint8)[...,0]
        return self._model.predict( image )
    
    @property
    def known_classes(self):
        return self._known_labels

    def get_halo_shape(self, data_axes):
        # FIXME: What halo does IIBoost require?
        return (0,) * len(data_axes)

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
