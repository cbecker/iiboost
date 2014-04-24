Learning Context Cues for Synapse Segmentation
==============================================
This code implements a synapse segmentation algorithm, as introduced in [1]. 
This is based on the first version available at http://cvlab.epfl.ch/software/synapse

Please check http://cvlab.epfl.ch/software/synapse for updated instructions
on how to use this code.

-------------
xx WARNING xx: THIS IS A BETA VERSION and DOES NOT FULLY IMPLEMENT THE METHOD IN [1].
xx WARNING xx: If you want to run the approach of [1], download the original version
xx WARNING xx: from http://cvlab.epfl.ch/software/synapse
-------------


GETTING STARTED
---------------
You need CMake to compile the code. So far, only Linux is supported.

First create a build folder, for example:
  cd <where_iiboost_is>
  mkdir build

Now configure with ccmake:
  ccmake ../

Choose to build the Python Wrapper (BUILD_PYTHON_WRAPPER set to ON),
specify the path to ITK 4.0 (ITK_DIR) and set CMAKE_BUILD_TYPE to RELEASE.

Finally, if you are using ILASTIK, set PYTHON_BASE_PATH to the path where ilastik is.
Otherwise, point it to your root folder (eg "/").

You can now compile with make, and try the python wrapper with
  python2 python_test_class.py


REFERENCES
----------

  For more information about the synapse segmentation algorithm,
  please check the following article:
  
  [1] Learning Context Cues for Synapse Segmentation
      C. Becker, K. Ali, G. Knott and P. Fua.
      IEEE Transactions on Medical Imaging (TMI) 2013
  
  [2] Learning Context Cues for Synapse Segmentation in EM Volumes.
      C. J. Becker, K. Ali, G. Knott and P. Fua. 
      International Conference on Medical Image Computing and 
      Computer Assisted Intervention (MICCAI), Lecture Notes in Computer Science, 2012.

  
CONTACT
-------

  Please mail carlos.becker@epfl.ch for bug reports, comments and questions.
