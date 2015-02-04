cmake_minimum_required (VERSION 2.8) 

set(PYTHON_BASE_PATH "/usr/" CACHE FILEPATH "Python: path to base system")

include_directories( ${PYTHON_BASE_PATH}/lib/python2.7/site-packages/numpy/core/include/numpy/ )
include_directories( ${PYTHON_BASE_PATH}/lib/python2.7/dist-packages/numpy/core/include/numpy/ )
include_directories( ${PYTHON_BASE_PATH}/include/python2.7 )

link_directories( ${PYTHON_BASE_PATH}/lib )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

add_library( iiboost_python SHARED src/iiboost_python.cpp ../slic/LKM.cpp ../slic/utils.cpp )
install(TARGETS iiboost_python
        LIBRARY DESTINATION ${PYTHON_BASE_PATH}/lib/python2.7/site-packages/iiboost)

if (APPLE)
    target_link_libraries( iiboost_python ${COMMON_LIBS} python2.7 )
else()
    target_link_libraries( iiboost_python ${COMMON_LIBS} python2.7 rt)
endif()

install(FILES iiboost/__init__.py 
              iiboost/booster.py
        DESTINATION ${PYTHON_BASE_PATH}/lib/python2.7/site-packages/iiboost)

#_create_symlink(iiboost_python "${CMAKE_SOURCE_DIR}/testData/img.jlb" "img.jlb")
#_create_symlink(iiboost_python "${CMAKE_SOURCE_DIR}/testData/gt.jlb" "gt.jlb")
#_create_symlink(iiboost_python "${CMAKE_SOURCE_DIR}/python/python_test_raw.py"   "python_test_raw.py")
#_create_symlink(iiboost_python "${CMAKE_SOURCE_DIR}/python/python_computeII_example.py" "python_computeII_example.py")
#_create_symlink(iiboost_python "${CMAKE_SOURCE_DIR}/python/python_test_class.py" "python_test_class.py")
#_create_symlink(iiboost_python "${CMAKE_SOURCE_DIR}/python/python_light_test_class.py" "python_light_test_class.py")
#_create_symlink(iiboost_python "${CMAKE_SOURCE_DIR}/python/python_nochannels_test_class.py" "python_test_class.py")
#_create_symlink(iiboost_python "${CMAKE_SOURCE_DIR}/python/python_channels_test_class.py" "python_channels_test_class.py")
#_create_symlink(iiboost_python "${CMAKE_SOURCE_DIR}/python/IIBoost.py" "IIBoost.py")
