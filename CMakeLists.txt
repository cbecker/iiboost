project (ContextCueBoosting)

# we need v2.8 for ExternalProject
cmake_minimum_required (VERSION 2.8) 
include(ExternalProject)

## Needs ITK at least v4
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

if( "${ITK_VERSION_MAJOR}" LESS 4 )
  message(FATAL_ERROR "ITK version 4 required")
endif()

## RANDOMLIB
set(RANDOMLIB_DESTPATH "${CMAKE_BINARY_DIR}/randomlib")
set(RANDOMLIB_LIB "${RANDOMLIB_DESTPATH}/lib/libRandom.a")
set(RANDOMLIB_INCPATH "${RANDOMLIB_DESTPATH}/include")


# If you want to comile the python wrapper, for Ilastik point PYTHON_BASE_PATH it to where Ilastik is
option(BUILD_PYTHON_WRAPPER "Build the python wrapper" ON)
if (BUILD_PYTHON_WRAPPER)
    set(PYTHON_BASE_PATH "/data/phd/software/ilastik-1.0.3-Linux" CACHE FILEPATH "Python: path to base system")
    
    include_directories( ${PYTHON_BASE_PATH}/lib/python2.7/site-packages/numpy/core/include/numpy/ )
    include_directories( ${PYTHON_BASE_PATH}/lib/python2.7/dist-packages/numpy/core/include/numpy/ )
    include_directories( ${PYTHON_BASE_PATH}/include/python2.7 )

    link_directories( ${PYTHON_BASE_PATH}/lib )

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

externalproject_add( 
    randomlibBuild
    GIT_REPOSITORY http://git.code.sf.net/p/randomlib/code
    GIT_TAG "r1.6"
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${RANDOMLIB_DESTPATH}" "-DCMAKE_BUILD_TYPE=RELEASE" "-DMAINTAINER=OFF" "-DRANDOM_SHARED_LIB=OFF" "-DCMAKE_C_FLAGS=-fpic" "-DCMAKE_CXX_FLAGS=-fpic" "-DDISABLE_BOOST=ON" "-DRANDOMLIB_DOCUMENTATION=OFF"
    UPDATE_COMMAND "" )

add_library(randomlib STATIC IMPORTED DEPENDS)
add_dependencies( randomlib randomlibBuild)
set_property(TARGET randomlib PROPERTY IMPORTED_LOCATION ${RANDOMLIB_LIB})

## LIBCONFIG
set(LIBCONFIG_DESTPATH "${CMAKE_BINARY_DIR}/libconfig")
set(LIBCONFIG_LIB "${LIBCONFIG_DESTPATH}/lib/libconfig++.a")
set(LIBCONFIG_INCPATH "${LIBCONFIG_DESTPATH}/include")

externalproject_add( 
    configlibBuild
    GIT_REPOSITORY https://bitbucket.org/cbecker/libconfig.git
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${LIBCONFIG_DESTPATH}"
    UPDATE_COMMAND "" )

add_library(libconfig STATIC IMPORTED)
add_dependencies( libconfig configlibBuild)
set_property(TARGET libconfig PROPERTY IMPORTED_LOCATION ${LIBCONFIG_LIB})


# Needed for log loss
#set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /home/cjbecker/filer/bitbucket/liblbfgs-1.10/lib/.libs)

include_directories(${LIBCONFIG_INCPATH} ${RANDOMLIB_INCPATH})


# Flags
# static release is pretty generic, possibly not the fastest one though
set(CMAKE_CXX_FLAGS_RELEASE_ALLSTATIC "-fopenmp -O3")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE_ALLSTATIC "-static-libgcc -static-libstdc++ -static")

set(CMAKE_CXX_FLAGS_RELEASE "-fopenmp -O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "-fopenmp -msse2 -msse3 -mssse3 -march=native -g")

if (NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release")
endif()

include_directories( ${ContextCueBoosting_SOURCE_DIR} )
include_directories( ${ContextCueBoosting_SOURCE_DIR}/extra )
include_directories( ${ContextCueBoosting_SOURCE_DIR}/eigen )

add_executable( ccboost test.cpp slic/LKM.cpp slic/utils.cpp )
target_link_libraries( ccboost ${ITK_LIBRARIES} libconfig randomlib rt )

## PYTHON wrapper
if (BUILD_PYTHON_WRAPPER)
    add_library( iiboost_python SHARED python/iiboost_python.cpp slic/LKM.cpp slic/utils.cpp )
    target_link_libraries( iiboost_python ${ITK_LIBRARIES} libconfig randomlib rt python2.7 )

    # copy python test files
    configure_file("testData/img.jlb" "img.jlb" COPYONLY)
    configure_file("testData/gt.jlb"  "gt.jlb" COPYONLY)

    configure_file("python/python_test_raw.py"   "python_test_raw.py"   COPYONLY)
    configure_file("python/python_test_class.py" "python_test_class.py" COPYONLY)
    configure_file("python/IIBoost.py" "IIBoost.py" COPYONLY)
endif()

